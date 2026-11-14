import copy,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from hybrid_tsad import roca_baseline as r

class RoCATests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1)
  root=Path(os.environ['ROCA_SOURCE_ROOT']);m=json.loads(Path(os.environ['ROCA_SOURCE_MANIFEST']).read_text())
  cls.core=r.load_official(root,{e['relative']:e['sha256'] for e in m['files']})
 def cfg(self,template='IOpsCompetition',channels=1):
  return r.config_from_template(self.core,template,channels,dict(batch_size=4,nu=.25,num_epoch=2,warmup=2,change_center_epoch=2))
 def run_data(self,role='FIT',start=0,n=100):
  a=np.random.default_rng(2).normal(size=(n,1)).astype(np.float32)
  return SimpleNamespace(values=a,observed=np.ones_like(a,bool),role=role,start=start,domain='SYNTHETIC',entity='tiny',run='record')
 def test_scaler_actual_sklearn_near_constant(self):
  for x in [np.ones((100,3))*1e12,np.c_[np.arange(100),np.ones(100),1e10+np.arange(100)*1e-5]]:
   run=SimpleNamespace(values=x,observed=np.ones_like(x,bool),role='FIT')
   ours=r.FitScaler.fit([run]);ref=StandardScaler().fit(x)
   np.testing.assert_array_equal(ours.mean,ref.mean_);np.testing.assert_array_equal(ours.scale,ref.scale_)
  with self.assertRaises(PermissionError):r.FitScaler.fit([self.run_data('QUERY')])
 def test_roles_and_endpoint_no_future(self):
  run=self.run_data();s=r.FitScaler.fit([run]);x,clock=r.endpoint_windows(run,16,[15,99],s,'FIT')
  self.assertEqual(clock[-1]['input_bounds'],[84,100]);self.assertEqual(clock[-1]['available_index'],99)
  with self.assertRaises(PermissionError):r.endpoint_windows(run,16,[14],s,'FIT')
  with self.assertRaises(PermissionError):r.endpoint_windows(run,16,[15],s,'QUERY')
  with self.assertRaises(PermissionError):r.validate_role_boundaries([run,self.run_data('CAL',110)],15)
 def test_author_formula_and_actual_component_gradients(self):
  cfg=self.cfg();q=torch.randn(12,20,requires_grad=True);qr=torch.randn(12,20,requires_grad=True);c=torch.randn(20)
  oc,var,ln,la=self.core.functions['train'](q,qr,c,2,cfg,'cpu')
  torch.testing.assert_close(r.invariance_scores(q,qr,c),ln,rtol=0,atol=0)
  for component in (oc,var,la.mean()):
   grads=torch.autograd.grad(component,(q,qr),retain_graph=True)
   self.assertTrue(all(torch.isfinite(g).all() for g in grads));self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
 def test_one_update_exact_author_loop(self):
  cfg=self.cfg();torch.manual_seed(3);a=self.core.Model(cfg,'cpu');b=copy.deepcopy(a)
  windows=np.random.default_rng(1).normal(size=(4,1,16)).astype(np.float32);center=torch.randn(20)
  oa=torch.optim.Adam(a.parameters(),lr=cfg.lr,betas=(cfg.beta1,cfg.beta2),weight_decay=cfg.weight)
  ob=torch.optim.Adam(b.parameters(),lr=cfg.lr,betas=(cfg.beta1,cfg.beta2),weight_decay=cfg.weight)
  torch.manual_seed(4);self.core.functions['model_train'](a,oa,r.make_loader(self.core,windows,cfg,5,True),center,cfg,'cpu',2)
  torch.manual_seed(4);rows=r.train_epoch(self.core,b,ob,r.make_loader(self.core,windows,cfg,5,True),center,cfg,2)
  for name,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[name],rtol=0,atol=0)
  self.assertEqual(sum(rows[0]['OE_selected_groups'].values()),3)
  self.assertGreater(rows[0]['component_projection_gradient_L1']['OE'],0)
 def test_two_templates_actual_forward(self):
  for template,channels in [('IOpsCompetition',1),('SWaT',38)]:
   cfg=self.cfg(template,channels);model=self.core.Model(cfg,'cpu').eval()
   with torch.no_grad():a,b=model(torch.randn(2,channels,cfg.window_size))
   self.assertEqual(a.shape,(2,cfg.project_channels));self.assertTrue(torch.isfinite(b).all())
 def test_mode_rng_restore_and_exception(self):
  cfg=self.cfg();model=self.core.Model(cfg,'cpu');modes=[m.training for m in model.modules()];state=torch.random.get_rng_state().clone()
  try:
   with r.eval_preserving_modes(model):raise RuntimeError('fixture')
  except RuntimeError:pass
  self.assertEqual(modes,[m.training for m in model.modules()]);self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
 def test_complete_training_center_reload_and_no_query(self):
  cfg=self.cfg();torch.manual_seed(9);model=self.core.Model(cfg,'cpu');windows=np.random.default_rng(9).normal(size=(8,1,16)).astype(np.float32)
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'discarded';report=r.fit(self.core,model,r.make_loader(self.core,windows,cfg,9,True),cfg,p,windows[:2])
   self.assertEqual(report['scientific_status'],'VALID_FINITE_ENDPOINT');self.assertEqual(report['updates'],4);self.assertEqual(report['OE_selected_total'],6)
   self.assertEqual([e.get('updated') for e in report['center_history'][1:]],[True,False])
   a,c,_=r.reload_endpoint(self.core,p/'endpoint.pt','cpu');b,d,_=r.reload_endpoint(self.core,p/'endpoint.pt','cpu')
   np.testing.assert_array_equal(r.score_windows(a,windows,c),r.score_windows(b,windows,d))
 def test_inverse_cdf_and_permission(self):
  values=np.arange(512);self.assertEqual(r.calibrate(values,'CAL')['threshold'],506)
  with self.assertRaises(PermissionError):r.calibrate(values,'QUERY')

if __name__=='__main__':unittest.main(verbosity=2)
