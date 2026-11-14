"""Actual author-runtime, synthetic-only incremental core gates."""
import copy,hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
W=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(W/'src')) if (W/'src').exists() else None
from hybrid_tsad.hybrid_v1 import joint_cycle_core as core
from hybrid_tsad.hybrid_v1.data import Run,window,arrived_window
from hybrid_tsad.hybrid_v1.training import tensor_batch,passes
from hybrid_tsad.hybrid_v1.scoring import residual_components
torch.set_num_threads(1)
class Gate(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.workspace=Path(os.environ.get('RESEARCH_WORKSPACE',Path(__file__).resolve().parents[2]))
  p=cls.workspace/'results/hybrid_v1_stage1_full_N8000_seed2025_20261102_v1/source_train'
  cls.cfg=json.loads((p/'config.json').read_text(encoding='utf-8'));cls.cfg['device']='cpu'
  cls.model,cls.mapping=core.construct(cls.cfg,shared_initial=p/'initial.pt',expected_initial_sha256='9a8f18ba57f64bc0ecbf17cab20d4fc1bc9758b433667f135de6eaa27ace6071')
  cls.model.eval();cls.initial=copy.deepcopy(cls.model.state_dict())
 def setUp(self):self.model.load_state_dict(self.initial);self.model._adaptation_route='STAGE1';self.model.eval();self.model.zero_grad(set_to_none=True)
 def batch(self,c=2,missing=False,arrived=False):
  x=np.random.default_rng(42).normal(size=(560,c)).astype('float32');o=np.ones_like(x,bool)
  if missing:o[510,0]=False;x[510,0]=np.nan
  r=Run('SYNTHETIC','synthetic','fixture','FIT',x,o)
  return tensor_batch((arrived_window if arrived else window)(r,512,512,32))
 def objective(self,b):
  f,r=passes(self.model,b,2025)
  return core.joint_objective(f,r,b,'huber',frft=self.model.tf_backbone.frft),f,r
 def test_shared_initial_and_removed_heads(self):
  self.assertTrue(self.mapping['shared_exact']);self.assertGreater(len(self.mapping['new_names']),0)
  self.assertFalse(hasattr(self.model,'tf_fusion_delta'));self.assertFalse(hasattr(self.model.tf_backbone.author,'decoder'))
 def test_C1_C38_and_original_residual_ABI(self):
  with torch.no_grad():
   for c in (1,38):
    loss,f,r=self.objective(self.batch(c))
    self.assertEqual(tuple(f['prediction'].shape),(1,32,c));self.assertEqual(tuple(r['tf_reconstruction'].shape),(1,min(2,c),2,100,c))
    self.assertEqual(tuple(r['cycle_spectrum'].shape),(1,min(2,c),2,c,100))
    self.assertTrue(torch.isfinite(loss['total']));self.assertEqual(set(residual_components(f,r,self.batch(c))['scalar'])&{'r_tf','r_ms','r_pred'},{'r_tf','r_ms','r_pred'})
 def test_all_new_and_retained_paths_have_finite_gradient(self):
  self.model.train();loss,_,_=self.objective(self.batch());loss['total'].backward()
  required=('joint_fusion.','time_cycle_decoder.','fractional_cycle_decoder.','tf_backbone.author.net.','tf_backbone.author.encoder.','temporal.','lag.','ms_decoder.','forecast_head.')
  sums={prefix:sum(float(p.grad.abs().sum()) for n,p in self.model.named_parameters() if n.startswith(prefix) and p.grad is not None) for prefix in required}
  self.assertTrue(all(v>0 and np.isfinite(v) for v in sums.values()),sums)
  self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))
 def test_hidden_GT_and_future_not_encoder_inputs(self):
  b=self.batch();changed={k:v.clone() if torch.is_tensor(v) else v for k,v in b.items()}
  changed['reconstruction_context'][:,-1,0]+=999;changed['future_targets']+=555
  with torch.no_grad():
   f,r=passes(self.model,b,2025);ff,rr=passes(self.model,changed,2025)
  self.assertTrue(torch.equal(f['prediction'],ff['prediction']))
  self.assertTrue(torch.equal(r['tf_reconstruction'][:,0],rr['tf_reconstruction'][:,0]));self.assertTrue(torch.equal(r['stats']['mean'][:,0],rr['stats']['mean'][:,0]))
  # Group1 may legitimately observe group0's current channel: do not demand its invariance.
 def test_natural_missing_and_spectral_target_support(self):
  with torch.no_grad():loss,_,_=self.objective(self.batch(missing=True,arrived=True))
  self.assertEqual(int(loss['spectral_target_complete_channel_count']),1);self.assertTrue(torch.isfinite(loss['total']))
 def test_no_complete_spectrum_keeps_original_normal_update(self):
  self.model.train();b=self.batch(c=1,missing=True);loss,f,r=self.objective(b)
  from hybrid_tsad.hybrid_v1.objectives import ssl_objective
  original=ssl_objective(f,r,b,'huber')
  self.assertTrue(torch.equal(loss['valid_windows'],original['valid_windows']));self.assertTrue(bool(loss['valid_windows'].all()))
  self.assertEqual(int(loss['spectral_target_complete_channel_count']),0);self.assertEqual(float(loss['parts']['analysis']),0.)
  before=copy.deepcopy(self.model.state_dict());opt=torch.optim.AdamW(self.model.parameters(),lr=.0003)
  loss['total'].backward();opt.step()
  self.assertTrue(any(p.grad is not None and bool((p.grad!=0).any()) for n,p in self.model.named_parameters() if n.startswith('fractional_cycle_decoder.')))
  self.assertTrue(any(not torch.equal(before[n],p) for n,p in self.model.state_dict().items() if n.startswith('fractional_cycle_decoder.')))
 def test_mode_rng_and_strict_reload(self):
  from hybrid_tsad.hybrid_v1 import training
  state=copy.deepcopy(self.model.state_dict());rng=torch.get_rng_state().clone();old=training.ssl_objective
  with torch.no_grad():a=self.objective(self.batch())[1]['prediction']
  self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(all(torch.equal(state[n],v) for n,v in self.model.state_dict().items()))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);core.save_checkpoint(self.model,p/'model.pt',p/'identity.json');m=core.load_checkpoint(self.cfg,p/'model.pt',p/'identity.json')
   with torch.no_grad():b=m.forecast(self.batch()['history'],self.batch()['observed'],mask_seed=2025)['prediction']
   self.assertTrue(torch.equal(a,b))
   meta=json.loads((p/'identity.json').read_text());meta['architecture']='HYBRID_V1';(p/'wrong.json').write_text(json.dumps(meta))
   with self.assertRaises(PermissionError):core.load_checkpoint(self.cfg,p/'model.pt',p/'wrong.json')
  with self.assertRaises(RuntimeError):
   with core.training_context(self.model):raise RuntimeError('restore')
  self.assertIs(training.ssl_objective,old)
 def test_Z_U_F_trainable_routes(self):
  for route in ('Z','U','F','STAGE1'):
   names=core.update_policy(self.model,route);self.model.train(route!='Z')
   if route=='Z':self.assertEqual(names,[])
   if route in ('U','F'):
    self.assertFalse(self.model.temporal.training);self.assertFalse(self.model.tf_backbone.training);self.assertTrue(self.model.joint_fusion.training)
    self.assertEqual(self.model.event_head.training,route=='F');self.assertEqual(self.model.event_head.weight.requires_grad,route=='F')
   if route=='STAGE1':self.assertTrue(self.model.temporal.training)
 def test_real_U_and_F_updates_use_new_objective(self):
  def run(name,role):
   x=np.random.default_rng(len(name)).normal(size=(550,1)).astype('float32')
   return Run('SYNTHETIC','e',name,role,x,np.ones_like(x,bool))
  for route in ('U','F'):
   if route=='U':fits=[run('fit','FIT')];sels=[run('select','SELECT')];extra={}
   else:
    fits=[run('event','SUPPORT'),run('normal','SUPPORT')];sels=[run('select','NORMAL_SELECT')]
    extra=dict(support=dict(K=1,events=[dict(event_id='E1',entity='e',run='event',start=513,end=515)],normal=[dict(matched_event_id='E1',entity='e',run='normal',start=513,end=515)],forbidden=[]))
   m,report=core.adapt(self.model,route,fits,sels,history=512,horizon=32,steps=2,lr=.0003,kind='huber',seed=2025,validate_every=1,**extra)
   self.assertEqual(report['updates'],2);self.assertTrue(any(not torch.equal(m.state_dict()[n],self.model.state_dict()[n]) for n in m.state_dict() if n.startswith('fractional_cycle_decoder.')))
   self.assertTrue(all(torch.equal(m.state_dict()[n],self.model.state_dict()[n]) for n in m.state_dict() if n.startswith(('temporal.','tf_backbone.','lag.','ms_filter.'))))
 def test_F_zero_support_is_exact_Z(self):
  before=copy.deepcopy(self.model.state_dict())
  result,report=core.adapt(self.model,'F',[],[],support=dict(K=0))
  self.assertEqual(report['route'],'Z');self.assertEqual(report['updates'],0)
  self.assertTrue(all(torch.equal(before[n],p) for n,p in result.state_dict().items()))
  self.assertTrue(all(not p.requires_grad for p in result.parameters()));self.assertFalse(result.training)
if __name__=='__main__':unittest.main(verbosity=2)
