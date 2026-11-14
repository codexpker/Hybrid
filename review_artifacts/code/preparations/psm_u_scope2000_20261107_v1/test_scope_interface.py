import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent;W=P.parents[1]
sys.path.insert(0,str(W/'preparations/psm_matched_u_ABC_operator_20261107_v1/runtime/src'))
import torch,numpy as np
from hybrid_tsad.hybrid_v1 import stage2_u_matched as u,training,data,objectives
s=importlib.util.spec_from_file_location('hybrid_tsad.hybrid_v1.stage2_u_scope',W/'hybrid_ad/src/hybrid_tsad/hybrid_v1/stage2_u_scope.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Toy(torch.nn.Module):
 def __init__(self):
  super().__init__();self.backbone=torch.nn.Linear(2,2)
  for n in u.ALLOWED:setattr(self,n,torch.nn.Linear(2,2))
  self.event_head=torch.nn.Linear(2,1);self.register_buffer('fixed_buffer',torch.ones(1))
class Run:domain='PSM';entity='toy';run='FIT';start=0
def passes(model,batch,seed):
 x=model.backbone(torch.ones(1,2));y=sum(getattr(model,n)(x) for n in u.ALLOWED)
 f=dict(prediction=y[:,None].expand(1,32,2),stats=dict(mean=torch.zeros(1,1,2),scale=torch.ones(1,1,2)))
 r=dict(tf_reconstruction=y,ms_reconstruction=y*2,hidden_mask=torch.ones(1,2,dtype=torch.bool))
 return f,r
def objective(f,r,b,kind,active):return dict(total=f['prediction'].square().mean(),valid_windows=torch.ones(1,dtype=torch.bool))
def window(run,t,h,p):return dict(footprint=[t-h,t+p],current_target=torch.ones(1,2),current_observed=torch.ones(1,2,dtype=torch.bool))
class Gate(unittest.TestCase):
 def test_paired_scope_forward_mode_rng_and_frozen_state(self):
  torch.manual_seed(11);source=Toy();plan=dict(history=512,horizon=32,seed=2025,FIT=[{},{}],SELECT=[{},{}])
  result=[]
  with tempfile.TemporaryDirectory() as folder,patch.object(u,'resolve_u_plan',side_effect=lambda *a,**k:[(Run(),512),(Run(),513)]),patch.object(u,'validate',return_value=1.),patch.object(training,'passes',passes),patch.object(training,'tensor_batch',side_effect=lambda b,d:b),patch.object(data,'window',window),patch.object(objectives,'ssl_objective',objective):
   for scope in m.SCOPES:
    torch.manual_seed(2025);out=Path(folder)/scope
    model,report,opt=m.train_endpoint(source,[Run()],[Run()],plan,scope=scope,artifact_dir=out,technical=True);result.append((model,report))
   a,b=result
   self.assertEqual(a[1]['module_modes'],b[1]['module_modes']);self.assertEqual(a[1]['initial_torch_rng'],b[1]['initial_torch_rng'])
   self.assertTrue(torch.equal(a[0].backbone.weight,source.backbone.weight));self.assertFalse(torch.equal(b[0].backbone.weight,source.backbone.weight))
   self.assertTrue(torch.equal(b[0].event_head.weight,source.event_head.weight));self.assertIn('event_head.weight',b[1]['unused_trainable_names'])
   self.assertEqual(a[1]['sample_mask_digest'],b[1]['sample_mask_digest'])
   with np.load(Path(folder)/'HEAD/FIRST_FORWARD.npz') as x,np.load(Path(folder)/'FULL/FIRST_FORWARD.npz') as y:
    for key in x.files:self.assertEqual(x[key].tobytes(),y[key].tobytes())
 def test_buffer_and_frozen_parameter_changes_are_rejected(self):
  before={'fusion.weight':1,'backbone.weight':2,'buffer':3};after=dict(before,buffer=4)
  self.assertFalse(m.scope_state_difference(before,after,{'fusion.weight','backbone.weight'},'FULL')['PASS'])
  after=dict(before);after['backbone.weight']=4
  self.assertFalse(m.scope_state_difference(before,after,{'fusion.weight','backbone.weight'},'HEAD')['PASS'])
  self.assertTrue(m.scope_state_difference(before,after,{'fusion.weight','backbone.weight'},'FULL')['PASS'])
 def test_exception_evaluation_restores_heterogeneous_modes_and_rng(self):
  model=Toy();u.adaptation_mode(model);modes=[v.training for v in model.modules()];state=torch.get_rng_state().clone()
  with self.assertRaises(RuntimeError):
   with u.evaluation_state(model,8):torch.rand(7);raise RuntimeError('fixture')
  self.assertEqual(modes,[v.training for v in model.modules()]);self.assertTrue(torch.equal(state,torch.get_rng_state()))
if __name__=='__main__':unittest.main(verbosity=2)
