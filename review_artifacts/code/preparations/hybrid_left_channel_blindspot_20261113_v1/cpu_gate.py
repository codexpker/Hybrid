from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
from types import SimpleNamespace
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path[:0]=[str(R/'hybrid_ad/src'),str(R/'preparations/left_baseline_build_20261101_v1/upstream')]
from models.Left.LEFT import Left
from hybrid_tsad.left_channel_blindspot import channel_blindspot_class
torch.set_num_threads(2);torch.manual_seed(2025);start=time.monotonic()
c=json.loads((R/'preparations/hybrid_native_predictive_cycle_20261113_v1/configs/PSM.json').read_text());c['enc_in']=4
cfg=SimpleNamespace(**c);klass=channel_blindspot_class(Left,'blind');m=klass(cfg)
assert not any(isinstance(v,torch.nn.modules.batchnorm._BatchNorm)for v in m.modules())
x=torch.randn(2,192,4);j=torch.tensor([0,3]);altered=x.clone();altered[0,:,0]+=torch.arange(192)*3;altered[1,:,3]-=100
m.eval();a=m.masked_forward(x,j);b=m.masked_forward(altered,j)
for k in [1,3,4,9]:assert torch.equal(a[k],b[k])
for k in ['max_prob','entropy']:
    assert torch.equal(a[10]['aux_t'][k],b[10]['aux_t'][k])
assert torch.equal(a[10]['p_f_aligned'],b[10]['p_f_aligned'])
sa=m.score_one_channel(x,j);sb=m.score_one_channel(altered,j);assert not torch.equal(sa,sb)
# Masked-target losses use only selected targets; other target columns are
# excluded except in native auxiliary branch and learned MS target transform.
loss_a,_=m.masked_target_loss(a,x,j);loss_b,_=m.masked_target_loss(a,altered,j);assert loss_a!=loss_b
native=Left(cfg);native.load_state_dict(m.state_dict(),strict=True)
control=channel_blindspot_class(Left,'native')(cfg);control.load_state_dict(native.state_dict(),strict=True)
native.train();control.train();native.step=control.step=c['warmup_steps']+c['ramp_steps']+1
for q in [native,control]:q.conf_threshold=-1.;q.update_q_err=1.;q.update_q_ad=1.
rng=torch.get_rng_state();native.zero_grad();ln=native(x,None,None,None)[0];ln.backward()
torch.set_rng_state(rng);control.zero_grad();lc=control(x,j)[0];lc.backward();assert torch.equal(ln,lc)
assert native.step==control.step
for (kn,vn),(kc,vc) in zip(native.state_dict().items(),control.state_dict().items()):
    assert kn==kc and torch.equal(vn,vc),kn
for (kn,vn),(kc,vc) in zip(native.named_parameters(),control.named_parameters()):
    assert (vn.grad is None)==(vc.grad is None),kn
    if vn.grad is not None:assert torch.equal(vn.grad,vc.grad),kn
# Explicit active-EMA path: masked runs first and reads pre-native snapshots.
m.train();m.step=c['warmup_steps']+c['ramp_steps']+1;m.conf_threshold=-1.;m.update_q_err=1.;m.update_q_ad=1.
before_step=m.step;before_proto=m.proto_t.prototypes.detach().clone();m.zero_grad()
na,ma,joint=m.paired_losses(x,j);joint.backward();assert m.step==before_step+1
assert torch.isfinite(joint)and all(torch.isfinite(v.grad).all()for v in m.parameters()if v.grad is not None)
groups={name:float(sum((p.grad.detach().abs().sum()for n,p in m.named_parameters()if n.startswith(name)and p.grad is not None),torch.tensor(0.)))for name in ['Et','Ef','encoder','fusion','ms_decoder','Dt','Df','z_fuse']}
assert all(v>0 for v in groups.values()),groups
assert not torch.equal(before_proto,m.proto_t.prototypes)
assert len(m.state_dict())==len(native.state_dict())
# Replay identical state and dropout RNG; target perturbation cannot influence
# masked reconstruction before the native EMA update.
aa=klass(cfg);bb=klass(cfg);bb.load_state_dict(aa.state_dict());aa.train();bb.train();aa.step=bb.step=c['warmup_steps']+c['ramp_steps']+1
rng=torch.get_rng_state();pa=aa.masked_forward(x,j);torch.set_rng_state(rng);pb=bb.masked_forward(altered,j)
for k in [1,3,4,9]:assert torch.equal(pa[k],pb[k])
out=dict(status='PASS',seconds=time.monotonic()-start,fixture='B2/H192/C4 CPU; frozen PSM model other settings',whole_target_history_perturbation_latent_decoders_gate_exact=True,selected_target_changes_only_loss_score=True,native_continued_loss_gradient_state_RNG_exact=True,active_EMA_native_only_once=True,masked_native_backward_valid_after_EMA=True,gradient_groups=groups,new_parameters=0,code_sha256=hashlib.sha256((R/'hybrid_ad/src/hybrid_tsad/left_channel_blindspot.py').read_bytes()).hexdigest())
(P/'CPU_GATE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
