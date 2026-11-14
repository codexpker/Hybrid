from pathlib import Path
import sys,json,time
import torch
from types import SimpleNamespace
P=Path(__file__).resolve().parent;R=P.parents[1]
sys.path[:0]=[str(R/'hybrid_ad/src'),str(R/'preparations/left_baseline_build_20261101_v1/upstream')]
from models.Left.LEFT import Left
from hybrid_tsad.left_channel_blindspot import channel_blindspot_class
torch.set_num_threads(2);torch.manual_seed(2025);t=time.monotonic()
c=json.loads((R/'preparations/hybrid_native_predictive_cycle_20261113_v1/configs/PSM.json').read_text());c['enc_in']=4
m=channel_blindspot_class(Left,'blind')(SimpleNamespace(**c));m.train();m.step=c['warmup_steps']+c['ramp_steps']+1
x=torch.randn(2,192,4);j=torch.tensor([0,3]);p=m.masked_forward(x,j);loss,_=m.masked_target_loss(p,x,j);loss.backward()
groups={name:float(sum((p.grad.detach().abs().sum()for n,p in m.named_parameters()if n.startswith(name)and p.grad is not None),torch.tensor(0.)))for name in ['Et','Ef','encoder','fusion','ms_decoder','Dt','Df','z_fuse']}
assert all(v>0 for v in groups.values()),groups
out=dict(status='PASS',only_masked_target_loss_backward=True,no_native_auxiliary_loss_backward=True,gradient_groups=groups,seconds=time.monotonic()-t)
(P/'CPU_MASK_GRADIENT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
