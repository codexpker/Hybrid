from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,time
import numpy as np
import torch
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path[:0]=[str(P),str(P/'runtime'),str(R/'preparations/left_baseline_build_20261101_v1/upstream')]
from worker import Data
from models.Left.LEFT import Left
from hybrid_tsad.left_channel_blindspot import channel_blindspot_class
torch.set_num_threads(2);torch.manual_seed(2025);t=time.monotonic();d=Data(['FIT','SELECT']);v,_=d.load('FIT');sv,_=d.load('SELECT')
try:d.load('CAL');raise AssertionError('role guard')
except PermissionError:pass
c=json.loads((P/'MODEL.json').read_text());m=channel_blindspot_class(Left,'blind')(SimpleNamespace(**c));m.eval();m.score_one_channel=lambda *a,**k:(_ for _ in()).throw(AssertionError('masked inference forbidden'))
with torch.no_grad():q=m.native_readout(torch.from_numpy(np.stack([v[:192],v[192:384]])))
assert q.shape==(2,192,38)and torch.isfinite(q).all()
plans=[]
for seed in [2025,2026,2027]:
 with np.load(P/('PLAN_'+str(seed)+'.npz'))as z:
  assert z['source'].shape==(4000,128)and z['paired'].shape==z['channels'].shape==(2000,128)
  assert min(z['source'].min(),z['paired'].min())>=0 and max(z['source'].max(),z['paired'].max())<=17087-192
  assert z['channels'].min()>=0 and z['channels'].max()<38
  plans.append(dict(seed=seed,SHA=hashlib.sha256((P/('PLAN_'+str(seed)+'.npz')).read_bytes()).hexdigest()))
out=dict(status='PASS',roles=d.access,native_readout_no_mask=True,native_shape_C38H192=True,all3_plans_bounded=True,SELECT_ends=(np.arange(0,len(sv)-191,192)+191).tolist(),plans=plans,seconds=time.monotonic()-t,GPU_hours=0,reused_core_gates='hybrid_left_channel_blindspot_20261113_v1/CPU_GATE.json+CPU_MASK_GRADIENT.json')
(P/'CPU_GATE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
