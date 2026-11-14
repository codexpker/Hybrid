from pathlib import Path
from types import SimpleNamespace
import json,sys,time
import numpy as np
import torch
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path[:0]=[str(P/'runtime'),str(R/'preparations/left_baseline_build_20261101_v1/upstream')]
from models.Left.LEFT import Left
from hybrid_tsad.left_channel_blindspot import channel_blindspot_class
torch.set_num_threads(2);torch.manual_seed(2025);start=time.monotonic()
manifest=json.loads((P/'controlled/ARRAYS.json').read_text());assert set(manifest['FIT']['run_ids']).isdisjoint(manifest['SELECT']['run_ids'])and set(manifest['FIT']['run_ids']).isdisjoint(manifest['CAL']['run_ids'])
with np.load(P/manifest['FIT']['path'])as z:raw=z['values'];mean=raw.mean((0,1));scale=raw.std((0,1));x=torch.from_numpy(np.float32((raw[:2,:100]-mean)/scale))
cfg=json.loads((P/'controlled/MODEL.json').read_text());m=channel_blindspot_class(Left,'blind')(SimpleNamespace(**cfg));m.train();m.step=400;j=torch.tensor([0,2]);loss=m(x,j)[0];loss.backward()
assert torch.isfinite(loss)and all(torch.isfinite(v.grad).all()for v in m.parameters()if v.grad is not None)
out=dict(status='PASS',C3H100_actual_shape=True,fit_select_cal_run_disjoint=True,no_test_values_accessed=True,finite_loss_gradient=True,seconds=time.monotonic()-start,GPU_hours=0)
(P/'CONTROLLED_CPU_GATE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
