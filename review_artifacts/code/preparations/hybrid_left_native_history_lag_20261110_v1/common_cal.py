import sys,json,time,hashlib,importlib.util,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name
s=importlib.util.spec_from_file_location('history_worker',P/'worker.py');w=importlib.util.module_from_spec(s);s.loader.exec_module(w)
arm=sys.argv[1];torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction(30*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats(0)
start=time.monotonic();out=O/'N'/'cal_common'if arm=='N'else O/arm/'cal_common';out.mkdir(parents=True,exist_ok=False)
if arm=='N':
 sys.path.insert(0,str(w.ROOT));from models.Left.LEFT import Left
 random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed(2025);model=Left(SimpleNamespace(**json.loads((w.ROOT/'configs/PSM/model_configs_Left.json').read_text()))).cuda();pt=R/'results/left_psm_R_seed2025_20261101_v1/model/PSM/Left/checkpoints_0/checkpoint.pth';res=model.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False);assert not res.missing_keys;model.eval()
else:model,hashpt=w.load(arm)
st=json.loads((R/'results/left_psm_R_seed2025_20261101_v1/SCALER_R.json').read_text())['statistics'];v=(np.load(R/'preparations/left_baseline_build_20261101_v1/normal_prefix_R_v1/prefix.npy')-np.asarray(st['mean']))/np.asarray(st['scale']);ends=np.linspace(106496,132480,512,dtype=np.int64);assert ends.min()-512>=105984
scores=[]
with torch.no_grad():
 for a in range(0,512,256):
  x=w.windows(v,ends[a:a+256],105984);assert np.isfinite(x).all();tensor=torch.from_numpy(x[:,-192:]if arm=='N'else x).cuda();raw,_=model.infer(tensor,None,None,None);_,z=w.scoring.author_numpy_scores([raw.cpu().numpy()]);scores.append(z.reshape(len(tensor),192)[:,-1])
scores=np.concatenate(scores);tau=float(np.quantile(scores,.99,method='higher'));np.savez_compressed(out/'scores.npz',index=ends,score=scores)
record=dict(status='COMPLETED',phase='cal_common',arm=arm,records=512,threshold=tau,quantile=.99,method='higher',role='SELECT_reuse_exploratory',full_history_in_role=True,first_end=int(ends[0]),last_end=int(ends[-1]),labels_read=False,seconds=time.monotonic()-start);(out/'RECEIPT.json').write_text(json.dumps(record,indent=2));(out/'GPU_PEAK.json').write_text(json.dumps(dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())));print(json.dumps(record),flush=True)
