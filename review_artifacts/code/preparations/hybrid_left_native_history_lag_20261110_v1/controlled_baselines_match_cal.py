"""Mechanical CAL index alignment; Ridge weights and all predictions unchanged."""
import json,time,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name/'controlled_baselines';D=R/'results'/P.name/'controlled_baselines_matched512'
sys.path.insert(0,str(P));from controlled_baselines import load,predict
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
start=time.monotonic();D.mkdir(parents=True,exist_ok=False);m=json.loads((P/'CONTROLLED_DATA_MANIFEST.json').read_text())
with np.load(O/'NORMAL_FIT_RIDGE.npz')as z:W=z['weight'].copy();mean=z['mean'].copy();scale=z['scale'].copy();assert float(z['ridge_alpha'])==1.
cal={};methods=['ARX_Ridge512','persistence','HistoryMean512']
for method in methods:
 residuals=[];indices=[]
 for e in m['roles']:
  if e['role']!='CAL':continue
  v=load(e,mean,scale);ends=np.linspace(512,len(v)-1,256,dtype=np.int64);p=predict(v,W,method);residuals.append(v[ends]-p[ends-512]);indices.append(dict(seed=e['meta']['seed'],index=ends.tolist()))
 residual=np.concatenate(residuals);sc=np.maximum(np.median(np.abs(residual),axis=0),1e-6);s=(np.abs(residual)/sc).mean(1);tau=float(np.quantile(s,.99,method='higher'));cal[method]=dict(scale=sc.tolist(),threshold=tau,points=512,indices=indices,normal_CAL_RMSE=float(np.sqrt(np.mean(residual**2))))
 out=D/method;out.mkdir()
 for e in m['roles']:
  if e['role']!='QUERY':continue
  old=O/method/(e['meta']['case']+'_'+str(e['meta']['seed'])+'.npz')
  with np.load(old)as z:arrays={k:z[k].copy()for k in z.files if k!='score'}
  score=(np.abs(arrays['observed']-arrays['prediction'])/sc).mean(1);np.savez_compressed(out/old.name,**arrays,score=score)
dump(D/'CALIBRATIONS.json',cal);dump(D/'ALL_RAW_SEALED_BEFORE_LABELS.json',dict(labels_read=False,reason='match all methods fixed normalCAL512, not performance search',old_full7168CAL_preserved=True,Ridge_weight_source_sha256=sha(O/'NORMAL_FIT_RIDGE.npz'),alpha_fixed=1.,new_training=False,new_GPU_forward=False,allbanks=[dict(path=str(p),sha256=sha(p))for p in D.rglob('*.npz')]))
print(json.dumps(dict(status='CPU_MATCHED512_RAW_READY',seconds=time.monotonic()-start)),flush=True)
