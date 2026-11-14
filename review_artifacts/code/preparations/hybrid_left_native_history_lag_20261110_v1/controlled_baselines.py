"""Fixed lag512 Ridge, persistence, mean controls; normal-only fit/cal."""
import json,time,hashlib,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name/'controlled_baselines'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def features(v):
 w=np.lib.stride_tricks.sliding_window_view(v,512,axis=0)[:-1]
 return np.concatenate((w.reshape(len(w),-1),np.ones((len(w),1))),axis=1)
def load(e,mean,scale):
 with np.load(e['path'])as z:v=z['values'].astype(np.float64)
 return (v-mean)/scale
def predict(v,W,method):
 if method=='ARX_Ridge512':return features(v)@W
 if method=='persistence':return v[511:-1]
 if method=='HistoryMean512':
  s=np.concatenate((np.zeros((1,4)),np.cumsum(v,axis=0)),axis=0);return (s[512:-1]-s[:-513])/512
 raise ValueError(method)
def main():
 start=time.monotonic();O.mkdir(parents=True,exist_ok=False);m=json.loads((P/'CONTROLLED_DATA_MANIFEST.json').read_text());fits=[e for e in m['roles']if e['role']=='FIT'];raw=[]
 for e in fits:
  with np.load(e['path'])as z:raw.append(z['values'])
 mean=np.concatenate(raw).mean(0,dtype=np.float64);scale=np.concatenate(raw).std(0,dtype=np.float64);dim=2049;gram=np.zeros((dim,dim));rhs=np.zeros((dim,4))
 for e in fits:
  v=load(e,mean,scale);X=features(v);gram+=X.T@X;rhs+=X.T@v[512:]
 penalty=np.eye(dim);penalty[-1,-1]=0;W=np.linalg.solve(gram+penalty,rhs);np.savez_compressed(O/'NORMAL_FIT_RIDGE.npz',weight=W,mean=mean,scale=scale,ridge_alpha=np.array(1.))
 calibrations={};methods=('ARX_Ridge512','persistence','HistoryMean512')
 for method in methods:
  errors=[]
  for e in m['roles']:
   if e['role']=='CAL':
    v=load(e,mean,scale);errors.append(v[512:]-predict(v,W,method))
  residual=np.concatenate(errors);sc=np.maximum(np.median(np.abs(residual),axis=0),1e-6);score=(np.abs(residual)/sc).mean(1);tau=float(np.quantile(score,.99,method='higher'));calibrations[method]=dict(scale=sc.tolist(),threshold=tau,normal_CAL_RMSE=float(np.sqrt(np.mean(residual**2))))
  out=O/method;out.mkdir()
  for e in m['roles']:
   if e['role']!='QUERY':continue
   v=load(e,mean,scale);pred=predict(v,W,method);res=v[512:]-pred;s=(np.abs(res)/sc).mean(1);case=e['meta']['case'];seed=e['meta']['seed'];np.savez_compressed(out/(case+'_'+str(seed)+'.npz'),index=np.arange(512,len(v)),prediction=pred,observed=v[512:],score=s,available_index=np.arange(512,len(v)),valid=np.ones(len(s),bool))
 dump(O/'CALIBRATIONS.json',calibrations);banks=[p for p in O.rglob('*.npz')];dump(O/'ALL_RAW_SEALED_BEFORE_LABELS.json',dict(labels_read=False,banks=[dict(path=str(p),sha256=sha(p))for p in banks],FIT_seeds=[e['meta']['seed']for e in fits],alpha=1.,scaler_role='FIT_only',lag_features='all4channels lag1..512, no current observation'))
 # Diagnostic labels are consulted after fitting/calibration and every score is fixed.
 from sklearn.metrics import average_precision_score,roc_auc_score
 rows=[]
 for method in methods:
  for e in m['roles']:
   if e['role']!='QUERY':continue
   with np.load(O/method/(e['meta']['case']+'_'+str(e['meta']['seed'])+'.npz'))as z:s=z['score'];obs=z['observed'];pred=z['prediction'];idx=z['index']
   with np.load(e['truth'])as z:y=z['labels'][idx]
   tau=calibrations[method]['threshold'];hit=s>tau;steady=(idx>=3584)&(idx<5120)
   rows.append(dict(method=method,case=e['meta']['case'],seed=e['meta']['seed'],AP=float(average_precision_score(y,s))if y.any()else None,ROC=float(roc_auc_score(y,s))if y.any()else None,TP=int((hit&y.astype(bool)).sum()),FP=int((hit&~y.astype(bool)).sum()),steady_RMSE=float(np.sqrt(np.mean((obs[steady]-pred[steady])**2))),steady_hits=int(hit[steady].sum()),normal_CAL_RMSE=calibrations[method]['normal_CAL_RMSE']))
 dump(O/'CPU_MECHANISM_RESULTS.json',dict(status='COMPLETED_FIXED_BASELINES',rows=rows,seconds=time.monotonic()-start));print(json.dumps(dict(status='CPU_ARX_GATE_COMPLETE',seconds=time.monotonic()-start,normal_cal=calibrations)),flush=True)
if __name__=='__main__':main()
