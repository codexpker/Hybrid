"""Fixed FIT-only CPU controls; never opens labels or chooses test parameters."""
from pathlib import Path
import json,hashlib
import numpy as np
from sklearn.linear_model import Ridge
P=Path(__file__).resolve().parent; R=P.parents[1]; O=R/'results'/P.name
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def data(domain,role):
    if domain=='PSM':
        base=R/'preparations/hybrid_native_predictive_cycle_20261113_v1';p=base/'data'/('PSM_'+role+'.npz');z=load(p);v=z['values'][0].astype(float);start=int(z['original_start'][0]);sf=base/'PSM_FIT_ONLY_SCALER.json'
    else:
        base=R/'preparations/left_multievent_mixed_u_20261111_v1';p=base/'data'/(role+'.npz');z=load(p);v=z['values'].astype(float);starts={'FIT':0,'CAL':445661,'QUERY':608023};start=starts[role];sf=base/'FIT_ONLY_SCALER.json'
    s=json.loads(sf.read_text());v=(v-np.array(s['mean']))/np.array(s['scale']);assert np.isfinite(v).all()
    return v,start,dict(path=str(p),sha256=sha(p),scaler=str(sf),scaler_sha256=sha(sf))
def controls(domain):
    out=O/'cpu_controls'/domain;out.mkdir(parents=True,exist_ok=False)
    fit,_,fitid=data(domain,'FIT'); C=fit.shape[1];ix=np.linspace(0,len(fit)-1,min(8192,len(fit)),dtype=int);models=[]
    for j in range(C):
        cols=np.arange(C)!=j;models.append(Ridge(alpha=1.).fit(fit[ix][:,cols],fit[ix,j]))
    np.savez_compressed(out/'Ridge_parameters.npz',coefficient=np.stack([m.coef_ for m in models]),intercept=np.array([m.intercept_ for m in models]),FIT_index=ix)
    receipts=[]
    for role in ['CAL','QUERY']:
        x,start,identity=data(domain,role);ends=np.arange(512,len(x));cs=np.vstack([np.zeros(C),np.cumsum(x,axis=0)]);pred=(cs[ends]-cs[ends-512])/512
        ridge=np.column_stack([m.predict(x[ends][:,np.arange(C)!=j])for j,m in enumerate(models)])
        scores={'HistoryMean512':np.square(x[ends]-pred).mean(1),'MeanFIT':np.square(x[ends]).mean(1),'InstantRidge':np.square(x[ends]-ridge).mean(1)}
        for method,score in scores.items():
            f=out/(method+'_'+role+'.npz');np.savez_compressed(f,index=ends+start,available_index=ends+start,score=score,target_standardized=x[ends]);receipts.append(dict(method=method,role=role,path=str(f),sha256=sha(f),points=len(ends),input=identity))
    (out/'RECEIPT.json').write_text(json.dumps(dict(FIT=fitid,alpha=1,FIT_count=len(ix),score='mean_C squared FIT-standardized residual',labels_read=False,entries=receipts),indent=2))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('domain',choices=['PSM','SWAT']);a=ap.parse_args();controls(a.domain)
