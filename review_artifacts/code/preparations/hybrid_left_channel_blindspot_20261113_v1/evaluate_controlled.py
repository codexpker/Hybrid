"""Independent controlled runs, fixed scores and per-run evaluation, no point concatenation."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from fixed_metrics import evaluate,rolling_max64
from evaluate_fixed import load,sha,table,P,O
def main():
    base=O/'controlled';out=O/'evaluation'/'controlled';out.mkdir(parents=True,exist_ok=False)
    spec=json.loads((P/'controlled/ARRAYS.json').read_text());sources=[];by={}
    for arm in ['native','blind']:
        rc=json.loads((base/arm/'RECEIPT.json').read_text());assert rc['status']=='COMPLETED'
        bank={}
        for role in spec:
            if role in ['FIT','SELECT']:continue
            f=base/arm/(role+'.npz');z=load(f);assert np.array_equal(z['index'],np.arange(99,2560));assert np.array_equal(z['available_index'],z['index']);sources.append(dict(path=str(f),sha256=sha(f)));bank[role]=z
        for name,key in ([('LEFT_native','native_score'),('LEFT_native_trained_masked_readout','score')]if arm=='native'else[('Hybrid_blindspot','score'),('Hybrid_nativeoff','native_score')]):by[name]={r:z[key]for r,z in bank.items()}
    by['LEFT_past64max']={r:np.stack([rolling_max64(s)for s in arr])for r,arr in by['LEFT_native'].items()}
    fit=load(P/spec['FIT']['path'])['values'].astype(float);mean=fit.mean((0,1));scale=fit.std((0,1));scale=np.where(scale>0,scale,1.)
    ff=((fit-mean)/scale).reshape(-1,3);fi=np.linspace(0,len(ff)-1,min(8192,len(ff)),dtype=int);models=[Ridge(alpha=1.).fit(ff[fi][:,np.arange(3)!=j],ff[fi,j])for j in range(3)]
    for method in ['MeanFIT','InstantRidge','HistoryMean512']:
        by[method]={}
        for role,e in spec.items():
            if role in ['FIT','SELECT']:continue
            x=(load(P/e['path'])['values']-mean)/scale;ix=np.arange(99,2560);pred=np.zeros((len(x),len(ix),3))
            if method=='InstantRidge':pred=np.stack([np.column_stack([m.predict(v[ix][:,np.arange(3)!=j])for j,m in enumerate(models)])for v in x])
            elif method=='HistoryMean512':
                ix=np.arange(512,2560);cs=np.concatenate([np.zeros((len(x),1,3)),np.cumsum(x,axis=1)],axis=1);pred=(cs[:,ix]-cs[:,ix-512])/512
            by[method][role]=np.square(x[:,ix]-pred).mean(2)
    (out/'RAW_SEAL.json').write_text(json.dumps(dict(status='SEALED_BEFORE_TRUTH_JOIN',sources=sources,methods=list(by),fixed_rule='each method own normalCAL q99higher; independent runs not concatenated; full model99 axis and allmethod512 axis'),indent=2))
    truth=load(P/'controlled/TRUTH.npz');rows=[];ev=[]
    for scope,start in [('model_complete',99),('all_methods_common',512)]:
        ix=np.arange(start,2560)
        for method,banks in by.items():
            if start==99 and method=='HistoryMean512':continue
            orig=512 if method=='HistoryMean512' else 99;cal=banks['CAL'][:,start-orig:].reshape(-1)
            for case in truth:
                for run in range(5):
                    rr,ee=evaluate(ix,banks[case][run,start-orig:],cal,truth[case][run],steady_after=100)
                    rows.extend(dict(scope=scope,method=method,case=case,run=90000+run,**x)for x in rr);ev.extend(dict(scope=scope,method=method,case=case,run=90000+run,**x)for x in ee)
    table(out/'per_run.csv',rows);table(out/'per_event.csv',ev)
    (out/'RECEIPT.json').write_text(json.dumps(dict(status='COMPLETED',cases=list(truth),test_runs=5,normal_controls_OOD=True,no_QUERY_selection=True,steady_after=100,long_min_length=512),indent=2))
if __name__=='__main__':main()
