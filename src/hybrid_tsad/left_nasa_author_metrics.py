"""Entity-first shared-LEFT NASA benchmarks and cached causal detector intersections."""
import argparse
import ast
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from .left_swat_author import read,dump,sha
from .left_native_swat_metrics import continuous_metrics,load_author_metric
from .left_nasa_author import OUT
from . import native_event_space as nasa
from .native_v_conditioning_runner import dump

METRICS=['AP','ROC_AUC','VUS_PR','VUS_ROC']

def entity(w,domain,e,record):
    root=w/OUT/domain;seal=read(root/'SCORES.json');leftref=next(r for r in seal['banks'] if r['entity']==e)
    p=root/(e+'.npz');assert sha(p)==leftref['bank_sha256']
    with np.load(p) as z:
        ix=z['index'];left=z['score'];available=z['available_index']
    W=96 if domain=='MSL' else 192
    n=leftref['input_points'];assert n==int(record['num_values'])
    np.testing.assert_array_equal(ix,np.arange(n//W*W))
    np.testing.assert_array_equal(available,(ix//W+1)*W-1)
    y=np.zeros(n,dtype=np.int8)
    for a,b in ast.literal_eval(record['anomaly_sequences']):y[a:b+1]=1
    metric=load_author_metric(w/'preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py')
    rows=[]
    full,_=continuous_metrics(y[ix],left,metric)
    rows.append(dict(domain=domain,entity=e,method='LEFT',axis='full',first=0,stop=len(ix),points=len(ix),**{k:full[k] for k in METRICS}))
    axis=np.arange(99,len(ix));scores={'LEFT':left[99:]};binds={}
    for family,path in [('e_vs_v',Path('results/native_e_vs_v_20261123_v1/space')),('MLP',Path('results/native_supervised_mlp_20261123_v1/readout')),('repeats',Path('results/native_event_space_repeat_20261122_v1'))]:
        p=w/path/domain/(e+'.npz');r=read(p.with_suffix('.json'));assert sha(p)==r['sha256']
        with np.load(p) as z:
            qi=z['query_index'];take=np.searchsorted(qi,axis)
            np.testing.assert_array_equal(qi[take],axis)
            vals=z['query'][take];keys=z['keys'].tolist() if 'keys' in z else r['arms']
        for j,key in enumerate(keys):scores[str(key)]=vals[:,j]
        binds[family]=dict(path=str(p.relative_to(w)),sha256=r['sha256'])
    for method,score in scores.items():
        values,_=continuous_metrics(y[axis],score,metric)
        rows.append(dict(domain=domain,entity=e,method=method,axis='common',first=99,stop=len(ix),points=len(axis),**{k:values[k] for k in METRICS}))
    return rows,dict(domain=domain,entity=e,LEFT_sha256=leftref['bank_sha256'],full_points=len(ix),common_points=len(axis),omitted_tail=n-len(ix),**binds)


def evaluate(w):
    labels={}
    with (w/nasa.MSL/'labels.csv').open() as f:
        for r in csv.DictReader(f):labels[r['chan_id']]=r
    jobs=[(w,d,e,labels[e]) for d in ['MSL','SMAP'] for e in nasa.entity_list(w,d)]
    rows=[];bindings=[]
    with ProcessPoolExecutor(max_workers=6) as p:
        for rr,b in p.map(run_one,jobs):rows.extend(rr);bindings.append(b)
    root=w/OUT
    frame=pd.DataFrame(rows);frame.to_csv(root/'entity_metrics.csv',index=False)
    frame.groupby(['domain','axis','method'],sort=False)[METRICS].mean().reset_index().to_csv(root/'macro.csv',index=False)
    common=frame[frame.axis=='common'].set_index(['domain','entity','method'])
    paired=[]
    for d in ['MSL','SMAP']:
        ix=common.loc[d]
        for method in ['official_V','official_E','official_V_plus_E','copy_V','Vtrain_V','Etrain_E','MLP_full','MLP_240','2026_copy','2026_point_weighted','2027_copy','2027_point_weighted']:
            a=ix.xs(method,level='method')[METRICS];b=ix.xs('LEFT',level='method')[METRICS]
            diff=a-b
            for metric in METRICS:
                vals=diff[metric]
                paired.append(dict(domain=d,method=method,reference='LEFT',metric=metric,macro_delta=vals.mean(),median_delta=vals.median(),wins=int((vals>0).sum()),ties=int((vals==0).sum()),losses=int((vals<0).sum()),entities=len(vals)))
    pd.DataFrame(paired).to_csv(root/'paired.csv',index=False)
    # Recompute repetition summaries on this exact LEFT intersection.
    repeats=[]
    for d in ['MSL','SMAP']:
        macro=frame[(frame.domain==d)&(frame.axis=='common')].groupby('method')[METRICS].mean()
        for family,seed25 in [('copy','copy_V'),('point_weighted','Vtrain_V')]:
            keys=[seed25,'2026_'+family,'2027_'+family]
            v=macro.loc[keys]
            repeats.append(dict(domain=d,method=family,n=3,ddof=1,**{k+'_mean':v[k].mean() for k in METRICS},**{k+'_sd':v[k].std(ddof=1) for k in METRICS}))
    pd.DataFrame(repeats).to_csv(root/'repeats_common.csv',index=False)
    dump(root/'METRIC_BINDINGS.json',dict(entities=bindings,implementation='pinned LEFT RangeAUC_volume_opt, windowSize100, thre250',entity_macro='equal weight after per-entity metric',threshold_operations='NA: domain scaler includes target normal validation/project CAL'))


def run_one(args):return entity(*args)

if __name__=='__main__':evaluate(Path.cwd())
