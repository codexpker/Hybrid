"""Fixed full-window and same-weight endpoint ranking on legal common axes."""
import argparse
import ast
import csv
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np
import pandas as pd
from . import native_event_space as nasa
from . import native_event_smd28 as smd
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .native_event_rank_readout import LABEL_SHA, asset
from .native_event_smd28_metrics import LABELS, LABELS_SHA
from .native_v_conditioning_runner import read, dump, sha
from .left_swat_replay_repeats import write_csv

OUT=Path('results/native_full_window_20261124_v1')
METRICS=('AP','ROC_AUC','VUS_PR','VUS_ROC')
WEIGHTS=('frozen_native_V','2025_copy','2025_point_weighted','2026_copy','2026_point_weighted','2027_copy','2027_point_weighted','2025_E')
METHODS=('official_V','official_E','official_V_plus_E','2025_copy','2026_copy','2027_copy','2025_point_weighted','2026_point_weighted','2027_point_weighted','Etrain_E')
MAP={'copy_V':'2025_copy','Vtrain_V':'2025_point_weighted'}
AUTHOR=Path('preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py')


def jobs(w):
    result=[('PSM','PSM')]
    result.extend((d,e) for d in ('MSL','SMAP') for e in nasa.entity_list(w,d))
    result.extend(('SMD',r['id']) for r in smd.records(w))
    assert len(result)==109
    return result


def checked_bank(path,receipt=None,sha_key='sha256'):
    r=read(receipt or path.with_suffix('.json'))
    assert sha(path)==r[sha_key],str(path)
    return np.load(path,allow_pickle=False),r


def truth(w,d,e):
    if d=='PSM':return np.load(asset(w,'data/common_psm_20261028_v1/test_label.npy',LABEL_SHA)).reshape(-1)
    if d=='SMD':
        assert sha(w/LABELS)==LABELS_SHA
        r=next(x for x in read(w/LABELS)['records'] if x['id']==e)
        with np.load(asset(w,r['path'],r['sha256'])) as z:return z['label'].reshape(-1)
    with (w/nasa.MSL/'labels.csv').open() as f:r=next(x for x in csv.DictReader(f) if x['spacecraft']==d and x['chan_id']==e)
    y=np.zeros(int(r['num_values']),np.int8)
    for a,b in ast.literal_eval(r['anomaly_sequences']):y[a:b+1]=1
    return y


def causal(w,d,e):
    scores={};bindings=[]
    if d=='PSM':
        path=w/'results/native_e_vs_v_psm_20261123_v1/banks.npz'
        z,r=checked_bank(path,path.with_name('SCORES.json'))
        with z:
            ix=z['index'];scores.update({MAP.get(k,k):z['query'][:,j] for j,k in enumerate(z['keys'].tolist()) if k in ('official_V','official_E','official_V_plus_E','copy_V','Vtrain_V','Etrain_E')})
        bindings.append(dict(path=str(path.relative_to(w)),sha256=r['sha256']))
        path=w/'results/native_event_rank_seed_repeats_20261122_v1/PSM/banks.npz'
        z,r=checked_bank(path,path.with_name('SCORES_FROZEN.json'))
        with z:
            np.testing.assert_array_equal(z['index'],ix)
            scores.update({k:z['score'][:,j] for j,k in enumerate(z['keys'].tolist())})
        bindings.append(dict(path=str(path.relative_to(w)),sha256=r['sha256']))
    else:
        path=w/(Path('results/native_e_vs_v_smd28_20261123_v1') if d=='SMD' else Path('results/native_e_vs_v_20261123_v1/space')/d)/(e+'.npz')
        z,r=checked_bank(path)
        with z:
            ix=z['query_index'];scores.update({MAP.get(k,k):z['query'][:,j] for j,k in enumerate(z['keys'].tolist()) if k in ('official_V','official_E','official_V_plus_E','copy_V','Vtrain_V','Etrain_E')})
        bindings.append(dict(path=str(path.relative_to(w)),sha256=r['sha256']))
        path=w/(smd.OUT if d=='SMD' else Path('results/native_event_space_repeat_20261122_v1')/d)/(e+'.npz')
        z,r=checked_bank(path)
        with z:
            np.testing.assert_array_equal(z['query_index'],ix)
            keys=z['keys'].tolist() if 'keys' in z else r['arms']
            scores.update({k:z['query'][:,j] for j,k in enumerate(keys) if k in METHODS})
        bindings.append(dict(path=str(path.relative_to(w)),sha256=r['sha256']))
    assert set(scores)==set(METHODS),(d,e,set(scores))
    return ix,scores,bindings


def left(w,d,e):
    if d=='PSM':
        manifest=read(w/'preparations/native_supervised_mlp_20261123_v1/PSM_SWAT_COMMON_AXIS.json')
        p=next(x for x in manifest['panels'] if x['dataset']=='PSM' and x['track']=='offline-readout')
        r=next(x for x in p['rows'] if x['method']=='LEFT native W192')
        path=asset(w,r['path'],r['sha256'])
        with np.load(path) as z:return z[r['index_key']].reshape(-1)-r['index_offset'],z[r['score_key']].reshape(-1),dict(path=r['path'],sha256=r['sha256'])
    if d=='SMD':
        path=w/'results/left_smd28_author_20261123_v1'/e/'scores.npz'
        z,r=checked_bank(path,path.with_name('SCORES.json'),'bank_sha256')
    else:
        root=w/'results/left_nasa_author_20261123_v1'/d
        r=next(x for x in read(root/'SCORES.json')['banks'] if x['entity']==e)
        path=asset(root,e+'.npz',r['bank_sha256']);z=np.load(path)
    with z:
        ix=z['index'];s=z['score'].reshape(-1);a=z['available_index']
        W=96 if d=='MSL' else 192
        np.testing.assert_array_equal(a,(ix//W+1)*W-1)
    return ix,s,dict(path=str(path.relative_to(w)),sha256=r['bank_sha256'])


def entity_job(args):
    w,d,e=args
    root=w/OUT;path=root/d/(e+'.npz');r=read(path.with_suffix('.json'))
    dest=root/'metrics_parts'/d/(e+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        result=read(dest);assert result['full_bank_sha256']==r['sha256'];return result
    z,r=checked_bank(path)
    y=truth(w,d,e);n=len(y)
    with z:
        ix=z['index'];V=z['V'];E=z['E'];keys=z['keys'].tolist()
        np.testing.assert_array_equal(ix,np.arange(n//100*100))
        np.testing.assert_array_equal(z['window_start'],ix//100*100)
        np.testing.assert_array_equal(z['available_index'],(ix//100+1)*100-1)
    assert keys==list(WEIGHTS) and V.shape==E.shape==(len(ix),8)
    full={'official_V':V[:,0],'official_E':E[:,0],'official_V_plus_E':V[:,0]+E[:,0],'Etrain_E':E[:,-1]}
    full.update({k:V[:,j] for j,k in enumerate(keys) if k in METHODS})
    assert set(full)==set(METHODS)
    ci,cs,cb=causal(w,d,e);li,ls,lb=left(w,d,e)
    axis=np.intersect1d(np.intersect1d(ix,ci,assume_unique=True),li,assume_unique=True)
    np.testing.assert_array_equal(axis,np.arange(99,min(len(ix),len(li))))
    cp=np.searchsorted(ci,axis);fp=np.searchsorted(ix,axis);lp=np.searchsorted(li,axis)
    assert len(axis)>0 and np.isfinite(V).all() and np.isfinite(E).all()
    # Every W100 endpoint is also the same W100 input in archived causal scoring.
    ends=ix[99::100];where=np.searchsorted(ci,ends)
    for k in METHODS:
        np.testing.assert_allclose(full[k][99::100],cs[k][where],rtol=2e-5,atol=2e-6,err_msg=f'{d}/{e}/{k} identical-window endpoint')
    metric=load_author_metric(w/AUTHOR);rows=[]
    def add(method,mode,kind,points,score):
        m,_=continuous_metrics(y[points],np.asarray(score,dtype=np.float64),metric)
        rows.append(dict(domain=d,entity=e,method=method,readout=mode,axis=kind,first=int(points[0]),stop=int(points[-1]+1),points=len(points),**{k:m[k] for k in METRICS}))
    for k in METHODS:
        add(k,'full_window','native_full',ix,full[k])
        add(k,'full_window','common',axis,full[k][fp])
        add(k,'causal_endpoint','common',axis,cs[k][cp])
    add('LEFT','full_window','common',axis,ls[lp])
    add('LEFT','full_window','native_full',li,ls)
    result=dict(domain=d,entity=e,full_bank_sha256=r['sha256'],input_points=n,full_points=len(ix),common_points=len(axis),left_points=len(li),causal_bindings=cb,left_binding=lb,rows=rows)
    dump(dest,result)
    return result


def summaries(w,results):
    root=w/OUT;rows=[r for result in results for r in result['rows']]
    frame=pd.DataFrame(rows);frame.to_csv(root/'entity_metrics.csv',index=False)
    macro=frame.groupby(['domain','readout','axis','method'],sort=False)[list(METRICS)].mean().reset_index()
    macro.to_csv(root/'macro.csv',index=False)
    common=frame[frame.axis=='common'].set_index(['domain','entity','method','readout'])
    differences=[];distribution=[];seedmacro=[]
    for d in ('PSM','SMD','MSL','SMAP'):
        data=common.loc[d]
        contrasts=[(k,'full_window',k,'causal_endpoint','readout_change') for k in METHODS]
        for readout in ('full_window','causal_endpoint'):
            for seed in (2025,2026,2027):
                for ref in ('official_V',f'{seed}_copy','LEFT'):
                    refmode='full_window' if ref=='LEFT' else readout
                    contrasts.append((f'{seed}_point_weighted',readout,ref,refmode,'source_learning' if ref!='LEFT' else 'versus_LEFT'))
        for method,mode,reference,refmode,kind in contrasts:
            a=data.xs((method,mode),level=('method','readout'))[list(METRICS)]
            b=data.xs((reference,refmode),level=('method','readout'))[list(METRICS)]
            delta=a-b
            for entity,values in delta.iterrows():
                differences.append(dict(domain=d,entity=entity,method=method,readout=mode,reference=reference,reference_readout=refmode,contrast=kind,**values.to_dict()))
            for metric in METRICS:
                values=delta[metric]
                distribution.append(dict(domain=d,method=method,readout=mode,reference=reference,reference_readout=refmode,contrast=kind,metric=metric,macro_delta=values.mean(),median_delta=values.median(),wins=int((values>0).sum()),ties=int((values==0).sum()),losses=int((values<0).sum()),entities=len(values)))
        for mode in ('full_window','causal_endpoint'):
            m=macro[(macro.domain==d)&(macro.axis=='common')&(macro.readout==mode)].set_index('method')
            for family in ('copy','point_weighted'):
                v=m.loc[[f'{seed}_{family}' for seed in (2025,2026,2027)],list(METRICS)]
                seedmacro.append(dict(domain=d,readout=mode,family=family,n=3,ddof=1,**{k+'_mean':v[k].mean() for k in METRICS},**{k+'_sd':v[k].std(ddof=1) for k in METRICS}))
    pd.DataFrame(differences).to_csv(root/'paired_entity.csv',index=False)
    pd.DataFrame(distribution).to_csv(root/'paired_distribution.csv',index=False)
    pd.DataFrame(seedmacro).to_csv(root/'repeats_common.csv',index=False)
    # Paired training-seed effects are computed after equal-entity aggregation.
    diff=pd.DataFrame(differences);diffs=[]
    for d in ('PSM','SMD','MSL','SMAP'):
        for mode in ('full_window','causal_endpoint'):
            for ref in ('official_V','copy','LEFT'):
                chosen=diff[(diff.domain==d)&(diff.readout==mode)&diff.method.str.endswith('_point_weighted')&(diff.reference.str.endswith('_copy') if ref=='copy' else (diff.reference==ref))]
                v=chosen.groupby('method')[list(METRICS)].mean()
                assert len(v)==3
                diffs.append(dict(domain=d,readout=mode,reference=ref,n=3,ddof=1,**{k+'_mean':v[k].mean() for k in METRICS},**{k+'_sd':v[k].std(ddof=1) for k in METRICS}))
        for family in ('copy','point_weighted'):
            chosen=diff[(diff.domain==d)&(diff.contrast=='readout_change')&diff.method.str.endswith('_'+family)]
            v=chosen.groupby('method')[list(METRICS)].mean();assert len(v)==3
            diffs.append(dict(domain=d,readout='full_minus_endpoint',reference=family,n=3,ddof=1,**{k+'_mean':v[k].mean() for k in METRICS},**{k+'_sd':v[k].std(ddof=1) for k in METRICS}))
    pd.DataFrame(diffs).to_csv(root/'paired_seed_summary.csv',index=False)
    dump(root/'METRIC_BINDINGS.json',dict(entities=[{k:v for k,v in r.items() if k!='rows'} for r in results],axes='common=intersection of causal, fullW100 and native LEFT scored points; entity-first equal macro',metrics='pinned LEFT author continuous_metrics: AP,ROC,RangeAUC_volume_opt(windowSize100,thre250)',calibration='No new threshold or operation metrics',MLP='Retained only in prior causal panels; no window-logit broadcast'))


def run(w):
    alljobs=jobs(w);pending=set(alljobs);active={};done=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        while pending or active:
            for d,e in sorted(pending):
                if len(active)>=6:break
                if (w/OUT/d/(e+'.json')).exists():
                    f=pool.submit(entity_job,(w,d,e));active[f]=(d,e);pending.remove((d,e))
            if active:
                ready,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for f in ready:
                    name=active.pop(f);result=f.result();done.append(result);print(name,len(done),len(alljobs),flush=True)
            else:time.sleep(5)
    summaries(w,done)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','one']);p.add_argument('--domain');p.add_argument('--entity');a=p.parse_args()
    if a.action=='run':run(Path.cwd())
    else:entity_job((Path.cwd(),a.domain,a.entity))
