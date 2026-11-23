"""Frozen official representations and two source-selected MLP readouts on fixed public axes."""
import argparse
import csv
import fcntl
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from . import native_event_smd28 as smd
from . import native_event_space as nasa
from .native_e_vs_v_psm import AXIS, SUFFIX, inputs
from .native_v_conditioning_runner import read, dump, setup
from .native_v_conditioning import sha
from .native_event_rank_readout import asset, LABEL_SHA, METRIC_SHA
from .native_event_smd28_metrics import LABELS, LABELS_SHA
from .native_event_space_summary import RANKS, group_operating, read_csv
from .left_native_swat_metrics import load_author_metric, continuous_metrics
from .left_swat_replay_repeats import write_csv
from .native_v_conditioning_evaluation import exact_cal, causal_timing
from .timercd_swat_evaluation import evaluate_score

OUT=Path('results/native_supervised_mlp_20261123_v1/readout')
METHODS=('MLP_full','MLP_240')


def load_models(w):
    # This interface is filled once source selection is sealed, before target inference.
    from .native_supervised_mlp import load_heads
    from .native_v_conditioning import load_native
    import json
    for attempt in range(60):
        try:
            heads,seal=load_heads(w,'cuda')
            break
        except json.JSONDecodeError:
            time.sleep(1)
    else:
        raise RuntimeError('Incomplete source selection seal')
    assert seal['status']=='COMPLETED_BEFORE_ANY_QUERY'
    return load_native(w,'cuda'),heads,seal


def jobs(w):
    result=[('PSM','PSM',87742)]
    for domain in ('MSL','SMAP'):
        for entity in nasa.entity_list(w,domain):
            old=read(w/nasa.OUT/domain/f'{entity}.json')
            result.append((domain,entity,old['hash_identical_windows']))
    result.extend(('SMD',r['id'],r['query_points']) for r in smd.records(w))
    return sorted(result,key=lambda v:v[2],reverse=True)


def input_bank(w,domain,entity):
    if domain=='PSM':
        x,ci,sc=inputs(w)
        return np.stack([x[t-99:t+1] for t in ci]),np.stack([x[t-99:t+1] for t in AXIS]),ci,AXIS,dict(scaler_sha256=sc,query_size=len(x))
    if domain in ('MSL','SMAP'):
        windows,ci,qi,binding=nasa.read_roles(w,domain,entity)
        old=read(w/nasa.OUT/domain/f'{entity}.json')
        return windows[0][:,:,None],windows[1][:,:,None],ci,qi,dict(binding,query_size=old['query_size'])
    rec=next(r for r in smd.records(w) if r['id']==entity)
    cx,co=smd.data(w,rec,'CAL');qx,qo=smd.data(w,rec,'QUERY');assert qo==0
    ci=np.linspace(99,len(cx)-1,512,dtype=np.int64)+co;qi=np.arange(99,len(qx))
    return np.stack([cx[t-co-99:t-co+1] for t in ci]),np.stack([qx[t-99:t+1] for t in qi]),ci,qi,dict(scaler_sha256=rec['scaler']['sha256'],query_size=len(qx))


@torch.inference_mode()
def score(w,lane,lanes):
    setup(12)
    encoder,heads,identity=load_models(w)
    same_head=all(torch.equal(v,heads[1].state_dict()[key]) for key,v in heads[0].state_dict().items())
    from .native_supervised_mlp import encode
    root=w/OUT;root.mkdir(parents=True,exist_ok=True)
    alljobs=jobs(w)
    for domain,entity,_ in alljobs[lane::lanes]+alljobs:
        dest=root/domain;dest.mkdir(exist_ok=True)
        bank=dest/f'{entity}.npz';receipt=bank.with_suffix('.json')
        with bank.with_suffix('.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if receipt.exists():
                old=read(receipt);assert sha(bank)==old['sha256'] and old['weights']==identity
                continue
            start=time.monotonic()
            ca,qu,ci,qi,binding=input_bank(w,domain,entity)
            windows=np.concatenate((ca,qu));n=len(windows)
            # NASA can contain exact repeated windows; only identical full inputs are reused.
            if domain in ('MSL','SMAP'):
                flat=np.ascontiguousarray(windows)
                _,first,inverse=np.unique(flat.reshape(len(flat),-1).view('V400').ravel(),return_index=True,return_inverse=True)
                windows=flat[first]
            else:inverse=np.arange(n)
            parts=dest/'parts'/entity;parts.mkdir(parents=True,exist_ok=True)
            vals=[]
            for lo in range(0,len(windows),1024):
                path=parts/f'{lo:06}.npz';meta=path.with_suffix('.json')
                if meta.exists():
                    r=read(meta);assert r['weights']==identity and sha(path)==r['sha256']
                    with np.load(path) as p:vals.append(p['scores'])
                    continue
                out=torch.empty((min(1024,len(windows)-lo),len(METHODS)),device='cuda')
                for j,x in enumerate(windows[lo:lo+1024]):
                    xx=torch.from_numpy(x).cuda()[None]
                    f=encode(encoder,xx)
                    out[j,0]=heads[0](f)
                    out[j,1]=out[j,0] if same_head else heads[1](f)
                ar=out.cpu().numpy();assert np.isfinite(ar).all()
                np.savez_compressed(path,scores=ar)
                dump(meta,dict(sha256=sha(path),weights=identity))
                vals.append(ar)
                print(domain,entity,lo+len(ar),len(windows),round(time.monotonic()-start,1),flush=True)
            values=np.concatenate(vals)[inverse]
            np.savez_compressed(bank,keys=METHODS,cal_index=ci,query_index=qi,cal=values[:len(ci)],query=values[len(ci):],cal_window_start=ci-99,query_window_start=qi-99,cal_available_index=ci,query_available_index=qi)
            dump(receipt,dict(dataset=domain,entity=entity,sha256=sha(bank),weights=identity,cal_points=len(ci),query_points=len(qi),unique_inputs=len(windows),same_head=same_head,**binding,seconds=time.monotonic()-start,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),labels_read=False))


def truth(w,domain,entity):
    if domain=='PSM':return np.load(asset(w,'data/common_psm_20261028_v1/test_label.npy',LABEL_SHA)).reshape(-1)
    if domain=='SMD':
        assert sha(w/LABELS)==LABELS_SHA
        rec=next(r for r in read(w/LABELS)['records'] if r['id']==entity)
        with np.load(asset(w,rec['path'],rec['sha256'])) as z:return z['label'].reshape(-1)
    import ast
    with (w/nasa.MSL/'labels.csv').open() as f:r=next(r for r in csv.DictReader(f) if r['spacecraft']==domain and r['chan_id']==entity)
    y=np.zeros(int(r['num_values']),int)
    for lo,hi in ast.literal_eval(r['anomaly_sequences']):y[lo:hi+1]=1
    return y


def evaluate_one(arg):
    w,domain,entity=arg;dest=w/OUT/domain;bank=dest/f'{entity}.npz'
    assert sha(bank)==read(bank.with_suffix('.json'))['sha256']
    y=truth(w,domain,entity)
    metric=load_author_metric(asset(w,'preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py',METRIC_SHA))
    rankrows,ops,events,common=[],[],[],[]
    with np.load(bank) as z:
        qi=z['query_index'];ci=z['cal_index'];np.testing.assert_array_equal(qi,np.arange(99,len(y)))
        for j,name in enumerate(METHODS):
            s=z['query'][:,j].astype(float);cal=z['cal'][:,j].astype(float)
            rank,_=continuous_metrics(y[qi],s,metric)
            rankrows.append(dict(dataset=domain,entity=entity,method=name,**rank))
            tau=float(np.quantile(cal,.99,method='higher'))
            if domain=='PSM':
                assert len(ci)==512 and y[ci].all()
                oi=SUFFIX;sv=s[oi-99]
                rr,ee,_,_=evaluate_score(oi,sv,y,tau,oi,query_start=int(oi[0]),query_stop=int(oi[-1])+1,original_length=len(y),expected_events=(48,5))
                causal_timing(rr,ee)
            else:rr,ee=nasa.operating_metrics(qi,s,y,tau,rank)
            identity=dict(dataset=domain,entity=entity,method=name)
            ops.extend(dict(r,**identity) for r in rr);events.extend(dict(e,**identity) for e in ee)
            if domain in ('PSM','SMAP','SMD'):
                keep=qi>=512 if domain!='SMD' else qi<(len(y)//192*192)
                m,_=continuous_metrics(y[qi[keep]],s[keep],metric)
                common.append(dict(**identity,points=int(keep.sum()),axis_start=int(qi[keep][0]),axis_stop=int(qi[keep][-1])+1,**m))
    return rankrows,ops,events,common



def cached_metrics(arg):
    w,domain,entity=arg
    root=w/OUT/domain/'metric_parts'/entity
    root.mkdir(parents=True,exist_ok=True)
    with (root/'work.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        source=read(w/OUT/domain/f'{entity}.json')['sha256']
        seal=root/'SOURCE.json'
        if seal.exists():
            r=read(seal);assert r['source_sha256']==source
            for file,digest in r['files'].items():assert sha(root/file)==digest
        else:
            groups=evaluate_one(arg)
            digests={}
            for key,rows in zip(('ranking','operating','events','common'),groups):
                if rows:
                    path=root/(key+'.csv');write_csv(path,rows);digests[path.name]=sha(path)
            dump(seal,dict(source_sha256=source,files=digests))
        output=[]
        for key in ('ranking','operating','events','common'):
            path=root/(key+'.csv')
            rows=read_csv(path) if path.exists() else []
            # Restore exact numeric scalars while leaving event status strings intact.
            if key in ('ranking','common'):
                for row in rows:
                    for metric in RANKS:row[metric]=float(row[metric])
                    if 'points' in row:row['points']=int(row['points'])
            if key=='operating':
                for row in rows:
                    for name in ('TP','TN','FP','FN','M','Npred','Ntrue','normal_alarm_fragments','auxiliary_any_hit_events'):
                        row[name]=int(row[name])
            output.append(rows)
        return output


def stage_metrics(w):
    pending={(d,e):(w,d,e) for d,e,_ in jobs(w)}
    with ProcessPoolExecutor(max_workers=4) as pool:
        while pending:
            ready=[key for key in pending if (w/OUT/key[0]/f'{key[1]}.json').exists()]
            for _ in pool.map(cached_metrics,[pending[key] for key in ready]):pass
            for key in ready:pending.pop(key)
            if pending:time.sleep(10)

def evaluate(w):
    result=[[],[],[],[]]
    js=[(w,d,e) for d,e,_ in jobs(w)]
    assert len(js)==109 and all((w/OUT/d/f'{e}.json').exists() for _,d,e in js)
    with ProcessPoolExecutor(max_workers=6) as pool:
        for group in pool.map(cached_metrics,js):
            for acc,part in zip(result,group):acc.extend(part)
    root=w/OUT
    for name,rows in zip(('entity_ranking','entity_operating','events','common_entity'),result):write_csv(root/f'{name}.csv',rows)
    ranks,ops,common=[] ,[],[]
    for domain in ('PSM','MSL','SMAP','SMD'):
        for method in METHODS:
            selected=[r for r in result[0] if r['dataset']==domain and r['method']==method]
            ranks.append(dict(dataset=domain,method=method,entities=len(selected),**{k:np.mean([r[k] for r in selected]) for k in RANKS}))
            for stream in ('strict','confirmed3'):
                rr=[r for r in result[1] if r['dataset']==domain and r['method']==method and r['stream']==stream]
                ee=[{k:'' if v is None else str(v) for k,v in r.items()} for r in result[2] if r['dataset']==domain and r['method']==method and r['stream']==stream]
                ops.append(dict(dataset=domain,method=method,stream=stream,**group_operating(rr,ee)))
            selected=[r for r in result[3] if r['dataset']==domain and r['method']==method]
            if selected:common.append(dict(dataset=domain,method=method,entities=len(selected),points=sum(r['points'] for r in selected),**{k:np.mean([r[k] for r in selected]) for k in RANKS}))
    for name,rows in (('ranking_macro',ranks),('operating',ops),('common_macro',common),('banks',[read(w/OUT/d/f'{e}.json') for _,d,e in js])):write_csv(root/f'{name}.csv',rows)


def external_psm(w):
    root=w/OUT
    binding=w/'preparations/native_supervised_mlp_20261123_v1/PSM_SWAT_COMMON_AXIS.json'
    manifest=read(binding)
    panel=next(p for p in manifest['panels'] if p['dataset']=='PSM' and p['track']=='offline-readout')
    axis=np.arange(512,87744)
    y=truth(w,'PSM','PSM')
    metric=load_author_metric(asset(w,'preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py',METRIC_SHA))
    rows=[]
    missing=[]
    def add(name,score,permission,identity):
        assert score.shape==axis.shape and np.isfinite(score).all()
        m,_=continuous_metrics(y[axis],score,metric)
        rows.append(dict(dataset='PSM',method=name,points=len(axis),axis_start=512,axis_stop=87744,permission=permission,cache_sha256=identity,**{k:m[k] for k in RANKS}))
    bank=root/'PSM/PSM.npz'
    rec=read(bank.with_suffix('.json'));assert sha(bank)==rec['sha256']
    with np.load(bank) as z:
        ix=np.searchsorted(z['query_index'],axis)
        np.testing.assert_array_equal(z['query_index'][ix],axis)
        for j,name in enumerate(METHODS):add(name,z['query'][ix,j].astype(float),'frozen official encoder; real public source-label MLP; W100 endpoint; target FIT scaler',rec['sha256'])
    old=w/'results/native_e_vs_v_psm_20261123_v1/banks.npz'
    rec=read(old.with_name('SCORES.json'));assert sha(old)==rec['sha256']
    with np.load(old) as z:
        ix=np.searchsorted(z['index'],axis);np.testing.assert_array_equal(z['index'][ix],axis)
        for j,name in enumerate(z['keys']):add(str(name),z['query'][ix,j].astype(float),'native causal W100; archived source permissions',rec['sha256'])
    for r in panel['rows']:
        if r['method'] not in ('LEFT native W192','CATCH W192','PCA residual'):continue
        if not (w/r['path']).exists():
            missing.append(dict(method=r['method'],seed=r['archived_seed'],relative=r['path'],reason='Archived bank unavailable on execution host; no replacement inference'))
            continue
        bank=asset(w,r['path'],r['sha256'])
        with np.load(bank) as z:
            ix=z[r['index_key']].reshape(-1)-r['index_offset']
            score=z[r['score_key']].reshape(-1)
            where=np.searchsorted(ix,axis);np.testing.assert_array_equal(ix[where],axis)
            name=r['method']+((' seed'+str(r['archived_seed'])) if r['method']=='CATCH W192' else '')
            add(name,score[where].astype(float),r['permission'],r['sha256'])
    write_csv(root/'PSM/external_compatible.csv',rows)
    dump(root/'PSM/EXTERNAL_MISSING.json',missing)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['score','evaluate','external','stage-metrics']);p.add_argument('--lane',type=int,default=0);p.add_argument('--lanes',type=int,default=4);a=p.parse_args()
    if a.action=='score':score(Path.cwd(),a.lane,a.lanes)
    elif a.action=='external':external_psm(Path.cwd())
    elif a.action=='stage-metrics':stage_metrics(Path.cwd())
    else:evaluate(Path.cwd())
