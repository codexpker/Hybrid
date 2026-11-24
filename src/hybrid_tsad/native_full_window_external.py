"""Cache-only external panels on their declared intersection axes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from .native_full_window_metrics import OUT,METHODS,METRICS,truth,causal,left,jobs,AUTHOR
from .native_v_conditioning_runner import read,dump,sha
from .native_event_rank_readout import asset
from .left_native_swat_metrics import continuous_metrics,load_author_metric
from .native_rank_psm_extension import EXTERNAL
from .native_full_window_tables import table,number

RCD=Path('results/timercd_fixed_20261117_v1')


def timercd(w,domain,entity,axis):
    record='PSM' if domain=='PSM' else entity
    r=read(w/RCD/'RAW_BANK_SEAL.json')
    assert r['status']=='COMPLETED'
    result=[]
    for arm in ('A_official_role','B_fit_only'):
        rel=f'{record}/{arm}/QUERY.npz';entry=next(x for x in r['banks'] if x['relative']==rel)
        path=asset(w/RCD,rel,entry['sha256'])
        with np.load(path) as z:
            ix=z['index'];where=np.searchsorted(ix,axis)
            np.testing.assert_array_equal(ix[where],axis)
            np.testing.assert_array_equal(z['logit'],z['head_logits'][:,1]-z['head_logits'][:,0])
            assert z['valid'].all() and len(ix)==len(truth(w,domain,entity))
            score=z['logit'][where].astype(float)
        result.append((f'TimeRCD {arm} logit',score,'official pretrained discriminative network; W5000 offline; '+('whole-role statistics including QUERY' if arm.startswith('A') else 'archived target FIT-only statistics'),entry['sha256']))
    return result


def external_one(args):
    w,d,e=args;root=w/OUT
    result=root/'external_parts'/d/(e+'.json');result.parent.mkdir(parents=True,exist_ok=True)
    if result.exists():return read(result)['rows']
    left_ix,_,_=left(w,d,e);raw_n=len(truth(w,d,e));stop=min(raw_n//100*100,len(left_ix));axis=np.arange(512 if d=='PSM' else 99,stop)
    y=truth(w,d,e);scores=timercd(w,d,e,axis)
    if d=='PSM':
        binding=read(w/'preparations/native_supervised_mlp_20261123_v1/PSM_SWAT_COMMON_AXIS.json')
        panel=next(p for p in binding['panels'] if p['dataset']=='PSM' and p['track']=='offline-readout')
        for r in panel['rows']:
            if r['method'] not in ('LEFT native W192','CATCH W192','PCA residual'):continue
            path=asset(w,r['path'],r['sha256'])
            with np.load(path) as z:
                ix=z[r['index_key']].reshape(-1)-r['index_offset'];where=np.searchsorted(ix,axis)
                np.testing.assert_array_equal(ix[where],axis);score=z[r['score_key']].reshape(-1)[where]
            name=r['method']+(f" seed{r['archived_seed']}" if r['method']=='CATCH W192' else '')
            scores.append((name,score,r['permission'],r['sha256']))
        for name in ('HistoryMean512',):
            rel,digest=EXTERNAL[name];path=asset(w,rel,digest)
            with np.load(path) as z:
                where=np.searchsorted(z['index'],axis);np.testing.assert_array_equal(z['index'][where],axis);score=z['score'].reshape(-1)[where]
            scores.append((name,score,'past512 causal residual; archived target normal statistics',digest))
        path=root/d/(e+'.npz');receipt=read(path.with_suffix('.json'));assert sha(path)==receipt['sha256']
        with np.load(path) as z:
            keys=z['keys'].tolist();vals=z['V'];errors=z['E'];index=z['index'];take=np.searchsorted(index,axis)
            full={'official_V':vals[:,0],'official_E':errors[:,0],'official_V_plus_E':vals[:,0]+errors[:,0],'Etrain_E':errors[:,-1]}
            full.update({k:vals[:,j] for j,k in enumerate(keys) if k in METHODS})
            for k,s in full.items():scores.append(('full_window '+k,s[take],'project fixed weight W100/stride100 offline',receipt['sha256']))
        ix,cs,refs=causal(w,d,e);take=np.searchsorted(ix,axis);np.testing.assert_array_equal(ix[take],axis)
        for k,s in cs.items():scores.append(('causal_endpoint '+k,s[take],'archived causal W100 B1 same masks',str(refs)))
    metric=load_author_metric(w/AUTHOR);rows=[]
    for method,s,permission,digest in scores:
        m,_=continuous_metrics(y[axis],np.asarray(s,float),metric)
        rows.append(dict(domain=d,entity=e,method=method,first=int(axis[0]),stop=int(axis[-1]+1),points=len(axis),permission=permission,cache_sha256=digest,**{k:m[k] for k in METRICS}))
    dump(result,dict(domain=d,entity=e,rows=rows))
    return rows


def run(w):
    selected=[(w,d,e) for d,e in jobs(w) if d in ('PSM','SMD')]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(external_one,selected):rows.extend(result)
    root=w/OUT;frame=pd.DataFrame(rows);frame.to_csv(root/'external_entity.csv',index=False)
    frame.groupby(['domain','method','permission'])[list(METRICS)].mean().reset_index().to_csv(root/'external_macro.csv',index=False)
    text=['# Compatible external cache panels','', 'PSM uses [512,87744), jointly covered by W100, causal endpoints, LEFT/CATCH and HistoryMean512. SMD uses the main per-machine intersection [99,min(100 floor(N/100),192 floor(N/192))). These panels preserve the external models’ original training, normalization and context permissions. TimeRCD uses its already frozen raw class-logit difference, with A/B both retained. A normalizes its whole passed role, including query observations; B uses archived FIT-only statistics. Neither is a causal alarm detector. There is no model inference or new threshold.']
    for d in ('PSM','SMD'):
        data=frame[frame.domain==d].groupby('method')[list(METRICS)].mean()
        text.extend(['',f'## {d}','',table(['Method',*METRICS],[[name,*[number(r[k]) for k in METRICS]] for name,r in data.iterrows()])])
    text.extend(['','For SMD, the main common-axis native/LEFT rows from Table1 are on these exact same points. The underlying [external entity rows](external_entity.csv) retain source score SHA and permission. Missing NASA TimeRCD/CATCH/PCA caches are NA; no inference was added to complete them.'])
    (root/'TABLE_EXTERNAL.md').write_text('\n'.join(text)+'\n',encoding='utf8')

if __name__=='__main__':run(Path.cwd())
