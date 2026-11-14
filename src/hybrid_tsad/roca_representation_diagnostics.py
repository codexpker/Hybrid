"""Frozen RoCA q/qr branch diagnostics. These outputs are never new detectors.

q = projection_head(CNN.flatten); qr also traverses LSTM encoder/decoder.
No cross-checkpoint center swap, QUERY reference fitting, or model calls.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bound(a):
    if sha(a['path'])!=a['sha256']:raise PermissionError('Frozen diagnostic input changed')
    return json.loads(Path(a['path']).read_text())
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def table(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def branch_distance(q,c):
    u=unit(np.asarray(q,np.float64));v=unit(np.asarray(c,np.float64))
    return 1-(u*v).sum(-1)/(np.maximum(np.linalg.norm(u,axis=-1),1e-6)*max(float(np.linalg.norm(v)),1e-6))
def posthoc_normal_rank(score,y):
    """Label-based descriptive percentile only, never an operational score."""
    normal=np.sort(score[y==0]);n=len(normal)
    if not n:return np.full(len(score),np.nan)
    return (np.searchsorted(normal,score,'left')+np.searchsorted(normal,score,'right'))/(2*n)
def stats(x):
    x=np.asarray(x,float);x=x[np.isfinite(x)]
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,
        std=float(x.std()) if len(x) else None,q10=float(np.quantile(x,.1)) if len(x) else None,q90=float(np.quantile(x,.9)) if len(x) else None)
def ranking(score,y):
    if len(np.unique(y))!=2:return dict(AP=None,ROC=None,reason='SINGLE_CLASS')
    return dict(AP=float(average_precision_score(y,score)),ROC=float(roc_auc_score(y,score)),reason=None)

def run(cfg):
    start=time.monotonic();scope=bound(cfg['scope']);old=bound(cfg['accepted_summary'])
    if sha(__file__)!=cfg['source_sha256'] or scope['new_detector'] or scope['reference_fit_from_QUERY']:
        raise PermissionError('Fixed descriptive-only scope required')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    errors=[];descriptions=[];rank_rows=[];event_rows=[];center_rows=[];bindings=[];pool={}
    for domain in ('SMAP','SMD'):
        desc=bound(cfg['banks'][domain]);by={}
        for d in desc:by.setdefault(d['entity'],{})[d['method']]=d
        for entity,pair in sorted(by.items()):
            if set(pair)!={'RoCA_FULL_T','RoCA_INIT_T'}:raise ValueError('Paired full/init required')
            data={};centers={}
            truth=Path(cfg['evaluation_directory'])/'entities'/domain/entity/'RoCA_FULL_T'/'calculation_inputs.npz'
            with np.load(truth,allow_pickle=False) as z:index=z['index'].copy();y=z['full_labels'][index]
            hard_spec=cfg['static_banks'][domain][entity]
            if sha(hard_spec['path'])!=hard_spec['sha256']:raise PermissionError('Static bank changed')
            with np.load(hard_spec['path'],allow_pickle=False) as z:
                if not np.array_equal(z['index'],index):raise ValueError('Static and q original axis mismatch')
                hard=(y==0)&(z['score']>hard_spec['CAL_tau'])
            for method,d in pair.items():
                p=Path(cfg['workspace'])/d['scores']['path'].split('/home/a312/tsad_research/',1)[1]
                if sha(p)!=d['scores']['sha256']:raise PermissionError('Original projection bank changed')
                with np.load(p,allow_pickle=False) as z:
                    if not np.array_equal(index,z['index']):raise ValueError('Paired original indices differ')
                    q=z['projection'].astype(float);qr=z['reconstructed_projection'].astype(float);c=z['center'].astype(float);total=z['score'].astype(float)
                a=branch_distance(q,c);b=branch_distance(qr,c);error=float(np.max(np.abs(a+b-total)))
                if error>2e-6:raise ValueError('Saved projection/center cannot reproduce original L_n')
                centers[method]=c;features={'q':q,'qr':qr}
                values={'Dq':a,'Dqr':b,'canonical_Ln':total};rank={k:posthoc_normal_rank(s,y) for k,s in values.items()}
                data[method]=dict(values=values,rank=rank,features=features)
                errors.append(dict(domain=domain,entity=entity,method=method,max_branch_sum_error=error,
                    correlation_Dq_Dqr=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None))
                bindings.append(dict(path=str(p),sha256=sha(p),truth_inputs=str(truth),truth_sha256=sha(truth),static=hard_spec))
                masks={'ALL':np.ones(len(y),bool),'OFFICIAL_NORMAL':y==0,'OFFICIAL_ANOMALY':y==1,
                    'HARD_NORMAL_STATIC_CAL_TAU':hard,'OTHER_NORMAL':(y==0)&~hard}
                for name,s in values.items():
                    for group,m in masks.items():descriptions.append(dict(domain=domain,entity=entity,method=method,branch=name,group=group,**stats(s[m])))
                    # Canonical total metrics are referenced, not repeated.
                    if name!='canonical_Ln':rank_rows.append(dict(domain=domain,entity=entity,method=method,branch=name,**ranking(s,y)))
                    pool.setdefault((domain,method,name),[]).append((s,y))
                for group,m in masks.items():descriptions.append(dict(domain=domain,entity=entity,method=method,branch='Dq_minus_Dqr_DESCRIPTIVE_ONLY',group=group,**stats((a-b)[m])))
                for field,v in features.items():
                    u=unit(v)
                    descriptions.append(dict(domain=domain,entity=entity,method=method,branch=field+'_unit_spread',group='ALL',
                        n=len(v),mean=float(u.var(0).sum()),median=None,std=None,q10=None,q90=None))
            ci,cf=centers['RoCA_INIT_T'],centers['RoCA_FULL_T']
            center_rows.append(dict(domain=domain,entity=entity,array_exact=bool(np.array_equal(ci,cf)),
                L2_difference=float(np.linalg.norm(ci-cf)),cosine=float((unit(ci)*unit(cf)).sum()),
                center_swapped=False,interpretation='Same center isolates its movement only; distinct latent coordinates forbid center swapping'))
            save=dict(index=index,labels=y,hard_normal_static_CAL=hard)
            for branch in ('Dq','Dqr','canonical_Ln'):
                ri=data['RoCA_INIT_T']['rank'][branch];rf=data['RoCA_FULL_T']['rank'][branch];delta=rf-ri
                for method in pair:save[method+'_'+branch]=data[method]['values'][branch];save[method+'_'+branch+'_posthoc_normal_percentile']=data[method]['rank'][branch]
                event_meta=Path(cfg['evaluation_directory'])/'entities'/domain/entity/'RoCA_FULL_T'/'event_coverage.csv'
                with event_meta.open() as f:events=list(csv.DictReader(f))
                for e in events:
                    m=(index>=int(e['start']))&(index<int(e['end_exclusive']))
                    event_rows.append(dict(domain=domain,entity=entity,source_event_id=e['source_event_id'],branch=branch,
                        main_eligible=e['main_eligible'],score_complete=e['complete'],boundary_censored=e['boundary_censored'],
                        original_points=int(e['total_points']),observed_points=int(m.sum()),
                        INIT_percentile_mean=float(ri[m].mean()) if m.any() else None,FULL_percentile_mean=float(rf[m].mean()) if m.any() else None,
                        percentile_delta_mean=float(delta[m].mean()) if m.any() else None,
                        percentile_down_points=int((delta[m]<0).sum()),percentile_up_points=int((delta[m]>0).sum()),
                        reference='POSTHOC QUERY-label0 CDF for diagnosis, NOT FIT reference or new detector'))
            directory=out/'arrays'/domain/entity;directory.mkdir(parents=True)
            np.savez_compressed(directory/'branches_and_posthoc_ranks.npz',**save)
    pooled=[]
    for (domain,method,name),parts in sorted(pool.items()):
        s=np.concatenate([x[0] for x in parts]);y=np.concatenate([x[1] for x in parts]);summary=ranking(s,y) if name!='canonical_Ln' else None
        if summary is None:
            r=next(r for r in old['dataset'] if r['domain']==domain and r['method']==method)
            summary=dict(AP=r['micro_point_ranking']['AP'],ROC=r['micro_point_ranking']['ROC_AUC'],reason='ORIGINAL_ACCEPTED_CANONICAL_TOTAL_REUSED')
        entity=[r for r in rank_rows if (r['domain'],r['method'],r['branch'])==(domain,method,name)]
        pooled.append(dict(domain=domain,method=method,branch=name,**summary,
            entity_macro_AP=float(np.mean([r['AP'] for r in entity if r['AP'] is not None])) if entity else None))
    table(out/'BRANCH_RANKING.csv',rank_rows);table(out/'BRANCH_DESCRIPTIONS.csv',descriptions)
    table(out/'POOLED_BRANCH_RANKING.csv',pooled);table(out/'PAIRED_EVENT_POSTHOC_PERCENTILES.csv',event_rows)
    table(out/'CENTER_MOVEMENT.csv',center_rows);dump(out/'CLOSURE_CHECKS.json',errors);dump(out/'SOURCE_ARRAY_BINDINGS.json',bindings)
    dump(out/'CONFIG.json',cfg);dump(out/'RESULT.json',dict(status='COMPLETED_CPU_DESCRIPTIVE_ONLY',entities=54,projection_banks=108,
        branch_rankings=pooled,max_branch_closure_error=max(r['max_branch_sum_error'] for r in errors),seconds=time.monotonic()-start,
        new_forward=False,new_detector=False,QUERY_reference_fit=False,GPUh=0,missing=['FIT frozen-reference q/qr','CNN preprojection QUERY/FIT','LSTM hidden QUERY/FIT'],
        limitations=scope['limitations']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Fixed diagnostic config required')
    run(json.loads(Path(a.config).read_text()))

if __name__=='__main__':main()
