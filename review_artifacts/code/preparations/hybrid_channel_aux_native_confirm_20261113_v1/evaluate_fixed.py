"""All-three-seed raw-sealed native-readout confirmation; fixed roles, no QUERY fitting."""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name
sys.path.insert(0,str(R/'preparations/hybrid_left_channel_blindspot_20261113_v1'))
from fixed_metrics import evaluate
def load(f):
    with np.load(f,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def table(f,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with f.open('x',newline='',encoding='utf8')as h:w=csv.DictWriter(h,keys);w.writeheader();w.writerows(rows)
def run(manifest):
    spec=json.loads(Path(manifest).read_text());entries=spec['methods'];assert len(entries)==9
    assert {(int(e['seed']),e['method'])for e in entries}=={(s,m)for s in [2025,2026,2027]for m in ['source','native','aux']}
    ci=np.arange(23839,28479);qi=np.arange(512,28479);banks=[]
    for e in entries:
        v={};identities={}
        for role,ix in [('CAL',ci),('QUERY',qi)]:
            f=Path(e[role]);z=load(f);assert np.array_equal(z['index'],ix)and np.array_equal(z['available_index'],ix)
            score=z['score'];assert score.shape==(len(ix),)and np.isfinite(score).all()
            if 'per_channel_last'in z:assert np.allclose(score,z['per_channel_last'].mean(1),rtol=1e-6,atol=1e-6)
            v[role]=score;identities[role]=dict(path=str(f),sha256=sha(f))
        banks.append(dict(method=e['method'],seed=e['seed'],scores=v,source=dict(manifest_entry=e,raw_banks=identities)))
    for name in ['HistoryMean512','MeanFIT']:
        v={role:load(O/'cpu_controls'/(name+'_'+role+'.npz'))['score']for role in ['CAL','QUERY']};banks.append(dict(method=name,seed=0,scores=v,source='fixed FIT-only CPU'))
    dest=O/'evaluation';dest.mkdir(parents=True,exist_ok=False)
    seal=[dict(method=e['method'],seed=e['seed'],threshold=float(np.quantile(e['scores']['CAL'],.99,method='higher')),source=e['source'])for e in banks]
    (dest/'RAW_SEAL.json').write_text(json.dumps(dict(status='ALL3SEEDS_SEALED_BEFORE_LABEL_JOIN',methods=seal,manifest=str(manifest),manifest_sha256=sha(manifest),scaler_sha256=sha(P/'FIT_ONLY_SCALER.json'),role_sha256=sha(P/'ROLES.json')),indent=2))
    labels=load(P/'audit_only/QUERY_LABELS.npz')['labels'];assert labels.shape==(28479,)
    ss=json.loads((P/'FIT_ONLY_SCALER.json').read_text());mean=np.array(ss['mean']);scale=np.array(ss['scale']);fit=(load(P/'data/FIT.npz')['values']-mean)/scale;q=(load(P/'data/QUERY.npz')['values']-mean)/scale
    ht=float(np.quantile(np.abs(np.diff(fit,axis=0)).mean(1),.95,method='higher'));hard=np.r_[False,np.abs(np.diff(q,axis=0)).mean(1)>ht];rows=[];events=[]
    for e in banks:
        rr,ee=evaluate(qi,e['scores']['QUERY'],e['scores']['CAL'],labels,steady_after=192,hard_normal=hard)
        rows.extend(dict(method=e['method'],seed=e['seed'],**r)for r in rr);events.extend(dict(method=e['method'],seed=e['seed'],**r)for r in ee)
    table(dest/'per_seed.csv',rows);table(dest/'per_event.csv',events)
    d=pd.DataFrame(rows);numeric=['AP','ROC','Recall','F1','M','one_to_one_eventPrecision','one_to_one_eventRecall','one_to_one_eventF1','steady_macro_coverage','steady_pooled_coverage','normal_FPR','normal_alarm_fragments_per_1000_points','hard_normal_FPR','max_event_miss']
    d.groupby(['method','stream'])[numeric].agg(['mean','std','min','max']).to_csv(dest/'training_seed_summary.csv')
    dif=[]
    for stream in ['strict','confirmed3']:
        a=d[d.stream==stream]
        for seed in [2025,2026,2027]:
            target=a[(a.method=='aux')&(a.seed==seed)].iloc[0]
            for m in ['source','native','HistoryMean512','MeanFIT']:
                b=a[(a.method==m)&(a.seed==(0 if m in ['HistoryMean512','MeanFIT']else seed))].iloc[0];dif.append(dict(seed=seed,stream=stream,contrast='aux-'+m,**{k:None if pd.isna(target[k])or pd.isna(b[k])else target[k]-b[k]for k in numeric}))
    table(dest/'paired_seed_differences.csv',dif)
    (dest/'RECEIPT.json').write_text(json.dumps(dict(status='COMPLETED',rows=len(rows),all_original_events=8,long_events=3,long_min_length=512,steady_after=192,hard_normal_FITdiff_q95=ht,source_history_exposed=True,seed_uncertainty='3 training seeds on one entity; not3 independent domains',no_QUERY_selection=True),indent=2))
if __name__=='__main__':run(sys.argv[1])
