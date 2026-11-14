"""Single sealed-bank fixed evaluation; no fitting on QUERY labels."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
from cpu_controls import data
from fixed_metrics import evaluate,rolling_max64,R
P=Path(__file__).resolve().parent;O=R/'results'/P.name
def load(p):
    with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='',encoding='utf8')as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def main(domain):
    dest=O/'evaluation'/domain;dest.mkdir(parents=True,exist_ok=False);banks={};raw=[]
    cv,cs,_=data(domain,'CAL');qv,qs,_=data(domain,'QUERY');ci=np.arange(cs+512,cs+len(cv));qi=np.arange(qs+512,qs+len(qv))
    for arm in ['source','native','blind']:
        d=O/(domain+'_'+arm)/'score';receipt=json.loads((d/'SCORE_RECEIPT.json').read_text());assert receipt['status']=='COMPLETED'
        bank={}
        for role,ix in [('CAL',ci),('QUERY',qi)]:
            f=d/(role+'.npz');z=load(f);assert np.array_equal(z['index'],ix)and np.array_equal(z['available_index'],ix)
            assert np.allclose(z['score'],z['per_channel_last'].mean(1),rtol=1e-6,atol=1e-6);raw.append(dict(path=str(f),sha256=sha(f)));bank[role]=z
        for name,key in ([('LEFT_source','native_score'),('Hybrid_mask_zero_update','score')]if arm=='source'else[('LEFT_continued','score')]if arm=='native'else[('Hybrid_blindspot','score'),('Hybrid_nativeoff','native_score')]):banks[name]={role:bank[role][key]for role in bank}
    banks['LEFT_past64max']={r:rolling_max64(s)for r,s in banks['LEFT_source'].items()}
    for name in ['HistoryMean512','MeanFIT','InstantRidge']:
        e={}
        for role,ix in [('CAL',ci),('QUERY',qi)]:
            f=O/'cpu_controls'/domain/(name+'_'+role+'.npz');z=load(f);assert np.array_equal(z['index'],ix);e[role]=z['score'];raw.append(dict(path=str(f),sha256=sha(f)))
        banks[name]=e
    for name,e in banks.items():assert all(np.isfinite(x).all()for x in e.values())
    (dest/'RAW_SEAL.json').write_text(json.dumps(dict(status='SEALED_BEFORE_LABEL_JOIN',raw=raw,methods=list(banks),CAL=[int(ci[0]),int(ci[-1])],QUERY=[int(qi[0]),int(qi[-1])],source_code_sha256=sha(__file__),threshold='own continuous commonCAL q99higher; no QUERY fitted parameters'),indent=2))
    if domain=='SWAT':truth=load(R/'preparations/left_multievent_mixed_u_20261111_v1/SWAT_LABEL_METADATA.npz')['label']
    else:
        cfg=json.loads((R/'preparations/hybrid_left_joint_cycle_20261109_v2/EVALUATION.ACTUAL.json').read_text());f=Path(cfg['truth']['path'].replace('__RESEARCH_ROOT__',str(R)));truth=load(f)['full_labels']
    fit,_,_=data(domain,'FIT');fg=np.abs(np.diff(fit,axis=0)).mean(1);hardtau=float(np.quantile(fg,.95,method='higher'));hard=np.zeros(len(truth),bool);g=np.r_[0.,np.abs(np.diff(qv,axis=0)).mean(1)];hard[qs:qs+len(qv)]=g>hardtau
    rows=[];events=[]
    for name,e in banks.items():
        rr,ev=evaluate(qi,e['QUERY'],e['CAL'],truth,steady_after=192,hard_normal=hard)
        rows.extend(dict(domain=domain,method=name,**x)for x in rr);events.extend(dict(domain=domain,method=name,**x)for x in ev)
    table(dest/'summary.csv',rows);table(dest/'per_event.csv',events)
    (dest/'RECEIPT.json').write_text(json.dumps(dict(status='COMPLETED',methods=len(banks),hard_normal_definition='meanC abs standardized firstdiff > FITq95higher; intersect evaluation normal labels only',hard_normal_threshold=hardtau,steady_after=192,long_min_length=512,past64max='min_periods1 per role reset; first63 partial',single_sequence=True,no_QUERY_selection=True),indent=2))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('domain',choices=['PSM','SWAT']);main(ap.parse_args().domain)
