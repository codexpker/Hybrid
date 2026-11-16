"""Fixed three-seed analysis, reusing accepted 2024 metrics without rerunning it.

Training randomness is the repeated unit. Entities remain separate; curves are
never pooled across seeds/machines. Query labels open only after both repeat
seeds' complete 48-bank global seal, except accepted 2024 curve-only export.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from . import timeradar_cross_variable_evaluation as ev

SEEDS=(2024,2025,2026)
ARMS=('decoder_head','self','cross')
RECORDS=('PSM','machine-1-1','machine-1-2','machine-1-3')
OLD_CSV_SHA='61697563a1aa7d99a403b6e3e1062246f5be3f0697a825c09e04d4cb100edd90'
OLD_SEAL_SHA='e073be13716f3b11e3121043c9c74086a20f560b6f5519cf619b9b3d8e91f146'
EVALUATOR_SHA='6077ccf194eee92323cc45238f246ab89468e60314cd39de5bffe5fef1900cd0'
DATA_SHA='d10cd11eca48f570f0e4a122f133cd8be60792c743ba68e0554f293ec4e82149'
EVAL_MANIFEST_SHA='7e2bbe4bc79b9e089ec2c9a69397b9b3bdac3edfeb873a023527ad71691302a7'


def _read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def _verified(path,expected):
    if ev.sha(path)!=expected:raise PermissionError('Frozen input identity changed: '+str(path))
    return path


def old_rows(workspace):
    path=Path(workspace)/'results/timeradar_cross_variable_fixed_evaluation_20261116_v1/metrics.csv'
    _verified(path,OLD_CSV_SHA);rows=[]
    with path.open(newline='',encoding='utf8')as handle:
        for raw in csv.DictReader(handle):
            if raw['arm']not in ARMS:continue
            row={}
            for key,value in raw.items():
                if key in ('record','arm','mode'):row[key]=value
                elif not value:row[key]=None
                else:row[key]=json.loads(value)
            row['seed']=2024;row['metric_origin']='ACCEPTED_2024_CSV_UNCHANGED';rows.append(row)
    if len(rows)!=12:raise ValueError('Accepted 2024 nonzero arms must contain 12 rows')
    return rows,path


def paired_statistics(rows,*,seeds=SEEDS):
    """Entity-wise exact seed pairing; sample SD, no point-level inference."""
    lookup={}
    for row in rows:
        key=(row['record'],row['arm'],row['seed'])
        if key in lookup:raise ValueError('Duplicate entity/method/seed')
        lookup[key]=row
    expected={(r,a,s)for r in RECORDS for a in ARMS for s in seeds}
    if set(lookup)!=expected:raise ValueError('All fixed entities, methods and seeds must be present')
    metrics=[k for k,v in rows[0].items()if isinstance(v,(int,float,type(None)))
             and k not in ('seed','threshold','scored_points','positive_points','unscored_points')]
    summary=[];paired=[]
    for rid in RECORDS:
        for arm in ARMS:
            for key in metrics:
                values=[lookup[rid,arm,s].get(key)for s in seeds]
                present=[float(v)for v in values if v is not None and np.isfinite(v)]
                summary.append(dict(record=rid,arm=arm,metric=key,seeds=list(seeds),values=values,n=len(present),
                    mean=float(np.mean(present))if len(present)==len(seeds)else None,
                    sample_SD=float(np.std(present,ddof=1))if len(present)==len(seeds)and len(present)>1 else None,
                    scope='training-seed variability within one fixed exposed entity; not independent datasets'))
        for comparator in ('self','decoder_head'):
            for key in metrics:
                diffs=[]
                for s in seeds:
                    a,b=lookup[rid,'cross',s].get(key),lookup[rid,comparator,s].get(key)
                    diffs.append(float(a-b)if a is not None and b is not None and np.isfinite(a)and np.isfinite(b)else None)
                present=[v for v in diffs if v is not None]
                paired.append(dict(record=rid,comparison='cross-minus-'+comparator,metric=key,seeds=list(seeds),
                    paired_differences=diffs,n=len(present),mean_difference=float(np.mean(present))if len(present)==len(seeds)else None,
                    sample_SD_difference=float(np.std(present,ddof=1))if len(present)==len(seeds)and len(present)>1 else None,
                    positive_seeds=sum(v>0 for v in present),negative_seeds=sum(v<0 for v in present),
                    scope='paired training seeds; no point/event resampling or significance claim'))
    return summary,paired


def curve_arrays(labels,score):
    labels,score=np.asarray(labels),np.asarray(score)
    if labels.ndim!=1 or labels.shape!=score.shape or not np.isfinite(score).all()or not np.isin(labels,[0,1]).all()or len(np.unique(labels))!=2:
        raise ValueError('Finite original continuous scores and both classes required')
    precision,recall,pr_threshold=precision_recall_curve(labels,score)
    fpr,tpr,roc_threshold=roc_curve(labels,score,drop_intermediate=False)
    return dict(precision=precision,recall=recall,pr_threshold=pr_threshold,
                FPR=fpr,TPR=tpr,roc_threshold=roc_threshold)


def cal_operating_point(labels,score,threshold):
    """Actual frozen score>tau bits; sklearn curve threshold ties use >=."""
    labels,score=np.asarray(labels),np.asarray(score)
    alarm=score>threshold
    tp=int((alarm&(labels==1)).sum());fp=int((alarm&(labels==0)).sum())
    positives=int((labels==1).sum());normals=int((labels==0).sum())
    return dict(TP=tp,FP=fp,TPR=tp/positives if positives else None,
                FPR=fp/normals if normals else None,
                precision=tp/(tp+fp)if tp+fp else None,
                recall=tp/positives if positives else None,
                threshold=float(threshold),rule='actual score>CAL_tau; not nearest sklearn >= threshold')


def _labels(w,rid):
    path=w/'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json'
    _verified(path,EVAL_MANIFEST_SHA)
    spec=next(r for r in _read(path)['records']if r['id']==rid);source=w/spec['path']
    _verified(source,spec['sha256'])
    with np.load(source,allow_pickle=False)as z:
        y=z['label'].copy()
        if not np.array_equal(z['index'],np.arange(len(y))):raise ValueError('Complete original label axis required')
    return y,source


def _curve(folder,index,score,labels,*,rid,arm,seed,inputs,metric_row):
    folder.mkdir(parents=True,exist_ok=False)
    arrays=curve_arrays(labels[index],score)
    point=cal_operating_point(labels[index],score,metric_row['threshold'])
    if point['TP']!=metric_row['strict_TP']or point['FP']!=metric_row['strict_FP']:
        raise ValueError('Fixed CAL operating point differs from accepted strict alarm counts')
    file=folder/'PR_ROC.npz';np.savez_compressed(file,**arrays)
    ev._dump(folder/'CURVE.json',dict(record=rid,arm=arm,seed=seed,scored_points=len(index),
        curve_sha256=ev.sha(file),inputs=inputs,AP=metric_row['AP'],ROC_AUC=metric_row['ROC_AUC'],
        threshold=metric_row['threshold'],threshold_rule='unchanged own CAL q.99 higher, score>',
        CAL_operating_point=point,
        permissions='native offline complete window and batch-shared FRFT order',
        notes='Full empirical PR/ROC only; no QUERY operating threshold or model selection; no pooled seed points',
        low_FPR_view='Fixed plotting zoom [0,.05]; no new metric or alarm rule',
        event_metrics='See per-seed metrics and original UNKNOWN-aware event details'))


def accepted_curves(workspace,output):
    """Read accepted 2024 scores, never repeat its training/inference/metrics."""
    w,output=Path(workspace),Path(output);output.mkdir(parents=True,exist_ok=False)
    rows,csvpath=old_rows(w);root=w/'results/timeradar_cross_variable_20261116_v1'
    seal=root/'RAW_BANK_SEAL.json';_verified(seal,OLD_SEAL_SHA)
    banks={b['relative']:b['sha256']for b in _read(seal)['banks']}
    for row in rows:
        rid,arm=row['record'],row['arm'];path=root/rid/arm/'score/QUERY.npz'
        _verified(path,banks[f'{rid}/{arm}/score/QUERY.npz'])
        index,score,_=ev._load_bank(path);labels,lp=_labels(w,rid)
        _curve(output/rid/arm/'seed2024',index,score,labels,rid=rid,arm=arm,seed=2024,
               inputs={str(p):ev.sha(p)for p in (path,csvpath,lp,seal)},metric_row=row)
    with(output/'curve_array_producer_snapshot.py').open('xb')as handle:
        handle.write(Path(__file__).read_bytes())
    ev._dump(output/'RECEIPT.json',dict(status='COMPLETED_ACCEPTED_2024_CURVES_ONLY',rows=12,
        old_metrics_csv_sha256=OLD_CSV_SHA,new_training=False,new_forward=False,old_metrics_recomputed=False,
        GPU_hours=0,source_sha256=ev.sha(__file__)))
    return rows


def reuse_accepted_curves(workspace,output):
    """Copy already accepted curve arrays after identity checks; no recompute."""
    w,output=Path(workspace),Path(output)
    source=w/'results/timeradar_cross_variable_accepted2024_curves_20261116_v1'
    receipt=_read(source/'RECEIPT.json')
    if receipt['status']!='COMPLETED_ACCEPTED_2024_CURVES_ONLY'or receipt['old_metrics_csv_sha256']!=OLD_CSV_SHA:
        raise PermissionError('Accepted 2024 curve cache identity missing')
    _verified(source/'curve_array_producer_snapshot.py',receipt['source_sha256'])
    rows,_=old_rows(w)
    for row in rows:
        root=source/row['record']/row['arm']/'seed2024';description=_read(root/'CURVE.json')
        if (description['record']!=row['record']or description['arm']!=row['arm']or description['seed']!=2024
                or description['AP']!=row['AP']or description['ROC_AUC']!=row['ROC_AUC']
                or description['threshold']!=row['threshold']):
            raise PermissionError('Accepted curve metadata differs from accepted metric row')
        _verified(root/'PR_ROC.npz',description['curve_sha256'])
    shutil.copytree(source,output)
    ev._dump(output/'REUSE_RECEIPT.json',dict(status='COPIED_VERIFIED_ACCEPTED_ARRAYS',
        source_receipt_sha256=ev.sha(source/'RECEIPT.json'),source=str(source),
        curve_arrays_recomputed=False,old_metrics_recomputed=False,GPU_hours=0))


def _repeat_gate(w):
    global_path=w/'preparations/timeradar_cross_variable_repeat_20261116_v1/ALL_REPEAT_RAW_BANK_SEAL.json'
    global_seal=_read(global_path)
    if global_seal['status']!='COMPLETED':raise FileNotFoundError('Global 48-bank seal not completed')
    entries=global_seal['seeds']
    if not isinstance(entries,list)or len(entries)!=2:raise ValueError('Exactly two globally bound seed seals required')
    seed_entries={int(r['seed']):r for r in entries}
    if set(seed_entries)!={2025,2026}:raise ValueError('Global repeat seeds must be exactly 2025 and 2026')
    if 'banks'in global_seal and len(global_seal['banks'])!=48:raise ValueError('Global seal must bind 48 banks')
    roots={};assets={str(global_path):ev.sha(global_path)}
    base=_read(w/'hybrid_ad/configs/timeradar-cross-variable-20261116.json')
    for seed in (2025,2026):
        cp=w/f'hybrid_ad/configs/timeradar-cross-variable-repeat-seed{seed}-20261116.json'
        cfg=_read(cp)
        if {k:v for k,v in cfg.items()if k not in ('id','output','seed')}!={k:v for k,v in base.items()if k not in ('id','output','seed')}or cfg['seed']!=seed:
            raise PermissionError('Repeat config differs beyond registered seed/output identity')
        if cfg['score_seed']!=2024 or cfg['select_seed']!=84551:raise PermissionError('Fixed scoring/selection RNG changed')
        root=w/cfg['output'];seal=root/'RAW_BANK_SEAL.json';raw=_read(seal)
        expected=seed_entries[seed].get('raw_seal_sha256',seed_entries[seed].get('raw_seal_sha'))
        if not isinstance(expected,str)or len(expected)!=64:raise ValueError('Global seed entry must pin its raw seal SHA')
        _verified(seal,expected)
        if raw['status']!='COMPLETED'or len(raw['banks'])!=24:raise FileNotFoundError('Each repeat requires all 24 banks')
        for bank in raw['banks']:_verified(root/bank['relative'],bank['sha256'])
        roots[seed]=(root,cfg,cp);assets.update({str(p):ev.sha(p)for p in (cp,seal)})
    # Per-seed complete seals are checked before any new label access.
    return roots,assets


def run(workspace,output,*,draw=False):
    started=time.monotonic();w,output=Path(workspace),Path(output)
    _verified(Path(ev.__file__),EVALUATOR_SHA)
    binding=w/'preparations/timeradar_cross_variable_20261116_v1/data_binding.json'
    _verified(binding,DATA_SHA);contract={r['id']:r for r in _read(binding)['records']}
    roots,assets=_repeat_gate(w)
    rows,oldcsv=old_rows(w);assets[str(oldcsv)]=ev.sha(oldcsv)
    output.mkdir(parents=True,exist_ok=False)
    # Preserve 2024 metrics by direct CSV reuse. Curves read its accepted banks.
    reuse_accepted_curves(w,output/'accepted_2024_curves')
    for seed,(root,cfg,cp)in roots.items():
        for rid in RECORDS:
            rolemap={r['role']:r for r in contract[rid]['roles']}
            for arm in ARMS:
                folder=root/rid/arm/'score';rp=folder/'SCORE_RECEIPT.json';receipt=_read(rp)
                if receipt['status']!='COMPLETED'or receipt['record']!=rid or receipt['arm']!=arm or receipt['config_sha256']!=ev.sha(cp):
                    raise PermissionError('Score receipt/config identity mismatch')
                roles={r['role']:r for r in receipt['roles']}
                if any(roles[r]['sha256']!=rolemap[r]['sha256']for r in ('CAL','QUERY')):raise PermissionError('Different physical role data')
                bankmap={Path(r['path']).name:r['sha256']for r in receipt['banks']}
                for name in ('CAL.npz','QUERY.npz'):_verified(folder/name,bankmap[name])
                cix,cal,_=ev._load_bank(folder/'CAL.npz');index,score,available=ev._load_bank(folder/'QUERY.npz')
                for role,actual in (('CAL',cix),('QUERY',index)):
                    lo=rolemap[role]['bounds_half_open'][0]
                    if not np.array_equal(actual,np.arange(lo,lo+rolemap[role]['scored_points'])):raise ValueError('Changed native score axis')
                tau=ev.cal_threshold(cal);labels,lp=_labels(w,rid)
                result=ev.evaluate_arrays(index,score,labels,tau,available_index=available)
                result.update(CAL_points=len(cal),CAL_exceedance_rate=float(np.mean(cal>tau)),seed=seed)
                inputs={str(p):ev.sha(p)for p in (folder/'CAL.npz',folder/'QUERY.npz',rp,lp,cp,binding)}
                ev._save_method(output/'evaluation'/rid/arm/f'seed{seed}',result,index,score,labels,available,inputs)
                row=ev._flat(rid,arm,result);row.update(seed=seed,metric_origin='NEW_SEALED_REPEAT');rows.append(row)
                _curve(output/'curves'/rid/arm/f'seed{seed}',index,score,labels,rid=rid,arm=arm,seed=seed,inputs=inputs,metric_row=row)
    rows.sort(key=lambda r:(RECORDS.index(r['record']),r['seed'],ARMS.index(r['arm'])))
    summary,paired=paired_statistics(rows)
    ev._write_csv(output/'per_entity_seed_metrics.csv',rows)
    ev._write_csv(output/'mean_sample_SD.csv',summary);ev._write_csv(output/'paired_cross_differences.csv',paired)
    ev._dump(output/'STATISTICS.json',dict(summary=summary,paired=paired,
        scope='Three training seeds per fixed exposed entity; sample SD ddof1; no independent-domain or point-level inference',
        no_pooled_curves=True,no_pvalues=True,no_QUERY_selection=True))
    if draw:plot_curves(output,rows)
    ev._dump(output/'RESULT.json',dict(status='COMPLETED',rows=rows,CPU_seconds=time.monotonic()-started,
        GPU_hours=0,new_training=False,new_forward=False,old_2024_metrics_reused=True,
        inputs=assets,source_sha256=ev.sha(__file__),evaluator_sha256=ev.sha(ev.__file__)))
    return rows,summary,paired


def plot_curves(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);lookup={(r['record'],r['arm'],r['seed']):r for r in rows}
    seeds=tuple(sorted({r['seed']for r in rows}))
    colors=dict(decoder_head='#4c78a8',self='#f58518',cross='#54a24b')
    styles={2024:'-',2025:'--',2026:':'};markers={2024:'o',2025:'s',2026:'^'}
    for rid in RECORDS:
        fig,axes=plt.subplots(3,3,figsize=(15,12))
        for col,arm in enumerate(ARMS):
            for seed in seeds:
                root=output/('accepted_2024_curves'if seed==2024 else 'curves')/rid/arm/f'seed{seed}'
                if not root.is_dir()and seeds==(2024,):root=output/rid/arm/'seed2024'
                with np.load(root/'PR_ROC.npz',allow_pickle=False)as z:
                    row=lookup[rid,arm,seed]
                    kwargs=dict(color=colors[arm],linestyle=styles[seed],lw=1)
                    axes[0,col].plot(z['recall'],z['precision'],label=f'{seed}: AP {row["AP"]:.4f}',**kwargs)
                    for ri in (1,2):axes[ri,col].plot(z['FPR'],z['TPR'],label=f'{seed}: ROC {row["ROC_AUC"]:.4f}',**kwargs)
                    point=_read(root/'CURVE.json').get('CAL_operating_point')
                    if point is None:
                        # Accepted early curve-only export: recover actual >tau
                        # confusion counts from the hash-checked historical CSV,
                        # never a nearest sklearn >=threshold operating point.
                        tp,fp,fn,tn=(row['strict_'+k]for k in ('TP','FP','FN','TN'))
                        point=dict(FPR=fp/(fp+tn),TPR=tp/(tp+fn),recall=tp/(tp+fn),
                                   precision=tp/(tp+fp)if tp+fp else None)
                    if point['precision']is not None:
                        axes[0,col].plot(point['recall'],point['precision'],marker=markers[seed],ms=5,
                                         color=colors[arm],markerfacecolor='white',linestyle='none')
                    for ri in (1,2):
                        axes[ri,col].plot(point['FPR'],point['TPR'],marker=markers[seed],ms=5,
                                         color=colors[arm],markerfacecolor='white',linestyle='none')
            for ri in range(3):
                ax=axes[ri,col];ax.set_ylim(0,1);ax.set_xlim(0,.05 if ri==2 else 1)
                ax.set_xlabel('Recall'if ri==0 else'False positive rate');ax.set_ylabel('Precision'if ri==0 else'True positive rate')
                ax.set_title(arm+(' PR'if ri==0 else' ROC'if ri==1 else' ROC: fixed low-FPR view'))
                ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle(rid+' — method color / seed line style; markers=unchanged CAL > rule; offline, no pooled points')
        fig.tight_layout();fig.savefig(output/f'{rid}_PR_ROC_separate_seeds.png',dpi=150)
        fig.savefig(output/f'{rid}_PR_ROC_separate_seeds.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--accepted-2024-curves-only',action='store_true');p.add_argument('--draw',action='store_true')
    a=p.parse_args()
    if a.accepted_2024_curves_only:accepted_curves(a.workspace,a.output)
    else:run(a.workspace,a.output,draw=a.draw)


if __name__=='__main__':main()
