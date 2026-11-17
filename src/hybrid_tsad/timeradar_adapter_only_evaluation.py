"""One decoder-freeze factor: canonical V plus fixed auxiliary decomposition.

Old native16 and diagnostic72 rows are reused by hash. Only eight new model
rows are evaluated after their 16-bank seal. Auxiliary E/E+.9V never fit a
threshold or replace the official native L1 score.
"""
import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from . import timeradar_cross_variable_evaluation as ev
from .timeradar_smd_full_evaluation import read,local,checked,checked_bank,_csv_values,METRICS

RECORDS=('PSM','machine-1-1','machine-1-2','machine-1-3')
ARMS=('self','cross')
NEW_NAMES={a:a+'_adapter_only'for a in ARMS}
OLD_NATIVE_SHA='61697563a1aa7d99a403b6e3e1062246f5be3f0697a825c09e04d4cb100edd90'
OLD_AUX_SHA='51584365e3bd45ccacc1c61313e73500387d5b275276fcf0697990ed25becfdf'
BINDING_SHA='d10cd11eca48f570f0e4a122f133cd8be60792c743ba68e0554f293ec4e82149'
LABEL_SHA='7e2bbe4bc79b9e089ec2c9a69397b9b3bdac3edfeb873a023527ad71691302a7'


def auxiliary_scores(mean,variance,target):
    """Original 72-row pointwise float32 formula, not a scalar MSE loss."""
    mean,variance,target=map(np.asarray,(mean,variance,target))
    if (mean.ndim!=2 or mean.shape!=variance.shape or mean.shape!=target.shape
            or any(a.dtype!=np.float32 or not np.isfinite(a).all()for a in (mean,variance,target))
            or np.any(variance<0)):
        raise ValueError('Aligned finite float32 [time,channel] mean, sample variance and target required')
    v=variance.mean(-1);e=np.square(mean-target).mean(-1)
    return {'V':v,'E':e,'E+0.9V':e+.9*v}


def diagnostic_row(record,arm,readout,index,score,labels):
    y=labels[index];both=len(np.unique(y))==2
    row=dict(record=record,arm=arm,readout=readout,points=len(index),start=int(index[0]),end_exclusive=int(index[-1])+1,
        AP=float(average_precision_score(y,score))if both else None,
        ROC_AUC=float(roc_auc_score(y,score))if both else None)
    for group,mask in [('normal',y==0),('anomaly',y==1)]:
        a=score[mask];row[group+'_n']=len(a)
        values=dict(mean=float(a.mean()),std=float(a.std()),q05=float(np.quantile(a,.05)),
                    median=float(np.median(a)),q95=float(np.quantile(a,.95)),q99=float(np.quantile(a,.99)))if len(a)else{key:None for key in ('mean','std','q05','median','q95','q99')}
        row.update({group+'_'+key:value for key,value in values.items()})
    return row


def complete_seal(raw):
    expected={f'{r}/{a}/score/{role}.npz'for r in RECORDS for a in ARMS for role in ('CAL','QUERY')}
    keys={r['relative']for r in raw['banks']}
    if raw['status']!='COMPLETED'or len(raw['banks'])!=16 or keys!=expected:
        raise ValueError('All 16 adapter-only banks must seal before new QUERY labels')


def freeze_proof(folder):
    paths=[]
    for phase in ('train','score'):
        path=folder/phase/'NATIVE_FREEZE_RECEIPT.json';proof=read(path)
        if proof['status']!='PASS_ENTIRE_NATIVE_STATE'or not proof['includes_decoder_and_buffers']or not proof['proofs']:
            raise PermissionError('Whole native freeze evidence is required')
        if phase=='train'and not proof['selected_and_final_checked']:raise PermissionError('Selected/final native state must be checked')
        for item in proof['proofs']:
            if (not item['all_native_state_unchanged']or not item['all_native_parameters_frozen']
                    or not item['all_native_modules_eval']or item['initial']!=item['final']):
                raise PermissionError('Native decoder/parameter/buffer/mode freeze failed')
        paths.append(path)
    return paths


def merged_comparison(primary,auxiliary):
    lookup={(r['record'],r['arm'],r['readout']):r for r in auxiliary};rows=[]
    for original in primary:
        row=dict(original,primary_readout='official canonical bank.score, L1 V')
        for readout,prefix in [('V','aux_Vnumpy_'),('E','aux_E_'),('E+0.9V','aux_Eplus0p9V_')]:
            diagnostic=lookup[row['record'],row['arm'],readout]
            if diagnostic['points']!=row['scored_points']:raise ValueError('Primary and auxiliary axes must match')
            for key,value in diagnostic.items():
                if key not in ('record','arm','readout'):row[prefix+key]=value
        rows.append(row)
    return rows


def paired_rows(rows):
    lookup={(r['record'],r['arm']):r for r in rows};result=[]
    comparisons=[('cross_adapter_only','self_adapter_only'),('self_adapter_only','self'),('cross_adapter_only','cross'),
                 ('self_adapter_only','zero'),('cross_adapter_only','zero')]
    metrics=METRICS+('aux_E_AP','aux_E_ROC_AUC','aux_Eplus0p9V_AP','aux_Eplus0p9V_ROC_AUC')
    for rid in RECORDS:
        for candidate,reference in comparisons:
            a,b=lookup[rid,candidate],lookup[rid,reference]
            if a['scored_points']!=b['scored_points']:raise ValueError('Fixed matched native time axis required')
            for metric in metrics:
                av,bv=a.get(metric),b.get(metric)
                result.append(dict(record=rid,comparison=candidate+'-minus-'+reference,metric=metric,
                    difference=float(av-bv)if av is not None and bv is not None else None,seed=2024,
                    auxiliary_metric=metric.startswith('aux_'),
                    scope='Single fixed development seed; auxiliary metrics do not select readout or threshold'))
    return result


def run(workspace,output,*,config=None):
    start=time.monotonic();w,output=Path(workspace),Path(output)
    cp=(w/'hybrid_ad/configs/timeradar-adapter-only-20261117.json'if config is None else local(w,config))
    cfg=read(cp);root=local(w,cfg['output']);seal=root/'RAW_BANK_SEAL.json';raw=read(seal);complete_seal(raw)
    if cfg['seed']!=2024 or cfg['score_seed']!=2024 or cfg['select_seed']!=84551 or cfg['updates']!=2000:
        raise PermissionError('Fixed original seeds/update/selection recipe required')
    if 'config_sha256'in raw and raw['config_sha256']!=ev.sha(cp):raise PermissionError('Config/global seal mismatch')
    for bank in raw['banks']:checked(root/bank['relative'],bank['sha256'])
    bp=local(w,cfg['data_binding']);checked(bp,BINDING_SHA)
    if cfg['data_binding_sha256']!=BINDING_SHA:raise PermissionError('Original physical roles required')
    binding={r['id']:r for r in read(bp)['records']}
    proofs=[]
    for rid in RECORDS:
        for arm in ARMS:
            proofs+=freeze_proof(root/rid/arm)
            train=read(root/rid/arm/'train/TRAIN_RECEIPT.json')
            if train['trainable_parameters']!=16912:raise PermissionError('Only original r16 adapter parameters may update')
    # New labels are unopened until complete bank identities and freeze proofs.
    native_path=w/'results/timeradar_cross_variable_fixed_evaluation_20261116_v1/metrics.csv'
    auxiliary_path=w/'results/timeradar_error_decomposition_20261117_v1/comparison.csv'
    primary=_csv_values(native_path,OLD_NATIVE_SHA);auxiliary=_csv_values(auxiliary_path,OLD_AUX_SHA)
    if len(primary)!=16 or len(auxiliary)!=72:raise ValueError('Full accepted historical tables required')
    for row in primary:row.update(metric_origin='ACCEPTED_OLD_OFFICIAL_V',decoder_policy='no_target_update'if row['arm']=='zero'else'decoder_only'if row['arm']=='decoder_head'else'decoder_plus_adapter')
    for row in auxiliary:row.update(metric_origin='FROZEN_HISTORICAL_72',readout_scope='auxiliary_posthoc_no_threshold')
    ep=w/'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json';checked(ep,LABEL_SHA)
    labelmeta={r['id']:r for r in read(ep)['records']}
    output.mkdir(parents=True,exist_ok=False);new_aux=[]
    for rid in RECORDS:
        record=binding[rid];qp=next(r for r in record['roles']if r['role']=='QUERY');xp=w/qp['path'];checked(xp,qp['sha256'])
        with np.load(xp,allow_pickle=False)as z:x,xi=z['x'].copy(),z['index'].copy()
        lp=w/labelmeta[rid]['path'];checked(lp,labelmeta[rid]['sha256'])
        with np.load(lp,allow_pickle=False)as z:
            labels=z['label'].copy()
            if not np.array_equal(z['index'],np.arange(len(labels))):raise ValueError('Complete original labels required')
        for arm in ARMS:
            folder=root/rid/arm/'score';index,canonical,available,tau,rp=checked_bank(folder,rid,arm,record,cp)
            name=NEW_NAMES[arm]
            inputs={str(p):ev.sha(p)for p in (folder/'CAL.npz',folder/'QUERY.npz',rp,xp,lp,cp,bp,seal)}
            result=ev.evaluate_arrays(index,canonical,labels,tau,available_index=available)
            ev._save_method(output/'official_V'/rid/name,result,index,canonical,labels,available,inputs)
            trainpath=root/rid/arm/'train/TRAIN_RECEIPT.json';train=read(trainpath)
            row=ev._flat(rid,name,result);row.update(metric_origin='NEW_ADAPTER_ONLY_OFFICIAL_V',source_arm=arm,
                decoder_policy='entire_native_frozen_eval_adapter_only',trainable_parameters=train['trainable_parameters'],
                updates=train['updates'],selected_update=train['selected_update'],seed=2024,
                training_receipt=trainpath.relative_to(w).as_posix(),training_receipt_sha256=ev.sha(trainpath))
            primary.append(row)
            with np.load(folder/'QUERY.npz',allow_pickle=False)as z:
                mean=z['reconstruction_mean'].copy();variance=z['channel_variance'].copy()
            pos=np.searchsorted(xi,index)
            if np.any(pos>=len(xi))or not np.array_equal(xi[pos],index):raise ValueError('Per-point target/reconstruction alignment mismatch')
            scores=auxiliary_scores(mean,variance,x[pos])
            if not np.allclose(scores['V'],canonical,rtol=2e-6,atol=1e-8):raise ValueError('Cached sample variance differs from official score beyond known reduction rounding')
            for readout,values in scores.items():
                diagnostic=diagnostic_row(rid,name,readout,index,values,labels)
                diagnostic.update(metric_origin='NEW_ADAPTER_ONLY_AUXILIARY',readout_scope='auxiliary_posthoc_no_threshold')
                auxiliary.append(diagnostic);new_aux.append(diagnostic)
            np.savez_compressed(output/'official_V'/rid/name/'auxiliary_inputs.npz',index=index,mean=mean,channel_variance=variance,
                observed_target=x[pos],official_V=canonical,numpy_V=scores['V'],E=scores['E'],Eplus0p9V=scores['E+0.9V'])
    primary.sort(key=lambda r:(RECORDS.index(r['record']),('zero','decoder_head','self','cross','self_adapter_only','cross_adapter_only').index(r['arm'])))
    comparison=merged_comparison(primary,auxiliary);pairs=paired_rows(comparison)
    ev._write_csv(output/'official_V_24rows.csv',primary);ev._write_csv(output/'auxiliary_decomposition_96rows.csv',auxiliary)
    ev._write_csv(output/'paired_comparison.csv',comparison);ev._write_csv(output/'paired_differences.csv',pairs)
    references=[]
    for row in primary:
        rid,arm=row['record'],row['arm']
        folder=(output/'official_V'/rid/arm if arm in NEW_NAMES.values()else
                w/'results/timeradar_cross_variable_fixed_evaluation_20261116_v1'/rid/arm)
        for stream in ('strict','confirmed3'):
            path=folder/f'{stream}_event_details.json'
            if not path.is_file():raise FileNotFoundError('Required original/new event details missing')
            references.append(dict(record=rid,arm=arm,stream=stream,path=path.relative_to(w).as_posix(),sha256=ev.sha(path)))
    ev._dump(output/'event_detail_references.json',dict(references=references,old_details_recomputed=False,
        timing='Native offline_first_hit_offset only; misses and UNKNOWN remain NA, never zero-filled'))
    ev._dump(output/'RESULT.json',dict(status='COMPLETED',primary_rows=primary,auxiliary_new_rows=new_aux,
        paired_rows=comparison,paired_differences=pairs,CPU_seconds=time.monotonic()-start,GPU_hours=0,
        new_training=False,new_forward=False,old_primary_metrics_recomputed=False,old_72_recomputed=False,
        inputs={str(p):ev.sha(p)for p in (native_path,auxiliary_path,cp,bp,ep,seal)+tuple(proofs)},
        source_sha256=ev.sha(__file__),evaluator_sha256=ev.sha(ev.__file__),
        primary='Canonical original bank.score V; own CAL .99 higher and strict>',
        auxiliary='V_numpy=mean(channel_variance), E=mean((reconstruction_mean-target)^2), E+.9V_numpy; all float32 pointwise; no thresholds',
        rounding='Historical72 NumPy variance reduction can differ minutely from official torch canonical V; primary V never substituted',
        scope='Four fixed exposed development cases; one seed2024; three SMD machines are not independent domains'))
    return comparison,pairs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path);a=p.parse_args();run(a.workspace,a.output,config=a.config)
