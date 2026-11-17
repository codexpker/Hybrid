"""Fixed PSM + SMD28 TimeRCD comparison, original and shared axes separate.

Only new TimeRCD banks are evaluated. Accepted TimeRadar/H512/LEFT rows are
copied by immutable CSV identity. No ranking, threshold or model is selected
from QUERY. Both TimeRCD statistics tracks remain explicitly offline.
"""
import argparse
from pathlib import Path
import time

import numpy as np
from . import timeradar_cross_variable_evaluation as ev
from .timeradar_smd_full_evaluation import read,local,checked,_csv_values,METRICS

ARMS=('A_official_role','B_fit_only')
OLD_ARMS=('zero','decoder_head','self','cross')
COMMON_ARMS=ARMS+OLD_ARMS+('history_mean512','LEFT_native_same_roles')
NATIVE_ARMS=ARMS+OLD_ARMS
OLD_SHA='61697563a1aa7d99a403b6e3e1062246f5be3f0697a825c09e04d4cb100edd90'
SMD_FULL_SHA='360fd27735e968c7e9034a119ad4c80cc12750ef000cd700e7381984d2bd7c03'
SMD_COMMON_SHA='d41ebd68aed36c0a9e3749a6e0b48c1edfc0c93a26dfdcc95eb6f48c338e5fda'
LEFT_COMMON_SHA='a9b42297e16b196399a370db42bec2933b2896abc717feed475a1da21dd6289b'
DATA_SHA='11e8b786b090928afa2bfdaf7ebe61e8bc2cbb93a6113aa7b073f22ef55e96b1'
LABEL_OLD_SHA='7e2bbe4bc79b9e089ec2c9a69397b9b3bdac3edfeb873a023527ad71691302a7'
LABEL_FULL_SHA='aee4c628d138cc2355535fb5e0bf24fa103d88fd44a50953b8ed2d4e2a05132b'
LEFT_CASES={'PSM','machine-1-1','machine-1-2','machine-1-3'}


def fixed_common_indices(record):
    n=record['query_points'];end=n//100*100
    if record['id']in LEFT_CASES:end=min(end,n//192*192)
    return np.arange(512,end,dtype=np.int64)


def terminal_gate(raw,records):
    expected={f'{r}/{a}/{role}.npz'for r in records for a in ARMS for role in ('CAL','QUERY')}
    actual=[b['relative']for b in raw['banks']]
    if (raw['status']not in ('COMPLETED','COMPLETED_WITH_FAILURES')or len(set(actual))!=len(actual)
            or not set(actual)<=expected or not raw['no_labels_decoded']):
        raise ValueError('Wrong terminal sealed TimeRCD bank set or label permission')
    if raw['status']=='COMPLETED'and set(actual)!=expected:
        raise ValueError('Completed comparison must contain all 116 fixed banks')
    return set(actual)


def load_new_bank(path,record,role,arm,descriptor,*,score_expression="probability"):
    checked(path,descriptor['sha256'])
    spec=next(r for r in record['roles']if r['role']==role);lo,hi=spec['bounds_half_open']
    expected=np.arange(lo,hi,dtype=np.int64)
    with np.load(path,allow_pickle=False)as z:
        index,score,available,maximum,end,valid=(z[k].copy()for k in
            ('index','score','available_index','max_input_index','window_end_index','valid'))
        if (not np.array_equal(index,expected)or score.shape!=index.shape or valid.dtype!=np.bool_
                or valid.shape!=index.shape or not valid.all()or not np.isfinite(score).all()
                or np.any((score<0)|(score>1))or not np.array_equal(score,z['prob'])):
            raise ValueError('Native probabilities must retain every real role point and no padding points')
        if not np.array_equal(available,maximum):raise ValueError('Complete time availability required')
        width=min(hi-lo,5000);expected_end=np.minimum(lo+(np.arange(hi-lo)//width+1)*width,hi)-1
        if not np.array_equal(end,expected_end):raise ValueError('Original nonoverlapping native5000 window identity changed')
        expected_available=np.full(len(index),hi-1,dtype=np.int64)if arm=='A_official_role'else end
        if not np.array_equal(available,expected_available):raise ValueError('Statistics-pool availability not disclosed correctly')
        if score_expression == 'logit':
            logit=z['logit'].copy()
            if not np.array_equal(logit,z['head_logits'][:,1]-z['head_logits'][:,0]) or not np.isfinite(logit).all():
                raise ValueError('Archived finite original head-logit difference required')
            score=logit
        elif score_expression != 'probability':
            raise ValueError('Fixed probability or original logit expression only')
    return index,score,available


def describe(rows,entities,methods,*,paired=False):
    """Per-entity descriptions; fixed n denominator, no pooled-point inference."""
    lookup={(r['record'],r['arm']):r for r in rows}
    if len(lookup)!=len(rows)or set(lookup)!={(r,m)for r in entities for m in methods}:
        raise ValueError('One row per fixed entity/method, including NA, required')
    macro=[];differences=[];pairs=[]
    for method in methods:
        for metric in METRICS:
            values=[lookup[r,method].get(metric)for r in entities]
            valid=[float(v)for v in values if v is not None and np.isfinite(v)]
            macro.append(dict(method=method,metric=metric,expected_entities=len(entities),n_valid=len(valid),
                macro_mean=float(np.mean(valid))if valid else None,median=float(np.median(valid))if valid else None,
                scope='SMD one exposed domain; entities are not independent datasets; unavailable/unknown is not zero'))
    if paired:
        for arm in ARMS:
            for reference in methods:
                if reference==arm:continue
                for metric in METRICS:
                    deltas=[]
                    for rid in entities:
                        a,b=lookup[rid,arm].get(metric),lookup[rid,reference].get(metric)
                        d=float(a-b)if a is not None and b is not None and np.isfinite(a)and np.isfinite(b)else None
                        differences.append(dict(record=rid,candidate=arm,reference=reference,metric=metric,difference=d))
                        if d is not None:deltas.append(d)
                    pairs.append(dict(candidate=arm,reference=reference,metric=metric,expected_entities=len(entities),n_valid=len(deltas),
                        macro_difference=float(np.mean(deltas))if deltas else None,median_difference=float(np.median(deltas))if deltas else None,
                        wins=sum(d>0 for d in deltas),ties=sum(d==0 for d in deltas),losses=sum(d<0 for d in deltas),
                        signs='Strict numerical sign; a positive false-alarm difference is a cost, not a benefit'))
    return dict(macro=macro,paired_rows=differences,paired_summary=pairs)


def inherited_tables(w,records):
    inputs={}
    def table(name,digest):
        path=w/name;inputs[str(path)]=digest;return _csv_values(path,digest)
    old=table('results/timeradar_cross_variable_fixed_evaluation_20261116_v1/metrics.csv',OLD_SHA)
    full=table('results/timeradar_smd_full_evaluation_20261116_v1/native_full_28x4.csv',SMD_FULL_SHA)
    shared=table('results/timeradar_smd_full_evaluation_20261116_v1/common_28x5.csv',SMD_COMMON_SHA)
    left=table('results/left_same_permission_comparison_20261116_v1/matched_external_4x6.csv',LEFT_COMMON_SHA)
    native=[dict(r)for r in old if r['record']=='PSM']+[dict(r)for r in full]
    common=[dict(r)for r in left]+[dict(r)for r in shared if r['record']not in LEFT_CASES]
    if len(native)!=116 or len(common)!=149:raise ValueError('Accepted immutable comparison rows incomplete')
    for r in native:
        record=records[r['record']]
        if r['scored_points']!=record['query_points']//100*100:raise ValueError('Accepted native100 axis changed')
        r.update(metric_origin='ACCEPTED_IMMUTABLE_CSV',axis_start=0,axis_end_exclusive=r['scored_points'],
                 statistics_permission='Historical TimeRadar one FIT-only StandardScaler; native model norm0/1 retained')
    for r in common:
        ix=fixed_common_indices(records[r['record']])
        if r['scored_points']!=len(ix):raise ValueError('Accepted shared-axis table mismatch')
        r.update(metric_origin='ACCEPTED_IMMUTABLE_CSV',axis_start=int(ix[0]),axis_end_exclusive=int(ix[-1])+1)
    for rid in records:
        if rid not in LEFT_CASES:
            common.append(dict(record=rid,arm='LEFT_native_same_roles',status='NOT_RUN_AS_PREDECLARED',AP=None,ROC_AUC=None,
                               reason='No same-role LEFT asset; not trained or replaced in this fixed comparison'))
    return native,common,inputs


def run(workspace,output,*,config,score_expression="probability"):
    began=time.monotonic();w,output=Path(workspace),Path(output);cp=local(w,config);cfg=read(cp)
    root=local(w,cfg['output']);sp=root/'RAW_BANK_SEAL.json';seal=read(sp)
    bp=w/'preparations/timercd_fixed_20261117_v1/data_binding.json';checked(bp,DATA_SHA)
    if cfg['data_binding_sha256']!=DATA_SHA or ev.sha(local(w,cfg['data_binding']))!=DATA_SHA:
        raise PermissionError('Original-unit raw roles/FIT-statistics binding mismatch')
    records={r['id']:r for r in read(bp)['records']}
    if len(records)!=29:raise ValueError('Exactly PSM plus all SMD28 retained')
    actual=terminal_gate(seal,records);descriptors={b['relative']:b for b in seal['banks']}
    if seal.get('config_sha256')!=ev.sha(cp):raise PermissionError('Frozen TimeRCD config identity mismatch')
    for b in seal['banks']:checked(root/b['relative'],b['sha256'])
    # No new QUERY labels are opened before the entire terminal bank set seals.
    ep=w/'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json';checked(ep,LABEL_OLD_SHA)
    fp=w/'preparations/timeradar_smd_full_20261116_v1/evaluation_manifest_full28.json';checked(fp,LABEL_FULL_SHA)
    labelsmeta={r['id']:r for r in read(fp)['records']};labelsmeta['PSM']=next(r for r in read(ep)['records']if r['id']=='PSM')
    native,common,inputs=inherited_tables(w,records)
    inputs.update({str(p):ev.sha(p)for p in (cp,sp,bp,ep,fp)})
    output.mkdir(parents=True,exist_ok=False);new=[];events=[]
    for rid,record in records.items():
        labels=None;ixcommon=fixed_common_indices(record)
        for arm in ARMS:
            keys={role:f'{rid}/{arm}/{role}.npz'for role in ('CAL','QUERY')}
            if not set(keys.values())<=actual:
                missing=dict(record=rid,arm=arm,status='MISSING_OR_FAILED',AP=None,ROC_AUC=None,
                             reason='No complete paired CAL/QUERY banks in terminal seal; fixed entity retained')
                native.append(dict(missing));common.append(dict(missing));continue
            banks={role:load_new_bank(root/key,record,role,arm,descriptors[key],score_expression=score_expression)for role,key in keys.items()}
            tau=ev.cal_threshold(banks['CAL'][1]);index,score,available=banks['QUERY']
            if labels is None:
                lp=local(w,labelsmeta[rid]['path']);checked(lp,labelsmeta[rid]['sha256']);inputs[str(lp)]=ev.sha(lp)
                with np.load(lp,allow_pickle=False)as z:
                    labels=z['label'].copy()
                    if not np.array_equal(z['index'],np.arange(len(labels))):raise ValueError('Full original event axis required')
            for scope,axis,rows in [('native_full',index,native),('shared_common',ixcommon,common)]:
                pos=np.searchsorted(index,axis)
                if np.any(pos>=len(index))or not np.array_equal(index[pos],axis):raise ValueError('Common axis must be actual scored intersection')
                provenance={str(root/keys[r]):descriptors[keys[r]]['sha256']for r in ('CAL','QUERY')}
                provenance.update({str(p):ev.sha(p)for p in (cp,sp,bp,lp)})
                result=ev.evaluate_arrays(axis,score[pos],labels,tau,available_index=available[pos])
                folder=output/scope/rid/arm
                ev._save_method(folder,result,axis,score[pos],labels,available[pos],provenance)
                row=ev._flat(rid,arm,result);row.update(status='OK',metric_origin='NEW_FROZEN_TIMERCd_BANK',
                    axis_start=int(axis[0]),axis_end_exclusive=int(axis[-1])+1,
                    statistics_permission='whole CAL and QUERY each normalized independently; offline future-role statistics'if arm==ARMS[0]
                    else'raw FIT mean/std(ddof0), std0->1e-8 once; frozen model; offline within-window observations',
                    score_definition=('Archived abnormal minus normal head logit; own CAL logit q99higher strict>' if score_expression=='logit' else 'Official pretrained anomaly_head mean channel logits then softmax class1; no logit substitution'),score_expression=score_expression)
                rows.append(row);new.append(row)
                for stream in ('strict','confirmed3'):
                    path=folder/(stream+'_event_details.json');events.append(dict(record=rid,arm=arm,scope=scope,stream=stream,
                        path=path.relative_to(w).as_posix(),sha256=ev.sha(path)))
    for rows,methods in [(native,NATIVE_ARMS),(common,COMMON_ARMS)]:
        rows.sort(key=lambda r:(r['record']!='PSM',r['record'],methods.index(r['arm'])))
    names=[r for r in records if r!='PSM']
    native_stats=describe([r for r in native if r['record']!='PSM'],names,NATIVE_ARMS)
    common_stats=describe([r for r in common if r['record']!='PSM'],names,COMMON_ARMS,paired=True)
    ev._write_csv(output/'native_29x6.csv',native);ev._write_csv(output/'common_29x8.csv',common)
    for scope,stats in [('native',native_stats),('common',common_stats)]:
        ev._write_csv(output/(scope+'_SMD_macro_median.csv'),stats['macro'])
        if stats['paired_rows']:ev._write_csv(output/(scope+'_SMD_paired_deltas.csv'),stats['paired_rows'])
        if stats['paired_summary']:ev._write_csv(output/(scope+'_SMD_paired_summary.csv'),stats['paired_summary'])
    ev._dump(output/'event_detail_references.json',dict(new_references=events,old_details_recomputed=False,
        timing='Offline_first_hit_offset only; missed/unobserved intervals remain UNKNOWN/NA, never online delay'))
    ev._dump(output/'RESULT.json',dict(status='COMPLETED'if len(actual)==116 else'INCOMPLETE',native_rows=native,common_rows=common,
        native_SMD=native_stats,common_SMD=common_stats,old_metrics_recomputed=False,new_method_evaluations=len(new),
        CPU_seconds=time.monotonic()-began,GPU_hours=0,new_forward=False,score_expression=score_expression,inputs=inputs,
        source_sha256=ev.sha(__file__),evaluator_sha256=ev.sha(ev.__file__),
        axis_scope='Native rows have method-own axes and are descriptive, not matched. All common rows share actual fixed intersection.',
        statistics='One exposed SMD domain. Macro/median and strict signs describe fixed entities; no pooled-point independent inference.',
        unavailable_LEFT='25 predeclared missing LEFT entity rows are kept NA; no added training',
        tracks='A versus B tests normalization fit pool only; not a loss/mechanism comparison. No weights trained or thresholds selected from QUERY.'))
    return native,common


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--score-expression',choices=['probability','logit'],default='probability');a=p.parse_args();run(a.workspace,a.output,config=a.config,score_expression=a.score_expression)
