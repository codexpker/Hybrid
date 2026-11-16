"""Fixed CAL evaluation of native TimeRadar banks; no model or score selection.

TimeRadar window/batch scores remain OFFLINE. Original event boundaries and
unknown positions are preserved. This module never converts availability into
an online delay claim. All thresholds use CAL scores only, without CAL labels.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .engine.events import binary_events, greedy_assign, real_blocks
from .hybrid_v1.event_details import event_details

ARMS = ('zero', 'decoder_head', 'self', 'cross')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dump(path, value):
    with Path(path).open('x', encoding='utf8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _ratio(a, b):
    return float(a / b) if b else None


def cal_threshold(score):
    """Fixed quantile; query labels are deliberately absent from this API."""
    score = np.asarray(score)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise ValueError('Finite nonempty CAL scores required')
    return float(np.quantile(score, .99, method='higher'))


def confirm3(index, alarm):
    """Current-point confirmation, never retrospective episode backfilling."""
    result = np.zeros(len(index), bool)
    for block in real_blocks(index, np.ones(len(index), bool)):
        p = alarm[block]
        if len(p) >= 3:
            result[block[2:]] = p[2:] & p[1:-1] & p[:-2]
    return result


def _stream(index, alarm, labels, available, *, online):
    y = labels[index]
    tp = int((alarm & (y == 1)).sum())
    fp = int((alarm & (y == 0)).sum())
    positives, normals = int(y.sum()), int((y == 0).sum())
    # event_details uses a strict > minimum; 511 implements frozen >=512.
    details = event_details(labels, index, alarm, steady_after=192, long_min_length=511)
    eligible = [e for e in details if e['eligible_for_event_metric']]
    # Incomplete/boundary-censored positive regions cannot silently become
    # truth events or normal regions. They split the matching axis.
    event_valid = y == 0
    for e in eligible:
        event_valid |= (index >= e['start']) & (index < e['end_exclusive'])
    matched, alarm_rows = set(), []
    for bi, ix in enumerate(real_blocks(index, event_valid)):
        truth = [(e['event_id'], e['start'], e['end_exclusive']) for e in eligible
                 if index[ix[0]] <= e['start'] and e['end_exclusive'] <= index[ix[-1]] + 1]
        pred = [(int(index[ix[a]]), int(index[ix[b-1]]) + 1)
                for a, b in binary_events(alarm[ix])]
        for (a, b), hit in zip(pred, greedy_assign(pred, truth)):
            if hit is not None:
                matched.add(hit)
            alarm_rows.append(dict(block=bi, start=a, end_exclusive=b, matched_event=hit))
    normal_fragments = sum(len(binary_events(alarm[ix]))
                           for ix in real_blocks(index, y == 0))
    any_hits = 0
    for e in details:
        hits = np.flatnonzero((index >= e['start']) & (index < e['end_exclusive']) & alarm)
        eligible_event = e['eligible_for_event_metric']
        any_hit = bool(len(hits)) if eligible_event else None
        any_hits += int(bool(any_hit))
        first = int(hits[0]) if len(hits) else None
        e.update(any_hit=any_hit,
                 one_to_one_matched=(e['event_id'] in matched) if eligible_event else None,
                 first_observed_alarm_index=int(index[first]) if first is not None else None,
                 first_available_alarm_index=int(available[first]) if first is not None else None,
                 offline_first_hit_offset=(int(index[first]-e['start'])
                                           if eligible_event and first is not None and not online else None),
                 online_delay_points=(int(available[first]-e['start'])
                                      if eligible_event and first is not None and online else None),
                 timing_status='CAUSAL_SAMPLE_DELAY' if online else 'OFFLINE_INDEX_OFFSET_NOT_CAUSAL_DELAY')
    ntrue, npred, m = len(eligible), len(alarm_rows), len(matched)
    longs = [e for e in eligible if e['total_points'] >= 512]
    steady = [e for e in longs if e['steady_coverage'] is not None]
    covered = sum(e['observed_alarm_points'] for e in eligible)
    metric = dict(TP=tp, FP=fp, FN=positives-tp, TN=normals-fp,
        point_Precision=_ratio(tp, tp+fp), point_Recall=_ratio(tp, positives),
        point_F1=_ratio(2*tp, 2*tp+fp+positives-tp),
        normal_exposure_points=normals, normal_false_alarm_points=fp,
        normal_FPR=_ratio(fp, normals), normal_FP_points_per_1000=_ratio(1000*fp, normals),
        normal_alarm_fragments=normal_fragments,
        normal_alarm_fragments_per_1000=_ratio(1000*normal_fragments, normals),
        original_events=len(details), complete_events=sum(e['observation_status']=='COMPLETE' for e in details),
        main_eligible_events=ntrue, any_hit_events=any_hits, any_hit_recall=_ratio(any_hits,ntrue),
        matched_events=m, predicted_alarm_segments=npred,
        one_to_one_eventPrecision=_ratio(m,npred), one_to_one_eventRecall=_ratio(m,ntrue),
        one_to_one_eventF1=_ratio(2*m,npred+ntrue),
        complete_event_point_coverage=_ratio(covered,sum(e['total_points'] for e in eligible)),
        long_events=len(longs), long_steady_macro_coverage=(float(np.mean([e['steady_coverage'] for e in steady])) if steady else None),
        long_steady_pooled_coverage=(_ratio(sum(e['steady_coverage']*e['steady_total_points'] for e in steady),
                                              sum(e['steady_total_points'] for e in steady))),
        longest_complete_event_miss=max((e['longest_miss_points'] for e in eligible), default=None),
        matching_rule='chronological_first_unmatched_overlap_greedy',
        event_eligibility='fully_scored_and_not_original_boundary_censored',
        normal_fragment_definition='normal-region fragments, clipped at anomaly boundaries; not new onsets')
    return dict(metrics=metric, events=details, alarms=alarm_rows)


def evaluate_arrays(index, score, full_labels, threshold=None, *, available_index=None, online=False):
    """One immutable score axis. A missing threshold means ranking-only."""
    index, score, labels = np.asarray(index), np.asarray(score), np.asarray(full_labels)
    if (index.ndim != 1 or index.dtype.kind not in 'iu' or not len(index)
            or score.shape != index.shape or not np.isfinite(score).all()
            or np.any(np.diff(index) <= 0) or labels.ndim != 1
            or not np.isin(labels,[0,1]).all() or index[0] < 0 or index[-1] >= len(labels)):
        raise ValueError('Finite scores, unique original indices and full binary labels required')
    available = index.copy() if available_index is None else np.asarray(available_index)
    if available.shape != index.shape or available.dtype.kind not in 'iu' or np.any(available < index):
        raise ValueError('Actual original maximum input/availability indices required')
    if online and np.any(available > index):
        raise PermissionError('Offline window/batch scores cannot become online by cropping')
    y = labels[index]
    both = len(np.unique(y)) == 2
    ranking = dict(AP=float(average_precision_score(y,score)) if both else None,
                   ROC_AUC=float(roc_auc_score(y,score)) if both else None,
                   scored_points=len(index), positive_points=int(y.sum()),
                   unscored_points=len(labels)-len(index))
    result = dict(ranking=ranking, threshold=threshold,
        threshold_rule='CAL_q.99_higher_strict_greater' if threshold is not None else 'UNAVAILABLE_NO_NEW_QUERY_THRESHOLD',
        mode='CAUSAL' if online else 'OFFLINE',
        index_units='original sample positions; no verified physical sampling clock')
    if threshold is None:
        result.update(strict=None, confirmed3=None)
    else:
        if not np.isfinite(threshold):
            raise ValueError('Finite fixed threshold required')
        alarm = score > threshold
        result.update(strict=_stream(index,alarm,labels,available,online=online),
                      confirmed3=_stream(index,confirm3(index,alarm),labels,available,online=online))
    return result


def _load_bank(path):
    with np.load(path,allow_pickle=False) as bank:
        index, score = bank['index'].copy(), bank['score'].copy()
        available = bank['available_index'].copy()
        maximum = bank['max_input_index'].copy()
    if not np.array_equal(available,maximum):
        raise ValueError('Native score availability must retain batch maximum input identity')
    return index,score,available


def _save_method(folder, result, index, score, labels, available, inputs):
    folder.mkdir(parents=True,exist_ok=False)
    p=folder/'calculation_inputs.npz'
    np.savez_compressed(p,index=index,score=score,full_labels=labels,available_index=available)
    result['provenance']=dict(inputs=inputs,calculation_inputs_sha256=sha(p),evaluator_sha256=sha(__file__))
    _dump(folder/'metrics.json',result)
    for stream in ('strict','confirmed3'):
        if result[stream] is not None:
            _dump(folder/(stream+'_event_details.json'),result[stream]['events'])


def _flat(record,arm,result):
    row=dict(record=record,arm=arm,mode=result['mode'],threshold=result['threshold'],**result['ranking'])
    for stream in ('strict','confirmed3'):
        if result[stream] is not None:
            row.update({stream+'_'+k:v for k,v in result[stream]['metrics'].items()
                        if isinstance(v,(int,float,type(None)))})
    return row


def _write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='',encoding='utf8') as handle:
        writer=csv.DictWriter(handle,fields);writer.writeheader();writer.writerows(rows)


def run(workspace, results, output, *, allow_incomplete=False):
    w,results,output=Path(workspace),Path(results),Path(output)
    binding=w/'preparations/timeradar_cross_variable_20261116_v1/data_binding.json'
    contract=json.loads(binding.read_text(encoding='utf8'))
    evalpath=w/'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json'
    evaluation=json.loads(evalpath.read_text(encoding='utf8'))
    labels_by_id={r['id']:r for r in evaluation['records']}
    pending=[];ready=[]
    for record in contract['records']:
        for arm in ARMS:
            folder=results/record['id']/arm/'score'
            if all((folder/name).is_file() for name in ('CAL.npz','QUERY.npz','SCORE_RECEIPT.json')):
                ready.append((record['id'],arm,folder))
            else:
                pending.append([record['id'],arm])
    if pending and not allow_incomplete:
        raise FileNotFoundError('Complete score banks still pending: '+repr(pending))
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for rid,arm,folder in ready:
        receipt=json.loads((folder/'SCORE_RECEIPT.json').read_text())
        if receipt['record']!=rid or receipt['arm']!=arm or receipt['status']!='COMPLETED':
            raise ValueError('Completed score receipt identity required')
        record=next(r for r in contract['records'] if r['id']==rid)
        rolemap={r['role']:r for r in record['roles']}
        receipt_roles={r['role']:r for r in receipt['roles']}
        for role in ('CAL','QUERY'):
            if receipt_roles[role]['sha256']!=rolemap[role]['sha256']:
                raise PermissionError('Score receipt uses different physical data roles')
        banks={Path(r['path']).name:r for r in receipt['banks']}
        for name in ('CAL.npz','QUERY.npz'):
            if sha(folder/name)!=banks[name]['sha256']:
                raise PermissionError('Score archive differs from completed runner receipt')
        _,cal,_=_load_bank(folder/'CAL.npz')
        tau=cal_threshold(cal)  # No query labels participate in this operation.
        index,score,available=_load_bank(folder/'QUERY.npz')
        calindex,_,_=_load_bank(folder/'CAL.npz')
        for role,actual in (('CAL',calindex),('QUERY',index)):
            lo=rolemap[role]['bounds_half_open'][0]
            if not np.array_equal(actual,np.arange(lo,lo+rolemap[role]['scored_points'])):
                raise ValueError('Native score axis differs from frozen role windows')
        lm=labels_by_id[rid];lp=w/lm['path']
        if sha(lp)!=lm['sha256']:
            raise PermissionError('Evaluation labels differ from fixed manifest')
        with np.load(lp,allow_pickle=False) as z:
            labels=z['label'].copy()
            if not np.array_equal(z['index'],np.arange(len(labels))):
                raise ValueError('Original complete label axis required')
        result=evaluate_arrays(index,score,labels,tau,available_index=available)
        result['CAL_exceedance_rate']=float(np.mean(cal>tau))
        result['CAL_points']=len(cal)
        inputs={str(p):sha(p) for p in (folder/'CAL.npz',folder/'QUERY.npz',folder/'SCORE_RECEIPT.json',lp,binding,evalpath)}
        _save_method(output/rid/arm,result,index,score,labels,available,inputs)
        rows.append(_flat(rid,arm,result))
    _write_csv(output/'metrics.csv',rows)
    _dump(output/'RESULT.json',dict(status='INCOMPLETE' if pending else 'COMPLETED',rows=rows,pending=pending,
        fixed_protocol='CAL .99 higher; >; C3 current point; offline; long>=512, steady_after192',
        GPU_hours=0,new_training=False,new_forward=False))
    return rows


def external_psm(workspace, results, output):
    """Fixed [512,87744) external comparison; no new model forward or fitting."""
    w,results,output=Path(workspace),Path(results),Path(output)
    common=np.arange(512,87744,dtype=np.int64)
    label_path=w/'data/timeradar_cross_variable_20261116_v1/evaluation/PSM_labels.npz'
    with np.load(label_path,allow_pickle=False) as z:labels=z['label'].copy()
    candidates=[]
    lp=w/'results/left_psm_R_score_seed2025_20261101_v1/raw_scores.npz'
    if sha(lp)!='e6457b6081688625babd57fced9c9492efdda42086cb68df65e51d63e7632702':
        raise PermissionError('Accepted LEFT score hash changed')
    with np.load(lp,allow_pickle=False) as z:
        ix=z['absolute_time_index']-132481;s=z['canonical_score'].copy()
        if not np.array_equal(s,z['channel_scores'].mean(-1).reshape(-1)) or z['channel_names'].tolist()!=['feature_'+str(i) for i in range(25)]:
            raise ValueError('LEFT canonical channel mean/order changed')
        candidates.append(('LEFT_R_targettrained',ix,s,z['point_batch_end']-132481,None,False,lp,
                           'Target full-pretest training/scaler; native offline; no bound independent CAL threshold'))
    hp=w/'results/hybrid_v1_psm_Z1_fixed_evaluation_20261102_v1/history_mean512/calculation_inputs.npz'
    if sha(hp)!='f1c792db8c65d95783c6ac183f5bbca22de3621164384d70c0aac4e40c6a423a':
        raise PermissionError('Accepted HistoryMean score hash changed')
    with np.load(hp,allow_pickle=False) as z:
        if not np.array_equal(labels,z['full_labels']):raise ValueError('Original PSM labels changed')
        ok=z['valid']
        candidates.append(('HistoryMean512_ZS',z['index'][ok],z['score'][ok],z['available_index'][ok],
                           2.9690243291906797,True,hp,'Historical source CAL; causal; different preprocessing permissions'))
    for arm in ARMS:
        folder=results/'PSM'/arm/'score'
        if not(folder/'SCORE_RECEIPT.json').is_file():continue
        receipt=json.loads((folder/'SCORE_RECEIPT.json').read_text())
        banksha={Path(x['path']).name:x['sha256'] for x in receipt['banks']}
        for name in ('CAL.npz','QUERY.npz'):
            if sha(folder/name)!=banksha[name]:raise PermissionError('Current bank hash changed')
        _,cal,_=_load_bank(folder/'CAL.npz');ix,s,a=_load_bank(folder/'QUERY.npz')
        candidates.append((arm,ix,s,a,cal_threshold(cal),False,folder/'QUERY.npz','Strict FIT-only scaler; native offline; same-role independent CAL'))
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for name,ix,s,a,tau,online,path,permission in candidates:
        pos=np.searchsorted(ix,common)
        if np.any(pos>=len(ix)) or not np.array_equal(ix[pos],common):
            raise ValueError('Fixed common axis is not completely scored: '+name)
        result=evaluate_arrays(common,s[pos],labels,tau,available_index=a[pos],online=online)
        if name=='HistoryMean512_ZS':result['threshold_rule']='UNCHANGED_ACCEPTED_SOURCE_CAL_THRESHOLD'
        result['permissions']=permission
        _save_method(output/name,result,common,s[pos],labels,a[pos],{str(path):sha(path),str(label_path):sha(label_path)})
        row=_flat('PSM_common512_87744',name,result);row['permissions']=permission;rows.append(row)
    _write_csv(output/'metrics.csv',rows)
    _dump(output/'RESULT.json',dict(rows=rows,common_axis=[512,87744],GPU_hours=0,new_forward=False,
        CATCH='Historical only; input/protocol incompatibility, not silently reused',
        warning='Equal points do not equal training/statistical/temporal permissions'))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--external-psm',action='store_true');p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args()
    rows=(external_psm(a.workspace,a.results,a.output) if a.external_psm else
          run(a.workspace,a.results,a.output,allow_incomplete=a.allow_incomplete))
    print(json.dumps(rows,indent=2,allow_nan=False))


if __name__=='__main__':main()
