"""Only new-row CPU evaluation for the single selective-context experiment.

Original 16 native rows and 24 matching-data external rows are hash-checked and
reused, never rescored. New QUERY labels open only after all 16 banks seal.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from . import timeradar_cross_variable_evaluation as ev
from .timeradar_smd_full_evaluation import checked,checked_bank,local,_csv_values,METRICS

RECORDS=('PSM','machine-1-1','machine-1-2','machine-1-3')
NEW_ARMS=('selective','capacity_self')
OLD_ARMS=('zero','decoder_head','self','cross')
OLD_NATIVE_SHA='61697563a1aa7d99a403b6e3e1062246f5be3f0697a825c09e04d4cb100edd90'
OLD_EXTERNAL_SHA='a9b42297e16b196399a370db42bec2933b2896abc717feed475a1da21dd6289b'
BINDING_SHA='d10cd11eca48f570f0e4a122f133cd8be60792c743ba68e0554f293ec4e82149'
LABEL_MANIFEST_SHA='7e2bbe4bc79b9e089ec2c9a69397b9b3bdac3edfeb873a023527ad71691302a7'


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def fixed_keys():
    return {f'{r}/{a}/score/{role}.npz'for r in RECORDS for a in NEW_ARMS for role in ('CAL','QUERY')}


def check_complete_seal(seal):
    banks=seal['banks'];keys={r['relative']for r in banks}
    if seal['status']!='COMPLETED'or len(banks)!=16 or len(keys)!=16 or keys!=fixed_keys():
        raise ValueError('Exactly all four records, two fixed arms and 16 CAL/QUERY banks must seal first')


def paired_rows(rows,*,scope):
    """Entity-wise, single seed, same-axis differences without dataset pooling."""
    lookup={(r['record'],r['arm']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate entity/method rows')
    result=[]
    comparators=('capacity_self','cross','self','decoder_head','zero')
    if scope=='matching_external':comparators+=('LEFT_native_same_roles','history_mean512')
    for rid in RECORDS:
        candidate=lookup[rid,'selective']
        for name in comparators:
            control=lookup[rid,name]
            if candidate['scored_points']!=control['scored_points']:
                raise ValueError('Paired comparison cannot mix scoring axes')
            if scope=='matching_external'and(candidate['common_start'],candidate['common_end_exclusive'])!=(control['common_start'],control['common_end_exclusive']):
                raise ValueError('External common-axis boundaries differ')
            for metric in METRICS:
                a,b=candidate.get(metric),control.get(metric)
                result.append(dict(record=rid,comparison='selective-minus-'+name,metric=metric,
                    delta=float(a-b)if a is not None and b is not None else None,seed=2024,scope=scope,
                    note='Single fixed seed; no independent-domain, significance or same-false-alarm-budget claim'))
    return result


def _cost(jobs,rid,arm):
    items=[r for r in jobs if r['record']==rid and r['arm']==arm]
    result={phase+'_GPU_hours':sum(r.get('GPU_hours',r.get('gpu_hours',0.))for r in items if r['phase']==phase)
            for phase in ('train','score')}
    result['selected_FIT_diagnostic_GPU_hours']=sum(r.get('GPU_hours',r.get('gpu_hours',0.))for r in items if r['phase']=='selected_mechanism')
    return result


def _parameters(w,root,rid,arm):
    if arm in ('zero','history_mean512'):return dict(trainable_parameters=0,target_updates=0,parameter_source='No target-trained parameters')
    path=root/rid/arm/'train/TRAIN_RECEIPT.json'
    if path.is_file():
        receipt=read(path)
        return dict(trainable_parameters=receipt['trainable_parameters'],target_updates=receipt['updates'],
                    parameter_source=path.relative_to(w).as_posix(),parameter_receipt_sha256=ev.sha(path))
    return dict(trainable_parameters=None,target_updates=None,parameter_source='RECEIPT_NOT_AVAILABLE_NO_GUESS')


def check_parameter_match(selective,capacity,old_mean):
    values=(selective['trainable_parameters'],capacity['trainable_parameters'],old_mean['trainable_parameters'])
    if any(type(v)is not int for v in values)or values[0]!=values[1]or values[0]-values[2]!=512:
        raise ValueError('Selective and capacity control require exact same parameters, old mean plus fixed 512 Q/K parameters')


def run(workspace,output,*,config=None):
    started=time.monotonic();w,output=Path(workspace),Path(output)
    cp=(w/'hybrid_ad/configs/timeradar-selective-context-20261116.json'if config is None else local(w,config))
    cfg=read(cp);root=local(w,cfg['output']);seal_path=root/'RAW_BANK_SEAL.json';seal=read(seal_path)
    check_complete_seal(seal)
    if cfg['seed']!=2024 or cfg['score_seed']!=2024 or cfg['select_seed']!=84551 or cfg['updates']!=2000:
        raise PermissionError('The fixed training/scoring/SELECT recipe changed')
    if 'config_sha256'in seal and seal['config_sha256']!=ev.sha(cp):raise PermissionError('Seal uses another config')
    bp=local(w,cfg['data_binding']);checked(bp,BINDING_SHA)
    if cfg['data_binding_sha256']!=BINDING_SHA:raise PermissionError('Original physical roles must be reused')
    binding={r['id']:r for r in read(bp)['records']}
    if set(binding)!=set(RECORDS):raise ValueError('Original four records must remain fixed')
    for bank in seal['banks']:checked(root/bank['relative'],bank['sha256'])
    # No new labels have been decoded before the complete 16-bank hash gate.
    oldnative=w/'results/timeradar_cross_variable_fixed_evaluation_20261116_v1/metrics.csv'
    oldexternal=w/'results/left_same_permission_comparison_20261116_v1/matched_external_4x6.csv'
    native=_csv_values(oldnative,OLD_NATIVE_SHA);external=_csv_values(oldexternal,OLD_EXTERNAL_SHA)
    if len(native)!=16 or len(external)!=24:raise ValueError('Keep all original comparison rows')
    if {(r['record'],r['arm'])for r in native}!={(r,a)for r in RECORDS for a in OLD_ARMS}:raise ValueError('Original native identities changed')
    oldmodel=w/'results/timeradar_cross_variable_20261116_v1';oldseal=oldmodel/'RAW_BANK_SEAL.json'
    checked(oldseal,'e073be13716f3b11e3121043c9c74086a20f560b6f5519cf619b9b3d8e91f146')
    oldjobs=read(oldseal)['jobs'];jobs=seal['jobs']
    for row in native:
        row.update(metric_origin='ACCEPTED_ORIGINAL_2024_ROW',seed=2024)
        row.update(_parameters(w,oldmodel,row['record'],row['arm']))
        row.update(_cost(oldjobs,row['record'],row['arm']))
    for row in external:
        row['metric_origin']='ACCEPTED_MATCHED_EXTERNAL_ROW'
        if row['arm']in OLD_ARMS+('history_mean512',):
            row.update(_parameters(w,oldmodel,row['record'],row['arm']))
        elif row['arm']=='LEFT_native_same_roles':
            path=w/'results/left_same_permission_20261116_v1'/row['record']/'train/RECEIPT.json';receipt=read(path)
            row.update(trainable_parameters=receipt['parameters'],target_updates=receipt['updates'],target_epochs=receipt['epochs'],
                       parameter_source=path.relative_to(w).as_posix(),parameter_receipt_sha256=ev.sha(path))
    for rid in RECORDS:
        check_parameter_match(_parameters(w,root,rid,'selective'),_parameters(w,root,rid,'capacity_self'),_parameters(w,oldmodel,rid,'cross'))
    ep=w/'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json';checked(ep,LABEL_MANIFEST_SHA)
    evalmeta={r['id']:r for r in read(ep)['records']}
    output.mkdir(parents=True,exist_ok=False)
    for rid in RECORDS:
        oldaxis=[r for r in external if r['record']==rid]
        endpoints={(r['common_start'],r['common_end_exclusive'])for r in oldaxis}
        if len(endpoints)!=1:raise ValueError('Accepted external methods must have one shared axis')
        lo,hi=endpoints.pop();common=np.arange(lo,hi,dtype=np.int64)
        lp=w/evalmeta[rid]['path'];checked(lp,evalmeta[rid]['sha256'])
        with np.load(lp,allow_pickle=False)as z:
            labels=z['label'].copy()
            if not np.array_equal(z['index'],np.arange(len(labels))):raise ValueError('Complete original labels required')
        for arm in NEW_ARMS:
            folder=root/rid/arm/'score'
            index,score,available,tau,rp=checked_bank(folder,rid,arm,binding[rid],cp)
            inputs={str(p):ev.sha(p)for p in (folder/'CAL.npz',folder/'QUERY.npz',rp,lp,cp,bp,ep,seal_path)}
            result=ev.evaluate_arrays(index,score,labels,tau,available_index=available)
            ev._save_method(output/'native_full'/rid/arm,result,index,score,labels,available,inputs)
            row=ev._flat(rid,arm,result);row.update(metric_origin='NEW_SEALED_NATIVE_ROW',seed=2024)
            row.update(_parameters(w,root,rid,arm));row.update(_cost(jobs,rid,arm));native.append(row)
            pos=np.searchsorted(index,common)
            if np.any(pos>=len(index))or not np.array_equal(index[pos],common):raise ValueError('New arm must cover the already fixed external axis')
            result=ev.evaluate_arrays(common,score[pos],labels,tau,available_index=available[pos])
            ev._save_method(output/'matching_external'/rid/arm,result,common,score[pos],labels,available[pos],inputs)
            erow=ev._flat(rid,arm,result);erow.update(common_start=lo,common_end_exclusive=hi,
                metric_origin='NEW_SEALED_MATCHED_AXIS_ROW',seed=2024)
            erow.update(_parameters(w,root,rid,arm));erow.update(_cost(jobs,rid,arm))
            erow['training_information']='Official pretrained TimeRadar frozen backbone; decoder+one adapter target FIT 2000 reconstruction updates; native L1'
            erow['data_CAL_information']='Original same physical FIT/SELECT/CAL, FIT-only standardizer; own CAL q99higher and strict>';external.append(erow)
    order=OLD_ARMS+NEW_ARMS
    native.sort(key=lambda r:(RECORDS.index(r['record']),order.index(r['arm'])))
    external.sort(key=lambda r:(RECORDS.index(r['record']),str(r['arm'])))
    ablation=[dict(r,aggregation={'self':'original_self_lowrank','cross':'stopped_uniform_other_channels',
        'capacity_self':'matched_QK_self_only_capacity_control','selective':'single_selective_other_channel_context'}[r['arm']])
        for r in native if r['arm']in ('self','cross','capacity_self','selective')]
    paired_native=paired_rows(native,scope='native_full');paired_external=paired_rows(external,scope='matching_external')
    ev._write_csv(output/'native_full_4x6.csv',native);ev._write_csv(output/'matched_external_4x8.csv',external)
    ev._write_csv(output/'aggregation_ablation.csv',ablation)
    ev._write_csv(output/'paired_native_differences.csv',paired_native);ev._write_csv(output/'paired_external_differences.csv',paired_external)
    references=[]
    for scope,table in (('native_full',native),('matching_external',external)):
        for row in table:
            rid,arm=row['record'],row['arm']
            if arm in NEW_ARMS:folder=output/scope/rid/arm
            elif scope=='native_full':folder=w/'results/timeradar_cross_variable_fixed_evaluation_20261116_v1'/rid/arm
            elif arm=='LEFT_native_same_roles'or rid=='machine-1-3':folder=w/'results/left_same_permission_comparison_20261116_v1'/rid/arm
            elif rid=='PSM':
                old_name='HistoryMean512_strict_targetCAL'if arm=='history_mean512'else arm
                folder=w/'results/timeradar_cross_variable_external_20261116_v1'/old_name
            else:folder=w/'results/timeradar_cross_variable_simple_common_20261116_v1'/rid/arm
            for stream in ('strict','confirmed3'):
                path=folder/f'{stream}_event_details.json'
                if not path.is_file():raise FileNotFoundError('Original/new complete event-detail reference missing: '+str(path))
                references.append(dict(scope=scope,record=rid,arm=arm,stream=stream,
                    path=path.relative_to(w).as_posix(),sha256=ev.sha(path),
                    timing='TimeRadar/LEFT offline_first_hit_offset is not online delay; misses/UNKNOWN remain NA'))
    ev._dump(output/'event_detail_references.json',dict(references=references,
        old_event_details_recomputed=False,new_event_details='Generated only for the eight new method-record combinations'))
    ev._dump(output/'RESULT.json',dict(status='COMPLETED',native_rows=native,external_rows=external,
        paired_native=paired_native,paired_external=paired_external,CPU_seconds=time.monotonic()-started,
        GPU_hours=0,new_training=False,new_forward=False,old_rows_recomputed=False,
        source_sha256=ev.sha(__file__),evaluator_sha256=ev.sha(ev.__file__),
        event_detail_references_sha256=ev.sha(output/'event_detail_references.json'),
        inputs={str(p):ev.sha(p)for p in (oldnative,oldexternal,oldseal,cp,bp,ep,seal_path)},
        notes=['All four are exposed development cases; three SMD machines are not three independent domains',
               'All TimeRadar/LEFT native scores remain offline; first hit offsets are not causal delays',
               'Same data permissions do not equate LEFT target-native20epoch to official-pretrained TimeRadar2000 adaptation',
               'Old uniform-cross and blindspot recipes remain stopped; no new same-weight uniform QUERY variant was evaluated']))
    return native,external,ablation,paired_native,paired_external


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path)
    a=p.parse_args();run(a.workspace,a.output,config=a.config)
