"""Fixed all-28 SMD evaluation: old3 reused, new25 opened after global seal.

Native full-axis and five-method history-compatible axes are separate tables.
Machine summaries describe one exposed SMD domain, never independent domains.
Missing methods retain explicit NA rows and make the whole study INCOMPLETE.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from . import timeradar_cross_variable_evaluation as ev

OLD=('machine-1-1','machine-1-2','machine-1-3')
FULL_METHODS=ev.ARMS
COMMON_METHODS=FULL_METHODS+('history_mean512',)
FULL_SHA='61697563a1aa7d99a403b6e3e1062246f5be3f0697a825c09e04d4cb100edd90'
COMMON_SHA='70820741b08603a857d6ec7acd312d2f1754b2bc57f470f164bc02477c25214e'
METRICS=('AP','ROC_AUC')+tuple(p+k for p in ('strict_','confirmed3_')for k in (
    'point_Precision','point_Recall','point_F1','any_hit_events','any_hit_recall',
    'one_to_one_eventPrecision','one_to_one_eventRecall','one_to_one_eventF1',
    'normal_FPR','normal_FP_points_per_1000','normal_alarm_fragments_per_1000',
    'complete_event_point_coverage','long_steady_macro_coverage','long_steady_pooled_coverage',
    'longest_complete_event_miss'))


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def local(w,path):
    text=str(path);prefix='/home/a312/tsad_research/'
    if text.startswith(prefix):return w/text[len(prefix):]
    p=Path(path);return p if p.is_absolute()else w/p


def checked(path,expected):
    if ev.sha(path)!=expected:raise PermissionError('Fixed input hash mismatch: '+str(path))
    return path


def reuse_old_csv(path,expected,methods):
    checked(path,expected);rows=[]
    with Path(path).open(newline='',encoding='utf8')as handle:
        for original in csv.DictReader(handle):
            if original['record']not in OLD or original['arm']not in methods:continue
            row={}
            for key,value in original.items():
                if key in ('record','arm','mode'):row[key]=value
                elif not value:row[key]=None
                else:row[key]=json.loads(value)
            row.update(status='OK',metric_origin='IMMUTABLE_OLD3_2024_CSV',seed=2024);rows.append(row)
    if len(rows)!=len(OLD)*len(methods):raise ValueError('Accepted old3 table incomplete')
    return rows


def summaries(rows,entities,methods):
    """Macro/median and paired signs with explicit missing denominators."""
    lookup={}
    for row in rows:
        key=row['record'],row['arm']
        if key in lookup:raise ValueError('Duplicate entity/method row')
        lookup[key]=row
    expected={(r,m)for r in entities for m in methods}
    if set(lookup)!=expected:raise ValueError('Keep one row for every fixed entity/method, including NA')
    groups={'old3':[r for r in entities if r in OLD],
            'new25':[r for r in entities if r not in OLD],'all28':list(entities)}
    macro=[];pairs=[];pair_summary=[]
    for group,names in groups.items():
        for method in methods:
            for metric in METRICS:
                values=[lookup[r,method].get(metric)for r in names]
                valid=[float(v)for v in values if v is not None and np.isfinite(v)]
                macro.append(dict(group=group,method=method,metric=metric,expected_entities=len(names),n_valid=len(valid),
                    macro_mean=float(np.mean(valid))if valid else None,median=float(np.median(valid))if valid else None,
                    denominator_note='valid n / fixed expected; NA never replaced by zero; long-event NA may mean no eligible long events'))
        for comparator in methods:
            if comparator=='cross':continue
            for metric in METRICS:
                deltas=[]
                for rid in names:
                    a,b=lookup[rid,'cross'].get(metric),lookup[rid,comparator].get(metric)
                    delta=float(a-b)if a is not None and b is not None and np.isfinite(a)and np.isfinite(b)else None
                    deltas.append(delta)
                    if group=='all28':pairs.append(dict(record=rid,comparison='cross-minus-'+comparator,metric=metric,
                        delta=delta,status='PAIRED'if delta is not None else'NA',seed=2024))
                valid=[v for v in deltas if v is not None]
                pair_summary.append(dict(group=group,comparison='cross-minus-'+comparator,metric=metric,
                    expected_entities=len(names),paired_n=len(valid),mean_delta=float(np.mean(valid))if valid else None,
                    median_delta=float(np.median(valid))if valid else None,
                    positive=sum(v>0 for v in valid),tie=sum(v==0 for v in valid),negative=sum(v<0 for v in valid),
                    wins=sum(v>0 for v in valid)if metric in ('AP','ROC_AUC')else None,
                    losses=sum(v<0 for v in valid)if metric in ('AP','ROC_AUC')else None,
                    rule='strict raw sign, zero tolerance; AP/ROC higher is better; other signs are not benefit labels'))
    complete=all(r.get('status')=='OK'for r in rows)
    return dict(status='COMPLETED'if complete else'INCOMPLETE',macro=macro,paired_rows=pairs,paired_summary=pair_summary)


def verify_resource_recovery(w,root,seal,expected):
    """Validate one unstarted score recovery without rewriting the 198-bank seal."""
    prep=w/'preparations/timeradar_smd_full_20261116_v1'
    if seal!=root/'RECOVERED_RAW_BANK_SEAL.json':raise ValueError('Explicit supported recovery seal path required')
    raw=read(seal);original_path=root/'RAW_BANK_SEAL.json'
    original_global=prep/'ALL_SMD_EXTENSION_RAW_BANK_SEAL.json'
    start_path=prep/'RESOURCE_RECOVERY_START.json';final_path=prep/'RESOURCE_RECOVERY_FINAL_SEAL.json'
    final,start=read(final_path),read(start_path)
    checked(seal,final['recovered_result_seal_sha256'])
    if {k:v for k,v in final.items()if k!='recovered_result_seal_sha256'}!=raw:raise PermissionError('Recovery wrapper does not bind the same result')
    checked(original_path,raw['original_controller_result_sha256']);checked(original_global,raw['original_controller_global_sha256'])
    checked(start_path,raw['recovery_start_sha256'])
    original=read(original_path);prior_global=read(original_global)
    if prior_global['result_seal_sha256']!=ev.sha(original_path):raise PermissionError('Original global/result seal chain mismatch')
    if (original['status']!='COMPLETED_WITH_FAILURES'or len(original['banks'])!=198
            or raw['status']!='COMPLETED'or not raw['resource_recovery_only']or not raw['original_198_banks_unchanged']):
        raise ValueError('Only the declared 198+2 resource recovery is accepted')
    old={b['relative']:b for b in original['banks']};new={b['relative']:b for b in raw['banks']}
    if len(old)!=198 or len(new)!=200 or set(new)!=expected:raise ValueError('Recovered bank set must be exactly the fixed 200')
    if any(new.get(k)!=v for k,v in old.items()):raise PermissionError('An original 198-bank identity changed')
    job=raw['recovery_job'];rid,arm=job['record'],job['arm']
    additions={f'{rid}/{arm}/score/{r}.npz'for r in ('CAL','QUERY')}
    if set(new)-set(old)!=additions or job['phase']!='score'or job['exit_code']!=0:raise ValueError('Recovery must add one successful score job and its two banks')
    if (len(original['jobs'])!=174 or len(raw['jobs'])!=175 or raw['jobs']!=original['jobs']+[job]
            or any(j['exit_code']!=0 for j in raw['jobs'])
            or any((j['record'],j['arm'],j['phase'])==(rid,arm,'score')for j in original['jobs'])):
        raise ValueError('Recovery must not repeat an already executed score or any training')
    if (start['original_result_sha256']!=ev.sha(original_path)or start['original_global_sha256']!=ev.sha(original_global)
            or not start['no_retraining']or not start['no_prior_score_execution']
            or start['preflight_rejection_GPU_hours']!=0 or start['recovery_gpu']!=job['gpu']):
        raise PermissionError('Unstarted resource recovery authorization identity mismatch')
    rejection=local(w,start['preflight_rejection_path']);checked(rejection,start['preflight_rejection_sha256'])
    reject=read(rejection)
    if (reject['record'],reject['arm'])!=(rid,arm)or raw['resource_preflight_rejections']!=original['failures']:
        raise PermissionError('Original resource rejection must remain disclosed')
    prefix=prep/'jobs'/f'{rid}__{arm}__score'
    launch_path=Path(str(prefix)+'_LAUNCH.json');exit_path=Path(str(prefix)+'_EXIT.json')
    launch=read(launch_path)
    if read(exit_path)!=job or launch['pid']!=job['pid']or launch['gpu']!=job['gpu']:raise PermissionError('Recovery launch/exit job identity mismatch')
    command=launch['command']
    if ('score'not in command or command[command.index('--record')+1]!=rid or command[command.index('--arm')+1]!=arm):
        raise PermissionError('Recovery command is not the original score identity')
    cp=local(w,command[command.index('--config')+1]);checked(cp,start['config_sha256'])
    wave=next(v for v in raw['waves']if rid in v['records'])
    if wave['config_sha256']!=start['config_sha256']:raise PermissionError('Recovery changed wave configuration')
    train_path=root/rid/arm/'train/TRAIN_RECEIPT.json';score_path=root/rid/arm/'score/SCORE_RECEIPT.json'
    train,score=read(train_path),read(score_path)
    if (train['record']!=rid or train['arm']!=arm or score['record']!=rid or score['arm']!=arm or score['status']!='COMPLETED'
            or train['updates']!=2000 or train['selected_sha256']!=start['selected_sha256']
            or score['checkpoint_sha256']!=train['selected_sha256']or score['config_sha256']!=start['config_sha256']):
        raise PermissionError('Recovery must reuse the selected trained checkpoint and same config')
    hashes={Path(b['path']).name:b['sha256']for b in score['banks']}
    for role in ('CAL','QUERY'):
        relative=f'{rid}/{arm}/score/{role}.npz'
        if hashes[f'{role}.npz']!=new[relative]['sha256']:raise PermissionError('Recovery score receipt bank identity mismatch')
        checked(root/relative,new[relative]['sha256'])
    return {str(p):ev.sha(p)for p in (original_path,original_global,start_path,final_path,rejection,
                                     launch_path,exit_path,cp,train_path,score_path)}


def gate(w,registry,*,model_seal=None):
    root=w/'results/timeradar_smd_full_20261116_v1'
    seal=root/'RAW_BANK_SEAL.json'if model_seal is None else local(w,model_seal)
    raw=read(seal)
    if raw['status']not in ('COMPLETED','COMPLETED_WITH_FAILURES'):
        raise FileNotFoundError('A complete global terminal seal is required before labels')
    new={r['id']for r in registry['records']if r['id']not in OLD}
    expected={f'{r}/{a}/score/{role}.npz'for r in new for a in FULL_METHODS for role in ('CAL','QUERY')}
    recovery_inputs=verify_resource_recovery(w,root,seal,expected)if model_seal is not None else{}
    actual={b['relative']for b in raw['banks']}
    if len(actual)!=len(raw['banks'])or not actual<=expected:raise ValueError('Unexpected/duplicate sealed banks')
    if raw['status']=='COMPLETED'and actual!=expected:raise ValueError('Successful global seal must bind all 200 banks')
    for b in raw['banks']:checked(root/b['relative'],b['sha256'])
    configs={};assets={str(seal):ev.sha(seal),**recovery_inputs}
    for wave in raw['waves']:
        cp=local(w,wave['config_path']);checked(cp,wave['config_sha256']);cfg=read(cp)
        bp=local(w,cfg['data_binding']);checked(bp,cfg['data_binding_sha256']);binding=read(bp)
        if cfg['seed']!=2024 or cfg['score_seed']!=2024 or cfg['select_seed']!=84551:
            raise PermissionError('Fixed training/scoring/SELECT seeds changed')
        for rid in wave['records']:
            if rid in configs or rid not in new:raise ValueError('Wave records overlap or include reused old3')
            configs[rid]=(cp,cfg,{r['id']:r for r in binding['records']}[rid])
        assets.update({str(cp):ev.sha(cp),str(bp):ev.sha(bp)})
    if set(configs)!=new:raise ValueError('Every fixed new entity needs a bound wave config')
    return root,configs,assets


def checked_bank(folder,rid,arm,record,config=None):
    rp=folder/'SCORE_RECEIPT.json';receipt=read(rp)
    if receipt['status']!='COMPLETED'or receipt['record']!=rid or receipt['arm']!=arm:
        raise PermissionError('Completed method/record score identity required')
    if config is not None and receipt['config_sha256']!=ev.sha(config):raise PermissionError('Wrong wave config')
    physical={r['role']:r for r in record['roles']};used={r['role']:r for r in receipt['roles']}
    if any(physical[r]['sha256']!=used[r]['sha256']for r in ('CAL','QUERY')):raise PermissionError('Changed physical roles')
    hashes={Path(r['path']).name:r['sha256']for r in receipt['banks']}
    for name in ('CAL.npz','QUERY.npz'):checked(folder/name,hashes[name])
    cix,cal,_=ev._load_bank(folder/'CAL.npz');ix,score,available=ev._load_bank(folder/'QUERY.npz')
    for role,actual in (('CAL',cix),('QUERY',ix)):
        lo=physical[role]['bounds_half_open'][0]
        expected=(np.arange(lo+512,physical[role]['bounds_half_open'][1])if arm=='history_mean512'else
                  np.arange(lo,lo+physical[role]['scored_points']))
        if not np.array_equal(actual,expected):raise ValueError('Changed native/history score axis')
    return ix,score,available,ev.cal_threshold(cal),rp


def _na(rid,arm,reason):return dict(record=rid,arm=arm,status='MISSING_OR_FAILED',reason=reason,seed=2024,AP=None,ROC_AUC=None)


def run(workspace,output,*,model_seal=None):
    started=time.monotonic();w,output=Path(workspace),Path(output)
    prep=w/'preparations/timeradar_smd_full_20261116_v1'
    registry_path=prep/'full28_registry.json';registry=read(registry_path)
    names=[r['id']for r in registry['records']]
    if len(set(names))!=28 or registry['expected_entities']!=28:raise ValueError('Fixed complete 28-name registry required')
    root,configs,inputs=gate(w,registry,model_seal=model_seal)  # All new model banks sealed before labels.
    binding_path=prep/'data_binding_full28.json';binding={r['id']:r for r in read(binding_path)['records']}
    ep=prep/'evaluation_manifest_full28.json';evalmeta={r['id']:r for r in read(ep)['records']}
    inputs.update({str(p):ev.sha(p)for p in (registry_path,binding_path,ep)})
    oldfull=w/'results/timeradar_cross_variable_fixed_evaluation_20261116_v1/metrics.csv'
    oldcommon=w/'results/timeradar_cross_variable_simple_common_20261116_v1/metrics.csv'
    fullrows=reuse_old_csv(oldfull,FULL_SHA,FULL_METHODS);commonrows=reuse_old_csv(oldcommon,COMMON_SHA,COMMON_METHODS)
    inputs.update({str(oldfull):ev.sha(oldfull),str(oldcommon):ev.sha(oldcommon)})
    output.mkdir(parents=True,exist_ok=False)
    # Old3 rows and their archived event details stay exactly where accepted;
    # references below expose both source tables without recomputing old metrics.
    for rid in names:
        if rid in OLD:continue
        record=binding.get(rid);labels=None
        for arm in COMMON_METHODS:
            folder=(w/'results/timeradar_smd_full_history512_20261116_v1'if arm=='history_mean512'else root)/rid/arm/'score'
            files=[folder/name for name in ('SCORE_RECEIPT.json','CAL.npz','QUERY.npz')]
            if record is None or not all(p.is_file()for p in files):
                row=_na(rid,arm,'MISSING_DATA_OR_COMPLETED_SCORE_BANK')
                commonrows.append(dict(row))
                if arm in FULL_METHODS:fullrows.append(dict(row))
                continue
            cp=configs[rid][0]if arm!='history_mean512'else None
            ix,score,available,tau,rp=checked_bank(folder,rid,arm,record,cp)
            if labels is None:
                spec=evalmeta[rid];lp=w/spec['path'];checked(lp,spec['sha256'])
                with np.load(lp,allow_pickle=False)as z:
                    labels=z['label'].copy()
                    if not np.array_equal(z['index'],np.arange(len(labels))):raise ValueError('Original label axis required')
            provenance={str(p):ev.sha(p)for p in files+[lp,binding_path,ep]}
            if arm in FULL_METHODS:
                result=ev.evaluate_arrays(ix,score,labels,tau,available_index=available)
                ev._save_method(output/'native_full'/rid/arm,result,ix,score,labels,available,provenance)
                row=ev._flat(rid,arm,result);row.update(status='OK',metric_origin='NEW25_FROZEN_NATIVE',seed=2024);fullrows.append(row)
            common=np.arange(512,record['query_points']//100*100);pos=np.searchsorted(ix,common)
            if np.any(pos>=len(ix))or not np.array_equal(ix[pos],common):raise ValueError('Incomplete fixed common axis')
            result=ev.evaluate_arrays(common,score[pos],labels,tau,available_index=available[pos],online=arm=='history_mean512')
            ev._save_method(output/'five_method_common'/rid/arm,result,common,score[pos],labels,available[pos],provenance)
            row=ev._flat(rid,arm,result);row.update(status='OK',metric_origin='NEW25_FIXED_COMMON',seed=2024);commonrows.append(row)
    fullrows.sort(key=lambda r:(r['record'],FULL_METHODS.index(r['arm'])))
    commonrows.sort(key=lambda r:(r['record'],COMMON_METHODS.index(r['arm'])))
    full=summaries(fullrows,names,FULL_METHODS);common=summaries(commonrows,names,COMMON_METHODS)
    ev._write_csv(output/'native_full_28x4.csv',fullrows);ev._write_csv(output/'common_28x5.csv',commonrows)
    for scope,stats in [('native_full',full),('five_method_common',common)]:
        ev._write_csv(output/f'{scope}_macro_median.csv',stats['macro'])
        ev._write_csv(output/f'{scope}_paired_deltas.csv',stats['paired_rows'])
        ev._write_csv(output/f'{scope}_paired_summary.csv',stats['paired_summary'])
    status='COMPLETED'if full['status']==common['status']=='COMPLETED'else'INCOMPLETE'
    ev._dump(output/'RESULT.json',dict(status=status,native_full_rows=fullrows,common_rows=commonrows,
        native_full_statistics=full,five_method_common_statistics=common,CPU_seconds=time.monotonic()-started,
        GPU_hours=0,new_training=False,new_forward=False,old3_metrics_recomputed=False,inputs=inputs,
        source_sha256=ev.sha(__file__),evaluator_sha256=ev.sha(ev.__file__),
        scope='One historical/exposed SMD domain; fixed seed2024; machine heterogeneity is not independent-domain confirmation',
        axes='Native full [0,floor(N/100)*100); five-method common [512,floor(N/100)*100); original boundaries/UNKNOWN retained',
        old3_event_details='Immutable results/timeradar_cross_variable_{fixed_evaluation,simple_common}_20261116_v1; unchanged'))
    return fullrows,commonrows,full,common


def common_indices(banks,*,start=512):
    """Intersect actual score masks/indices only, never labels or model quality."""
    common=None
    for index,score,available in banks:
        if len(index)!=len(score)or len(index)!=len(available)or np.any(np.diff(index)<=0):raise ValueError('Unique aligned source axes required')
        ix=index[index>=start]
        common=ix if common is None else np.intersect1d(common,ix)
    if common is None or not len(common):raise ValueError('No legal shared scored points')
    return common


def _csv_values(path,expected):
    checked(path,expected);result=[]
    with Path(path).open(newline='',encoding='utf8')as handle:
        for row in csv.DictReader(handle):
            parsed={}
            for k,v in row.items():
                if not v:parsed[k]=None;continue
                try:parsed[k]=json.loads(v)
                except json.JSONDecodeError:parsed[k]=v
            result.append(parsed)
    return result


def _left_bank(folder,role,descriptor,record):
    file=folder/f'{role}.npz';checked(file,descriptor['sha256'])
    with np.load(file,allow_pickle=False)as z:
        index,score,valid,available=(z[k].copy()for k in ('index','score','valid','available_index'))
        if valid.dtype!=np.bool_ or valid.shape!=index.shape or not np.array_equal(z['per_channel'].mean(-1),score):
            raise ValueError('LEFT official NumPy channel mean/mask must remain unchanged')
        if score.shape!=index.shape or available.shape!=index.shape or not np.isfinite(score[valid]).all():raise ValueError('Invalid LEFT scored arrays')
    spec=next(r for r in record['roles']if r['role']==role);lo,hi=spec['bounds_half_open']
    if not np.array_equal(index,np.arange(lo,lo+(hi-lo)//192*192)):raise ValueError('LEFT native192 original role axis changed')
    return index[valid],score[valid],available[valid]


def compare_left(workspace,output):
    """Four frozen matching-data cases; independently gated from all-SMD jobs."""
    w,output=Path(workspace),Path(output);started=time.monotonic()
    left=w/'results/left_same_permission_20261116_v1';seal=left/'RAW_SCORE_GLOBAL_SEAL.json';sealed=read(seal)
    records=('PSM',)+OLD
    if sealed['status']!='SEALED'or sealed['labels_read']is not False or {r['record']for r in sealed['cases']}!=set(records)or len(sealed['cases'])!=4:
        raise FileNotFoundError('All four LEFT cases must seal before label access')
    config=w/'hybrid_ad/configs/left-same-permission-20261116.json';checked(config,sealed['config_sha256']);cfg=read(config)
    bp=local(w,cfg['data_binding']);checked(bp,cfg['data_binding_sha256']);binding={r['id']:r for r in read(bp)['records']}
    cases={r['record']:r for r in sealed['cases']}
    for case in sealed['cases']:
        checked(local(w,case['score_receipt_path']),case['score_receipt_sha256'])
        checked(local(w,case['train_receipt_path']),case['train_receipt_sha256'])
        for bank in case['banks']:checked(local(w,bank['path']),bank['sha256'])
    # All model inputs now sealed; this fixed four-case external comparison does
    # not wait for or open the unrelated new25 TimeRadar queries.
    oldmodel=w/'results/timeradar_cross_variable_20261116_v1'
    modelseal=oldmodel/'RAW_BANK_SEAL.json';checked(modelseal,'e073be13716f3b11e3121043c9c74086a20f560b6f5519cf619b9b3d8e91f146')
    oldjobs=read(modelseal)['jobs'];leftjobs=read(left/'QUEUE_RECEIPT.json')['jobs']
    oldcommon=w/'results/timeradar_cross_variable_simple_common_20261116_v1/metrics.csv'
    oldrows=_csv_values(oldcommon,COMMON_SHA)
    oldexternal=w/'results/timeradar_cross_variable_external_20261116_v1/external_main_comparison.csv'
    psmrows=_csv_values(oldexternal,'9e38b2640dda32e71618f41fac383b4f40f900f18b3388b370f4a3677ad30a91')
    ep=w/'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json'
    evalmeta={r['id']:r for r in read(ep)['records']}
    output.mkdir(parents=True,exist_ok=False);rows=[];axes=[]
    for rid in records:
        record=binding[rid];folder=left/rid/'score';case=cases[rid]
        receipt=read(folder/'RECEIPT.json')
        if receipt['status']!='COMPLETED'or receipt['record']!=rid:raise PermissionError('LEFT receipt identity mismatch')
        physical={r['role']:r for r in record['roles']};used={r['role']:r for r in receipt['roles']}
        if any(physical[r]['sha256']!=used[r]['sha256']for r in ('CAL','QUERY')):raise PermissionError('LEFT used different physical roles')
        descriptors={r['role']:r for r in case['banks']}
        _,cal,_=_left_bank(folder,'CAL',descriptors['CAL'],record)
        lbank=_left_bank(folder,'QUERY',descriptors['QUERY'],record);tau=ev.cal_threshold(cal)
        calibration=folder/'CALIBRATION.json'
        if read(calibration)['threshold']!=tau or read(calibration)['comparison']!='>':raise ValueError('Original LEFT own CAL changed')
        banks={'LEFT_native_same_roles':(lbank,tau,folder/'RECEIPT.json')}
        for arm in FULL_METHODS:
            mf=oldmodel/rid/arm/'score';ix,s,a,t,rp=checked_bank(mf,rid,arm,record)
            banks[arm]=((ix,s,a),t,rp)
        hf=w/'results/timeradar_cross_variable_history512_20261116_v1'/rid/'history_mean512'/'score'
        ix,s,a,t,rp=checked_bank(hf,rid,'history_mean512',record);banks['history_mean512']=((ix,s,a),t,rp)
        common=common_indices([b[0]for b in banks.values()])
        axes.append(dict(record=rid,points=len(common),start=int(common[0]),end_exclusive=int(common[-1])+1,
            contiguous=bool(np.all(np.diff(common)==1)),index_rule='intersection of actual six score axes, index>=512; no label filtering'))
        lp=w/evalmeta[rid]['path'];checked(lp,evalmeta[rid]['sha256'])
        with np.load(lp,allow_pickle=False)as z:
            labels=z['label'].copy()
            if not np.array_equal(z['index'],np.arange(len(labels))):raise ValueError('Complete original label axis required')
        for method,((ix,s,a),threshold,rp)in banks.items():
            candidates=(psmrows if rid=='PSM'else oldrows)
            old_name='HistoryMean512_strict_targetCAL'if rid=='PSM'and method=='history_mean512'else method
            accepted=next((r for r in candidates if r['arm']==old_name and(rid=='PSM'or r['record']==rid)),None)
            previous_axis=(np.arange(512,87744)if rid=='PSM'else np.arange(512,record['query_points']//100*100))
            if method!='LEFT_native_same_roles'and accepted is not None and np.array_equal(common,previous_axis):
                row=dict(accepted,record=rid,arm=method,metric_origin='ACCEPTED_EXACT_COMMON_AXIS_ROW')
                if row['threshold']!=threshold:raise ValueError('Reuse requires unchanged actual CAL threshold')
            else:
                pos=np.searchsorted(ix,common)
                if not np.array_equal(ix[pos],common):raise ValueError('Score common-axis alignment mismatch')
                result=ev.evaluate_arrays(common,s[pos],labels,threshold,available_index=a[pos],online=method=='history_mean512')
                sources={str(p):ev.sha(p)for p in (rp,lp,bp,seal,modelseal,ep)}
                ev._save_method(output/rid/method,result,common,s[pos],labels,a[pos],sources)
                row=ev._flat(rid,method,result);row['metric_origin']='FROZEN_SCORES_NEW_COMMON_AXIS_ONLY'
            row.update(seed=2024,common_start=int(common[0]),common_end_exclusive=int(common[-1])+1)
            jobs=(leftjobs if method=='LEFT_native_same_roles'else oldjobs if method!='history_mean512'else [])
            relevant=[j for j in jobs if j['record']==rid and(method=='LEFT_native_same_roles'or j['arm']==method)]
            row['train_GPU_hours']=sum(j.get('gpu_hours',j.get('GPU_hours',0.))for j in relevant if j['phase']=='train')
            row['score_GPU_hours']=sum(j.get('gpu_hours',j.get('GPU_hours',0.))for j in relevant if j['phase']=='score')
            row['training_information']=('native LEFT random initialization; target FIT all-parameter native20epoch/patience3; normal SELECT'if method=='LEFT_native_same_roles'else
                'official source-pretrained TimeRadar; no target parameter update'if method=='zero'else
                'fixed untrained past512 statistic'if method=='history_mean512'else
                'official source-pretrained frozen backbone; target FIT 2000 reconstruction updates; normal SELECT')
            row['data_CAL_information']='Same physical FIT-only scaler/isolated roles; own complete-footprint CAL q99higher, strict>; QUERY only common actual points'
            rows.append(row)
    ev._write_csv(output/'matched_external_4x6.csv',rows)
    ev._dump(output/'RESULT.json',dict(status='COMPLETED',rows=rows,common_axes=axes,CPU_seconds=time.monotonic()-started,
        GPU_hours=0,new_training=False,new_forward=False,source_sha256=ev.sha(__file__),
        inputs={str(p):ev.sha(p)for p in (seal,modelseal,config,bp,oldcommon,oldexternal,ep)},
        warnings=['Same data permissions are not equal initialization or optimization budgets',
                  'LEFT192 and TimeRadar100 scores remain offline; offsets are not online delays',
                  'Historical different-permission LEFT_R and CATCH results remain separate and are not reused as matched rows']))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--left-comparison',action='store_true')
    p.add_argument('--model-seal',type=Path)
    a=p.parse_args()
    if a.left_comparison:compare_left(a.workspace,a.output)
    else:run(a.workspace,a.output,model_seal=a.model_seal)
