"""Matched U-head/U-full endpoint scope comparison; no neural computation.

Two complete canonical banks seal before labels. Z-T is c00 residuals with
the existing target CAL512, not Z-S and not the earlier partial pilot result.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .pretrained_psm_comparison import bound, read, write, sha, archive_scalars
from .source_discrimination_psm_comparison import (
    exact_fields, import_bound, crop, table, corrected_operating_columns, fixed_old_calibration_scores, reuse_operating)

SOURCE='c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812'
ZT_CAL='b23dfc24d251136d5d916c3f585d43dcd2b0e8a6eb19493ce6bb78338e40e880'
SCOPES=('HEAD','FULL')
IDENTITIES={'input_sha256','config_sha256','checkpoint_sha256','calibration_sha256','source_sha256'}
FULL_RANGE=[dict(domain='PSM',entity='anonymous_PSM',run='test_original',start=512,end=87841)]


def target_calibration(calibration):
    exact_fields(calibration,dict(valid_windows=512,source_domain_equal_weight=False,domains=['PSM'],
                                 weights=dict(r_tf=.25,r_ms=.25,r_pred=.5)))


def admit(config):
    if config.get('status')!='FROZEN_READY' or config.get('seed')!=2025 or config.get('common_axis')!=[512,87744]:
        raise PermissionError('Only the frozen full-query target-scope experiment')
    for p,h in config['source_sha256'].items():
        if sha(p)!=h:raise PermissionError('Changed evaluation dependency: '+p)
    science=read(config['science_freeze'])
    if not config['science_expected_fields']:raise PermissionError('Actual root science binding required')
    exact_fields(science,config['science_expected_fields'])
    plan=read(config['common_plan'])
    exact_fields(plan,dict(history=512,horizon=32,seed=2025))
    if any(len(plan[role])!=count for role,count in (('FIT',2000),('SELECT',32),('CAL',512))):
        raise ValueError('Shared fixed FIT2000 / SELECT32 / CAL512 required')
    if [d['scope'] for d in config['adapted_methods']]!=list(SCOPES):
        raise ValueError('Both prespecified scopes required, not the better endpoint')
    if config['ZT_calibration']['sha256']!=ZT_CAL:raise ValueError('Existing Z-T calibration required')
    target_calibration(read(config['ZT_calibration']));bound(config['ZT_lineage_reference'])
    if not config['resource_receipts'] or not config['required_evidence']:
        raise PermissionError('Bind actual execution and mode/RNG/scope evidence')
    for a in config['resource_receipts']:bound(a)
    for e in config['required_evidence']:exact_fields(read(e['asset']),e['expected_fields'])
    gates=[]
    for d in config['adapted_methods']:
        receipts=[read(d[k]) for k in ('train_receipt','cal_receipt','query_receipt')]
        train,cal,query=receipts
        for r,phase in zip(receipts,('train','cal','query')):
            exact_fields(r,dict(status='COMPLETED',phase=phase,seed=2025,labels_read=False))
            exact_fields(r,d['expected_shared_receipt_fields'])
        exact_fields(train,dict(attempts=2000,updates=2000,scientific_training_status='VALID_ALL_ATTEMPTS_UPDATED'))
        exact_fields(train,dict(scope=d['scope'],source_checkpoint_sha256=SOURCE,
            common_plan_sha256=config['common_plan']['sha256'],selection='FIXED_ATTEMPT_ENDPOINT_NO_BEST_CHECKPOINT',
            mode_policy='root_train_allowed_children_train_others_eval'))
        exact_fields(train,d['expected_train_scope_fields'])
        checkpoint=d['checkpoint']['sha256'];bound(d['checkpoint'])
        if any(r['checkpoint_sha256']!=checkpoint for r in receipts):raise ValueError('Mixed endpoint lineage')
        if d['common_plan']['sha256']!=config['common_plan']['sha256']:raise ValueError('Both scopes share FIT/SELECT/CAL')
        bound(d['common_plan']);bound(d['train_config']);bound(d['training_report']);bound(d['cal_records'])
        exact_fields(train,dict(config_sha256=d['train_config']['sha256']))
        calibration=read(d['calibration']);target_calibration(calibration)
        exact_fields(cal,dict(records=512,calibration_sha256=d['calibration']['sha256']))
        exact_fields(query,dict(points=87329,calibration_sha256=d['calibration']['sha256'],bank_manifest_sha256=d['manifest']['sha256']))
        exact_fields(query,dict(scientific_status='COMPLETE_PENDING_FIXED_EVALUATION'))
        manifest=read(d['manifest']);exact_fields(manifest,dict(complete=True,committed_records=87329))
        exact_fields(manifest['contract'],dict(ranges=FULL_RANGE,threshold=calibration['threshold']))
        if set(d['expected_identity'])!=IDENTITIES:raise ValueError('All five producer identities required')
        exact_fields(manifest['contract']['identity'],d['expected_identity'])
        exact_fields(manifest['contract']['identity'],dict(checkpoint_sha256=checkpoint,calibration_sha256=d['calibration']['sha256']))
        gates.append(dict(scope=d['scope'],checkpoint_sha256=checkpoint,manifest_sha256=d['manifest']['sha256'],
                          calibration_sha256=d['calibration']['sha256']))
    seal=read(config['whole_raw_seal'])
    exact_fields(seal,dict(labels_read=False))
    if seal['methods']!=gates:raise ValueError('Whole raw seal must bind both actual endpoints')
    return gates


def prepare_zt(cache,calibration):
    """Original source residuals, exactly the fixed target CAL; no query fitting."""
    index=np.asarray(cache['index'])
    if not np.array_equal(index,np.arange(512,87841)) or not np.asarray(cache['valid']).all():
        raise ValueError('Complete original source87329 scalar cache required')
    target_calibration(calibration)
    score=fixed_old_calibration_scores({k:cache[k] for k in ('r_tf','r_ms','r_pred')},calibration)
    return dict(index=index.copy(),score=score,valid=np.ones(len(index),bool),available_index=index.copy())


def reuse_full_mean(descriptor,index,score,available,labels):
    """Reuse already accepted original full-axis Mean512 metrics, not common crop."""
    if not np.array_equal(index,np.arange(512,87841)):
        raise ValueError('Full87329 Mean cannot be substituted by the common crop or pilot')
    previous=descriptor['accepted_full_evaluation'];result=read(previous['result']);provenance=read(previous['provenance'])
    if provenance['calculation_inputs_sha256']!=previous['inputs']['sha256']:
        raise ValueError('Prior full Mean input provenance changed')
    with np.load(bound(previous['inputs']),allow_pickle=False) as old:
        for key,value in dict(index=index,score=score,available_index=available,full_labels=labels).items():
            if not np.array_equal(old[key],value):raise ValueError('Prior full Mean must have exact same original inputs')
    gate=next(v for v in provenance['provenance']['gates'] if v['id']==previous['original_gate_id'])
    if gate['threshold']!=descriptor['threshold'] or gate['records']!=87329:
        raise ValueError('Prior full Mean threshold/coverage differs')
    return result


def evaluate_online(config,folder,data,index,labels,threshold,descriptor,formatter):
    from . import evaluation
    from .public_native_evaluation import event_diagnostics
    from .controlled_wave_evaluation import reject_dependency_na
    where=crop(data,index);score=data['score'][where];available=data.get('available_index',data.get('available'))[where]
    result=evaluation.evaluate_arrays(index=index,score=score,valid=np.ones(len(index),bool),available_index=available,
        invalid_reasons=[None]*len(index),full_labels=labels,threshold=threshold,bank_complete=True,
        original_length=87841,warmup_points=512,affiliation_identity=bound(config['affiliation_identity']))
    reject_dependency_na(result)
    evaluation.save_evaluation(folder,result,index=index,score=score,valid=np.ones(len(index),bool),available_index=available,
        invalid_reasons=[None]*len(index),full_labels=labels,provenance=dict(config=config,method=descriptor['id']))
    phases=event_diagnostics(result)
    return corrected_operating_columns(formatter,result,phases),phases


def run(config):
    began=time.monotonic();gates=admit(config)
    from sklearn.metrics import average_precision_score,roc_auc_score
    from .controlled_wave_evaluation import require_metric_dependencies
    from .public_native_evaluation import event_diagnostics
    require_metric_dependencies()
    loader=import_bound(config['baseline_loader'],'accepted_psm_scope_loader')
    formatter=import_bound(config['operating_formatter'],'accepted_psm_scope_formatter').operating_columns
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    archives=[];arrays={}
    for d in config['adapted_methods']:
        a=archive_scalars(d,out/(d['id']+'_canonical_scalars.npz'));archives.append(dict(scope=d['scope'],raw=a))
        with np.load(a['path'],allow_pickle=False) as raw:
            arrays[d['id']]={k:raw[k].copy() for k in ('index','score','valid','available_index')}
    with np.load(bound(config['source_scalar_cache']),allow_pickle=False) as raw:
        zt=prepare_zt({k:raw[k].copy() for k in ('index','valid','r_tf','r_ms','r_pred')},read(config['ZT_calibration']))
    with (out/'ZT_original_source_with_target_CAL512.npz').open('xb') as h:np.savez_compressed(h,**zt)
    write(out/'ALL_RAW_SEALED_BEFORE_LABELS.json',dict(labels_read=False,gates=gates,archives=archives,
        ZT=dict(path=str(out/'ZT_original_source_with_target_CAL512.npz'),sha256=sha(out/'ZT_original_source_with_target_CAL512.npz'))))
    # Numerical labels are opened only after both new banks and fixed Z-T scores seal.
    with np.load(bound(config['truth']),allow_pickle=False) as raw:labels=raw['full_labels'].copy()
    if labels.shape!=(87841,):raise ValueError('Keep original native full truth axis')
    arrays['Z_T_source2025']=zt
    descriptors=[dict(id='Z_T_source2025',online=True,threshold=read(config['ZT_calibration'])['threshold'],
        protocol='c00_h1_R3_original_targetCAL512_no_target_updates',role_permissions='target CAL statistics; no target FIT updates')]
    descriptors += [dict(id=d['id'],online=True,threshold=read(d['calibration'])['threshold'],
        protocol='U_'+d['scope']+'_fixed_endpoint2000_h1_R3_own_targetCAL512',
        role_permissions='target FIT2000 updates and normal SELECT diagnostic only; target CAL512') for d in config['adapted_methods']]
    descriptors += config['external_methods']
    if [d['id'] for d in config['external_methods']]!=['HistoryMean512_ZS','TimeRadar_official_native','LEFT_R_targettrained'] or len(descriptors)!=6:
        raise ValueError('Exactly six frozen main rows')
    rows=[];phases=[];fullrows=[];fullphases=[];common=np.arange(512,87744)
    for d in descriptors:
        name=d['id'];data=arrays.get(name)
        if data is None:data=loader.load_method(d,labels)
        where=crop(data,common);score=data['score'][where];available=data.get('available_index',data.get('available'))[where]
        row=dict(method=name,points=len(common),positive_points=int(labels[common].sum()),AP=float(average_precision_score(labels[common],score)),
            ROC_AUC=float(roc_auc_score(labels[common],score)),online=d['online'],threshold=d.get('threshold'),protocol=d['protocol'],role_permissions=d['role_permissions'])
        folder=out/name;folder.mkdir()
        with (folder/'common_axis_inputs.npz').open('xb') as h:
            np.savez_compressed(h,index=common,score=score,valid=np.ones(len(common),bool),available_index=available,full_labels=labels)
        if d['online']:
            if (available>common).any():raise PermissionError('No offline-to-online conversion')
            if name=='HistoryMean512_ZS':
                result=reuse_operating(d,common,score,available,labels);p=event_diagnostics(result)
                operating=corrected_operating_columns(formatter,result,p)
                write(folder/'REUSED_ACCEPTED_OPERATING_RESULT.json',d['accepted_common_evaluation'])
            else:operating,p=evaluate_online(config,folder/'fixed_target_tau_common',data,common,labels,d['threshold'],d,formatter)
            row.update(operating);phases.extend(dict(method=name,**v) for v in p)
        else:row['online_NA_reason']='native retrospective windows/batch; target statistics/training differ; AP same axis does not match permissions'
        rows.append(row)
        if name in {a['id'] for a in config['adapted_methods']} | {'Z_T_source2025','HistoryMean512_ZS'}:
            axis=np.arange(512,87841);where=crop(data,axis);s=data['score'][where]
            r=dict(method=name,points=len(axis),positive_points=int(labels[axis].sum()),AP=float(average_precision_score(labels[axis],s)),
                ROC_AUC=float(roc_auc_score(labels[axis],s)),online=True,threshold=d['threshold'],protocol=d['protocol'],role_permissions=d['role_permissions'])
            if name=='HistoryMean512_ZS':
                result=reuse_full_mean(d,axis,s,data.get('available_index',data.get('available'))[where],labels)
                p=event_diagnostics(result);op=corrected_operating_columns(formatter,result,p)
                write(folder/'REUSED_ACCEPTED_FULL_OPERATING_RESULT.json',d['accepted_full_evaluation'])
            else:op,p=evaluate_online(config,folder/'fixed_target_tau_full87329',data,axis,labels,d['threshold'],d,formatter)
            r.update(op);fullrows.append(r);fullphases.extend(dict(method=name,**v) for v in p)
    lookup={r['method']:r for r in rows};head,full=[lookup[d['id']] for d in config['adapted_methods']]
    delta={k:float(full[k]-v) for k,v in head.items() if k not in ('points','positive_points','threshold','online')
        and isinstance(v,(int,float)) and not isinstance(v,bool) and isinstance(full.get(k),(int,float))}
    table(out/'six_methods_common_axis.csv',rows);table(out/'four_causal_methods_full87329.csv',fullrows)
    table(out/'full_minus_head.csv',[delta]);table(out/'common_event_phases.csv',phases);table(out/'full_event_phases.csv',fullphases)
    write(out/'RESULT.json',dict(status='COMPLETED_TARGET_UPDATE_SCOPE_PSM',seconds=time.monotonic()-began,common_axis=[512,87744],
        rows=rows,full87329=fullrows,full_minus_head=delta,labels_after_both_raw_sealed=True,source_config=config,
        new_forward=False,new_training=False,GPUh=0,limits=config['limits']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Frozen configuration hash required')
    run(json.loads(Path(a.config).read_text(encoding='utf-8')))


if __name__=='__main__':main()
