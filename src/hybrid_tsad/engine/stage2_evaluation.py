"""Independent evaluation runtime. Never executes or relaxes the readout gate."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

from ..config import read_json,resolve_path,sha256
from .identity import SCORE_SPACE,operation_manifest,source_calibration
from .stage2_identity import Assets,canonical_sha
from .readout_manifest import record,entity_profile,json_write
from .evaluation import evaluate_fixed_scores,validate_coverage
from .events import binary_events,greedy_assign,real_blocks
from .official_affiliation import load_official_affiliation
from .pipeline import _json_safe
from .stage2_run_statistics import NATIVE,LEGACY,MODES,run_statistics,delay_summary,normal_prefix_summary,fixed_grid

STAGE='U_NORMAL_HEAD_EVALUATION'
READOUT_STAGE='U_NORMAL_HEAD_READOUT'
CONTRACT_SHA='71fde43d68ec20f70f8bf5218e35d94ffae87332d59674648dd1bf77e0df843c'
INPUTS={'seal','gate_acceptance','gate_summary','gate_config','gate_os_wait','gate_resource','readout_runtime','readout_copy_manifest','query_manifest','affiliation','runtime'}


def evaluation_limit_seconds(profile):
    """Pre-label resource caps only; never selected from scores or throughput."""
    limits={'SYNTHETIC':300,'TEP100':900}
    if profile.get('name') not in limits:raise ValueError('Unknown evaluation resource profile')
    return limits[profile['name']]


def runtime_identity():
    root=Path(__file__).resolve().parents[1]
    return dict(stage=STAGE,files={'src/hybrid_tsad/'+p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*.py'))})


def load_operation(path):
    c=read_json(path)
    if (set(c)!={'engine_schema_version','stage','operation','contract','approval','inputs','parameters','output','score_space'}
            or c['engine_schema_version']!=7 or c['stage']!=STAGE or c['operation']!='evaluate' or c['score_space']!=SCORE_SPACE
            or set(c['inputs'])!=INPUTS or set(c['parameters'])!={'profile','readout_runtime_root'}):raise ValueError('Independent schema7 evaluation inputs required; no fit/device/metric selection')
    for v in [c['contract'],c['approval'],*c['inputs'].values()]:
        if set(v)!={'path','sha256'}:raise ValueError('Exact path/SHA records required')
    paths={k:resolve_path(v['path'],path) for k,v in {'contract':c['contract'],'approval':c['approval'],**c['inputs']}.items()};out=resolve_path(c['output'],path)
    manifest,_=operation_manifest(c,paths,out);manifest.update(stage=STAGE,permissions='evaluation only; no inference/training/calibration/F')
    return c,paths,out,manifest,canonical_sha(manifest)


def authorize(c,paths,out,digest,base):
    """Only hashes/JSON/source bytes before permission; zero numerical array access."""
    a=Assets()
    for k,ref in {'contract':c['contract'],'approval':c['approval'],**c['inputs']}.items():a.file({'path':str(paths[k]),'sha256':ref['sha256']},base)
    if c['contract']['sha256']!=CONTRACT_SHA or out.exists():raise ValueError('Wrong evaluation contract or reused output')
    approval=read_json(paths['approval']);profile=c['parameters']['profile'];synthetic=profile.get('name')=='SYNTHETIC'
    permission=dict(labels=True,normal_prefix_values=True,evaluate=True,inference=False,training=False,calibration=False,F=False)
    if (approval.get('status')!='APPROVED' or approval.get('stage')!=STAGE or approval.get('operation_manifest_sha256')!=digest
            or approval.get('contract_sha256')!=CONTRACT_SHA or approval.get('permissions')!=permission
            or any(type(v) is not bool for v in approval.get('permissions',{}).values())):raise ValueError('Exact independent evaluation/label release absent')
    if approval.get('scope')!=('SYNTHETIC_CPU_ENGINEERING_ONLY' if synthetic else 'U_NORMAL_HEAD_EVALUATION_EXPLICIT_LABEL_RELEASE'):
        raise ValueError('Synthetic release cannot become production evaluation')
    fixed_grid(profile)
    if read_json(paths['runtime'])!=runtime_identity():raise ValueError('Independent evaluation runtime changed')
    frozen=read_json(paths['readout_runtime']);root=Path(c['parameters']['readout_runtime_root'])
    files={'src/hybrid_tsad/'+p.relative_to(root/'src/hybrid_tsad').as_posix():sha256(p) for p in (root/'src/hybrid_tsad').rglob('*.py')}
    if a.file(c['inputs']['readout_copy_manifest'],base)!=(root/'COPY_MANIFEST.json').resolve():raise ValueError('Frozen COPY manifest root mismatch')
    copy=read_json(paths['readout_copy_manifest'])
    if (len(files)!=32 or frozen!=dict(stage=READOUT_STAGE,files=files) or copy.get('identity')!=frozen
            or copy.get('python_files')!=32 or 'src/hybrid_tsad/engine/stage2_readout.py' not in files):raise ValueError('Original frozen32 readout snapshot changed; never import new dynamic gate')
    for n,h in files.items():a.file({'path':str(root/n),'sha256':h},base)
    for n,h in copy.get('extra_files',{}).items():a.file({'path':str(root/n),'sha256':h},base)
    current_root=Path(__file__).resolve().parents[1]
    for name in ('engine/evaluation.py','engine/events.py','engine/statistics.py','engine/official_affiliation.py','protocol.py'):
        if sha256(current_root/name)!=files['src/hybrid_tsad/'+name]:raise ValueError('Frozen metric/event/score kernels changed across runtimes')
    gate_accept=read_json(paths['gate_acceptance']);gate=read_json(paths['gate_summary']);gc=read_json(paths['gate_config'])
    oswait=read_json(paths['gate_os_wait']);resource=read_json(paths['gate_resource']);seal=read_json(paths['seal'],max_bytes=64*1024*1024)
    expected_status='SYNTHETIC_GATE_ACCEPTANCE_ONLY' if synthetic else 'PRIMARY_ACCEPTED_U_READOUT_GATE'
    if gate_accept.get('status')!=expected_status or gate_accept.get('stage')!=READOUT_STAGE:raise ValueError('Actual frozen gate independent/primary acceptance required')
    for key in ('seal','gate_summary','gate_config','gate_os_wait','gate_resource','readout_runtime','readout_copy_manifest'):
        if gate_accept.get(key)!=c['inputs'][key]:raise ValueError('Gate acceptance does not bind this exact input chain')
    review=read_json(a.file(gate_accept['independent_review'],paths['gate_acceptance']))
    if synthetic:
        if review.get('status')!='SYNTHETIC_REVIEW_ONLY' or resource.get('synthetic_fixture') is not True:raise ValueError('CPU gate fixture marker required')
    elif not str(review.get('status','')).startswith('PASS') or resource.get('synthetic_fixture') is not False:
        raise ValueError('Fixture PASS is not a production gate review')
    if (gc.get('engine_schema_version')!=6 or gc.get('stage')!=READOUT_STAGE or gc.get('operation')!='gate'
            or gc['inputs'].get('seal')!=c['inputs']['seal'] or gc['parameters'].get('runtime')!=c['inputs']['readout_runtime']
            or gc['parameters'].get('profile')!=profile or gc['inputs'].get('catalog')!=seal.get('catalog')):raise ValueError('Frozen actual gate config/seal/runtime mismatch')
    gp={k:resolve_path(v['path'],paths['gate_config']) for k,v in {'contract':gc['contract'],'approval':gc['approval'],**gc['inputs']}.items()}
    gm,_=operation_manifest(gc,gp,resolve_path(gc['output'],paths['gate_config']));gm.update(stage=READOUT_STAGE,permissions='no training/source-fit/labels/F; explicit operation only')
    ga=read_json(a.file(gc['approval'],paths['gate_config']))
    gate_permission=dict(operation='gate',training=False,source_fit=False,query_infer=False,target_normal_cal=False,labels=False,F=False)
    if (ga.get('status')!='APPROVED' or ga.get('stage')!=READOUT_STAGE or ga.get('operation_manifest_sha256')!=canonical_sha(gm)
            or ga.get('contract_sha256')!=gc['contract']['sha256'] or ga.get('permissions')!=gate_permission):raise ValueError('Original gate operation was not explicitly approved')
    if (gate.get('status')!='COMPLETE_SEAL_ELIGIBLE_FOR_SEPARATE_PRIMARY_LABEL_RELEASE' or gate.get('labels_permission') is not False
            or gate.get('seal')!=c['inputs']['seal'] or gate.get('counts')!=seal.get('counts')
            or resolve_path(gc['output'],paths['gate_config'])!=paths['gate_summary'].parent):raise ValueError('No complete actual frozen gate summary')
    if (oswait.get('actual_OS_wait_exit_code')!=0 or not oswait.get('process_seconds',0)>0
            or resource.get('status')!='COMPLETED_PENDING_INDEPENDENT_REVIEW' or resource.get('summary_sha256')!=c['inputs']['gate_summary']['sha256']
            or resource.get('current_runtime',{}).get('shared_sha256')!=files['src/hybrid_tsad/engine/stage2_readout.py']
            or resource.get('current_runtime',{}).get('runtime_identity_sha256')!=c['inputs']['readout_runtime']['sha256']):raise ValueError('Actual gate OS/runtime/summary receipts missing')
    if (seal.get('status')!='ALL_U_CANONICAL_SCORES_SEALED_LABELS_LOCKED' or seal.get('stage')!=READOUT_STAGE
            or seal.get('complete_group') is not True or seal.get('labels_opened') is not False):raise ValueError('Incomplete seal cannot release labels')
    query=read_json(paths['query_manifest']);entities=entity_profile(query,profile)
    if any(q.get('event_start_0based')!=profile['event_start'] or q.get('event_end_exclusive')!=profile['points'] for q in entities.values()):raise ValueError('Frozen original event bounds changed')
    if gate_accept.get('query_manifest')!=c['inputs']['query_manifest']:raise ValueError('Gate acceptance query identity mismatch')
    return a,seal,entities,synthetic


def registry(a,seal,entities,base,query_record):
    """Rehash all frozen objects/threshold refs before the first label is opened."""
    catalog_path=a.file(seal['catalog'],base);catalog=read_json(catalog_path);definition_path=a.file(catalog['definition'],catalog_path);definition=read_json(definition_path)
    if definition['query_manifest']!=query_record:raise ValueError('Accepted catalog/query label population mismatch')
    rows=definition['u_models'];jobs={f'u_normal_head_{m}_s{s}_{i}':(s,m,i) for m in MODES for s in range(3) for i in ('source_pretrained','random_init')}
    models={r['job_id']:r for r in rows}
    if len(rows)!=18 or set(models)!=set(jobs):raise ValueError('Exact18 fitted models required')
    expected={READOUT_STAGE+'/'+j+'/'+e for j in jobs for e in entities}
    if set(seal['u_banks'])!=expected:raise ValueError('Missing/extra U bank before labels')
    result=[]
    for key,v in seal['u_banks'].items():
        job=v['job_id'];s,m,i=jobs[job];entity=v['entity'];row=models[job]
        if key!=READOUT_STAGE+'/'+job+'/'+entity or v['canonical']!={'file':v['bank'],'score_field':'standardized_score','index_field':'time_index'}:raise ValueError('U canonical identity mismatch')
        if set(v['tracks'])!={'source','target'} or any(t['score_object']!=v['canonical'] for t in v['tracks'].values()):raise ValueError('S/T must use ONE canonical file/field/index/SHA')
        for part in ('bank','rng','metadata'):a.file(v[part],base)
        for part in ('source_calibration','target_threshold','checkpoint_metadata','training_summary','train_archive','cal_archive','primary_acceptance'):a.file(row[part],definition_path)
        source=read_json(a.file(row['source_calibration'],definition_path));target=read_json(a.file(row['target_threshold'],definition_path))
        if (source.get('role')!='source_cal_after_adaptation' or target.get('role')!='target_threshold_only'
                or source.get('fitted_checkpoint_sha256')!=row['checkpoint']['sha256'] or target.get('fitted_checkpoint_sha256')!=row['checkpoint']['sha256']
                or v['tracks']['source']['threshold']!=source['threshold99'] or v['tracks']['target']['threshold']!=target['threshold99']):raise ValueError('U threshold/calibration/fitted identity mismatch')
        result.append(dict(stage='U_NORMAL_HEAD',initialization=i,method=m,seed=s,entity=entity,checkpoint_sha256=row['checkpoint']['sha256'],object=v['canonical'],thresholds={k:t['threshold'] for k,t in v['tracks'].items()}))
    old_path=a.file(seal['z_seal'],base);old=read_json(old_path,max_bytes=32*1024*1024)
    if old['query_manifest']!=query_record:raise ValueError('Old Z query population differs')
    zc_path=a.file(definition['z_catalog'],definition_path);zc=read_json(zc_path);zmodels={(m['seed'],m['mode']):m for m in zc['models']};controls={(c['seed'],c['method']):c for c in zc['controls']}
    threshold_path=a.file(seal['z_thresholds'],base);zthreshold=read_json(threshold_path)
    zexpected={f's{s}/{m}/{e}' for s in range(3) for m in MODES+('late','history_mean','persistence') for e in entities}
    if set(old['scores'])!=zexpected or set(old['native_banks'])!={f's{s}/{m}/{e}' for s in range(3) for m in MODES for e in entities}:raise ValueError('Old Z reference set incomplete')
    if set(zthreshold['thresholds'])!={f's{s}/{m}' for s in range(3) for m in MODES}:raise ValueError('All9 Z target thresholds required')
    for key,v in old['scores'].items():
        s,m,e=v['seed'],v['method'],v['entity']
        if key!=f's{s}/{m}/{e}' or v['threshold']!=controls[s,m]['threshold']:raise ValueError('Old Z score/source threshold changed')
        a.file(v['file'],old_path);thresholds={'source':v['threshold']};checkpoint='NO_CHECKPOINT_FIXED_CONTROL'
        if m in MODES:
            model=zmodels[s,m];checkpoint=model['checkpoint']['sha256'];cal=source_calibration(a.file(model['calibration'],zc_path))
            t=read_json(a.file(zthreshold['thresholds'][f's{s}/{m}'],threshold_path))
            if (t.get('role')!='Z_TARGET_NORMAL_THRESHOLD_ONLY' or t['checkpoint']!=model['checkpoint'] or t['source_calibration']!=model['calibration']
                    or t['source_error_scale']!=cal['normal_error_scale'] or t['target_global_rescale'] is not False):raise ValueError('Z target/source role/units mismatch')
            thresholds['target']=t['threshold99']
        result.append(dict(stage='Z',initialization='source_pretrained',method=m,seed=s,entity=e,checkpoint_sha256=checkpoint,
            object=dict(file=v['file'],score_field='score',index_field='time_index'),thresholds=thresholds))
    for q in entities.values():
        a.file({'path':q['label_path'],'sha256':q['label_sha256_from_frozen_parent']},base)
        a.file({'path':q['array_path'],'sha256':q['array_sha256_from_frozen_parent']},base)
    return result


def threshold_only(index, labels, score, threshold, coverage, *, affiliation=None, rank):

    index, y, s = np.asarray(index), np.asarray(labels), np.asarray(score)
    if index.ndim != 1 or y.shape != index.shape or s.shape != index.shape:
        raise ValueError("Index, labels and score must be equal one-dimensional arrays")
    if not len(index) or not np.issubdtype(index.dtype, np.integer) or np.any(np.diff(index) <= 0) or not np.isfinite(s).all():
        raise ValueError("Need finite scores and strictly increasing original time indices")
    if not np.array_equal(np.unique(y), [0, 1]) or not np.isfinite(threshold):
        raise ValueError("Fixed evaluation requires both binary classes and a finite source threshold")
    coverage = validate_coverage(index, y, coverage)
    complete = [e for e in coverage if e.eligible_complete]
    event_id = np.full(len(index), "", dtype="<U64")
    for e in coverage:
        event_id[(index >= e.start) & (index < e.end_exclusive)] = e.event_id
    event_valid = (y == 0) | np.isin(event_id, [e.event_id for e in complete])
    pred = s > threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    hits = matched_count = predicted_count = normal_fragments = 0
    events, alarms, affinity = [], [], []
    aff_f1 = float("nan")
    blocks = real_blocks(index, event_valid)
    for bi, ix in enumerate(blocks):
        pe = [(int(index[ix[lo]]), int(index[ix[hi - 1]]) + 1)
              for lo, hi in binary_events(pred[ix])]
        truth = [(e.event_id, e.start, e.end_exclusive) for e in complete
                 if e.start >= index[ix[0]] and e.end_exclusive <= index[ix[-1]] + 1]
        assignment = greedy_assign(pe, truth)
        matched_count += sum(hit is not None for hit in assignment)
        predicted_count += len(pe)
        for (lo, hi), hit in zip(pe, assignment):
            alarms.append(dict(block=bi, start=lo, end_exclusive=hi, matched_event=hit))
        normal_fragments += len(binary_events(pred[ix] & (y[ix] == 0)))
        if affiliation is not None:
            gt = binary_events(y[ix])
            p = r = block_f1 = float("nan")
            reason = "NO_GROUND_TRUTH_EVENTS"
            if gt:
                ar = affiliation(binary_events(pred[ix]), gt, (0, len(ix)))
                p, r = float(ar["precision"]), float(ar["recall"])
                block_f1 = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and p + r > 0 else float("nan")
                reason = None if np.isfinite(block_f1) else "OFFICIAL_UNDEFINED"
            affinity.append(dict(block=bi, start=int(index[ix[0]]), end_exclusive=int(index[ix[-1]]) + 1,
                                 Aff_P=p, Aff_R=r, Aff_F1=block_f1, reason=reason))
    aff_reason = "NOT_EVALUATED_EXTERNAL_CALLBACK_REQUIRED"
    if affiliation is not None:
        aff_reason = "MULTIPLE_REAL_BLOCKS_NO_GLOBAL_AGGREGATION" if len(blocks) > 1 else "NO_VALID_BLOCKS"
        if len(blocks) == 1:
            aff_f1, aff_reason = affinity[0]["Aff_F1"], affinity[0]["reason"]
    for e in coverage:
        eligible = bool(e.eligible_complete)
        ix = np.flatnonzero((event_id == e.event_id) & pred)
        hit = bool(len(ix)) if eligible else None
        if eligible:
            hits += int(hit)
        events.append(dict(event_id=e.event_id, complete=eligible, any_hit=hit,
                           delay_sample_points=int(index[ix[0]] - e.start) if eligible and hit else None,
                           scored_points=e.scored_points, total_points=e.total_points))
    ap, roc = rank["AP"], rank["ROC"]
    if rank["positive_fraction"] != float(y.mean()):raise ValueError("Cached rank population mismatch")
    metrics = dict(AP=ap, ROC=roc, positive_fraction=float(y.mean()),
                   AP_over_positive_fraction=ap / y.mean(), threshold=threshold,
                   normal_FPR=fp / (fp + tn), point_Precision=tp / (tp + fp) if tp + fp else float("nan"),
                   TP=tp, FP=fp, TN=tn, FN=fn, complete_event_hits=hits,
                   complete_events=len(complete), original_events=len(coverage),
                   complete_event_recall_any_hit_aux=hits / len(complete) if complete else float("nan"),
                   predicted_events=predicted_count,
                   one_to_one_event_precision=matched_count / predicted_count if predicted_count else float("nan"),
                   normal_alarm_fragments=normal_fragments, Aff_F1=aff_f1,
                   Aff_F1_reason=aff_reason,
                   affiliation_status="CALLBACK_USED" if affiliation is not None else "NOT_EVALUATED_EXTERNAL_CALLBACK_REQUIRED",
                   delay_units="original sample points; no reliable physical clock")
    return {"metrics": metrics, "events": events, "alarms": alarms, "affinity": affinity,
            "complete_event_valid": event_valid}


def _bytes_identity(value):
    x=np.ascontiguousarray(value)
    return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def evaluate_registry(objects,entities,callback,profile):
    rows=[];events=[];alarms=[];affinity=[];prefix=[];coordinates=[];calls={'native':0,'legacy':0}
    for entity,q in entities.items():
        labels=np.load(q['label_path'],mmap_mode='r',allow_pickle=False)
        expected=(np.arange(profile['points'])>=q['event_start_0based'])&(np.arange(profile['points'])<q['event_end_exclusive'])
        if labels.shape!=(profile['points'],) or not np.array_equal(labels,expected.astype(labels.dtype)):raise ValueError('Frozen event labels/boundaries changed')
        features=np.load(q['array_path'],mmap_mode='r',allow_pickle=False)
        if features.shape!=(profile['points'],profile['channels']) or features.dtype!=np.float32:raise ValueError('Feature-prefix source shape/dtype mismatch')
        feature_hash=_bytes_identity(features[:q['event_start_0based']])
        index_expected=np.concatenate([np.arange(b,b+5) for b in q['origins']])
        for item in [o for o in objects if o['entity']==entity]:
            obj=item['object'];path=Path(obj['file']['path'])
            with np.load(path,allow_pickle=False) as bank:index=bank[obj['index_field']].copy();score=bank[obj['score_field']].copy()
            if score.dtype!=np.float32 or not np.array_equal(index,index_expected) or score.shape!=index.shape or not np.isfinite(score).all():raise ValueError('Canonical/index float32 population mismatch')
            y=labels[index];coverage=[dict(event_id=entity+'_event',start=q['event_start_0based'],end_exclusive=q['event_end_exclusive'],eligible_complete=True,
                scored_points=int((y==1).sum()),total_points=q['event_end_exclusive']-q['event_start_0based'])]
            rank_id=canonical_sha({k:item[k] for k in ('stage','initialization','method','seed','checkpoint_sha256','object')}|dict(label_sha256=q['label_sha256_from_frozen_parent'],index_sha256=_bytes_identity(index)))
            source=evaluate_fixed_scores(index,y,score,item['thresholds']['source'],coverage,affiliation=callback)
            calls['native' if item['method'] in MODES else 'legacy']+=1
            rank={k:source['metrics'][k] for k in ('AP','ROC','positive_fraction')}
            results={'source':source}
            if 'target' in item['thresholds']:results['target']=threshold_only(index,y,score,item['thresholds']['target'],coverage,affiliation=callback,rank=rank)
            for track,result in results.items():
                base={k:item[k] for k in ('stage','initialization','method','seed','checkpoint_sha256')};base.update(threshold_track=track,entity=entity,fault_id=q['fault_id'],run_id=q['run_id'],rank_object_id=rank_id)
                metrics=result['metrics'];pred=score>item['thresholds'][track];normal=index<q['event_start_0based'];hours=float(normal.sum()*profile['sample_interval_minutes']/60)
                metrics.update(point_Recall=metrics['TP']/(metrics['TP']+metrics['FN']),normal_observation_hours=hours,
                    normal_alarm_fragments_per_hour=metrics['normal_alarm_fragments']/hours if hours else float('nan'),
                    one_to_one_event_recall=sum(a['matched_event'] is not None for a in result['alarms'])/len(coverage),
                    pre_fault_alarm=bool(pred[normal].any()),alarm_continuing_across_fault_start=any(a['start']<q['event_start_0based']<a['end_exclusive'] for a in result['alarms']),
                    delay_units='original sample points; minutes uses frozen sample_interval_minutes',
                    operating_condition=q.get('operating_condition'),operating_condition_reason=None if q.get('operating_condition') is not None else 'NOT_PROVIDED_BY_FROZEN_QUERY_METADATA')
                metrics.update(point_Precision_reason=None if np.isfinite(metrics['point_Precision']) else 'NO_PREDICTED_POSITIVES',
                    normal_duration_reason=None if hours else 'NO_NORMAL_SCORED_OBSERVATION')
                rows.append(base|metrics)
                for event in result['events']:
                    event['delay_minutes']=None if event['delay_sample_points'] is None else event['delay_sample_points']*profile['sample_interval_minutes']
                    event.update(delay_reason=None if event['delay_sample_points'] is not None else 'NOT_DETECTED' if event['complete'] else 'INCOMPLETE_EVENT',
                        pre_fault_alarm=metrics['pre_fault_alarm'],alarm_continuing_across_fault_start=metrics['alarm_continuing_across_fault_start'],
                        original_start=q['event_start_0based'],original_end_exclusive=q['event_end_exclusive'],delay_caveat='zero may be continuing pre-fault alarm; no miss imputation')
                    events.append(base|event)
                alarms.extend(base|a for a in result['alarms']);affinity.extend(base|a for a in result['affinity'])
                prefix.append(base|dict(feature_prefix_sha256=feature_hash,score_prefix_sha256=_bytes_identity(score[normal]),binary_prefix_sha256=_bytes_identity(pred[normal]),
                    normal_points=int(normal.sum()),normal_false_points=metrics['FP'],normal_alarm_fragments=metrics['normal_alarm_fragments']))
            if 'target' in results and any(results['target']['metrics'][k]!=source['metrics'][k] for k in ('AP','ROC')):raise ValueError('S/T rank results changed')
            coordinates.append(dict(entity=entity,rank_object_id=rank_id,index_sha256=_bytes_identity(index),label_sha256=q['label_sha256_from_frozen_parent'],score_object=obj,
                history_start=[b-100 for b in q['origins']],forecast_origin=q['origins'],target_end_exclusive=[b+5 for b in q['origins']]))
    return rows,events,alarms,affinity,prefix,coordinates,calls


def run_evaluation(path,*,execute=False):
    c,paths,out,manifest,digest=load_operation(path)
    if not execute:return dict(executed=False,operation_manifest=manifest,operation_manifest_sha256=digest,scope='preview only; no assets/labels/arrays opened')
    a,seal,entities,synthetic=authorize(c,paths,out,digest,path)
    objects=registry(a,seal,entities,paths['seal'],c['inputs']['query_manifest'])
    callback,aff_provenance=load_official_affiliation(paths['affiliation'])
    out.mkdir(parents=True,exist_ok=False);json_write(out/'operation_manifest.json',manifest);json_write(out/'input_registry.json',dict(objects=objects,gate_summary=c['inputs']['gate_summary'],readout_runtime=c['inputs']['readout_runtime']))
    started=time.monotonic();code=runtime_identity()
    try:
        rows,events,alarms,affinity,prefix,coordinates,calls=evaluate_registry(objects,entities,callback,c['parameters']['profile'])
        stats=run_statistics(rows,c['parameters']['profile']);normal=normal_prefix_summary(prefix,c['parameters']['profile']);delays=delay_summary(events)
        if time.monotonic()-started>evaluation_limit_seconds(c['parameters']['profile']):raise ValueError('CPU evaluation profile time budget exceeded')
        from .multi_evaluation import write_csv
        np.savez_compressed(out/'paired_run_bootstrap.npz',**stats.pop('bootstrap_arrays'))
        for name,data in [('entity_metrics',rows),('event_results',events),('alarm_events',alarms),('entity_affiliation',affinity),('normal_prefix_by_run',normal),('delay_summary',delays),
            ('macro_summary',stats['macro']),('entity_seed_mean',stats['entity_seed_mean']),('fault_summary',stats['per_fault']),('run_summary',stats['per_run']),('worst_cases',stats['worst']),('paired_run_intervals',stats['intervals'])]:
            write_csv(out/(name+'.csv'),data,fields=None if data else ['stage','initialization','method','seed','reason'])
        write_csv(out/'failed_entity_metrics.csv',stats['failures'],fields=['stage','initialization','method','threshold_track','seed','fault_id','run_id','reason'])
        unsupported=[{k:o[k] for k in ('stage','initialization','method','seed','entity')}|dict(threshold_track='target',threshold=None,reason='NOT_IN_CONTRACT_TARGET_CALIBRATION_SOURCE_ONLY_CONTROL') for o in objects if o['method'] not in MODES]
        write_csv(out/'unsupported_target_tracks.csv',unsupported)
        json_write(out/'coordinates.json',coordinates)
        if runtime_identity()!=code:raise ValueError('Evaluation code bytes changed during execution')
        if any(sha256(Path(aff_provenance['directory'])/n)!=v for n,v in aff_provenance['files'].items()):raise ValueError('External official Aff changed')
        a.recheck()
        report=dict(status='COMPLETE_FIXED_STAGE2_EVALUATION' if stats['status']=='COMPLETE' else 'INCOMPLETE',stage=STAGE,synthetic_fixture=synthetic,
            metric_rows=len(rows),rank_call_counts=calls,total_rank_function_pairs=sum(calls.values()),native_rank_objects=27*len(entities),legacy_rank_objects=9*len(entities),
            score_tracks_single_object=True,point_adjustment=False,threshold_search=False,overall_Aff_F1=None,overall_Aff_reason='NO_OFFICIAL_GLOBAL_DEFINITION_ACROSS_ENTITIES',
            gate_acceptance=c['inputs']['gate_acceptance'],readout_runtime=c['inputs']['readout_runtime'],evaluation_runtime=c['inputs']['runtime'],affiliation=aff_provenance,
            normal_prefix_groups=len(normal),delay_summary=delays,failures=stats['failures'],seconds=time.monotonic()-started,
            oracle_rank_calls_included=False,inference=False,training=False,GPU_executed=False,scope='synthetic CPU engineering only' if synthetic else 'fixed label evaluation after separately accepted original-runtime gate')
        if sum(calls.values())!=36*len(entities):raise ValueError('Once-only rank call count incomplete')
        json_write(out/'evaluation_summary.json',_json_safe(report))
        json_write(out/'evaluation_manifest.json',dict(status='COMPLETE_FIXED_EVALUATION_ARCHIVE' if stats['status']=='COMPLETE' else 'INCOMPLETE',files={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()},requires_absence_of_failure_marker=True))
        return report
    except Exception as exc:
        json_write(out/'evaluation_failure.json',dict(status='FAILED_NO_COMPLETE_REPORT',error=str(exc)));raise
