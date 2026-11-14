"""Fixed post-production CPU h1 error diagnostics; no model or new score.

Uses exact archived raw predictions/targets, shared valid channel intersections,
and production evaluation's original labels/events. Never calls a neural model.
"""
import argparse
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path, PurePosixPath
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hybrid_tsad.hybrid_v1.chunked_bank import read_records
from hybrid_tsad.hybrid_v1.controlled_mechanism import _unpack

METHODS=('HybridV1_Z_S','persistence','history_mean512')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bound(spec):
    if sha(spec['path'])!=spec['sha256']:raise PermissionError('Changed bound diagnostic input')
    return json.loads(Path(spec['path']).read_text(encoding='utf-8'))
def dump(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def original_artifact(cfg,path):
    """Resolve bytes of an archived remote artifact without rewriting provenance."""
    mapping=cfg['path_mapping'];prefix=mapping['remote_root'].rstrip('/')+'/'
    if not path.startswith(prefix):raise PermissionError('Artifact is outside the one frozen remote workspace')
    relative=PurePosixPath(path[len(prefix):])
    if relative.is_absolute() or '..' in relative.parts:raise PermissionError('Unsafe remote artifact path')
    root=Path(mapping['local_root']).resolve();p=(root/Path(*relative.parts)).resolve()
    if root not in p.parents:raise PermissionError('Mapped artifact escapes local workspace')
    return p

def exact_h1(record,method):
    """Original archived prediction[0], never r_pred or normalized output."""
    try:
        p=np.asarray(record['prediction'][0],np.float64)
        if method=='HybridV1_Z_S':
            replay=record['replay'];y=np.asarray(replay['current_target'][0],np.float64)
            observed=_unpack(replay['current_observed'])[0]
            available=_unpack(replay['forecast_valid'])[0,0]
            bounds=replay['forecast_history_bounds']
        else:
            y=np.asarray(record['current_target'],np.float64)
            observed=np.asarray(record['current_observed'],bool)
            available=np.asarray(record['prediction_valid'],bool)
            bounds=record['forecast_history_bounds']
    except KeyError as exc:
        raise ValueError('EXACT_ORIGINAL_H1_OR_TARGET_ARCHIVE_MISSING: '+str(exc)) from exc
    t=record['index']
    if (p.ndim!=1 or p.shape!=y.shape or observed.shape!=p.shape or available.shape!=p.shape
            or record['available_index']!=t or record['forecast_issued_index']!=t-1
            or bounds!=[t-512,t]):
        raise ValueError('Archived h1 original shape/time/permission mismatch')
    return p,y,observed,available & observed & np.isfinite(p) & np.isfinite(y)

def summarize(errors,absolute,count,select):
    n=int(count[select].sum());points=int(select.sum());usable=int((select & (count>0)).sum())
    return dict(candidate_points=points,common_valid_points=usable,
        excluded_no_common_valid_points=points-usable,valid_channel_observations=n,
        squared_error_sum=float(errors[select].sum()),absolute_error_sum=float(absolute[select].sum()),
        original_input_scale_MSE=float(errors[select].sum()/n) if n else None,
        original_input_scale_MAE=float(absolute[select].sum()/n) if n else None,
        NA_reason=None if n else 'NO_COMMON_VALID_CHANNEL_OBSERVATIONS')

def check_production(cfg):
    scope=bound(cfg['diagnostic_scope']);receipt=bound(cfg['production_receipt'])
    terminal=bound(cfg['production_terminal']);summary=bound(cfg['production_summary']);production=bound(cfg['production_config'])
    if (scope['methods']!=list(METHODS) or scope['training'] or scope['new_forward'] or scope['new_score']
            or terminal.get('OS_exit_code')!=0 or receipt.get('status')!='COMPLETED'
            or receipt.get('banks')!=162 or receipt.get('entities')!=54
            or summary.get('status')!='COMPLETED_PENDING_INDEPENDENT_AUDIT'
            or not receipt.get('all_raw_before_truth')):
        raise PermissionError('All162 raw banks and real production OS0 label gate required')
    if sha(__file__)!=cfg['script_sha256']:raise PermissionError('Frozen diagnostic source required')
    for spec in cfg['source_dependencies']:
        if sha(spec['path'])!=spec['sha256']:raise PermissionError('Original raw decoder changed')
    by={(d['method'],d['domain'],d['entity'],d['run']):d for d in production['banks']}
    if len(by)!=162 or {k[0] for k in by}!=set(METHODS):raise PermissionError('Exact production method/cohort matrix required')
    # All manifests are checked before any additional archived labels are read.
    for d in by.values():
        path=original_artifact(cfg,d['directory'])/'manifest.json'
        if sha(path)!=d['manifest_sha256']:raise PermissionError('Production raw bank changed')
        m=json.loads(path.read_text())
        if not m['complete'] or m['contract']['identity']!=d['identity']:raise PermissionError('Incomplete/mismatched production raw bank')
    return production,by

def run(cfg):
    began=time.monotonic();production,by=check_production(cfg)
    evaluation_dir=Path(cfg['production_summary']['path']).parent
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    entity_rows=[];event_rows=[];pair_rows=[];array_files=[];missing=[]
    for domain,entity,run_id in sorted({k[1:] for k in by}):
        descriptors=[by[(m,domain,entity,run_id)] for m in METHODS]
        # Reuse production's preserved full-axis truth, not a new label variant.
        calc=evaluation_dir/'entities'/domain/entity/'HybridV1_Z_S'/'calculation_inputs.npz'
        result_file=calc.parent/'result.json';provenance=calc.parent/'provenance.json'
        prov=json.loads(provenance.read_text());result=json.loads(result_file.read_text())
        if sha(calc)!=prov['calculation_inputs_sha256']:raise PermissionError('Production calculation inputs changed')
        with np.load(calc,allow_pickle=False) as z:
            y=z['full_labels'];expected=z['index'].copy()
        if hashlib.sha256(y.tobytes()).hexdigest()!=result['input_identity']['label_array_sha256']:
            raise ValueError('Production full original truth differs')
        squared=[];absolute=[];counts=[];indices=[]
        iters=[read_records(original_artifact(cfg,d['directory'])) for d in descriptors]
        sentinel=object()
        try:
            for records in zip_longest(*iters,fillvalue=sentinel):
                if any(r is sentinel for r in records):raise ValueError('Unequal method raw point counts')
                t=records[0]['index']
                if any(tuple(r[k] for k in ('domain','entity','run'))!=(domain,entity,run_id) or r['index']!=t for r in records):
                    raise ValueError('Original native method axes differ')
                parts=[exact_h1(r,m) for r,m in zip(records,METHODS)]
                target=parts[0][1];observed=parts[0][2]
                if any(not np.array_equal(v[1],target,equal_nan=True) or not np.array_equal(v[2],observed) for v in parts[1:]):
                    raise ValueError('Archived original targets/observations differ across methods')
                common=np.logical_and.reduce([v[3] for v in parts]);counts.append(int(common.sum()));indices.append(t)
                errors=np.array([p[common]-target[common] for p,_,_,_ in parts])
                squared.append(np.square(errors).sum(axis=1));absolute.append(np.abs(errors).sum(axis=1))
        except ValueError as exc:
            if not str(exc).startswith('EXACT_ORIGINAL_H1_OR_TARGET_ARCHIVE_MISSING'):raise
            missing.append(dict(domain=domain,entity=entity,run=run_id,status='ARCHIVE_MISSING_NO_REFORWARD',reason=str(exc)))
            continue
        index=np.asarray(indices,np.int64);count=np.asarray(counts,np.int64)
        se=np.asarray(squared,np.float64);ae=np.asarray(absolute,np.float64)
        if not np.array_equal(index,expected) or se.shape!=(len(expected),3):raise ValueError('Production raw axis/order missing')
        positive=y[index]==1;onset=np.zeros(len(index),bool);steady=np.zeros(len(index),bool)
        phases=[]
        for event in result['event_coverage']:
            lo,hi=event['start'],event['end_exclusive'];cut=min(lo+512,hi)
            onset|=(index>=lo)&(index<cut);steady|=(index>=cut)&(index<hi)
            for phase,a,b in [('all',lo,hi),('onset512',lo,cut),('steady_after512',cut,hi)]:
                phases.append((event,phase,(index>=a)&(index<b)))
        if not np.array_equal(onset|steady,positive) or (onset&steady).any():raise ValueError('Original event phases differ from labels')
        groups={'ALL':np.ones(len(index),bool),'NORMAL':~positive,'ANOMALY':positive,'ANOMALY_ONSET512':onset,'ANOMALY_STEADY_AFTER512':steady}
        for j,method in enumerate(METHODS):
            for group,select in groups.items():
                entity_rows.append(dict(domain=domain,entity=entity,run=run_id,method=method,group=group,
                    **summarize(se[:,j],ae[:,j],count,select)))
            for event,phase,select in phases:
                event_rows.append(dict(domain=domain,entity=entity,run=run_id,method=method,
                    source_event_id=event.get('source_event_id',event['event_id']),start=event['start'],end_exclusive=event['end_exclusive'],
                    main_event_eligible=event['main_eligible'],boundary_censored=event['boundary_censored'],phase=phase,
                    **summarize(se[:,j],ae[:,j],count,select)))
        for group,select in groups.items():
            for j in (1,2):
                denom=int(count[select].sum())
                pair_rows.append(dict(domain=domain,entity=entity,group=group,reference=METHODS[j],candidate=METHODS[0],
                    valid_channel_observations=denom,
                    candidate_minus_reference_MSE=float((se[select,0]-se[select,j]).sum()/denom) if denom else None,
                    positive_means_candidate_worse=True,statistical_confirmation=False))
        path=out/'arrays'/domain/entity;path.mkdir(parents=True)
        np.savez_compressed(path/'original_h1_errors.npz',index=index,point_labels=y[index],common_valid_channel_count=count,
            squared_error_sum_by_method=se,absolute_error_sum_by_method=ae,anomaly_onset512=onset,anomaly_steady_after512=steady)
        array_files.append(dict(domain=domain,entity=entity,path=str(path/'original_h1_errors.npz'),sha256=sha(path/'original_h1_errors.npz'),
            production_inputs_sha256=sha(calc),banks=[dict(method=d['method'],manifest_sha256=d['manifest_sha256']) for d in descriptors]))
    dataset=[]
    for domain in ('SMAP','SMD'):
        for method in METHODS:
            for group in ('ALL','NORMAL','ANOMALY','ANOMALY_ONSET512','ANOMALY_STEADY_AFTER512'):
                rows=[r for r in entity_rows if r['domain']==domain and r['method']==method and r['group']==group]
                n=sum(r['valid_channel_observations'] for r in rows);ss=sum(r['squared_error_sum'] for r in rows);aa=sum(r['absolute_error_sum'] for r in rows)
                macro=[r['original_input_scale_MSE'] for r in rows if r['original_input_scale_MSE'] is not None]
                dataset.append(dict(domain=domain,method=method,group=group,entities=len(rows),valid_channel_observations=n,
                    common_valid_points=sum(r['common_valid_points'] for r in rows),original_input_scale_MSE=ss/n if n else None,
                    original_input_scale_MAE=aa/n if n else None,entity_macro_MSE=float(np.mean(macro)) if macro else None,
                    entity_macro_denominator=len(macro)))
    csvwrite(out/'ENTITY_H1_ERRORS.csv',entity_rows);csvwrite(out/'EVENT_H1_ERRORS.csv',event_rows)
    csvwrite(out/'DATASET_H1_ERRORS.csv',dataset);csvwrite(out/'PAIRED_ENTITY_MSE_DIFFERENCES.csv',pair_rows)
    dump(out/'ARRAY_BINDINGS.json',array_files);dump(out/'MISSING_ARCHIVES.json',missing)
    dump(out/'CONFIG.json',cfg)
    dump(out/'RESULT.json',dict(status='COMPLETED_DESCRIPTIVE_ONLY' if not missing else 'INCOMPLETE_MISSING_ORIGINAL_ARCHIVES',
        entities=len(array_files),methods=list(METHODS),event_intervals=len(event_rows)//9,
        interpretation='Published input units, not certified physical units; lower normal MSE is not anomaly-detection evidence',
        common_valid_channel_intersection=True,original_event_boundaries=True,normalized_residual_used=False,
        new_score=False,new_forward=False,new_training=False,GPUh=0,seconds=time.monotonic()-began,dataset=dataset))
    return dataset

def self_test():
    """Two tiny fixtures; no real inputs/truth are read."""
    y=np.array([3.,4.]);m=np.ones(2,bool)
    packed=lambda a:dict(shape=list(a.shape),packbits_hex=np.packbits(a.reshape(-1),bitorder='little').tobytes().hex(),unpacked_sha256=hashlib.sha256(a.tobytes()).hexdigest())
    base=dict(index=512,available_index=512,forecast_issued_index=511,prediction=[[2.,6.]],
        current_target=y.tolist(),current_observed=m.tolist(),prediction_valid=m.tolist(),forecast_history_bounds=[0,512])
    p,t,o,v=exact_h1(base,'persistence');assert np.array_equal(p-t,[-1,2]) and v.all()
    neural=dict(index=512,available_index=512,forecast_issued_index=511,prediction=[[2.,6.]],replay=dict(
        current_target=[y.tolist()],current_observed=packed(m[None]),forecast_valid=packed(m[None,None]),forecast_history_bounds=[0,512]))
    a=exact_h1(neural,'HybridV1_Z_S');assert np.array_equal(a[0],p) and np.array_equal(a[1],t)
    r=summarize(np.array([5.,0.]),np.array([3.,0.]),np.array([2,0]),np.array([True,True]))
    assert r['original_input_scale_MSE']==2.5 and r['original_input_scale_MAE']==1.5 and r['excluded_no_common_valid_points']==1
    mapped=original_artifact(dict(path_mapping=dict(remote_root='/home/a312/tsad_research',local_root=str(ROOT))),
        '/home/a312/tsad_research/results/fixed/bank')
    assert mapped==ROOT/'results/fixed/bank'
    print(json.dumps(dict(status='CPU_FIXTURES_PASS',fixtures=3,no_real_truth=True)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--trusted-config-sha256');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:return self_test()
    if not a.config or sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Bound diagnostic config required')
    run(json.loads(Path(a.config).read_text()))

if __name__=='__main__':main()
