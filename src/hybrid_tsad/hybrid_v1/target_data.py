"""Label-free target CSV derivation; no model, fit, score or label loader.

CSV parsing/canonical float32 values and finite-observed semantics reuse the
existing data.load_manifest implementation. No missingness is guessed from
repeats, zeros or an earlier LOCF transformation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np

OPERATION="HYBRID_V1_PSM_TARGET_INPUT_PREPARATION"
RAW_SHA="816e0e2a5356b3b09d356925cd9a0fe1d327e3b350e9336d26ee3e650c72e112"
ROLES=dict(FIT=[0,79488],SELECT=[80032,105984],CAL=[106528,132481])
PERMISSIONS=dict(existing_target_train=True,parse_once=True,query=False,labels=False,model=False,fit_or_calibrate=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def admit(path,release,trusted):
    cfg=json.loads(Path(path).read_text())
    if not release or not trusted:raise PermissionError('Named root target-input release required')
    if sha(release)!=trusted:raise PermissionError('Root release SHA changed')
    signed=json.loads(Path(release).read_text())
    if (signed.get('status')!='APPROVED' or signed.get('approved_by')!='root' or signed.get('operation')!=OPERATION
        or signed.get('allowed_actions')!=['prepare_target_inputs'] or signed.get('config_sha256')!=sha(path)
        or cfg.get('operation')!=OPERATION or cfg.get('permissions')!=PERMISSIONS):
        raise PermissionError('Pending/wrong target-input release')
    if Path(cfg['output']).exists():raise FileExistsError('Fresh derived output required')
    spec=cfg['raw_csv']
    if (spec.get('sha256')!=RAW_SHA or spec.get('expected_shape')!=[132481,25] or spec.get('format')!='csv'
        or spec.get('has_header') is not True or spec.get('columns')!=['feature_%d'%i for i in range(25)]
        or cfg.get('roles')!=ROLES or cfg.get('history')!=512 or cfg.get('horizon')!=32
        or cfg.get('selection_counts')!={'SELECT':32,'CAL':512} or cfg.get('sampling_seed')!=2025):
        raise PermissionError('Frozen original-train, roles, columns and observed-only sampling required')
    own=cfg['source_sha256'];base=Path(__file__).resolve().parents[2]
    if signed.get('source_sha256')!=own:raise PermissionError('Source closure changed')
    required={'hybrid_tsad/hybrid_v1/target_data.py','hybrid_tsad/hybrid_v1/data.py'}
    if not required.issubset(own):raise PermissionError('Parser and existing loader identities required')
    for name,digest in own.items():
        f=base/name
        if base not in f.resolve().parents or sha(f)!=digest:raise PermissionError('Derived-input source bytes changed')
    return cfg


def fixed_positions(observed,roles,*,history,horizon,counts,seed,identity):
    """Only finite-observed masks determine eligibility, never values/loss.

    Require a current observed channel with>=2 visible past observations in
    [t-H+1,t); this also meets forecast history support. SELECT reserves H+P,
    CAL H+1. Draw SELECT then CAL once with one local Generator, no sorting.
    """
    observed=np.asarray(observed)
    if observed.dtype!=np.bool_ or observed.ndim!=2 or history<3 or horizon<1:raise ValueError('Observed bool matrix and legal H/P required')
    if set(roles)!={'FIT','SELECT','CAL'}:raise ValueError('Three explicit roles required')
    ordered=[roles[k] for k in ('FIT','SELECT','CAL')]
    if any(not 0<=lo<hi<=len(observed) for lo,hi in ordered) or any(a[1]>b[0] for a,b in zip(ordered,ordered[1:])):
        raise ValueError('Original role ranges overlap or leave original axis')
    cumulative=np.concatenate([np.zeros((1,observed.shape[1]),np.int64),observed.cumsum(axis=0,dtype=np.int64)])
    rng=np.random.default_rng(seed);plans={};candidate_counts={}
    for role in ('SELECT','CAL'):
        lo,hi=roles[role];p=horizon if role=='SELECT' else 1
        candidates=np.arange(lo+history,hi-p+1,dtype=np.int64)
        visible_count=cumulative[candidates]-cumulative[candidates-history+1]
        candidates=candidates[np.all(observed[candidates] & (visible_count>=2),axis=1)]
        candidate_counts[role]=len(candidates)
        if len(candidates)<counts[role]:raise ValueError('Insufficient observed-only eligible '+role+' points; no fallback/resampling')
        chosen=rng.choice(candidates,size=counts[role],replace=False)
        plans[role]=[dict(**identity,role=role,index=int(t),footprint=[int(t)-history,int(t)+p]) for t in chosen]
    return plans,candidate_counts


def derive(cfg):
    """One CSV parse via the accepted loader, no learner is ever invoked."""
    from .data import load_manifest
    spec=cfg['raw_csv'];output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=False)
    if sha(spec['path'])!=spec['sha256']:raise ValueError('Original target CSV bytes changed before parse')
    with Path(spec['path']).open(encoding='utf-8-sig',newline='') as f:header=next(csv.reader(f))
    if header!=[cfg['timestamp_column']]+spec['columns']:raise ValueError('Original header/order changed')
    identity={k:cfg[k] for k in ('domain','entity','run')}
    # Internal metadata-only adapter to the existing parser; never an emitted
    # training-role manifest. Its complete raw view is not given to a learner.
    entry=dict(spec,**identity,role='FIT',bounds=[0,spec['expected_shape'][0]],original_start=0)
    with tempfile.TemporaryDirectory() as temporary:
        p=Path(temporary)/'parse_only.json';p.write_text(json.dumps(dict(runs=[entry])))
        runs=load_manifest(p,roles={'FIT'})
    if len(runs)!=1:raise ValueError('One original CSV file view required')
    raw=runs[0]
    if not np.array_equal(raw.observed,np.isfinite(raw.values)):raise ValueError('Canonical finite mask differs; no repair')
    plans,candidates=fixed_positions(raw.observed,cfg['roles'],history=cfg['history'],horizon=cfg['horizon'],
        counts=cfg['selection_counts'],seed=cfg['sampling_seed'],identity=identity)
    npz=output/'values_observed_original_axis.npz'
    np.savez_compressed(npz,values=raw.values,observed=raw.observed,original_index=np.arange(len(raw.values),dtype=np.int64),
                        columns=np.asarray(spec['columns'],dtype=str))
    digest=sha(npz)
    role_manifest=dict(runs=[dict(**identity,role=role,path=str(npz.resolve()),format='npz',key='values',observed_key='observed',
        sha256=digest,expected_shape=list(raw.values.shape),bounds=bounds,original_start=0) for role,bounds in cfg['roles'].items()],
        source_csv=dict(path=spec['path'],sha256=spec['sha256']),normality='Training-source roles, not certified industrial normal labels')
    (output/'target_role_manifest.json').write_text(json.dumps(role_manifest,indent=2))
    (output/'target_plans.json').write_text(json.dumps(dict(history=cfg['history'],horizon=cfg['horizon'],seed=cfg['sampling_seed'],**plans),indent=2))
    report=dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',source_csv=spec,actual_source_sha256=sha(spec['path']),
        derived_npz_sha256=digest,shape=list(raw.values.shape),values_dtype=str(raw.values.dtype),observed_dtype=str(raw.observed.dtype),
        observed_definition='np.isfinite(stored CSV numeric values), existing load_manifest/Run canonical float32; NaNs retained, Inf rejected by existing loader',
        upstream_imputation_recovery=False,zero_repeat_or_LOCF_mask_guess=False,physical_clock='NOT_ASSERTED; original_index is CSV row axis',
        timestamp_column_excluded_from_features=cfg['timestamp_column'],source_numeric_parse_calls=1,
        roles=cfg['roles'],candidate_counts=candidates,selected_counts={k:len(v) for k,v in plans.items()},
        per_role_missing={k:int((~raw.observed[a:b]).sum()) for k,(a,b) in cfg['roles'].items()},
        no_model_fit_or_calibration_or_labels=True,sampling='Local Generator2025, SELECT then CAL, no replacement, returned order retained')
    (output/'report.json').write_text(json.dumps(report,indent=2));return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--release')
    p.add_argument('--trusted-release-sha256');p.add_argument('--execute',action='store_true');a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,operation=OPERATION,raw_body_or_model_opened=False)));return
    cfg=admit(a.config,a.release,a.trusted_release_sha256);print(json.dumps(derive(cfg)))


if __name__=='__main__':main()
