"""Release-gated normal20 score loader; no model, positive labels or bootstrap."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import time
from . import tep_normal_control as identity
from . import tep_normal_control_evaluation as core

OP='TEP_NORMAL_CONTROL_EVALUATION'
RAW_SHA='4f45afafa469793eeb7203fb9ed10ed0b1724c73c9c95537f15a0889ade0ebd4'
CORE_SHA='2391dd746baa7b9085cbb2e639c1c64c388972366add7fe5e545292a78827c0f'

def sha(data):return hashlib.sha256(data).hexdigest()
def source_hashes():
    p=Path(__file__).resolve().parent
    paths=dict(runner=Path(__file__),core=Path(core.__file__),identity=Path(identity.__file__),
        events=p/'engine/events.py',engine_init=p/'engine/__init__.py',package_init=p/'__init__.py')
    return {k:sha(v.read_bytes()) for k,v in paths.items()}

def ref_path(ref,workspace,suffix='.json'):
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise PermissionError('Exact metadata reference')
    identity.digest(ref['sha256']);p=Path(ref['path']).resolve(strict=True);root=Path(workspace).resolve(strict=True)
    if not Path(ref['path']).is_absolute() or root not in p.parents or not p.is_file() or p.suffix.lower()!=suffix:
        raise PermissionError('Absolute in-workspace closed file reference')
    if suffix=='.npz' and p.name!='compact_scores.npz':raise PermissionError('Compact score payload only')
    return p

def verified_bytes(ref,workspace,suffix='.json'):
    data=ref_path(ref,workspace,suffix).read_bytes()
    if sha(data)!=ref['sha256']:raise PermissionError('Referenced bytes changed')
    return data

def meta(ref,workspace):
    def unique(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('Duplicate JSON key')
            d[k]=v
        return d
    return json.loads(verified_bytes(ref,workspace),object_pairs_hook=unique,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def verify_metadata(config_ref,release_ref,trusted,workspace):
    c=meta(config_ref,workspace)
    if c.get('status')!='BOUND_NORMAL_CONTROL_EVALUATION' or not isinstance(c.get('score_seal'),dict):
        raise PermissionError('PENDING_NORMAL_SCORE_SEAL before any score access')
    out=Path(c.get('output',''));root=Path(workspace).resolve()
    if not out.is_absolute() or out.resolve().parent!=root/'results' or out.name!='tep_normal_control_evaluation_20261102_v1' or out.exists():
        raise PermissionError('Fresh dedicated normal output before score access')
    identity.digest(trusted)
    if release_ref.get('sha256')!=trusted:raise PermissionError('External root release digest required')
    release=meta(release_ref,workspace)
    if (release.get('status')!='PRIMARY_RELEASED_TEP_NORMAL_CONTROL_EVALUATION' or release.get('approved_by')!='root'
            or release.get('operation')!=OP or release.get('config')!=config_ref
            or release.get('allow_normal_score_evaluation') is not True or release.get('allow_model_or_fit') is not False
            or release.get('allow_bootstrap') is not False or source_hashes()['core']!=CORE_SHA or release.get('source_hashes')!=source_hashes()
            or release.get('score_seal')!=c['score_seal']):raise PermissionError('Exact root normal-only evaluation release')
    seal=meta(c['score_seal'],workspace)
    if (seal.get('status')!='PRIMARY_SEALED_ALL_TEP_NORMAL_CONTROL_SCORES' or seal.get('approved_by')!='root'
            or seal.get('operation')!=OP or seal.get('entities')!=20 or seal.get('points_per_entity')!=861
            or seal.get('endpoints_0based')!=[99,959]):raise PermissionError('Complete normal20 score seal required')
    accepted=meta(seal['input_acceptance'],workspace);manifest=meta(seal['input_manifest'],workspace)
    if (accepted.get('status')!='PRIMARY_ACCEPTED_LIMITED_TEP_NORMAL_CONTROL_INPUTS' or accepted.get('approved_by')!='root'
            or accepted.get('entities')!=20 or accepted.get('source_sha256')!=RAW_SHA
            or accepted.get('input_manifest_sha256')!=seal['input_manifest']['sha256']
            or manifest.get('raw_sha256')!=RAW_SHA):raise PermissionError('Accepted normal20 input identity required')
    records=identity.validate_manifest(manifest)
    models=seal.get('score_models')
    if not isinstance(models,list) or [(m.get('family'),m.get('seed')) for m in models]!=[('LEFT',2025),('LEFT',2026),('LEFT',2027),('RIDGE_ZMAX',None)]:
        raise PermissionError('Three original seeds and deterministic baseline, no substitution')
    calibrations=[]
    for model in models:
        left=model['family']=='LEFT'
        if left and type(model['seed']) is not int:raise PermissionError('Integer training seed')
        a=meta(model['score_acceptance'],workspace);report=meta(model['score_report'],workspace)
        final=meta(model['score_manifest'],workspace);cal=meta(model['calibration'],workspace)
        expected='PRIMARY_ACCEPTED_LIMITED_NORMAL20_LEFT_SCORES' if left else 'PRIMARY_ACCEPTED_LIMITED_NORMAL20_RIDGE_SCORES'
        if (a.get('status')!=expected or a.get('approved_by')!='root' or a.get('entities')!=20
                or a.get('raw_bank_count')!=20 or a.get('scored_points_per_entity')!=861 or a.get('endpoints_0based')!=[99,959]):
            raise PermissionError('Independent full20 score acceptance required')
        for field,key in [('source_manifest_sha256','score_manifest'),('report_sha256','score_report'),
                          ('compact_scores_sha256','compact_scores'),('calibration_sha256','calibration')]:
            if a.get(field)!=model[key]['sha256']:raise PermissionError('Score acceptance chain differs: '+field)
        if a.get('input_manifest_sha256')!=seal['input_manifest']['sha256']:raise PermissionError('Wrong input source in score acceptance')
        parent=ref_path(model['score_manifest'],workspace).parent;indexed={}
        for f in final.get('files',[]):
            p=Path(f['path'])
            if p.is_absolute() or '..' in p.parts or str(p) in indexed:raise PermissionError('Unsafe/duplicate score manifest member')
            indexed[str(p)]=identity.digest(f['sha256'])
        for key in ['score_report','compact_scores']:
            p=ref_path(model[key],workspace,'.npz' if key=='compact_scores' else '.json')
            if indexed.get(str(p.relative_to(parent)))!=model[key]['sha256']:raise PermissionError('Manifest does not bind report/compact')
        if (report.get('status')!='COMPLETED_PENDING_INDEPENDENT_REVIEW' or report.get('role')!='NORMAL_CONTROL'
                or [b.get('entity_id') for b in report.get('banks',[])]!=list(identity.ENTITIES)
                or report.get('compact_scores_sha256')!=model['compact_scores']['sha256']):raise PermissionError('Incomplete normal bank report')
        refs=report.get('source_refs',{})
        if refs.get('input_manifest')!=seal['input_manifest'] or refs.get('calibration')!=model['calibration']:
            raise PermissionError('Original input/CAL refs changed')
        keys={'last','mean32'} if left else {m+'_'+d for m in core.METHODS[1:] for d in core.READOUTS}
        if (cal.get('n')!=8020 or cal.get('rank_1based')!=7941 or cal.get('comparison')!='strict_greater'
                or set(cal.get('thresholds',{}))!=keys or report.get('thresholds')!=cal['thresholds']):raise PermissionError('Original fixed normal CAL thresholds only')
        import math
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in cal['thresholds'].values()):raise PermissionError('Finite CAL threshold')
        scaler=identity.digest(model['scaler_sha256'])
        if cal.get('scaler_sha256')!=scaler or refs.get('scaler',{}).get('sha256')!=scaler:raise PermissionError('Original scaler identity')
        if left:
            checkpoint=identity.digest(model['checkpoint_sha256'])
            if (a.get('seed')!=model['seed'] or report.get('seed')!=model['seed'] or cal.get('seed')!=model['seed']
                    or a.get('checkpoint_sha256')!=checkpoint or cal.get('checkpoint_sha256')!=checkpoint
                    or refs.get('checkpoint',{}).get('sha256')!=checkpoint):raise PermissionError('Original seed/checkpoint/CAL identity')
        else:
            coefficient=identity.digest(model['coefficients_sha256'])
            if (cal.get('coefficient_sha256')!=coefficient or refs.get('coefficients',{}).get('sha256')!=coefficient
                    or a.get('coefficients_sha256')!=coefficient):raise PermissionError('Original deterministic coefficients')
        calibrations.append(cal)
    verified_bytes(c['contract'],workspace,'.md')
    return dict(config=c,seal=seal,records=records,calibrations=calibrations)

def validate_compact(arrays,family):
    import numpy as np
    common={'entity_id','end_index_0based'}
    required=common|({'last','mean32','channel_last','channel_mean32','available_index_0based'} if family=='LEFT' else {m+'_'+d for m in core.METHODS[1:] for d in core.READOUTS})
    if family not in ['LEFT','RIDGE_ZMAX'] or set(arrays)!=required:raise ValueError('Exact normal20 compact schema')
    if arrays['entity_id'].dtype.kind!='U' or arrays['entity_id'].tolist()!=list(identity.ENTITIES):raise ValueError('Ordered20normalIDs, never400faults')
    ends=arrays['end_index_0based']
    if ends.dtype.kind not in 'iu' or ends.shape!=(20,861) or not np.array_equal(ends,np.tile(np.arange(99,960),(20,1))):raise ValueError('Complete861point coordinate for each normalrun')
    for name,v in arrays.items():
        if name in common:continue
        if name=='available_index_0based':
            if v.dtype.kind not in 'iu' or not np.array_equal(v,ends):raise ValueError('Window-end availability')
        else:
            expected=(20,861,3) if name.startswith('channel_') else (20,861)
            if v.shape!=expected or v.dtype.kind not in 'fi' or not np.isfinite(v).all():raise ValueError('Finite complete continuous score arrays')

def load_verified_scores(config_ref,release_ref,trusted,workspace):
    state=verify_metadata(config_ref,release_ref,trusted,workspace)
    import numpy as np
    banks=[]
    for model in state['seal']['score_models']:
        data=verified_bytes(model['compact_scores'],workspace,'.npz')
        with np.load(io.BytesIO(data),allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        validate_compact(a,model['family']);banks.append(a)
    return state,banks

def execute(config_ref,release_ref,trusted,workspace):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise PermissionError('CPU-only single-thread resource shell')
    start=time.monotonic();cpu=time.process_time()
    state,banks=load_verified_scores(config_ref,release_ref,trusted,workspace)
    out=Path(state['config']['output']);root=Path(workspace).resolve()
    if not out.is_absolute() or out.resolve().parent!=root/'results' or out.name!='tep_normal_control_evaluation_20261102_v1' or out.exists():raise PermissionError('Fresh dedicated normal evaluation output')
    rows=[]
    for model,cal,bank in zip(state['seal']['score_models'],state['calibrations'],banks):
        for i,run in enumerate(core.RUNS):
            for method in (('LEFT',) if model['family']=='LEFT' else core.METHODS[1:]):
                for d in core.READOUTS:
                    key=d if method=='LEFT' else method+'_'+d
                    rows.append(dict(method=method,readout=d,seed=model['seed'],run=run,
                        result=core.evaluate_normal(bank[key][i],bank['end_index_0based'][i],cal['thresholds'][key])))
    summary=core.summarize_normal_rows(rows)
    if time.monotonic()-start>300:raise TimeoutError('Fixed300s CPU evaluation budget; no reduced protocol')
    out.mkdir(exist_ok=False)
    def dump(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    dump('all_rows_segments_and_NA.json',rows);dump('summary.json',summary)
    with (out/'normal_metrics.csv').open('x',newline='',encoding='utf-8') as f:
        fields=['method','readout','seed','run']+list(rows[0]['result']['metrics'])+['first_eligible_new_confirmation_sample','first_new_confirmation_delay_minutes','first_new_confirmation_administratively_right_censored']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow(dict({k:row[k] for k in fields[:4]},**row['result']['metrics'],**{k:row['result'][k] for k in fields if k.startswith('first_')}))
    import resource
    dump('execution_receipt.json',dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',config=config_ref,release=release_ref,
        score_seal=state['config']['score_seal'],source_hashes=source_hashes(),input_hashes=state['seal'],
        elapsed_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,GPU_hours=0,
        rows=320,independent_runs=20,bootstrap_performed=False,scientific_decision='HOLD',innovation='UNSUPPORTED'))
    dump('FINAL_MANIFEST.json',dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',files=[dict(path=p.name,sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in sorted(out.iterdir())]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',required=True);p.add_argument('--config',required=True);p.add_argument('--config-sha256',required=True);p.add_argument('--release',required=True);p.add_argument('--trusted-release-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    c=dict(path=a.config,sha256=a.config_sha256);r=dict(path=a.release,sha256=a.trusted_release_sha256)
    if a.execute:execute(c,r,a.trusted_release_sha256,a.workspace)
    else:verify_metadata(c,r,a.trusted_release_sha256,a.workspace)

if __name__=='__main__':main()
