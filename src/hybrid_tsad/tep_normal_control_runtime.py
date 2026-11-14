"""Stage-separated normal-control runtime candidate; exact root release only.

Parse is independently freezeable. No inference implementation is yet released.
No method modifies frozen400fault, normal training, CAL or model source.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from . import tep_normal_control as control
from .tep_faulty_inputs import unique_dataframe

ROOT=Path('/home/a312/tsad_research')
RAW=ROOT/'data/tep_faultfree_testing_20261102_v1/TEP_FaultFree_Testing.RData'
RAW_SHA='4f45afafa469793eeb7203fb9ed10ed0b1724c73c9c95537f15a0889ade0ebd4'
RAW_MD5='38ad9810fc871026157086ae2c2f0ee9'
RAW_BYTES=47327663
OUT=ROOT/'results/tep_normal_control_inputs_run51_70_20261102_v1'
PARSE_OP='TEP_NORMAL_CONTROL_INPUT_EXTRACTION'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def verify_config(config_path):
    c=read(config_path)
    if (c.get('operation')!=PARSE_OP or c.get('raw_path')!=str(RAW) or c.get('output')!=str(OUT)
            or c.get('raw_sha256')!=RAW_SHA or c.get('raw_MD5')!=RAW_MD5 or c.get('raw_bytes')!=RAW_BYTES
            or c.get('object_name')!='fault_free_testing' or c.get('run_ids')!=list(control.RUNS)
            or c.get('columns')!=list(control.COLUMNS) or c.get('max_seconds')!=300):
        raise PermissionError('Exact normal20input extraction identity only')
    return c

def require_release(config_path,release_path,trusted,config):
    control.digest(trusted)
    if sha(release_path)!=trusted:raise PermissionError('External root trusted releaseSHA required')
    r=read(release_path)
    expected=dict(status='PRIMARY_RELEASED_TEP_NORMAL_CONTROL_INPUT_EXTRACTION',approved_by='root',
        operation=PARSE_OP,config_sha256=sha(config_path),allow_one_read_r=True,
        allow_normal20_inputs=True,allow_model_scoring=False,allow_metrics=False,
        helper_hashes=config.get('helper_hashes'))
    if any(r.get(k)!=v for k,v in expected.items()) or config.get('binding_state')!='BOUND_NORMAL_CONTROL_PARSE':
        raise PermissionError('PENDING/exact parse-only root release required')
    required={'__init__.py','tep_faulty_inputs.py','tep_normal_control.py','tep_normal_control_runtime.py'}
    if set(config.get('helper_hashes',{}))!=required:raise PermissionError('Exact four-helper closure required')
    for name,h in config['helper_hashes'].items():
        if sha(Path(__file__).parent/name)!=h:raise PermissionError('Frozen parse helper changed')
    for key in ('acquisition_receipt','contract'):
        ref=config.get(key)
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or sha(ref['path'])!=ref['sha256']:
            raise PermissionError('Actual metadata ref required: '+key)
    acquisition=read(config['acquisition_receipt']['path'])
    if acquisition.get('status')!='ACQUIRED_IDENTITY_VERIFIED_NO_PARSE' or acquisition.get('actual_SHA256')!=RAW_SHA or acquisition.get('target')!=str(RAW):
        raise PermissionError('Actual acquired source identity differs')
    return r

def read_normal_once(raw_path,reader,*,group_function=control.group_normal_control_arrays):
    """Injectable once-only parser core. Reader never retried even on failure."""
    objects=reader(str(raw_path))
    name,frame=unique_dataframe(objects)
    if name!='fault_free_testing':raise ValueError('Published normal object name differs')
    groups=group_function(frame)
    if [run for run,a in groups]!=list(control.RUNS):raise ValueError('Twenty normal run ordering differs')
    for run,arrays in groups:control.validate_arrays(arrays)
    return name,groups

def parse_and_write(raw_path,output,reader,*,group_function=control.group_normal_control_arrays):
    """No stat/hash/download in pure parser hook; caller must bind raw bytes first."""
    out=Path(output)
    if out.exists():raise FileExistsError('Fresh parse output; no overwrite/retry')
    start=time.monotonic();out.mkdir(parents=False,exist_ok=False)
    report=dict(status='RUNNING',operation=PARSE_OP,read_r_calls=0,raw_sha256=RAW_SHA,
        role='NORMAL_CONTROL',entities=20,GPU_hours=0,no_model_or_metrics=True,
        post_hoc_control=True,not_strict_blind=True,no_known_lag_truth=True)
    try:
        report['read_r_calls']=1
        name,groups=read_normal_once(raw_path,reader,group_function=group_function)
        import numpy as np
        inputs=out/'inputs';inputs.mkdir();records=[]
        for run,arrays in groups:
            entity='normal_run_%04d'%run;path=inputs/(entity+'.npz')
            np.savez(path,**arrays)
            records.append(dict(entity_id=entity,role='NORMAL_CONTROL',faultNumber=0,simulationRun=run,
                path=str(path),sha256=sha(path),bytes=path.stat().st_size,shape=[960,3],dtype='float64',
                columns=list(control.COLUMNS),raw_sha256=RAW_SHA))
        manifest=dict(status='PUBLIC_SIMULATION_NORMAL_CONTROL_INPUTS',records=records,
            raw_source=RAW.name,raw_sha256=RAW_SHA,sample_axis=[1,960,1],
            imputation='none',standardization='none',sorting='none')
        control.validate_manifest(manifest)
        dump(inputs/'MANIFEST.json',manifest)
        report.update(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',actual_object_name=name,
            shape_full=[480000,55],input_manifest_sha256=sha(inputs/'MANIFEST.json'),
            elapsed_seconds=time.monotonic()-start)
        dump(out/'report.json',report)
        files=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size)
               for p in sorted(out.rglob('*')) if p.is_file()]
        dump(out/'FINAL_MANIFEST.json',dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',files=files))
        return report
    except BaseException as exc:
        report.update(status='FAILED_NO_RETRY',error=repr(exc),elapsed_seconds=time.monotonic()-start)
        dump(out/'report.json',report)
        raise

def parse_stage(config_path,release_path,trusted):
    c=verify_config(config_path);require_release(config_path,release_path,trusted,c)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):
        raise PermissionError('CPU-only single-thread resource shell required')
    if OUT.exists():raise FileExistsError('Fresh output required before raw read')
    # These actual raw bytes are touched only after root binding, never in preview.
    before=RAW.stat();md5=hashlib.md5();h=hashlib.sha256()
    if before.st_size!=RAW_BYTES:raise ValueError('Raw byte count changed')
    with RAW.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):
            md5.update(block);h.update(block)
    after=RAW.stat()
    if (h.hexdigest()!=RAW_SHA or md5.hexdigest()!=RAW_MD5 or
            (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino)):
        raise ValueError('Actual acquired source changed before parse')
    import pyreadr
    def once_reader(path):
        objects=pyreadr.read_r(path)
        final=RAW.stat()
        if (after.st_size,after.st_mtime_ns,after.st_ino)!=(final.st_size,final.st_mtime_ns,final.st_ino):
            raise ValueError('Source changed during the sole parse')
        return objects
    return parse_and_write(RAW,OUT,once_reader)

def verified_ref(config,key,*,json_body=True):
    ref=config.get(key)
    if not isinstance(ref,dict) or set(ref)!={'path','sha256'} or sha(ref['path'])!=ref['sha256']:
        raise PermissionError('Exact accepted reference required: '+key)
    return read(ref['path']) if json_body else Path(ref['path'])

def require_readout_release(config_path,release_path,trusted,stage):
    c=read(config_path)
    # Missing normal input acceptance rejects before any actual reference/body.
    if c.get('binding_state')!='BOUND_NORMAL_CONTROL_READOUT' or c.get('input_acceptance') is None or c.get('input_manifest') is None:
        raise PermissionError('PENDING_NORMAL_CONTROL_INPUT_ACCEPTANCE')
    control.digest(trusted)
    if sha(release_path)!=trusted:raise PermissionError('External root readout digest required')
    r=read(release_path)
    expected=dict(status='PRIMARY_RELEASED_TEP_NORMAL_CONTROL_READOUT',approved_by='root',
        operation='TEP_NORMAL_CONTROL_'+stage.upper()+'_READOUT',config_sha256=sha(config_path),
        allow_normal20_scoring=True,allow_fit=False,allow_recalibration=False,allow_metrics=False)
    if c.get('operation')!=expected['operation'] or any(r.get(k)!=v for k,v in expected.items()):
        raise PermissionError('Exact stage-specific normal-control release required')
    if c.get('run_ids')!=list(control.RUNS) or c.get('readouts')!=['last','mean32']:
        raise PermissionError('Fixed20runs/dualreadouts required')
    suffix='left_normal_CONTROL_seed%d_20261102_v2'%c.get('seed',0) if stage=='left' else 'tep_ridge_normal_CONTROL_20261102_v1'
    if c.get('output')!=str(ROOT/'results'/suffix) or Path(c['output']).exists():
        raise PermissionError('Distinct fresh normal-control output required')
    if stage=='left':
        from .normal_b1_readout import SEED_DEVICES
        if type(c.get('seed')) is not int or c['seed'] not in SEED_DEVICES or (c.get('physical_gpu_index'),c.get('expected_gpu_uuid'))!=SEED_DEVICES[c['seed']]:
            raise PermissionError('Fixed existing seed/GPU binding')
    required={'__init__.py','tep_faulty_inputs.py','tep_normal_control.py','tep_normal_control_runtime.py',
              'normal_b1_readout.py','left_baseline.py','tep_normal_ridge.py','tep_ridge_readout.py','tep_ridge_query.py'}
    if set(c.get('helper_hashes',{}))!=required or r.get('helper_hashes')!=c['helper_hashes']:
        raise PermissionError('Exact closed runtime helpers required')
    for name,h in c['helper_hashes'].items():
        if sha(Path(__file__).parent/name)!=h:raise PermissionError('Frozen runtime helper changed')
    accepted=verified_ref(c,'input_acceptance');manifest=verified_ref(c,'input_manifest')
    if (accepted.get('status')!='PRIMARY_ACCEPTED_LIMITED_TEP_NORMAL_CONTROL_INPUTS'
            or accepted.get('approved_by')!='root' or accepted.get('entities')!=20
            or accepted.get('input_manifest_sha256')!=c['input_manifest']['sha256']):
        raise PermissionError('Actual independent normal20 input acceptance required')
    rows=control.validate_manifest(manifest)
    if manifest['raw_sha256']!=RAW_SHA:raise PermissionError('Actual acquired FaultFreeTesting source only')
    verified_ref(c,'contract',json_body=False)
    cal_accept=verified_ref(c,'calibration_acceptance');cal=verified_ref(c,'calibration')
    if cal_accept.get('calibration_json_sha256')!=c['calibration']['sha256'] or cal.get('n')!=8020 or cal.get('rank_1based')!=7941 or cal.get('comparison')!='strict_greater':
        raise PermissionError('Original normal CAL only, never recalibrate')
    expected_keys={'last','mean32'} if stage=='left' else {m+'_'+d for m in ['SELF','SYNC','PEAK','MULTI','ZMAX'] for d in ['last','mean32']}
    if set(cal.get('thresholds',{}))!=expected_keys or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not __import__('math').isfinite(v) for v in cal['thresholds'].values()):
        raise PermissionError('Exact original finite CAL threshold keys')
    if stage=='left':
        from .normal_b1_readout import bind_checkpoint_schema
        accepted=verified_ref(c,'training_acceptance');training=verified_ref(c,'training_report')
        events=verified_ref(c,'training_events');keys=verified_ref(c,'checkpoint_schema');scaler=verified_ref(c,'scaler')
        if accepted.get('status')!='PRIMARY_ACCEPTED_LIMITED_TEP_NORMAL_TRAINING' or accepted.get('seed')!=c['seed']:
            raise PermissionError('Existing accepted normal-training seed required')
        for field,key in [('report_sha256','training_report'),('events_sha256','training_events'),('state_keys_sha256','checkpoint_schema')]:
            if accepted.get(field)!=c[key]['sha256']:raise PermissionError('Original training metadata chain differs')
        for key in ['checkpoint','scaler']:
            if any(accepted[key].get(k)!=c[key].get(k) for k in ['path','sha256']):
                raise PermissionError('Original model/scaler reference changed')
        science,extras=bind_checkpoint_schema(events,keys,training)
        if accepted['scientific_state_schema']!=science or accepted['exact_thop_extra_schema']!=extras:
            raise PermissionError('Exact actual scientific/THOP schema required')
        if (cal_accept.get('status')!='PRIMARY_ACCEPTED_LIMITED_NORMAL_CALIBRATION'
                or cal.get('seed')!=c['seed'] or cal.get('checkpoint_sha256')!=c['checkpoint']['sha256']
                or cal.get('scaler_sha256')!=c['scaler']['sha256']
                or cal_accept.get('checkpoint_sha256')!=c['checkpoint']['sha256']
                or cal_accept.get('scaler_sha256')!=c['scaler']['sha256']):
            raise PermissionError('Frozen per-seed checkpoint/scaler/CAL chain required')
        return c,rows,dict(training=training,science=science,extras=extras,scaler=scaler,calibration=cal)
    accepted=verified_ref(c,'ridge_acceptance');selection=verified_ref(c,'selection');scaler=verified_ref(c,'scaler')
    if (accepted.get('status')!='PRIMARY_ACCEPTED_LIMITED_NORMAL_RIDGE_FIT_SELECT'
            or accepted.get('coefficients_sha256')!=c['coefficients']['sha256']
            or accepted.get('model_selection_sha256')!=c['selection']['sha256']
            or accepted.get('scaler_sha256')!=c['scaler']['sha256']
            or cal_accept.get('status')!='PRIMARY_ACCEPTED_LIMITED_RIDGE_NORMAL_CALIBRATION'
            or cal.get('coefficient_sha256')!=c['coefficients']['sha256']
            or cal.get('scaler_sha256')!=c['scaler']['sha256']):
        raise PermissionError('Existing fixed Ridge parameters/scaler/CAL only')
    return c,rows,dict(selection=selection,scaler=scaler,calibration=cal)

def normal_values(row):
    import numpy as np
    if sha(row['path'])!=row['sha256']:raise ValueError('Normal input bytes changed')
    with np.load(row['path'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    return control.validate_arrays(arrays),arrays

def seal_output(out):
    files=[dict(path=str(p.relative_to(out)),sha256=sha(p),bytes=p.stat().st_size)
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='FINAL_MANIFEST.json']
    dump(out/'FINAL_MANIFEST.json',dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',files=files))

def append_saved_left_bank(banks,compacts,row,seed,bank,saved):
    """Aggregate the actual save_bank result without duplicate keyword fields."""
    record=dict(saved)
    if (record.get('run')!=row['simulationRun'] or record.get('seed')!=seed
            or record.get('role')!='NORMAL_CONTROL'):
        raise ValueError('Saved normal bank run/seed/role mismatch')
    if 'entity_id' in record and record['entity_id']!=row['entity_id']:
        raise ValueError('Saved normal bank entity mismatch')
    record['entity_id']=row['entity_id']
    banks.append(record)
    compacts.append({k:bank[k] for k in ('last','mean32','channel_last','channel_mean32','end_index_0based','available_index_0based')})

def save_left_compact(out,compacts):
    import numpy as np
    np.savez(Path(out)/'compact_scores.npz',entity_id=np.asarray(control.ENTITIES),
        **{k:np.stack([b[k] for b in compacts]) for k in compacts[0]})

def ridge_stage(config_path,release_path,trusted):
    c,rows,meta=require_readout_release(config_path,release_path,trusted,'ridge')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU only')
    from .tep_ridge_query import score_values
    import numpy as np
    if sha(c['coefficients']['path'])!=c['coefficients']['sha256']:raise ValueError('Coefficients changed')
    with np.load(c['coefficients']['path'],allow_pickle=False) as w:weights={k:w[k].copy() for k in w.files}
    out=Path(c['output']);out.mkdir(parents=False,exist_ok=False);banks=[];compact={};start=time.monotonic()
    for row in rows:
        values,axes=normal_values(row);a=score_values(values,meta['scaler'],meta['selection'],weights)
        bank=out/(row['entity_id']+'.npz');np.savez(bank,**a,**{k:v for k,v in axes.items() if k!='values'})
        banks.append(dict(entity_id=row['entity_id'],run=row['simulationRun'],path=str(bank),sha256=sha(bank)))
        for method in ('SELF','SYNC','PEAK','MULTI','ZMAX'):
            for d in ('last','mean32'):compact.setdefault(method+'_'+d,[]).append(a[method+'_'+d])
        if time.monotonic()-start>120:raise TimeoutError('Fixed120s CPU budget')
    np.savez(out/'compact_scores.npz',entity_id=np.asarray(control.ENTITIES),
        end_index_0based=np.tile(np.arange(99,960),(20,1)),**{k:np.stack(v) for k,v in compact.items()})
    dump(out/'report.json',dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',role='NORMAL_CONTROL',banks=banks,
        source_refs=c,thresholds=meta['calibration']['thresholds'],no_metrics=True,GPU_hours=0,
        compact_scores_sha256=sha(out/'compact_scores.npz')))
    seal_output(out)

def left_stage(config_path,release_path,trusted):
    start=time.perf_counter()
    c,rows,meta=require_readout_release(config_path,release_path,trusted,'left')
    from . import left_baseline as baseline
    from .normal_b1_readout import checkpoint_key_contract,save_bank
    from types import SimpleNamespace
    import numpy as np
    root=Path(c['source_root']);seed=c['seed']
    required=set(baseline.MODEL_FILES+['exp/exp_anomaly_detection.py',
        'configs/PSM/model_configs_Left.json','configs/PSM/model_configs_0.json','configs/PSM/train_configs.json']+
        ['models/CrossAD/'+name+'.py' for name in ['Basic_CrossAD','Attention_Blocks','EncDec','Context_Blocks']])
    if set(c['source_hashes'])!=required:raise PermissionError('Exact original official source closure required')
    for name,h in c['source_hashes'].items():
        if sha(root/name)!=h:raise PermissionError('Official model source changed')
    if c['upstream_commit']!=baseline.COMMIT or c['max_allocated_GiB']!=20 or c['max_seconds']!=600:
        raise PermissionError('Fixed original model/resource identity')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(c['physical_gpu_index']):raise PermissionError('Fixed seed GPU')
    if sha(c['checkpoint']['path'])!=c['checkpoint']['sha256']:raise ValueError('Checkpoint changed')
    model_cfg=dict(read(root/'configs/PSM/model_configs_Left.json'),seq_len=100,enc_in=3,warmup_steps=123,ramp_steps=246)
    if model_cfg!=meta['training']['effective_configs']['model']:raise ValueError('Actual fixed model config')
    torch,Left,loaded=baseline.model_runtime(c,root);props=torch.cuda.get_device_properties(0)
    if str(props.uuid).lower().removeprefix('gpu-')!=c['expected_gpu_uuid']:raise ValueError('GPU UUID changed')
    model=Left(SimpleNamespace(**model_cfg)).cuda()
    describe=lambda state:{k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state.items()}
    if describe(model.state_dict())!=meta['science']:raise ValueError('Fresh science schema changed')
    state=torch.load(c['checkpoint']['path'],map_location='cpu',weights_only=True)
    checkpoint_key_contract(describe(state),meta['science'],meta['extras'])
    model.load_state_dict({k:state[k] for k in meta['science']},strict=True);del state;model.eval()
    if model.step!=0 or model._curriculum_lambda()!=.3:raise ValueError('Fresh0/eval.3 required')
    def tensor_hash(items):
        h=hashlib.sha256()
        for name,value in items:
            h.update(name.encode());h.update(b'None' if value is None else value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def fingerprint():
        import random
        nstate=np.random.get_state()
        return dict(state=tensor_hash(model.state_dict().items()),gradients=tensor_hash((k,p.grad) for k,p in model.named_parameters()),
            modes={k:m.training for k,m in model.named_modules()},step=model.step,
            rng=dict(python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                numpy=hashlib.sha256(nstate[1].tobytes()+repr((nstate[0],nstate[2:])).encode()).hexdigest(),
                torch_cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()))
    def scorer(batch):
        if time.perf_counter()-start>600:raise TimeoutError('Fixed600s GPU budget')
        with torch.no_grad():
            score,aux=model.infer(torch.from_numpy(batch).to('cuda:0'),None,None,None)
            if aux is not None:raise ValueError('Nativeaux changed')
            a=score.detach().cpu().numpy().copy();torch.cuda.synchronize()
        if torch.cuda.max_memory_allocated()>20*1024**3:raise MemoryError('20GiB allocator cap')
        return a
    scaler=meta['scaler'];mean=np.asarray(scaler['mean']);scale=np.asarray(scaler['scale'])
    if (scaler['fit_ids']!=[['TEP_REACTOR','FIT',i] for i in list(range(1,21))+list(range(51,61))]
            or scaler['fit_points']!=15000 or scaler['ddof']!=0 or scaler['SELECT_used_for_fit'] is not False or scaler['columns']!=list(control.COLUMNS)):
        raise PermissionError('Original FIT-only scaler required')
    if mean.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Existing fixed scaler')
    out=Path(c['output']);out.mkdir(parents=False,exist_ok=False);banks=[];compacts=[]
    for row in rows:
        values,axes=normal_values(row)
        observed=[]
        def observed_fingerprint():
            current=fingerprint();observed.append(current);return current
        normalized=(values-mean)/scale
        bank=control.collect_normal_bank(normalized,run=row['simulationRun'],seed=seed,scorer=scorer,fingerprint=observed_fingerprint)
        bank['normalized_input_windows']=np.stack([normalized[i:i+100].astype(np.float32) for i in range(861)])
        bank.update(source_values_float64=values,**{k:v for k,v in axes.items() if k!='values'})
        saved=save_bank(out/row['entity_id'],bank,dict(role='NORMAL_CONTROL',run=row['simulationRun'],seed=seed,source=row,
            fingerprint_before=observed[0],fingerprint_after=observed[1],
            observation_granularity='full state/gradients/modes/step/RNG at both bank edges only'),pipeline_start=start)
        append_saved_left_bank(banks,compacts,row,seed,bank,saved)
    save_left_compact(out,compacts)
    dump(out/'report.json',dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',role='NORMAL_CONTROL',seed=seed,banks=banks,
        source_refs=c,loaded_modules=loaded,thresholds=meta['calibration']['thresholds'],no_metrics=True,
        best_checkpoint_step=meta['training']['best_checkpoint_python_step_at_save'],training_terminal_step=meta['training']['final_python_step'],
        fresh_inference_python_step=model.step,eval_lambda=.3,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        compact_scores_sha256=sha(out/'compact_scores.npz')))
    seal_output(out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['parse','left','ridge']);parser.add_argument('--config',required=True)
    parser.add_argument('--release',required=True);parser.add_argument('--trusted-release-sha256',required=True)
    args=parser.parse_args()
    if args.stage=='parse':return parse_stage(args.config,args.release,args.trusted_release_sha256)
    if args.stage=='left':return left_stage(args.config,args.release,args.trusted_release_sha256)
    return ridge_stage(args.config,args.release,args.trusted_release_sha256)

if __name__=='__main__':main()
