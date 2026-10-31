"""Bounded, opt-in source L0 diagnostic; never writes training checkpoints.

This module calls the production batch/optimizer kernels for exactly five
updates, plus at most two source missingness checks. It cannot authorize train
or read any target query. Run as ``python -m hybrid_tsad.engine.source_gate``.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from ..config import OFFICIAL_SHA, read_json, resolve_path, sha256
from .identity import operation_manifest
from .pipeline import load_operation


def prepare(config_path):
    cfg=read_json(config_path)
    expected={'diagnostic_schema_version','purpose','contract','approval','source_operations',
              'cases','package_code','parameters','output'}
    if set(cfg)!=expected or cfg['diagnostic_schema_version']!=1 or cfg['purpose']!='source_gate_l0':
        raise ValueError('Expected bounded source_gate_l0 diagnostic schema1')
    if cfg['parameters'].keys()!={'optimizer_steps','batch','device','seed_by_job','mask_seed',
                                'max_seconds_per_process','max_total_process_GPU_hours'}:
        raise ValueError('Unknown diagnostic parameter')
    if cfg['parameters']['optimizer_steps']!=5 or cfg['parameters']['batch']!=64 or cfg['parameters']['mask_seed']!=84551:
        raise ValueError('Diagnostic is fixed to five batch64 updates and mask84551')
    if cfg['parameters']['max_seconds_per_process']!=120 or cfg['parameters']['max_total_process_GPU_hours']!=.1:
        raise ValueError('Source L0 budget is fixed to120seconds per3 processes,0.1GPUhours total')
    if len(cfg['source_operations'])!=3 or set(cfg['parameters']['seed_by_job'])!=set(cfg['source_operations']):
        raise ValueError('Exactly three named source modes and explicit job seeds required')
    records={k:cfg[k] for k in ('contract','approval','cases')}
    records.update({f'op:{k}':v for k,v in cfg['source_operations'].items()})
    paths={}
    for name,r in records.items():
        if set(r)!={'path','sha256'} or len(r['sha256'])!=64:
            raise ValueError('Diagnostic assets require exact path/SHA256')
        paths[name]=resolve_path(r['path'],config_path)
    output=resolve_path(cfg['output'],config_path)
    code={str(resolve_path(p,config_path)):h for p,h in cfg['package_code'].items()}
    source_root=Path(__file__).resolve().parents[1]
    required_code={str(p.resolve()) for p in source_root.rglob('*.py')}
    if not required_code<=set(code) or any(not isinstance(h,str) or len(h)!=64 for h in code.values()):
        raise ValueError('Diagnostic must bind its actual kernels and unchanged core code SHA256')
    manifest={'diagnostic_schema_version':1,'purpose':cfg['purpose'],
        'inputs':{name:{'path':str(paths[name]),'sha256':records[name]['sha256']}
                  for name in records if name!='approval'},
        'package_code':code,'parameters':cfg['parameters'],'output':str(output)}
    encoded=json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False)
    digest=hashlib.sha256(encoded.encode()).hexdigest()
    return cfg,records,paths,output,manifest,digest


def validate_inputs(config_path,job):
    cfg,records,paths,output,manifest,digest=prepare(config_path)
    for name,p in paths.items():
        if not p.is_file() or sha256(p)!=records[name]['sha256']:
            raise ValueError('Source diagnostic identity mismatch: '+name)
    approval=read_json(paths['approval'])
    if (approval.get('status')!='APPROVED' or approval.get('scope')!='SOURCE_L0_ONLY'
        or approval.get('contract_sha256')!=records['contract']['sha256']
        or approval.get('diagnostic_manifest_sha256')!=digest
        or approval.get('training_released') is not False):
        raise ValueError('Explicit SOURCE_L0_ONLY approval required; never a training release')
    for p,h in manifest['package_code'].items():
        if sha256(p)!=h:raise ValueError('Diagnostic package code identity changed')
    if job not in cfg['source_operations']:raise ValueError('Only the approved diagnostic jobs may execute')
    if set(approval.get('allowed_jobs',[]))!=set(cfg['source_operations']):
        raise ValueError('Source L0 approval must name all three allowed jobs')
    if (output/job).exists():raise ValueError('Diagnostic job output must be a new directory')
    operations={};modes=set()
    for job in cfg['source_operations']:
        op,assets,op_paths,op_output=load_operation(paths['op:'+job],'train')
        if op['engine_schema_version']!=3 or op['parameters']['job']!=job:
            raise ValueError('Source operation job/schema mismatch')
        for name,p in op_paths.items():
            if not p.is_file() or sha256(p)!=assets[name]['sha256']:
                raise ValueError('Source operation input identity mismatch: '+name)
        if assets['contract']['sha256']!=records['contract']['sha256']:
            raise ValueError('Diagnostic and source contracts differ')
        wave=read_json(op_paths['wave_config']);inputs=read_json(op_paths['source_inputs'])
        recipe={'steps':10000,'batch':64,'lr':.0003,'weight_decay':.01,'warmup':500,'validate_every':500}
        if any(wave.get(k)!=v for k,v in recipe.items()) or wave.get('query_evaluation') is not False:
            raise ValueError('Source recipe differs from fixed production recipe')
        history=wave['history'];spec=wave['jobs'][job]
        if history not in (100,400) or wave.get('horizon',5)!=5 or spec['mode'] not in ('temporal','tf','hybrid'):
            raise ValueError('Unsupported source history/horizon/mode')
        actual_seed=spec.get('seed',wave.get('seed',0))
        if type(actual_seed) is not int or actual_seed!=cfg['parameters']['seed_by_job'][job]:
            raise ValueError('Diagnostic seed differs from authoritative source job seed')
        if sha256(op_paths['official'])!=OFFICIAL_SHA:
            raise ValueError('Official source identity changed')
        if Path(inputs['manifest']).resolve()!=op_paths['source_manifest'] or Path(inputs['official']).resolve()!=op_paths['official']:
            raise ValueError('Source input paths differ')
        if any(sha256(p)!=h for p,h in inputs['hashes'].items()):
            raise ValueError('Original source input identity changed')
        rows=read_json(op_paths['rows']);source=read_json(op_paths['source_manifest'])
        from ..data import canonical_domain
        target=canonical_domain(wave['target'])
        expected_rows=[r for r in source['records'] if r['dataset'] in spec['sources'] and canonical_domain(r['dataset'])!=target]
        if rows!=expected_rows or target in [canonical_domain(x) for x in spec['sources']]:
            raise ValueError('Source rows/target exclusion mismatch')
        if any(r['history']!=history or r['horizon']!=5 for r in rows):
            raise ValueError('Source rows disagree with history/horizon')
        if op_output.exists():raise ValueError('Cannot diagnose over an existing formal training job')
        modes.add(spec['mode']);operations[job]=(wave,inputs,spec,rows,op_paths)
    if modes!={'temporal','tf','hybrid'}:raise ValueError('Three distinct source modes required')
    return cfg,paths,output,manifest,digest,operations


def _create_model(official,history,mode,device):
    from ..model import HybridPredictor
    return HybridPredictor(official,history=history,mode=mode).to(device)


def run(config_path,*,execute=False,job=None):
    invoked=time.monotonic()
    cfg,_,_,output,manifest,digest=prepare(config_path)
    if not execute:
        return {'executed':False,'diagnostic_manifest':manifest,'diagnostic_manifest_sha256':digest,
                'scope':'source-only bounded L0 preview; no data/model/target opened'}
    cfg,paths,output,manifest,digest,operations=validate_inputs(config_path,job)
    import numpy as np
    import torch
    from ..data import ManifestSampler
    from ..model import objective
    from .source import read_plan
    from .training import optimizer_step,source_batch,evaluate_source
    device=torch.device(cfg['parameters']['device']);is_cuda=device.type=='cuda'
    if device.type not in ('cpu','cuda'):raise ValueError('Only CPU/CUDA diagnostic devices supported')
    cases=read_json(paths['cases'])
    if set(cases)!={'history','horizon','first5_batch_slices','source_mask_cases'}:
        raise ValueError('Unknown source cases; target/query cases are forbidden')
    if cases['horizon']!=5 or cases['first5_batch_slices']!=[[i*64,(i+1)*64] for i in range(5)]:
        raise ValueError('Source diagnostic batches changed')
    if set(cases['source_mask_cases'])-{'first_internal_history_missing','first_partial_future_missing'}:
        raise ValueError('Unknown extra source diagnostic case')
    torch.set_num_threads(2);results=[];started=time.time()
    output.mkdir(parents=True,exist_ok=True)
    output=output/job;output.mkdir(exist_ok=False)
    for job,(wave,inputs,spec,rows,op_paths) in [(job,operations[job])]:
        history=wave['history']
        if cases['history']!=history:raise ValueError('Case history disagrees with source operations')
        sampler=ManifestSampler(inputs['manifest'],exclude_dataset=wave['target'])
        plan=read_plan(op_paths['plan'])
        for role,n in (('fit',640000),('select',256),('cal',256)):
            for key,dtype in (('record','uint16'),('origin','uint32'),('channel','uint8')):
                if plan[role+'_'+key].shape!=(n,) or plan[role+'_'+key].dtype!=np.dtype(dtype):
                    raise ValueError('Diagnostic plan exposure/dtype mismatch')
            ri,bb,cc=[plan[role+'_'+k] for k in ('record','origin','channel')]
            for j in np.unique(ri):
                if int(j)>=len(rows) or rows[int(j)]['role']!=role:
                    raise ValueError('Source diagnostic plan role mismatch')
                row=rows[int(j)];sel=ri==j;b=bb[sel].astype('int64');c=cc[sel].astype('int64')
                if (not np.isin(b,row['forecast_origins']).all() or (c>=len(row['columns'])).any()
                    or (b-history<row['local_start']).any() or (b+5>row['local_end_exclusive']).any()):
                    raise ValueError('Source diagnostic plan footprint/channel mismatch')
            domains=np.array([rows[int(j)]['dataset'] for j in ri])
            expected=np.asarray(spec['sources'])[(np.arange(n)//64)%len(spec['sources'])]
            if not np.array_equal(domains,expected):raise ValueError('Fixed source batch domain order changed')
        seed=cfg['parameters']['seed_by_job'][job]
        torch.manual_seed(seed);np.random.seed(seed)
        model=_create_model(op_paths['official'],history,spec['mode'],device)
        initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        active=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(active,lr=.0003,weight_decay=.01)
        if is_cuda:torch.cuda.reset_peak_memory_stats(device)
        curve=[];clock=time.time()
        for step in range(1,6):
            if time.monotonic()-invoked>120:raise ValueError('Source L0 process time budget exceeded')
            if is_cuda:torch.cuda.synchronize(device)
            whole_start=time.perf_counter()
            x,y=source_batch(sampler,plan,rows,'fit',(step-1)*64,device=device,history=history)
            if x.shape!=(64,history,1) or y.shape!=(64,5,1):raise ValueError('Source batch shape mismatch')
            if is_cuda:torch.cuda.synchronize(device)
            start=time.perf_counter()
            loss,parts,lr,gn=optimizer_step(model,optimizer,x,y,objective,step)
            if is_cuda:torch.cuda.synchronize(device)
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
                raise ValueError('Active gradient missing/nonfinite')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError('Inactive gradient present')
            curve.append(dict(step=step,lr=lr,loss=float(loss.detach()),future=float(parts['future'].detach()),
                              tf=float(parts['tf'].detach()),gradient_norm=gn,
                              optimizer_step_seconds=time.perf_counter()-start,
                              whole_step_seconds=time.perf_counter()-whole_start))
            if is_cuda and torch.cuda.max_memory_allocated(device)>20*1024**3:
                raise ValueError('Source diagnostic exceeded20GiB allocated')
        select_started=time.perf_counter()
        select_loss,select_arrays=evaluate_source(model,sampler,plan,rows,'select',objective,wave,
                                                 device=device,history=history)
        if is_cuda:torch.cuda.synchronize(device)
        select_seconds=time.perf_counter()-select_started
        np.savez_compressed(output/'diagnostic_select_after5.npz',**select_arrays)
        mask_results=[]
        devices=[device.index if device.index is not None else torch.cuda.current_device()] if is_cuda else []
        for kind,case in cases['source_mask_cases'].items():
            if time.monotonic()-invoked>120:raise ValueError('Source L0 process time budget exceeded')
            i=case['original_plan_index']
            if type(i) is not int or not 0<=i<640000:raise ValueError('Mask plan index outside fixed source fit')
            ri=int(plan['fit_record'][i]);b=int(plan['fit_origin'][i]);c=int(plan['fit_channel'][i])
            if (ri,b,c)!=(case['record'],case['origin'],case['channel']):raise ValueError('Source mask case identity changed')
            row=rows[ri]
            if row['role']!='fit':raise ValueError('Mask diagnostic must come only from source fit')
            ar=sampler.array(row);xx=ar[b-history:b,c];yy=ar[b:b+5,c]
            if not np.isfinite(xx[0]) or np.isinf(xx).any() or np.isinf(yy).any() or not np.isfinite(yy).any():
                raise ValueError('Invalid source case cannot be relaxed')
            if kind=='first_internal_history_missing' and np.isfinite(xx).all():raise ValueError('Missingness case has no internal missing')
            if kind=='first_partial_future_missing' and np.isfinite(yy).all():raise ValueError('Partial target case has no missing')
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(84551);model.train();model.zero_grad(set_to_none=True)
                x=torch.tensor(xx[None,:,None].copy(),device=device);y=torch.tensor(yy[None,:,None].copy(),device=device)
                o=model(x);loss,_=objective(o,y);loss.backward()
                if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
                    raise ValueError('Source missingness mask has nonfinite loss/gradient')
            mask_results.append(dict(kind=kind,plan_index=i,finite_history=int(np.isfinite(xx).sum()),
                                     finite_future=int(np.isfinite(yy).sum()),loss=float(loss.detach()),optimizer_updated=False))
        changes={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in model.named_parameters()}
        if not any(changes[n]>0 for n,p in model.named_parameters() if p.requires_grad):raise ValueError('No actual active update')
        if any(changes[n]!=0 for n,p in model.named_parameters() if not p.requires_grad):raise ValueError('Inactive parameter changed')
        peak=torch.cuda.max_memory_allocated(device) if is_cuda else None
        if is_cuda and peak>20*1024**3:raise ValueError('Source mask diagnostic exceeded20GiB allocated')
        results.append(dict(job=job,mode=spec['mode'],seed=seed,history=history,optimizer_steps=5,
            exposure_windows=320,model_discarded=True,curve=curve,mask_checks=mask_results,
            peak_allocated_bytes=peak,seconds=time.time()-clock,parameter_changes=changes,
            parameter_counts=model.parameter_counts(),source_select_after5_loss=select_loss,
            source_select_256_windows_seconds=select_seconds,
            estimate_10k_seconds=float(np.median([r['whole_step_seconds'] for r in curve[2:]])*10000+select_seconds*22),
            estimate_caveat='three steady steps only;22 source-evaluation passes approximated by one select measurement; not a runtime guarantee'))
        del model,optimizer
        if is_cuda:torch.cuda.empty_cache()
    for path,h in manifest['package_code'].items():
        if sha256(path)!=h:raise ValueError('Package changed during source diagnostic')
    if time.monotonic()-invoked>120:raise ValueError('Source L0 process time budget exceeded')
    report=dict(status='SOURCE_L0_PASS_AWAIT_PRIMARY_TRAIN_RELEASE' if is_cuda else 'CPU_ONLY_NOT_A_GPU_ENTRY_GATE',
        diagnostic_manifest_sha256=digest,results=results,seconds=time.time()-started,
        process_seconds=time.monotonic()-invoked,
        GPU_executed=is_cuda,training_released=False,formal_checkpoints_written=False,
        target_values_or_labels_read=False,scope='Bounded source diagnostic; all weights discarded')
    (output/'source_gate_report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--job',help='One of the three manifest-bound jobs; one bounded process per mode')
    args=p.parse_args(argv)
    try:print(json.dumps(run(args.config,execute=args.execute,job=args.job),indent=2,allow_nan=False))
    except (OSError,ValueError,KeyError,TypeError) as e:
        print(json.dumps({'valid':False,'error':str(e)}));return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
