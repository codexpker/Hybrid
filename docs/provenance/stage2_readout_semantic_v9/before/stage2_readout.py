"""U-native readout assembly. All model, scoring and RNG kernels are reused."""
from pathlib import Path
import hashlib
import time
import numpy as np

from ..config import read_json,sha256
from ..protocol import score_block
from .bank import collect_forecasts,write_bank,inference_metadata
from .readout_manifest import record,json_write
from .identity import source_calibration
from .stage2_identity import canonical_sha
from .stage2_readout_identity import (STAGE,JOBS,Assets,u_key,z_key,expected_u,whitelist,runtime_identity,
    validate_definition,load_operation,approve)


def load_catalog(path,profile):
    path=Path(path)
    if (path.parent/'failure.json').exists() or not (path.parent/'summary.json').is_file():raise ValueError('Incomplete/failed catalog archive')
    c=read_json(path)
    if read_json(path.parent/'summary.json')!=c:raise ValueError('Catalog differs from completed operation summary')
    if c.get('status')!='U_READOUT_CATALOG_ALL18_ACCEPTED':raise ValueError('Unaccepted/partial catalog')
    dpath=Assets().file(c['definition'],path);d=read_json(dpath)
    a=validate_definition(d,dpath,profile)
    if c.get('models')!={k:canonical_sha(v['row']) for k,v in a['models'].items()}:raise ValueError('Catalog model identities changed')
    a['catalog_path']=Path(path);return a


def build_catalog(definition,output,profile):
    d=read_json(definition);a=validate_definition(d,definition,profile)
    c=dict(status='U_READOUT_CATALOG_ALL18_ACCEPTED',stage=STAGE,definition=record(definition),profile=profile,
        models={k:canonical_sha(v['row']) for k,v in a['models'].items()},labels_opened=False,training=False)
    json_write(output/'catalog.json',c);return c


def production_backend(profile,device):
    if profile['name']=='TEP100':
        from .physical_validation import physical_validation_spec,validate_physical_runtime,CUDA_BACKEND
        validate_physical_runtime(physical_validation_spec(dict(backend=CUDA_BACKEND,device=device),profile))
    elif device!='cpu':raise ValueError('Synthetic CPU fixture cannot become GPU')


def whole_state(model):
    """Readout freezes head too; training's frozen_hashes intentionally excludes it."""
    return {name:hashlib.sha256(str(value.dtype).encode()+str(tuple(value.shape)).encode()+value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name,value in model.state_dict().items()}


def infer(a,cells,output,*,profile,scope,device,operation_digest=None):
    import torch
    from ..api import create_predictor
    production_backend(profile,device);torch.set_num_threads(2)
    cells=whitelist(cells,a['entities'],scope);banks={}
    for cell in cells:
        started=time.perf_counter();job=cell['job_id'];seed,mode,init=JOBS[job];q=a['entities'][cell['entity']];spec=a['models'][job]
        if torch.device(device).type=='cuda':torch.cuda.set_device(device);torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
        array=Path(q['array_path'])
        a['assets'].file({'path':str(array),'sha256':q['array_sha256_from_frozen_parent']},a['base'])
        torch.manual_seed(seed)
        model=create_predictor(a['official'],mode=mode,history=100,checkpoint=spec['paths']['checkpoint'],checkpoint_metadata=spec['paths']['checkpoint_metadata'],device=device)
        frozen=whole_state(model)
        data=np.load(array,mmap_mode='r',allow_pickle=False)
        if data.shape!=(profile['points'],profile['channels']) or data.dtype!=np.float32:raise ValueError('Query shape/dtype changed')
        index=np.concatenate([np.arange(b,b+5) for b in cell['origins']]);scale=spec['source']['normal_error_scale']
        arrays,rng,times=collect_forecasts(model,data,cell['origins'],scale,mask_seed=84551,expected_index=index)
        if frozen!=whole_state(model):raise ValueError('Frozen readout state (including head/buffers) changed')
        key=u_key(job,cell['entity']);folder=output/'native'/key
        metadata=inference_metadata(key,dict(stage=STAGE,job_id=job,seed=seed,mode=mode,initialization=init,entity=cell['entity'],
            history=100,source_scale=scale,source_calibration_sha256=spec['row']['source_calibration']['sha256'],
            target_threshold_sha256=spec['row']['target_threshold']['sha256'],scope=scope,operation_manifest_sha256=operation_digest,
            readout_runtime_sha256=a.get('readout_runtime',{}).get('sha256')),
            points=len(index),blocks=len(cell['origins']),mask_seed=84551,checkpoint_sha256=spec['row']['checkpoint']['sha256'],
            query_array_sha256=q['array_sha256_from_frozen_parent'],seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
            forward_p50_seconds=float(np.median(times)),forward_p95_seconds=float(np.quantile(times,.95)))
        metadata.update(forward_timing_samples=len(times),seconds_scope='model/input-I-O/compute before bank serialization',
            forward_timing_scope='primary forward only; first2 RNG replays excluded',catalog_sha256=sha256(a['catalog_path']))
        write_bank(folder,arrays,rng,metadata)
        banks[key]=dict(bank=record(folder/'predictions_and_scores.npz'),rng=record(folder/'rng_states.npz'),metadata=record(folder/'inference.json'),
            origins=cell['origins'],whole_entity_seconds=time.perf_counter()-started)
        del model,data
    result=dict(stage=STAGE,scope=scope,status='PARTIAL_U_BANKS_LABELS_LOCKED',catalog=record(a['catalog_path']),
        whitelist=cells,banks=banks,labels_opened=False,complete_group=False,operation_manifest_sha256=operation_digest,readout_runtime=a.get('readout_runtime'))
    json_write(output/'shard.json',result);return result


def z_normal_cal(a,output,*,profile,device):
    from ..api import create_predictor
    from .stage2 import evaluate_target_cal,module_modes,rng_identity
    from .stage2_data import TargetNormal
    production_backend(profile,device);out={}
    normal_path=a['assets'].file(a['definition']['normal_manifest'],a['base'])
    normal=TargetNormal(a['normal'],a['plans'][0],a['assets'].file(a['normal']['cache'],normal_path))
    for (seed,mode),spec in sorted(a['z_models'].items()):
        normal.plan=a['plans'][seed]
        cp=a['assets'].file(spec['checkpoint'],a['source_refs_path']);meta=a['assets'].file(spec['checkpoint_metadata'],a['source_refs_path'])
        calpath=a['assets'].file(spec['calibration'],a['source_refs_path']);cal=source_calibration(calpath)
        model=create_predictor(a['official'],mode=mode,history=100,checkpoint=cp,checkpoint_metadata=meta,device=device)
        for parameter in model.parameters():parameter.requires_grad_(False)
        before=whole_state(model);flags=module_modes(model);rng_before=rng_identity(device)
        values=evaluate_target_cal(model,normal,cal['normal_error_scale'],device=device)
        if before!=whole_state(model) or flags!=module_modes(model) or rng_before!=rng_identity(device):raise ValueError('Z normal-cal changed frozen weights/modes/RNG')
        folder=output/f's{seed}'/mode;folder.mkdir(parents=True)
        np.savez_compressed(folder/'physical_cal_bank.npz',**{k:v for k,v in values.items() if k!='canonical_score'})
        np.savez_compressed(folder/'canonical.npz',**{k:values[k] for k in ('canonical_score','run','origin')})
        identity=dict(schema_version=1,role='Z_TARGET_NORMAL_THRESHOLD_ONLY',stage=STAGE,seed=seed,mode=mode,
            catalog=record(a['catalog_path']),readout_runtime=a['readout_runtime'],operation_manifest_sha256=a['operation_digest'],
            operation_config=a['operation_config'],
            backend='original_cuda_a40_float32_v1' if profile['name']=='TEP100' else 'original_cpu_float32_v1',
            checkpoint=spec['checkpoint'],checkpoint_metadata=spec['checkpoint_metadata'],source_calibration=spec['calibration'],
            source_error_scale=cal['normal_error_scale'],normal_manifest=a['definition']['normal_manifest'],normal_plan=a['definition']['normal_plans'][str(seed)],
            bank=record(folder/'physical_cal_bank.npz'),canonical=record(folder/'canonical.npz'),threshold99=float(np.quantile(values['canonical_score'],.99,method='linear')),
            parameter_training_excluded_target='TEP',threshold_target='TEP',target_threshold_fitted=True,weights_unchanged=True,
            frozen_before=before,frozen_after=whole_state(model),target_global_rescale=False,query_forwards=0,labels_opened=False)
        json_write(folder/'threshold.json',identity);validate_z_threshold(folder/'threshold.json',a,seed,mode)
        out[f's{seed}/{mode}']=record(folder/'threshold.json');del model
    result=dict(status='ALL9_Z_NORMAL_THRESHOLDS_LABELS_LOCKED',stage=STAGE,catalog=record(a['catalog_path']),thresholds=out,query_forwards=0,labels_opened=False)
    json_write(output/'z_thresholds.json',result);return result


def validate_z_threshold(path,a,seed,mode):
    t=read_json(path);parent=a['z_models'][seed,mode];assets=a['assets']
    required={'schema_version','role','stage','seed','mode','checkpoint','checkpoint_metadata','source_calibration','source_error_scale',
        'normal_manifest','normal_plan','bank','canonical','threshold99','parameter_training_excluded_target','threshold_target',
        'target_threshold_fitted','weights_unchanged','frozen_before','frozen_after','target_global_rescale','query_forwards','labels_opened',
        'catalog','readout_runtime','operation_manifest_sha256','backend','operation_config'}
    if (set(t)!=required or t['schema_version']!=1 or t['role']!='Z_TARGET_NORMAL_THRESHOLD_ONLY' or t['stage']!=STAGE
            or t['seed']!=seed or t['mode']!=mode or t['parameter_training_excluded_target']!='TEP' or t['threshold_target']!='TEP'
            or t['target_threshold_fitted'] is not True or t['weights_unchanged'] is not True or t['target_global_rescale'] is not False
            or t['query_forwards']!=0 or t['labels_opened'] is not False or t['frozen_before']!=t['frozen_after']
            or t['catalog']!=record(a['catalog_path'])):raise ValueError('Wrong Z normal-threshold role/permission/fitted identity')
    if read_json(assets.file(t['readout_runtime'],path))!=runtime_identity():raise ValueError('Z normal-cal readout runtime changed')
    original_config=assets.file(t['operation_config'],path);cfg,paths,_,_,digest=load_operation(original_config)
    approved=approve(cfg,paths,Path(path).parent,digest,original_config);approved.recheck()
    if (cfg['operation']!='z-normal-cal' or digest!=t['operation_manifest_sha256'] or record(paths['catalog'])!=t['catalog']
            or cfg['parameters']['runtime']['sha256']!=t['readout_runtime']['sha256']):raise ValueError('Z normal-cal operation/approval binding mismatch')
    if t['backend']!=('original_cuda_a40_float32_v1' if a['definition']['profile']['name']=='TEP100' else 'original_cpu_float32_v1'):raise ValueError('Z normal-cal backend changed')
    for field in ('checkpoint','checkpoint_metadata','source_calibration'):
        original=parent['calibration' if field=='source_calibration' else field]
        if assets.file(t[field],path)!=assets.file(original,a['source_refs_path']):raise ValueError('Z checkpoint/source calibration replaced')
    source=source_calibration(assets.file(t['source_calibration'],path))
    if t['source_error_scale']!=source['normal_error_scale']:raise ValueError('Z source scale cannot be refitted')
    for field,key in [('normal_manifest','normal_manifest'),('normal_plan',None)]:
        ref=a['definition'][key] if key else a['definition']['normal_plans'][str(seed)]
        if assets.file(t[field],path)!=assets.file(ref,a['base']):raise ValueError('Z normal-cal role/plan identity changed')
    plan=a['plans'][seed]
    normal_manifest_path=assets.file(t['normal_manifest'],path)
    normal_data=np.load(assets.file(a['normal']['cache'],normal_manifest_path),mmap_mode='r',allow_pickle=False)
    actual_target=np.stack([normal_data[int(run)-1,int(origin):int(origin)+5] for run,origin in zip(plan['cal_run'],plan['cal_origin'])])
    with np.load(assets.file(t['bank'],path),allow_pickle=False) as bank:
        if set(bank.files)!={'prediction','target','history_scale','run','origin'} or any(not np.array_equal(bank[k],plan['cal_'+k]) for k in ('run','origin')):raise ValueError('Z normal-cal plan order changed')
        if any(bank[k].shape!=shape or bank[k].dtype!=np.float32 for k,shape in [('prediction',(256,5,52)),('target',(256,5,52)),('history_scale',(256,1,52))]):raise ValueError('Z normal-cal physical bank shape/dtype')
        if not np.array_equal(bank['target'],actual_target):raise ValueError('Z normal-cal target differs from actual frozen normal values/plan')
        expected=np.stack([score_block(p,y,s,source['normal_error_scale']) for p,y,s in zip(bank['prediction'],bank['target'],bank['history_scale'])])
    with np.load(assets.file(t['canonical'],path),allow_pickle=False) as scores:
        if (set(scores.files)!={'canonical_score','run','origin'} or any(not np.array_equal(scores[k],plan['cal_'+k]) for k in ('run','origin'))
                or scores['canonical_score'].dtype!=np.float32 or not np.array_equal(scores['canonical_score'],expected)
                or t['threshold99']!=float(np.quantile(expected,.99,method='linear'))):raise ValueError('Z canonical/threshold not closed to physical bank and original source units')
    return t


def physical_bridge(a,profile):
    """Use actual old Z banks, not an assumed extra reference artifact."""
    assets=a['assets'];d=a['definition'];p=assets.file(d['physical_bridge'],a['base']);b=read_json(p)
    old_path=assets.file(d['z_seal'],a['base']);old=read_json(old_path,max_bytes=32*1024*1024)
    if (b.get('status')!='PRIMARY_ACCEPTED_Z_PHYSICAL_REFERENCE_BRIDGE' or b.get('profile')!=profile
            or b.get('z_seal')!=d['z_seal'] or b.get('query_manifest')!=d['query_manifest']
            or old.get('status')!='ALL_FIXED_SCORES_SEALED_LABELS_LOCKED' or old.get('labels_opened') is not False
            or old.get('complete_group') is not True or old.get('query_manifest')!=d['query_manifest']
            or old.get('catalog')!=d['z_catalog'] or old.get('profile')!=profile):raise ValueError('HOLD: complete accepted old Z physical bridge unavailable')
    if type(b.get('gold_seed')) is not int or b['gold_seed']!=0 or b.get('gold_mode')!='hybrid':
        raise ValueError('Physical bridge gold must be the predeclared seed0/hybrid; no reference reselection')
    gold_seed,gold_mode=b['gold_seed'],b['gold_mode']
    expected={z_key(s,m,e) for s in range(3) for m in ('temporal','tf','hybrid') for e in a['entities']}
    from .readout_manifest import expected_scores
    if set(old['native_banks'])!=expected or set(old['scores'])!=expected_scores(a['entities']):raise ValueError('Old Z all-bank/score reference set incomplete')
    for value in old['native_banks'].values():
        for part in ('bank','rng','metadata'):assets.file(value[part],old_path)
    for value in old['scores'].values():assets.file(value['file'],old_path)
    if profile['name']=='TEP100':
        from .physical_validation import CUDA_BACKEND
        if (b.get('backend')!=CUDA_BACKEND or old.get('physical_validation')!={'backend':CUDA_BACKEND,'device':'cuda:0'}
                or b.get('execution_evidence')!=old.get('execution_evidence')):raise ValueError('Old Z backend/execution provenance not bound')
        assets.file(b['independent_physical_audit'],p);assets.file(b['execution_evidence'],old_path)
    references={}
    if set(b.get('entities',{}))!=set(a['entities']):raise ValueError('Physical reference lacks actual entity banks')
    for entity,q in a['entities'].items():
        descriptor=old['native_banks'][z_key(gold_seed,gold_mode,entity)]
        if b['entities'][entity]!=descriptor:raise ValueError('Reference is not the fixed original Z seed0 hybrid bank')
        meta=read_json(assets.file(descriptor['metadata'],old_path))
        if (meta.get('query_array_sha256')!=q['array_sha256_from_frozen_parent'] or meta.get('history')!=100 or meta.get('mask_seed')!=84551
                or meta.get('checkpoint_sha256')!=a['z_models'][gold_seed,gold_mode]['checkpoint']['sha256']
                or meta.get('bank_sha256')!=descriptor['bank']['sha256'] or meta.get('seed')!=gold_seed or meta.get('mode')!=gold_mode
                or meta.get('entity')!=entity):raise ValueError('Z goldbank physical source/history identity mismatch')
        data_path=Path(q['array_path'])
        assets.file({'path':str(data_path),'sha256':q['array_sha256_from_frozen_parent']},a['base'])
        data=np.load(data_path,mmap_mode='r',allow_pickle=False);index=np.concatenate([np.arange(o,o+5) for o in q['origins']])
        if data.shape!=(profile['points'],profile['channels']) or data.dtype!=np.float32 or not np.isfinite(data).all():raise ValueError('Original query shape/dtype/finite contract')
        with np.load(assets.file(descriptor['bank'],old_path),allow_pickle=False) as gold:
            scale=gold['history_scale_by_block'].copy()
            if (not np.array_equal(gold['time_index'],index) or not np.array_equal(gold['forecast_origin'],q['origins'])
                    or not np.array_equal(gold['target'],data[index]) or scale.shape!=(len(q['origins']),1,profile['channels'])
                    or not np.isfinite(scale).all() or np.any(scale<=0)):raise ValueError('Original Z physical reference invalid')
        references[entity]=(data,index,scale)
    return references,old


def check_executions(executions,shards,base,profile):
    if profile['name']=='SYNTHETIC':
        if executions!={'scope':'SYNTHETIC_CPU_ENGINEERING_ONLY'}:raise ValueError('Synthetic executions cannot claim production evidence')
        return
    if set(executions)!={sha256(p) for p in shards}:raise ValueError('Missing actual execution receipts')
    from .physical_validation import CUDA_RUNTIME
    for p in shards:
        entry=executions[sha256(p)];a=Assets();launch=read_json(a.file(entry['launch'],base));oswait=read_json(a.file(entry['os_wait'],base));resource=read_json(a.file(entry['resource'],base));shard=read_json(p)
        if (oswait.get('actual_OS_wait_exit_code')!=0 or resource.get('status')!='COMPLETED_PENDING_INDEPENDENT_REVIEW'
                or launch.get('operation_manifest_sha256')!=shard['operation_manifest_sha256']
                or resource.get('summary_sha256')!=sha256(p) or not oswait.get('process_seconds',0)>0):raise ValueError('Actual U execution/OS/shard evidence mismatch')
        runtime=resource['current_runtime']
        if (runtime.get('torch')!=CUDA_RUNTIME['torch_version'] or runtime.get('cuda')!=CUDA_RUNTIME['cuda_version'] or runtime.get('name')!=CUDA_RUNTIME['gpu_name']
                or runtime.get('shared_sha256')!=sha256(__file__)):raise ValueError('Actual U production backend/source mismatch')


def seal(a,shard_refs,z_thresholds,executions,output,*,profile,write=True):
    from .multi_readout import verify_physical_bank
    references,old=physical_bridge(a,profile);assets=a['assets'];banks={};shards=[]
    for ref in shard_refs:
        p=assets.file(ref,a['base']);shards.append(p);shard=read_json(p,max_bytes=64*1024*1024)
        if (p.parent/'failure.json').exists() or not (p.parent/'summary.json').is_file():raise ValueError('Incomplete/failed U shard archive')
        # Rehashed descriptors can be independently restaged, but a real
        # operation's terminal summary must still match the selected shard.
        if read_json(p.parent/'summary.json',max_bytes=64*1024*1024)!=shard:raise ValueError('Shard differs from its terminal summary')
        if (shard.get('stage')!=STAGE or shard.get('scope')!='shard' or shard.get('complete_group') is not False
                or shard.get('labels_opened') is not False or shard.get('catalog')!=record(a['catalog_path'])):raise ValueError('L0/foreign shard cannot seal')
        if read_json(assets.file(shard['readout_runtime'],p))!=runtime_identity():raise ValueError('Shard readout runtime changed')
        cells=whitelist(shard['whitelist'],a['entities'],'shard')
        if set(shard['banks'])!={u_key(c['job_id'],c['entity']) for c in cells}:raise ValueError('Whitelist/output key mismatch')
        for key,value in shard['banks'].items():
            if key in banks:raise ValueError('Duplicate bank across shards')
            cell=next(c for c in cells if u_key(c['job_id'],c['entity'])==key);job=cell['job_id'];entity=cell['entity'];spec=a['models'][job]
            bp=assets.file(value['bank'],p);meta=read_json(assets.file(value['metadata'],p));rp=assets.file(value['rng'],p)
            expected=dict(stage=STAGE,job_id=job,entity=entity,history=100,mask_seed=84551,scope='shard',
                checkpoint_sha256=spec['row']['checkpoint']['sha256'],source_scale=spec['source']['normal_error_scale'],
                source_calibration_sha256=spec['row']['source_calibration']['sha256'],target_threshold_sha256=spec['row']['target_threshold']['sha256'],
                operation_manifest_sha256=shard['operation_manifest_sha256'],readout_runtime_sha256=shard['readout_runtime']['sha256'],
                catalog_sha256=sha256(a['catalog_path']),query_array_sha256=a['entities'][entity]['array_sha256_from_frozen_parent'])
            if meta.get('bank_sha256')!=sha256(bp) or meta.get('labels_opened') is not False or any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('U bank fitted/score/physical metadata mismatch')
            verify_physical_bank(bp,a['entities'][entity],profile,references[entity]);index=references[entity][1]
            with np.load(rp,allow_pickle=False) as rng:
                if rng['before_each_block'].dtype!=np.uint8 or rng['before_each_block'].ndim!=2 or len(rng['before_each_block'])!=len(cell['origins']):raise ValueError('Missing exact per-block RNG states')
            with np.load(bp,allow_pickle=False) as bank:
                shape=(len(index),profile['channels'])
                if any(bank[k].dtype!=np.float32 or bank[k].shape!=shape or not np.isfinite(bank[k]).all() for k in ('prediction','target','normalized_channel_residual')):raise ValueError('Native float32 physical bank invalid')
                if (bank['history_scale_by_block'].dtype!=np.float32 or bank['valid'].dtype!=np.bool_
                        or any(bank[k].dtype!=np.float32 or bank[k].shape!=(len(index),) for k in ('raw_score','standardized_score'))):raise ValueError('Canonical float32 units/valid dtype mismatch')
                expanded=np.repeat(bank['history_scale_by_block'],5,axis=1).reshape(shape);residual=np.abs(bank['prediction']-bank['target'])/expanded
                if (not np.array_equal(residual,bank['normalized_channel_residual']) or not np.array_equal(bank['raw_score'],residual.mean(-1))
                        or not np.array_equal(bank['standardized_score'],bank['raw_score']/spec['source']['normal_error_scale'])
                        or not np.array_equal(bank['valid'],np.ones(len(index),bool))):raise ValueError('Canonical score not closed to physical bank/source units')
            banks[key]={**value,'job_id':job,'entity':entity,'canonical':dict(file=value['bank'],score_field='standardized_score',index_field='time_index'),
                'tracks':{'source':{'score_object':dict(file=value['bank'],score_field='standardized_score',index_field='time_index'),'threshold':spec['source']['threshold99']},
                          'target':{'score_object':dict(file=value['bank'],score_field='standardized_score',index_field='time_index'),'threshold':spec['tracks']['target']['threshold']}}}
    if set(banks)!=expected_u(a['entities']):raise ValueError('INCOMPLETE: all7200 U banks (or full synthetic grid) required')
    check_executions(executions,shards,a['base'],profile)
    z=read_json(assets.file(z_thresholds,a['base']))
    zp=Path(z_thresholds['path'])
    if (zp.parent/'failure.json').exists() or not (zp.parent/'summary.json').is_file() or read_json(zp.parent/'summary.json')!=z:raise ValueError('Incomplete/failed Z normal-cal archive')
    expected_z={f's{s}/{m}' for s in range(3) for m in ('temporal','tf','hybrid')}
    if z.get('catalog')!=record(a['catalog_path']) or set(z.get('thresholds',{}))!=expected_z:raise ValueError('All9 Z normal thresholds required')
    for seed in range(3):
        for mode in ('temporal','tf','hybrid'):validate_z_threshold(assets.file(z['thresholds'][f's{seed}/{mode}'],z_thresholds['path']),a,seed,mode)
    result=dict(status='ALL_U_CANONICAL_SCORES_SEALED_LABELS_LOCKED',stage=STAGE,complete_group=True,labels_opened=False,
        catalog=record(a['catalog_path']),shards=shard_refs,z_thresholds=z_thresholds,executions=executions,u_banks=banks,
        z_seal=a['definition']['z_seal'],physical_bridge=a['definition']['physical_bridge'],counts=dict(u_banks=len(banks),z_native_banks=len(old['native_banks']),z_fixed_scores=len(old['scores'])),
        z_query_new_forwards=0,overall_Aff_F1=None)
    assets.recheck()
    if write:json_write(output/'seal.json',result)
    return result


def gate(path,a,output,*,profile):
    value=read_json(path,max_bytes=64*1024*1024)
    if (Path(path).parent/'failure.json').exists():raise ValueError('Failed archive cannot become a complete seal')
    if value.get('status')!='ALL_U_CANONICAL_SCORES_SEALED_LABELS_LOCKED' or value.get('stage')!=STAGE:raise ValueError('No complete U seal')
    # Re-derive the same closed seal before returning eligibility. No label loader.
    recreated=seal(a,value['shards'],value['z_thresholds'],value['executions'],output,profile=profile,write=False)
    if recreated!=value:raise ValueError('Sealed descriptors/threshold tracks changed')
    return dict(status='COMPLETE_SEAL_ELIGIBLE_FOR_SEPARATE_PRIMARY_LABEL_RELEASE',labels_permission=False,seal=record(path),counts=value['counts'])


def run_readout(config_path,*,execute=False):
    cfg,paths,output,manifest,digest=load_operation(config_path)
    if not execute:return dict(executed=False,operation_manifest=manifest,operation_manifest_sha256=digest,scope='preview only; no assets/model/labels opened')
    assets=approve(cfg,paths,output,digest,config_path)
    from .stage2 import _snapshot,_finish
    code=_snapshot(output,cfg,manifest,{'assets':assets})
    try:
        profile=cfg['parameters']['profile'];op=cfg['operation']
        if op=='catalog':result=build_catalog(paths['definition'],output,profile)
        else:
            a=load_catalog(paths['catalog'],profile)
            a['readout_runtime']=record(Assets().file(cfg['parameters']['runtime'],config_path));a['operation_digest']=digest;a['operation_config']=record(config_path)
            if op=='z-normal-cal':result=z_normal_cal(a,output,profile=profile,device=cfg['parameters']['device'])
            elif op=='infer':
                result=infer(a,read_json(paths['whitelist']),output,profile=profile,scope=cfg['parameters']['scope'],device=cfg['parameters']['device'],operation_digest=digest)
            elif op=='seal':
                shard_refs=[record(Assets().file(ref,paths['shards'])) for ref in read_json(paths['shards'])]
                result=seal(a,shard_refs,record(paths['z_thresholds']),read_json(paths['executions']),output,profile=profile)
            else:result=gate(paths['seal'],a,output,profile=profile)
            a['assets'].recheck()
        _finish({'assets':assets},code);json_write(output/'summary.json',result);return result
    except Exception as exc:
        json_write(output/'failure.json',dict(status='FAILED_NO_RELEASE',error=str(exc)));raise
