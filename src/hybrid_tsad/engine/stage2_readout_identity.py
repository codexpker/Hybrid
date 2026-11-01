"""Schema6 identities, separate from immutable source/Z and U-training policy."""
from pathlib import Path
import re

from ..config import read_json, resolve_path, sha256, OFFICIAL_SHA
from .identity import SCORE_SPACE, operation_manifest
from .stage2_identity import Assets, canonical_sha
from .readout_manifest import entity_profile, record

STAGE='U_NORMAL_HEAD_READOUT'
CONTRACT_SHA='56f4106f9c042fd20ec969b0dfd4535d7f9c3827dd382cc9472213c25471327d'
QUERY_SHA='25a3349e1debc419fe00a5b5b93e8340cbb93d45bf510126cedb8fb4aca93213'
PROFILE_SHA='0058a545ea95c6080e504e6cb02633e452c454c6bff3ce8a332724e9f7d0eb56'
SOURCE_SHA='095bd53b7b9a5faf04428270c635c28264094359afc6f2b5743497f287640236'
TRAIN_ENTRY_SHA='9579090dc13ee368c58c2cb4b96745ff7667bf68105c2028a00719ed1b9dd6da'
TRAIN_RECIPE_SHA='fcecfa04f5f24f2cb98a40c4d96991ee8d45b87d5dfc73aff3daf876d56b2b78'
MODES=('temporal','tf','hybrid')
INITS=('source_pretrained','random_init')
JOBS={f'u_normal_head_{m}_s{s}_{i}':(s,m,i) for m in MODES for s in range(3) for i in INITS}
INPUTS={'catalog':{'definition'},'z-normal-cal':{'catalog'},'infer':{'catalog','whitelist'},
        'seal':{'catalog','shards','z_thresholds','executions'},'gate':{'catalog','seal'}}


def u_key(job,entity):
    if job not in JOBS:raise ValueError('Unknown U model identity')
    return STAGE+'/'+job+'/'+entity


def z_key(seed,mode,entity):return f's{seed}/{mode}/{entity}'


def expected_u(entities):return {u_key(j,e) for j in JOBS for e in entities}


def runtime_identity():
    root=Path(__file__).resolve().parents[1]
    return dict(stage=STAGE,files={'src/hybrid_tsad/'+p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*.py'))})


def load_operation(path):
    cfg=read_json(path)
    if (set(cfg)!={'engine_schema_version','stage','operation','contract','approval','inputs','parameters','output','score_space'}
            or cfg['engine_schema_version']!=6 or cfg['stage']!=STAGE or cfg['operation'] not in INPUTS
            or cfg['score_space']!=SCORE_SPACE or set(cfg['inputs'])!=INPUTS[cfg['operation']]):
        raise ValueError('Independent schema6 operation/inputs/score-space required')
    allowed={'profile','scope','runtime'}|({'device'} if cfg['operation'] in ('infer','z-normal-cal') else set())
    if set(cfg['parameters'])!=allowed or cfg['parameters']['scope'] not in ('full','shard','l0'):
        raise ValueError('No labels, fit, score variants or arbitrary query parameters')
    if cfg['operation'] not in ('infer',) and cfg['parameters']['scope']!='full':raise ValueError('Only infer permits L0/shard scope')
    if cfg['parameters'].get('device','cpu') not in ('cpu','cuda:0'):raise ValueError('Invalid device; no fallback')
    refs={'contract':cfg['contract'],'approval':cfg['approval'],**cfg['inputs']}
    for ref in [*refs.values(),cfg['parameters']['runtime']]:
        if set(ref)!={'path','sha256'} or not re.fullmatch('[0-9a-f]{64}',ref['sha256']):raise ValueError('Exact path/SHA references required')
    paths={k:resolve_path(v['path'],path) for k,v in refs.items()};output=resolve_path(cfg['output'],path)
    manifest,_=operation_manifest(cfg,paths,output);manifest.update(stage=STAGE,permissions='no training/source-fit/labels/F; explicit operation only')
    return cfg,paths,output,manifest,canonical_sha(manifest)


def approve(cfg,paths,output,digest,base):
    a=Assets()
    for name,ref in {'contract':cfg['contract'],'approval':cfg['approval'],**cfg['inputs']}.items():a.file({'path':str(paths[name]),'sha256':ref['sha256']},base)
    if cfg['contract']['sha256']!=CONTRACT_SHA:raise ValueError('Wrong readout contract')
    approval=read_json(paths['approval']);profile=cfg['parameters']['profile']
    permission=dict(operation=cfg['operation'],training=False,source_fit=False,query_infer=cfg['operation']=='infer',
                    target_normal_cal=cfg['operation']=='z-normal-cal',labels=False,F=False)
    if (approval.get('status')!='APPROVED' or approval.get('stage')!=STAGE or approval.get('contract_sha256')!=CONTRACT_SHA
            or approval.get('operation_manifest_sha256')!=digest or approval.get('permissions')!=permission
            or any(type(approval['permissions'].get(k)) is not bool for k in permission if k!='operation')):
        raise ValueError('Approval does not bind schema6 operation/permissions')
    if profile.get('name')=='SYNTHETIC':
        if approval.get('scope')!='SYNTHETIC_CPU_ENGINEERING_ONLY' or cfg['parameters'].get('device','cpu')!='cpu':
            raise ValueError('Synthetic fixtures are CPU-only, never real release')
    elif profile.get('name')=='TEP100':
        if approval.get('scope')!='U_NORMAL_HEAD_READOUT_EXPLICIT_OPERATION':raise ValueError('Real operation requires separate explicit release')
        if cfg['operation'] in ('infer','z-normal-cal') and cfg['parameters']['device']!='cuda:0':raise ValueError('Real production backend cannot fall back to CPU')
    else:raise ValueError('Unknown query profile')
    runtime=read_json(a.file(cfg['parameters']['runtime'],base))
    if runtime!=runtime_identity():raise ValueError('New readout runtime bytes differ from release')
    return a


def verify_archive(ref,assets,base,required):
    p=assets.file(ref,base);value=read_json(p)
    if value.get('status')!='STAGE2_ARCHIVE_COMPLETE' or value.get('stage')!='U_NORMAL_HEAD':raise ValueError('Incomplete U train/cal archive')
    for name,digest in value['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Archive escapes its root')
        assets.file({'path':str(p.parent/name),'sha256':digest},p)
    for target in required:
        if target.parent!=p.parent or value['files'].get(target.name)!=sha256(target):raise ValueError('U artifact is outside complete archive')
    return value


def whitelist(rows,entities,scope):
    seen=set()
    first=min(entities,key=lambda x:(entities[x]['fault_id'],entities[x]['run_id']))
    for row in rows:
        if set(row)!={'job_id','entity','origins'} or row['job_id'] not in JOBS or row['entity'] not in entities:raise ValueError('Unknown init/job/entity whitelist')
        key=u_key(row['job_id'],row['entity']);allowed=entities[row['entity']]['origins']
        if key in seen:raise ValueError('Duplicate U bank')
        if scope=='l0':
            if JOBS[row['job_id']][0]!=0 or row['entity']!=first or row['origins']!=allowed[:2]:raise ValueError('L0 is seed0 first entity first2 origins')
        elif scope=='shard':
            if row['origins']!=allowed:raise ValueError('Shard needs full original entity origins')
        else:raise ValueError('Infer scope must be l0/shard')
        seen.add(key)
    if not seen:raise ValueError('Empty whitelist')
    if scope=='l0' and seen!={u_key(j,first) for j in JOBS if JOBS[j][0]==0}:raise ValueError('L0 requires exact six seed0 models/initializations')
    return rows


def validate_definition(d,base,profile):
    """Metadata/source/normal-cal verification only; never query arrays or labels."""
    from .stage2_data import validate_target,validate_source_trace
    from .stage2 import calibration_tracks
    from ..api import verify_checkpoint_contract
    from .multi_readout import load_catalog as load_z_catalog
    a=Assets();real=profile['name']=='TEP100'
    required={'schema_version','stage','profile','official','query_manifest','source_references','z_catalog','z_seal',
              'normal_manifest','normal_plans','source_traces','training_runtime','training_runtime_root','u_models','all18_acceptance','physical_bridge'}
    if set(d)!=required or d['schema_version']!=1 or d['stage']!=STAGE or d['profile']!=profile:raise ValueError('Malformed independent readout definition')
    qp=a.file(d['query_manifest'],base)
    bridge=read_json(a.file(d['physical_bridge'],base))
    if (bridge.get('status')!='PRIMARY_ACCEPTED_Z_PHYSICAL_REFERENCE_BRIDGE' or bridge.get('profile')!=profile
            or bridge.get('query_manifest')!=d['query_manifest'] or bridge.get('z_seal')!=d['z_seal']):raise ValueError('HOLD: accepted original Z reference bridge unavailable')
    a.file(d['z_seal'],base)
    if real and sha256(qp)!=QUERY_SHA:
        raise ValueError('Original query/profile identity changed or physical bridge missing')
    entities=entity_profile(read_json(qp),profile)
    refs_path=a.file(d['source_references'],base);refs=read_json(refs_path)
    if real and sha256(refs_path)!=SOURCE_SHA:raise ValueError('Original source references changed')
    official=a.file(d['official'],base)
    if sha256(official)!=OFFICIAL_SHA or a.file(refs['official'],refs_path)!=official:raise ValueError('External official source changed')
    zpath=a.file(d['z_catalog'],base);zc,zm,_=load_z_catalog(zpath,require_full=True)
    originals={(r['seed'],r['mode']):r for r in refs['models']}
    if set(originals)!={(s,m) for s in range(3) for m in MODES}:raise ValueError('Need exactly9 original Z parents')
    for key,parent in originals.items():
        for field in ('checkpoint','checkpoint_metadata','calibration','cal_bank','cal_scores'):
            if a.file(parent[field],refs_path)!=a.file(zm[key][field],zpath):raise ValueError('Z reference substituted')
    runtime_path=a.file(d['training_runtime'],base);training_runtime=read_json(runtime_path);rr=Path(d['training_runtime_root'])
    actual={'src/hybrid_tsad/'+p.relative_to(rr/'src/hybrid_tsad').as_posix() for p in (rr/'src/hybrid_tsad').rglob('*.py')}
    if set(training_runtime['files'])!=actual or real and training_runtime['files'].get('src/hybrid_tsad/engine/stage2.py')!=TRAIN_ENTRY_SHA:
        raise ValueError('Frozen training runtime identity incomplete/changed')
    for name,h in training_runtime['files'].items():a.file({'path':str(rr/name),'sha256':h},base)
    normal_path=a.file(d['normal_manifest'],base);plans={};source_traces={}
    for seed in range(3):
        pp=a.file(d['normal_plans'][str(seed)],base);normal,plans[seed]=validate_target(normal_path,pp,seed,a)
        tr=a.file(d['source_traces'][str(seed)],base)
        parent_meta=read_json(a.file(originals[seed,'hybrid']['checkpoint_metadata'],refs_path))
        validate_source_trace(refs,refs_path,tr,seed,parent_meta,a);source_traces[seed]=tr
    rows=d['u_models']
    if len(rows)!=18 or {r['job_id'] for r in rows}!=set(JOBS):raise ValueError('All18 accepted U models required; no partial real catalog')
    accepted=read_json(a.file(d['all18_acceptance'],base))
    if (accepted.get('status')!='PRIMARY_ACCEPTED_ALL18_U_TRAIN_CAL' or accepted.get('stage')!='U_NORMAL_HEAD'
            or accepted.get('models')!={r['job_id']:canonical_sha(r) for r in rows}):raise ValueError('Primary all18 train+cal acceptance absent/changed')
    model_map={}
    for row in rows:
        seed,mode,init=JOBS[row['job_id']]
        headers=dict(stage='U_NORMAL_HEAD',seed=seed,mode=mode,initialization=init,step=1000)
        if any(row.get(k)!=v for k,v in headers.items()):raise ValueError('U row stage/init/seed/mode/step mismatch')
        names=('checkpoint','checkpoint_metadata','training_summary','train_archive','cal_archive','source_calibration','target_threshold','primary_acceptance')
        if set(row)!={'job_id',*headers,*names}:raise ValueError('Unexpected U row fields')
        paths={k:a.file(row[k],base) for k in names};m=read_json(paths['checkpoint_metadata']);summary=read_json(paths['training_summary'])
        if paths['checkpoint'].name!='step1000.pt' or m.get('trainable_parameters')!=['head.weight','head.bias']:raise ValueError('Only fixed head645 primary step1000 is permitted')
        parent=originals[seed,mode];pmeta=read_json(a.file(parent['checkpoint_metadata'],refs_path))
        if (pmeta.get('excluded_target')!='TEP' or set(pmeta.get('source_domains',[]))!={'PSM','SMD_SUBSET'} or pmeta.get('step')!=10000):raise ValueError('Wrong original Z parent')
        expected={**headers,'checkpoint_sha256':row['checkpoint']['sha256'],'parent_checkpoint_sha256':parent['checkpoint']['sha256'],
            'parent_weights_loaded':init=='source_pretrained','initial_checkpoint_sha256':parent['checkpoint']['sha256'] if init=='source_pretrained' else None,
            'pretraining_domains':['PSM','SMD_SUBSET'] if init=='source_pretrained' else [],'pretraining_excluded_target':'TEP' if init=='source_pretrained' else None,
            'adaptation_target':'TEP','target_labels_used':False,'query_access':False,'trainable_count':645,
            'target_manifest_sha256':d['normal_manifest']['sha256'],'target_plan_sha256':d['normal_plans'][str(seed)]['sha256'],
            'source_cal_trace_sha256':d['source_traces'][str(seed)]['sha256'],'source_references_sha256':d['source_references']['sha256'],
            'runtime_sha256':d['training_runtime']['sha256'],'recipe_sha256':TRAIN_RECIPE_SHA}
        if (any(m.get(k)!=v for k,v in expected.items()) or 'excluded_target' in m or m.get('frozen_before')!=m.get('frozen_after')
                or summary.get('status')!='U_NORMAL_HEAD_STEP1000_COMPLETE' or summary.get('identity')!=m or summary.get('job_id')!=row['job_id']
                or summary.get('primary')!=row['checkpoint'] or summary.get('steps')!=1000):raise ValueError('U fitted parent/role/trace/primary identity mismatch')
        verify_checkpoint_contract(paths['checkpoint'],paths['checkpoint_metadata'],history=100,mode=mode)
        verify_archive(row['train_archive'],a,base,[paths[x] for x in ('checkpoint','checkpoint_metadata','training_summary')])
        verify_archive(row['cal_archive'],a,base,[paths[x] for x in ('source_calibration','target_threshold')])
        acceptance=read_json(paths['primary_acceptance'])
        if acceptance.get('stage')!='U_NORMAL_HEAD' or acceptance.get('status') not in ('PRIMARY_ACCEPTED_FORMAL_PAIR','PRIMARY_ACCEPTED_NAMED_WAVE'):
            raise ValueError('Complete train+cal primary acceptance required; train-only is insufficient')
        train_accept_path=a.file(acceptance['accepted_train'],paths['primary_acceptance']);train_accept=read_json(train_accept_path)
        cal_review=read_json(a.file(acceptance['independent_cal_review'],paths['primary_acceptance']))
        expected_status='PASS_LIMITED_FORMAL_CAL_PAIR' if acceptance['status']=='PRIMARY_ACCEPTED_FORMAL_PAIR' else f"PASS_LIMITED_WAVE{acceptance['wave']}_CAL"
        if (train_accept.get('stage')!='U_NORMAL_HEAD' or train_accept.get('status')!='PRIMARY_ACCEPTED_FORMAL_TRAIN_PAIR'
                or row['job_id'] not in train_accept.get('allowed_jobs',[]) or train_accept.get('summaries',{}).get(row['job_id'])!=row['training_summary']
                or cal_review.get('status')!=expected_status):raise ValueError('Per-wave accepted train summary/cal review does not cover this U row')
        tracks=calibration_tracks(paths['source_calibration'],paths['target_threshold'],row['checkpoint']['sha256'])
        source=read_json(paths['source_calibration']);target=read_json(paths['target_threshold'])
        if (source['source_cal_trace_sha256']!=expected['source_cal_trace_sha256'] or source['source_references_sha256']!=expected['source_references_sha256']
                or target['target_manifest_sha256']!=expected['target_manifest_sha256'] or target['target_plan_sha256']!=expected['target_plan_sha256']
                or source['runtime_sha256']!=expected['runtime_sha256']):raise ValueError('U calibration identity differs from fitted trace/runtime')
        model_map[row['job_id']]=dict(row=row,paths=paths,source=source,tracks=tracks)
    a.recheck()
    return dict(assets=a,definition=d,base=base,entities=entities,models=model_map,z_models=originals,z_catalog=zc,
                z_catalog_path=zpath,source_refs=refs,source_refs_path=refs_path,normal=normal,plans=plans,official=official)
