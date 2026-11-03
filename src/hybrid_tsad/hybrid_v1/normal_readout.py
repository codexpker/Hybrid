"""Explicit SMD matched N/Z query route. No fitting, labels or F/U aliases."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from . import event_query as helper, event_readout as original_F, normal_control as normal
from .scoring import recipe

OPERATION='HYBRID_V1_SMD_MATCHED_CONTROL_READOUT'
ACTION='matched_control_infer'
PROFILE='SMD21_MATCHED_CONTROLS_H512'
SOURCE_CAL='da239eac4c5ab1e76718cd651c48f6f89b8d5658b5b9b2a20cb400d88f83d585'
SOURCE_ACCEPT='d0820c42cc41edd8c5a1305102d9b2f341c7ca274901088209df087464ffabf5'
COMMON=original_F.ASSETS
NORMAL_EXTRA={'normal_plan','support','reference_trace','reference_receipt','reference_acceptance','normal_input_manifest','normal_plans'}
RUNTIME_FILES=normal.RUNTIME_FILES|{'hybrid_tsad/hybrid_v1/'+n+'.py' for n in ('event_query','event_readout','normal_readout')}
read=normal.event.read
write=normal.event.write
sha=helper.sha

def accepted(document,choice):
    if document.get('approved_by')!='root' or not document.get('status','').startswith('ACCEPTED'):raise PermissionError('Accepted own root asset required')
    return next((x for x in document['cases'] if x.get('choice')==choice),{}) if 'cases' in document else document

def fixed_R3(cal,source):
    active,weights=recipe()
    if (cal.get('source_domain_equal_weight') is not source or cal.get('active_components')!=list(active) or cal.get('weights')!=weights
        or set(cal.get('scales',{}))!=set(active) or not all(np.isfinite(v) and v>0 for v in cal['scales'].values())
        or not np.isfinite(cal.get('threshold',np.nan)) or cal.get('valid_windows')!=(1024 if source else 512)
        or cal.get('domains')!=(['SMD','Sunspot'] if source else ['SMD'])):raise ValueError('Fixed full R3 CAL origin/units/count differ')

def binding(cfg,docs,seal):
    """Metadata only; testable before tensor/array loading."""
    kind=cfg.get('mode');is_N=kind=='N_NEG300';assets=cfg['assets'];choice=cfg.get('choice')
    expected=COMMON|NORMAL_EXTRA if is_N else COMMON
    if (cfg.get('operation')!=OPERATION or cfg.get('profile')!=PROFILE or kind not in {'N_NEG300','Z_SOURCE_R3'}
        or set(assets)!=expected or cfg.get('active_components')!=['r_tf','r_ms','r_pred']
        or any(cfg.get(k)!=v for k,v in dict(history=512,horizon=32,source_seed=2025,inference_seed=2025,chunk_size=64).items())
        or choice not in (helper.CHOICES if is_N else {'COMMON_SOURCE'})
        or cfg.get('method_id')!=('N_NEG300_'+choice+'_EVENT_LOGIT' if is_N else 'Z_SOURCE_R3')
        or cfg.get('score_kind')!=('raw_event_logit' if is_N else 'source_calibrated_R3')):raise PermissionError('Explicit fixed N/Z identity required')
    if seal!=dict(status='ACCEPTED_SMD_MATCHED_QUERY_ASSETS',approved_by='root',mode=kind,choice=choice,assets=assets):raise PermissionError('Exact new N/Z asset seal required')
    source,source_r,train,trained,calcfg,calr,cal=[docs[k] for k in ('source_config','source_receipt','training_config','training_receipt','calibration_config','calibration_receipt','calibration')]
    ckpt=assets['checkpoint']['sha256']
    if (assets['parent_source_checkpoint']['sha256']!=normal.SOURCE or source.get('seed')!=2025
        or source_r.get('phase')!='source_train' or source_r.get('files',{}).get('source.pt')!=normal.SOURCE
        or source_r['files'].get('config.json')!=assets['source_config']['sha256']):raise ValueError('Actual source parent/config/receipt differ')
    for key in ('model','sources','history','horizon','active_components'):
        if any(doc.get(key)!=cfg.get(key) for doc in (source,train,calcfg)):raise ValueError('Saved full architecture/recipe differs')
    ta=accepted(docs['training_acceptance'],choice);ca=accepted(docs['calibration_acceptance'],choice)
    if is_N:
        if ckpt==normal.SOURCE:raise ValueError('N cannot masquerade as frozen Z')
        for doc,phase,roles in ((trained,'target_normal_train',['NORMAL_SELECT','SUPPORT']),(calr,'target_normal_calibrate',['NORMAL_CAL'])):
            if (doc.get('operation')!=normal.OPERATION or doc.get('status')!='COMPLETED' or doc.get('mode')!='N_NEG300'
                or doc.get('phase')!=phase or doc.get('loaded_roles')!=roles or doc.get('query_or_labels') is not False):raise ValueError('N is neither F nor U')
        for key in ('choice','source_checkpoint_sha256','manifest_sha256','plans_sha256','support_sha256','normal_plan_sha256','reference_trace_sha256'):
            if any(doc.get(key)!=train.get(key) for doc in (trained,calcfg,calr)):raise ValueError('N train/ownCAL lineage differs')
        for key,asset in [('manifest','normal_input_manifest'),('plans','normal_plans'),('support','support'),('normal_plan','normal_plan'),('reference_trace','reference_trace'),('reference_receipt','reference_receipt'),('reference_acceptance','reference_acceptance')]:
            if train.get(key+'_sha256')!=assets[asset]['sha256']:raise ValueError('N normal exposure/CAL provenance differs')
        if (train.get('support_sha256')!=normal.event.SUPPORT[choice] or any(train.get(k+'_sha256')!=v for k,v in normal.event.ACCEPTED.items())
            or train.get('source_checkpoint_sha256')!=normal.SOURCE or trained.get('adapted_checkpoint_sha256')!=ckpt
            or trained.get('updates')!=300 or trained.get('positive_forward_exposures')!=0
            or trained.get('selected_attempt') not in (49,99,149,199,249,299)
            or calcfg.get('adapted_checkpoint_sha256')!=ckpt or calcfg.get('adapted_receipt_sha256')!=assets['training_receipt']['sha256']
            or calr.get('updates')!=0 or calr.get('target_calibration_contact') is not True or calr.get('calibrated_checkpoint_sha256')!=ckpt
            or calr.get('calibration_file')!='event_logit_calibration.json' or calr.get('threshold')!=cal.get('threshold')):raise ValueError('N checkpoint or own normalCAL invalid')
        fixed=dict(steps=300,validate_every=50,lr=1e-4,weight_decay=.01,kind='huber',select_count=32,cal_count=512,seed=2025)
        if any(doc.get(k)!=v for doc in (train,calcfg) for k,v in fixed.items()):raise ValueError('N fixed budget/SELECT/CAL recipe differs')
        if (cal.get('score_kind')!='event_logit' or cal.get('q')!=.99 or cal.get('method')!='higher'
            or cal.get('strict_greater') is not True or not np.isfinite(cal.get('threshold',np.nan))):raise ValueError('N raw-logit CAL required')
        if (ta.get('adapted_checkpoint_sha256',ta.get('checkpoint_sha256'))!=ckpt or ta.get('receipt_sha256')!=assets['training_receipt']['sha256']
            or ca.get('calibration_sha256')!=assets['calibration']['sha256'] or ca.get('receipt_sha256')!=assets['calibration_receipt']['sha256']):raise PermissionError('N root acceptance does not bind own PT/CAL/receipt')
        ref=docs['reference_receipt'];refcase=accepted(docs['reference_acceptance'],choice)
        if (ref.get('phase')!='target_event_train' or ref.get('status')!='COMPLETED'
            or refcase.get('adapted_checkpoint_sha256',refcase.get('checkpoint_sha256'))!=ref.get('adapted_checkpoint_sha256')
            or any(ref.get(k)!=train.get(k) for k in ('choice','source_checkpoint_sha256','manifest_sha256','plans_sha256','support_sha256'))):raise ValueError('Accepted F normal exposure reference differs')
        expected_plan=normal.normal_plan_from_f_trace(docs['reference_trace'],docs['support'],history=512,horizon=32,seed=2025,choice=choice,trace_sha256=assets['reference_trace']['sha256'])
        if docs['normal_plan']!=expected_plan or len(expected_plan['rows'])!=300:raise ValueError('Fixed matched normal plan changed')
    else:
        if (ckpt!=normal.SOURCE or assets['training_config']!=assets['source_config'] or assets['training_receipt']!=assets['source_receipt']
            or assets['training_acceptance']['sha256']!=SOURCE_ACCEPT or assets['calibration']['sha256']!=SOURCE_CAL
            or train.get('operation')!='HYBRID_V1_SOURCE_STAGE1' or calcfg.get('operation')!='HYBRID_V1_SOURCE_STAGE1'
            or calr.get('phase')!='calibrate_source' or calr.get('synthetic_non_scientific_results') is not False
            or trained.get('synthetic_non_scientific_results') is not False or calcfg.get('checkpoint_sha256')!=ckpt
            or calcfg.get('train_receipt_sha256')!=assets['training_receipt']['sha256']
            or calr.get('files',{}).get('source_calibration.json')!=assets['calibration']['sha256']
            or calr['files'].get('config.json')!=assets['calibration_config']['sha256']):raise ValueError('Frozen source/CAL identity differs; targetCAL is separate overlay')
        if not {ckpt,assets['calibration']['sha256'],assets['calibration_receipt']['sha256']}<={x.get('sha256') for x in ca.get('assets',[]) if x.get('seed')==2025}:raise PermissionError('Accepted source CAL assets differ')
        fixed_R3(cal,True)
    return dict(mode=kind,choice=choice,method_id=cfg['method_id'],score_kind=cfg['score_kind'],threshold=cal['threshold'],geometry=helper.geometry(),
                target_contact='normal_negative_fit_and_normalCAL' if is_N else 'none',seed_rule='2025 + original_index - 17129')

def code_gate(cfg,signed,required=RUNTIME_FILES):
    own=cfg['source_sha256'];base=Path(__file__).resolve().parents[2]
    if signed.get('source_sha256')!=own or set(own)!=set(required):raise PermissionError('Exact matched control closure required')
    for name,digest in own.items():
        path=(base/name).resolve()
        if base not in path.parents or path.suffix!='.py' or sha(path)!=digest:raise PermissionError('Changed matched control source')

def input_binding(run,npz_sha256):
    """Once-per-run byte identity in the original iter_infer replay schema."""
    return dict(domain=run.domain,entity=run.entity,run=run.run,start=run.start,length=len(run.values),sha256=npz_sha256,
        input_identity=dict(values_float32_sha256=hashlib.sha256(run.values.tobytes()).hexdigest(),
            observed_sha256=hashlib.sha256(run.observed.tobytes()).hexdigest(),start=run.start,shape=list(run.values.shape)))

def preflight(config,release,trusted):
    if not release or not trusted or sha(release)!=trusted:raise PermissionError('Exact new root N/Z query release required')
    cfg,signed=read(config),read(release)
    if (signed.get('status')!='APPROVED' or signed.get('approved_by')!='root' or signed.get('operation')!=OPERATION or cfg.get('operation')!=OPERATION
        or signed.get('allowed_actions')!=[ACTION] or signed.get('allow_labels') is not False or signed.get('config_sha256')!=sha(config)
        or cfg.get('permissions')!={'query':True,'labels':False,'training':False,'threshold_fit':False,'target_fit':False}):raise PermissionError('New zero-update matched route only')
    if Path(cfg['output']).exists():raise FileExistsError('Fresh query output; no retry/resume')
    code_gate(cfg,signed);helper._no_annotation({k:v for k,v in cfg.items() if k!='permissions'})
    seal_spec=cfg['asset_seal']
    if sha(seal_spec['path'])!=seal_spec.get('sha256') or signed.get('asset_seal_sha256')!=seal_spec['sha256']:raise PermissionError('Exact root seal required')
    seal=read(seal_spec['path'])
    # Root seal is checked before any PT/query file-byte access.
    if seal!=dict(status='ACCEPTED_SMD_MATCHED_QUERY_ASSETS',approved_by='root',mode=cfg['mode'],choice=cfg['choice'],assets=cfg['assets']):raise PermissionError('Changed query assets/seal')
    for asset in cfg['assets'].values():
        if any(x in Path(asset['path']).name.lower() for x in ('label','ledger')) or not asset.get('sha256') or sha(asset['path'])!=asset['sha256']:raise PermissionError('Changed/annotation query asset')
    docs={k:read(v['path']) for k,v in cfg['assets'].items() if k not in {'checkpoint','parent_source_checkpoint'}}
    result=binding(cfg,docs,seal);m=docs['query_manifest'];r=original_F.query_metadata(m);qa=docs['query_input_acceptance']
    if (qa.get('approved_by')!='root' or qa.get('status')!='ACCEPTED_SMD_EVENT_QUERY_INPUTS' or qa.get('manifest_sha256')!=cfg['assets']['query_manifest']['sha256']
        or qa.get('NPZ_sha256')!=r['sha256'] or qa.get('numeric_bounds')!=[17129,23694] or qa.get('labels_read') is not False
        or sha(r['path'])!=r['sha256']):raise PermissionError('Accepted independent native38 suffix required')
    from .provenance import OfficialSources
    if signed.get('external_source_sha256')!={x['path']:x['sha256'] for x in OfficialSources(**cfg['sources']).manifest()}:raise PermissionError('Official source identity differs')
    return cfg,signed,docs['calibration'],result

def infer_loaded(model,runs,calibration,output,*,mode,identity,history,horizon,seed):
    """Tiny shared CPU-test spine; real public geometry is enforced at execute."""
    from . import event_adaptation as event, readout
    if mode not in {'N_NEG300','Z_SOURCE_R3'}:raise ValueError('Explicit N/Z score family only')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);model.eval();before=event.state(model)
    flags={n:p.requires_grad for n,p in model.named_parameters()};rng=readout._rng_json(readout._rng())
    for name,value in [('state_before',before),('parameter_flags_before',flags),('RNG_before',rng)]:write(out/(name+'.json'),value)
    if mode=='N_NEG300':helper.write_event_bank(model,runs,calibration,out/'bank',identity=identity,history=history,horizon=horizon,seed=seed,feature_dim=2*model.config.width+3)
    else:readout.write_model_bank(model,runs,calibration,out/'bank',identity=identity,history=history,horizon=horizon,seed=seed,active_components=['r_tf','r_ms','r_pred'])
    after=event.state(model);afterflags={n:p.requires_grad for n,p in model.named_parameters()};afterrng=readout._rng_json(readout._rng())
    for name,value in [('state_after',after),('parameter_flags_after',afterflags),('RNG_after',afterrng)]:write(out/(name+'.json'),value)
    if before!=after or flags!=afterflags:raise RuntimeError('Query updated model/state/flags')
    m=read(out/'bank/manifest.json');receipt=dict(status='COMPLETED_PENDING_RAW_ACCEPTANCE',operation=OPERATION,mode=mode,updates=0,labels_read=False,
        score_kind='raw_event_logit' if mode=='N_NEG300' else 'source_calibrated_R3',records=m['committed_records'],bank_manifest_sha256=sha(out/'bank/manifest.json'),
        model_state_unchanged=True,RNG_transition_recorded=True,RNG_unchanged=rng==afterrng)
    write(out/'receipt.json',receipt);return receipt

def execute(config,release,trusted):
    cfg,signed,cal,bound=preflight(config,release,trusted);cap=read(Path(signed['lifecycle'])/'GPU_CAP_BEFORE_MODEL.json')
    canon=lambda v:str(v).replace('GPU-','')
    if (os.environ.get('HYBRID_MATCHED_QUERY_CHILD')!=trusted or cap.get('release_sha256')!=trusted or cap.get('config_sha256')!=signed['config_sha256']
        or cap.get('set_before_model_construction') is not True or cap.get('limit_bytes')!=30*1024**3 or cap.get('physical_gpu')!=cfg['physical_gpu']
        or canon(cap.get('GPU_uuid'))!=canon(cfg['expected_gpu_uuid'])):raise PermissionError('Owned UUID bound pre-model cap required')
    from . import cli
    from .data import load_manifest
    a=cfg['assets'];weights={k:sha(a[k]['path']) for k in ('checkpoint','parent_source_checkpoint')}
    model=cli.load_model(dict(cfg,seed=2025),a['checkpoint']['path']);runs=load_manifest(a['query_manifest']['path'],roles={'QUERY'})
    if len(runs)!=1 or runs[0].start!=17129 or runs[0].values.shape!=(6565,38):raise ValueError('Independent QUERY geometry changed')
    code=hashlib.sha256(json.dumps(cfg['source_sha256'],sort_keys=True).encode()).hexdigest()
    identity=dict(input_sha256=helper.RAW_SHA,config_sha256=signed['config_sha256'],checkpoint_sha256=a['checkpoint']['sha256'],calibration_sha256=a['calibration']['sha256'],source_sha256=code,
        input_runs=[input_binding(r,read(a['query_manifest']['path'])['runs'][0]['sha256']) for r in runs])
    result=infer_loaded(model,runs,cal,cfg['output'],mode=cfg['mode'],identity=identity,history=512,horizon=32,seed=2025);out=Path(cfg['output'])
    if result['records']!=6053:raise ValueError('Full6053 required')
    cli.save_runtime(out,model);write(out/'config.json',cfg);write(out/'binding.json',bound)
    if any(sha(a[k]['path'])!=v for k,v in weights.items()):raise RuntimeError('Checkpoint bytes changed')
    write(out/'weights_post.json',dict(sha256=weights,unchanged=True));write(out/'execution_identity.json',dict(config_sha256=signed['config_sha256'],release_sha256=trusted,source_sha256=cfg['source_sha256'],asset_seal=cfg['asset_seal']))
    with (out/'normal_readout.py').open('xb') as handle:handle.write(Path(__file__).read_bytes())
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','release','trusted-release-sha256'):p.add_argument('--'+key)
    p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,model_or_data_opened=False)));return
    if a.preflight_only:preflight(a.config,a.release,a.trusted_release_sha256);print(json.dumps(dict(metadata_PASS=True,arrays_or_model_loaded=False)));return
    print(json.dumps(execute(a.config,a.release,a.trusted_release_sha256)))

if __name__=='__main__':main()
