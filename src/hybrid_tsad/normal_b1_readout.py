"""Normal-run B1 score-bank primitives and separately released frozen runtime.

Preview/tests use no Torch, checkpoint or real data. Exact root approval is
mandatory before runtime reference/data/Torch access. No training or labels.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path,PurePosixPath

FINGERPRINT_KEYS = {'state', 'gradients', 'modes', 'step', 'rng'}
COLUMNS = ('xmeas_9', 'xmeas_21', 'xmv_10')
SEED_DEVICES = {2025:(0,'895053ba-a4d6-49a6-d831-865d5982f009'),
                2026:(1,'514a18de-bcd9-8992-c47f-775f98aed4b6'),
                2027:(2,'a3b69732-11e4-3841-7a95-735c2ac27e47')}
CAL_OPERATION='LEFT_TEP_NORMAL_CALIBRATION'
QUERY_OPERATION='LEFT_TEP_FIXED_QUERY_READOUT'


def score_identity(metadata):
    """Reject queries, CAL, training roles and alternative run/shape choices."""
    calibration=metadata.get('purpose')=='NORMAL_CALIBRATION'
    query=metadata.get('purpose')=='FIXED_PUBLIC_QUERY'
    if query:
        allowed={'domain','role','purpose','entity_id','seed','points','window','batch_size','columns',
                 'array_sha256','scaler_sha256','checkpoint_sha256','feature_source_sha256','best_checkpoint_step'}
        if (set(metadata)!=allowed
                or metadata.get('role')!='QUERY' or metadata.get('entity_id') not in ['entity_%04d'%i for i in range(1,401)]
                or metadata.get('seed') not in SEED_DEVICES or metadata.get('points')!=960):
            raise PermissionError('Only opaque query identities without fault/run/labels')
    role_ok=(metadata.get('role')=='CAL' and metadata.get('run_id') in range(81,101)
             and metadata.get('seed') in SEED_DEVICES) if calibration else (
                 metadata.get('role')=='SELECT' and metadata.get('run_id')==61 and metadata.get('seed')==2025)
    if (metadata.get('domain') != 'TEP_REACTOR' or (not query and (not role_ok or metadata.get('fault_number') != 0))
            or tuple(metadata.get('columns', ())) != COLUMNS
            or metadata.get('window') != 100 or metadata.get('batch_size') != 1
            or metadata.get('points') != (960 if query else 500)):
        raise PermissionError('Only predeclared normal SELECT run61 B1 cost gate')
    required = ('array_sha256', 'scaler_sha256', 'checkpoint_sha256',
                'feature_source_sha256', 'best_checkpoint_step')
    if any(key not in metadata for key in required):
        raise ValueError('Missing immutable input/model provenance')
    for key in required[:-1]:
        value = metadata[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError('Invalid SHA256: ' + key)
    if (not isinstance(metadata['best_checkpoint_step'], int)
            or metadata['best_checkpoint_step'] < 369):
        raise ValueError('Actual curriculum-eligible checkpoint step required')


def checkpoint_key_contract(actual, scientific, exact_thop_extras):
    """Metadata-only equality of names, shape AND dtype, never strict=False.

    Dictionaries contain {name: {'shape': [...], 'dtype': '...'}} from saved
    audited checkpoint metadata. No checkpoint is opened by this function.
    An eventual Torch runner still must strict-load the returned scientific keys.
    """
    if set(scientific) & set(exact_thop_extras):
        raise ValueError('Scientific and exact THOP keys overlap')
    expected = dict(scientific, **exact_thop_extras)
    if not scientific or set(actual) != set(expected):
        raise ValueError('Missing/unexpected checkpoint keys')
    if any(actual[k] != expected[k] for k in expected):
        raise ValueError('Checkpoint dtype/shape differs')
    return tuple(scientific)


def collect_bank(normalized_values, metadata, scorer, fingerprint, *, clock=time.perf_counter):
    """Score one already available normalized normal run; no file/model IO.

    scorer accepts one [1,100,3] float32 window and must return a detached,
    synchronized CPU numpy [1,100,3] array. Conversion/transfer belongs inside
    scorer and its measured duration. Full fingerprint runs only at bank edges.
    """
    import numpy as np
    score_identity(metadata)
    values = np.asarray(normalized_values)
    points=metadata['points'];count=points-99
    if values.shape != (points, 3) or values.dtype not in (np.float32, np.float64) or not np.isfinite(values).all():
        raise ValueError('Finite normalized 500x3 floating input required')
    observed_start = clock()
    before = fingerprint()
    if set(before) != FINGERPRINT_KEYS:
        raise ValueError('Explicit full state/gradient/mode/step/RNG fingerprint required')
    # JSON copy prevents callbacks returning a mutable shared dict from hiding changes.
    before = json.loads(json.dumps(before, sort_keys=True, allow_nan=False))
    raw = np.empty((count, 100, 3), np.float32)
    input_windows = np.empty_like(raw)
    call_seconds = np.empty(count, np.float64)
    for start in range(count):
        batch = np.array(values[start:start+100], dtype=np.float32, copy=True)[None]
        input_windows[start] = batch[0]
        begin = clock()
        score = scorer(batch)
        call_seconds[start] = clock() - begin
        if not np.array_equal(batch[0], input_windows[start]):
            raise RuntimeError('Scorer mutated its causal input window')
        if (not isinstance(score, np.ndarray) or score.shape != (1, 100, 3)
                or score.dtype != np.float32 or not np.isfinite(score).all()):
            raise ValueError('Synchronous CPU float32 [1,100,3] score required')
        raw[start] = score[0]
    after = json.loads(json.dumps(fingerprint(), sort_keys=True, allow_nan=False))
    if after != before:
        raise RuntimeError('Frozen model/gradients/modes/step/RNG changed within bank')
    indices = np.arange(99, points, dtype=np.int64)
    point = raw.mean(axis=2, dtype=np.float32)
    arrays = dict(raw_scores=raw, normalized_input_windows=input_windows,
                  end_index_0based=indices, available_index_0based=indices.copy(),
                  end_sample_1based=indices+1, window_start_0based=np.arange(count, dtype=np.int64),
                  last=point[:, -1], mean32=point[:, -32:].mean(axis=1, dtype=np.float32),
                  channel_last=raw[:,-1,:].copy(),channel_mean32=raw[:,-32:,:].mean(axis=1,dtype=np.float32),
                  scorer_and_cpu_return_seconds=call_seconds)
    report = dict(status='BANK_CONSTRUCTED_NOT_FORMAL_MODEL_VALIDATION', metadata=metadata,
                  fingerprint_before=before, fingerprint_after=after,
                  observation_granularity='full fingerprints before/after bank only',
                  observed_collection_seconds=clock()-observed_start,
                  first_call_seconds=float(call_seconds[0]), steady_calls=count-1,
                  steady_mean_seconds=float(call_seconds[1:].mean()),
                  steady_p50_seconds=float(np.quantile(call_seconds[1:], .5)),
                  steady_p95_seconds=float(np.quantile(call_seconds[1:], .95)),
                  timing_scope='scorer synchronous CPU return; includes transfer only when implemented by scorer',
                  no_actual_online_collection_claim=True,
                  no_latency_upper_bound_claim=True)
    return arrays, report


def save_bank(output, arrays, report, *, pipeline_start=None, clock=time.perf_counter):
    """Persist continuous scores/indices; never overwrite an existing directory.

    pipeline_start must be captured by future runner before actual file/model IO;
    if omitted, no end-to-end-with-I/O cost is invented.
    """
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = clock()
    path = output / 'raw_bank.npz'
    np.savez(path, **arrays)
    elapsed = clock() - start
    result = dict(report, archive_write_seconds=elapsed, raw_bank_bytes=path.stat().st_size,
                  raw_bank_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  pipeline_through_bank_write_seconds=(None if pipeline_start is None else clock()-pipeline_start),
                  complete_pipeline_timing_available=pipeline_start is not None)
    (output / 'bank_report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


def preview(config_path):
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if cfg.get('operation')==QUERY_OPERATION:
        return dict(status='PENDING_INPUT_ACCEPTANCE_NO_QUERY_ACCESS',seed=cfg.get('seed'),
            entities=400,points=960,bank_shape=[861,100,3],endpoints=[99,959],
            binding_state=cfg.get('binding_state'),allowed_real_execution=False)
    if cfg.get('operation') in ('LEFT_TEP_B1_NORMAL_COST',CAL_OPERATION):
        if cfg.get('binding_state')=='EXACT_REFERENCES_BOUND':validate_remote_paths(cfg)
    elif cfg.get('operation') != 'NORMAL_B1_READOUT_COST_PREPARATION' or cfg.get('status') != 'PENDING':
        raise PermissionError('Only pending preparation is currently implemented')
    return dict(status='PREPARATION_ONLY_NO_MODEL_ARRAY_OR_CHECKPOINT_ACCESS',
                fixed_input=(dict(role='CAL',run_ids=list(range(81,101)),seed=cfg.get('seed'),points=500,window=100,batch_size=1)
                             if cfg.get('operation')==CAL_OPERATION else dict(role='SELECT', run_id=61, points=500, window=100, batch_size=1)),
                score_shape=[401, 100, 3], endpoint_range_0based=[99, 499],
                frozen_checkpoint_runtime='IMPLEMENTED_CANDIDATE_NOT_BOUND_OR_MODEL_VALIDATED',
                allowed_real_execution=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def require_runtime_release(cfg, config_path, release_path):
    """Only config/release are read. No reference/data/Torch access on refusal."""
    cal=cfg.get('operation')==CAL_OPERATION
    query=cfg.get('operation')==QUERY_OPERATION
    seed=cfg.get('seed');device=SEED_DEVICES.get(seed)
    if query:
        if (cfg.get('binding_state')!='EXACT_REFERENCES_BOUND' or device is None
                or cfg.get('max_seconds')!=5400 or cfg.get('max_allocated_GiB')!=20
                or cfg.get('physical_gpu_index')!=device[0] or cfg.get('expected_gpu_uuid')!=device[1]
                or cfg.get('batch_size')!=1 or cfg.get('window')!=100
                or cfg.get('infer_calls')!=344400
                or cfg.get('entity_ids')!=['entity_%04d'%i for i in range(1,401)]
                or cfg.get('input_acceptance') is None or cfg.get('input_manifest') is None):
            raise PermissionError('Query input acceptance not bound; no input/model access')
        validate_query_paths(cfg)
        release=read_json(release_path)
        expected=dict(status='PRIMARY_RELEASED_LEFT_TEP_FIXED_QUERY_READOUT',approved_by='root',
            approved_seed=seed,config_sha256=sha(config_path),adapter_hashes=cfg['adapter_hashes'],
            allow_opaque_query_inputs=True,allow_labels=False,allow_sealed_identity=False,allow_CAL=False)
        if any(release.get(k)!=v for k,v in expected.items()):raise PermissionError('Exact per-model query release required')
        return release
    if cal:
        if (device is None or cfg.get('run_ids')!=list(range(81,101))
                or cfg.get('max_seconds')!=300 or cfg.get('infer_calls')!=8020
                or cfg.get('physical_gpu_index')!=device[0] or cfg.get('expected_gpu_uuid')!=device[1]
                or cfg.get('quantile_rank_1based')!=7941 or cfg.get('threshold_comparison')!='strict_greater'):
            raise PermissionError('Only per-seed fixed CAL81..100 contract')
    if (cfg.get('operation') not in ('LEFT_TEP_B1_NORMAL_COST',CAL_OPERATION)
            or cfg.get('binding_state') != 'EXACT_REFERENCES_BOUND'
            or (not cal and (seed != 2025 or cfg.get('max_seconds') != 180
                or cfg.get('physical_gpu_index') != 0 or cfg.get('expected_gpu_uuid') != SEED_DEVICES[2025][1]
                or cfg.get('run_id') != 61 or cfg.get('infer_calls') != 401))
            or cfg.get('batch_size') != 1 or cfg.get('window') != 100
            or cfg.get('max_allocated_GiB') != 20
            or cfg.get('checkpoint_initialization') != 'accepted_normal_best'):
        raise PermissionError('Unbound or non-predeclared normal B1 cost config')
    validate_remote_paths(cfg)
    release = read_json(release_path)
    expected = dict(status=('PRIMARY_RELEASED_LEFT_TEP_NORMAL_CALIBRATION' if cal else 'PRIMARY_RELEASED_LEFT_TEP_B1_NORMAL_COST'), approved_by='root',
                    config_sha256=sha(config_path),
                    allow_CAL=False, allow_Faulty_Testing=False, allow_labels=False,
                    adapter_hashes=cfg.get('adapter_hashes'))
    if cal:expected.update(approved_seed=seed,allow_normal_CAL81_100=True,allow_SELECT_run61=False)
    else:expected['allow_SELECT_run61']=True
    # CAL is allowed only by its separately named role permission, not generic query access.
    expected['allow_CAL']=bool(cal)
    if any(release.get(k) != v for k, v in expected.items()):
        raise PermissionError('Exact root release required before Torch/reference/data access')
    return release


def validate_query_paths(cfg):
    if cfg.get('output')!='/home/a312/tsad_research/results/left_tep_fixed_QUERY_seed%d_20261102_v1'%cfg['seed']:
        raise PermissionError('Fixed query output path required')
    paths=[cfg['source_root']]
    for key in ('training_acceptance','normal_source_acceptance','contract','training_report','training_events',
                'checkpoint_schema','scaler','checkpoint','calibration_acceptance','calibration','input_acceptance','input_manifest'):
        r=cfg.get(key)
        if not isinstance(r,dict) or set(r)!={'path','sha256'}:raise PermissionError('Exact query reference required')
        paths.append(r['path'])
    for p in paths:
        if (not isinstance(p,str) or '\\' in p or not p.startswith('/home/a312/tsad_research/')
                or not PurePosixPath(p).is_absolute() or '..' in PurePosixPath(p).parts or 'sealed_identity' in PurePosixPath(p).parts):
            raise PermissionError('No sealed/relative/nonworkspace query references')


def validate_query_manifest(manifest):
    expected=['entity_%04d'%i for i in range(1,401)]
    top={'status','records','raw_source','raw_sha256','sample_axis','imputation','standardization','sorting'}
    def valid_sha(value):
        return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    if (set(manifest)!=top or manifest.get('raw_source')!='/home/a312/tsad_research/data/tep_rieth2017_v1/TEP_Faulty_Testing.RData'
            or not valid_sha(manifest.get('raw_sha256'))
            or manifest.get('sample_axis')!='original 1..960; no absolute datetime'
            or any(manifest.get(k) is not False for k in ('imputation','standardization','sorting'))
            or not isinstance(manifest.get('records'),list)
            or any(not isinstance(r,dict) for r in manifest['records'])):
        raise PermissionError('Exact numerical producer top-level schema required; no label/identity fields')
    if manifest.get('status')!='PUBLIC_SIMULATION_INPUTS_NO_LABEL_FEATURES' or [r['entity_id'] for r in manifest['records']]!=expected:
        raise PermissionError('All400 predetermined opaque entities required')
    for r in manifest['records']:
        p='/home/a312/tsad_research/results/tep_faulty_inputs_run51_70_20261102_v1/inputs/'+r['entity_id']+'.npz'
        if (set(r)!={'entity_id','path','sha256','bytes','shape','dtype','columns','raw_sha256'}
                or r['path']!=p or r['shape']!=[960,3]
                or not valid_sha(r['sha256']) or r['raw_sha256']!=manifest['raw_sha256']
                or type(r['bytes']) is not int or r['bytes']<=0
                or r['dtype']!='float64' or type(r['columns']) is not list
                or any(type(c) is not str for c in r['columns'])
                or r['columns']!=list(COLUMNS)):raise PermissionError('Opaque source manifest schema differs')
    return manifest['records']


def validate_remote_paths(cfg):
    cal=cfg.get('operation')==CAL_OPERATION
    expected=('/home/a312/tsad_research/results/left_tep_normal_CAL_seed%d_20261101_v1'%cfg['seed'] if cal
              else '/home/a312/tsad_research/results/left_tep_b1_normal_cost_seed2025_20261101_v1')
    if cfg.get('output')!=expected:raise PermissionError('Exact POSIX absolute output required')
    paths=[cfg.get('source_root')]
    for key in ('training_acceptance','normal_source_acceptance','contract','training_report',
                'training_events','checkpoint_schema','scaler','normal_metadata','checkpoint'):
        ref=cfg.get(key)
        if not isinstance(ref,dict):raise PermissionError('Bound remote reference required: '+key)
        paths.append(ref.get('path'))
    array_refs=cfg.get('arrays') if cal else [cfg.get('array')]
    if not isinstance(array_refs,list) or len(array_refs)!=(20 if cal else 1):raise PermissionError('Exact number of normal arrays required')
    for ref in array_refs:
        if not isinstance(ref,dict):raise PermissionError('Bound array reference required')
        paths.append(ref.get('path'))
    for path in paths:
        if (not isinstance(path,str) or '\\' in path or not PurePosixPath(path).is_absolute()
                or not path.startswith('/home/a312/tsad_research/')
                or '..' in PurePosixPath(path).parts):
            raise PermissionError('Only absolute workspace-contained POSIX references')


def verified_ref(cfg, key, *, json_body=True):
    ref=cfg[key]
    if set(ref) != {'path','sha256'} or sha(ref['path']) != ref['sha256']:
        raise ValueError('Changed reference: '+key)
    return read_json(ref['path']) if json_body else Path(ref['path'])


def bind_checkpoint_schema(events, schema, training_report):
    basics=[e for e in events if e.get('kind')=='official_basic']
    if len(basics)!=1:raise ValueError('One actual native basic event required')
    basic=basics[0];before=basic['before']['state'];after=basic['after']['state']
    if (not before or set(schema)!=set(after) or not set(before).issubset(after)
            or basic['before']['step']!=0 or basic['after']['step']!=0
            or any(after[k]!=before[k] for k in before)
            or training_report['initial_model_state_and_RNG']['state']!=before):
        raise ValueError('Actual pre/post-basic state/schema provenance differs')
    extras={k:schema[k] for k in set(after)-set(before)}
    if any(not k.endswith(('total_ops','total_params')) or schema[k]!={'shape':[1],'dtype':'torch.float64'}
           for k in extras):
        raise ValueError('Only individually named actual THOP counters may be excluded')
    return {k:schema[k] for k in before}, extras


def calibrate_normal_pool(banks,*,seed,checkpoint_sha256,scaler_sha256):
    """Fixed per-model normal q99; never concatenate run time axes as events."""
    import numpy as np
    if seed not in SEED_DEVICES or len(banks)!=20:raise ValueError('Fixed model and twenty CAL runs required')
    if [b['metadata'].get('run_id') for b in banks]!=list(range(81,101)):
        raise ValueError('CAL81..100 exactly once in original run order')
    source=None
    for bank in banks:
        meta=bank['metadata'];score_identity(meta)
        if (meta.get('purpose')!='NORMAL_CALIBRATION' or meta['seed']!=seed
                or meta['checkpoint_sha256']!=checkpoint_sha256 or meta['scaler_sha256']!=scaler_sha256):
            raise PermissionError('Do not mix roles, seeds, model weights or calibration scalers')
        trace=(meta['feature_source_sha256'],meta['best_checkpoint_step'])
        if source is None:source=trace
        if trace!=source:raise PermissionError('Mixed extraction implementations/checkpoint steps')
        if not np.array_equal(bank['end_index_0based'],np.arange(99,500)):
            raise ValueError('Each run must preserve its own 401 endpoint indices')
        for key in ('last','mean32'):
            a=bank[key]
            if not isinstance(a,np.ndarray) or a.shape!=(401,) or a.dtype!=np.float32 or not np.isfinite(a).all():
                raise ValueError('Complete finite original float32 CAL vectors required')
    dist={key:np.concatenate([b[key] for b in banks]) for key in ('last','mean32')}
    dist.update(run_id=np.repeat(np.arange(81,101,dtype=np.int64),401),
                end_index_0based=np.tile(np.arange(99,500,dtype=np.int64),20))
    rank=(99*(8020+1)+99)//100
    assert rank==7941
    result=dict(status='NORMAL_ONLY_FIXED_THRESHOLDS_NOT_DETECTION_EVIDENCE',seed=seed,
                checkpoint_sha256=checkpoint_sha256,scaler_sha256=scaler_sha256,n=8020,
                quantile=.99,rank_1based=rank,comparison='strict_greater',
                run_ids=list(range(81,101)),no_cross_run_event_concatenation=True,
                thresholds={key:float(np.sort(dist[key],kind='stable')[rank-1]) for key in ('last','mean32')})
    result['observed_CAL_point_exceedance']={key:int(np.count_nonzero(dist[key]>result['thresholds'][key])) for key in ('last','mean32')}
    result['limitation']='Empirical point calibration only; no event-FPR or unseen-domain guarantee'
    return dist,result


def execute_cost(config_path, release_path):
    """Run once only after exact root authorization; no fit or label path."""
    cfg=read_json(config_path)
    require_runtime_release(cfg,config_path,release_path)
    cal=cfg['operation']==CAL_OPERATION;query=cfg['operation']==QUERY_OPERATION;seed=cfg['seed']
    from . import left_baseline as baseline
    root=Path(cfg['source_root'])
    required=set(baseline.MODEL_FILES+['exp/exp_anomaly_detection.py',
        'configs/PSM/model_configs_Left.json','configs/PSM/model_configs_0.json','configs/PSM/train_configs.json']+
        ['models/CrossAD/'+n+'.py' for n in ('Basic_CrossAD','Attention_Blocks','EncDec','Context_Blocks')])
    if set(cfg['source_hashes'])!=required:raise ValueError('Fixed thirteen official sources required')
    for name,digest in cfg['source_hashes'].items():
        if sha(root/name)!=digest:raise ValueError('Official source changed: '+name)
    if set(cfg['adapter_hashes'])!={'__init__.py','left_baseline.py','normal_b1_readout.py'}:
        raise ValueError('Exact three-helper runtime closure required')
    for name,digest in cfg['adapter_hashes'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('Own source changed: '+name)
    if cfg.get('upstream_commit')!=baseline.COMMIT:raise ValueError('Wrong official commit')
    if Path(cfg['output']).exists():raise FileExistsError(cfg['output'])
    pipeline_start=time.perf_counter()
    report={'status':'RUNNING','operation':cfg['operation'],'seed':seed,
            'config_sha256':sha(config_path),'release_sha256':sha(release_path),
            'actual_array_reads':[],'CAL_only':cal,'QUERY_only':query,'no_labels':True,
            'no_fault_query_or_labels':not query}
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);torch=None
    try:
        train_receipt=verified_ref(cfg,'training_acceptance')
        if train_receipt.get('status')!='PRIMARY_ACCEPTED_LIMITED_TEP_NORMAL_TRAINING':
            raise PermissionError('Normal training primary acceptance required')
        source_receipt=verified_ref(cfg,'normal_source_acceptance')
        if (source_receipt.get('status')!='PRIMARY_ACCEPTED_LIMITED_NORMAL_ASSETS'
                or source_receipt.get('source_manifest_sha256')!='1e2cca586b9632cfac6b5cf8d4a73e9af486a490173753611796e8ee51958d1e'):
            raise PermissionError('Accepted normal source required')
        # Contract bytes are bound but do not themselves authorize execution.
        verified_ref(cfg,'contract',json_body=False)
        training=verified_ref(cfg,'training_report');events=verified_ref(cfg,'training_events')
        schema=verified_ref(cfg,'checkpoint_schema');scaler=verified_ref(cfg,'scaler')
        metadata=verified_ref(cfg,'input_manifest' if query else 'normal_metadata')
        for field,refname in [('report_sha256','training_report'),('events_sha256','training_events'),('state_keys_sha256','checkpoint_schema')]:
            if train_receipt[field]!=cfg[refname]['sha256']:raise ValueError('Training receipt cross-reference differs: '+field)
        if (any(train_receipt['checkpoint'][k]!=cfg['checkpoint'][k] for k in ('path','sha256'))
                or any(train_receipt['scaler'][k]!=cfg['scaler'][k] for k in ('path','sha256'))):
            raise ValueError('Training receipt model/scaler identity differs')
        if (training['status']!='COMPLETED_PENDING_INDEPENDENT_REVIEW'
                or training['operation']!='LEFT_TEP_C3_H100_NORMAL_TRAIN' or training['seed']!=seed
                or training['checkpoint']['sha256']!=cfg['checkpoint']['sha256']
                or training['scaler_sha256']!=cfg['scaler']['sha256']):
            raise ValueError('Actual normal training references differ')
        science,extras=bind_checkpoint_schema(events,schema,training)
        if train_receipt['scientific_state_schema']!=science:
            raise ValueError('Primary receipt scientific schema differs from actual basic')
        if (train_receipt['exact_thop_extra_schema']!=extras
                or train_receipt['best_step']!=training['best_checkpoint_python_step_at_save']
                or train_receipt['training_terminal_python_step']!=training['final_python_step']):
            raise ValueError('Primary receipt exact extras or best/terminal step differs')
        expected_fit=[['TEP_REACTOR','FIT',i] for i in list(range(1,21))+list(range(51,61))]
        if (scaler['fit_ids']!=expected_fit or scaler['fit_points']!=15000 or scaler['ddof']!=0
                or scaler['SELECT_used_for_fit'] is not False or tuple(scaler['columns'])!=COLUMNS):
            raise ValueError('FIT-only scaler provenance differs')
        if query:
            acceptance=verified_ref(cfg,'input_acceptance')
            if (acceptance.get('status')!='PRIMARY_ACCEPTED_LIMITED_TEP_FAULTY_INPUTS'
                    or acceptance.get('input_manifest_sha256')!=cfg['input_manifest']['sha256']):
                raise PermissionError('Actual primary-accepted input manifest required')
            calibration_receipt=verified_ref(cfg,'calibration_acceptance');calibration=verified_ref(cfg,'calibration')
            if (calibration_receipt.get('status')!='PRIMARY_ACCEPTED_LIMITED_NORMAL_CALIBRATION'
                    or calibration_receipt['seed']!=seed or calibration['seed']!=seed
                    or calibration_receipt['calibration_json_sha256']!=cfg['calibration']['sha256']
                    or calibration_receipt['checkpoint_sha256']!=cfg['checkpoint']['sha256']
                    or calibration['checkpoint_sha256']!=cfg['checkpoint']['sha256']
                    or calibration_receipt['scaler_sha256']!=cfg['scaler']['sha256']
                    or calibration['scaler_sha256']!=cfg['scaler']['sha256']
                    or calibration_receipt['thresholds']!=calibration['thresholds']):raise PermissionError('Frozen per-model CAL references differ')
            rows=validate_query_manifest(metadata);run_ids=[r['entity_id'] for r in rows]
            refs=[{'path':r['path'],'sha256':r['sha256']} for r in rows]
        else:
            run_ids=list(range(81,101)) if cal else [61]
            records=metadata['records'] if cal else metadata['roles']['SELECT']
            rows=[r for r in records if r['role']==('cal' if cal else 'select') and r['simulationRun'] in run_ids]
            if [r['simulationRun'] for r in rows]!=run_ids:raise ValueError('Unique ordered normal runs required')
            refs=cfg['arrays'] if cal else [cfg['array']]
            for ref,bound in zip(rows,refs):
                if (ref['faultNumber']!=0 or ref['shape']!=[500,3] or ref['dtype']!='float64'
                        or tuple(ref['column_names'])!=COLUMNS or ref['array_path']!=bound['path']
                        or ref['sha256']!=bound['sha256']
                        or ref['raw_sha256']!='12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940'):
                    raise ValueError('Exact accepted normal array identity differs')
        checkpoint_path=verified_ref(cfg,'checkpoint',json_body=False)
        author=read_json(root/'configs/PSM/model_configs_Left.json')
        expected_model=dict(author,seq_len=100,enc_in=3,warmup_steps=123,ramp_steps=246)
        if training['effective_configs']['model']!=expected_model:raise ValueError('Actual model config differs')
        import os
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(cfg['physical_gpu_index']):raise RuntimeError('Fixed physical GPU required')
        import numpy as np
        mean=np.asarray(scaler['mean']);scale=np.asarray(scaler['scale'])
        if mean.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('Invalid fitted scaler')
        report['metadata_and_IO_seconds']=time.perf_counter()-pipeline_start
        torch,Left,loaded=baseline.model_runtime(cfg,root)
        props=torch.cuda.get_device_properties(0)
        if str(props.uuid).lower().removeprefix('gpu-')!=cfg['expected_gpu_uuid']:
            raise RuntimeError('Actual GPU UUID differs')
        from types import SimpleNamespace
        model=Left(SimpleNamespace(**expected_model)).cuda()
        fresh={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in model.state_dict().items()}
        if fresh!=science:raise ValueError('Fresh model schema not exact actual pre-basic science schema')
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        actual={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in checkpoint.items()}
        checkpoint_key_contract(actual,science,extras)
        model.load_state_dict({k:checkpoint[k] for k in science},strict=True)
        del checkpoint
        model.eval()
        if model.step!=0 or model._curriculum_lambda()!=.3:raise ValueError('Fresh step/eval lambda differs')
        def tensor_hash(items):
            h=hashlib.sha256()
            for name,value in items:
                h.update(name.encode());h.update(b'None' if value is None else value.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        def fingerprint():
            import random
            nstate=np.random.get_state()
            return dict(state=tensor_hash(model.state_dict().items()),
                        gradients=tensor_hash((k,p.grad) for k,p in model.named_parameters()),
                        modes={k:m.training for k,m in model.named_modules()},step=model.step,
                        rng=dict(python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                            numpy=hashlib.sha256(nstate[1].tobytes()+repr((nstate[0],nstate[2:])).encode()).hexdigest(),
                            torch_cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                            cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()))
        def scorer(batch):
            if time.perf_counter()-pipeline_start>cfg['max_seconds']:raise TimeoutError('Contract elapsed-time limit exceeded')
            with torch.no_grad():
                score,aux=model.infer(torch.from_numpy(batch).to('cuda:0'),None,None,None)
                if aux is not None:raise ValueError('Unexpected native auxiliary output')
                result=score.detach().cpu().numpy().copy()
                torch.cuda.synchronize()
            if torch.cuda.max_memory_allocated()>20*1024**3:raise MemoryError('20GiB contract exceeded')
            return result
        report.update(loaded_modules=loaded,scientific_keys=list(science),excluded_exact_thop_schema=extras,
            best_checkpoint_step=training['best_checkpoint_python_step_at_save'],training_terminal_step=training['final_python_step'],
            fresh_inference_python_step=model.step,eval_lambda=.3,no_warmup_before_first_B1=True,
            setup_through_model_load_seconds=time.perf_counter()-pipeline_start,
            hardware={'GPU':props.name,'uuid':str(props.uuid),'torch':torch.__version__,'cuda':torch.version.cuda})
        pools=[];report['banks']=[];compact=[];storage_bytes=0
        for run_id,ref in zip(run_ids,refs):
            array_path=Path(ref['path'])
            if sha(array_path)!=ref['sha256']:raise ValueError('Frozen normal array changed')
            if query:
                with np.load(array_path,allow_pickle=False) as z:
                    if set(z.files)!={'values','sample_1based','source_row_0based'}:raise PermissionError('Only opaque numerical input fields')
                    values=z['values'].copy();samples=z['sample_1based'].copy();source_rows=z['source_row_0based'].copy()
                if not np.array_equal(samples,np.arange(1,961)) or source_rows.shape!=(960,) or source_rows.dtype!=np.int64 or (source_rows<0).any() or not (np.diff(source_rows)>0).all():raise ValueError('Original query sample/row axis differs')
            else:values=np.load(array_path,allow_pickle=False)
            report['actual_array_reads'].append(str(array_path));points=960 if query else 500
            if values.shape!=(points,3) or values.dtype!=np.float64 or not np.isfinite(values).all():
                raise ValueError('Finite original float64 normal run required')
            normalized=(values-mean)/scale
            identity=dict(domain='TEP_REACTOR',role='CAL' if cal else 'SELECT',run_id=run_id,fault_number=0,columns=COLUMNS,
                seed=seed,points=points,window=100,batch_size=1,array_sha256=ref['sha256'],
                scaler_sha256=cfg['scaler']['sha256'],checkpoint_sha256=cfg['checkpoint']['sha256'],
                feature_source_sha256=sha(__file__),best_checkpoint_step=training['best_checkpoint_python_step_at_save'])
            if cal:identity['purpose']='NORMAL_CALIBRATION'
            if query:
                identity.pop('run_id');identity.pop('fault_number')
                identity.update(role='QUERY',purpose='FIXED_PUBLIC_QUERY',entity_id=run_id)
            bank,bank_report=collect_bank(normalized,identity,scorer,fingerprint)
            bank['source_values_float64' if query else 'source_normal_values_float64']=values
            directory=out/'banks'/run_id if query else out/('run%03d'%run_id if cal else 'bank')
            saved=save_bank(directory,bank,bank_report,pipeline_start=pipeline_start)
            report['banks'].append(dict(**({'entity_id':run_id} if query else {'run_id':run_id}),path=str(directory),**saved))
            if query:
                compact.append({k:bank[k].copy() for k in ('last','mean32','channel_last','channel_mean32','end_index_0based','available_index_0based')})
                storage_bytes+=saved['raw_bank_bytes']
                if storage_bytes>8*1024**3:raise RuntimeError('8GiB raw storage contract exceeded')
            elif cal:pools.append(dict(metadata=identity,end_index_0based=bank['end_index_0based'],
                                     last=bank['last'].copy(),mean32=bank['mean32'].copy()))
            else:report.update(bank=saved,bank_directory=str(directory))
        if query:
            np.savez(out/'compact_scores.npz',entity_id=np.asarray(run_ids),**{k:np.stack([b[k] for b in compact]) for k in compact[0]})
            report.update(opaque_entities=run_ids,raw_bank_count=400,scored_points=344400,raw_bank_bytes=storage_bytes,
                          compact_scores_sha256=sha(out/'compact_scores.npz'),no_label_evaluation=True)
        if cal:
            distribution,calibration=calibrate_normal_pool(pools,seed=seed,checkpoint_sha256=cfg['checkpoint']['sha256'],scaler_sha256=cfg['scaler']['sha256'])
            np.savez(out/'calibration_scores.npz',**distribution)
            calibration['distribution_sha256']=sha(out/'calibration_scores.npz')
            (out/'calibration.json').write_text(json.dumps(calibration,indent=2,allow_nan=False),encoding='utf-8')
            report['calibration']=calibration
        report.update(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',parameter_count=sum(p.numel() for p in model.parameters()),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as e:
        import traceback
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',error=type(e).__name__,traceback=traceback.format_exc())
        raise
    finally:
        report['pipeline_before_report_write_seconds']=time.perf_counter()-pipeline_start
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release')
    args = parser.parse_args()
    if args.execute:
        if not args.release:raise PermissionError('Exact root release required before config/reference/model access')
        print(json.dumps(execute_cost(args.config,args.release),indent=2));return
    print(json.dumps(preview(args.config), indent=2))


if __name__ == '__main__':
    main()
