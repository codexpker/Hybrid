"""CPU-only SMD deterministic baselines; one prediction bank, two fixed CAL views."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from . import baselines,normal_control as normal,event_readout,event_query,normal_readout as binding,smd_recalibration as storage
from .chunked_bank import ChunkedBank,_fingerprint
from .scoring import apply_calibration,fit_calibration

OPERATION='HYBRID_V1_SMD_MATCHED_BASELINES'
PHASES={'normal_calibrate','query_source','target_view'}
METHODS=baselines.METHODS
BASELINE_SHA='e14922884c6cbb01fddfe3ba9da9d245b3e7ab3116144a3e958aa904720f6dbd'
SOURCE_CAL={'persistence':'5513840e1454ef3cd84e373c510a336660d716d60971ad6e14b752e1c946d1cd',
            'history_mean512':'16b3a4959a08998ea2dedbf3c2373e7a95e1fae15607f21d065ce42d9825fb9d'}
RUNTIME_FILES=binding.RUNTIME_FILES|storage.RUNTIME_FILES|{'hybrid_tsad/hybrid_v1/smd_baselines.py','hybrid_tsad/hybrid_v1/baselines.py'}
read=normal.event.read
write=normal.event.write
sha=normal.event.sha

def algorithm(method):
    if method not in METHODS:raise ValueError('Only two original deterministic baselines')
    if sha(baselines.__file__)!=BASELINE_SHA:raise PermissionError('Accepted baseline algorithm bytes changed')
    return hashlib.sha256(('NO_TRAINED_CHECKPOINT::'+method+'::'+BASELINE_SHA).encode()).hexdigest()

def calibration(cal,source):
    if (cal.get('source_domain_equal_weight') is not source or cal.get('active_components')!=['r_pred'] or cal.get('weights')!={'r_pred':1.}
        or set(cal.get('scales',{}))!={'r_pred'} or not np.isfinite(cal['scales']['r_pred']) or cal['scales']['r_pred']<=0
        or not np.isfinite(cal.get('threshold',np.nan)) or cal.get('domains')!=(['SMD','Sunspot'] if source else ['SMD'])):raise ValueError('Own r_pred SOURCE/TCAL recipe required')
    return cal

def source_cal(spec,acceptance,method):
    if spec.get('sha256')!=SOURCE_CAL[method]:raise PermissionError('Reuse accepted own baseline source CAL')
    cal=storage.metadata(spec);a=storage.metadata(acceptance);binding.accepted(a,method)
    if next((r.get('calibration_sha256') for r in a.get('methods',[]) if r.get('method')==method),None)!=spec['sha256']:
        raise PermissionError('Source baseline acceptance differs')
    return calibration(cal,True)

def target_cal(spec,receipt,acceptance,method):
    cal=storage.metadata(spec);r=storage.metadata(receipt);a=storage.metadata(acceptance);case=binding.accepted(a,method)
    if 'methods' in a:case=next((x for x in a['methods'] if x.get('method')==method),{})
    if (r.get('operation')!=OPERATION or r.get('phase')!='normal_calibrate' or r.get('method')!=method or r.get('status')!='COMPLETED'
        or r.get('records')!=512 or r.get('plans_sha256')!=normal.event.ACCEPTED['plans'] or r.get('manifest_sha256')!=normal.event.ACCEPTED['manifest']
        or r.get('algorithm_sha256')!=algorithm(method) or r.get('calibration_sha256')!=spec['sha256']
        or r.get('labels_read') is not False or case.get('calibration_sha256')!=spec['sha256'] or case.get('receipt_sha256')!=receipt['sha256']):raise ValueError('Accepted own same512 baseline target CAL required')
    return calibration(cal,False)

def fit_target(runs,points,method,history=512):
    arrived=[dict(p,footprint=[p['index']-history,p['index']+1]) for p in points]
    records=list(baselines.iter_baselines(runs,method=method,history=history,index_plan=arrived))
    if len(records)!=len(points) or [r['index'] for r in records]!=[p['index'] for p in points]:raise ValueError('Fixed CAL origins changed; no resampling')
    return records,fit_calibration(records,source=False,active_components=['r_pred'])

def write_source(runs,method,cal,out,*,identity,history=512,iterator=None):
    """Exactly one original prediction iterator; scoring math is unchanged."""
    iterator=baselines.iter_baselines if iterator is None else iterator
    ranges=[dict(domain=r.domain,entity=r.entity,run=r.run,start=r.start+history,end=r.start+len(r.values)) for r in runs]
    lookup={(r.domain,r.entity,r.run):binding.input_binding(r,'UNBOUND_INTERNAL_TEST')['input_identity'] for r in runs}
    with ChunkedBank(out,identity=identity,ranges=ranges,threshold=cal['threshold'],chunk_size=64) as bank:
        for raw in iterator(runs,method=method,history=history):
            row=dict(raw);row['baseline_intrinsic_reason']=raw.get('reason');row['replay']={'input_identity':lookup[tuple(row[k] for k in ('domain','entity','run'))]}
            derived=apply_calibration(raw,cal);row.update(derived);row['reason']=derived.get('reason')
            bank.append(row)
        bank.finalize()

def view_gate(cfg,fresh=True):
    p,e=cfg['producer'],cfg['destination'];method=cfg['method'];a=cfg['assets']
    if p['identity']['checkpoint_sha256']!=algorithm(method) or e['identity']['checkpoint_sha256']!=algorithm(method):raise ValueError('Baseline has no trained PT')
    sc=source_cal(a['source_calibration'],a['source_acceptance'],method);tc=target_cal(a['target_calibration'],a['target_receipt'],a['target_acceptance'],method)
    if (p['identity']['calibration_sha256']!=a['source_calibration']['sha256'] or e['identity']['calibration_sha256']!=a['target_calibration']['sha256']
        or p['ranges']!=storage.RANGES or e['ranges']!=p['ranges'] or p['identity']['input_sha256']!=event_query.RAW_SHA
        or e['identity']['input_sha256']!=p['identity']['input_sha256'] or e['identity'].get('input_runs')!=p['identity'].get('input_runs')
        or e.get('active_components')!=['r_pred'] or e.get('calibration_origin')!='target' or e.get('adaptation_mode')!='BASELINE_TCAL'
        or e['identity']['config_sha256']!=storage.original.recipe_identity(p,e)):raise ValueError('Source/raw/own TCAL view identity differs')
    storage.original.bank_metadata(p)
    if fresh and Path(e['directory']).exists():raise FileExistsError('Fresh baseline target overlay required')
    return sc,tc

def materialized(spec):
    r=storage.metadata(spec['receipt']);cfg=r['view_config']
    if r.get('operation')!=OPERATION or r.get('phase')!='target_view':raise ValueError('Explicit baseline overlay receipt required')
    _,cal=view_gate(cfg,False);p,e=cfg['producer'],cfg['destination'];storage.original.bank_metadata(dict(e,manifest_sha256=r['manifest_sha256']))
    rows=iter(storage.original.checked_rows(e));count=0
    for count,raw in enumerate(storage.original.checked_rows(p),1):
        v=next(rows,None);expected=apply_calibration(raw,cal)
        if (v is None or v.get('producer_raw_fingerprint')!=_fingerprint(raw) or v.get('producer_position')!=count-1
            or v.get('producer_manifest_sha256')!=p['manifest_sha256'] or v.get('valid')!=expected['valid']
            or (expected['valid'] and (v.get('score')!=expected['score'] or v.get('alarm')!=expected['alarm']))
            or (not expected['valid'] and ('alarm' in v or v.get('reason')!=expected['reason']))):raise ValueError('Noncanonical baseline overlay')
        value=dict(raw);value['producer_derived']={k:raw.get(k) for k in ('score','valid','reason','alarm')};value.update(expected);value['reason']=expected.get('reason')
        if not expected['valid']:value.pop('alarm',None)
        yield value
    if next(rows,None) is not None or count!=r['records']:raise ValueError('Baseline overlay coverage differs')

def preflight(cfg,phase):
    if (phase not in PHASES or cfg.get('operation')!=OPERATION or cfg.get('permissions')!={'phase':phase,'labels':False,'GPU':False,'model':False}
        or cfg.get('history')!=512 or cfg.get('methods')!=list(METHODS)):raise PermissionError('Fixed two deterministic CPU controls only')
    if Path(cfg['output']).exists():raise FileExistsError('Fresh baseline namespace')
    a=cfg['assets']
    if phase=='normal_calibrate':
        m=storage.metadata(a['manifest']);p=storage.metadata(a['plans'])
        if a['manifest']['sha256']!=normal.event.ACCEPTED['manifest'] or a['plans']['sha256']!=normal.event.ACCEPTED['plans'] or len(p['NORMAL_CAL'])!=512:
            raise PermissionError('Original common normal512 plans required')
        if p.get('history')!=512 or p.get('horizon')!=32:raise ValueError('Conservative H512P32 eligibility retained')
    elif phase=='query_source':
        qm=storage.metadata(a['query_manifest']);event_readout.query_metadata(qm);qa=storage.metadata(a['query_input_acceptance'])
        if (qa.get('status')!='ACCEPTED_SMD_EVENT_QUERY_INPUTS' or qa.get('approved_by')!='root' or qa.get('manifest_sha256')!=a['query_manifest']['sha256']
            or qa.get('NPZ_sha256')!=qm['runs'][0]['sha256'] or qa.get('numeric_bounds')!=[17129,23694] or qa.get('labels_read') is not False):raise PermissionError('Accepted suffix input only')
        for method in METHODS:source_cal(a[method+'_source_calibration'],a['source_acceptance'],method)
    else:
        if set(cfg.get('views',{}))!=set(METHODS):raise ValueError('Both predeclared baseline views required')
        for method in METHODS:
            if cfg['views'][method]['method']!=method or Path(cfg['views'][method]['destination']['directory']).resolve()!=(Path(cfg['output'])/method/'bank').resolve():raise ValueError('Own fresh view namespace')
            view_gate(cfg['views'][method])

def execute(cfg,phase):
    preflight(cfg,phase);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);write(out/'config.json',cfg);a=cfg['assets'];outputs={}
    from .data import load_manifest
    if phase=='normal_calibrate':
        runs=load_manifest(a['manifest']['path'],roles={'NORMAL_CAL'});points=storage.metadata(a['plans'])['NORMAL_CAL']
        for method in METHODS:
            rows,cal=fit_target(runs,points,method);folder=out/method;folder.mkdir();from .chunked_bank import _pack
            np.savez_compressed(folder/'raw_CAL.npz',**_pack(rows));write(folder/'target_calibration.json',cal)
            write(folder/'receipt.json',dict(operation=OPERATION,status='COMPLETED',phase=phase,method=method,algorithm_sha256=algorithm(method),records=512,valid_windows=cal['valid_windows'],
                manifest_sha256=a['manifest']['sha256'],plans_sha256=a['plans']['sha256'],calibration_sha256=sha(folder/'target_calibration.json'),labels_read=False))
            outputs[method]=dict(calibration_sha256=sha(folder/'target_calibration.json'),receipt_sha256=sha(folder/'receipt.json'),records=512,valid_windows=cal['valid_windows'])
    elif phase=='query_source':
        runs=load_manifest(a['query_manifest']['path'],roles={'QUERY'});query_manifest=storage.metadata(a['query_manifest'])
        if len(runs)!=1 or runs[0].start!=17129 or runs[0].values.shape!=(6565,38):raise ValueError('Original suffix array differs')
        for method in METHODS:
            spec=a[method+'_source_calibration'];cal=source_cal(spec,a['source_acceptance'],method)
            identity=dict(input_sha256=event_query.RAW_SHA,config_sha256=cfg['execution_config_sha256'],checkpoint_sha256=algorithm(method),calibration_sha256=spec['sha256'],
                source_sha256=hashlib.sha256(json.dumps(cfg['source_sha256'],sort_keys=True).encode()).hexdigest(),input_runs=[binding.input_binding(runs[0],query_manifest['runs'][0]['sha256'])])
            write_source(runs,method,cal,out/method/'bank',identity=identity)
            bank_manifest=read(out/method/'bank/manifest.json')
            if not bank_manifest['complete'] or bank_manifest['committed_records']!=6053:raise ValueError('Both full6053 baseline raw banks required')
            outputs[method]=dict(directory=str(out/method/'bank'),manifest_sha256=sha(out/method/'bank/manifest.json'),records=6053,identity=identity)
    else:
        for method in METHODS:
            view=cfg['views'][method];_,cal=view_gate(view);result=storage.build_overlay(view['producer'],view['destination'],cal)
            write(Path(view['destination']['directory'])/'baseline_overlay_receipt.json',dict(operation=OPERATION,phase=phase,method=method,view_config=view,**result))
            outputs[method]=result
    receipt=dict(status='COMPLETED_PENDING_ACCEPTANCE',operation=OPERATION,phase=phase,methods=outputs,learned_parameters=0,training_seeds=0,labels_read=False,GPUs=0)
    write(out/'receipt.json',receipt);write(out/'execution_identity.json',dict(operation=OPERATION,phase=phase,config_sha256=cfg['execution_config_sha256'],source_sha256=cfg['source_sha256'],labels_read=False,GPUs=0));return receipt

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('config','release','trusted-release-sha256','phase'):p.add_argument('--'+key,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true');args=p.parse_args(argv)
    if not args.execute:print(json.dumps(dict(preview=True,arrays_or_labels_opened=False)));return
    cfg=read(args.config);signed=storage.metadata(dict(path=args.release,sha256=args.trusted_release_sha256))
    if (signed.get('status')!='APPROVED' or signed.get('approved_by')!='root' or signed.get('operation')!=OPERATION or signed.get('allowed_actions')!=[args.phase]
        or signed.get('config_sha256')!=sha(args.config) or signed.get('allow_labels') is not False or os.environ.get('CUDA_VISIBLE_DEVICES')!=''):raise PermissionError('New exact CPU-only baseline release')
    binding.code_gate(cfg,signed,RUNTIME_FILES)
    seal=storage.metadata(cfg['input_seal'])
    if signed.get('input_seal_sha256')!=cfg['input_seal']['sha256'] or seal!=dict(status='ACCEPTED_SMD_BASELINE_INPUTS',approved_by='root',phase=args.phase,assets=cfg['assets'],views=cfg.get('views')):raise PermissionError('Root baseline assets/view seal required')
    if args.preflight_only:preflight(cfg,args.phase);return
    cfg['execution_config_sha256']=sha(args.config);print(json.dumps(execute(cfg,args.phase)))

if __name__=='__main__':main()
