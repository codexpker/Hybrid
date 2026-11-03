"""SMD frozen-Z raw -> target-normal-CAL overlay. No fitting or model calls."""
import argparse,json
from pathlib import Path
import numpy as np
from . import bank_recalibration as original, normal_readout as query
from . import normal_control as normal
from .chunked_bank import ChunkedBank,_fingerprint
from .scoring import apply_calibration

OPERATION='HYBRID_V1_SMD_MATCHED_CONTROL_RECALIBRATION'
ACTION='recalibrate_smd_Z_raw'
RANGES=[dict(domain='SMD',entity='machine-2-1',run='machine-2-1_test',start=17641,end=23694)]
read=query.read
write=query.write
sha=query.sha
EXTRA={'calibration_config','calibration_acceptance','normal_plans','normal_manifest'}
RUNTIME_FILES={'hybrid_tsad/__init__.py'}|{'hybrid_tsad/hybrid_v1/'+x+'.py' for x in
    ('__init__','smd_recalibration','bank_recalibration','normal_readout','normal_control','event_adaptation','event_data','target_data','data','scoring','chunked_bank','event_query','event_readout')}

def metadata(spec):
    path=Path(spec['path'])
    if path.suffix!='.json' or any(word in path.name.lower() for word in ('label','ledger')):raise PermissionError('Only named metadata JSON, no annotation/array/PT')
    return original.bound(spec)

def target_calibration(e,producer_config):
    cal=metadata(e['calibration']);receipt=metadata(e['calibration_receipt']);docs={k:metadata(e[k]) for k in EXTRA}
    c=docs['calibration_config'];accepted=query.accepted(docs['calibration_acceptance'],'COMMON_SOURCE')
    query.fixed_R3(cal,False)
    if (e.get('calibration_origin')!='target' or e.get('adaptation_mode')!='Z_TCAL_R3' or e.get('active_components')!=['r_tf','r_ms','r_pred']
        or e['identity']['calibration_sha256']!=e['calibration']['sha256']):raise ValueError('Explicit SMD target R3 overlay only')
    if (receipt.get('operation')!=normal.OPERATION or receipt.get('mode')!='Z_TCAL_R3' or receipt.get('phase')!='source_target_calibrate'
        or receipt.get('status')!='COMPLETED' or receipt.get('updates')!=0 or receipt.get('loaded_roles')!=['NORMAL_CAL']
        or receipt.get('query_or_labels') is not False or receipt.get('target_calibration_contact') is not True
        or receipt.get('choice')!='COMMON_SOURCE' or receipt.get('calibrated_checkpoint_sha256')!=normal.SOURCE
        or receipt.get('source_checkpoint_sha256')!=normal.SOURCE or receipt.get('calibration_file')!='target_r3_calibration.json'
        or receipt.get('threshold')!=cal['threshold']):raise ValueError('SMD Z targetCAL is not PSM U/F/N')
    if (c.get('operation')!=normal.OPERATION or c.get('mode')!='Z_TCAL_R3' or c.get('scope')!='public_target'
        or c.get('permissions')!={'phase':'source_target_calibrate','query':False,'labels':False}
        or c.get('source_checkpoint_sha256')!=normal.SOURCE or c.get('choice')!='COMMON_SOURCE'
        or c.get('select_count')!=32 or c.get('cal_count')!=512 or c.get('seed')!=2025
        or any(c.get(k)!=producer_config.get(k) for k in ('model','sources','history','horizon','active_components'))
        or any(c.get(k) for k in ('adapted_checkpoint','normal_plan','support'))):raise ValueError('Frozen Z CAL configuration/parent differs')
    for key,name in [('plans','normal_plans'),('manifest','normal_manifest')]:
        if e[name]['sha256']!=normal.event.ACCEPTED[key] or c.get(key+'_sha256')!=e[name]['sha256'] or receipt.get(key+'_sha256')!=e[name]['sha256']:
            raise ValueError('Same SMD normal512 CAL pool required')
    if (accepted.get('calibration_sha256')!=e['calibration']['sha256'] or accepted.get('receipt_sha256')!=e['calibration_receipt']['sha256']
        or accepted.get('config_sha256')!=e['calibration_config']['sha256']):raise PermissionError('Own root accepted SMD Z CAL required')
    return cal

def gate(cfg,*,fresh=True):
    if (cfg.get('operation')!=OPERATION or cfg.get('permissions')!={'raw_bank':True,'labels':False,'model_forward':False,'threshold_fit':False}
        or any(k in cfg for k in ('labels','weights','fit','ensemble'))):raise PermissionError('Label-free SMD Z overlay only')
    seal=metadata(cfg['input_seal'])
    expected={k:cfg[k] for k in ('producer','producer_evidence','destination')}
    if seal!=dict(status='ACCEPTED_SMD_Z_RECALIBRATION_INPUTS',approved_by='root',**expected):raise PermissionError('Exact new producer/CAL root seal required')
    parent,e=cfg['producer'],cfg['destination'];ev=cfg['producer_evidence']
    if set(ev)!={'config','receipt','acceptance','execution_identity'}:raise PermissionError('Exact producer metadata only')
    docs={k:metadata(v) for k,v in ev.items()}
    p=docs['config'];r=docs['receipt'];accept=docs['acceptance'];execution=docs['execution_identity']
    if (set(ev)!={'config','receipt','acceptance','execution_identity'} or p.get('operation')!=query.OPERATION or p.get('mode')!='Z_SOURCE_R3'
        or p.get('method_id')!='Z_SOURCE_R3' or p.get('profile')!=query.PROFILE or p.get('choice')!='COMMON_SOURCE'
        or p.get('assets',{}).get('checkpoint',{}).get('sha256')!=normal.SOURCE or p['assets']['calibration']['sha256']!=query.SOURCE_CAL
        or r.get('operation')!=query.OPERATION or r.get('mode')!='Z_SOURCE_R3' or r.get('score_kind')!='source_calibrated_R3'
        or r.get('status')!='COMPLETED_PENDING_RAW_ACCEPTANCE' or r.get('updates')!=0 or r.get('labels_read') is not False
        or r.get('model_state_unchanged') is not True or r.get('records')!=6053 or r.get('bank_manifest_sha256')!=parent['manifest_sha256']
        or execution.get('config_sha256')!=ev['config']['sha256'] or execution.get('source_sha256')!=p.get('source_sha256')
        or parent['identity']['config_sha256']!=ev['config']['sha256'] or parent['identity']['calibration_sha256']!=query.SOURCE_CAL):raise ValueError('Only accepted source-Z producer, never U/N or a borrowed bank')
    query.accepted(accept,'COMMON_SOURCE')
    if accept.get('bank_manifest_sha256')!=parent['manifest_sha256'] or accept.get('checkpoint_sha256')!=normal.SOURCE or accept.get('config_sha256')!=ev['config']['sha256']:
        raise PermissionError('Root raw acceptance identifies different producer')
    manifest=original.bank_metadata(parent)
    if (parent['ranges']!=RANGES or e['ranges']!=RANGES or parent['identity']['input_sha256']!=query.helper.RAW_SHA
        or parent['identity']['checkpoint_sha256']!=normal.SOURCE or e['identity']['checkpoint_sha256']!=normal.SOURCE
        or e['identity']['input_sha256']!=parent['identity']['input_sha256'] or e['identity'].get('input_runs')!=parent['identity'].get('input_runs')
        or e['identity']['config_sha256']!=original.recipe_identity(parent,e)):raise ValueError('Original axis/model/raw input/recipe differs')
    inputs=parent['identity'].get('input_runs',[])
    if (len(inputs)!=1 or any(inputs[0].get(k)!=v for k,v in dict(domain='SMD',entity='machine-2-1',run='machine-2-1_test',start=17129,length=6565).items())
        or not inputs[0].get('sha256') or inputs[0].get('input_identity',{}).get('shape')!=[6565,38]
        or inputs[0]['input_identity'].get('start')!=17129
        or any(len(inputs[0]['input_identity'].get(k,''))!=64 for k in ('values_float32_sha256','observed_sha256'))):raise ValueError('Native independent suffix raw identity required')
    if fresh and Path(e['directory']).exists():raise FileExistsError('Fresh SMD Z overlay required')
    return manifest,target_calibration(e,p)

def build_overlay(parent,destination,calibration):
    """Shared pure-array storage spine; public identities are checked by gate."""
    position=0
    with ChunkedBank(destination['directory'],identity=destination['identity'],ranges=destination['ranges'],threshold=calibration['threshold'],chunk_size=64) as bank:
        for raw in original.checked_rows(parent):
            derived=apply_calibration(raw,calibration);row={k:raw[k] for k in ('domain','entity','run','index','available_index')}
            row.update({k:raw.get(k) for k in destination['active_components']});row.update(derived);row['reason']=derived.get('reason')
            if 'input_identity' in raw.get('replay',{}):row['replay']={'input_identity':raw['replay']['input_identity']}
            row.update(producer_position=position,producer_manifest_sha256=parent['manifest_sha256'],producer_raw_fingerprint=_fingerprint(raw))
            bank.append(row);position+=1
        bank.finalize()
    original.bank_metadata(parent)
    return dict(records=position,manifest_sha256=sha(Path(destination['directory'])/'manifest.json'))

def run(cfg):
    _,cal=gate(cfg);result=build_overlay(cfg['producer'],cfg['destination'],cal)
    receipt=dict(status='COMPLETED_PENDING_ACCEPTANCE',operation=OPERATION,producer=cfg['producer'],destination=cfg['destination'],
        producer_evidence=cfg['producer_evidence'],input_seal=cfg['input_seal'],source_sha256=cfg.get('source_sha256'),model_or_fit_executed=False,labels_accessed=False,threshold_fitted=False,**result)
    out=Path(cfg['destination']['directory']);write(out/'recalibration_receipt.json',receipt);write(out/'operation_config.json',cfg)
    with (out/'smd_recalibration.py').open('xb') as handle:handle.write(Path(__file__).read_bytes())
    return receipt

def iter_materialized(overlay):
    receipt=metadata(overlay['receipt']);parent=overlay['producer'];e=overlay['destination']
    if receipt.get('operation')!=OPERATION or receipt.get('producer')!=parent or receipt.get('destination')!=e:raise ValueError('SMD overlay receipt/lineage differs')
    ev=receipt['producer_evidence'];cfg=dict(operation=OPERATION,permissions=dict(raw_bank=True,labels=False,model_forward=False,threshold_fit=False),
        producer=parent,producer_evidence=ev,destination=e,input_seal=receipt['input_seal'])
    _,cal=gate(cfg,fresh=False);original.bank_metadata(dict(e,manifest_sha256=receipt['manifest_sha256']))
    rows=iter(original.checked_rows(e));count=0
    for count,raw in enumerate(original.checked_rows(parent),1):
        row=next(rows,None);expected=apply_calibration(raw,cal)
        if (row is None or row.get('producer_position')!=count-1 or row.get('producer_manifest_sha256')!=parent['manifest_sha256']
            or row.get('producer_raw_fingerprint')!=_fingerprint(raw) or row.get('valid')!=expected['valid']
            or (expected['valid'] and (row.get('score')!=expected['score'] or row.get('alarm')!=expected['alarm']))
            or (not expected['valid'] and 'alarm' in row)
            or row.get('reason')!=expected.get('reason')
            or _fingerprint({k:row.get(k) for k in e['active_components']})!=_fingerprint({k:raw.get(k) for k in e['active_components']})):raise ValueError('Noncanonical SMD overlay/raw changed')
        result=dict(raw);result['producer_derived']={k:raw.get(k) for k in ('score','valid','reason','alarm')};result.update(expected);result['reason']=expected.get('reason')
        if not expected['valid']:result.pop('alarm',None)
        result.update(producer_position=count-1,producer_manifest_sha256=parent['manifest_sha256']);yield result
    if next(rows,None) is not None or count!=receipt['records']:raise ValueError('Overlay original point count differs')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('config','release','trusted-release-sha256'):p.add_argument('--'+k)
    p.add_argument('--execute',action='store_true');p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,bank_or_labels_opened=False)));return
    cfg=read(a.config);signed=original.bound(dict(path=a.release,sha256=a.trusted_release_sha256))
    if (signed.get('operation')!=OPERATION or signed.get('approved_by')!='root' or signed.get('status')!='APPROVED'
        or signed.get('allowed_actions')!=[ACTION] or signed.get('allow_labels') is not False or signed.get('config_sha256')!=sha(a.config)
        or signed.get('input_seal_sha256')!=cfg['input_seal']['sha256']):raise PermissionError('New exact SMD label-free CPU release required')
    query.code_gate(cfg,signed,RUNTIME_FILES)
    if a.preflight_only:gate(cfg);print(json.dumps(dict(metadata_PASS=True,model_or_labels_opened=False)));return
    print(json.dumps(run(cfg)))

if __name__=='__main__':main()
