"""Mechanical actual receipt/identity binding, then one fixed CPU evaluation."""
import argparse,hashlib,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def asset(p):return dict(path=str(p),sha256=sha(p))
def dump(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def bind(s):
    root=Path(s['output_root']);tr=root/'source_train';cal=root/'calibrate_source';q=root/'query_merged'
    r=read(q/'RECEIPT.json');manifest=q/'bank/manifest.json';m=read(manifest)
    assert r['status']=='COMPLETED' and r['points']==87329 and m['complete'] and m['committed_records']==87329
    assert r['architecture']==s['architecture'] and r['implementation_sha256']==s['implementation_sha256']
    seal=dict(status='COMPLETE_WHOLE_RAW_FOR_FIXED_EVALUATION',architecture=s['architecture'],implementation_sha256=s['implementation_sha256'],
        labels_read=False,bank_manifest_sha256=sha(manifest),checkpoint_sha256=sha(tr/'source.pt'),calibration_sha256=sha(cal/'source_calibration.json'),
        query_receipt=asset(q/'RECEIPT.json'),shards=r['shard_receipts'],source_training_receipt=asset(tr/'RECEIPT.json'),source_CAL_receipt=asset(cal/'RECEIPT.json'))
    sealp=P/'WHOLE_RAW_SEAL.ACTUAL.json'
    if sealp.exists():assert read(sealp)==seal
    else:dump(sealp,seal)
    c=read(s['evaluation_template']['path']);assert sha(s['evaluation_template']['path'])==s['evaluation_template']['sha256']
    c.update(status='FROZEN_READY',science_freeze=s['science_freeze'],whole_raw_seal=asset(sealp),
        expected_shared_receipt_fields=dict(architecture=s['architecture'],implementation_sha256=s['implementation_sha256']),
        expected_training_config_fields=dict(operation='HYBRID_LEFT_JOINT_FRFT_CYCLE',phase='train',seed=2025),
        required_evidence=[dict(asset=asset(root/'l0_active/FULL_MODEL_CHECKS.json'),expected_fields=dict(status='PASS'))] if (root/'l0_active/FULL_MODEL_CHECKS.json').exists() else [],
        resource_receipts=[])
    c['required_evidence']=[dict(asset=s['l0_receipt'],expected_fields=dict(status='COMPLETED',scientific_status='L0_PASS_DISCARDED',architecture=s['architecture'])),dict(asset=s['CPU_gate_OS'],expected_fields=dict(exit_code=0,GPUh=0)),dict(asset=s['CPU_schema_receipt'],expected_fields=dict(status='CPU_SCHEMA_PASS',records=2,labels_read=False))]
    base=read(s['base_config']['path']);original=read(base['original_config']['path'])
    c['candidate']=dict(id='Hybrid_LEFT_joint_cycle_source2025',training_receipt=asset(tr/'RECEIPT.json'),calibration_receipt=asset(cal/'RECEIPT.json'),query_receipt=asset(q/'RECEIPT.json'),
        checkpoint=asset(tr/'source.pt'),source_CAL1024=asset(cal/'source_calibration.json'),bank_manifest=asset(manifest),expected_five_identity=m['contract']['identity'],
        initial=base['initial'],source_plan=asset(original['plans']),source_CAL_records=asset(cal/'source_calibration_records.json'),training_config=asset(P/'TRAIN.CONFIG.json'),
        original_source_config=base['original_config'],training_report=asset(tr/'source_training.json'),implementation=asset(Path(base['runtime_src'])/'hybrid_tsad/hybrid_v1/joint_cycle_core.py'),
        checkpoint_metadata=asset(tr/'CHECKPOINT_IDENTITY.json'),initial_mapping=asset(tr/'INITIAL_MAPPING.json'))
    # Includes resource refusal, L0 and all executed model phases, no double charge.
    for p in sorted((P/'lifecycles').glob('*/OS_WAIT.json')):c['resource_receipts'].append(asset(p))
    c['source_sha256']=s['evaluation_source_sha256'];c['runtime_src']=s['evaluation_runtime_src']
    cp=P/'EVALUATION.ACTUAL.json'
    if cp.exists():assert read(cp)==c
    else:dump(cp,c)
    return cp
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    assert sha(a.config)==a.trusted_config_sha256;cp=bind(read(a.config));print(json.dumps(asset(cp)),flush=True)
