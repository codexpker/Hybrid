"""Original-source native Z-S consumer; no adaptation, fitting, or label permission."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--trusted-config-sha256',required=True);a=ap.parse_args()
 assert sha(a.config)==a.trusted_config_sha256;c=json.loads(Path(a.config).read_text())
 assert c['operation']=='HYBRID_V1_ORIGINAL_NATIVE_SOURCE_ZS' and c['phase'] in ('profile','readout')
 for p,h in c['immutable_sha256'].items():assert sha(p)==h,p
 science=json.loads(Path(c['science_config']).read_text());assert sha(c['science_config'])==c['science_config_sha256']
 assert science['checkpoint']['sha256']=='c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812'
 assert science['calibration']['sha256']=='da239eac4c5ab1e76718cd651c48f6f89b8d5658b5b9b2a20cb400d88f83d585'
 assert science['history']==512 and science['horizon']==32 and science['inference_seed']==2025
 sys.path.insert(0,c['runtime_source_root'])
 from hybrid_tsad.hybrid_v1 import public_native_zs as native,stage2_u_matched as u
 assert sha(native.__file__)==c['shared_module_sha256']
 import torch
 props=torch.cuda.get_device_properties(0);norm=lambda s:str(s)[4:] if str(s).startswith('GPU-') else str(s)
 assert norm(props.uuid)==norm(c['expected_uuid']) and 'A40' in props.name
 limit=30*1024**3;assert c['allocator_GiB']==30
 torch.cuda.set_per_process_memory_fraction(limit/props.total_memory,0);torch.cuda.reset_peak_memory_stats(0)
 life=Path(c['lifecycle']);dump(life/'CAP_BEFORE_MODEL.json',dict(uuid=str(props.uuid),name=props.name,bytes=limit,before_model=True))
 from hybrid_tsad.hybrid_v1 import cli,readout
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
 try:
  random.seed(2025);np.random.seed(2025);torch.manual_seed(2025)
  cfg=json.loads(Path(science['source_config']['path']).read_text());cfg['device']='cuda:0'
  t=time.monotonic();model=cli.load_model(cfg,science['checkpoint']['path']);model.eval();load_seconds=time.monotonic()-t
  assert model.config.history==512 and model.config.horizon==32 and model.config.self_lag_enabled and model.config.cross_lag_enabled
  # Root-fixed entity boundary: group/order cannot alter fresh native streams.
  initial=readout._rng();state=readout._state(model);dump(out/'INITIAL_ENTITY_RNG.json',readout._rng_json(initial));dump(out/'MODEL_STATE_BEFORE.json',state);cli.save_runtime(out,model)
  cal=json.loads(Path(science['calibration']['path']).read_text())
  assert cal['active_components']==['r_tf','r_ms','r_pred'] and cal['weights']=={'r_tf':.25,'r_ms':.25,'r_pred':.5}
  assert cal['source_domain_equal_weight'] is True and cal['domains']==['SMD','Sunspot']
  assert all(np.isfinite(x) and x>0 for x in cal['scales'].values()) and np.isfinite(cal['threshold'])
  allruns=[]
  for asset in science['query_manifests'].values():
   assert sha(asset['path'])==asset['sha256'];allruns.extend(u.load_roles(asset['path'],{'QUERY'}))
  lookup={native.entity_key(r):r for r in allruns};assert len(lookup)==54
  allowed=science['groups'][c['group_id']]['entity_keys'] if c['phase']=='readout' else [science['profile_entity_key']]
  assert c['entity_keys']==allowed and len(set(allowed))==len(allowed)
  result=[]
  for key in allowed:
   run=lookup[key];assert run.role=='QUERY'
   if run.domain=='SMAP':assert run.values.shape[1]==1 and run.start==0
   else:assert run.domain=='SMD' and run.entity=='machine-1-4' and run.values.shape==(15261,38) and run.start==8446
   readout._restore(initial)
   identity=dict(input_sha256=native.native_identity(run),source_sha256=c['science_source_map_sha256'],config_sha256=c['science_config_sha256'],checkpoint_sha256=science['checkpoint']['sha256'],calibration_sha256=science['calibration']['sha256'])
   directory=Path(science['bank_root'])/key/'bank'
   resume=(c['phase']=='readout' and key==science['profile_entity_key'])
   if resume:
    parent=json.loads((directory/'manifest.json').read_text());assert parent['committed_records']==64 and parent['complete'] is False
   row=native.write_source_entity(model,run,cal,directory,identity=identity,max_new_points=64 if c['phase']=='profile' else None,resume=resume)
   if c['phase']=='profile':
    assert row['new_points']==64 and row['committed_records']==64 and not row['complete']
    import shutil
    shutil.copytree(directory,out/'PROFILE_BANK_SNAPSHOT')
   else:assert row['complete'] and row['committed_records']==len(run.values)-512
   row.update(bank_manifest=str(directory/'manifest.json'));result.append(row)
   dump(out/('ENTITY_'+str(len(result)).zfill(3)+'.json'),row);print(json.dumps(dict(event='NATIVE_ENTITY_SEGMENT_COMPLETED',**row)),flush=True)
  assert readout._state(model)==state;dump(out/'MODEL_STATE_AFTER.json',readout._state(model));dump(out/'BANKS.json',result)
  dump(out/'RECEIPT.json',dict(status='PROFILE64_COMPLETE_PENDING_CONTINUATION' if c['phase']=='profile' else 'COMPLETED_PENDING_INDEPENDENT_AUDIT',phase=c['phase'],group_id=c.get('group_id'),entities=len(result),new_points=sum(r['new_points'] for r in result),covered_points=sum(r['committed_records'] for r in result),model_load_seconds=load_seconds,writer_seconds=sum(r['seconds'] for r in result),profile_prefix_reused_once=any(r['prefix']==64 for r in result),source_unchanged=True,training_updates=0,labels_read=False,target_fit=False,target_CAL=False,threshold_fit=False))
  for p,h in c['immutable_sha256'].items():assert sha(p)==h,p
 finally:dump(life/'GPU_PEAK.json',dict(allocated=torch.cuda.max_memory_allocated(0),reserved=torch.cuda.max_memory_reserved(0),limit=limit,uuid=str(props.uuid)))
if __name__=='__main__':main()
