"""Explicit independent joint-cycle factory; original source13/data/score untouched."""
import argparse,hashlib,json,sys,time
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def metadata(model,pt,a):
 return dict(a.identity(model),checkpoint_sha256=sha(pt),model_config=json.loads(json.dumps(model.config.to_dict())))
def scored_bank(model,run,plan,cal,directory,identity):
 import torch
 from hybrid_tsad.hybrid_v1 import chunked_bank as cb,original_stream_training as stream,readout,cli
 ranges=[]
 for e in plan:
  if ranges and ranges[-1]['end']==e['index']:ranges[-1]['end']+=1
  else:ranges.append(dict(domain=run.domain,entity=run.entity,run=run.run,start=e['index'],end=e['index']+1))
 state=readout._state(model);modes=[(m,m.training) for m in model.modules()]
 devices=[0] if next(model.parameters()).is_cuda else []
 try:
  with torch.random.fork_rng(devices=devices),cb.ChunkedBank(directory,identity=identity,ranges=ranges,threshold=cal['threshold'],chunk_size=64) as bank:
   for row in cb.iter_pointwise_infer(model,[run],history=512,horizon=32,seed=2025,index_plan=plan,infer_fn=stream.iter_infer):bank.append(cli.score_records([row],cal)[0])
   bank.finalize()
 finally:
  for m,mode in modes:m.training=mode
 assert readout._state(model)==state
def finite_l0(c,cfg,old,a,out):
 import torch,numpy as np
 from hybrid_tsad.hybrid_v1 import data,training,readout,chunked_bank as cb,original_stream_training as stream
 runs=old.load_manifest(cfg['manifest'],roles={'FIT','SELECT'});fits=[r for r in runs if r.role=='FIT'];sels=[r for r in runs if r.role=='SELECT'];p=read(cfg['plans'])
 fp=[next(e for e in p['FIT'] if e['domain']=='SMD'),p['FIT'][26]];assert fp[1]['domain']=='Sunspot'
 sp=[next(e for e in p['SELECT'] if e['domain']==d) for d in ('SMD','Sunspot')]
 if c.get('resume_technical_from'):
  prior=Path(c['resume_technical_from']);train=prior/'discarded_train';mapping=read(prior/'INITIAL_MAPPING.json');report=read(train/'training.json')
  assert report['attempts']==report['updates']==2
  model=a.load_checkpoint(cfg,train/'discarded.pt',train/'CHECKPOINT_IDENTITY.json')
  train_seconds=sum(x['train_sync_wall_seconds']+x['SELECT_wall_seconds']['total']+x['artifact_wall_seconds'] for x in report['steps'])
  dump(out/'REUSED_DISCARDED_UPDATES.json',dict(training_report_sha256=sha(train/'training.json'),checkpoint_sha256=sha(train/'discarded.pt'),updates_reused=2,new_training_updates=0))
 else:
  model,mapping=a.construct(cfg,shared_initial=c['initial']['path'],expected_initial_sha256=c['initial']['sha256']);dump(out/'INITIAL_MAPPING.json',mapping)
  torch.save(readout._rng(),out/'INITIAL_RNG.pt');train=out/'discarded_train';train.mkdir()
  start=time.monotonic()
  model,report=a.fit(model,fits,sels,history=512,horizon=32,steps=2,lr=cfg['lr'],kind=cfg['kind'],seed=2025,validate_every=1,
   sampling_plan=fp,validation_plan=sp,weight_decay=cfg['weight_decay'],artifact_dir=train)
  torch.cuda.synchronize();train_seconds=time.monotonic()-start
  assert report['attempts']==report['updates']==2 and report['steps'][1]['valid_counts']['analysis']==[0] and report['steps'][1]['updated'];dump(train/'training.json',report)
  a.save_checkpoint(model,train/'discarded.pt',train/'CHECKPOINT_IDENTITY.json');model.eval()
 loaded=a.load_checkpoint(cfg,train/'discarded.pt',train/'CHECKPOINT_IDENTITY.json');assert readout._state(model)==readout._state(loaded)
 rr,t=data.resolve_plan(fits,[fp[0]],512,32)[0];b=training.tensor_batch(data.window(rr,t,512,32),'cuda:0')
 with torch.no_grad():f,r=training.passes(model,b,2025);ff,rrr=training.passes(loaded,b,2025)
 parity={k:float((v-w).abs().max()) for k,v,w in [('prediction',f['prediction'],ff['prediction']),('tf',r['tf_reconstruction'],rrr['tf_reconstruction']),('ms',r['ms_reconstruction'],rrr['ms_reconstruction'])]}
 assert all(v<=1e-5 for v in parity.values())
 # Actual CAL path with old scalar R3 and the new loaded model, no source CAL opened.
 start=time.monotonic();records=old.infer(loaded,fits,history=512,horizon=32,seed=2025,index_plan=[dict(e,footprint=[e['index']-512,e['index']+1]) for e in fp])
 cal=old.calibrate_route(records,'STAGE1');torch.cuda.synchronize();cal_seconds=time.monotonic()-start
 dump(out/'technical_CAL_records.json',records);dump(out/'technical_calibration.json',cal)
 vals=rr.values[t-512:t+65,:25].copy();obs=rr.observed[t-512:t+65,:25].copy();assert vals.shape==(577,25)
 run=data.Run('SYNTHETIC_FIT','C25','technical','QUERY',vals,obs);q=[dict(domain=run.domain,entity=run.entity,run=run.run,role='QUERY',index=i) for i in range(512,577)]
 identity={k:'0'*64 for k in cb.IDENTITY_KEYS};state=readout._state(loaded);rng=readout._rng();modes=[(m,m.training) for m in loaded.modules()]
 start=time.monotonic();scored_bank(loaded,run,q,cal,out/'full65',identity);torch.cuda.synchronize();query_seconds=time.monotonic()-start
 rows=list(cb.read_records(out/'full65'));assert len(rows)==65 and [r['index'] for r in rows]==list(range(512,577))
 assert all(len(r['prediction'])==32 and len(r['prediction'][0])==25 for r in rows)
 assert readout._state(loaded)==state and torch.equal(rng['cpu'],readout._rng()['cpu'])
 assert all(torch.equal(x,y) for x,y in zip(rng['cuda'],readout._rng()['cuda'])) and all(m.training==mode for m,mode in modes)
 check=dict(missing_spectrum_valid_normal_update=True,analysis_zero_but_update=True,status='L0_PASS_DISCARDED',architecture=a.ARCHITECTURE,updates=2,shared9a8_exact=mapping['shared_exact'],full_old_initial_exact=False,
  reload_maxabs=parity,records=65,forecast_shape=[32,25],labels_read=False,source_CAL_read=False,original_source13_unchanged=True,
  RNG_exact=True,modes_exact=True,state_unchanged_in_eval=True,train2_with_SELECT4_seconds=train_seconds,CAL2_seconds=cal_seconds,query65_seconds=query_seconds,
  parameter_count=sum(p.numel() for p in model.parameters()),note='New heads initialized seed2025; two SMD/Sunspot technical updates discarded; no production training')
 dump(out/'L0_CHECKS.json',check);return dict(scientific_status='L0_PASS_DISCARDED',updates=2,records=65)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);p.add_argument('--metadata-only',action='store_true');p.add_argument('--cpu-schema-output');z=p.parse_args()
 assert sha(z.config)==z.trusted_config_sha256;c=read(z.config);assert c['operation']=='HYBRID_LEFT_JOINT_FRFT_CYCLE' and c['seed']==2025
 for path,digest in c['immutable_sha256'].items():assert sha(path)==digest,path
 assert not Path(c['output']).exists()
 if z.metadata_only:print(json.dumps(dict(status='METADATA_PASS',torch_imported='torch' in sys.modules)));return
 sys.path.insert(0,c['runtime_src'])
 import torch
 from hybrid_tsad.hybrid_v1 import cli as old,readout,joint_cycle_core as a
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 assert c['loss_weights']==dict(analysis=.5,synthesis=.5,ms=.5,consistency=.1,pred=1.)
 if z.cpu_schema_output:
  import numpy as np
  from hybrid_tsad.hybrid_v1 import data,chunked_bank as cb
  out=Path(z.cpu_schema_output);out.mkdir(parents=True,exist_ok=False)
  cfg=read(c['original_config']['path']);cfg['device']='cpu'
  m,mapping=a.construct(cfg,shared_initial=c['initial']['path'],expected_initial_sha256=c['initial']['sha256']);m.eval()
  x=np.random.default_rng(2025).normal(size=(514,2)).astype(np.float32);run=data.Run('SYNTHETIC','fixture','CPU','QUERY',x,np.ones_like(x,bool))
  plan=[dict(domain=run.domain,entity=run.entity,run=run.run,role='QUERY',index=i) for i in (512,513)]
  cal=dict(scales={k:1. for k in ('r_tf','r_ms','r_pred')},threshold=1.,weights={'r_tf':.25,'r_ms':.25,'r_pred':.5},active_components=['r_tf','r_ms','r_pred'])
  scored_bank(m,run,plan,cal,out/'bank',{k:'0'*64 for k in cb.IDENTITY_KEYS});records=list(cb.read_records(out/'bank'))
  assert len(records)==2 and records[-1]['index']==513 and len(records[-1]['prediction'])==32
  dump(out/'RECEIPT.json',dict(status='CPU_SCHEMA_PASS',records=2,shared_initial_exact=mapping['shared_exact'],labels_read=False,GPUh=0,architecture=a.ARCHITECTURE,implementation_sha256=sha(a.__file__)))
  print(json.dumps(dict(status='CPU_SCHEMA_PASS',records=2,GPUh=0)));return
 assert c['execution_status']=='EXECUTION_ENABLED', 'Implementation stage only; no formal GPU authorized by this config'
 prop=torch.cuda.get_device_properties(0);assert str(prop.uuid).removeprefix('GPU-')==c['expected_uuid'] and 'A40' in prop.name
 limit=c['allocator_GiB']*1024**3;torch.cuda.set_per_process_memory_fraction(limit/prop.total_memory,0);torch.cuda.reset_peak_memory_stats(0)
 cfg=read(c['original_config']['path']);cfg['device']='cuda:0';cfg['outputs']=c['outputs'];cfg['candidate_architecture']=dict(architecture=a.ARCHITECTURE,implementation_sha256=sha(a.__file__),loss_weights=c['loss_weights'])
 out=Path(c['output']);start=time.monotonic()
 if c['phase']=='l0':out.mkdir(parents=True,exist_ok=False);receipt=finite_l0(c,cfg,old,a,out)
 elif c['phase']=='train':
  oldconstruct,oldfit=old.construct,old.fit;capture={}
  def construct(settings):
   m,mapping=a.construct(settings,shared_initial=c['initial']['path'],expected_initial_sha256=c['initial']['sha256']);capture['model']=m
   dump(out/'INITIAL_MAPPING.json',mapping);torch.save(readout._rng(),out/'INITIAL_RNG.pt');return m
  old.construct,old.fit=construct,a.fit
  try:old.source_action(cfg,'source_train',out)
  finally:old.construct,old.fit=oldconstruct,oldfit
  report=read(out/'source_training.json');assert report['attempts']==report['updates']==8000
  dump(out/'CHECKPOINT_IDENTITY.json',metadata(capture['model'],out/'source.pt',a))
  receipt=dict(scientific_status='VALID_SOURCE_SELECTED_ENDPOINT',attempts=8000,updates=8000,selected_attempt=report['selected_attempt'],checkpoint_sha256=sha(out/'source.pt'),shared_initial_sha256=c['initial']['sha256'])
 elif c['phase']=='cal':
  train=read(c['training_receipt']['path']);assert train['scientific_status']=='VALID_SOURCE_SELECTED_ENDPOINT' and train['updates']==8000 and train['architecture']==a.ARCHITECTURE
  cfg.update(checkpoint=c['checkpoint']['path'],checkpoint_sha256=c['checkpoint']['sha256'],train_receipt=c['training_receipt']['path'],train_receipt_sha256=c['training_receipt']['sha256'])
  base=old.load_model;old.load_model=lambda settings,path:a.load_checkpoint(settings,path,c['checkpoint_identity']['path'])
  try:old.source_action(cfg,'calibrate_source',out)
  finally:old.load_model=base
  assert len(read(out/'source_calibration_records.json'))==1024
  receipt=dict(scientific_status='VALID_SOURCE_CAL',records=1024,checkpoint_sha256=c['checkpoint']['sha256'],calibration_sha256=sha(out/'source_calibration.json'))
 elif c['phase']=='query':
  out.mkdir(parents=True,exist_ok=False);m=a.load_checkpoint(cfg,c['checkpoint']['path'],c['checkpoint_identity']['path'])
  runs=old.load_manifest(c['query_manifest']['path'],roles={'QUERY'});assert len(runs)==1;run=runs[0]
  assert (run.domain,run.entity,run.run,run.start,run.values.shape)==('PSM','anonymous_PSM','test_original',0,(87841,25))
  plan=[dict(domain=run.domain,entity=run.entity,run=run.run,role='QUERY',index=i) for i in range(*c['index_range'])]
  scored_bank(m,run,plan,read(c['calibration']['path']),out/'bank',c['bank_identity'])
  receipt=dict(scientific_status='COMPLETE_SHARD_PENDING_RAW_MERGE',points=len(plan),index_range=c['index_range'],bank_manifest_sha256=sha(out/'bank/manifest.json'),checkpoint_sha256=c['checkpoint']['sha256'],calibration_sha256=c['calibration']['sha256'])
 else:raise ValueError(c['phase'])
 for path,digest in c['immutable_sha256'].items():assert sha(path)==digest,path
 dump(out/'WORKER_GPU_PEAK.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(0),reserved_bytes=torch.cuda.max_memory_reserved(0),allocator_limit_bytes=limit))
 receipt.update(status='COMPLETED',phase=c['phase'],architecture=a.ARCHITECTURE,implementation_sha256=sha(a.__file__),seed=2025,labels_read=False,config_sha256=z.trusted_config_sha256,seconds=time.monotonic()-start)
 dump(out/'RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
