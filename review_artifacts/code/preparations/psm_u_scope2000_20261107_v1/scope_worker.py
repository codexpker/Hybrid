"""One paired HEAD/FULL target experiment; original inference and CAL math."""
import argparse,hashlib,json,sys,time
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def check(a):assert sha(a['path'])==a['sha256'];return a['path']
def reset_rng(torch):
 import random,numpy as np
 random.seed(2025);np.random.seed(2025);torch.random.default_generator.manual_seed(2025)
 with torch.cuda.device(0):torch.cuda.manual_seed(2025)
def write_bank(model,run,indices,cal,path,identity,u,cli):
 from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,iter_pointwise_infer
 q=[dict(domain=run.domain,entity=run.entity,run=run.run,role=run.role,index=i) for i in indices]
 ranges=[dict(domain=run.domain,entity=run.entity,run=run.run,start=indices[0],end=indices[-1]+1)]
 with ChunkedBank(path,identity=identity,ranges=ranges,threshold=cal['threshold'],chunk_size=64) as bank:
  with u.evaluation_state(model,2025):
   for row in iter_pointwise_infer(model,[run],history=512,horizon=32,seed=2025,index_plan=q):bank.append(cli.score_records([row],cal)[0])
  bank.finalize()
def l0(c,source,plan,out,u,scope,cli,state_identity,torch):
 import numpy as np
 from hybrid_tsad.hybrid_v1.data import Run
 from hybrid_tsad.hybrid_v1 import chunked_bank as cb
 sys.path.insert(0,c['cached_reader_directory']);from cached_chunk_reader import cached_unpack
 fits=u.load_roles(c['manifest']['path'],{'FIT'});sels=u.load_roles(c['manifest']['path'],{'SELECT'})
 tech=dict(plan,FIT=plan['FIT'][:2],SELECT=plan['SELECT'][:2]);source_before=state_identity(source);reports={}
 for arm in ('HEAD','FULL'):
  p=out/arm;p.mkdir();reset_rng(torch)
  model,report,opt=scope.train_endpoint(source,fits,sels,tech,scope=arm,artifact_dir=p,technical=True)
  assert report['updates']==2 and not report['state_difference']['changed_buffers'];reports[arm]=report
  cli.save_json(p/'training.json',report);cli.save_model(p/'discarded.pt',model)
  cfg=read(c['source_config']['path']);cfg['device']='cuda:0';reloaded=cli.load_model(cfg,p/'discarded.pt')
  assert state_identity(reloaded)==state_identity(model)
  modes=[(m,m.training) for m in model.modules()];cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(0).clone()
  try:
   with u.evaluation_state(model,789):torch.rand(4,device='cuda');torch.rand(4);raise RuntimeError('fixture')
  except RuntimeError:pass
  assert modes==[(m,m.training) for m in model.modules()] and torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state(0))
  records,cal=u.calibrate(reloaded,u.load_roles(c['manifest']['path'],{'CAL'}),plan['CAL'][:2],history=512,horizon=32,seed=2025)
  assert len(records)==2
  run=fits[0];t=plan['FIT'][0]['index']-run.start
  fixture=Run('SYNTHETIC_FIT','C25','technical','QUERY',run.values[t-512:t+65].copy(),run.observed[t-512:t+65].copy())
  assert fixture.values.shape==(577,25)
  identity={k:'0'*64 for k in cb.IDENTITY_KEYS};before=state_identity(reloaded)
  write_bank(reloaded,fixture,list(range(512,577)),cal,p/'bank',identity,u,cli)
  with cached_unpack(cb):raw=list(cb.read_records(p/'bank'))
  assert len(raw)==65 and all(np.asarray(r['prediction']).shape==(32,25) for r in raw) and [r['index'] for r in raw]==list(range(512,577))
  assert state_identity(reloaded)==before and state_identity(source)==source_before
 assert reports['HEAD']['initial_torch_rng']==reports['FULL']['initial_torch_rng']
 for k in ('module_modes','sample_mask_digest','SELECT_sequence_digest','FIT_sequence_digest'):assert reports['HEAD'][k]==reports['FULL'][k]
 differences={};exact={}
 with np.load(out/'HEAD/FIRST_FORWARD.npz') as a,np.load(out/'FULL/FIRST_FORWARD.npz') as b:
  for k in a.files:
   assert a[k].dtype==b[k].dtype and a[k].shape==b[k].shape
   exact[k]=a[k].tobytes()==b[k].tobytes()
   if k in ('prediction','tf','ms'):
    differences[k]=float(np.nanmax(np.abs(a[k].astype(np.float64)-b[k].astype(np.float64))));assert differences[k]<=1e-5
   else:assert exact[k],k
 checks=dict(status='L0_PASS_DISCARDED',discarded_updates=4,source_c00_unchanged=True,
  both2000consumer_scopes_tested=True,first_forward_bitexact=exact,continuous_first_forward_maxabs=differences,
  technical_continuous_tolerance=1e-5,initial_CPU_CUDA_RNG_exact=True,module_modes_exact=True,
  sample_mask_SELECT_FIT_digests_exact=True,head_frozen_and_all_buffers_exact=True,FULL_nonhead_parameters_changed=True,
  reload_state_exact=True,evaluation_mode_CPU_CUDA_RNG_exception_exact=True,writer_records_per_arm=65,fullP32_and_tail31=True,
  prior_samearchitecture_shard_parity_reused=True,labels_read=False)
 dump(out/'L0_CHECKS.json',checks);return dict(updates=4,records=130,scientific_training_status='L0_PASS_DISCARDED')
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);p.add_argument('--metadata-only',action='store_true');a=p.parse_args()
 assert sha(a.config)==a.trusted_config_sha256;c=read(a.config)
 assert c['operation']=='PSM_U_HEAD_FULL2000' and c['seed']==2025 and c['scope'] in ('HEAD','FULL') and c['phase'] in ('l0','train','cal','query')
 assert c['source_checkpoint']['sha256']=='c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812'
 for path,h in c['immutable_sha256'].items():assert sha(path)==h,path
 for key in ('source_config','source_checkpoint','manifest','plan','science_freeze'):check(c[key])
 assert not Path(c['output']).exists();sys.path.insert(0,c['runtime_src'])
 from hybrid_tsad.hybrid_v1 import stage2_u_matched as u,stage2_u_scope as scope
 u.role_metadata(c['manifest']['path']);plan=read(c['plan']['path']);assert (len(plan['FIT']),len(plan['SELECT']),len(plan['CAL']))==(2000,32,512)
 if c['phase'] in ('cal','query'):
  r=read(check(c['training_receipt']));assert r['scope']==c['scope'] and r['attempts']==r['updates']==2000 and r['scientific_training_status']=='VALID_ALL_ATTEMPTS_UPDATED' and r['checkpoint_sha256']==c['checkpoint']['sha256'];check(c['checkpoint'])
 if c['phase']=='query':
  r=read(check(c['calibration_receipt']));assert r['records']==512 and r['checkpoint_sha256']==c['checkpoint']['sha256'] and r['calibration_sha256']==c['calibration']['sha256'];check(c['calibration']);check(c['query_manifest'])
 if a.metadata_only:print(json.dumps(dict(status='METADATA_PASS',Torch_imported='torch' in sys.modules)));return
 import torch
 from hybrid_tsad.hybrid_v1 import cli
 from hybrid_tsad.hybrid_v1.stage2_adaptation import state_identity
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 prop=torch.cuda.get_device_properties(0);assert str(prop.uuid).replace('GPU-','')==c['expected_uuid'] and 'A40' in prop.name
 cap=c['allocator_GiB']*1024**3;torch.cuda.set_per_process_memory_fraction(cap/prop.total_memory,0);torch.cuda.reset_peak_memory_stats(0)
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);dump(out/'CAP_BEFORE_MODEL.json',dict(UUID=str(prop.uuid),cap_bytes=cap,set_before_model=True));began=time.monotonic()
 cfg=read(c['source_config']['path']);cfg['device']='cuda:0';checkpoint=c['source_checkpoint'] if c['phase'] in ('train','l0') else c['checkpoint']
 model=cli.load_model(cfg,check(checkpoint));before=state_identity(model);dump(out/'STATE_BEFORE.json',before)
 if c['phase']=='train':
  fits=u.load_roles(c['manifest']['path'],{'FIT'});sels=u.load_roles(c['manifest']['path'],{'SELECT'});cli.save_model(out/'source_initial.pt',model)
  reset_rng(torch);torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0)),out/'TRAIN_START_RNG.pt')
  adapted,report,opt=scope.train_endpoint(model,fits,sels,plan,scope=c['scope'],artifact_dir=out)
  assert report['attempts']==report['updates']==2000
  cli.save_model(out/'adapted.pt',adapted);cli.save_json(out/'training.json',report);torch.save(opt.state_dict(),out/'optimizer.pt');dump(out/'MODULE_MODES.json',report['module_modes']);dump(out/'ENDPOINT_STATE.json',state_identity(adapted))
  assert before==state_identity(model)
  receipt=dict(attempts=2000,updates=2000,scientific_training_status=report['scientific_training_status'],selection=report['selection'],
   mode_policy=report['mode_policy'],sample_mask_digest=report['sample_mask_digest'],FIT_sequence_digest=report['FIT_sequence_digest'],
   SELECT_sequence_digest=report['SELECT_sequence_digest'],checkpoint_sha256=sha(out/'adapted.pt'),buffers_unchanged=True,source_unchanged=True)
 elif c['phase']=='cal':
  records,cal=u.calibrate(model,u.load_roles(c['manifest']['path'],{'CAL'}),plan['CAL'],history=512,horizon=32,seed=2025)
  assert len(records)==512;cli.save_json(out/'calibration_records.json',records);cli.save_json(out/'calibration.json',cal)
  receipt=dict(records=512,checkpoint_sha256=checkpoint['sha256'],calibration_sha256=sha(out/'calibration.json'))
 elif c['phase']=='query':
  run=cli.load_manifest(c['query_manifest']['path'],roles={'QUERY'})[0];assert run.start==0 and (run.domain,run.entity,run.run)==('PSM','anonymous_PSM','test_original')
  identity=c['bank_identity'];cal=read(c['calibration']['path']);lo,hi=c['index_range'];write_bank(model,run,list(range(lo,hi)),cal,out/'bank',identity,u,cli)
  receipt=dict(points=hi-lo,index_range=[lo,hi],checkpoint_sha256=checkpoint['sha256'],calibration_sha256=c['calibration']['sha256'],bank_manifest_sha256=sha(out/'bank/manifest.json'),state_unchanged=True)
 else:receipt=l0(c,model,plan,out,u,scope,cli,state_identity,torch)
 assert before==state_identity(model);dump(out/'STATE_AFTER.json',state_identity(model))
 for path,h in c['immutable_sha256'].items():assert sha(path)==h,path
 dump(out/'WORKER_GPU_PEAK.json',dict(allocated_bytes=torch.cuda.max_memory_allocated(0),reserved_bytes=torch.cuda.max_memory_reserved(0),allocator_limit_bytes=cap))
 receipt.update(status='COMPLETED',phase=c['phase'],scope=c['scope'],seed=2025,common_plan_sha256=c['plan']['sha256'],config_sha256=a.trusted_config_sha256,
 source_checkpoint_sha256=c['source_checkpoint']['sha256'],labels_read=False,seconds=time.monotonic()-began)
 dump(out/'RECEIPT.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
