import argparse,json,hashlib,sys,os,time,random
from pathlib import Path
import numpy as np
R=Path('__RESEARCH_ROOT__');P=Path(__file__).resolve().parent;O=R/'results'/P.name
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def prepare():
 import shutil
 assert not (P/'DATA_READY.json').exists()
 src=R/'preparations/psm_u_scope2000_20261107_v1/runtime_v1/src';shutil.copytree(src,P/'runtime/src')
 shutil.copy2(P/'target_fullshot.py',P/'runtime/src/hybrid_tsad/hybrid_v1/target_fullshot.py')
 x=np.load(R/'preparations/left_baseline_build_20261101_v1/normal_prefix_R_v1/prefix.npy');assert x.shape==(132481,25)
 scaler=json.loads((R/'results/left_psm_R_seed2025_20261101_v1/SCALER_R.json').read_text());print(scaler['statistics'].keys(),flush=True)
 mean=np.asarray(scaler['statistics']['mean']);scale=np.asarray(scaler['statistics']['scale'])
 assert np.allclose(x.mean(axis=0),mean,rtol=1e-12,atol=1e-12)
 assert np.allclose(x.std(axis=0),scale,rtol=1e-10,atol=1e-12)
 d=P/'roles';d.mkdir();entries=[]
 for role,lo,hi in [('FIT',0,105984),('SELECT',105984,118960),('CAL',119504,132481)]:
  v=((x[lo:hi]-mean)/scale).astype('float32');path=d/(role+'.npz');np.savez(path,values=v,observed=np.isfinite(v))
  entries.append(dict(domain='PSM',entity='anonymous_PSM',run='train_original',role=role,path=str(path),sha256=sha(path),expected_shape=list(v.shape),original_start=lo))
 q=np.load(R/'data/common_psm_raw_20261029_v1/test.npy');assert q.shape==(87841,25)
 v=((q-mean)/scale).astype('float32');path=d/'QUERY.npz';np.savez(path,values=v,observed=np.isfinite(v))
 entries.append(dict(domain='PSM',entity='anonymous_PSM',run='test_original',role='QUERY',path=str(path),sha256=sha(path),expected_shape=list(v.shape),original_start=0))
 dump(P/'MANIFEST.json',dict(runs=entries));rng=np.random.default_rng(2025);plan={}
 for role,n,arrived in [('FIT',8000,False),('SELECT',64,False),('CAL',512,True)]:
  e=next(e for e in entries if e['role']==role);N=e['expected_shape'][0];end=N if arrived else N-32+1
  ts=rng.integers(512,end,size=n) if role=='FIT' else np.linspace(512,end-1,n,dtype=int)
  plan[role]=[dict(domain=e['domain'],entity=e['entity'],run=e['run'],role=role,index=int(t)+e['original_start'],footprint=[int(t)+e['original_start']-512,int(t)+e['original_start']+(1 if arrived else 32)]) for t in ts]
 dump(P/'PLAN.json',plan);dump(P/'DATA_READY.json',dict(prefix_sha=sha(R/'preparations/left_baseline_build_20261101_v1/normal_prefix_R_v1/prefix.npy'),query_sha=sha(R/'data/common_psm_raw_20261029_v1/test.npy'),scaler_sha=sha(R/'results/left_psm_R_seed2025_20261101_v1/SCALER_R.json'),full_pretest_statistics_include_SELECT_CAL=True,query_labels_read=False,plan_sha=sha(P/'PLAN.json'),manifest_sha=sha(P/'MANIFEST.json'),FIT=8000,SELECT=64,CAL=512));print('CPU_DATA_PREP_PASS',flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--shard',type=int);a=p.parse_args()
 if a.phase=='prepare':return prepare()
 sys.path.insert(0,str(P/'runtime/src'));sys.path.insert(0,str(R/'preparations/psm_u_scope2000_20261107_v1'))
 import torch
 from hybrid_tsad.hybrid_v1 import cli,stage2_u_matched as u,target_fullshot as f
 from hybrid_tsad.hybrid_v1.stage2_adaptation import state_identity
 from scope_worker import write_bank
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction((20 if a.phase in ('train','l0') else 30)*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats(0)
 random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed(2025)
 phase=a.phase+('_'+str(a.shard) if a.phase=='query' else '');out=O/phase;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 old=json.loads((R/'preparations/psm_u_scope2000_20261107_v1/FULL_train.CONFIG.json').read_text());cfg=json.loads(Path(old['source_config']['path']).read_text());cfg['device']='cuda:0'
 checkpoint=Path(old['source_checkpoint']['path']) if a.phase in ('train','l0') else O/'train/selected.pt'
 model=cli.load_model(cfg,checkpoint);before=state_identity(model);plan=json.loads((P/'PLAN.json').read_text());manifest=P/'MANIFEST.json'
 if a.phase in ('train','l0'):
  report=f.train(model,f.load_roles(manifest,{'FIT'}),f.load_roles(manifest,{'SELECT'}),plan,out,technical=a.phase=='l0')
  assert report['updates']==(2 if a.phase=='l0' else 8000)
  reloaded=cli.load_model(cfg,out/'selected.pt');assert all(torch.isfinite(v).all() for v in reloaded.state_dict().values())
  if a.phase=='l0':
   records,cal=u.calibrate(reloaded,f.load_roles(manifest,{'CAL'}),plan['CAL'][:2],history=512,horizon=32,seed=2025);assert len(records)==2
   from hybrid_tsad.hybrid_v1.data import Run
   fit=f.load_roles(manifest,{'FIT'})[0];run=Run('PSM','anonymous_PSM','test_original','QUERY',fit.values[:577],fit.observed[:577],0)
   write_bank(reloaded,run,list(range(512,577)),cal,out/'bank',{k:'0'*64 for k in ('checkpoint_sha256','calibration_sha256','input_sha256','source_sha256','config_sha256')},u,cli)
   assert state_identity(model)==before;report.update(records=65,discarded=True)
 elif a.phase=='cal':
  records,cal=u.calibrate(model,f.load_roles(manifest,{'CAL'}),plan['CAL'],history=512,horizon=32,seed=2025);assert len(records)==512
  cli.save_json(out/'calibration_records.json',records);cli.save_json(out/'calibration.json',cal);report=dict(records=512,checkpoint_sha256=sha(checkpoint),calibration_sha256=sha(out/'calibration.json'))
 else:
  assert a.phase=='query';cal=json.loads((O/'cal/calibration.json').read_text());run=f.load_roles(manifest,{'QUERY'})[0]
  borders=[512,22345,44177,66009,87841];lo,hi=borders[a.shard:a.shard+2]
  ident=dict(checkpoint_sha256=sha(checkpoint),calibration_sha256=sha(O/'cal/calibration.json'),input_sha256=sha(manifest),source_sha256=sha(P/'runtime/src/hybrid_tsad/hybrid_v1/target_fullshot.py'),config_sha256=sha(P/'PLAN.json'))
  write_bank(model,run,list(range(lo,hi)),cal,out/'bank',ident,u,cli);report=dict(points=hi-lo,index_range=[lo,hi],checkpoint_sha256=sha(checkpoint),calibration_sha256=ident['calibration_sha256'],bank_manifest_sha256=sha(out/'bank/manifest.json'))
 assert state_identity(model)==before
 dump(out/'GPU_PEAK.json',dict(allocated=torch.cuda.max_memory_allocated(0),reserved=torch.cuda.max_memory_reserved(0)))
 report.update(status='COMPLETED',phase=a.phase,labels_read=False,seconds=time.monotonic()-start);dump(out/'RECEIPT.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
