import argparse,sys,json,time,hashlib,random,importlib.util
from pathlib import Path
import numpy as np,torch
from types import SimpleNamespace
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name;ROOT=R/'preparations/left_baseline_build_20261101_v1/upstream'
sys.path.insert(0,str(R/'preparations/left_baseline_build_20261101_v1/deps_thop_transport2_v1/site'));sys.path.insert(0,str(P/'runtime_v2'));from left_native_history_lag import candidate_class,ARMS
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def module(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
adapter=module(R/'results/left_psm_R_seed2025_20261101_v1/adapter_snapshot.py','training_adapter');scoring=module(R/'results/left_psm_R_score_seed2025_20261101_v1/adapter_snapshot.py','scoring_adapter')
OriginalDataset=adapter.NormalPrefixDataset
class HistoryDataset(OriginalDataset):
 def __getitem__(self,index):
  self.footprint(index);end=index*self.step+192;start=max(0,end-513);v=np.full((513,self.values.shape[1]),np.nan,np.float32);v[-(end-start):]=np.float32(self.values[start:end]);assert np.isfinite(v[-192:]).all()
  return v,np.zeros((192,1),np.float32)
def common_hashes(model):return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()if not k.startswith('history_context.')and not k.endswith(('total_ops','total_params'))}
def train(out,arm):
 cfg=json.loads((R/'results/left_psm_R_seed2025_20261101_v1/train_R_v4.APPROVED.json').read_text());cfg['output']=str(out);cfg['max_seconds']=21600;adapter.NormalPrefixDataset=HistoryDataset;original_runtime=adapter.model_runtime
 def runtime(c,root):
  torch,native,loaded=original_runtime(c,root);official=adapter.import_official(root,'exp.exp_anomaly_detection');Hybrid=candidate_class(native,arm);official.Left=Hybrid;oldbasic=official.Exp_Anomaly_Detection._basic;expected=next(v for v in json.loads((R/'results/left_psm_R_seed2025_20261101_v1/normal_events.json').read_text())if v['kind']=='official_basic')
  def basic(exp):
   for when in ['before']:
    assert common_hashes(exp.model)=={k:v for k,v in expected[when]['state_sha256'].items()if not k.endswith(('total_ops','total_params'))}
    assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()==expected[when]['torch_rng_sha256'];assert hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()==expected[when]['cuda_rng_sha256']
   exp.model.allow_basic_padding=True
   try:result=oldbasic(exp)
   finally:exp.model.allow_basic_padding=False
   assert common_hashes(exp.model)=={k:v for k,v in expected['after']['state_sha256'].items()if not k.endswith(('total_ops','total_params'))};assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()==expected['after']['torch_rng_sha256'];assert hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()==expected['after']['cuda_rng_sha256']
   dump(out/'INITIAL_BASIC_PARITY.json',dict(shared_native_initial_exact=True,shared_after_basic_exact=True,CPU_CUDA_RNG_after_basic_exact=True,original192random_draws_internal_pad513=True,new_branch_initial_output_zero=True,arm=arm))
   return result
  official.Exp_Anomaly_Detection._basic=basic;return torch,Hybrid,loaded
 adapter.model_runtime=runtime;out.mkdir(parents=True,exist_ok=False);report={};adapter.train(cfg,ROOT,out,report);dump(out/'TRAIN_REPORT.json',report)
 pt=out/'model/PSM/Left/checkpoints_0/checkpoint.pth';return dict(checkpoint_sha256=sha(pt),final_step=report['final_python_step'],epochs=len([e for e in report['validation_events']if e['kind']=='normal_validation']))
def windows(v,ends,start=0):
 x=np.full((len(ends),513,v.shape[1]),np.nan,np.float32)
 for i,t in enumerate(ends):
  lo=max(start,int(t)-512);seq=np.float32(v[lo:t+1]);x[i,-len(seq):]=seq
 return x
def load(arm):
 sys.path.insert(0,str(ROOT));from models.Left.LEFT import Left
 random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed(2025);model=candidate_class(Left,arm)(SimpleNamespace(**json.loads((ROOT/'configs/PSM/model_configs_Left.json').read_text()))).cuda();pt=O/arm/'train/model/PSM/Left/checkpoints_0/checkpoint.pth';r=model.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False);assert not r.missing_keys and all(k.endswith(('total_ops','total_params'))for k in r.unexpected_keys);model.eval();return model,sha(pt)
def scores(model,v,ends,start=0,capture_dir=None):
 out=[];fixed_cases={512,2048,8192,16384,32768,49152,65536,87840}
 with torch.no_grad():
  for a in range(0,len(ends),256):
   ids=ends[a:a+256];rows=[j for j,t in enumerate(ids)if int(t)in fixed_cases]if capture_dir else[];model.capture_rows=rows if rows else None
   x=torch.from_numpy(windows(v,ids,start)).cuda();raw,_=model.infer(x,None,None,None);_,s=scoring.author_numpy_scores([raw.cpu().numpy()]);out.append(s.reshape(len(x),192)[:,-1])
   if rows:
    arrays={k:z.cpu().numpy()for k,z in model.captured.items()};arrays.update(index=ids[rows],input=x[rows].cpu().numpy(),native_score=raw[rows].cpu().numpy());np.savez_compressed(capture_dir/('same_forward_'+str(int(ids[0]))+'.npz'),**arrays)
   model.capture_rows=None;model.captured=None;model.history_context.captured=None
 return np.concatenate(out)
def main():
 a=argparse.ArgumentParser();a.add_argument('--phase',required=True);a.add_argument('--arm',required=True,choices=ARMS);args=a.parse_args();phase=args.phase;arm=args.arm;torch.set_num_threads(1);torch.set_num_interop_threads(1);mem=20 if phase in('train','l0')else 30;torch.cuda.set_per_process_memory_fraction(mem*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats(0);out=O/arm/phase;start=time.monotonic()
 if phase=='train':report=train(out,arm)
 else:
  out.mkdir(parents=True,exist_ok=False)
  if phase=='l0':
   sys.path.insert(0,str(ROOT));from models.Left.LEFT import Left
   torch.manual_seed(2025);model=candidate_class(Left,arm)(SimpleNamespace(**json.loads((ROOT/'configs/PSM/model_configs_Left.json').read_text()))).cuda();model.train();opt=torch.optim.Adam(model.parameters(),lr=1e-4);x=torch.randn(128,513,25,device='cuda');steps=[]
   for i in range(2):
    opt.zero_grad();loss=model(x,None,None,None)[0];assert torch.isfinite(loss);loss.backward();g={k:float(v.grad.abs().sum())if v.grad is not None else None for k,v in model.named_parameters()if k in('history_context.input.weight','history_context.self_kernel','history_context.cross_kernel')};steps.append(g)
    if i==1:
     assert g['history_context.input.weight']>0
     if arm in('LS','LSC'):assert g['history_context.self_kernel']>0
     if arm in('LC','LSC'):assert g['history_context.cross_kernel']>0
    opt.step()
   model.eval()
   with torch.no_grad():
    s=model.infer(x[:2],None,None,None)[0];b=model.infer(x[:1],None,None,None)[0];p=model.infer(x[:2].flip(0),None,None,None)[0].flip(0);assert torch.allclose(s[:1],b,atol=1e-4,rtol=1e-4)and torch.allclose(s,p,atol=1e-4,rtol=1e-4)
   report=dict(discarded_updates=2,steps=steps,batch_tolerance='atol=rtol1e-4_native_float32_development_fixture',finite_B128=True)
  else:
   model,pt=load(arm);st=json.loads((R/'results/left_psm_R_seed2025_20261101_v1/SCALER_R.json').read_text())['statistics'];mean=np.asarray(st['mean']);scale=np.asarray(st['scale'])
   if phase=='cal':
    v=(np.load(R/'preparations/left_baseline_build_20261101_v1/normal_prefix_R_v1/prefix.npy')-mean)/scale;ends=np.linspace(106175,132480,512,dtype=np.int64);s=scores(model,v,ends,105984);tau=float(np.quantile(s,.99,method='higher'));np.savez_compressed(out/'scores.npz',index=ends,score=s);report=dict(records=512,threshold=tau,SELECT_CAL_reuse_exploratory=True,checkpoint_sha256=pt)
   elif phase=='query':
    v=(np.load(R/'data/common_psm_raw_20261029_v1/test.npy')-mean)/scale;ends=np.arange(512,87841);s=scores(model,v,ends,capture_dir=out);np.savez_compressed(out/'scores.npz',index=ends,score=s,available_index=ends,valid=np.ones(len(ends),bool));report=dict(records=87329,checkpoint_sha256=pt,causal513_input=True)
   else:raise ValueError(phase)
 report.update(status='COMPLETED',phase=phase,arm=arm,labels_read=False,seconds=time.monotonic()-start);dump(out/'RECEIPT.json',report);dump(out/'GPU_PEAK.json',dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()));print(json.dumps(report),flush=True)
if __name__=='__main__':main()

