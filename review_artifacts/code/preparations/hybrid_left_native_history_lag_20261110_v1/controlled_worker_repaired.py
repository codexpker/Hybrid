"""Same-distribution controlled normal training for native N and frozen L family."""
import sys,json,random,time,hashlib,importlib.util,copy,bisect
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name/'controlled';ROOT=R/'preparations/left_baseline_build_20261101_v1/upstream'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(P/'runtime_v2'));sys.path.insert(0,str(R/'preparations/left_baseline_build_20261101_v1/deps_thop_transport2_v1/site'))
from models.Left.LEFT import Left
from left_native_history_lag import candidate_class
from native_score_numpy import native_numpy_scores
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_anomaly_detection import Configs
s=importlib.util.spec_from_file_location('score_adapter',R/'results/left_psm_R_score_seed2025_20261101_v1/adapter_snapshot.py');sa=importlib.util.module_from_spec(s);s.loader.exec_module(sa)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
MANIFEST=json.loads((P/'CONTROLLED_DATA_MANIFEST.json').read_text());STATS=json.loads((P/'CONTROLLED_NORMALIZATION.json').read_text())
def values(e):
 assert sha(e['path'])==e['sha256']
 with np.load(e['path'])as z:v=z['values'].astype(np.float64)
 return (v-np.asarray(STATS['mean']))/np.asarray(STATS['scale'])
class Native513(Left):
 def __init__(self,cfg):super().__init__(cfg);self.allow_basic_padding=False;self.capture_rows=None;self.captured=None
 def _forward(self,x,*args):
  if x.shape[1]==192:
   if not self.allow_basic_padding:raise ValueError('192only_basic')
  elif x.shape[1]!=513:raise ValueError('513required')
  x=x[:,-192:];assert torch.isfinite(x).all();r=super()._forward(x,*args)
  self.captured=None if self.capture_rows is None else dict(native_freq_to_time=r[3][self.capture_rows].detach(),native_time=r[4][self.capture_rows].detach(),native_target=r[5][self.capture_rows].detach(),native_MS=r[9][self.capture_rows].detach())
  return r
CLASSES={a:candidate_class(Left,a)for a in ('L','LS','LC','LSC')};CLASSES['N']=Native513
class RunsDataset(Dataset):
 def __init__(self,role,step):
  self.entries=[e for e in MANIFEST['roles']if e['role']==role];self.runs=[values(e).astype(np.float32)for e in self.entries];self.step=step;self.counts=[(len(v)-192)//step+1 for v in self.runs];self.cumulative=np.cumsum(self.counts).tolist()
 def __len__(self):return self.cumulative[-1]
 def __getitem__(self,i):
  j=bisect.bisect_right(self.cumulative,i);k=i-(self.cumulative[j-1]if j else 0);end=k*self.step+192;lo=max(0,end-513);x=np.full((513,4),np.nan,np.float32);x[-(end-lo):]=self.runs[j][lo:end];assert np.isfinite(x[-192:]).all();return x,np.zeros((192,1),np.float32)
def hashstate(m):return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in m.state_dict().items()if not k.startswith('history_context.')and not k.endswith(('total_ops','total_params'))}
def reset():random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed(2025)
def train(out,arm):
 reset();events=[]
 class Exp(Exp_Anomaly_Detection):
  def _build_model(self):
   m=CLASSES[arm](self.hyper_args);actual_rng=torch.get_rng_state().clone()
   with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(2025);ref=Left(self.hyper_args);ref_rng=torch.get_rng_state().clone()
   assert torch.equal(actual_rng,ref_rng)and hashstate(m)==hashstate(ref)
   return m
  def _get_data(self,flag,step=None):
   role='FIT'if flag=='train'else'SELECT';ds=RunsDataset(role,1 if flag=='train'else 192);events.append(dict(kind='loader',role=role,runs=[e['meta']['seed']for e in ds.entries],windows=len(ds),step=ds.step))
   return ds,DataLoader(ds,batch_size=128,shuffle=flag=='train',drop_last=False,num_workers=0)
  def _basic(self):
   before=dict(common_state=hashstate(self.model),CPU_rng=sha_bytes(torch.get_rng_state().numpy().tobytes()),CUDA_rng=sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes()));self.model.allow_basic_padding=True
   try:r=super()._basic()
   finally:self.model.allow_basic_padding=False
   after=dict(common_state=hashstate(self.model),CPU_rng=sha_bytes(torch.get_rng_state().numpy().tobytes()),CUDA_rng=sha_bytes(torch.cuda.get_rng_state().cpu().numpy().tobytes()));dump(out/'INITIAL_BASIC.json',dict(before=before,after=after,shared_native_init_checked=True));return r
  def vali(self,loader):
   v=super().vali(loader);events.append(dict(kind='validation',loss=float(v),step=self.model.step));return v
  def test(self,**kw):raise PermissionError('No query during train')
 args=SimpleNamespace(data='CONTROLLED',data_origin='DADA',model_class='Left',gpu=0,use_gpu=True,gpu_type='cuda',use_multi_gpu=False,configs_path=str(P/'controlled_configs'),model_save_path=str(out/'model'),results_save_path=str(out/'unused_results'))
 out.mkdir(parents=True,exist_ok=False);exp=Exp(args,id=0)
 try:exp.train()
 finally:dump(out/'EVENTS.json',events)
 pt=out/'model/CONTROLLED/Left/checkpoints_0/checkpoint.pth';return dict(epochs=len([e for e in events if e['kind']=='validation']),steps=exp.model.step,checkpoint_sha256=sha(pt))
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def load(arm):
 reset();cfg=SimpleNamespace(**json.loads((P/'controlled_configs/CONTROLLED/model_configs_Left.json').read_text()));m=CLASSES[arm](cfg).cuda();pt=O/arm/'train/model/CONTROLLED/Left/checkpoints_0/checkpoint.pth';r=m.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False);assert not r.missing_keys and all(k.endswith(('total_ops','total_params'))for k in r.unexpected_keys);m.eval();return m,sha(pt)
def windows(v,ids):return np.stack([np.float32(v[t-512:t+1])for t in ids])
def score(m,v,ids,capture=None):
 parts=[];represent={512,2048,3072,3584,4608,5120,6144,8191}
 with torch.no_grad():
  for i in range(0,len(ids),256):
   idx=ids[i:i+256];sel=[j for j,t in enumerate(idx)if int(t)in represent]if capture else[];m.capture_rows=sel if sel else None;x=torch.from_numpy(windows(v,idx)).cuda();raw,_=m.infer(x,None,None,None);_,s=native_numpy_scores([raw.cpu().numpy()],4);parts.append(s.reshape(len(x),192)[:,-1])
   if sel:
    a={k:z.cpu().numpy()for k,z in m.captured.items()};a.update(index=idx[sel],input=x[sel].cpu().numpy(),score=raw[sel].cpu().numpy());np.savez_compressed(capture/('same_forward_'+str(int(idx[0]))+'.npz'),**a)
   m.capture_rows=None;m.captured=None
   if hasattr(m,'history_context'):m.history_context.captured=None
 return np.concatenate(parts)
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=CLASSES);p.add_argument('--phase',required=True,choices=('train','cal','query'));a=p.parse_args();arm=a.arm;phase=a.phase;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_per_process_memory_fraction((20 if phase=='train'else 30)*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats(0);out=O/arm/('cal_repaired'if phase=='cal'else phase);start=time.monotonic()
 if phase=='train':report=train(out,arm)
 else:
  out.mkdir(parents=True,exist_ok=False);m,pt=load(arm)
  if phase=='cal':
   bank=[]
   for i,e in enumerate(e for e in MANIFEST['roles']if e['role']=='CAL'):
    v=values(e);ids=np.linspace(512,len(v)-1,256,dtype=np.int64);s=score(m,v,ids);bank.append(s);np.savez_compressed(out/('normal_run_'+str(i)+'.npz'),index=ids,score=s)
   threshold=float(np.quantile(np.concatenate(bank),.99,method='higher'));report=dict(records=512,threshold=threshold,normal_CAL_runs_independent_of_SELECT=True,checkpoint_sha256=pt)
  else:
   manifest=[]
   for e in MANIFEST['roles']:
    if e['role']!='QUERY':continue
    v=values(e);ids=np.arange(512,len(v));folder=out/(e['meta']['case']+'_'+str(e['meta']['seed']));folder.mkdir();s=score(m,v,ids,folder);f=folder/'scores.npz';np.savez_compressed(f,index=ids,score=s,available_index=ids,valid=np.ones(len(ids),bool));manifest.append(dict(case=e['meta']['case'],seed=e['meta']['seed'],path=str(f),sha256=sha(f),truth=e['truth']))
   dump(out/'BANKS.json',manifest);report=dict(runs=len(manifest),points=40*7680,checkpoint_sha256=pt)
 report.update(status='COMPLETED',phase=phase,arm=arm,labels_read=False,seconds=time.monotonic()-start);dump(out/'RECEIPT.json',report);dump(out/'GPU_PEAK.json',dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()));print(json.dumps(report),flush=True)
if __name__=='__main__':main()

