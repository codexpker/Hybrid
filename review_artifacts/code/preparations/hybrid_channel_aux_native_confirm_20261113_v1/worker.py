import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,hashlib,random
import numpy as np
import torch
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name
sys.path[:0]=[str(P/'runtime'),str(R/'preparations/left_baseline_build_20261101_v1/upstream')]
from models.Left.LEFT import Left
from hybrid_tsad.left_channel_blindspot import channel_blindspot_class
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def statehash(m):return{k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in m.state_dict().items()}
class Data:
    def __init__(self,allowed):self.allowed=set(allowed);self.roles=read(P/'ROLES.json');self.sc=read(P/'FIT_ONLY_SCALER.json');self.access=[];assert sha(P/'FIT_ONLY_SCALER.json')==read(P/'FREEZE.json')['scaler_sha256']
    def load(self,role):
        if role not in self.allowed:raise PermissionError(role)
        e=self.roles[role];p=R/e['path'];assert sha(p)==e['sha256']
        with np.load(p)as z:raw=z['values'];assert z['observed'].all();v=np.float32((raw-np.asarray(self.sc['mean']))/np.asarray(self.sc['scale']))
        assert v.shape==(e['length'],38)and np.isfinite(v).all();self.access.append(dict(role=role,sha256=e['sha256'],shape=list(v.shape),start=e['start']));return v,e['start']
def setup(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(2);torch.cuda.set_device(0);c=read(P/'MODEL.json');assert sha(P/'MODEL.json')==read(P/'FREEZE.json')['model_sha256'];m=channel_blindspot_class(Left,'blind')(SimpleNamespace(**c)).cuda();assert not any(isinstance(s,torch.nn.modules.batchnorm._BatchNorm)for s in m.modules());return m,c
def window(v,starts):return torch.from_numpy(np.stack([v[s:s+192]for s in starts])).cuda()
def select(m,v):
    modes={n:s.training for n,s in m.named_modules()};before=statehash(m);step=m.step;tr=torch.get_rng_state();cr=torch.cuda.get_rng_state();m.eval();starts=np.arange(0,len(v)-191,192);sums=0.;n=0
    try:
        with torch.no_grad():
            for a in range(0,len(starts),128):
                x=window(v,starts[a:a+128]);score=m.native_readout(x)[:,-1,:].mean(1);sums+=float(score.double().sum());n+=len(x)
        assert before==statehash(m)and step==m.step;return sums/n
    finally:
        for name,s in m.named_modules():s.training=modes[name]
        torch.set_rng_state(tr);torch.cuda.set_rng_state(cr)
def savept(p,m,opt,update,selection):
    q=p.with_suffix('.tmp');torch.save(dict(state=m.state_dict(),optimizer=opt.state_dict(),update=update,python_step=m.step,SELECT_native_score=selection),q);os.replace(q,p)
def train(seed,arm,out):
    m,c=setup(seed);data=Data(['FIT','SELECT']);fit,_=data.load('FIT');val,_=data.load('SELECT');source=None
    if arm!='source':
        source=O/str(seed)/'source/train/selected.pth';pt=torch.load(source,map_location='cpu',weights_only=True);m.load_state_dict(pt['state'],strict=True);m.step=int(pt['python_step'])
    initial=statehash(m);initial_step=m.step;opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    with np.load(P/('PLAN_'+str(seed)+'.npz'))as plan:draw=plan['source'if arm=='source'else'paired'];channels=plan['channels']
    limit=4000 if arm=='source'else 2000;trace=[];best=float('inf');chosen=None;t=time.monotonic();initial_select=select(m,val)
    for u in range(1,limit+1):
        m.train();x=window(fit,draw[u-1]);opt.zero_grad()
        loss=m(x,torch.from_numpy(channels[u-1]).cuda())[0]if arm=='aux'else Left.forward(m,x,None,None,None)[0]
        assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all()for p in m.parameters()if p.grad is not None);opt.step()
        if u%200==0:
            s=select(m,val);eligible=arm!='source'or u>=3000;row=dict(update=u,python_step=m.step,train_last_loss=float(loss.detach()),SELECT_native_score=s,eligible=eligible,seconds=time.monotonic()-t)
            if eligible and s<best:best=s;chosen=dict(update=u,python_step=m.step,SELECT_native_score=s);savept(out/'selected.pth',m,opt,u,s);row['selected']=True
            trace.append(row);dump(out/'TRAIN_TRACE.json',trace);print(json.dumps(dict(event='SELECT',seed=seed,arm=arm,**row)),flush=True)
    savept(out/('endpoint'+str(limit)+'.pth'),m,opt,limit,trace[-1]['SELECT_native_score'])
    dump(out/'TRAIN_RECEIPT.json',dict(status='COMPLETED',seed=seed,arm=arm,updates=limit,selected=chosen,initial_step=initial_step,initial_state=initial,initial_SELECT_native_score=initial_select,source_PT=str(source)if source else None,source_PT_sha256=sha(source)if source else None,selected_sha256=sha(out/'selected.pth'),plan_sha256=sha(P/('PLAN_'+str(seed)+'.npz')),roles=data.access,scaler_sha256=sha(P/'FIT_ONLY_SCALER.json'),seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),readout='native only; no mask SELECT'))
def score(seed,arm,out):
    m,c=setup(seed);p=out.parent/'train/selected.pth';pt=torch.load(p,map_location='cpu',weights_only=True);m.load_state_dict(pt['state'],strict=True);m.step=int(pt['python_step']);m.eval();before=statehash(m);step=m.step;data=Data(['CAL','QUERY']);banks=[];t=time.monotonic()
    for role in ['CAL','QUERY']:
        v,offset=data.load(role);ends=np.arange(512,len(v));parts=[]
        with torch.no_grad():
            for a in range(0,len(ends),128):
                x=window(v,ends[a:a+128]-191);parts.append(m.native_readout(x)[:,-1,:].cpu().numpy())
        channel=np.concatenate(parts);f=out/(role+'.npz');np.savez_compressed(f,index=ends+offset,available_index=ends+offset,per_channel_last=channel,score=channel.mean(1));banks.append(dict(role=role,path=str(f),sha256=sha(f),points=len(ends)))
    assert before==statehash(m)and step==m.step
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',seed=seed,arm=arm,selected_update=pt['update'],selected_step=step,PT_sha256=sha(p),banks=banks,roles=data.access,seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),only_native_readout=True,eval_state_immutable=True))
def l0(seed,arm,out):
    m,c=setup(seed);data=Data(['FIT']);v,_=data.load('FIT');x=window(v,np.arange(128));op=torch.optim.Adam(m.parameters(),lr=1e-4)
    for step in [0,3001]:m.step=step;m.train();op.zero_grad();loss=m(x,torch.arange(128,device='cuda')%38)[0];assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all()for p in m.parameters()if p.grad is not None);op.step()
    m.eval();before=statehash(m);step=m.step
    with torch.no_grad():z=m.native_readout(x)
    assert z.shape==(128,192,38)and torch.isfinite(z).all()and before==statehash(m)and step==m.step
    dump(out/'L0_RECEIPT.json',dict(status='PASS_DISCARDED',C38H192=True,B128=True,finite_native_aux_gradients=True,only_native_inference=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
if __name__=='__main__':
    phase=sys.argv[1];seed=int(sys.argv[2]);arm=sys.argv[3];torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction((20 if phase!='score'else 30)*1024**3/torch.cuda.get_device_properties(0).total_memory);out=O/str(seed)/arm/phase;out.mkdir(parents=True,exist_ok=False);t=time.monotonic()
    try:{'train':train,'score':score,'l0':l0}[phase](seed,arm,out)
    finally:dump(out/'OS_INTERNAL.json',dict(pid=os.getpid(),phase=phase,seed=seed,arm=arm,seconds=time.monotonic()-t,device=torch.cuda.get_device_name(0),GPU_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
