"""Only the frozen two-domain channel-blindspot experiment. No metric labels."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,hashlib,random,copy
import numpy as np
import torch
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name
sys.path[:0]=[str(P/'runtime'),str(R/'preparations/left_baseline_build_20261101_v1/upstream')]
from models.Left.LEFT import Left
from hybrid_tsad.left_channel_blindspot import channel_blindspot_class
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def read(p):return json.loads(Path(p).read_text())
def seed():random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
def statehash(m):return{k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in m.state_dict().items()}
class Data:
    def __init__(self,domain,allowed):self.domain=domain;self.allowed=set(allowed);self.access=[];self.cfg=read(P/'FIXED_EXPERIMENT.json')['domains'][domain];self.sc=read(R/self.cfg['scaler'])
    def load(self,role):
        if role not in self.allowed:raise PermissionError(role)
        e=self.cfg['roles'][role];p=R/e['path'];assert sha(p)==e['sha256']
        with np.load(p)as z:
            raw=z['values'];assert z['observed'].all()
            if raw.ndim==3:assert len(raw)==1;raw=raw[0]
            assert raw.ndim==2 and np.isfinite(raw).all();raw=raw.copy()
        self.access.append(dict(role=role,sha256=e['sha256'],shape=list(raw.shape),start=e['start']))
        return np.float32((raw-np.asarray(self.sc['mean']))/np.asarray(self.sc['scale'])),int(e['start'])
def model(domain,mode):
    d=read(P/'FIXED_EXPERIMENT.json')['domains'][domain];c=read(R/d['config']);m=channel_blindspot_class(Left,mode)(SimpleNamespace(**c)).cuda()
    assert not any(isinstance(s,torch.nn.modules.batchnorm._BatchNorm)for s in m.modules())
    source=R/d['source_PT'];assert sha(source)==d['source_sha256'];pt=torch.load(source,map_location='cpu',weights_only=True);state=pt['state']if 'state'in pt else pt
    missing=m.load_state_dict(state,strict=False);assert not missing.missing_keys and all(k.endswith(('total_ops','total_params'))for k in missing.unexpected_keys)
    m.step=int(d['source_python_step']);return m,c
def windows(v,starts):return torch.from_numpy(np.stack([v[s:s+192]for s in starts])).cuda()
def select(m,v,channels):
    states={n:s.training for n,s in m.named_modules()};before=statehash(m);step=m.step;tr=torch.get_rng_state();cr=torch.cuda.get_rng_state();m.eval();sums={};n=0
    starts=np.arange(0,len(v)-191,192)
    try:
        with torch.no_grad():
            for a in range(0,len(starts),128):
                b=windows(v,starts[a:a+128]);j=torch.from_numpy(channels[a:a+len(b)]).cuda();m.selection_loss(b,j)
                for k,z in m.last_blind_parts.items():sums[k]=sums.get(k,0.)+z*len(b)
                n+=len(b)
        assert before==statehash(m)and step==m.step
        return{k:z/n for k,z in sums.items()}
    finally:
        for name,s in m.named_modules():s.training=states[name]
        torch.set_rng_state(tr);torch.cuda.set_rng_state(cr)
def savept(p,m,opt,update,selection):
    q=p.with_suffix('.tmp');torch.save(dict(state=m.state_dict(),optimizer=opt.state_dict(),update=update,python_step=m.step,selection=selection),q);os.replace(q,p)
def train(domain,arm,out):
    seed();data=Data(domain,['FIT','SELECT']);fit,_=data.load('FIT');val,_=data.load('SELECT');m,c=model(domain,'blind'if arm=='blind'else'native');m.train();initial=statehash(m)
    with np.load(P/(domain+'_PLAN.npz'))as p:starts=p['starts'];channels=p['channels'];schannels=p['select_channels']
    assert starts.shape==channels.shape==(2000,128)and starts.min()>=0 and starts.max()<=len(fit)-192
    opt=torch.optim.Adam(m.parameters(),lr=1e-4);trace=[];best=float('inf');t=time.monotonic();selected=None
    s0=select(m,val,schannels);dump(out/'SOURCE_SELECT.json',s0)
    for u in range(1,2001):
        m.train();x=windows(fit,starts[u-1]);j=torch.from_numpy(channels[u-1]).cuda();opt.zero_grad();loss=m(x,j)[0];assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all()for p in m.parameters()if p.grad is not None);opt.step()
        if u==1 or u%100==0:print(json.dumps(dict(event='UPDATE',domain=domain,arm=arm,update=u,step=m.step,loss=float(loss.detach()))),flush=True)
        if u%200==0:
            training_last=dict(m.last_blind_parts)
            s=select(m,val,schannels);row=dict(update=u,python_step=m.step,SELECT=s,training_last=training_last,seconds=time.monotonic()-t)
            if s['joint']<best:best=s['joint'];selected=dict(update=u,python_step=m.step,SELECT=s);savept(out/'selected.pth',m,opt,u,s);row['selected']=True
            trace.append(row);dump(out/'TRAIN_TRACE.json',trace);print(json.dumps(dict(event='SELECT',domain=domain,arm=arm,**row)),flush=True)
    savept(out/'endpoint2000.pth',m,opt,2000,trace[-1]['SELECT'])
    dump(out/'TRAIN_RECEIPT.json',dict(status='COMPLETED',updates=2000,selected=selected,source=read(P/'FIXED_EXPERIMENT.json')['domains'][domain],plan_sha256=sha(P/(domain+'_PLAN.npz')),initial_state=initial,source_native_untouched=True,roles=data.access,seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),selected_sha256=sha(out/'selected.pth'),endpoint_sha256=sha(out/'endpoint2000.pth')))
def l0(domain,out):
    seed();data=Data(domain,['FIT']);fit,_=data.load('FIT');m,c=model(domain,'blind');x=windows(fit,np.arange(128));j=torch.arange(128,device='cuda')%c['enc_in'];m.train();op=torch.optim.Adam(m.parameters(),lr=1e-4);t=time.monotonic()
    for _ in range(2):op.zero_grad();loss=m(x,j)[0];assert torch.isfinite(loss);loss.backward();op.step()
    m.eval();before=statehash(m);step=m.step;starts=np.arange(16);v=windows(fit,starts);rep=v.repeat_interleave(c['enc_in'],dim=0);ch=torch.arange(c['enc_in'],device='cuda').repeat(len(v));st=time.monotonic()
    with torch.no_grad():score=m.score_one_channel(rep,ch)
    torch.cuda.synchronize();wall=time.monotonic()-st
    assert score.shape==(16*c['enc_in'],)and torch.isfinite(score).all()and before==statehash(m)and step==m.step
    dump(out/'L0_RECEIPT.json',dict(status='PASS_DISCARDED',domain=domain,B128_train=True,infer_windows=16,infer_masks=len(rep),infer_seconds=wall,windows_per_second=16/wall,peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-t,source_PT_unchanged=True))
def score(domain,arm,out):
    seed();m,c=model(domain,'blind');base_step=m.step
    if arm!='source':
        p=O/(domain+'_'+arm)/'train/selected.pth';pt=torch.load(p,map_location='cpu',weights_only=True);m.load_state_dict(pt['state'],strict=True);m.step=int(pt['python_step'])
    else:p=R/read(P/'FIXED_EXPERIMENT.json')['domains'][domain]['source_PT']
    m.eval();before=statehash(m);step=m.step;data=Data(domain,['CAL','QUERY']);banks=[];t=time.monotonic();C=c['enc_in'];do_mask=arm in['blind','source']
    for role in ['CAL','QUERY']:
        v,offset=data.load(role);ends=np.arange(512,len(v));chan=[];native=[]
        for a in range(0,len(ends),16):
            es=ends[a:a+16];x=windows(v,es-191)
            with torch.no_grad():
                ns=m.native_readout(x)[:,-1,:].cpu().numpy();native.append(ns)
                if do_mask:
                    xx=x.repeat_interleave(C,0);jj=torch.arange(C,device='cuda').repeat(len(x));s=m.score_one_channel(xx,jj).reshape(len(x),C).cpu().numpy();chan.append(s)
            if a==0 or a%16000==0:print(json.dumps(dict(event='SCORE',domain=domain,arm=arm,role=role,points=min(a+16,len(ends)),total=len(ends))),flush=True)
        nc=np.concatenate(native);cc=np.concatenate(chan)if do_mask else nc
        f=out/(role+'.npz');np.savez_compressed(f,index=ends+offset,available_index=ends+offset,score=cc.mean(1),per_channel_last=cc,native_score=nc.mean(1),native_channel_last=nc)
        banks.append(dict(role=role,path=str(f),sha256=sha(f),points=len(ends),index_start=int(ends[0]+offset),index_end=int(ends[-1]+offset)))
    assert statehash(m)==before and m.step==step
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',domain=domain,arm=arm,PT=str(p),PT_sha256=sha(p),python_step=step,source_python_step=base_step,banks=banks,roles=data.access,seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),all_channel_blind_readout=do_mask,eval_state_immutable=True))
if __name__=='__main__':
    torch.set_num_threads(2);torch.cuda.set_device(0);phase,job=sys.argv[1:3];domain,arm=job.split('_');cap=20 if phase in['train','l0']else 30
    torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(0).total_memory);out=O/job/phase;out.mkdir(parents=True,exist_ok=False);st=time.monotonic()
    try:{'train':train,'score':score,'l0':lambda d,a,o:l0(d,o)}[phase](domain,arm,out)
    finally:dump(out/'OS_INTERNAL.json',dict(pid=os.getpid(),phase=phase,job=job,seconds=time.monotonic()-st,device=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),GPU_visible=os.environ.get('CUDA_VISIBLE_DEVICES')))
