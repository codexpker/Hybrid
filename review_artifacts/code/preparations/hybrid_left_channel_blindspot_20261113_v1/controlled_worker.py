"""Same channel-blindspot/native pair on reader-frozen independent run arrays."""
from pathlib import Path
import os,sys,json,time,hashlib,random
import numpy as np
import torch
from types import SimpleNamespace
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name/'controlled'
sys.path[:0]=[str(P/'runtime'),str(R/'preparations/left_baseline_build_20261101_v1/upstream')]
from models.Left.LEFT import Left
from hybrid_tsad.left_channel_blindspot import channel_blindspot_class
from worker import statehash,savept
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(role):
    e=read(P/'controlled/ARRAYS.json')[role];f=P/e['path'];assert sha(f)==e['sha256']
    with np.load(f)as z:v=z['values'].copy()
    assert v.ndim==3 and v.shape[2]==3 and np.isfinite(v).all();return v,e
def batch(v,starts,runs,H):return torch.from_numpy(np.float32(np.stack([v[r,s:s+H]for s,r in zip(starts,runs)]))).cuda()
def selection(m,v,plan,H):
    modes={n:s.training for n,s in m.named_modules()};before=statehash(m);step=m.step;tr=torch.get_rng_state();cr=torch.cuda.get_rng_state();m.eval();losses=[]
    pairs=[(r,s)for r in range(len(v))for s in range(0,v.shape[1]-H+1,H)]
    try:
        with torch.no_grad():
            for a in range(0,len(pairs),128):
                pp=pairs[a:a+128];x=batch(v,[q[1]for q in pp],[q[0]for q in pp],H);j=torch.from_numpy(plan[a:a+len(pp)]).cuda();value=m.selection_loss(x,j);losses.append((len(pp),float(value)))
        assert before==statehash(m)and step==m.step;return sum(n*v for n,v in losses)/sum(n for n,v in losses)
    finally:
        for n,s in m.named_modules():s.training=modes[n]
        torch.set_rng_state(tr);torch.cuda.set_rng_state(cr)
def run(arm):
    random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025);torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory)
    c=read(P/'controlled/MODEL.json');H=c['seq_len'];m=channel_blindspot_class(Left,'blind'if arm=='blind'else'native')(SimpleNamespace(**c)).cuda();initial=statehash(m)
    fit,_=load('FIT');val,_=load('SELECT');mean=fit.mean((0,1));scale=fit.std((0,1));scale=np.where(scale>0,scale,1.)
    fit=np.float32((fit-mean)/scale);val=np.float32((val-mean)/scale)
    rng=np.random.default_rng(2025);draw=rng.integers(0,len(fit)*(fit.shape[1]-H+1),size=(2000,128));runs=draw//(fit.shape[1]-H+1);starts=draw%(fit.shape[1]-H+1)
    channels=np.random.default_rng(84551).integers(0,3,size=draw.shape,dtype=np.int64);schannels=np.random.default_rng(123045).integers(0,3,size=len(val)*len(range(0,val.shape[1]-H+1,H)),dtype=np.int64)
    out=O/arm;out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'PLAN_SCALER.npz',draw=draw,channels=channels,SELECT_channels=schannels,mean=mean,scale=scale)
    opt=torch.optim.Adam(m.parameters(),lr=1e-4);best=float('inf');trace=[];t=time.monotonic()
    for u in range(1,2001):
        m.train();x=batch(fit,starts[u-1],runs[u-1],H);j=torch.from_numpy(channels[u-1]).cuda();opt.zero_grad();loss=m(x,j)[0];assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(v.grad).all()for v in m.parameters()if v.grad is not None);opt.step()
        if u%200==0:
            s=selection(m,val,schannels,H);row=dict(update=u,step=m.step,SELECT=s,seconds=time.monotonic()-t);trace.append(row)
            if m.step>=c['warmup_steps']+c['ramp_steps']and s<best:best=s;savept(out/'selected.pth',m,opt,u,s);row['selected']=True
            dump(out/'TRAIN_TRACE.json',trace);print(json.dumps(dict(event='CONTROLLED_UPDATE',arm=arm,**row)),flush=True)
    state=torch.load(out/'selected.pth',map_location='cpu',weights_only=True);m.load_state_dict(state['state'],strict=True);m.step=state['python_step'];m.eval();before=statehash(m);banks=[]
    for role in read(P/'controlled/ARRAYS.json'):
        if role in['FIT','SELECT']:continue
        raw,e=load(role);v=np.float32((raw-mean)/scale);allscore=[];native=[]
        for runid in range(len(v)):
            ends=np.arange(H-1,len(v[runid]));ss=[];ns=[]
            for a in range(0,len(ends),64):
                es=ends[a:a+64];x=batch(v,es-H+1,[runid]*len(es),H)
                with torch.no_grad():
                    ns.append(m.native_readout(x)[:,-1,:].cpu().numpy());rep=x.repeat_interleave(3,0);j=torch.arange(3,device='cuda').repeat(len(x));ss.append(m.score_one_channel(rep,j).reshape(len(x),3).cpu().numpy())
            allscore.append(np.concatenate(ss));native.append(np.concatenate(ns))
        cc=np.stack(allscore);nc=np.stack(native);f=out/(role+'.npz');np.savez_compressed(f,index=np.arange(H-1,raw.shape[1]),available_index=np.arange(H-1,raw.shape[1]),per_channel_last=cc,score=cc.mean(2),native_channel_last=nc,native_score=nc.mean(2))
        banks.append(dict(role=role,path=str(f),sha256=sha(f),input_sha256=e['sha256'],shape=list(cc.shape)))
    assert before==statehash(m)
    dump(out/'RECEIPT.json',dict(status='COMPLETED',arm=arm,new_random_normal_training=True,updates=2000,selected_update=state['update'],selected_step=m.step,selected_sha256=sha(out/'selected.pth'),initial_state=initial,banks=banks,FIT_only_scaler=True,seconds=time.monotonic()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated(),labels_used_for_fit=False))
if __name__=='__main__':run(sys.argv[1])
