"""One isolated native/Hybrid job: fixed normal training then frozen scoring."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
from types import SimpleNamespace
import argparse,json,hashlib,time,random,sys,copy
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name;UP=R/'preparations/left_baseline_build_20261101_v1/upstream';sys.path[:0]=[str(UP),str(P/'runtime')]
from models.Left.LEFT import Left
from left_predictive_cycle import predictive_cycle_class
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def dump(f,x):Path(f).write_text(json.dumps(x,indent=2,allow_nan=False))
def seed():random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
def statehash(m):return {k:thash(v)for k,v in m.state_dict().items()}
class Roles:
    def __init__(self,domain,allowed):
        self.domain=domain;self.allowed=set(allowed);self.gate=json.loads((P/'DATA_GATE.json').read_text());self.scaler=json.loads((P/(domain+'_FIT_ONLY_SCALER.json')).read_text());self.access=[];self.raw={}
    def load(self,role):
        if role not in self.allowed:raise PermissionError('forbidden role BEFORE opening: '+role)
        e=next(x for x in self.gate['roles']if x['domain']==self.domain and x['role']==role);f=Path(e['path']);assert sha(f)==e['sha256']
        with np.load(f)as z:
            assert z['observed'].all();raw=z['values'];v=np.float32((raw-np.asarray(self.scaler['mean']))/np.asarray(self.scaler['scale']));run=z['run'].copy();fault=z['fault'].copy();offset=z['original_start'].copy()
        assert np.isfinite(v).all();self.raw[role]=raw;self.access.append(dict(role=role,sha256=e['sha256'],shape=list(v.shape)));return v,run,fault,offset
class Windows(Dataset):
    def __init__(self,values,H,items=None):self.v=values;self.H=H;self.per=values.shape[1]-H;self.items=items
    def __len__(self):return len(self.items)if self.items is not None else len(self.v)*self.per
    def __getitem__(self,i):
        run,end=self.items[i]if self.items is not None else(i//self.per,self.H+i%self.per)
        return self.v[run,end-self.H:end+1]
def model(domain,mode):
    cfg=json.loads((P/'configs'/(domain+'.json')).read_text());m=predictive_cycle_class(Left,mode)(SimpleNamespace(**cfg)).cuda();assert not any(isinstance(x,torch.nn.modules.batchnorm._BatchNorm)for x in m.modules());return m,cfg
def loader(dataset,sampler=None,batch=128,epoch=0):
    return DataLoader(dataset,batch_size=batch,shuffle=False,sampler=sampler,num_workers=0,generator=torch.Generator().manual_seed(2025+epoch))
def selection(m,vl):
    modes={n:s.training for n,s in m.named_modules()};step=m.step;before=statehash(m);rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state();m.eval();sums={};count=0
    try:
        with torch.no_grad():
            for b in vl:
                m.selection_loss(b.cuda());n=len(b);count+=n
                for k,v in m.last_loss_parts.items():sums[k]=sums.get(k,0)+v*n
        assert step==m.step and before==statehash(m)
        return {k:v/count for k,v in sums.items()}
    finally:
        for n,sub in m.named_modules():sub.training=modes[n]
        torch.set_rng_state(rng);torch.cuda.set_rng_state(cuda)
def savept(path,m,opt,epoch,selection_loss):
    tmp=path.with_suffix('.tmp');torch.save(dict(state=m.state_dict(),optimizer=opt.state_dict(),epoch=epoch,python_step=m.step,selection_loss=selection_loss),tmp);os.replace(tmp,path)
def train(domain,mode,out):
    seed();data=Roles(domain,['FIT','SELECT']);v,run,_,offset=data.load('FIT');sv,srun,_,_=data.load('SELECT');m,cfg=model(domain,mode);m.train();H=cfg['seq_len'];ds=Windows(v,H)
    try:data.load('CAL');raise AssertionError('role guard missing')
    except PermissionError:pass
    selitems=[(r,t)for r in range(len(sv))for t in range(H,sv.shape[1],H)];vl=loader(Windows(sv,H,selitems));planfile=P/'data'/(domain+'_DRAWPLAN.npz');plan=np.load(planfile)['order'];assert plan.shape==(20,len(ds))
    cfgfixed=json.loads((P/'FIXED_EXPERIMENT.json').read_text());assert len(ds)==(24000 if domain=='TEP'else 79296)
    initial=statehash(m);dump(out/'INITIAL.json',dict(state=initial,CPU_RNG=thash(torch.get_rng_state()),CUDA_RNG=thash(torch.cuda.get_rng_state()),python_step=m.step,model_sha256=sha(P/'runtime/left_predictive_cycle.py'),worker_sha256=sha(__file__),config_sha256=sha(P/'configs'/(domain+'.json')),plan_sha256=sha(planfile),scaler_sha256=sha(P/(domain+'_FIT_ONLY_SCALER.json')),source_roles=data.access,optimizer_updates_limit=(3760 if domain=='TEP'else 12400)))
    opt=torch.optim.Adam(m.parameters(),lr=1e-4);trace=[];best=float('inf');selected=None;updates=0;start=time.monotonic();schedule={2:5e-5,4:1e-5,6:5e-6,8:1e-6,10:5e-7,15:1e-7,20:5e-8};eligible_after=cfg['warmup_steps']+cfg['ramp_steps']
    for epoch in range(1,21):
        m.train();sums={};count=0;epstart=time.monotonic();grad_last=None
        for batch in loader(ds,plan[epoch-1].tolist(),epoch=epoch):
            opt.zero_grad();loss=m(batch.cuda(),None,None,None)[0];assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all()for p in m.parameters()if p.grad is not None);opt.step();updates+=1;n=len(batch);count+=n
            for key,value in m.last_loss_parts.items():sums[key]=sums.get(key,0)+value*n
            if updates%100==0 or updates==1:grad_last=float(torch.stack([p.grad.detach().norm()for p in m.parameters()if p.grad is not None]).norm())
        sel=selection(m,vl);eligible=m.step>=eligible_after;row=dict(epoch=epoch,updates=updates,python_step=m.step,train={k:v/count for k,v in sums.items()},SELECT=sel,eligible=eligible,learning_rate=opt.param_groups[0]['lr'],last_logged_grad_norm=grad_last,seconds=time.monotonic()-epstart)
        if eligible and sel['joint']<best:
            best=sel['joint'];selected=dict(epoch=epoch,python_step=m.step,SELECT=sel);savept(out/'selected.pth',m,opt,epoch,best);row['checkpoint_sha256']=sha(out/'selected.pth')
        trace.append(row);dump(out/'TRAIN_TRACE.json',trace);print(json.dumps(dict(event='EPOCH_COMPLETED',domain=domain,mode=mode,**row)),flush=True)
        if epoch in schedule:
            for group in opt.param_groups:group['lr']=schedule[epoch]
    assert updates==(3760 if domain=='TEP'else 12400)and selected is not None;savept(out/'final.pth',m,opt,20,trace[-1]['SELECT']['joint']);dump(out/'TRAIN_RECEIPT.json',dict(status='COMPLETED',updates=updates,selected=selected,seconds=time.monotonic()-start,source_roles=data.access,plan_sha256=sha(planfile),initial_parameters=initial,all_parameters_trainable=all(p.requires_grad for p in m.parameters()),all_modules_training=all(x.training for x in m.modules()),scaler_fit_only=True,CAL_QUERY_unopened=True,selected_sha256=sha(out/'selected.pth'),final_sha256=sha(out/'final.pth')))
def l0(domain,out):
    seed();data=Roles(domain,['FIT']);v,_,_,_=data.load('FIT');m,cfg=model(domain,'full');H=cfg['seq_len'];x=torch.from_numpy(np.stack([v[0,j:j+H+1]for j in range(128)])).cuda();opt=torch.optim.Adam(m.parameters(),lr=1e-4);losses=[]
    for step in [0,cfg['warmup_steps']+cfg['ramp_steps']+1]:
        m.step=step;m.train();opt.zero_grad();loss=m(x,None,None,None)[0];assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all()for p in m.parameters()if p.grad is not None);opt.step();losses.append(float(loss.detach()))
    m.eval();before=statehash(m);step=m.step
    with torch.no_grad():raw=m.infer(x[:2],None,None,None)[0];off=m.native_readout(x[:2])[0]
    assert raw.shape==(2,H,cfg['enc_in'])and off.shape==raw.shape and torch.isfinite(raw).all()and torch.isfinite(off).all()and statehash(m)==before and m.step==step
    dump(out/'L0_RECEIPT.json',dict(status='PASS_DISCARDED',domain=domain,B128=True,losses=losses,step_and_eval_state_immutable=True,shared_native_train=True,roles=data.access,peak_allocated_bytes=torch.cuda.max_memory_allocated(),selected_PT_saved=False))
def score(domain,mode,out):
    seed();m,cfg=model(domain,mode);pt=out.parent/'train/selected.pth';state=torch.load(pt,map_location='cpu',weights_only=True);m.load_state_dict(state['state'],strict=True);m.step=state['python_step'];m.eval();before=statehash(m);step=m.step;data=Roles(domain,['CAL','QUERY','NORMAL_TEST']);H=cfg['seq_len'];scaler=data.scaler;outputs=[]
    for role in ['CAL','QUERY']+(['NORMAL_TEST']if domain=='TEP'else[]):
        v,runs,faults,offsets=data.load(role);series=[]
        for k in range(len(v)):
            if domain=='PSM'and role=='QUERY':ends=np.arange(512,len(v[k]))
            else:ends=np.arange(H,len(v[k]))
            index=ends+int(offsets[k])+(1 if domain=='TEP'else 0);raws=[];offraw=[];preds=[];freqs=[];mspreds=[];nativepreds=[]
            with torch.no_grad():
                for a in range(0,len(ends),128):
                    es=ends[a:a+128];x=torch.from_numpy(np.stack([v[k,t-H:t+1]for t in es])).cuda();past,current=m._split(x)
                    if mode=='native':
                        result=Left._forward(m,current,None,None,None);raw=m._ms_anomaly_score(result[1],result[0],result[2],result[3],result[4],current,result[9],result[10])
                    else:
                        result=m.prediction_forward(past);raw=m._prediction_score(result,current)
                    raws.append(raw[:,-1].cpu().numpy());preds.append(result[4][:,-1].cpu().numpy());freqs.append(result[3][:,-1].cpu().numpy());mspreds.append(result[9][:,-1].cpu().numpy())
                    if mode!='native':
                        # Fixed score-off from the same candidate PT, no extra model.
                        nr=Left._forward(m,current,None,None,None);off=m._ms_anomaly_score(nr[1],nr[0],nr[2],nr[3],nr[4],current,nr[9],nr[10]);offraw.append(off[:,-1].cpu().numpy());nativepreds.append(nr[4][:,-1].cpu().numpy())
            ch=np.concatenate(raws);pred=np.concatenate(preds);mean=ch.mean(-1);scoreprimary=ch[:,0]if domain=='TEP'else mean;d=dict(index=index,run=np.full(len(index),runs[k]),fault=np.full(len(index),faults[k]),target_y=data.raw[role][k,ends,0],target_all_original=data.raw[role][k,ends],target_standardized=v[k,ends],prediction=pred*np.asarray(scaler['scale'])+np.asarray(scaler['mean']),time_prediction_standardized=pred,freq_prediction_standardized=np.concatenate(freqs),MS_prediction_standardized=np.concatenate(mspreds),raw_score=scoreprimary,channel_score=ch,all_channel_mean=mean)
            if mode!='native':
                off=np.concatenate(offraw);d.update(nativeoff_score=off[:,0]if domain=='TEP'else off.mean(-1),nativeoff_channel_score=off,nativeoff_prediction=np.concatenate(nativepreds)*np.asarray(scaler['scale'])+np.asarray(scaler['mean']))
            series.append(d)
        keys=series[0].keys();dest=out/(role+'.npz');np.savez_compressed(dest,**{key:np.stack([s[key]for s in series])for key in keys});outputs.append(dict(role=role,path=str(dest),sha256=sha(dest),series=len(series),points_per_series=len(series[0]['index']),primary='target9'if domain=='TEP'else'mean25',CAL_QUERY_scores_not_fitted_or_thresholded=True))
    assert statehash(m)==before and m.step==step;dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',selected_PT_sha256=sha(pt),selected_python_step=step,eval_states_modes_immutable=True,roles=data.access,outputs=outputs,score='native formula, current target only in error/GT; strictpast prediction output never current latent',no_labels_or_QUERY_statistics=True))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['l0','train','score']);ap.add_argument('job');args=ap.parse_args();domain=args.job.split('_')[0];mode=args.job[len(domain)+1:];assert mode in['native','full','history_only'];out=O/args.job/args.phase;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.cuda.set_device(0);cap=30 if args.phase=='score'else 20;torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(0).total_memory);torch.cuda.reset_peak_memory_stats();start=time.time()
    try:
        {'train':lambda:train(domain,mode,out),'l0':lambda:l0(domain,out),'score':lambda:score(domain,mode,out)}[args.phase]();dump(out/'OS.json',dict(status='COMPLETED',pid=os.getpid(),start_unix=start,end_unix=time.time(),seconds=time.time()-start,GPU_hours=(time.time()-start)/3600,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),exit_code=0));print(json.dumps(dict(event='PHASE_COMPLETED',phase=args.phase,job=args.job,seconds=time.time()-start)),flush=True)
    except BaseException:
        import traceback;dump(out/'FAILED.json',dict(status='FAILED',pid=os.getpid(),start_unix=start,end_unix=time.time(),seconds=time.time()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),traceback=traceback.format_exc()));raise
