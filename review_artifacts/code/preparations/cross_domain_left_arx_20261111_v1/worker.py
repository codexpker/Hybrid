"""Fixed native LEFT SMD and ordinary Ridge; explicit isolated physical roles."""
import argparse,hashlib,importlib.util,json,os,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name;UP=R/'preparations/left_baseline_build_20261101_v1/upstream'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def dump(f,d):
    with Path(f).open('x')as out:json.dump(d,out,indent=2,allow_nan=False)
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rel=module('rel',P/'runtime/relationship_residual.py');native_score=module('score',P/'runtime/native_score_numpy.py')

class RoleData:
    def __init__(self,allowed):self.allowed=set(allowed);self.roles={e['role']:e for e in read(P/'INPUT_QUALIFICATION.json')['roles']};self.scaler=read(P/'FIT_ONLY_SCALER.json');self.access=[]
    def load(self,role):
        if role not in self.allowed:raise PermissionError('role forbidden before file access: '+role)
        e=self.roles[role];f=P/'data'/(role+'.npz');assert sha(f)==e['sha256']
        with np.load(f)as z:v=z['values'].astype(np.float64);mask=z['observed'].copy()
        assert list(v.shape)==e['shape']and v.shape[1]==38 and mask.all()and np.isfinite(v).all()
        self.access.append(dict(role=role,path=str(f),sha256=e['sha256'],original_start=e['original_start']))
        return(v-np.asarray(self.scaler['mean']))/np.asarray(self.scaler['scale']),e['original_start']

class WindowDataset:
    def __init__(self,v,start,step):self.v=v;self.start=start;self.step=step
    def __len__(self):return(len(self.v)-192)//self.step+1
    def footprint(self,i):return[self.start+i*self.step,self.start+i*self.step+192]
    def __getitem__(self,i):return np.float32(self.v[i*self.step:i*self.step+192]),np.zeros((192,1),np.float32)

def seed(torch):
    random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)

def setup_torch(cap):
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats();return torch

def native(torch):
    sys.path.insert(0,str(UP));from models.Left.LEFT import Left
    return Left(SimpleNamespace(**read(P/'configs/SMD/model_configs_Left.json')))

def left_l0(torch,out):
    seed(torch);m=native(torch).cuda();m.train();opt=torch.optim.Adam(m.parameters(),lr=1e-4);data=RoleData(['FIT']);v,_=data.load('FIT');x=torch.from_numpy(np.stack([np.float32(v[i:i+192])for i in range(128)])).cuda()
    for _ in range(2):
        opt.zero_grad();loss=m(x,None,None,None)[0];assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all()for p in m.parameters()if p.grad is not None);opt.step()
    m.eval();before={k:p.detach().clone()for k,p in m.state_dict().items()};step=m.step
    with torch.no_grad():raw=m.infer(x[:2],None,None,None)[0]
    assert raw.shape==(2,192,38)and torch.isfinite(raw).all()and m.step==step and all(torch.equal(before[k],v)for k,v in m.state_dict().items())
    dump(out/'RECEIPT.json',dict(status='COMPLETED',discarded_updates=2,full_B128_finite=True,eval_state_immutable=True,labels_read=False,roles=data.access))

def left_train(torch,out):
    seed(torch);sys.path.insert(0,str(R/'preparations/left_baseline_build_20261101_v1/deps_thop_transport2_v1/site'));sys.path.insert(0,str(UP));from exp import exp_anomaly_detection as official
    from torch.utils.data import DataLoader
    data=RoleData(['FIT','SELECT']);fit,fitstart=data.load('FIT');val,valstart=data.load('SELECT');events=[];updates=[0];save_events=[]
    try:data.load('CAL');raise AssertionError('CAL guard missing')
    except PermissionError:pass
    class ExactEarlyStopping(official.EarlyStopping):
        def save_checkpoint(self,val_loss,model,path):
            super().save_checkpoint(val_loss,model,path)
            save_events.append(dict(epoch=sum(e['kind']=='SELECT'for e in events),SELECT_loss=float(val_loss),python_step=model.step,checkpoint_sha256=sha(Path(path)/'checkpoint.pth')))
    official.EarlyStopping=ExactEarlyStopping
    class Isolated(official.Exp_Anomaly_Detection):
        def _acquire_device(self):return torch.device('cuda:0') # preserve UUID visibility, no official override to physical0
        def _get_data(self,flag,step=None):
            if flag not in ['train','val']:raise PermissionError('train entry cannot open CAL/QUERY')
            values,start=(fit,fitstart)if flag=='train'else(val,valstart);stride=192 if step is None else step;ds=WindowDataset(values,start,stride)
            events.append(dict(kind='loader',flag=flag,windows=len(ds),stride=stride,first=ds.footprint(0),last=ds.footprint(len(ds)-1)))
            return ds,DataLoader(ds,batch_size=self.train_configs.batch_size,shuffle=flag=='train',drop_last=False,num_workers=0)
        def _select_optimizer(self):
            opt=super()._select_optimizer()
            def post(optimizer,args,kwargs):
                updates[0]+=1
                if updates[0]%100==0 or updates[0]==1:
                    grads=[p.grad for p in self.model.parameters()if p.grad is not None];assert grads and all(torch.isfinite(g).all()for g in grads)
                    events.append(dict(kind='optimizer',update=updates[0],python_step=self.model.step,lr=[g['lr']for g in optimizer.param_groups],gradient_norm=float(torch.stack([g.detach().float().norm()for g in grads]).norm())))
            opt.register_step_post_hook(post);return opt
        def vali(self,loader):
            value=super().vali(loader);events.append(dict(kind='SELECT',loss=float(value),python_step=self.model.step));return value
        def test(self,**kwargs):raise PermissionError('training cannot query')
        def evaluate_spot(self,**kwargs):raise PermissionError('training cannot query')
    args=SimpleNamespace(data='SMD',data_origin='DADA',model_class='Left',gpu=0,use_gpu=True,gpu_type='cuda',use_multi_gpu=False,configs_path=str(P/'configs'),model_save_path=str(out/'model'),results_save_path=str(out/'unused_results'))
    exp=Isolated(args,id=0);initial={k:sha_bytes(v)for k,v in exp.model.state_dict().items()};dump(out/'INITIAL_STATE.json',dict(state_sha256=initial,CPU_rng=sha_bytes(torch.get_rng_state()),CUDA_rng=sha_bytes(torch.cuda.get_rng_state()),python_step=exp.model.step))
    try:exp.train() # exact original basic/optimizer/scheduler/earlystop; explicit roles only
    finally:dump(out/'TRAIN_EVENTS.json',dict(events=events,checkpoint_saves=save_events,updates=updates[0]))
    checkpoint=out/'model/SMD/Left/checkpoints_0/checkpoint.pth';assert save_events and sha(checkpoint)==save_events[-1]['checkpoint_sha256']
    dump(out/'RUNTIME_STATE.json',dict(final_python_step=exp.model.step,selected_checkpoint_python_step=save_events[-1]['python_step'],selected_epoch=save_events[-1]['epoch'],selected_loss=save_events[-1]['SELECT_loss'],checkpoint_sha256=sha(checkpoint),train_return_mode=exp.model.training,eval_restore='selected PT params/buffers plus actual final Pythonstep, native official posttrain semantics',state_sha256={k:sha_bytes(v)for k,v in exp.model.state_dict().items()}))
    dump(out/'RECEIPT.json',dict(status='COMPLETED',labels_read=False,checkpoint_sha256=sha(checkpoint),updates=updates[0],epochs=sum(e['kind']=='SELECT'for e in events),selected_epoch=save_events[-1]['epoch'],roles=data.access,CAL_QUERY_not_accessed=True))

def sha_bytes(v):return hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def left_score(torch,out):
    seed(torch);m=native(torch).cuda();train=O/'left_train';state=read(train/'RUNTIME_STATE.json');pt=train/'model/SMD/Left/checkpoints_0/checkpoint.pth';assert sha(pt)==state['checkpoint_sha256']
    loaded=m.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False);assert not loaded.missing_keys and all(k.endswith(('total_ops','total_params'))for k in loaded.unexpected_keys);m.step=state['final_python_step'];m.eval();before={k:sha_bytes(v)for k,v in m.state_dict().items()}
    data=RoleData(['CAL','QUERY']);banks=[]
    for role in ['CAL','QUERY']:
        v,start=data.load(role);index=np.arange(start+512,start+len(v));scores=[];channel=[]
        with torch.no_grad():
            for a in range(0,len(index),256):
                ids=index[a:a+256]-start;x=np.stack([np.float32(v[t-191:t+1])for t in ids]);assert x.shape[1:]==(192,38)
                raw=m.infer(torch.from_numpy(x).cuda(),None,None,None)[0].cpu().numpy();_,s=native_score.native_numpy_scores([raw],38);scores.append(s.reshape(len(x),192)[:,-1]);channel.append(raw[:,-1])
        score=np.concatenate(scores);f=out/(role+'.npz');np.savez_compressed(f,index=index,score=score,per_channel_last=np.concatenate(channel),available_index=index,valid=np.ones(len(index),bool));banks.append(dict(role=role,path=str(f),sha256=sha(f),points=len(index)))
    assert m.step==state['final_python_step']and before=={k:sha_bytes(v)for k,v in m.state_dict().items()}
    dump(out/'RECEIPT.json',dict(status='COMPLETED',labels_read=False,banks=banks,checkpoint_sha256=sha(pt),python_step_restored=m.step,eval_parameters_buffers_immutable=True,roles=data.access,native_smooth51_inside_W192_only=True))

def arx(torch,out):
    torch.backends.cuda.matmul.allow_tf32=False;data=RoleData(['FIT']);fit,start=data.load('FIT');view=rel.feature_view(fit);dim=38*512+1;gram=torch.zeros((dim,dim),dtype=torch.float64,device='cuda');rhs=torch.zeros((dim,38),dtype=torch.float64,device='cuda');block=1024
    for a in range(0,len(view),block):
        b=min(a+block,len(view));x=torch.from_numpy(rel.feature_block(view,a,b)).cuda();y=torch.from_numpy(np.ascontiguousarray(fit[a+512:b+512])).cuda();gram.addmm_(x.T,x);rhs.addmm_(x.T,y);del x,y
        print(json.dumps(dict(event='ARX_GRAM_ROWS',rows=b,total=len(view))),flush=True)
    gram.diagonal()[:-1].add_(1.);chol,info=torch.linalg.cholesky_ex(gram);assert int(info)==0;wt=torch.cholesky_solve(rhs,chol);error=float((gram@wt-rhs).abs().max()/torch.clamp(rhs.abs().max(),min=1.));assert error<1e-8;W=wt.cpu().numpy();np.savez_compressed(out/'NORMAL_FIT_RIDGE.npz',weight=W,mean=data.scaler['mean'],scale=data.scaler['scale'],ridge_alpha=np.array(1.));dump(out/'FIT_RECEIPT.json',dict(rows=len(view),features=dim,alpha=1.,intercept_penalized=False,normal_equation_relative_max_error=error,solver='float64 primal blocked Gram Cholesky; no jitter',roles=data.access,labels_read=False))
    del gram,chol,rhs;torch.cuda.empty_cache()
    def predict(v):
        windows=rel.feature_view(v);ans=np.empty((len(windows),38),np.float64)
        for a in range(0,len(windows),block):
            b=min(a+block,len(windows));x=torch.from_numpy(rel.feature_block(windows,a,b)).cuda();ans[a:b]=(x@wt).cpu().numpy()
        return ans
    scored=RoleData(['CAL','QUERY']);cal,calstart=scored.load('CAL');cp=predict(cal);res=cal[512:]-cp;idx=np.arange(calstart+512,calstart+len(cal));ends=np.linspace(20020,23704,512,dtype=np.int64);pos=ends-idx[0];scale=np.maximum(np.median(np.abs(res[pos]),axis=0),1e-6)
    banks=[]
    for role,v,pred,offset in [('CAL',cal,cp,calstart)]:
        point=rel.residual_score(v[512:]-pred,scale);g=rel.accumulate(point);f=out/(role+'.npz');ix=np.arange(offset+512,offset+len(v));np.savez_compressed(f,index=ix,prediction=pred,observed=v[512:],residual=v[512:]-pred,score=point,g=g,available_index=ix,valid=np.ones(len(ix),bool));banks.append(dict(role=role,path=str(f),sha256=sha(f),points=len(ix)))
    q,qstart=scored.load('QUERY');qp=predict(q);point=rel.residual_score(q[512:]-qp,scale);g=rel.accumulate(point);qi=np.arange(qstart+512,qstart+len(q));f=out/'QUERY.npz';np.savez_compressed(f,index=qi,prediction=qp,observed=q[512:],residual=q[512:]-qp,score=point,g=g,available_index=qi,valid=np.ones(len(qi),bool));banks.append(dict(role='QUERY',path=str(f),sha256=sha(f),points=len(qi)))
    samples=[0,1,512,len(qp)-1];qview=rel.feature_view(q);cpu=np.stack([rel.feature_block(qview,i,i+1)[0]@W for i in samples]);replay=float(np.abs(cpu-qp[samples]).max());assert np.allclose(cpu,qp[samples],atol=1e-9,rtol=1e-9)
    dump(out/'CALIBRATION.json',dict(residual_scale=scale.tolist(),selected_index=ends.tolist(),normal_CAL_selected_RMSE=float(np.sqrt(np.mean(res[pos]**2))),normal_CAL_isolated_from_FIT_SELECT=True,background_health='nominal/unknown not certified',CPU_GPU_prediction_replay_max_abs_error=replay))
    dump(out/'RECEIPT.json',dict(status='COMPLETED',banks=banks,labels_read=False,roles=scored.access,no_SELECT_model_or_alpha_selection=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['left_l0','left_train','left_score','arx']);phase=ap.parse_args().phase;start=time.monotonic();out=O/phase;out.mkdir(parents=True,exist_ok=False)
    torch=setup_torch(20 if phase in ['left_l0','left_train','arx']else 30)
    {'left_l0':left_l0,'left_train':left_train,'left_score':left_score,'arx':arx}[phase](torch,out)
    dump(out/'GPU_PEAK.json',dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-start));print(json.dumps(dict(event='PHASE_COMPLETED',phase=phase,seconds=time.monotonic()-start)),flush=True)
