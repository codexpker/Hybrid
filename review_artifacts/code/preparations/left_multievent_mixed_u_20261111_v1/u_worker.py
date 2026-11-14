"""Ordinary full-native U, paired real-attack windows; no robust loss added."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name
def mod(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=mod('native_base',P/'native_baseline.py');u=mod('u_common',P/'runtime/left_native_u.py');constructor=mod('real_attack',P/'runtime/left_real_attack_u.py');observer=mod('obs',P/'runtime/left_loss_observer.py')
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(f,d):
    with Path(f).open('x')as out:json.dump(d,out,indent=2,allow_nan=False)
def hashes(m):return{k:base.sha_bytes(v)for k,v in m.state_dict().items()}
def rngs(torch):return dict(CPU=base.sha_bytes(torch.get_rng_state()),CUDA=[base.sha_bytes(x)for x in torch.cuda.get_rng_state_all()])
def initial(torch):
    base.seed(torch);m=base.native(torch).cuda();meta=read(O/'left_train/RUNTIME_STATE.json');pt=O/'left_train/model/SWAT/Left/checkpoints_0/checkpoint.pth';assert sha(pt)==meta['checkpoint_sha256'];load=m.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False);assert not load.missing_keys and all(k.endswith(('total_ops','total_params'))for k in load.unexpected_keys);m.step=meta['final_python_step'];return m,pt
def select(torch,m,values,update):
    before=hashes(m);oldstep=m.step;oldrng=rngs(torch);modes={n:x.training for n,x in m.named_modules()};starts=np.arange(0,len(values)-191,192);x=np.stack([np.float32(values[s:s+192])for s in starts])
    with u.diagnostic_eval(m,torch):value=float(m(torch.from_numpy(x).cuda(),None,None,None)[0])
    assert before==hashes(m)and oldstep==m.step and oldrng==rngs(torch)and modes=={n:x.training for n,x in m.named_modules()}
    return dict(update=update,SELECT_loss=value,windows=len(x),state_modes_step_RNG_restored=True,no_endpoint_selection=True)
def train(torch,c,out):
    m,pt=initial(torch);m.train();assert all(p.requires_grad for p in m.parameters());startstep=m.step;data=base.RoleData(['U_FIT','DONOR','SELECT']);fit,_=data.load('U_FIT');donor,_=data.load('DONOR');val,_=data.load('SELECT');gate=read(P/'LEARNER_RUNTIME.json')
    assert sha(P/'LEARNER_PLAN.npz')==gate['learner_plan_sha256']
    with np.load(P/'LEARNER_PLAN.npz')as z:plan={k:z[k].copy()for k in z.files}
    initial_meta=dict(checkpoint_sha256=sha(pt),state_sha256=hashes(m),rng=rngs(torch),source_step=startstep,plan_sha256=sha(P/'LEARNER_PLAN.npz'));dump(out/'INITIAL_IDENTITY.json',initial_meta);np.savez_compressed(out/'INITIAL_RNG.npz',CPU=torch.get_rng_state().cpu().numpy(),CUDA=torch.cuda.get_rng_state().cpu().numpy());before={k:v.detach().cpu().clone()for k,v in m.state_dict().items()};opt=torch.optim.Adam(m.parameters(),lr=1e-4);selected=[select(torch,m,val,0)];observed=observer.attach_native_loss_observer(m,torch);losses=[];traces=[]
    ema={'t_calls':0,'f_calls':0,'t_vectors':0,'f_vectors':0}
    for prefix,bank in [('t',m.proto_t),('f',m.proto_f)]:
        original=bank.ema_update_hard
        def count_ema(z,nn_idx,momentum=.99,original=original,prefix=prefix):
            ema[prefix+'_calls']+=1;ema[prefix+'_vectors']+=int(z.shape[0]);return original(z,nn_idx,momentum=momentum)
        bank.ema_update_hard=count_ema
    for step in range(2000):
        x=constructor.native_batch(fit,donor,plan,step,c);opt.zero_grad();loss=m(torch.from_numpy(x).cuda(),None,None,None)[0];assert torch.isfinite(loss);loss.backward();grads=[p.grad for p in m.parameters()if p.grad is not None];assert grads and all(torch.isfinite(g).all()for g in grads);norm=float(torch.stack([g.detach().float().norm()for g in grads]).norm());opt.step();assert m.step==startstep+step+1
        each=observed['last'];losses.append({k:each[k].copy()for k in ['ms','cycle_ft','cycle_tf','cycle','total','MS_segment']});traces.append(dict(update=step+1,loss=float(loss.detach()),lr=.0001,gradient_norm=norm,rng_after=rngs(torch),replacement_slots=int(plan['replacement_mask'][step].sum())if c!='A'else 0,python_step=m.step))
        if(step+1)%500==0:selected.append(select(torch,m,val,step+1));print(json.dumps(dict(event='U_UPDATE',condition=c,update=step+1,SELECT_loss=selected[-1]['SELECT_loss'],loss=traces[-1]['loss'])),flush=True)
    torch.save(m.state_dict(),out/'endpoint2000.pth');torch.save(opt.state_dict(),out/'optimizer2000.pth');np.savez_compressed(out/'FINAL_RNG.npz',CPU=torch.get_rng_state().cpu().numpy(),CUDA=torch.cuda.get_rng_state().cpu().numpy());np.savez_compressed(out/'PER_SAMPLE_NATIVE_LOSS.npz',**{k:np.stack([x[k]for x in losses])for k in losses[0]})
    delta={}
    for name,v in m.state_dict().items():
        group=name.split('.')[0];d=float((v.detach().cpu().float()-before[name].float()).norm());delta[group]=delta.get(group,0)+d*d
    dump(out/'TRAIN_TRACE.json',dict(condition=c,steps=traces,SELECT_diagnostics=selected,module_delta={k:v**.5 for k,v in delta.items()},prototype_EMA=ema,source_PT_unchanged=sha(pt)==initial_meta['checkpoint_sha256'],final_endpoint_only=True,loss_observer_detached_no_gradient_change=True))
    dump(out/'RECEIPT.json',dict(status='COMPLETED',condition=c,updates=2000,initial_step=startstep,final_python_step=m.step,checkpoint_sha256=sha(out/'endpoint2000.pth'),plan_sha256=sha(P/'LEARNER_PLAN.npz'),roles=data.access,labels_read=False,all_native_parameters_trainable=True,native_SmoothL1_unchanged=True))
def score(torch,c,out):
    base.seed(torch);m=base.native(torch).cuda();train=O/'U'/c/'train';r=read(train/'RECEIPT.json');pt=train/'endpoint2000.pth';assert sha(pt)==r['checkpoint_sha256'];z=m.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False);assert not z.missing_keys and all(k.endswith(('total_ops','total_params'))for k in z.unexpected_keys);m.step=r['final_python_step'];m.eval();before=hashes(m);rng=rngs(torch);data=base.RoleData(['CAL','QUERY']);banks=[]
    for role in ['CAL','QUERY']:
        v,start=data.load(role);ix=np.arange(start+512,start+len(v));scores=[];channels=[]
        with torch.no_grad():
            for a in range(0,len(ix),256):
                ends=ix[a:a+256]-start;x=np.stack([np.float32(v[t-191:t+1])for t in ends]);raw=m.infer(torch.from_numpy(x).cuda(),None,None,None)[0].cpu().numpy();_,s=base.native_score.native_numpy_scores([raw],31);scores.append(s.reshape(len(x),192)[:,-1]);channels.append(raw[:,-1])
        f=out/(role+'.npz');np.savez_compressed(f,index=ix,score=np.concatenate(scores),per_channel_last=np.concatenate(channels),available_index=ix,valid=np.ones(len(ix),bool));banks.append(dict(role=role,path=str(f),sha256=sha(f),points=len(ix)))
    assert before==hashes(m)and rng==rngs(torch)
    dump(out/'RECEIPT.json',dict(status='COMPLETED',condition=c,banks=banks,checkpoint_sha256=sha(pt),python_step=m.step,eval_state_modes_RNG_immutable=True,roles=data.access,labels_read=False))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['train','score']);ap.add_argument('condition',choices=['A','B','C']);args=ap.parse_args();out=O/'U'/args.condition/args.phase;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch=base.setup_torch(20 if args.phase=='train'else 30);train(torch,args.condition,out)if args.phase=='train'else score(torch,args.condition,out);dump(out/'GPU_PEAK.json',dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-start));print(json.dumps(dict(event='COMPLETED',phase=args.phase,condition=args.condition)),flush=True)
