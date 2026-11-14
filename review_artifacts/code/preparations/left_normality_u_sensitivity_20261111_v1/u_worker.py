"""Fixed ordinary within-target native LEFT U; generator tags never enter loss."""
import argparse,hashlib,importlib.util,json,random,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=module('accepted_native',R/'preparations/cross_domain_left_arx_20261111_v1/worker.py')
u=module('u_fixed',P/'runtime/left_native_u.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    with Path(p).open('x')as f:json.dump(d,f,indent=2,allow_nan=False)
def read(p):return json.loads(Path(p).read_text())
def load_plan():
    assert sha(P/'PAIRED_PLAN.npz')==read(P/'CPU_U_PLAN_TESTS.json')['plan_sha256']
    with np.load(P/'PAIRED_PLAN.npz')as z:return{k:z[k].copy()for k in z.files}
def hashes(m):return{k:base.sha_bytes(v)for k,v in m.state_dict().items()}
def rngs(torch):return dict(CPU=base.sha_bytes(torch.get_rng_state()),CUDA=[base.sha_bytes(s)for s in torch.cuda.get_rng_state_all()])
def initial(torch):
    base.seed(torch);m=base.native(torch).cuda();source=R/'results/cross_domain_left_arx_20261111_v1/left_train';state=read(source/'RUNTIME_STATE.json');pt=source/'model/SMD/Left/checkpoints_0/checkpoint.pth';assert sha(pt)==state['checkpoint_sha256'];loaded=m.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False);assert not loaded.missing_keys and all(k.endswith(('total_ops','total_params'))for k in loaded.unexpected_keys);m.step=2200;m.train();assert all(v.requires_grad for v in m.parameters());return m,sha(pt)
def select_diagnostic(torch,m,v,update):
    before=hashes(m);step=m.step;debug=m.last_x_res;oldrng=rngs(torch);modes={n:x.training for n,x in m.named_modules()}
    starts=np.arange(0,len(v)-192+1,192);x=np.stack([np.float32(v[t:t+192])for t in starts])
    with u.diagnostic_eval(m,torch):loss=float(m(torch.from_numpy(x).cuda(),None,None,None)[0])
    assert hashes(m)==before and m.step==step and m.last_x_res is debug and rngs(torch)==oldrng and {n:x.training for n,x in m.named_modules()}==modes
    return dict(update=update,SELECT_loss=loss,SELECT_windows=len(starts),python_step=step,mode_step_buffers_parameters_RNG_restored=True,no_checkpoint_selection=True)
def train(torch,condition,out):
    fixed=read(P/'FIXED_U_SENSITIVITY.json');assert fixed['updates']==200 and fixed['batch_size']==128
    m,source=initial(torch);data=base.RoleData(['FIT','SELECT']);fit,_=data.load('FIT');select,_=data.load('SELECT');plan=load_plan();opt=torch.optim.Adam(m.parameters(),lr=1e-4);before={k:v.detach().cpu().clone()for k,v in m.state_dict().items()};initial_rng=rngs(torch)
    np.savez(out/'INITIAL_RNG.npz',CPU=torch.get_rng_state().cpu().numpy(),CUDA=torch.cuda.get_rng_state().cpu().numpy())
    initial_meta=dict(source_PT_sha256=source,source_python_step=2200,initial_state_sha256=hashes(m),rng=initial_rng,plan_sha256=sha(P/'PAIRED_PLAN.npz'),optimizer='freshAdam lr1e-4 defaults; no scheduler/weight_decay',model_all_train=True)
    dump(out/'INITIAL_IDENTITY.json',initial_meta);selects=[select_diagnostic(torch,m,select,0)];assert abs(selects[0]['SELECT_loss']-read(R/'results/cross_domain_left_arx_20261111_v1/left_train/RUNTIME_STATE.json')['selected_loss'])<1e-6
    ema={'t_calls':0,'f_calls':0,'t_vectors':0,'f_vectors':0}
    for prefix,bank in [('t',m.proto_t),('f',m.proto_f)]:
        original=bank.ema_update_hard
        def observe(z,nn_idx,momentum=.99,original=original,prefix=prefix):
            ema[prefix+'_calls']+=1;ema[prefix+'_vectors']+=int(z.shape[0]);return original(z,nn_idx,momentum=momentum)
        bank.ema_update_hard=observe
    traces=[]
    for update in range(200):
        x=u.native_batch(fit,plan,update,condition);opt.zero_grad();loss=m(torch.from_numpy(x).cuda(),None,None,None)[0];assert torch.isfinite(loss);loss.backward();grads=[p.grad for p in m.parameters()if p.grad is not None];assert grads and all(torch.isfinite(g).all()for g in grads);norm=float(torch.stack([g.detach().float().norm()for g in grads]).norm());opt.step()
        assert m.step==2200+update+1
        traces.append(dict(update=update+1,loss=float(loss.detach()),lr=opt.param_groups[0]['lr'],gradient_norm=norm,actual_python_step=m.step,finite=True,rng_after=rngs(torch),constructor_replacement_slots=int(plan['replacement_mask'][update].sum())if condition!='A'else 0,constructor_synthetic_slots=int(plan['replacement_mask'][update].sum())if condition=='B'else 0))
        if(update+1)%50==0:
            selects.append(select_diagnostic(torch,m,select,update+1));print(json.dumps(dict(event='U_UPDATE',condition=condition,update=update+1,loss=traces[-1]['loss'],SELECT_loss=selects[-1]['SELECT_loss'])),flush=True)
    endpoint=out/'endpoint200.pth';torch.save(m.state_dict(),endpoint);torch.save(opt.state_dict(),out/'optimizer200.pth');np.savez(out/'FINAL_RNG.npz',CPU=torch.get_rng_state().cpu().numpy(),CUDA=torch.cuda.get_rng_state().cpu().numpy())
    deltas={};all_nonzero=0
    for name,v in m.state_dict().items():
        delta=float((v.detach().cpu().float()-before[name].float()).norm());all_nonzero+=delta>0;group=name.split('.')[0];deltas[group]=deltas.get(group,0)+delta*delta
    deltas={k:v**.5 for k,v in deltas.items()}
    dump(out/'TRAIN_TRACE.json',dict(condition=condition,steps=traces,SELECT_diagnostics=selects,module_parameter_buffer_delta_norm=deltas,changed_state_tensors=all_nonzero,prototype_EMA=ema,source_PT_sha256=source,source_file_unchanged=sha(R/'results/cross_domain_left_arx_20261111_v1/left_train/model/SMD/Left/checkpoints_0/checkpoint.pth')==source))
    dump(out/'RECEIPT.json',dict(status='COMPLETED',condition=condition,updates=200,final_python_step=2400,primary_endpoint=200,checkpoint_sha256=sha(endpoint),optimizer_sha256=sha(out/'optimizer200.pth'),plan_sha256=sha(P/'PAIRED_PLAN.npz'),labels_read=False,roles=data.access,SELECT_never_selects_checkpoint=True,all_native_parameters_trainable=True,fresh_optimizer=True))
def score(torch,condition,out):
    base.seed(torch);m=base.native(torch).cuda();train=O/'U'/condition/'train';r=read(train/'RECEIPT.json');pt=train/'endpoint200.pth';assert sha(pt)==r['checkpoint_sha256'];loaded=m.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False);assert not loaded.missing_keys and all(k.endswith(('total_ops','total_params'))for k in loaded.unexpected_keys);m.step=2400;m.eval();before=hashes(m);rng=rngs(torch);data=base.RoleData(['CAL','QUERY']);banks=[]
    for role in ['CAL','QUERY']:
        v,start=data.load(role);index=np.arange(start+512,start+len(v));scores=[];per_channel=[]
        with torch.no_grad():
            for a in range(0,len(index),256):
                ends=index[a:a+256]-start;x=np.stack([np.float32(v[t-191:t+1])for t in ends]);raw=m.infer(torch.from_numpy(x).cuda(),None,None,None)[0].cpu().numpy();_,s=base.native_score.native_numpy_scores([raw],38);scores.append(s.reshape(len(x),192)[:,-1]);per_channel.append(raw[:,-1])
        f=out/(role+'.npz');np.savez_compressed(f,index=index,score=np.concatenate(scores),per_channel_last=np.concatenate(per_channel),available_index=index,valid=np.ones(len(index),bool));banks.append(dict(role=role,path=str(f),sha256=sha(f),points=len(index)))
    assert hashes(m)==before and rngs(torch)==rng and m.step==2400
    dump(out/'RECEIPT.json',dict(status='COMPLETED',condition=condition,banks=banks,checkpoint_sha256=sha(pt),python_step=2400,eval_state_modes_RNG_immutable=True,labels_read=False,roles=data.access))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['train','score']);ap.add_argument('condition',choices=['A','B','C']);args=ap.parse_args();start=time.monotonic();out=O/'U'/args.condition/args.phase;out.mkdir(parents=True,exist_ok=False);torch=base.setup_torch(20 if args.phase=='train'else 30);train(torch,args.condition,out)if args.phase=='train'else score(torch,args.condition,out);dump(out/'GPU_PEAK.json',dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-start));print(json.dumps(dict(event='PHASE_COMPLETED',condition=args.condition,phase=args.phase,seconds=time.monotonic()-start)),flush=True)
