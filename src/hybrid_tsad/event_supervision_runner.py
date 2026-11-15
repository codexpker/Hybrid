"""Physical-role isolated paired F experiment; no QUERY labels or threshold fit."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .event_supervision import EventSupervisedLEFT,event_balanced_loss,TRAIN_PATHS
from .normality_density_runner import validation_mode


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,row):
    with Path(path).open('x',encoding='utf-8')as f:json.dump(row,f,indent=2,allow_nan=False)
def load(path):
    with np.load(path,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def state_hash(model,predicate=lambda k:True):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in model.state_dict().items()if predicate(k)}
def seed_all(seed,device):
    random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
    if device.type=='cuda':torch.cuda.default_generators[device.index or 0].manual_seed(seed)
def native(workspace,cfg):
    sys.path.insert(0,str(workspace/cfg['upstream']))
    from models.Left.LEFT import Left
    from types import SimpleNamespace
    conf=read(workspace/cfg['source_preparation']/'configs/SWAT/model_configs_Left.json')
    return Left(SimpleNamespace(**conf))


def physical(workspace,cfg,role):
    P=workspace/cfg['source_preparation'];meta=next(r for r in read(P/'INPUT_QUALIFICATION.json')['roles']if r['role']==role)
    path=P/'data'/f'{role}.npz';assert sha(path)==meta['sha256']
    z=load(path);assert z['observed'].all();sc=read(P/'FIT_ONLY_SCALER.json')
    v=np.float32((z['values'].astype(np.float64)-np.asarray(sc['mean']))/np.asarray(sc['scale']))
    assert np.isfinite(v).all()
    return v,meta


def pack(path,blocks):
    ptr=np.r_[0,np.cumsum([len(b['values'])for b in blocks])].astype(np.int64)
    np.savez_compressed(path,values=np.concatenate([b['values']for b in blocks]),ptr=ptr,
        original_start=np.array([b['original_start']for b in blocks],np.int64),
        group_id=np.array([b['group_id']for b in blocks],np.int64))


def prepare(workspace,config,cfg):
    P=workspace/cfg['preparation'];P.mkdir(parents=True,exist_ok=True)
    assert not(P/'DATA_MANIFEST.json').exists()
    src=workspace/cfg['source_preparation'];planfile=src/'PAIRED_PLAN.npz'
    assert sha(planfile)==read(src/'DATA_PLAN_GATE.json')['plan_sha256']
    # The only anomaly metadata seen here are the thirteen pre-QUERY donors.
    with np.load(planfile,allow_pickle=False)as z:starts=z['donor_event_starts'].copy();ends=z['donor_event_ends'].copy()
    assert len(starts)==13 and np.all(starts[1:]>=ends[:-1])and np.all(ends-starts>=192)
    donor,dr=physical(workspace,cfg,'DONOR');assert np.all(starts>=dr['original_start'])and np.all(ends<=dr['end'])
    groups=[]
    for i,(a,b)in enumerate(zip(starts,ends)):
        lo=int(a)-191;hi=int(b);assert lo>=dr['original_start']
        groups.append(dict(values=donor[lo-dr['original_start']:hi-dr['original_start']],original_start=lo,group_id=i))
    pack(P/'TRAIN_POSITIVE.npz',groups[:10]);pack(P/'SELECT_POSITIVE.npz',groups[10:])
    fit,fr=physical(workspace,cfg,'FIT');u,ur=physical(workspace,cfg,'U_FIT');normal=[]
    boundaries=np.linspace(0,len(fit),4,dtype=np.int64)
    for i,(a,b)in enumerate(zip(boundaries[:-1],boundaries[1:])):normal.append(dict(values=fit[a:b],original_start=int(a+fr['original_start']),group_id=i))
    normal.append(dict(values=u,original_start=ur['original_start'],group_id=3));pack(P/'TRAIN_NORMAL.npz',normal)
    select,sr=physical(workspace,cfg,'SELECT');bounds=np.linspace(0,len(select),5,dtype=np.int64)
    normal_select=[dict(values=select[a:b],original_start=int(a+sr['original_start']),group_id=i)for i,(a,b)in enumerate(zip(bounds[:-1],bounds[1:]))]
    pack(P/'SELECT_NORMAL.npz',normal_select)
    rng=np.random.default_rng(cfg['seed']);N=cfg['updates'];pg=np.empty((N,2),np.int64);ng=np.empty_like(pg)
    pe=np.empty((N,2,32),np.int64);ne=np.empty_like(pe)
    for step in range(N):
        pg[step]=(2*step+np.arange(2))%10;ng[step]=(2*step+np.arange(2))%4
        for j,g in enumerate(pg[step]):
            pe[step,j]=rng.integers(191,len(groups[g]['values']),size=32)
            pe[step,j,0]=rng.integers(382,len(groups[g]['values'])) # one fully internal positive window
        for j,g in enumerate(ng[step]):ne[step,j]=rng.integers(191,len(normal[g]['values']),size=32)
    np.savez_compressed(P/'FIT_PLAN.npz',positive_group=pg,normal_group=ng,positive_end=pe,normal_end=ne)
    # Validation is the complete three held-out positive events and fixed normal positions.
    vg=[];ve=[];kind=[]
    for g,b in enumerate(groups[10:]):
        e=np.arange(191,len(b['values']));vg.extend([g]*len(e));ve.extend(e);kind.extend([1]*len(e))
    for g,b in enumerate(normal_select):
        e=np.linspace(191,len(b['values'])-1,cfg['normal_validation_points_per_block'],dtype=np.int64)
        vg.extend([g]*len(e));ve.extend(e);kind.extend([0]*len(e))
    np.savez_compressed(P/'SELECT_PLAN.npz',group=np.asarray(vg,np.int64),end=np.asarray(ve,np.int64),positive=np.asarray(kind,np.int8))
    files={name:sha(P/name)for name in ['TRAIN_POSITIVE.npz','SELECT_POSITIVE.npz','TRAIN_NORMAL.npz','SELECT_NORMAL.npz','FIT_PLAN.npz','SELECT_PLAN.npz']}
    events=[dict(id=i,start=int(a),end=int(b),role='TRAIN'if i<10 else'SELECT',points=int(b-a))for i,(a,b)in enumerate(zip(starts,ends))]
    dump(P/'DATA_MANIFEST.json',dict(config_sha256=sha(config),files=files,events=events,source_roles=[fr,ur,sr,dr],
         scaler_sha256=sha(src/'FIT_ONLY_SCALER.json'),donor_event_plan_sha256=sha(planfile),
         source_PT_sha256=sha(workspace/cfg['source_checkpoint']),source_config_sha256=sha(src/'configs/SWAT/model_configs_Left.json'),
         source_LEFT_sha256=sha(workspace/cfg['upstream']/'models/Left/LEFT.py'),
         native_selected_epoch=13,native_final_python_step=read(workspace/'results/left_multievent_mixed_u_20261111_v1/left_train/RUNTIME_STATE.json')['final_python_step'],
         normal_train_blocks=[dict(group=b['group_id'],start=b['original_start'],end=b['original_start']+len(b['values']))for b in normal],
         normal_SELECT_blocks=[dict(group=b['group_id'],start=b['original_start'],end=b['original_start']+len(b['values']))for b in normal_select],
         train_positive_events=10,validation_positive_events=3,independent_run_count=1,
         QUERY_labels_or_values_not_accessed=True,CAL_not_accessed=True))


def batch(packed,groups,ends,device):
    rows=[]
    for g,t in zip(np.asarray(groups).ravel(),np.asarray(ends).ravel()):
        lo,hi=packed['ptr'][g:g+2];assert 191<=t<hi-lo
        rows.append(packed['values'][lo+t-191:lo+t+1])
    x=np.stack(rows);assert x.shape[1:]==(192,31)
    return torch.from_numpy(x).to(device)


def assets(workspace,config,cfg,roles):
    P=workspace/cfg['preparation'];m=read(P/'DATA_MANIFEST.json');assert m['config_sha256']==sha(config)
    allowed=set(roles);result={}
    for role in allowed:
        file=P/f'{role}.npz';assert sha(file)==m['files'][file.name];result[role]=load(file)
    return result,m


def initialise(workspace,config,cfg):
    P=workspace/cfg['preparation'];m=read(P/'DATA_MANIFEST.json');seed_all(cfg['seed'],torch.device('cpu'))
    base=native(workspace,cfg);pt=workspace/cfg['source_checkpoint'];assert sha(pt)==m['source_PT_sha256']
    result=base.load_state_dict(torch.load(pt,map_location='cpu',weights_only=True),strict=False)
    assert not result.missing_keys and all(k.endswith(('total_ops','total_params'))for k in result.unexpected_keys)
    base.step=m['native_final_python_step'];model=EventSupervisedLEFT(base)
    torch.save(dict(state=model.state_dict(),native_step=base.step,config_sha256=sha(config)),P/'INITIAL.pth')
    dump(P/'INITIAL_IDENTITY.json',dict(initial_sha256=sha(P/'INITIAL.pth'),state=state_hash(model),
         head=state_hash(model,lambda k:k.startswith('head.')),feature=state_hash(model,lambda k:k.startswith('native.')),
         source_PT_sha256=sha(pt),trainable=[n for n,p in model.named_parameters()if p.requires_grad],
         trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad)))


def model_from_initial(workspace,cfg,device):
    P=workspace/cfg['preparation'];seed_all(cfg['seed'],device);m=EventSupervisedLEFT(native(workspace,cfg))
    identity=read(P/'INITIAL_IDENTITY.json');assert sha(P/'INITIAL.pth')==identity['initial_sha256']
    state=torch.load(P/'INITIAL.pth',map_location='cpu',weights_only=True);m.load_state_dict(state['state'],strict=True)
    m.native.step=state['native_step'];m=m.to(device)
    assert state_hash(m)==identity['state'];return m,identity


def validation(model,data,device):
    p=data['SELECT_PLAN'];group_losses=[];normal=[]
    with validation_mode(model,device):
        for positive,name,number in [(1,'SELECT_POSITIVE',3),(0,'SELECT_NORMAL',4)]:
            per=[]
            for g in range(number):
                mask=(p['positive']==positive)&(p['group']==g);ends=p['end'][mask];loss=[]
                for i in range(0,len(ends),128):
                    e=ends[i:i+128];x=batch(data[name],np.full(len(e),g),e,device);s=model(x)
                    loss.append(F.binary_cross_entropy_with_logits(s,torch.full_like(s,float(positive)),reduction='none').cpu().numpy())
                per.append(float(np.concatenate(loss).mean()))
            (group_losses if positive else normal).extend(per)
    return .5*np.mean(group_losses)+.5*np.mean(normal),dict(positive_event_meanBCE=group_losses,normal_block_meanBCE=normal)


def frozen_hash(model):
    return state_hash(model,lambda k:k.startswith('native.')and k.split('.')[1]not in TRAIN_PATHS)


def cpu_check(workspace,config,cfg,out):
    data,meta=assets(workspace,config,cfg,['TRAIN_POSITIVE','TRAIN_NORMAL'])
    model,identity=model_from_initial(workspace,cfg,torch.device('cpu'));model.eval()
    x=batch(data['TRAIN_NORMAL'],np.array([0,1]),np.array([300,400]),torch.device('cpu'))
    captured=[];hook=model.native.fusion.register_forward_hook(lambda m,i,o:captured.append(tuple(v.detach().clone()for v in o)))
    with torch.no_grad():model.native.infer(x,None,None,None)
    hook.remove();expected=torch.cat([v.mean(1)for v in captured[0]],-1)
    with torch.no_grad():actual=model.features(x)
    assert torch.equal(expected,actual)
    protected=frozen_hash(model);step=model.native.step;model.train();optim=torch.optim.Adam([p for p in model.parameters()if p.requires_grad],lr=cfg['learning_rate'])
    result=model(x);loss=F.binary_cross_entropy_with_logits(result,torch.tensor([0.,1.]));loss.backward()
    grad={}
    for name,p in model.named_parameters():
        if p.grad is not None:
            assert p.requires_grad and torch.isfinite(p.grad).all();key=name.split('.')[1]if name.startswith('native.')else'head'
            grad[key]=grad.get(key,0.)+float(p.grad.square().sum())
    assert all(grad.get(k,0)>0 for k in ['head','encoder','Et','Ef','fusion','patch_embedding']),grad
    optim.step();assert frozen_hash(model)==protected and model.native.step==step
    try:model(torch.zeros(2,193,31));raise AssertionError('future row accepted')
    except ValueError:pass
    dump(out/'CPU_CHECK.json',dict(status='PASS_DISCARDED',native_fusion_tokens_bitexact=True,
         features=list(actual.shape),logits=list(result.shape),gradient_squares=grad,
         frozen_parameter_buffer_and_native_step_unchanged=True,future_extra_row_rejected=True,
         initial_identity_sha256=identity['initial_sha256'],no_CAL_QUERY_access=True))


def run_train(workspace,config,cfg,out,arm,device,l0=False):
    data,m=assets(workspace,config,cfg,['TRAIN_POSITIVE','TRAIN_NORMAL','FIT_PLAN','SELECT_POSITIVE','SELECT_NORMAL','SELECT_PLAN'])
    model,identity=model_from_initial(workspace,cfg,device);protected=frozen_hash(model);native_step=model.native.step
    optimizer=torch.optim.Adam([p for p in model.parameters()if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    q=cfg['arms'][arm];N=2 if l0 else cfg['updates'];labels=torch.cat((torch.ones(2,32),torch.zeros(2,32))).to(device)
    best=float('inf');selected=0;grad_modules={};initial_live={n:p.detach().cpu().clone()for n,p in model.named_parameters()if p.requires_grad}
    def select(step):
        nonlocal best,selected
        modes=[(module,module.training)for module in model.modules()];cpu=torch.get_rng_state().clone()
        cuda=torch.cuda.get_rng_state(device).clone()if device.type=='cuda'else None
        py=random.getstate();nr=np.random.get_state()
        value,parts=validation(model,data,device)
        assert all(module.training==mode for module,mode in modes)and torch.equal(cpu,torch.get_rng_state())
        if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state(device))
        assert random.getstate()==py and all(np.array_equal(a,b)for a,b in zip(nr,np.random.get_state()))
        row=dict(update=step,mean_BCE=float(value),modes_CPU_CUDA_Python_numpy_RNG_restored=True,**parts)
        with(out/'SELECT_TRACE.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
        if value<best:
            best=float(value);selected=step
            if not l0:
                torch.save(dict(state=model.state_dict(),config=cfg,arm=arm,update=step,native_step=native_step,
                                initial_sha256=identity['initial_sha256'],plan_sha256=m['files']['FIT_PLAN.npz']),out/'selected.tmp')
                os.replace(out/'selected.tmp',out/'selected.pth')
        return row
    if not l0:select(0)
    start=time.monotonic()
    for step in range(1,N+1):
        model.train();plan=data['FIT_PLAN'];pos=batch(data['TRAIN_POSITIVE'],plan['positive_group'][step-1,:,None].repeat(32,1),plan['positive_end'][step-1],device)
        neg=batch(data['TRAIN_NORMAL'],plan['normal_group'][step-1,:,None].repeat(32,1),plan['normal_end'][step-1],device)
        x=torch.cat((pos,neg));optimizer.zero_grad(set_to_none=True);logit=model(x).reshape(4,32)
        loss,grouped,raw=event_balanced_loss(logit,labels,q);assert torch.isfinite(loss);loss.backward()
        active={}
        for name,p in model.named_parameters():
            if p.grad is not None:
                assert p.requires_grad and bool(torch.isfinite(p.grad).all());key=name.split('.')[1]if name.startswith('native.')else'head'
                active[key]=active.get(key,0.)+float(p.grad.detach().square().sum())
        assert all(active.get(k,0)>0 for k in ['head','encoder','Et','Ef','fusion','patch_embedding']),active
        grad_modules=active;optimizer.step()
        assert frozen_hash(model)==protected and model.native.step==native_step
        row=dict(update=step,loss=float(loss.detach()),group_loss=grouped.detach().cpu().tolist(),
                 positive_event_ids=plan['positive_group'][step-1].tolist(),normal_block_ids=plan['normal_group'][step-1].tolist(),
                 group_gradient_squares=active,seconds=time.monotonic()-start)
        with(out/'TRAIN_TRACE.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
        if step%cfg['validation_every']==0 and not l0:select(step)
    if l0:
        select(N);model.eval()
        with torch.no_grad():s=model(x[:2]);assert s.shape==(2,)and torch.isfinite(s).all()
    delta={n:float((p.detach().cpu()-initial_live[n]).square().sum())for n,p in model.named_parameters()if p.requires_grad}
    if not l0:torch.save(dict(state=model.state_dict(),update=N),out/'final.pth')
    dump(out/'TRAIN_RECEIPT.json',dict(status='L0_PASS_DISCARDED'if l0 else'COMPLETED',arm=arm,q=q,updates=N,
         selected_update=selected,selected_mean_BCE=best,initial_sha256=identity['initial_sha256'],head_initial_identity=identity['head'],
         plan_sha256=m['files']['FIT_PLAN.npz'],data_sha256=m['files'],source_PT_sha256=m['source_PT_sha256'],
         frozen_state_unchanged=True,native_step_unchanged=True,gradient_squares=grad_modules,parameter_delta_squared=delta,
         selected_sha256=None if l0 else sha(out/'selected.pth'),final_sha256=None if l0 else sha(out/'final.pth'),
         QUERY_CAL_not_accessed=True,validation_meanBCE_same_for_both_arms=True))


def score(workspace,config,cfg,out,arm,device):
    train=out.parent/'train';receipt=read(train/'TRAIN_RECEIPT.json');assert sha(train/'selected.pth')==receipt['selected_sha256']
    model,_=model_from_initial(workspace,cfg,device);state=torch.load(train/'selected.pth',map_location='cpu',weights_only=True)
    assert state['config']==cfg and state['arm']==arm;model.load_state_dict(state['state'],strict=True);model.native.step=state['native_step'];model.eval()
    initial=state_hash(model);banks=[]
    with torch.no_grad():
        for role in ['CAL','QUERY']:
            values,meta=physical(workspace,cfg,role);lo,hi=cfg[role+'_axis'];assert meta['original_start']<=lo<hi<=meta['end']
            index=np.arange(lo,hi,dtype=np.int64);parts=[]
            for start in range(0,len(index),128):
                t=index[start:start+128]-meta['original_start'];assert np.all(t>=191)
                x=torch.from_numpy(np.stack([values[end-191:end+1]for end in t])).to(device);parts.append(model(x).cpu().numpy())
            logit=np.concatenate(parts);assert logit.shape==index.shape and np.isfinite(logit).all()
            f=out/f'{role}.npz';np.savez_compressed(f,index=index,score=logit,logit=logit,available_index=index,max_input_index=index)
            banks.append(dict(role=role,path=str(f),sha256=sha(f),points=len(index),physical_role_sha256=meta['sha256']))
    assert state_hash(model)==initial
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',arm=arm,banks=banks,selected_checkpoint_sha256=sha(train/'selected.pth'),
         selected_update=state['update'],seed=cfg['seed'],score_is_raw_logit=True,parameter_buffer_unchanged=True,QUERY_labels_not_accessed=True,
         CAL_is_original_pre_QUERY_role=True,no_threshold_fitted=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','initialise','cpu_check','l0','train','score'])
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--arm',choices=['tail25','mean'],default='tail25');p.add_argument('--device',default='cpu');args=p.parse_args()
    workspace=args.workspace.resolve();config=args.config.resolve();cfg=read(config)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if args.phase=='prepare':return prepare(workspace,config,cfg)
    if args.phase=='initialise':return initialise(workspace,config,cfg)
    device=torch.device(args.device);out=workspace/cfg['output']/args.arm/args.phase;out.mkdir(parents=True,exist_ok=False)
    begin=time.time();rc=0
    if device.type=='cuda':
        torch.cuda.set_device(device);cap=20 if args.phase in ['l0','train']else 30
        torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(device).total_memory,device);torch.cuda.reset_peak_memory_stats(device)
    dump(out/'START.json',dict(pid=os.getpid(),start_unix=begin,phase=args.phase,arm=args.arm,config_sha256=sha(config),runner_sha256=sha(__file__),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        if args.phase=='score':score(workspace,config,cfg,out,args.arm,device)
        elif args.phase=='cpu_check':cpu_check(workspace,config,cfg,out)
        else:run_train(workspace,config,cfg,out,args.arm,device,l0=args.phase=='l0')
    except BaseException:
        rc=1;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:
        end=time.time();dump(out/'OS.json',dict(pid=os.getpid(),exit_code=rc,start_unix=begin,end_unix=end,seconds=end-begin,
             GPU_hours=(end-begin)/3600 if device.type=='cuda'else 0,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=='cuda'else 0))


if __name__=='__main__':main()
