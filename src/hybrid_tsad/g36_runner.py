"""One fixed G36 simulation experiment: isolated roles and raw score archives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from .g36_model import ConditionalResponseFlow
from .normality_density_runner import validation_mode


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,row):
    with Path(path).open('x',encoding='utf-8')as f:json.dump(row,f,indent=2,allow_nan=False)
def seed(value,device):
    random.seed(value);np.random.seed(value);torch.random.default_generator.manual_seed(value)
    if device.type=='cuda':torch.cuda.default_generators[device.index or 0].manual_seed(value)
def hashes(model):return{k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in model.state_dict().items()}


class Data:
    def __init__(self,workspace,cfg,allowed):
        self.root=workspace/cfg['data'];self.manifest=read(self.root/'manifest.json');self.allowed=set(allowed);self.access=[]
        assert(self.manifest['history'],self.manifest['block'])==(48,4)
        assert len(self.manifest['response_columns'])==4 and len(self.manifest['context_columns'])==5
    def load(self,record):
        if record['role']not in self.allowed:raise PermissionError('Role forbidden before values access')
        path=self.root/record['path'];assert sha(path)==record['sha256']
        # NPZ is lazy: label, mode, and time arrays are never decoded here.
        with np.load(path,allow_pickle=False)as z:x=z['x'].copy();index=z['index'].copy()
        assert x.shape==(len(index),9)and x.dtype==np.float32 and np.isfinite(x).all()
        assert np.array_equal(index,np.arange(*record['bounds_half_open']))
        self.access.append(dict(run=record['run'],role=record['role'],sha256=record['sha256'],fields_read=['x','index']))
        return x,index
    def one(self,role):
        records=[r for r in self.manifest['records']if r['role']==role];assert len(records)==1
        return self.load(records[0])


def batch(values,ends,device):
    assert np.all(ends>=51)and np.all(ends<len(values))
    return torch.from_numpy(np.stack([values[t-51:t+1]for t in ends])).to(device)


def make(workspace,cfg,method,device):
    if method=='LEFT_native':
        sys.path.insert(0,str(workspace/cfg['upstream']));from models.Left.LEFT import Left
        config=read(workspace/cfg['source_model_config']);config.update(seq_len=48,enc_in=9)
        model=Left(SimpleNamespace(**config))
    else:model=ConditionalResponseFlow(method,cfg['hidden'],cfg['kernel'])
    return model.to(device)


def loss(model,method,arrived):
    return model(arrived[:,-48:],None,None,None)[0]if method=='LEFT_native'else model.loss(arrived)


def prepare(workspace,config,cfg):
    root=workspace/cfg['output'];root.mkdir(parents=True,exist_ok=True);folder=root/'shared';folder.mkdir(exist_ok=False)
    data=Data(workspace,cfg,['FIT']);fit,_=data.one('FIT');rng=np.random.default_rng(cfg['seed'])
    order=rng.integers(51,len(fit),size=(cfg['updates'],cfg['batch_size']),dtype=np.int64)
    np.savez_compressed(folder/'FIT_PLAN.npz',end=order)
    identities={}
    for method in cfg['methods']:
        seed(cfg['seed'],torch.device('cpu'));model=make(workspace,cfg,method,torch.device('cpu'))
        torch.save(dict(state=model.state_dict(),native_step=getattr(model,'step',None)),folder/(method+'_INITIAL.pth'))
        identities[method]=dict(sha256=sha(folder/(method+'_INITIAL.pth')),state=hashes(model),parameters=sum(p.numel()for p in model.parameters()))
    assert identities['full']['state']==identities['history_only']['state']
    dump(folder/'MANIFEST.json',dict(config_sha256=sha(config),data_manifest_sha256=sha(data.root/'manifest.json'),
         scaler_sha256=sha(data.root/'scaler.npz'),plan_sha256=sha(folder/'FIT_PLAN.npz'),initial=identities,
         source_LEFT_sha256=sha(workspace/cfg['upstream']/'models/Left/LEFT.py'),
         source_model_config_sha256=sha(workspace/cfg['source_model_config']),FIT_access=data.access,
         no_SELECT_CAL_QUERY_values_access=True,effective_TCN_receptive_field_points=31))


def initial(workspace,config,cfg,method,device):
    folder=workspace/cfg['output']/'shared';meta=read(folder/'MANIFEST.json');assert meta['config_sha256']==sha(config)
    assert meta['data_manifest_sha256']==sha(workspace/cfg['data']/'manifest.json')
    assert sha(folder/(method+'_INITIAL.pth'))==meta['initial'][method]['sha256']
    seed(cfg['seed'],device);model=make(workspace,cfg,method,device)
    state=torch.load(folder/(method+'_INITIAL.pth'),map_location='cpu',weights_only=True);model.load_state_dict(state['state'],strict=True)
    if method=='LEFT_native':model.step=state['native_step']
    assert hashes(model)==meta['initial'][method]['state']
    return model,meta


def select(model,method,values,device):
    total=0.;count=0
    with validation_mode(model,device):
        for begin in range(51,len(values),64):
            ends=np.arange(begin,min(begin+64,len(values)));value=loss(model,method,batch(values,ends,device));assert torch.isfinite(value)
            total+=float(value)*len(ends);count+=len(ends)
    return total/count


def train(workspace,config,cfg,out,method,device,l0=False):
    model,meta=initial(workspace,config,cfg,method,device);data=Data(workspace,cfg,['FIT','SELECT']);fit,_=data.one('FIT');val,_=data.one('SELECT')
    plan=workspace/cfg['output']/'shared/FIT_PLAN.npz';assert sha(plan)==meta['plan_sha256']
    with np.load(plan,allow_pickle=False)as z:order=z['end'].copy()
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    best=float('inf');selected=-1;N=2 if l0 else cfg['updates'];started=time.monotonic();trace=[]
    def validation(step):
        nonlocal best,selected
        states=[(m,m.training)for m in model.modules()];cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(device).clone()if device.type=='cuda'else None
        v=select(model,method,val,device)
        assert torch.equal(cpu,torch.get_rng_state())and all(m.training==s for m,s in states)
        if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state(device))
        row=dict(update=step,SELECT_loss=float(v),CPU_CUDA_modes_restored=True)
        with(out/'SELECT_TRACE.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
        if v<best:
            best=v;selected=step
            if not l0:
                torch.save(dict(state=model.state_dict(),native_step=getattr(model,'step',None),method=method,
                    update=step,config=cfg,plan_sha256=meta['plan_sha256']),out/'selected.tmp')
                os.replace(out/'selected.tmp',out/'selected.pth')
    if not l0:validation(0)
    for step in range(1,N+1):
        model.train();x=batch(fit,order[step-1],device);opt.zero_grad(set_to_none=True);value=loss(model,method,x);assert value.ndim==0 and torch.isfinite(value)
        value.backward();grads=[p.grad for p in model.parameters()if p.grad is not None];assert grads and all(torch.isfinite(g).all()for g in grads)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm']);assert torch.isfinite(norm)
        condition_grad=None
        if method!='LEFT_native':condition_grad=float(model.input.weight.grad[:,4:].abs().sum())
        if method=='history_only':assert condition_grad==0
        opt.step();row=dict(update=step,loss=float(value.detach()),gradient_norm_before_clip=float(norm),condition_input_weight_gradient_L1=condition_grad,seconds=time.monotonic()-started)
        if l0:trace.append(row)
        with(out/'TRAIN_TRACE.jsonl').open('a')as f:f.write(json.dumps(row)+'\n')
        if step%cfg['select_every']==0 and not l0:validation(step)
    if l0:
        validation(N)
        if method=='full':assert trace[-1]['condition_input_weight_gradient_L1']>0
        model.eval()
        with torch.no_grad():
            if method=='LEFT_native':s=model.infer(x[:,-48:],None,None,None)[0];assert s.shape==(64,48,9)and torch.isfinite(s).all()
            else:s=model.score(x);assert s.shape==(64,)and torch.isfinite(s).all()
    else:torch.save(dict(state=model.state_dict(),native_step=getattr(model,'step',None),update=N),out/'final.pth')
    dump(out/'TRAIN_RECEIPT.json',dict(status='L0_PASS_DISCARDED'if l0 else'COMPLETED',method=method,updates=N,
         selected_update=selected,SELECT_loss=best,initial_sha256=meta['initial'][method]['sha256'],initial_state=meta['initial'][method]['state'],
         plan_sha256=meta['plan_sha256'],roles=data.access,selected_sha256=None if l0 else sha(out/'selected.pth'),
         no_CAL_QUERY_values=True,no_labels_decoded=True,loss='native nine-channel loss'if method=='LEFT_native'else'conditional response NLL mean over16 scalars'))


def score(workspace,config,cfg,out,method,device):
    trainpath=out.parent/'train';receipt=read(trainpath/'TRAIN_RECEIPT.json');assert sha(trainpath/'selected.pth')==receipt['selected_sha256']
    model,meta=initial(workspace,config,cfg,method,device);state=torch.load(trainpath/'selected.pth',map_location='cpu',weights_only=True)
    assert state['method']==method and state['config']==cfg;model.load_state_dict(state['state'],strict=True)
    if method=='LEFT_native':model.step=state['native_step']
    model.eval();before=hashes(model);native_step=getattr(model,'step',None);data=Data(workspace,cfg,['SELECT','CAL','QUERY']);banks=[]
    with torch.no_grad():
        for record in data.manifest['records']:
            if record['role']not in ['SELECT','CAL','QUERY']:continue
            values,index=data.load(record);ends=np.arange(51,len(values));parts={}
            for begin in range(0,len(ends),64):
                e=ends[begin:begin+64];arrived=batch(values,e,device)
                if method=='LEFT_native':
                    raw=model.infer(arrived[:,-48:],None,None,None)[0].cpu().numpy()
                    terminal=raw[:,-1];fields=dict(score=np.mean(np.array(terminal[:,:4]),axis=-1),terminal_channel_score=terminal)
                else:fields={k:v.cpu().numpy()for k,v in model.components(arrived).items()}
                for key,v in fields.items():parts.setdefault(key,[]).append(v)
            fields={k:np.concatenate(v)for k,v in parts.items()};ix=index[ends];assert ix[0]==record['first_scored_index']and len(ix)==record['n_scored']
            assert fields['score'].shape==(len(ix),)and np.isfinite(fields['score']).all()
            file=out/(record['run']+'__'+record['role']+'.npz')
            np.savez_compressed(file,index=ix,available_index=ix,max_input_index=ix,
                max_condition_index=ix if method=='LEFT_native'else ix-4,min_input_index=ix-47 if method=='LEFT_native'else ix-51,**fields)
            banks.append(dict(run=record['run'],role=record['role'],path=str(file),sha256=sha(file),points=len(ix)))
    assert hashes(model)==before and getattr(model,'step',None)==native_step
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',method=method,banks=banks,roles=data.access,PT_sha256=sha(trainpath/'selected.pth'),
         selected_update=state['update'],parameter_buffer_step_immutable=True,no_labels_decoded=True,no_threshold_fitting=True,
         task='original native last-point four-response mean'if method=='LEFT_native'else'four-response four-point conditional NLL, context ends t-4'))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','l0','train','score']);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--method',choices=['full','history_only','LEFT_native']);p.add_argument('--device',default='cpu');args=p.parse_args();workspace=args.workspace.resolve();config=args.config.resolve();cfg=read(config)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if args.phase=='prepare':return prepare(workspace,config,cfg)
    device=torch.device(args.device);out=workspace/cfg['output']/args.method/args.phase;out.mkdir(parents=True,exist_ok=False);begin=time.time();rc=0
    if device.type=='cuda':
        torch.cuda.set_device(device);cap=30 if args.phase=='score'else 20
        torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(device).total_memory,device);torch.cuda.reset_peak_memory_stats(device)
    dump(out/'START.json',dict(pid=os.getpid(),phase=args.phase,method=args.method,start_unix=begin,config_sha256=sha(config),runner_sha256=sha(__file__),model_sha256=sha(Path(__file__).with_name('g36_model.py')),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        if args.phase=='score':score(workspace,config,cfg,out,args.method,device)
        else:train(workspace,config,cfg,out,args.method,device,l0=args.phase=='l0')
    except BaseException:
        rc=1;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:
        end=time.time();dump(out/'OS.json',dict(pid=os.getpid(),start_unix=begin,end_unix=end,seconds=end-begin,exit_code=rc,
             GPU_hours=(end-begin)/3600 if device.type=='cuda'else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=='cuda'else 0))


if __name__=='__main__':main()
