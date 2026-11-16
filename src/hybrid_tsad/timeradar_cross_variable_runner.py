"""Fixed official TimeRadar decoder/complex-channel adapter experiment.

Source, weight and physical role identities are required from the data contract.
No labels are decoded by this runner. Official native forward/scorer are reused.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .timeradar_cross_variable import TimeRadarCrossVariable
from .normality_density_runner import validation_mode


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,r):
    with Path(p).open('x',encoding='utf-8')as f:json.dump(r,f,indent=2,allow_nan=False)
def seed(n,device):
    random.seed(n);np.random.seed(n);torch.random.default_generator.manual_seed(n)
    if device.type=='cuda':torch.cuda.default_generators[device.index or 0].manual_seed(n)
def state_hashes(m):return{k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in m.state_dict().items()}
def resolve(w,p):return Path(p)if Path(p).is_absolute()else w/p


class RoleReader:
    def __init__(self,w,cfg,record,allowed):
        self.w,self.cfg,self.record=w,cfg,record;self.allowed=set(allowed);self.access=[]
        assert sha(resolve(w,cfg['data_binding']))==cfg['data_binding_sha256']
        self.binding=read(resolve(w,cfg['data_binding']))
        self.meta={v['id']:v for v in self.binding['records']}[record]
    def load(self,role):
        if role not in self.allowed:raise PermissionError('Role forbidden before values access')
        rec={v['role']:v for v in self.meta['roles']}[role];p=resolve(self.w,rec['path'])
        assert sha(p)==rec['sha256']
        with np.load(p,allow_pickle=False)as z:
            # Only these arrays are decoded. Evaluation-side labels are never opened.
            x,index=z['x'].copy(),z['index'].copy()
        assert x.dtype==np.float32 and x.ndim==2 and np.isfinite(x).all() and list(x.shape)==rec['shape']
        assert index.ndim==1 and len(index)==len(x)and np.all(np.diff(index)==1)
        self.access.append(dict(record=self.record,role=role,run=rec['run'],path=str(p),sha256=rec['sha256'],fields_read=['x','index']))
        return x,index


def model(w,cfg,arm,device):
    source=resolve(w,cfg['author_source']);assert sha(source)==cfg['author_source_sha256']
    checkpoint=resolve(w,cfg['checkpoint']);assert sha(checkpoint)==cfg['checkpoint_sha256']
    spec=importlib.util.spec_from_file_location('hybrid_official_timeradar_cross',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    native=module.Model(SimpleNamespace(**cfg['model']))
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict'in saved:saved=saved['state_dict']
    # The original public110-key file is a DDP state, including exactly three
    # CDL bookkeeping buffers; preserve them even though this task never uses CDL.
    if all(k.startswith('module.')for k in saved):saved={k[7:]:v for k,v in saved.items()}
    extras=set(saved)-set(native.state_dict())
    if extras!={'tau_n','tau_a','tau_inited'}:raise ValueError('Unexpected original checkpoint extra keys: '+str(extras))
    for name in sorted(extras):native.register_buffer(name,saved[name].detach().clone())
    native.load_state_dict(saved,strict=True)
    wrapped=TimeRadarCrossVariable(native,arm,cfg['adapter_rank']).to(device)
    return wrapped


def native_batches(values,batch_size):
    starts=np.arange(0,len(values)-99,100,dtype=np.int64)
    for begin in range(0,len(starts),batch_size):yield starts[begin:begin+batch_size]
def tensor_windows(values,starts,device):
    return torch.from_numpy(np.stack([values[t:t+100]for t in starts])).to(device)
def mse(copies,x):return (copies-x[None]).square().mean()


def parity(w,config,cfg,out,record,device):
    data=RoleReader(w,cfg,record,['FIT']);x,index=data.load('FIT');spec=cfg['records'][record]
    starts=next(iter(native_batches(x,spec['batch_size'])))[:4]
    # Real batch and original checkpoint; no adaptation update.
    batch=tensor_windows(x,starts,device)
    seed(cfg['seed'],device);wrapper=model(w,cfg,'cross',device).eval()
    before=state_hashes(wrapper.native);rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state(device).clone()if device.type=='cuda'else None
    # Remove the hook completely for the direct official-forward witness.
    wrapper._hook.remove();wrapper._hook=None
    try:
        with torch.no_grad():native_repr,native_out=wrapper.native(batch,norm=spec['norm'],copies=cfg['copies'])
    finally:wrapper._hook=wrapper.native.decoder.register_forward_pre_hook(wrapper._adapt)
    after_cpu=torch.get_rng_state().clone();after_cuda=torch.cuda.get_rng_state(device).clone()if device.type=='cuda'else None
    torch.set_rng_state(rng)
    if crng is not None:torch.cuda.set_rng_state(crng,device)
    wrapper.adapter_enabled=True
    with torch.no_grad():repr_,outputs=wrapper(batch,norm=spec['norm'],copies=cfg['copies'])
    assert torch.equal(native_repr,repr_)and torch.equal(native_out,outputs)
    assert torch.equal(after_cpu,torch.get_rng_state())
    if after_cuda is not None:assert torch.equal(after_cuda,torch.cuda.get_rng_state(device))
    score0=wrapper.cal_anomaly_score(batch,native_out,nn.MSELoss(),L=spec['L'])
    score1=wrapper.cal_anomaly_score(batch,outputs,nn.MSELoss(),L=spec['L'])
    assert torch.equal(score0,score1) and before==state_hashes(wrapper.native)
    np.savez_compressed(out/'PARITY.npz',starts=starts,index=np.concatenate([index[t:t+100]for t in starts]),score=score1.cpu().numpy(),native_copies=native_out.cpu().numpy())
    dump(out/'PARITY_RECEIPT.json',dict(status='PASS_BITEXACT',record=record,shape=list(batch.shape),repr_shape=list(repr_.shape),copies_shape=list(outputs.shape),source_sha256=sha(resolve(w,cfg['author_source'])),checkpoint_sha256=sha(resolve(w,cfg['checkpoint'])),roles=data.access,score_bitexact=True,RNG_consumption_identical=True,native_parameters_unchanged=True,adapter_zero=True,limits='Adapter does not mix B/copies; native FRFT order still aggregates batch/channel.'))
    wrapper.close()


def selection(m,values,device,cfg,record):
    spec=cfg['records'][record];total=0.;count=0
    with validation_mode(m,device):
        seed(cfg['select_seed'],device)
        for starts in native_batches(values,spec['batch_size']):
            x=tensor_windows(values,starts,device);_,out=m(x,norm=spec['norm'],copies=cfg['copies'])
            loss=mse(out,x);assert torch.isfinite(loss)
            total+=float(loss)*len(starts);count+=len(starts)
    return total/count


def optimizer(m,cfg):
    # Same author grouping: all trainable non-LayerNorm non-bias names decay.
    no_decay=set()
    for prefix,module in m.named_modules():
        if isinstance(module,nn.LayerNorm):no_decay.update(prefix+'.'+n for n,_ in module.named_parameters())
    groups=[dict(params=[p for n,p in m.named_parameters()if p.requires_grad and n not in no_decay and 'bias'not in n],weight_decay=cfg['weight_decay']),dict(params=[p for n,p in m.named_parameters()if p.requires_grad and(n in no_decay or 'bias'in n)],weight_decay=0.)]
    return torch.optim.AdamW(groups,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'])


def train(w,config,cfg,out,record,arm,device):
    data=RoleReader(w,cfg,record,['FIT','SELECT']);fit,ix=data.load('FIT');val,_=data.load('SELECT');spec=cfg['records'][record]
    # Stochastic full fixed100 windows, no role boundary crossings.
    available=np.arange(0,len(fit)-99,100,dtype=np.int64)
    plan=np.random.default_rng(cfg['seed']).choice(available,size=(cfg['updates'],spec['batch_size']),replace=True)
    np.savez_compressed(out/'FIT_PLAN.npz',starts=plan)
    seed(cfg['seed'],device);m=model(w,cfg,arm,device);initial=state_hashes(m)
    # Re-seed after constructing the arm, so adapter construction cannot change masks.
    seed(cfg['seed'],device);opt=optimizer(m,cfg);best=float('inf');selected=-1;first=None
    trainable=m.trainable_names();frozen={k:v for k,v in initial.items()if k.startswith('native.')and not k.startswith('native.decoder.')}
    def validate(step):
        nonlocal best,selected,first
        modes=[t.training for t in m.modules()];cpu=torch.get_rng_state().clone();cu=torch.cuda.get_rng_state(device).clone()if device.type=='cuda'else None
        v=selection(m,val,device,cfg,record)
        assert modes==[t.training for t in m.modules()]and torch.equal(cpu,torch.get_rng_state())
        if cu is not None:assert torch.equal(cu,torch.cuda.get_rng_state(device))
        if step==0:first=v
        with(out/'SELECT_TRACE.jsonl').open('a')as f:f.write(json.dumps(dict(update=step,SELECT_loss=v,mode_RNG_restored=True))+'\n')
        if v<best:
            best,selected=v,step
            torch.save(dict(state=m.state_dict(),record=record,arm=arm,update=step,config=cfg),out/'selected.tmp');os.replace(out/'selected.tmp',out/'selected.pth')
    m.train();validate(0);begin=time.monotonic()
    for step in range(1,cfg['updates']+1):
        m.train();lr=cfg['learning_rate']*min(step/cfg['warmup_steps'],1.)
        for group in opt.param_groups:group['lr']=lr
        x=tensor_windows(fit,plan[step-1],device);opt.zero_grad(set_to_none=True)
        mask_rng=hashlib.sha256((torch.cuda.get_rng_state(device)if device.type=='cuda'else torch.get_rng_state()).cpu().numpy().tobytes()).hexdigest()
        repr_,copies=m(x,norm=spec['norm'],copies=cfg['copies']);loss=mse(copies,x)
        assert torch.isfinite(loss);loss.backward()
        grads=[p.grad for p in m.parameters()if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all()for g in grads)
        assert all(p.grad is None for p in m.native.parameters()if not p.requires_grad)
        assert not m.native.training and not m.native.encoder.training
        norm=torch.linalg.vector_norm(torch.stack([g.detach().float().norm()for g in grads]))
        assert torch.isfinite(norm);opt.step()
        with(out/'TRAIN_TRACE.jsonl').open('a')as f:f.write(json.dumps(dict(update=step,loss=float(loss.detach()),lr=lr,gradient_norm=float(norm),mask_RNG_before_sha256=mask_rng,seconds=time.monotonic()-begin))+'\n')
        if step%cfg['select_every']==0:validate(step)
    end=state_hashes(m);assert all(end[k]==v for k,v in frozen.items())
    torch.save(dict(state=m.state_dict(),update=cfg['updates'],arm=arm,record=record),out/'final.pth')
    dump(out/'TRAIN_RECEIPT.json',dict(status='COMPLETED',record=record,arm=arm,updates=cfg['updates'],selected_update=selected,SELECT_loss=best,step0_SELECT_loss=first,initial_state=initial,frozen_state_unchanged=True,trainable_names=trainable,trainable_parameters=sum(p.numel()for p in m.parameters()if p.requires_grad),FIT_plan_sha256=sha(out/'FIT_PLAN.npz'),selected_sha256=sha(out/'selected.pth'),roles=data.access,no_labels_decoded=True,no_CAL_QUERY_values=True,original_source_sha256=sha(resolve(w,cfg['author_source']))))
    m.close()


def score(w,config,cfg,out,record,arm,device):
    if arm=='zero':
        seed(cfg['seed'],device);m=model(w,cfg,'decoder_head',device)
        checkpoint_sha=sha(resolve(w,cfg['checkpoint']));selected=0
    else:
        folder=w/cfg['output']/record/arm/'train';receipt=read(folder/'TRAIN_RECEIPT.json');assert sha(folder/'selected.pth')==receipt['selected_sha256']
        state=torch.load(folder/'selected.pth',map_location='cpu',weights_only=True);assert state['config']==cfg and state['arm']==arm and state['record']==record
        m=model(w,cfg,arm,device);m.load_state_dict(state['state'],strict=True);checkpoint_sha=receipt['selected_sha256'];selected=state['update']
    m.eval();before=state_hashes(m)
    native_before=state_hashes(m.native)
    identity_adapter=m.adapter is None or (torch.count_nonzero(m.adapter.up.weight)==0 and torch.count_nonzero(m.adapter.up.bias)==0)
    zero_folder=w/cfg['output']/record/'zero/score'
    if arm!='zero' and identity_adapter and (zero_folder/'SCORE_RECEIPT.json').exists():
        zero=read(zero_folder/'SCORE_RECEIPT.json')
        if zero['config_sha256']==sha(config) and zero['native_state_hashes']==native_before:
            reused=[]
            for row in zero['banks']:
                source=zero_folder/Path(row['path']).name;assert sha(source)==row['sha256']
                target=out/source.name;shutil.copyfile(source,target)
                reused.append(dict(row,path=str(target),sha256=sha(target)))
            for row in zero['roles']:assert sha(Path(row['path']))==row['sha256']
            dump(out/'SCORE_RECEIPT.json',dict(zero,arm=arm,selected_update=selected,checkpoint_sha256=checkpoint_sha,
                 banks=reused,reused_zero=True,reused_receipt_sha256=sha(zero_folder/'SCORE_RECEIPT.json'),
                 reuse_reason='Native tensors exactly equal and adapter correction identically zero; same config/data/mask stream',
                 roles_access='Reused zero role hashes, no second model/data forward'))
            m.close();return
    seed(cfg['score_seed'],device);data=RoleReader(w,cfg,record,['CAL','QUERY']);spec=cfg['records'][record];banks=[]
    with torch.no_grad():
        for role in ['CAL','QUERY']:
            values,index=data.load(role);scores=[];positions=[];available=[];window_end=[];rng_stream=[];channel_variance=[];reconstruction_mean=[]
            for starts in native_batches(values,spec['batch_size']):
                rng_stream.append(hashlib.sha256((torch.cuda.get_rng_state(device)if device.type=='cuda'else torch.get_rng_state()).cpu().numpy().tobytes()).hexdigest())
                x=tensor_windows(values,starts,device);_,outputs=m(x,norm=spec['norm'],copies=cfg['copies'])
                s=m.cal_anomaly_score(x,outputs,nn.MSELoss(),L=spec['L']);assert s.shape==x.shape[:2]and torch.isfinite(s).all()
                scores.append(s.cpu().numpy().reshape(-1));positions.extend(index[t:t+100]for t in starts)
                channel_variance.append(outputs.var(0).cpu().numpy().reshape(-1,x.shape[-1]))
                reconstruction_mean.append(outputs.mean(0).cpu().numpy().reshape(-1,x.shape[-1]))
                available.append(np.full(len(starts)*100,index[starts[-1]+99],dtype=np.int64))
                window_end.append(np.repeat(index[starts+99],100))
            idx=np.concatenate(positions);score_=np.concatenate(scores);assert len(idx)==len(values)//100*100
            p=out/(role+'.npz');np.savez_compressed(p,index=idx,score=score_,available_index=np.concatenate(available),max_input_index=np.concatenate(available),window_end_index=np.concatenate(window_end),channel_variance=np.concatenate(channel_variance),reconstruction_mean=np.concatenate(reconstruction_mean))
            banks.append(dict(record=record,role=role,path=str(p),sha256=sha(p),points=len(idx),unscored_tail=len(values)%100,mask_RNG_stream_sha256=hashlib.sha256(''.join(rng_stream).encode()).hexdigest(),batch_size=spec['batch_size'],score_seed=cfg['score_seed']))
    assert state_hashes(m)==before
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',config_sha256=sha(config),native_state_hashes=native_before,reused_zero=False,arm=arm,record=record,selected_update=selected,checkpoint_sha256=checkpoint_sha,banks=banks,roles=data.access,native_score='original cal_anomaly_score, original mean MSE argument, L1 variance only',mode='offline complete100 window; no strict online alarm claim',immutable=True,no_labels_decoded=True,no_threshold_fitting=True));m.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['parity','train','score']);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--record',required=True);p.add_argument('--arm',choices=['zero','decoder_head','self','cross']);p.add_argument('--device',default='cpu')
    a=p.parse_args();w=a.workspace.resolve();config=a.config.resolve();cfg=read(config);torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device(a.device)
    folder='parity'if a.phase=='parity'else a.arm+'/'+a.phase;out=w/cfg['output']/a.record/folder;out.mkdir(parents=True,exist_ok=False);start=time.time();rc=0
    if device.type=='cuda':
        torch.cuda.set_device(device);cap=20 if a.phase=='train'else 30;torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(device).total_memory,device);torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dump(out/'START.json',dict(pid=os.getpid(),phase=a.phase,record=a.record,arm=a.arm,start_unix=start,config_sha256=sha(config),runner_sha256=sha(__file__),adapter_sha256=sha(Path(__file__).with_name('timeradar_cross_variable.py')),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        if a.phase=='parity':parity(w,config,cfg,out,a.record,device)
        elif a.phase=='train':train(w,config,cfg,out,a.record,a.arm,device)
        else:score(w,config,cfg,out,a.record,a.arm,device)
    except BaseException:
        rc=1;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:
        end=time.time();dump(out/'OS.json',dict(pid=os.getpid(),start_unix=start,end_unix=end,seconds=end-start,exit_code=rc,GPU_hours=(end-start)/3600 if device.type=='cuda'else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=='cuda'else 0))


if __name__=='__main__':main()
