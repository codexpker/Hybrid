"""One matched persistent-response experiment; QUERY requires root normal gate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from .g36_persistent_model import PersistentResponse
from .g36_runner import Data,batch,dump,hashes,read,seed,sha
from .normality_density_runner import validation_mode


def make(cfg,method,device):
    return PersistentResponse(method,cfg['hidden'],cfg['state_dim']).to(device)


def prepare(w,config,cfg):
    out=w/cfg['output']/'shared';out.mkdir(parents=True,exist_ok=False)
    data=Data(w,cfg,['FIT']);fit,_=data.one('FIT')
    plan=w/cfg['existing_fit_plan']
    with np.load(plan)as z:order=z['end']
    assert np.array_equal(order,np.random.default_rng(2025).integers(51,len(fit),size=(3000,64),dtype=np.int64))
    shutil.copyfile(plan,out/'FIT_PLAN.npz');identities={}
    for method in cfg['methods']:
        seed(cfg['seed'],torch.device('cpu'));model=make(cfg,method,torch.device('cpu'))
        torch.save(dict(state=model.state_dict()),out/(method+'_INITIAL.pth'))
        identities[method]=dict(sha256=sha(out/(method+'_INITIAL.pth')),state=hashes(model),parameters=sum(p.numel()for p in model.parameters()))
    assert all(identities['full']['state'][k]==v for k,v in identities['linear']['state'].items())
    dump(out/'MANIFEST.json',dict(config_sha256=sha(config),data_manifest_sha256=sha(data.root/'manifest.json'),scaler_sha256=sha(data.root/'scaler.npz'),plan_sha256=sha(plan),initial=identities,FIT_access=data.access))


def initial(w,config,cfg,method,device):
    shared=w/cfg['output']/'shared';meta=read(shared/'MANIFEST.json')
    assert sha(config)==meta['config_sha256'] and sha(w/cfg['data']/'manifest.json')==meta['data_manifest_sha256']
    assert sha(w/cfg['data']/'scaler.npz')==meta['scaler_sha256']
    assert sha(shared/(method+'_INITIAL.pth'))==meta['initial'][method]['sha256']
    seed(cfg['seed'],device);model=make(cfg,method,device)
    state=torch.load(shared/(method+'_INITIAL.pth'),map_location='cpu',weights_only=True)
    model.load_state_dict(state['state']);assert hashes(model)==meta['initial'][method]['state']
    return model,meta


def select(model,values,device,cfg):
    total=0.;count=0
    with validation_mode(model,device):
        for begin in range(51,len(values),64):
            ends=np.arange(begin,min(begin+64,len(values)))
            v=model.loss_per_sample(batch(values,ends,device));assert torch.isfinite(v).all()
            total+=float(v.sum());count+=len(v)
    return total/count


def train(w,config,cfg,out,method,device,l0=False):
    model,meta=initial(w,config,cfg,method,device);data=Data(w,cfg,['FIT','SELECT'])
    fit,_=data.one('FIT');val,_=data.one('SELECT')
    plan=w/cfg['output']/'shared/FIT_PLAN.npz';assert sha(plan)==meta['plan_sha256']
    with np.load(plan)as z:order=z['end'].copy()
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    best=float('inf');selected=-1;first=None
    def validation(step):
        nonlocal best,selected,first
        modes=[m.training for m in model.modules()];cpu=torch.get_rng_state().clone();cu=torch.cuda.get_rng_state(device).clone()if device.type=='cuda'else None
        v=select(model,val,device,cfg)
        assert torch.equal(cpu,torch.get_rng_state())and modes==[m.training for m in model.modules()]
        if cu is not None:assert torch.equal(cu,torch.cuda.get_rng_state(device))
        if step==0:first=v
        with(out/'SELECT_TRACE.jsonl').open('a')as f:f.write(json.dumps(dict(update=step,SELECT_loss=v,modes_rng_restored=True))+'\n')
        if v<best:
            best,selected=v,step
            if not l0:
                torch.save(dict(state=model.state_dict(),method=method,update=step,config=cfg),out/'selected.tmp');os.replace(out/'selected.tmp',out/'selected.pth')
    validation(0);started=time.monotonic();N=2 if l0 else cfg['updates']
    for step in range(1,N+1):
        model.train();opt.zero_grad(set_to_none=True)
        v=model.loss(batch(fit,order[step-1],device));assert torch.isfinite(v)
        v.backward();grads=[p.grad for p in model.parameters()if p.grad is not None]
        assert grads and all(torch.isfinite(g).all()for g in grads)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm']);assert torch.isfinite(norm)
        opt.step()
        with(out/'TRAIN_TRACE.jsonl').open('a')as f:f.write(json.dumps(dict(update=step,loss=float(v.detach()),gradient_norm=float(norm),seconds=time.monotonic()-started))+'\n')
        if not l0 and step%cfg['select_every']==0:validation(step)
    if l0:
        validation(N);assert any(v!=meta['initial'][method]['state'][k]for k,v in hashes(model).items())
        with validation_mode(model,device):
            prior=model.fit_prior(torch.from_numpy(fit).to(device));r=model.components(batch(val,np.arange(51,55),device),prior)
            assert all(torch.isfinite(v).all()for v in r.values())
    else:torch.save(dict(state=model.state_dict(),update=N),out/'final.pth')
    dump(out/'TRAIN_RECEIPT.json',dict(status='L0_PASS_DISCARDED'if l0 else'COMPLETED',method=method,updates=N,selected_update=selected,SELECT_loss=best,step0_SELECT_loss=first,selected_improves_step0=selected>0 and best<first,initial_state=meta['initial'][method]['state'],plan_sha256=meta['plan_sha256'],roles=data.access,no_CAL_QUERY_values=True,no_labels_decoded=True,selected_sha256=None if l0 else sha(out/'selected.pth')))


def assess(w,config,cfg,out,method,device):
    trainpath=out.parent/'train';receipt=read(trainpath/'TRAIN_RECEIPT.json')
    assert sha(trainpath/'selected.pth')==receipt['selected_sha256']
    model,_=initial(w,config,cfg,method,device)
    state=torch.load(trainpath/'selected.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state['state']);model.eval();data=Data(w,cfg,['FIT'])
    fit,_=data.one('FIT')
    with torch.no_grad():
        prior=model.fit_prior(torch.from_numpy(fit).to(device));cov0=model.joint_covariance();cov1=model.joint_covariance(prior)
        delta=cov1-cov0
        kl=.5*(torch.trace(torch.linalg.solve(cov1,cov0))-len(cov0)+torch.linalg.slogdet(cov1)[1]-torch.linalg.slogdet(cov0)[1])
        e0=torch.linalg.eigvalsh(cov0);e1=torch.linalg.eigvalsh(cov1);ed=torch.linalg.eigvalsh(delta)
    np.savez(out/'FIT_PRIOR.npz',diagonal=prior.cpu().numpy())
    np.savez(out/'HYPOTHESIS_COVARIANCE.npz',Sigma0=cov0.cpu().numpy(),Sigma1=cov1.cpu().numpy())
    dump(out/'ASSESS_RECEIPT.json',dict(status='NORMAL_ASSESSMENT_COMPLETED',method=method,selected_update=state['update'],train_receipt_sha256=sha(trainpath/'TRAIN_RECEIPT.json'),checkpoint_sha256=sha(trainpath/'selected.pth'),prior_sha256=sha(out/'FIT_PRIOR.npz'),prior_diagonal=prior.tolist(),covariance_sha256=sha(out/'HYPOTHESIS_COVARIANCE.npz'),Sigma0_eigen_min=float(e0.min()),Sigma1_eigen_min=float(e1.min()),delta_eigen_min=float(ed.min()),delta_trace=float(torch.trace(delta)),delta_frobenius=float(delta.norm()),KL_H0_H1=float(kl),statistically_distinct=bool(kl>0 and delta.norm()>0),roles=data.access,no_CAL_QUERY_values=True,limits='Different normalized hypotheses only; not physical fault identifiability or efficacy.'))


def score(w,config,cfg,out,method,device,gate):
    release=read(gate);assert release['status']=='NORMAL_GATE_PASS_QUERY_RELEASED' and release['config_sha256']==sha(config)
    full_receipt=w/cfg['output']/'full/train/TRAIN_RECEIPT.json'
    assert release['full_train_receipt_sha256']==sha(full_receipt)
    assert read(full_receipt)['selected_improves_step0']
    trainpath=out.parent/'train';receipt=read(trainpath/'TRAIN_RECEIPT.json');assert sha(trainpath/'selected.pth')==receipt['selected_sha256']
    model,meta=initial(w,config,cfg,method,device);state=torch.load(trainpath/'selected.pth',map_location='cpu',weights_only=True)
    assert state['method']==method and state['config']==cfg;model.load_state_dict(state['state']);model.eval();before=hashes(model)
    assessment=out.parent/'assess';ar=read(assessment/'ASSESS_RECEIPT.json')
    assert ar['checkpoint_sha256']==sha(trainpath/'selected.pth')
    assert release['prior_sha256'][method]==ar['prior_sha256']==sha(assessment/'FIT_PRIOR.npz')
    with np.load(assessment/'FIT_PRIOR.npz')as z:prior=torch.from_numpy(z['diagonal'].copy()).to(device)
    shutil.copyfile(assessment/'FIT_PRIOR.npz',out/'FIT_PRIOR.npz')
    data=Data(w,cfg,['SELECT','CAL','QUERY']);banks=[]
    for record in data.manifest['records']:
        if record['role']not in ['SELECT','CAL','QUERY']:continue
        values,index=data.load(record);ends=np.arange(51,len(values));parts={}
        with torch.no_grad():
            for begin in range(0,len(ends),64):
                e=ends[begin:begin+64];r=model.components(batch(values,e,device),prior)
                assert all(torch.isfinite(v).all()for v in r.values())
                for k,v in r.items():parts.setdefault(k,[]).append(v.cpu().numpy())
        fields={k:np.concatenate(v)for k,v in parts.items()};ix=index[ends]
        assert ix[0]==record['first_scored_index']and len(ix)==record['n_scored']
        file=out/(record['run']+'__'+record['role']+'.npz')
        np.savez_compressed(file,index=ix,available_index=ix,max_input_index=ix,max_condition_index=ix-4,min_input_index=ix-51,**fields)
        banks.append(dict(run=record['run'],role=record['role'],path=str(file),sha256=sha(file),points=len(ix)))
    assert before==hashes(model)
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',method=method,banks=banks,roles=data.access,FIT_prior_access=ar['roles'],prior_sha256=sha(out/'FIT_PRIOR.npz'),selected_update=state['update'],PT_sha256=sha(trainpath/'selected.pth'),immutable=True,no_labels_decoded=True,no_threshold_fitting=True,gate_sha256=sha(gate)))



def frozen_linear_inputs(w,config,cfg,release_path,method):
    """Exact new readout authorization; never impersonates the old full gate."""
    release=read(release_path)
    if release.get('status')!='LINEAR_FROZEN_READOUT_RELEASED' or method!='linear' or release.get('method')!='linear':
        raise PermissionError('Frozen linear-only readout authorization required')
    root=w/cfg['output'];trainpath=root/'linear/train';assessment=root/'linear/assess'
    required=dict(config_sha256=config,train_receipt_sha256=trainpath/'TRAIN_RECEIPT.json',
                  checkpoint_sha256=trainpath/'selected.pth',assess_receipt_sha256=assessment/'ASSESS_RECEIPT.json',
                  prior_sha256=assessment/'FIT_PRIOR.npz',stop_gate_sha256=root/'NORMAL_GATE.json',
                  data_manifest_sha256=w/cfg['data']/'manifest.json',scaler_sha256=w/cfg['data']/'scaler.npz',
                  runner_sha256=Path(__file__),model_sha256=Path(__file__).with_name('g36_persistent_model.py'))
    for key,path in required.items():
        if release.get(key)!=sha(path):raise PermissionError('Frozen identity mismatch: '+key)
    tr=read(trainpath/'TRAIN_RECEIPT.json');ar=read(assessment/'ASSESS_RECEIPT.json')
    assert tr['method']=='linear' and tr['selected_update']==1200 and tr['selected_sha256']==release['checkpoint_sha256']
    assert ar['method']=='linear' and ar['selected_update']==1200 and ar['checkpoint_sha256']==release['checkpoint_sha256'] and ar['prior_sha256']==release['prior_sha256']
    assert read(root/'NORMAL_GATE.json')['status']=='NORMAL_GATE_FAILED_STOP'
    out=(w/release['output']).resolve()
    expected=(w/'results/g36_linear_frozen_readout_20261116_v1/linear/score').resolve()
    if out!=expected:raise PermissionError('Only the fixed new output directory is allowed')
    return release,trainpath,assessment,out


def frozen_linear_score(w,config,cfg,out,device,release_path):
    release,trainpath,assessment,expected=frozen_linear_inputs(w,config,cfg,release_path,'linear')
    assert out==expected
    tracked=[trainpath/'selected.pth',assessment/'FIT_PRIOR.npz',trainpath/'TRAIN_RECEIPT.json',assessment/'ASSESS_RECEIPT.json',w/cfg['output']/'NORMAL_GATE.json']
    originals={str(p):sha(p)for p in tracked}
    state=torch.load(trainpath/'selected.pth',map_location='cpu',weights_only=True)
    assert state['method']=='linear' and state['update']==1200 and state['config']==cfg
    model=make(cfg,'linear',device);model.load_state_dict(state['state'],strict=True);model.eval();before=hashes(model)
    with np.load(assessment/'FIT_PRIOR.npz',allow_pickle=False)as z:prior=torch.from_numpy(z['diagonal'].copy()).to(device)
    assert prior.shape==(16,) and torch.isfinite(prior).all()and(prior>0).all()
    data=Data(w,cfg,['SELECT','CAL','QUERY']);banks=[]
    with torch.no_grad():
        for record in data.manifest['records']:
            if record['role']not in ['SELECT','CAL','QUERY']:continue
            values,index=data.load(record);ends=np.arange(51,len(values));parts={}
            for begin in range(0,len(ends),64):
                e=ends[begin:begin+64];r=model.components(batch(values,e,device),prior)
                assert all(torch.isfinite(v).all()for v in r.values())
                for key,value in r.items():parts.setdefault(key,[]).append(value.cpu().numpy())
            fields={k:np.concatenate(v)for k,v in parts.items()};ix=index[ends]
            assert ix[0]==record['first_scored_index']and len(ix)==record['n_scored']
            file=out/(record['run']+'__'+record['role']+'.npz')
            np.savez_compressed(file,index=ix,available_index=ix,max_input_index=ix,max_condition_index=ix-4,min_input_index=ix-51,**fields)
            banks.append(dict(run=record['run'],role=record['role'],path=str(file),sha256=sha(file),points=len(ix)))
    assert len(banks)==6 and before==hashes(model)
    assert all(sha(Path(p))==value for p,value in originals.items())
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',method='linear',banks=banks,roles=data.access,
         selected_update=1200,PT_sha256=release['checkpoint_sha256'],prior_sha256=release['prior_sha256'],
         release_sha256=sha(release_path),immutable=True,no_labels_decoded=True,no_threshold_fitting=True,
         no_FIT_values_access=True,no_prior_refit=True,no_training=True,old_gate_unchanged=True,
         original_sha256=originals,readout='same frozen model: score=integrated LR, normal_nll retained for paired ablation'))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','l0','train','assess','score','linear_readout']);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--method',choices=['full','linear']);p.add_argument('--device',default='cpu');p.add_argument('--gate',type=Path)
    a=p.parse_args();w=a.workspace.resolve();config=a.config.resolve();cfg=read(config);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    if a.phase=='prepare':return prepare(w,config,cfg)
    if a.phase=='linear_readout':
        if a.gate is None:raise PermissionError('Frozen linear readout authorization required')
        _,_,_,out=frozen_linear_inputs(w,config,cfg,a.gate,a.method)
    else:out=w/cfg['output']/a.method/a.phase
    out.mkdir(parents=True,exist_ok=False);device=torch.device(a.device);start=time.time();rc=0
    if device.type=='cuda':
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction((30 if a.phase in ['score','linear_readout']else 20)*1024**3/torch.cuda.get_device_properties(device).total_memory,device);torch.cuda.reset_peak_memory_stats(device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dump(out/'START.json',dict(pid=os.getpid(),method=a.method,phase=a.phase,start_unix=start,config_sha256=sha(config),runner_sha256=sha(__file__),model_sha256=sha(Path(__file__).with_name('g36_persistent_model.py')),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        if a.phase=='linear_readout':frozen_linear_score(w,config,cfg,out,device,a.gate)
        elif a.phase=='score':
            if a.gate is None:raise PermissionError('Root normal gate required')
            score(w,config,cfg,out,a.method,device,a.gate)
        elif a.phase=='assess':assess(w,config,cfg,out,a.method,device)
        else:train(w,config,cfg,out,a.method,device,a.phase=='l0')
    except BaseException:
        rc=1;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:
        end=time.time();dump(out/'OS.json',dict(pid=os.getpid(),start_unix=start,end_unix=end,seconds=end-start,exit_code=rc,GPU_hours=(end-start)/3600 if device.type=='cuda'else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=='cuda'else 0))


if __name__=='__main__':main()
