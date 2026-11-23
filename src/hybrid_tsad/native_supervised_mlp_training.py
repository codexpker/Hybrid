"""Two fixed learning-rate trajectories for the standard supervised MLP readout."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from . import native_event_rank_training as source
from .native_supervised_mlp import OUT, SupervisedMLP, encode
from .native_v_conditioning import load_native, fixed_native_rng, state_hashes, sha, SOURCE_SHA, WEIGHT_SHA
from .native_v_conditioning_runner import setup, seed, read, dump
from .native_v_replay import optimizer

PLAN_SHA='df7e74047bb0fd49d25a2da0bb58e54a2fbe3327fd9f61b6e07ceca6422543e9'
PRIOR=Path('results/native_event_rank_psm_20261122_v1')
LRS=(1e-4,1e-3)

def main(w):
    started=time.monotonic();root=w/OUT;root.mkdir(parents=True,exist_ok=True)
    assert not (root/'ALL_SELECTION.json').exists()
    device=setup(30);seed(2025)
    assert sha(w/PRIOR/'PLAN.npz')==PLAN_SHA
    plan=source.load_plan(w/PRIOR)
    folder,task,fit,select,records=source.source_roles(w)
    train=(source.event_input(w,folder,task,fit),source.nominal_swat_input(w,folder,task,'FIT'),[source.load_smd(w,records[e],'FIT') for e in source.SMD_ENTITIES])
    valid=(source.event_input(w,folder,task,select),source.nominal_swat_input(w,folder,task,'SELECT'),[source.load_smd(w,records[e],'SELECT') for e in source.SMD_ENTITIES])
    native=load_native(w,device);identity=state_hashes(native)
    x=next(source.windows(train[1],plan['swat_fit'][0,:1],device))
    with torch.no_grad(),fixed_native_rng(device):h,_=native(x,norm=0,mask_mode='c',copies=10)
    extracted=encode(native,x)
    torch.testing.assert_close(extracted.reshape_as(h),h,rtol=0,atol=0)
    cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state(device).clone()
    encode(native,x)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    heads=[SupervisedMLP().to(device)];heads.append(deepcopy(heads[0]));opts=[optimizer(m) for m in heads]
    assert all(torch.equal(v,heads[1].state_dict()[k]) for k,v in heads[0].state_dict().items())
    initial_hash={k:sha_tensor(v) for k,v in heads[0].state_dict().items()}
    checks=dict(native_H_exact=True,RNG_isolated=True,official_parameters_frozen=True,source_plan_sha256=PLAN_SHA,head_parameters=sum(p.numel() for p in heads[0].parameters()),head_initial_hashes=initial_hash,encoder_source_sha256=SOURCE_SHA,official_weight_sha256=WEIGHT_SHA,labels='public SOURCE event labels only; no target reads',feature_cache='SELECT GPU float32 complex64; FIT streaming shared across both trajectories')
    dump(root/'CHECKS.json',checks)
    print('Extracting fixed SOURCE SELECT features',flush=True)
    def features(data,ends):return [encode(native,z) for z in source.windows(data,ends,device)]
    select_h={'positive':features(valid[0],plan['positive_select']), 'normal':features(valid[0],plan['normal_select']), 'nominal':features(valid[1],plan['swat_select'])}
    for role,ends in zip(valid[2],plan['smd_select']):select_h['nominal'].extend(features(role,ends))
    cache_seconds=time.monotonic()-started
    split=np.cumsum(plan['select_stratum_sizes'])[:-1]
    rows=[];curves=[];best=[None,None];aligned=[None,None]
    for step in range(2401):
        if step:
            row=(step-1)%240
            for lr,opt in zip(LRS,opts):
                opt.zero_grad(set_to_none=True)
                for group in opt.param_groups:group['lr']=lr*min(step/24,1)
            totals=np.zeros((2,3)); weights=plan['stratum_sizes'].reshape(-1).astype(float);weights/=weights.sum()
            groups=[]
            groups.extend((train[1],int(e),0.25/32,False,0) for e in plan['swat_fit'][row])
            groups.extend((train[2][i],int(e),0.25/32,False,0) for i,e in zip(plan['smd_entity'][row],plan['smd_fit'][row]))
            groups.extend((train[0],int(e),0.5*weights[j],True,1) for j,e in enumerate(plan['positive_fit'][row]))
            groups.extend((train[0],int(e),0.25*weights[j],False,2) for j,e in enumerate(plan['normal_fit'][row]))
            for data,end,weight,positive,kind in groups:
                window=next(source.windows(data,[end],device));rep=encode(native,window)
                for k,head in enumerate(heads):
                    logit=head(rep);loss=F.softplus(-logit if positive else logit)*weight
                    loss.backward();totals[k,kind]+=float(loss.detach())
            for k,opt in enumerate(opts):
                gradients=[p.grad for p in heads[k].parameters() if p.grad is not None]
                gn=float(torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients])))
                assert np.isfinite(gn) and gn>0 and all(torch.isfinite(g).all() for g in gradients)
                opt.step();curves.append(dict(step=step,lr=LRS[k],loss=float(totals[k].sum()),nominal=float(totals[k,0]),positive=float(totals[k,1]),pair_normal=float(totals[k,2]),gradient_norm=gn))
        if step%40==0:
            with torch.no_grad():
                for k,head in enumerate(heads):
                    scores={name:torch.stack([head(rep) for rep in reps]).double().cpu().numpy() for name,reps in select_h.items()}
                    p=np.logaddexp(0,-scores['positive']);n=np.logaddexp(0,scores['normal']);nom=np.logaddexp(0,scores['nominal'])
                    pos=float(np.mean([a.mean() for a in np.split(p,split)]));neg=float(np.mean([a.mean() for a in np.split(n,split)]));nominal=float(nom.mean())
                    risk=.5*pos+.25*neg+.25*nominal
                    record=dict(step=step,lr=LRS[k],risk=risk,positive=pos,pair_normal=neg,nominal=nominal,seconds=time.monotonic()-started)
                    rows.append(record)
                    if best[k] is None or risk<best[k]['risk']:
                        best[k]=record.copy();save_head(root,f'best_{k}.pt',head,record)
                    if step<=240 and (aligned[k] is None or risk<aligned[k]['risk']):
                        aligned[k]=record.copy();save_head(root,f'aligned_{k}.pt',head,record)
                    print('SOURCE_SELECT',step,LRS[k],risk,flush=True)
            dump(root/'SELECT_CURVE.json',rows);dump(root/'FIT_CURVE.json',curves)
    assert identity==state_hashes(native) and all(not m.training for m in native.modules()) and all(p.grad is None for p in native.parameters())
    selected={}
    for kind,records_ in [('formal',best),('aligned240',aligned)]:
        k=min(range(2),key=lambda i:(records_[i]['risk'],records_[i]['step'],LRS[i]))
        filename=f'best_{k}.pt' if kind=='formal' else f'aligned_{k}.pt'
        selected[kind]=dict(relative=filename,sha256=sha(root/filename),**records_[k])
    unique=[]
    for eventname,key,ends in [('SWAT_normal','swat_fit',plan['swat_fit']),('SWAT_positive','positive_fit',plan['positive_fit']),('SWAT_pair_normal','normal_fit',plan['normal_fit'])]:unique.append(dict(role=eventname,unique_windows=len(np.unique(ends)),exposures_per_track=ends.size*10))
    smd_ids=set(zip(plan['smd_entity'].ravel().tolist(),plan['smd_fit'].ravel().tolist()));unique.append(dict(role='SMD_normal',unique_windows=len(smd_ids),exposures_per_track=38400))
    all_swat_normal=set(plan['swat_fit'].ravel().tolist());all_pair_normal=set(plan['normal_fit'].ravel().tolist());assert all_swat_normal.isdisjoint(all_pair_normal)
    dump(root/'ALL_SELECTION.json',dict(status='COMPLETED_BEFORE_ANY_QUERY',**selected,steps_per_track=2400,learning_rates=LRS,seed=2025,shared_initialization=initial_hash,source_plan_sha256=PLAN_SHA,parameter_count=checks['head_parameters'],head_only=True,native_frozen_verified=True,source_window_exposure=unique,independent_FIT_events=5,independent_SELECT_events=4,actual_GPU_seconds=time.monotonic()-started,SELECT_cache_seconds=cache_seconds,peak_reserved_bytes=torch.cuda.max_memory_reserved(device),score='mean copy/channel raw MLP logit'))


def sha_tensor(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()

def save_head(root,name,head,record):
    torch.save(dict(head={k:v.detach().cpu().clone() for k,v in head.state_dict().items()},step=record['step'],lr=record['lr']),root/name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('workspace',type=Path);main(p.parse_args().workspace)
