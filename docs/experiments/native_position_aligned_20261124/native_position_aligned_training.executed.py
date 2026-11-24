"""Matched source training with a common full-position validation criterion."""
import argparse
from pathlib import Path
import shutil
import time
import os
import numpy as np
import torch
from torch.nn import functional as F
from . import native_event_rank_training as old
from .native_event_rank import native_endpoint_and_copy_loss, event_pair_loss
from .native_v_conditioning import fixed_native_rng, state_hashes, sha, WEIGHT_SHA
from .native_v_conditioning_runner import read, dump, setup, seed
from .native_v_replay import decoder_model, optimizer

OUT=Path('results/native_position_aligned_20261124_v1')
PRIOR=Path('results/native_event_rank_psm_20261122_v1')
ARMS=('copy','terminal','all_positions')
PLAN_SHA='df7e74047bb0fd49d25a2da0bb58e54a2fbe3327fd9f61b6e07ceca6422543e9'


def values(model,x):
    with fixed_native_rng(x.device):
        _,copies=model(x,norm=0,mask_mode='c',copies=10)
    assert copies.shape==(10,*x.shape)
    variance=model.cal_anomaly_score(x,copies,None,1)[0]
    return variance,F.mse_loss(copies,x.unsqueeze(0).expand_as(copies))


def position_pair(positive,normal,scale,weights=None):
    assert positive.shape==normal.shape and positive.shape[-1]==100
    losses=F.softplus(1-torch.log1p(positive/scale)+torch.log1p(normal/scale)).mean(-1)
    if weights is None:return losses.mean()
    mass=torch.as_tensor(weights,device=losses.device,dtype=losses.dtype)
    assert mass.shape==losses.shape and (mass>0).all()
    return (mass*losses).sum()/mass.sum()


def prepare(w):
    out=w/OUT;out.mkdir(parents=True,exist_ok=False)
    assert sha(w/PRIOR/'PLAN.npz')==PLAN_SHA
    for filename in ['PLAN.npz','PLAN.json','SCALE.json','FIT_NORMAL_V.npz']:
        shutil.copyfile(w/PRIOR/filename,out/filename)
    scan={}
    for arm in ('copy','point_weighted'):
        folder=w/PRIOR/'train'/arm
        files=sorted(str(p.relative_to(w)) for p in folder.rglob('*.pt'))
        scan[arm]=dict(checkpoints=files,complete_seven_checkpoint_trajectory=False,
                       action='rebuild fixed seeded trajectory; preserve historical files')
        assert files==[str((folder/'selected.pt').relative_to(w))]
    dump(out/'CONFIG.json',dict(seed=2025,arms=ARMS,updates=240,warmup=24,source_plan_sha256=PLAN_SHA,
        source_scale=read(out/'SCALE.json')['value'],source_event_fit=5,source_event_select=4,
        normal_windows_per_update=32,positive_pair_windows_per_update=15,normal_pair_windows_per_update=15,
        training='copy: native copy MSE; terminal: copy+endpoint99 pair; all_positions: copy+mean100 per-position pair',
        training_pair_weights='original positive-window sampling-stratum counts, normalized; overlapping positions are not uniform independent timepoints',
        selection='all arms: mean per-position pair risk, then equal original source event/stratum means; earliest minimum steps0,40,...240',
        optimizer='fresh original decoder AdamW lr1e-4 betas(.9,.95) eps1e-8 decay.1 except bias/LayerNorm; warmup24',
        checkpoints='all seven validation checkpoints and position-specific SOURCE SELECT arrays saved',
        legacy_trajectory_scan=scan,target_read=False))


def datasets(w):
    folder,task,fit,select,records=old.source_roles(w)
    def data(role):
        ev=fit if role=='FIT' else select
        return (old.event_input(w,folder,task,ev),old.nominal_swat_input(w,folder,task,role),
                [old.load_smd(w,records[e],role) for e in old.SMD_ENTITIES])
    return data('FIT'),data('SELECT')


def check(w):
    device=setup();seed(2025);root=w/OUT;plan=old.load_plan(root)
    fit,_=datasets(w);model=decoder_model(w,device);initial=state_hashes(model)
    x=next(old.windows(fit[1],plan['swat_fit'][0,:1],device))
    v,copy=values(model,x);legacy,legacy_copy=native_endpoint_and_copy_loss(model,x)
    torch.testing.assert_close(v[99],legacy,rtol=0,atol=0)
    torch.testing.assert_close(copy,legacy_copy,rtol=0,atol=0)
    parameters=[p for p in model.parameters() if p.requires_grad]
    scale=read(root/'SCALE.json')['value']
    # Same native score and derivative for the terminal-control path.
    a=F.softplus(1-torch.log1p(v[99]/scale)) + copy
    b=F.softplus(1-torch.log1p(legacy/scale)) + legacy_copy
    ga=torch.autograd.grad(a,parameters);gb=torch.autograd.grad(b,parameters)
    for left,right in zip(ga,gb):torch.testing.assert_close(left,right,rtol=0,atol=0)
    pp=torch.linspace(0,9,1500,device=device).reshape(5,3,100).requires_grad_()
    nn=torch.linspace(5,0,1500,device=device).reshape(5,3,100).requires_grad_()
    weights=plan['stratum_sizes'];loss=position_pair(pp,nn,scale,weights)
    manual=(torch.as_tensor(weights,device=device)*torch.stack([F.softplus(1-torch.log1p(pp[:,:,j]/scale)+torch.log1p(nn[:,:,j]/scale)) for j in range(100)]).mean(0)).sum()/weights.sum()
    torch.testing.assert_close(loss,manual)
    wrong=event_pair_loss(pp.mean(-1),nn.mean(-1),scale,phase_sizes=weights)
    assert abs(float(loss-wrong))>1e-6
    grad=torch.autograd.grad(loss,(pp,nn));assert all(torch.isfinite(g).all() for g in grad) and (grad[0]<0).all() and (grad[1]>0).all()
    assert initial==state_hashes(model)
    dump(root/'CHECKS.json',dict(passed=True,terminal_value_bit_exact=True,terminal_copy_bit_exact=True,
        terminal_decoder_gradient_bit_exact=True,position_mean_arithmetic=True,mean_before_softplus_rejected=True,
        correct_loss=float(loss),incorrect_meanV_loss=float(wrong),decoder_only=True,frozen_state_unchanged=True,
        official_initial_sha256=WEIGHT_SHA,source_plan_sha256=PLAN_SHA,source_only=True))


def validate(model,plan,data,scale,device,path):
    event,swat,smd=data
    with torch.inference_mode():
        def evaluate(role,ends):
            pairs=[values(model,x) for x in old.windows(role,ends,device)]
            return np.stack([v.cpu().numpy() for v,_ in pairs]),np.array([float(l) for _,l in pairs])
        p,_=evaluate(event,plan['positive_select']);n,_=evaluate(event,plan['normal_select'])
        sw,swloss=evaluate(swat,plan['swat_select'])
        ms=[evaluate(role,ends) for role,ends in zip(smd,plan['smd_select'])]
    position=np.logaddexp(0,1-np.log1p(p.astype(float)/scale)+np.log1p(n.astype(float)/scale))
    split=np.cumsum(plan['select_stratum_sizes'])[:-1]
    risks=np.array([a.mean() for a in np.split(position,split)])
    np.savez_compressed(path,positive_V=p,normal_V=n,nominal_swat_V=sw,nominal_smd_V=np.concatenate([a for a,_ in ms]),
        positive_endpoint=plan['positive_select'],normal_endpoint=plan['normal_select'],
        nominal_swat_endpoint=plan['swat_select'],nominal_smd_endpoint=plan['smd_select'],
        select_stratum_sizes=plan['select_stratum_sizes'],per_pair_position_loss=position,
        nominal_swat_copy_MSE=swloss,nominal_smd_copy_MSE=np.concatenate([b for _,b in ms]))
    return dict(risk=float(risks.mean()),stratum_risk=risks.tolist(),position_mean_risk=position.mean(0).tolist(),
                normal_reconstruction_MSE=float(np.concatenate((swloss,*[b for _,b in ms])).mean()),
                bank_sha256=sha(path),bank_file=path.name)


def train(w,arm):
    tic=time.monotonic();device=setup();seed(2025);root=w/OUT;assert read(root/'CHECKS.json')['passed']
    plan=old.load_plan(root);fit,select=datasets(w);scale=read(root/'SCALE.json')['value']
    model=decoder_model(w,device);before=state_hashes(model);cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state(device)
    opt=optimizer(model);d=root/'train'/arm;d.mkdir(parents=True,exist_ok=False)
    best=None;selections=[];trace=[]
    for step in range(241):
        if step:
            row=step-1;opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:group['lr']=1e-4*min(step/24,1)
            losses=[]
            for role,ends in [(fit[1],plan['swat_fit'][row])]+[(fit[2][i],[e]) for i,e in zip(plan['smd_entity'][row],plan['smd_fit'][row])]:
                for x in old.windows(role,ends,device):
                    _,loss=values(model,x);(loss/32).backward();losses.append(float(loss.detach()))
            pair=0.
            if arm!='copy':
                positive=torch.stack([values(model,x)[0] for x in old.windows(fit[0],plan['positive_fit'][row],device)]).reshape(5,3,100)
                normal=torch.stack([values(model,x)[0] for x in old.windows(fit[0],plan['normal_fit'][row],device)]).reshape(5,3,100)
                pairloss=(position_pair(positive,normal,scale,plan['stratum_sizes']) if arm=='all_positions' else
                          event_pair_loss(positive[:,:,99],normal[:,:,99],scale,phase_sizes=plan['stratum_sizes']))
                pairloss.backward();pair=float(pairloss.detach())
            grad=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            norm=float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grad])))
            assert np.isfinite(norm) and norm>0 and all(torch.isfinite(g).all() for g in grad)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            opt.step();trace.append(dict(step=step,normal_copy_loss=np.mean(losses),pair_loss=pair,gradient_norm=norm))
        if step%40==0:
            ck=d/f'step{step:03}.pt';torch.save(dict(model=model.state_dict(),update=step,arm=arm),ck)
            bank=d/f'select{step:03}.npz';risk=validate(model,plan,select,scale,device,bank)
            record=dict(step=step,checkpoint=ck.name,checkpoint_sha256=sha(ck),**risk);selections.append(record)
            if best is None or record['risk']<best['risk']:best=record
            print(arm,step,record['risk'],'best',best['step'],round(time.monotonic()-tic,1),flush=True)
    after=state_hashes(model)
    assert all(before[k]==v for k,v in after.items() if not k.startswith('decoder.'))
    assert all(not m.training for m in model.modules()) and torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(device))
    selected=d/'selected.pt';shutil.copyfile(d/best['checkpoint'],selected)
    dump(d/'TRACE.json',trace)
    dump(d/'SELECTION.json',dict(status='COMPLETED',arm=arm,updates=240,selected_step=best['step'],checkpoint_relative=str(selected.relative_to(root)),checkpoint_sha256=sha(selected),
         selections=selections,plan_sha256=PLAN_SHA,scale=scale,source_labels_for_selection=True,source_labels_for_gradients=arm!='copy',
         native_frozen_unchanged=True,eval=True,RNG_restored=True,seconds=time.monotonic()-tic,gpu=os.environ['CUDA_VISIBLE_DEVICES'],peak_reserved=torch.cuda.max_memory_reserved()))


def seal(w):
    root=w/OUT;arms={}
    for a in ARMS:
        p=root/'train'/a/'SELECTION.json';r=read(p);assert r['status']=='COMPLETED'
        assert sha(root/r['checkpoint_relative'])==r['checkpoint_sha256']
        arms[a]=dict(checkpoint_relative=r['checkpoint_relative'],checkpoint_sha256=r['checkpoint_sha256'],selected_step=r['selected_step'],selection_relative=str(p.relative_to(root)),selection_sha256=sha(p))
    dump(root/'ALL_MODELS_FROZEN.json',dict(status='COMPLETED_BEFORE_TARGET',arms=arms,source_plan_sha256=PLAN_SHA,query_read=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','check','train','seal']);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    {'prepare':lambda:prepare(Path.cwd()),'check':lambda:check(Path.cwd()),'train':lambda:train(Path.cwd(),a.arm),'seal':lambda:seal(Path.cwd())}[a.action]()
