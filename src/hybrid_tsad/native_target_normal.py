"""Matched target-normal decoder training with entity-isolated public windows."""
import argparse
import math
import os
import time
import shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from .native_v_conditioning import sha,state_hashes,WEIGHT_SHA
from .native_v_conditioning_runner import read,dump,setup,seed
from .native_v_replay import decoder_model,optimizer
from .native_position_aligned_training import values
from . import left_nasa_author as nasa

OUT=Path('results/native_target_normal_20261124_v1')
FULL_OUT=Path('results/native_target_normal_full_20261124_v1')
SOURCE=Path('results/native_position_aligned_20261124_v1')
ARMS=('official','copy','all_positions')
DOMAINS=('PSM','MSL')
EPOCHS=20


def normal_paths(w,domain):
    if domain=='PSM':return [('PSM',w/'data/common_psm_raw_20261029_v1/prefix.npy')]
    return [(e,nasa.raw_paths(w,'MSL','train',e)) for e in nasa.old.entity_list(w,'MSL')]


def normal_raw(w,domain):
    result={}
    for entity,path in normal_paths(w,domain):
        a=np.load(path,allow_pickle=False)
        if domain=='MSL':a=a[:,0:1]
        a=a.astype(np.float64)
        assert a.ndim==2 and a.shape[1]==(25 if domain=='PSM' else 1) and np.isfinite(a).all()
        result[entity]=a
    return result


def load_initial(w,arm,device):
    m=decoder_model(w,device)
    binding=dict(arm=arm,kind='official',checkpoint_sha256=WEIGHT_SHA,source_event_labels=False)
    if arm!='official':
        path=w/SOURCE/'train'/arm/'selected.pt';r=read(path.parent/'SELECTION.json')
        assert r['selected_step']=={'copy':120,'all_positions':200}[arm]
        assert sha(path)==r['checkpoint_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        before=state_hashes(m);m.load_state_dict(saved['model'],strict=True)
        assert all(before[k]==v for k,v in state_hashes(m).items() if not k.startswith('decoder.'))
        binding.update(kind='source_selected',checkpoint_relative=str(path.relative_to(w)),checkpoint_sha256=sha(path),
            selected_source_step=r['selected_step'],source_event_labels=True,
            source_gradient='normal reconstruction only' if arm=='copy' else 'normal reconstruction plus all-position source pairs',
            source_selection='full-position labeled source event risk')
    return m,binding


def prepare(w):
    root=w/OUT;root.mkdir(parents=True,exist_ok=False)
    for domain in DOMAINS:
        d=root/domain;d.mkdir();raw=normal_raw(w,domain);ids=list(raw)
        scaler=StandardScaler().fit(np.concatenate(list(raw.values())))
        np.savez_compressed(d/'scaler.npz',mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,n_samples_seen=scaler.n_samples_seen_)
        inputs=[];train=[];validation=[]
        for ei,(entity,path) in enumerate(normal_paths(w,domain)):
            n=len(raw[entity]);cut=int(.8*n)
            inputs.append(dict(entity=entity,path=str(path.relative_to(w)),sha256=sha(path),length=n,cut=cut,channels=raw[entity].shape[1]))
            train.extend((ei,int(t)) for t in range(0,cut-99,100))
            validation.extend((ei,int(t)) for t in range(cut,n-99,100))
        assert train and validation
        rng=np.random.default_rng(2025);orders=np.stack([rng.permutation(len(train)) for _ in range(EPOCHS)])
        np.savez_compressed(d/'PLAN.npz',train=np.array(train,np.int64),validation=np.array(validation,np.int64),orders=orders)
        updates=math.ceil(len(train)/32)*EPOCHS
        dump(d/'CONFIG.json',dict(domain=domain,seed=2025,inputs=inputs,entities=ids,epochs=EPOCHS,
            window=100,stride=100,batch=32,B=1,copies=10,mask_seed=84551,norm=0,correction=1,
            scaler_fit='all official normal train arrays including normal validation; exclude test',scaler_sha256=sha(d/'scaler.npz'),
            plan_sha256=sha(d/'PLAN.npz'),train_windows=len(train),validation_windows=len(validation),
            updates=updates,warmup_updates=math.ceil(.1*updates),selection='earliest minimum complete normal-validation per-copy MSE, including epoch0',
            optimizer=dict(name='AdamW',lr=1e-4,betas=[.9,.95],eps=1e-8,decay=.1,bias_LayerNorm_decay=0),
            source_initials=str(SOURCE),test_values_read=False,test_labels_read=False))
    dump(root/'CONFIG.json',dict(domains=DOMAINS,arms=ARMS,seed=2025,target_training='normal only; no source replay or pair gradient',
        source_checkpoint_roles='official / selected copy120 / selected all_positions200',global_seal_before_target_test=True))


def data(w,domain):
    d=w/OUT/domain;config=read(d/'CONFIG.json');raw=normal_raw(w,domain)
    for x in config['inputs']:assert sha(w/x['path'])==x['sha256'] and len(raw[x['entity']])==x['length']
    assert sha(d/'scaler.npz')==config['scaler_sha256'] and sha(d/'PLAN.npz')==config['plan_sha256']
    with np.load(d/'scaler.npz') as z:scaled=[((raw[e]-z['mean'])/z['scale']).astype(np.float32) for e in config['entities']]
    with np.load(d/'PLAN.npz') as z:plan={k:z[k].copy() for k in z.files}
    return scaled,plan,config


def window(scaled,row,device):
    entity,start=map(int,row);a=scaled[entity][start:start+100]
    assert len(a)==100
    return torch.from_numpy(a.copy()).to(device)[None]


@torch.no_grad()
def validate(model,scaled,rows,device):
    losses=[float(values(model,window(scaled,r,device))[1]) for r in rows]
    return float(np.mean(losses)),np.array(losses,dtype=np.float64)


def check(w,domain):
    t=time.monotonic();device=setup();seed(2025);scaled,plan,cfg=data(w,domain)
    for role,rows in [('train',plan['train']),('validation',plan['validation'])]:
        for ei,start in rows:
            inp=cfg['inputs'][int(ei)];lo,hi=(0,inp['cut']) if role=='train' else (inp['cut'],inp['length'])
            assert int(start)>=lo and int(start)+100<=hi
    model,binding=load_initial(w,'official',device);before=state_hashes(model)
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state(device)
    x=window(scaled,plan['train'][0],device);v,loss=values(model,x);loss.backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('decoder.'))
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.decoder.parameters())
    assert before==state_hashes(model) and torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(device))
    assert all(not m.training for m in model.modules())
    # One measured small logical batch, not a scientific short-training trial.
    model.zero_grad(set_to_none=True);count=min(32,len(plan['train']));torch.cuda.synchronize();start=time.monotonic()
    for row in plan['train'][:count]:
        _,loss=values(model,window(scaled,row,device));(loss/count).backward()
    torch.cuda.synchronize();seconds=time.monotonic()-start
    dump(w/OUT/domain/'CHECKS.json',dict(passed=True,window_roles=True,decoder_gradient=True,frozen_state_unchanged=True,
        eval=True,RNG_restored=True,per_window_B=1,logical_batch_windows=count,logical_batch_forward_backward_seconds=seconds,
        estimated_training_forward_backward_seconds=seconds/count*cfg['train_windows']*EPOCHS,
        normal_only=True,seconds=time.monotonic()-t))


def parameter_groups(model):
    groups = {}
    for name, parameter in model.named_parameters():
        groups.setdefault(name.split('.')[0], []).append((name, parameter))
    assert set(groups) == {'net', 'real_input_embed', 'imag_input_embed', 'encoder', 'decoder'}
    return groups


def check_full(w, domain, arm):
    device = setup()
    seed(2025)
    scaled, plan, cfg = data(w, domain)
    model, initial = load_initial(w, arm, device)
    before = state_hashes(model)
    buffers = {name: before[name] for name, _ in model.named_buffers()}
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    x = window(scaled, plan['train'][0], device)
    with torch.no_grad():
        v_ref, loss_ref = values(model, x)
    model.requires_grad_(True)
    v, loss = values(model, x)
    torch.testing.assert_close(v, v_ref, rtol=0, atol=0)
    torch.testing.assert_close(loss, loss_ref, rtol=0, atol=0)
    opt = optimizer(model)
    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(p) for p in model.parameters()}
    torch.cuda.synchronize()
    start = time.monotonic()
    batch = plan['train'][plan['orders'][0, :32]]
    for row in batch:
        _, loss = values(model, window(scaled, row, device))
        (loss/len(batch)).backward()
    groups = parameter_groups(model)
    gradient_norms = {}
    for group, params in groups.items():
        gradients = [p.grad for _, p in params if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        gradient_norms[group] = float(torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients])))
        assert gradient_norms[group] > 0
    for group in opt.param_groups:
        group['lr'] = 1e-4/cfg['warmup_updates']
    opt.step()
    torch.cuda.synchronize()
    elapsed = time.monotonic()-start
    after = state_hashes(model)
    changed = {group: sum(before[n] != after[n] for n, _ in params) for group, params in groups.items()}
    assert all(changed.values())
    assert all(after[n] == v for n, v in buffers.items())
    assert all(not m.training for m in model.modules())
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
    root = w/FULL_OUT/domain
    root.mkdir(parents=True, exist_ok=True)
    dump(root/f'CHECK_{arm}.json', dict(passed=True, initial=initial, bit_exact_loss_and_V=True,
         parameter_counts={g: sum(p.numel() for _, p in ps) for g, ps in groups.items()},
         gradient_norms=gradient_norms, changed_parameter_tensors=changed,
         logical_batch_seconds=elapsed, buffers_unchanged=True, eval=True, RNG_restored=True,
         normal_only=True, plan_sha256=cfg['plan_sha256'], scaler_sha256=cfg['scaler_sha256']))


def train(w,domain,arm, *, full=False):
    tic=time.monotonic();device=setup();seed(2025);scaled,plan,cfg=data(w,domain)
    out = FULL_OUT if full else OUT
    assert read(w/out/domain/(f'CHECK_{arm}.json' if full else 'CHECKS.json'))['passed']
    d=w/out/domain/'train'/arm;d.mkdir(parents=True,exist_ok=False)
    model,initial=load_initial(w,arm,device);before=state_hashes(model)
    if full:
        model.requires_grad_(True)
    opt=optimizer(model)
    groups = parameter_groups(model)
    cpu=torch.get_rng_state();gpu=torch.cuda.get_rng_state(device);updates=0;curve=[];best=None
    for epoch in range(EPOCHS+1):
        losses=[]
        if epoch:
            order=plan['orders'][epoch-1]
            for offset in range(0,len(order),32):
                batch=plan['train'][order[offset:offset+32]];opt.zero_grad(set_to_none=True)
                updates+=1;lr=1e-4*min(updates/cfg['warmup_updates'],1.)
                for g in opt.param_groups:g['lr']=lr
                for row in batch:
                    _,loss=values(model,window(scaled,row,device));(loss/len(batch)).backward();losses.append(float(loss.detach()))
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                opt.step()
        mse,val=validate(model,scaled,plan['validation'],device)
        ck=d/f'epoch{epoch:02}.pt';torch.save(dict(model=model.state_dict(),epoch=epoch,updates=updates),ck)
        vp=d/f'validation{epoch:02}.npz';np.savez_compressed(vp,copy_MSE=val,windows=plan['validation'])
        r=dict(epoch=epoch,updates=updates,validation_MSE=mse,training_MSE=float(np.mean(losses)) if losses else None,
            checkpoint=ck.name,checkpoint_sha256=sha(ck),validation_bank=vp.name,validation_bank_sha256=sha(vp),
            nominal_window_exposures=epoch*len(plan['train']))
        curve.append(r)
        if best is None or mse<best['validation_MSE']:best=r
        print(domain,arm,epoch,updates,mse,'best',best['epoch'],round(time.monotonic()-tic,1),flush=True)
    torch.cuda.synchronize()
    assert updates==cfg['updates']
    after = state_hashes(model)
    changed = {g: sum(before[n] != after[n] for n, _ in ps) for g, ps in groups.items()}
    assert all(before[n] == after[n] for n, _ in model.named_buffers())
    if full:
        assert all(changed.values())
    else:
        assert all(before[k]==v for k,v in after.items() if not k.startswith('decoder.'))
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(device))
    assert all(not m.training for m in model.modules())
    shutil.copyfile(d/best['checkpoint'],d/'selected.pt')
    dump(d/'CURVE.json',curve)
    dump(d/'SELECTION.json',dict(status='COMPLETED',domain=domain,arm=arm,seed=2025,initial=initial,executed_epochs=EPOCHS,
        executed_updates=updates,selected_epoch=best['epoch'],selected_updates=best['updates'],selected_validation_MSE=best['validation_MSE'],
        selected_window_exposures=best['nominal_window_exposures'],full_window_exposures=EPOCHS*len(plan['train']),
        checkpoint_relative=str((d/'selected.pt').relative_to(w/out)),checkpoint_sha256=sha(d/'selected.pt'),
        plan_sha256=cfg['plan_sha256'],scaler_sha256=cfg['scaler_sha256'],seconds=time.monotonic()-tic,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),device=os.environ.get('CUDA_VISIBLE_DEVICES'),
        native_frozen_unchanged=not full, train_scope='full' if full else 'decoder',
        changed_parameter_tensors_at_epoch20=changed, buffers_unchanged=True,
        RNG_restored=True,target_labels_read=False,test_values_read=False,
        curve=curve))


def seal(w, *, full=False):
    out = FULL_OUT if full else OUT
    domains={}
    for domain in DOMAINS:
        arms={}
        for arm in ARMS:
            r=read(w/out/domain/'train'/arm/'SELECTION.json');assert r['status']=='COMPLETED'
            assert sha(w/out/r['checkpoint_relative'])==r['checkpoint_sha256']
            assert r.get('train_scope', 'decoder') == ('full' if full else 'decoder')
            arms[arm]={k:r[k] for k in ('checkpoint_relative','checkpoint_sha256','selected_epoch','selected_updates','selected_validation_MSE')}
        domains[domain]=arms
    dump(w/out/'ALL_MODELS_FROZEN.json',dict(status='COMPLETED_BEFORE_TEST',domains=domains,test_values_read=False,test_labels_read=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','check','train','seal'));p.add_argument('--domain',choices=DOMAINS);p.add_argument('--arm',choices=ARMS);p.add_argument('--full',action='store_true');a=p.parse_args();w=Path.cwd()
    if a.action=='prepare':prepare(w)
    elif a.action=='seal':seal(w,full=a.full)
    elif a.action=='check':
        if a.full:check_full(w,a.domain,a.arm)
        else:check(w,a.domain)
    else:train(w,a.domain,a.arm,full=a.full)
