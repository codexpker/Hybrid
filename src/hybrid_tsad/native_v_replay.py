"""Native TimeRadar decoder-only replay, preserving B1 order and fixed masks.

This entry is independent of stopped adapters and conditional-V predictors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .native_v_conditioning import load_native, fixed_native_rng, state_hashes, sha
from .native_v_conditioning_runner import read, dump, role_values, setup, seed

EXPERIMENT='timeradar_native_v_replay_20261120_v1'
OLD='native_v_conditioning_20261119_v1'
TARGET_MEAN_EXPERIMENT='timeradar_target_mean_20261120_v1'
REPEAT_EXPERIMENT='timeradar_target_mean_repeat_20261120_v1'
SMD_EXPERIMENT='timeradar_target_mean_smd_20261120_v1'
SMD_ENTITIES=['machine-3-2','machine-3-3']


def training_seed(task):
    value=task['replay_training'].get('seed',2025)
    if isinstance(value,bool)or not isinstance(value,int)or value<0:raise ValueError('Non-negative integer training seed')
    return value


def sampling_plan(source_count,target_count,updates=2000,half=32,seed_value=2025):
    rngs=[np.random.default_rng(s)for s in np.random.SeedSequence(seed_value).spawn(3)]
    return dict(T=np.random.default_rng(seed_value).integers(target_count,size=(updates,half)),
        S=rngs[0].integers(source_count,size=(updates,half)),
        T_prime=rngs[1].integers(target_count,size=(updates,half)),
        S_prime=rngs[2].integers(source_count,size=(updates,half)))


def pool_starts(role):
    lo,hi=role
    return np.arange(lo,hi-99,100,dtype=np.int64)


def paired_starts(plan,source,target,arm,step):
    if arm=='R':return [('U_FIT',target[plan['T'][step]]),('SOURCE_FIT',source[plan['S'][step]])]
    if arm=='TT':return [('U_FIT',target[plan['T'][step]]),('U_FIT',target[plan['T_prime'][step]])]
    if arm=='SS':return [('SOURCE_FIT',source[plan['S_prime'][step]]),('SOURCE_FIT',source[plan['S'][step]])]
    raise ValueError(arm)


def prepare(w,root):
    root.mkdir(parents=True,exist_ok=False)
    for domain in ['SWAT','PSM']:
        out=root/domain;out.mkdir();old=w/'results'/OLD/domain
        task=read(old/'TASK.json');task['experiment']=EXPERIMENT;task['roles']['U_FIT']=[495000,585000]if domain=='SWAT'else[0,17568]
        task['arms']={'TT':['T','T_prime'],'SS':['S_prime','S'],'R':['T','S']}
        source=pool_starts(task['roles']['SOURCE_FIT']);target=pool_starts(task['roles']['U_FIT'])
        plan=sampling_plan(len(source),len(target));np.savez_compressed(out/'PLAN.npz',**plan,source_starts=source,target_starts=target)
        import shutil
        shutil.copyfile(old/'SCALER.npz',out/'SCALER.npz')
        for role in ['CAL','QUERY']:
            times=np.load(w/task['indexes'][role]['relative']);np.save(out/f'{role}_INDEX.npy',times)
            task['indexes'][role].update(relative=str((out/f'{role}_INDEX.npy').relative_to(w)),sha256=sha(out/f'{role}_INDEX.npy'))
        task['replay_training']=dict(seed=2025,updates=2000,batch=64,half=32,optimizer='AdamW',lr=1e-4,betas=[.9,.95],eps=1e-8,
            weight_decay=.1,bias_LayerNorm_weight_decay=0,warmup=200,checkpoint='endpoint2000_no_SELECT',
            trainable='native.decoder only; all modules eval',loss='per window mean10x100xC MSE /64, accumulated B1 then one optimizer step',
            SOURCE_FIT_pool=len(source),U_FIT_pool=len(target),sampling_plan_sha256=sha(out/'PLAN.npz'),
            pool='complete non-overlap100 windows anchored independently at each role lo',
            masks='each B1 CPU/CUDA fork_rng reset84551,copies10 complementary')
        task['source_noU_bank_root']=str((w/'results'/OLD/'banks'/domain/'A').relative_to(w))
        dump(out/'TASK.json',task)
        dump(out/'POOL_EXPOSURE.json',dict(pool_counts=dict(SOURCE_FIT=len(source),U_FIT=len(target)),
            streams={k:dict(exposure=int(v.size),unique_pool_windows=int(len(np.unique(v))),repeated_exposures=int(v.size-len(np.unique(v))))for k,v in plan.items()},
            independent_events_not_inferred=True,labels_read=False))
    dump(root/'CONFIG.json',dict(experiment=EXPERIMENT,domains=['SWAT','PSM'],arms=['TT','SS','R'],seed=2025,
        official_initial_checkpoint='528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c',labels_read=False))


def prepare_target_mean(w,root):
    import shutil
    old=w/'results'/EXPERIMENT;frozen=old/'GLOBAL_RAW_BANK_SEAL.json'
    assert sha(frozen)=='6220f28d1b865b83629ad9426933d64c164d5968b26f4c180ae8c0f88686d51e'
    root.mkdir(parents=True,exist_ok=False);binding=[]
    for domain in ['SWAT','PSM']:
        target=root/domain;target.mkdir()
        for name in ['TASK.json','SCALER.npz','PLAN.npz']:
            source=old/domain/name;shutil.copyfile(source,target/name)
            assert sha(source)==sha(target/name)
            binding.append(dict(domain=domain,name=name,original_relative=str(source.relative_to(w)),sha256=sha(source)))
    dump(root/'CONFIG.json',dict(experiment=TARGET_MEAN_EXPERIMENT,seed=2025,loss_profile='target_mean_source_copy',
        arms=['R_target_mean'],models=2,freeze_name='ALL_MODELS_FROZEN.json',inherited_copied_bytes=binding,
        reuse_global_relative=str(frozen.relative_to(w)),reuse_global_sha256=sha(frozen),
        reuse_metrics_sha256='2dfd4d51a434b90308db120905c9e709907f1b0d3abb49f30182e75c2f6f67df',
        actual_loss='U_FIT E_window; SOURCE_FIT copyMSE; per window /64 and one optimizer step',
        unchanged='official initial/scaler/PLAN/optimizer/warmup/budget/B1 masks/score/axes/CAL',labels_read=False))


def prepare_repeats(w,root):
    import shutil
    old=w/'results'/EXPERIMENT;mean=w/'results'/TARGET_MEAN_EXPERIMENT
    references=[dict(relative=str((r/'GLOBAL_RAW_BANK_SEAL.json').relative_to(w)),sha256=sha(r/'GLOBAL_RAW_BANK_SEAL.json'))for r in [old,mean]]
    assert [r['sha256']for r in references]==['6220f28d1b865b83629ad9426933d64c164d5968b26f4c180ae8c0f88686d51e','a54a115c3f007c761e38d93b6d95b68211d527b88a90a6bef789de0b51889d31']
    root.mkdir(parents=True,exist_ok=False);baseline_checks=[]
    for domain in ['SWAT','PSM']:
        with np.load(old/domain/'PLAN.npz')as z:base={k:z[k].copy()for k in z.files}
        recreated=sampling_plan(len(base['source_starts']),len(base['target_starts']),seed_value=2025)
        assert all(np.array_equal(recreated[k],base[k])for k in recreated)
        baseline_checks.append(dict(domain=domain,seed2025_arrays_equal=True,original_plan_sha256=sha(old/domain/'PLAN.npz')))
    for value in [2026,2027]:
        sr=root/f'seed{value}';sr.mkdir()
        for domain in ['SWAT','PSM']:
            out=sr/domain;out.mkdir();task=read(old/domain/'TASK.json')
            with np.load(old/domain/'PLAN.npz')as z:source=z['source_starts'].copy();target=z['target_starts'].copy()
            plan=sampling_plan(len(source),len(target),seed_value=value)
            np.savez_compressed(out/'PLAN.npz',**plan,source_starts=source,target_starts=target)
            shutil.copyfile(old/domain/'SCALER.npz',out/'SCALER.npz');assert sha(out/'SCALER.npz')==task['scaler_sha256']
            task['seed']=value;task['replay_training']['seed']=value
            task['replay_training']['sampling_plan_sha256']=sha(out/'PLAN.npz')
            task['experiment']=REPEAT_EXPERIMENT;task['matched_repeat_arms']=['R','R_target_mean']
            task['inherited_reference_not_executed']=['SOURCE_SELECT','source_permutation_sha256','original training(B/C/D)']
            dump(out/'TASK.json',task)
        dump(sr/'CONFIG.json',dict(experiment=REPEAT_EXPERIMENT,seed=value,matched_repeat=True,
            arms=['R','R_target_mean'],models=4,freeze_name='ALL_MODELS_FROZEN.json',labels_read=False))
    dump(root/'CONFIG.json',dict(experiment=REPEAT_EXPERIMENT,seeds=[2026,2027],new_models=8,new_banks=16,
        baseline_seed2025_reused=True,baseline_references=references,paired_arms=['R','R_target_mean'],labels_read=False))
    dump(root/'SEED2025_PLAN_EQUIVALENCE.json',dict(status='PASS',checks=baseline_checks,no_training_or_inference=True))


def decoder_model(w,device):
    model=load_native(w,device)
    model.decoder.requires_grad_(True);model.eval()
    assert all(not module.training for module in model.modules())
    assert all(p.requires_grad==name.startswith('decoder.')for name,p in model.named_parameters())
    return model


def optimizer(model):
    no_decay=set()
    for prefix,module in model.named_modules():
        if isinstance(module,torch.nn.LayerNorm):no_decay.update(prefix+'.'+n for n,_ in module.named_parameters())
    groups=[dict(params=[p for n,p in model.named_parameters()if p.requires_grad and n not in no_decay and 'bias'not in n],weight_decay=.1),
        dict(params=[p for n,p in model.named_parameters()if p.requires_grad and(n in no_decay or'bias'in n)],weight_decay=0.)]
    return torch.optim.AdamW(groups,lr=1e-4,betas=(.9,.95),eps=1e-8)


def input_windows(values,lo,starts,device):
    for start in starts:
        relative=int(start)-lo
        if relative<0 or relative+100>len(values):raise PermissionError('Window leaves role')
        yield torch.from_numpy(values[relative:relative+100].copy()).to(device)[None]


def reconstruction_components(outputs,x):
    """All terms keep their graph; variance is over ten copies, full100xC."""
    if outputs.shape[0]!=10:raise ValueError('Exactly ten copies required')
    return ((outputs-x[None]).square().mean(),(outputs.mean(0)-x).square().mean(),outputs.var(0,correction=1).mean())


def accumulate(model,windows,device,divisor,loss_mode='copy',diagnostics=None):
    losses=[];orders=[];hook=model.net.register_forward_hook(lambda m,a,o:orders.append(float(o.detach().reshape(-1)[0])))
    try:
        for x in windows:
            with fixed_native_rng(device):_,outputs=model(x,norm=0,mask_mode='c',copies=10)
            assert outputs.shape==(10,1,100,x.shape[-1])
            if loss_mode=='copy'and diagnostics is None:
                loss=(outputs-x[None]).square().mean()
            else:
                full,E,V=reconstruction_components(outputs,x)
                loss=E if loss_mode=='mean'else full
                if loss_mode not in ['mean','copy']:raise ValueError(loss_mode)
                if diagnostics is not None:diagnostics.append(dict(E=float(E.detach()),V=float(V.detach()),copy=float(full.detach())))
            assert torch.isfinite(loss)
            (loss/divisor).backward();losses.append(float(loss.detach()))
    finally:hook.remove()
    assert losses and all(p.grad is None or torch.isfinite(p.grad).all()for p in model.parameters())
    assert all(p.grad is None for p in model.parameters()if not p.requires_grad)
    return float(np.mean(losses)),orders


def role_arrays(w,root,domain,roles):
    task=read(root/domain/'TASK.json');arrays={}
    for role in roles:arrays[role]=role_values(w,root/domain,task,role)
    return task,arrays


def train(w,root,domain,arm,loss_profile='native'):
    assert read(root/domain/'GATE.json')['status']=='PASS_ENGINEERING_ONLY'
    out=root/'train'/domain/arm;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();device=setup(20)
    roles=['SOURCE_FIT']if arm=='SS'else['U_FIT']if arm=='TT'else['SOURCE_FIT','U_FIT']
    task,arrays=role_arrays(w,root,domain,roles);p=root/domain/'PLAN.npz';assert sha(p)==task['replay_training']['sampling_plan_sha256']
    with np.load(p)as z:plan={k:z[k].copy()for k in z.files}
    seed_value=training_seed(task);seed(seed_value);model=decoder_model(w,device);initial=state_hashes(model);opt=optimizer(model);trace=[]
    seed_binding=dict(seed=seed_value)if seed_value!=2025 else{}
    seed_receipt=dict(effective_torch_CPU_seed=int(torch.initial_seed()),effective_torch_CUDA_seed=int(torch.cuda.initial_seed()))if seed_value!=2025 else{}
    target_mean=loss_profile=='target_mean_source_copy'
    if loss_profile not in ['native','target_mean_source_copy']or(target_mean and arm!='R_target_mean'):raise ValueError('Fixed R target-mean profile only')
    pair_arm='R'if target_mean else arm
    for step in range(2000):
        lr=1e-4*min((step+1)/200,1.)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);losses=[];orders=[];components={}
        for role,starts in paired_starts(plan,plan['source_starts'],plan['target_starts'],pair_arm,step):
            values,_,lo,_=arrays[role]
            diagnostic=[]if target_mean else None
            mode='mean'if target_mean and role=='U_FIT'else'copy'
            loss,order=accumulate(model,input_windows(values,lo,starts,device),device,64,mode,diagnostic);losses.append(loss);orders+=order
            if target_mean:
                tag='T'if role=='U_FIT'else'S'
                components.update({k+'_'+tag:float(np.mean([r[k]for r in diagnostic]))for k in ['E','V','copy']})
        assert len(orders)==64
        gradient=float(torch.stack([p.grad.norm()for p in model.decoder.parameters()if p.grad is not None]).norm())
        assert np.isfinite(gradient);opt.step()
        trace.append(dict(update=step+1,loss=float(np.mean(losses)),gradient_norm=gradient,lr=lr,
            per_window_order_sha256=hashlib.sha256(np.asarray(orders,np.float32).tobytes()).hexdigest(),**components))
        if(step+1)%200==0:print(json.dumps(dict(event='TRAIN_PROGRESS',domain=domain,arm=arm,update=step+1,elapsed=time.monotonic()-start)),flush=True)
    final=state_hashes(model);frozen={k:v for k,v in initial.items()if not k.startswith('decoder.')}
    assert all(final[k]==v for k,v in frozen.items())and any(final[k]!=initial[k]for k in initial if k.startswith('decoder.'))
    assert all(not m.training for m in model.modules())
    checkpoint=out/'ENDPOINT.pth';torch.save(dict(model=model.state_dict(),update=2000,domain=domain,arm=arm,**seed_binding),checkpoint)
    torch.save(opt.state_dict(),out/'OPTIMIZER.pth');dump(out/'TRACE.json',trace)
    profile_binding=dict(loss_profile=loss_profile)if target_mean or seed_value!=2025 else{}
    dump(out/'TRAIN_RECEIPT.json',dict(status='COMPLETED',domain=domain,arm=arm,seed=seed_value,updates=2000,logical_batch=64,
        B1_forward_count=128000,exposures=128000,roles_read=roles,labels_read=False,CAL_QUERY_opened=False,
        checkpoint_relative=str(checkpoint.relative_to(w)),checkpoint_sha256=sha(checkpoint),optimizer_sha256=sha(out/'OPTIMIZER.pth'),
        plan_sha256=sha(p),scaler_sha256=task['scaler_sha256'],initial_state_hashes=initial,final_state_hashes=final,
        frozen_parameters_buffers_unchanged=True,all_modules_eval=True,trainable_names=[n for n,p in model.named_parameters()if p.requires_grad],
        trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),
        elapsed_GPU_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),**profile_binding,**seed_receipt))
    print(json.dumps(dict(event='TRAINING_COMPLETED',domain=domain,arm=arm)),flush=True)


def gate(w,root,domain):
    out=root/domain/'engineering_gate';out.mkdir();device=setup(20);started=time.monotonic()
    task,arrays=role_arrays(w,root,domain,['SOURCE_FIT','U_FIT']);seed();model=decoder_model(w,device);initial=state_hashes(model)
    with np.load(root/domain/'PLAN.npz')as z:plan={k:z[k].copy()for k in z.files}
    opt=optimizer(model);rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone());probe=[]
    for role,starts in paired_starts(plan,plan['source_starts'],plan['target_starts'],'R',0):
        values,_,lo,_=arrays[role];probe.extend(list(input_windows(values,lo,starts,device)))
    assert len(probe)==64
    # The loaded official state must reproduce the previous normal noU fixture.
    old=w/'results/timeradar_native_v_protocol_20261119_v1'/domain/'t512.npz'
    with np.load(old)as z:
        x=torch.from_numpy(z['input'].copy()).to(device)[None];expected=z['native_V_terminal'].copy()
    with torch.no_grad(),fixed_native_rng(device):
        _,copies=model(x,norm=0,mask_mode='c',copies=10);score=model.cal_anomaly_score(x,copies,anomaly_criterion=None,L=1)[0,99]
    assert np.array_equal(score.cpu().numpy(),expected)
    torch.cuda.synchronize();tt=time.monotonic();opt.zero_grad(set_to_none=True)
    loss,orders=accumulate(model,iter(probe),device,64);opt.step();torch.cuda.synchronize();elapsed=time.monotonic()-tt
    final=state_hashes(model)
    assert all(final[k]==v for k,v in initial.items()if not k.startswith('decoder.'))
    assert any(final[k]!=v for k,v in initial.items()if k.startswith('decoder.'))
    assert torch.equal(rng[0],torch.get_rng_state())and torch.equal(rng[1],torch.cuda.get_rng_state(device))
    # Independence uses actual B1 windows and fixed masks, not a mixed B>1 call.
    with torch.no_grad():
        for sample in [probe[0],probe[1],probe[0]]:
            with fixed_native_rng(device):_,outputs=model(sample,norm=0,mask_mode='c',copies=10)
            if sample is probe[0]:
                if 'witness'in locals():assert torch.equal(witness,outputs)
                else:witness=outputs.clone()
    receipt=dict(status='PASS_ENGINEERING_ONLY',domain=domain,logical_batch=64,B1_accumulation=True,all_modules_eval=True,
        frozen_except_decoder=True,original_noU_V_bitexact=True,RNG_restored=True,neighbor_independent_B1=True,
        actual_decoder_gradient_path=True,loss=loss,logical_update_seconds=elapsed,estimated_train_seconds_per_arm=elapsed*2000,
        orders=orders,trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),
        elapsed_GPU_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),labels_read=False,
        gate_model_discarded=True)
    dump(out/'GATE.json',receipt);dump(root/domain/'GATE.json',receipt)
    print(json.dumps(dict(event='GATE_COMPLETED',**receipt)),flush=True)


def gate_target_mean(w,root,domain):
    out=root/domain/'engineering_gate';out.mkdir();device=setup(20);started=time.monotonic();seed()
    task,arrays=role_arrays(w,root,domain,['SOURCE_FIT']);seed(training_seed(task));model=decoder_model(w,device);before=state_hashes(model);opt=optimizer(model)
    assert sha(root/domain/'PLAN.npz')==task['replay_training']['sampling_plan_sha256']
    values,_,lo,_=arrays['SOURCE_FIT'];windows=list(input_windows(values,lo,[lo+413,lo+925],device))
    rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone());terms=[]
    for x in windows:
        with fixed_native_rng(device):_,outputs=model(x,norm=0,mask_mode='c',copies=10)
        full,E,V=reconstruction_components(outputs,x);torch.testing.assert_close(full,E+.9*V,rtol=2e-5,atol=2e-6)
        terms.append((full,E,V))
    old_loss=(terms[0][0]+terms[1][0])/2;new_loss=(terms[0][1]+terms[1][0])/2;difference=.45*terms[0][2]
    torch.testing.assert_close(old_loss-new_loss,difference,rtol=2e-5,atol=2e-6)
    params=list(model.decoder.parameters());a=torch.autograd.grad(old_loss,params,retain_graph=True)
    b=torch.autograd.grad(new_loss,params,retain_graph=True);c=torch.autograd.grad(difference,params,retain_graph=True)
    for x,y,z in zip(a,b,c):torch.testing.assert_close(x-y,z,rtol=2e-4,atol=2e-5)
    maximum=max(float((x-y-z).abs().max())for x,y,z in zip(a,b,c))
    opt.zero_grad(set_to_none=True);new_loss.backward();assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in params)
    assert all(p.grad is None for p in model.parameters()if not p.requires_grad);opt.step()
    after=state_hashes(model);assert all(after[k]==v for k,v in before.items()if not k.startswith('decoder.'))
    assert all(not m.training for m in model.modules())and torch.equal(rng[0],torch.get_rng_state())and torch.equal(rng[1],torch.cuda.get_rng_state(device))
    receipt=dict(status='PASS_ENGINEERING_ONLY',domain=domain,seed=training_seed(task),nominal_SOURCE_FIT_windows=2,labels_read=False,
        loss_identity_absolute_error=float((old_loss-new_loss-difference).detach().abs()),gradient_identity_max_absolute_error=maximum,
        graph_connected_E=True,frozen_except_decoder=True,all_modules_eval=True,RNG_restored=True,
        plan_sha256=sha(root/domain/'PLAN.npz'),old_plan_bytes_reused=training_seed(task)==2025 and task.get('domain')!='SMD',paired_plan_verified=True,gate_models_discarded=True,
        estimated_train_seconds_per_arm=read(w/'results'/EXPERIMENT/task.get('timing_reference_domain',domain)/'GATE.json')['estimated_train_seconds_per_arm']*1.25,
        elapsed_GPU_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    dump(out/'GATE.json',receipt);dump(root/domain/'GATE.json',receipt)
    print(json.dumps(dict(event='GATE_COMPLETED',**receipt)),flush=True)


def matrix_spec(root):
    config=read(root/'CONFIG.json')
    if config.get('smd_extension'):return ['R','R_target_mean'],'ALL_MODELS_FROZEN.json',4
    if config.get('matched_repeat'):return ['R','R_target_mean'],'ALL_MODELS_FROZEN.json',4
    if config.get('loss_profile')=='target_mean_source_copy':return ['R_target_mean'],'ALL_MODELS_FROZEN.json',2
    return ['TT','SS','R'],'ALL_SIX_MODELS_FROZEN.json',6


def scoring_cases(w,root):
    import shutil
    arms,freeze_name,count=matrix_spec(root)
    frozen=read(root/freeze_name);assert frozen['status']=='COMPLETED'and len(frozen['trainings'])==count
    for arm in arms:
        case=root/'readout'/arm;case.mkdir(parents=True,exist_ok=False)
        dump(case/freeze_name,dict(status='COMPLETED',parent_relative=str((root/freeze_name).relative_to(w)),parent_sha256=sha(root/freeze_name)))
        for domain in ['SWAT','PSM']:
            target=case/domain;target.mkdir();shutil.copyfile(root/domain/'TASK.json',target/'TASK.json');shutil.copyfile(root/domain/'SCALER.npz',target/'SCALER.npz')
            receipt=root/'train'/domain/arm/'TRAIN_RECEIPT.json';selected=read(receipt)
            dump(target/'SCORE_BINDING.json',dict(domain=domain,arm=arm,checkpoint_relative=selected['checkpoint_relative'],
                actual_endpoint_sha256=selected['checkpoint_sha256'],train_receipt_relative=str(receipt.relative_to(w)),train_receipt_sha256=sha(receipt),
                actual_feature_window=100,feature_end='t inclusive',legacy512_predictor_not_executed=True,
                **(dict(seed=selected['seed'])if read(root/'CONFIG.json').get('matched_repeat')else{})))


def score_cache(w,root,domain,arm,role,part,parts):
    from .native_v_conditioning_runner import cache
    _,freeze_name,_=matrix_spec(root);frozen=read(root/freeze_name)
    if read(root/'CONFIG.json').get('smd_extension'):
        return smd_score_cache(w,root,domain,arm,role,part,parts)
    selected=next(r for r in frozen['trainings']if r['domain']==domain and r['arm']==arm)
    checkpoint=w/selected['checkpoint_relative'];assert sha(checkpoint)==selected['checkpoint_sha256']
    meta=dict(seed=selected['seed'],method=arm)if read(root/'CONFIG.json').get('matched_repeat')else None
    cache(w,root/'readout'/arm,domain,role,part,parts,native_checkpoint=checkpoint,freeze_name=freeze_name,metadata=meta)


def queue_score(w,root):
    from .native_v_conditioning_queue import run_jobs
    from .native_v_conditioning_runner import merge_cache
    scoring_cases(w,root);arms,freeze_name,_=matrix_spec(root)
    tasks=[(domain+'_'+arm+'_'+role+'_'+str(part),['score_cache','--domain',domain,'--arm',arm,'--role',role,'--part',str(part),'--parts',str(parts)],30,7200)
        for arm in arms for domain in ['SWAT','PSM']for role,parts in [('CAL',1),('QUERY',4)]for part in range(parts)]
    jobs=run_jobs(w,root,'native_readout',tasks,'hybrid_tsad.native_v_replay');banks=[];reused=[]
    for domain in ['SWAT','PSM']:
        for arm in arms:
            for role in ['CAL','QUERY']:
                case=root/'readout'/arm;merge_cache(w,case,domain,role);p=case/'cache'/domain/role/'SEALED.json';sealed=read(p)
                banks.append(dict(domain=domain,method=arm,role=role,relative=sealed['relative'],sha256=sealed['sha256'],
                    receipt_relative=str(p.relative_to(w)),receipt_sha256=sha(p)))
        for role in ([]if arms==['R_target_mean']else['CAL','QUERY']):
            p=w/'results'/OLD/'banks'/domain/'A'/role/'RECEIPT.json';r=read(p);assert sha(w/r['relative'])==r['sha256']
            reused.append(dict(domain=domain,method='noU',role=role,relative=r['relative'],sha256=r['sha256'],receipt_relative=str(p.relative_to(w)),receipt_sha256=sha(p)))
    alljobs=read(root/'GATES_COMPLETED.json')['jobs']+read(root/freeze_name)['jobs']+jobs
    trainings=[]
    for domain in ['SWAT','PSM']:
        for arm in arms:
            p=root/'train'/domain/arm/'TRAIN_RECEIPT.json';row=read(p)
            trainings.append(dict(row,train_receipt_relative=str(p.relative_to(w)),train_receipt_sha256=sha(p)))
    dump(root/'GLOBAL_RAW_BANK_SEAL.json',dict(status='COMPLETED',no_labels_decoded=True,labels_read=False,banks=banks,noU_reused=reused,
        trainings=trainings,jobs=alljobs,reuse_binding=read(root/'CONFIG.json').get('reuse_global_sha256'),actual_GPU_hours=sum(j['actual_GPU_seconds']for j in alljobs)/3600))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='GLOBAL_RAW_BANK_SEAL',new_banks=len(banks),reused_noU_banks=len(reused))),flush=True)


def queue_repeats(w,root):
    from .native_v_conditioning_queue import run_jobs
    from .native_v_conditioning_runner import merge_cache
    tasks=[(f'seed{value}_{domain}',['gate','--output',str(root/f'seed{value}'),'--domain',domain,'--loss-profile','target_mean_source_copy'],20,600)
        for value in [2026,2027]for domain in ['SWAT','PSM']]
    gates=run_jobs(w,root,'repeat_engineering_gate',tasks,'hybrid_tsad.native_v_replay')
    dump(root/'GATES_COMPLETED.json',dict(status='COMPLETED',jobs=gates,labels_read=False))
    tasks=[]
    for value in [2026,2027]:
        sr=root/f'seed{value}'
        for domain in ['SWAT','PSM']:
            timeout=max(3600,int(read(sr/domain/'GATE.json')['estimated_train_seconds_per_arm']*1.8+600))
            for arm in ['R','R_target_mean']:
                profile='target_mean_source_copy'if arm=='R_target_mean'else'native'
                tasks.append((f'seed{value}_{domain}_{arm}',['train','--output',str(sr),'--domain',domain,'--arm',arm,'--loss-profile',profile],20,timeout))
    training_jobs=run_jobs(w,root,'repeat_decoder_training',tasks,'hybrid_tsad.native_v_replay')
    for value in [2026,2027]:
        sr=root/f'seed{value}';rows=[read(sr/'train'/d/arm/'TRAIN_RECEIPT.json')for d in ['SWAT','PSM']for arm in ['R','R_target_mean']]
        assert len(rows)==4 and all(r['seed']==value and r['updates']==2000 for r in rows)
        dump(sr/'ALL_MODELS_FROZEN.json',dict(status='COMPLETED',models=4,seed=value,trainings=rows,labels_read=False))
    dump(root/'ALL_EIGHT_MODELS_FROZEN.json',dict(status='COMPLETED',models=8,jobs=training_jobs,labels_read=False,
        seed_seals=[dict(seed=value,relative=str((root/f'seed{value}'/'ALL_MODELS_FROZEN.json').relative_to(w)),sha256=sha(root/f'seed{value}'/'ALL_MODELS_FROZEN.json'))for value in [2026,2027]]))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='ALL_EIGHT_MODELS_FROZEN')),flush=True)
    for value in [2026,2027]:scoring_cases(w,root/f'seed{value}')
    tasks=[(f'seed{value}_{domain}_{arm}_{role}_{part}',
        ['score_cache','--output',str(root/f'seed{value}'),'--domain',domain,'--arm',arm,'--role',role,'--part',str(part),'--parts',str(parts)],30,7200)
        for value in [2026,2027]for arm in ['R','R_target_mean']for domain in ['SWAT','PSM']for role,parts in [('CAL',1),('QUERY',4)]for part in range(parts)]
    score_jobs=run_jobs(w,root,'repeat_native_readout',tasks,'hybrid_tsad.native_v_replay');banks=[];trainings=[]
    for value in [2026,2027]:
        sr=root/f'seed{value}'
        for domain in ['SWAT','PSM']:
            for arm in ['R','R_target_mean']:
                p=sr/'train'/domain/arm/'TRAIN_RECEIPT.json';trainings.append(dict(read(p),train_receipt_relative=str(p.relative_to(w)),train_receipt_sha256=sha(p)))
                for role in ['CAL','QUERY']:
                    case=sr/'readout'/arm;merge_cache(w,case,domain,role,metadata=dict(seed=value,method=arm))
                    p=case/'cache'/domain/role/'SEALED.json';sealed=read(p)
                    banks.append(dict(domain=domain,method=arm,seed=value,role=role,relative=sealed['relative'],sha256=sealed['sha256'],receipt_relative=str(p.relative_to(w)),receipt_sha256=sha(p)))
    jobs=gates+training_jobs+score_jobs
    dump(root/'GLOBAL_RAW_BANK_SEAL.json',dict(status='COMPLETED',models=8,banks=banks,trainings=trainings,jobs=jobs,
        baseline_seed2025_references=read(root/'CONFIG.json')['baseline_references'],no_labels_decoded=True,labels_read=False,
        actual_GPU_hours=sum(j['actual_GPU_seconds']for j in jobs)/3600))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='GLOBAL_RAW_BANK_SEAL',banks=16)),flush=True)



def prepare_smd(w,root):
    import copy
    root.mkdir(parents=True,exist_ok=False)
    base=w/'data/mechanism_validation_smd3_group_20261105_v1'
    template=read(w/'results'/EXPERIMENT/'PSM/TASK.json')
    for entity in SMD_ENTITIES:
        out=root/entity;out.mkdir();manifest=read(base/entity/'ROLE_MANIFEST.json')
        records={r['role']:r for r in manifest['roles']}
        task=copy.deepcopy(template);task.update(experiment=SMD_EXPERIMENT,domain='SMD',entity=entity,channels=38,
            channel_order=[f'original_array_column_{i}'for i in range(38)],timing_reference_domain='SWAT',
            channel_order_source={'description':'Official SMD original38 stored columns'},
            source_health='UNKNOWN_NATIVE_TRAIN; not independently certified healthy',
            authorization='Original DONOR value role is explicitly authorized as natural unlabeled U_FIT in current task',
            historical_exposure=True,blind_confirmation=False)
        task['roles']={};task['role_records']={};task['indexes']={}
        for role,oldrole in [('SOURCE_FIT','FIT'),('SOURCE_SELECT','SELECT'),('U_FIT','DONOR'),('CAL','CAL'),('QUERY','QUERY')]:
            r=records[oldrole];p=base/entity/f'{oldrole}.npz';assert sha(p)==r['sha256']
            task['roles'][role]=r['original_bounds']
            task['role_records'][role]=dict(relative=p.relative_to(w).as_posix(),sha256=sha(p),key='values',shape=r['expected_shape'],
                record_id=entity+'_'+r['parent_split'],parent_split=r['parent_split'],parent_sha256=r['parent_sha256'],storage_origin=r['original_start'])
        # Identity excludes role. Check all distinct permissions on the same physical record.
        rs=list(task['roles'])
        for i,a in enumerate(rs):
            for b in rs[i+1:]:
                if task['role_records'][a]['record_id']==task['role_records'][b]['record_id']:
                    assert max(task['roles'][a][0],task['roles'][b][0])>=min(task['roles'][a][1],task['roles'][b][1])
        task['data']={'source_train':task['role_records']['SOURCE_FIT'],'target_test':task['role_records']['QUERY']}
        with np.load(base/entity/'FIT.npz',allow_pickle=False)as z:fit=np.asarray(z['values'],dtype=np.float64)
        assert fit.shape==(14221,38)and np.isfinite(fit).all()
        mean=fit.mean(0);std=fit.std(0,ddof=0);std[std<1e-8]=1
        np.savez_compressed(out/'SCALER.npz',mean=mean,std=std);task['scaler_sha256']=sha(out/'SCALER.npz')
        eligibility=base/'audit_only'/f'{entity}_TARGET_ELIGIBILITY.npz'
        with np.load(eligibility,allow_pickle=False)as z:
            grid=z['CAL_arrived_target_original'].copy();query=z['QUERY_arrived_target_original'].copy()
        assert len(grid)>=512 and np.array_equal(query,np.arange(8957,23703))
        cal=grid[np.linspace(0,len(grid)-1,512,dtype=np.int64)]
        for role,idx in [('CAL',cal),('QUERY',query)]:
            lo,hi=task['roles'][role];assert np.all(idx>=lo+512)and np.all(idx<hi)
            p=out/f'{role}_INDEX.npy';np.save(p,idx)
            task['indexes'][role]=dict(relative=p.relative_to(w).as_posix(),sha256=sha(p),count=len(idx),first=int(idx[0]),last=int(idx[-1]),record_id=task['role_records'][role]['record_id'])
        source=pool_starts(task['roles']['SOURCE_FIT']);target=pool_starts(task['roles']['U_FIT'])
        plan=sampling_plan(len(source),len(target),seed_value=2025)
        np.savez_compressed(out/'PLAN.npz',**plan,source_starts=source,target_starts=target)
        task['replay_training']=copy.deepcopy(template['replay_training']);task['replay_training'].update(seed=2025,SOURCE_FIT_pool=len(source),U_FIT_pool=len(target),sampling_plan_sha256=sha(out/'PLAN.npz'))
        task['inherited_reference_not_executed']=['old source training/SELECT predictors','history512 predictor','source_noU_bank_root']
        for key in ['source_noU_bank_root','source_permutation_sha256','training','source_qualifications']:
            task.pop(key,None)
        dump(out/'TASK.json',task)
        dump(out/'POOL_EXPOSURE.json',dict(pool_counts={'SOURCE_FIT':len(source),'U_FIT':len(target)},
            streams={k:dict(exposure=int(v.size),unique_pool_windows=int(len(np.unique(v))),repeated_exposures=int(v.size-len(np.unique(v))))for k,v in plan.items()},labels_read=False))
        label=w/'data/event_smd_20261030_v1'/f'{entity}_test_label.txt'
        # Fingerprint only; training never opens this evaluation-only asset.
        dump(out/'EVALUATION_BINDING.json',dict(relative=label.relative_to(w).as_posix(),sha256=sha(label),format='txt',CAL_labels_available=False))
    dump(root/'CONFIG.json',dict(experiment=SMD_EXPERIMENT,smd_extension=True,domains=SMD_ENTITIES,arms=['R','R_target_mean'],seed=2025,
        models=4,freeze_name='ALL_MODELS_FROZEN.json',official_initial_checkpoint='528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c',
        CAL_rule='Original CAL_arrived_target_original fixed linspace512; q99 higher strict >',labels_read=False))


def smd_score_cache(w,root,domain,arm,role,part,parts):
    from .native_v_conditioning_runner import cache
    frozen=read(root/'ALL_MODELS_FROZEN.json');checkpoint=None
    if arm!='noU':
        selected=next(r for r in frozen['trainings']if r['domain']==domain and r['arm']==arm)
        checkpoint=w/selected['checkpoint_relative'];assert sha(checkpoint)==selected['checkpoint_sha256']
    cache(w,root/'readout'/arm,domain,role,part,parts,native_checkpoint=checkpoint,freeze_name='ALL_MODELS_FROZEN.json',
        metadata=dict(seed=2025,method=arm),canonical_native_V=True,normal_fixture_reuse=False)


def queue_smd(w,root):
    import shutil
    from .native_v_conditioning_queue import run_jobs
    from .native_v_conditioning_runner import merge_cache
    gates=run_jobs(w,root,'smd_engineering_gate',[(d,['gate','--domain',d,'--loss-profile','target_mean_source_copy'],20,600)for d in SMD_ENTITIES],'hybrid_tsad.native_v_replay')
    dump(root/'GATES_COMPLETED.json',dict(status='COMPLETED',jobs=gates))
    tasks=[]
    for d in SMD_ENTITIES:
        timeout=max(3600,int(read(root/d/'GATE.json')['estimated_train_seconds_per_arm']*1.8+600))
        for arm in ['R','R_target_mean']:
            profile='target_mean_source_copy'if arm=='R_target_mean'else'native'
            tasks.append((d+'_'+arm,['train','--domain',d,'--arm',arm,'--loss-profile',profile],20,timeout))
    trains=run_jobs(w,root,'smd_decoder_training',tasks,'hybrid_tsad.native_v_replay')
    rows=[]
    for d in SMD_ENTITIES:
        for arm in ['R','R_target_mean']:
            p=root/'train'/d/arm/'TRAIN_RECEIPT.json';rows.append(dict(read(p),train_receipt_relative=p.relative_to(w).as_posix(),train_receipt_sha256=sha(p)))
    dump(root/'ALL_MODELS_FROZEN.json',dict(status='COMPLETED',models=4,trainings=rows,jobs=trains,labels_read=False))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='ALL_FOUR_MODELS_FROZEN')),flush=True)
    for arm in ['noU','R','R_target_mean']:
        case=root/'readout'/arm;case.mkdir(parents=True,exist_ok=False)
        dump(case/'ALL_MODELS_FROZEN.json',dict(status='COMPLETED',parent_relative=(root/'ALL_MODELS_FROZEN.json').relative_to(w).as_posix(),parent_sha256=sha(root/'ALL_MODELS_FROZEN.json')))
        for d in SMD_ENTITIES:
            out=case/d;out.mkdir()
            for n in ['TASK.json','SCALER.npz']:shutil.copyfile(root/d/n,out/n)
    tasks=[(d+'_'+arm+'_'+role,['score_cache','--domain',d,'--arm',arm,'--role',role],30,3600)
        for d in SMD_ENTITIES for arm in ['noU','R','R_target_mean']for role in ['CAL','QUERY']]
    scores=run_jobs(w,root,'smd_native_readout',tasks,'hybrid_tsad.native_v_replay');banks=[]
    for d in SMD_ENTITIES:
        for arm in ['noU','R','R_target_mean']:
            for role in ['CAL','QUERY']:
                case=root/'readout'/arm;merge_cache(w,case,d,role,metadata=dict(seed=2025,method=arm))
                p=case/'cache'/d/role/'SEALED.json';r=read(p)
                banks.append(dict(domain=d,entity=d,method=arm,seed=2025,role=role,relative=r['relative'],sha256=r['sha256'],receipt_relative=p.relative_to(w).as_posix(),receipt_sha256=sha(p)))
    jobs=gates+trains+scores
    dump(root/'GLOBAL_RAW_BANK_SEAL.json',dict(status='COMPLETED',models=4,banks=banks,trainings=rows,jobs=jobs,no_labels_decoded=True,labels_read=False,actual_GPU_hours=sum(j['actual_GPU_seconds']for j in jobs)/3600))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='GLOBAL_RAW_BANK_SEAL',banks=12)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','gate','train','queue_gate','queue_train','score_cache','queue_score','prepare_repeats','queue_repeats','prepare_smd','queue_smd']);p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--domain',choices=['SWAT','PSM']+SMD_ENTITIES);p.add_argument('--arm',choices=['TT','SS','R','R_target_mean','noU'])
    p.add_argument('--loss-profile',choices=['native','target_mean_source_copy'],default='native')
    p.add_argument('--role',choices=['CAL','QUERY']);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1)
    a=p.parse_args();new=a.loss_profile=='target_mean_source_copy';root=a.output or a.workspace/'results'/(SMD_EXPERIMENT if a.action in ['prepare_smd','queue_smd']else REPEAT_EXPERIMENT if a.action in ['prepare_repeats','queue_repeats']else TARGET_MEAN_EXPERIMENT if new else EXPERIMENT)
    if a.action=='prepare_smd':prepare_smd(a.workspace,root)
    elif a.action=='queue_smd':queue_smd(a.workspace,root)
    elif a.action=='prepare_repeats':prepare_repeats(a.workspace,root)
    elif a.action=='queue_repeats':queue_repeats(a.workspace,root)
    elif a.action=='prepare':(prepare_target_mean if new else prepare)(a.workspace,root)
    elif a.action=='gate':(gate_target_mean if new else gate)(a.workspace,root,a.domain)
    elif a.action=='train':train(a.workspace,root,a.domain,a.arm,a.loss_profile)
    elif a.action=='score_cache':score_cache(a.workspace,root,a.domain,a.arm,a.role,a.part,a.parts)
    elif a.action=='queue_score':queue_score(a.workspace,root)
    else:
        from .native_v_conditioning_queue import run_jobs
        if a.action=='queue_gate':
            jobs=run_jobs(a.workspace,root,'engineering_gate',[(d,['gate','--domain',d]+(['--loss-profile',a.loss_profile]if new else[]),20,600)for d in ['SWAT','PSM']],'hybrid_tsad.native_v_replay')
            dump(root/'GATES_COMPLETED.json',dict(status='COMPLETED',jobs=jobs,labels_read=False))
        else:
            assert read(root/'GATES_COMPLETED.json')['status']=='COMPLETED'
            tasks=[]
            for d in ['SWAT','PSM']:
                seconds=read(root/d/'GATE.json')['estimated_train_seconds_per_arm']
                timeout=max(3600,int(seconds*1.8+600))
                tasks.extend((d+'_'+arm,['train','--domain',d,'--arm',arm]+(['--loss-profile',a.loss_profile]if new else[]),20,timeout)for arm in (['R_target_mean']if new else['TT','SS','R']))
            jobs=run_jobs(a.workspace,root,'decoder_training',tasks,'hybrid_tsad.native_v_replay')
            arms,freeze_name,count=matrix_spec(root)
            dump(root/freeze_name,dict(status='COMPLETED',models=count,jobs=jobs,labels_read=False,trainings=[
                read(root/'train'/d/arm/'TRAIN_RECEIPT.json')for d in ['SWAT','PSM']for arm in arms]))
            print(json.dumps(dict(event='STAGE_COMPLETED',stage=freeze_name.removesuffix('.json'))),flush=True)


if __name__=='__main__':main()
