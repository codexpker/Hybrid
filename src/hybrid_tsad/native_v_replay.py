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


def accumulate(model,windows,device,divisor):
    losses=[];orders=[];hook=model.net.register_forward_hook(lambda m,a,o:orders.append(float(o.detach().reshape(-1)[0])))
    try:
        for x in windows:
            with fixed_native_rng(device):_,outputs=model(x,norm=0,mask_mode='c',copies=10)
            assert outputs.shape==(10,1,100,x.shape[-1])
            loss=(outputs-x[None]).square().mean();assert torch.isfinite(loss)
            (loss/divisor).backward();losses.append(float(loss.detach()))
    finally:hook.remove()
    assert losses and all(p.grad is None or torch.isfinite(p.grad).all()for p in model.parameters())
    assert all(p.grad is None for p in model.parameters()if not p.requires_grad)
    return float(np.mean(losses)),orders


def role_arrays(w,root,domain,roles):
    task=read(root/domain/'TASK.json');arrays={}
    for role in roles:arrays[role]=role_values(w,root/domain,task,role)
    return task,arrays


def train(w,root,domain,arm):
    assert read(root/domain/'GATE.json')['status']=='PASS_ENGINEERING_ONLY'
    out=root/'train'/domain/arm;out.mkdir(parents=True,exist_ok=False);start=time.monotonic();device=setup(20)
    roles=['SOURCE_FIT']if arm=='SS'else['U_FIT']if arm=='TT'else['SOURCE_FIT','U_FIT']
    task,arrays=role_arrays(w,root,domain,roles);p=root/domain/'PLAN.npz';assert sha(p)==task['replay_training']['sampling_plan_sha256']
    with np.load(p)as z:plan={k:z[k].copy()for k in z.files}
    seed();model=decoder_model(w,device);initial=state_hashes(model);opt=optimizer(model);trace=[]
    for step in range(2000):
        lr=1e-4*min((step+1)/200,1.)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True);losses=[];orders=[]
        for role,starts in paired_starts(plan,plan['source_starts'],plan['target_starts'],arm,step):
            values,_,lo,_=arrays[role]
            loss,order=accumulate(model,input_windows(values,lo,starts,device),device,64);losses.append(loss);orders+=order
        assert len(orders)==64
        gradient=float(torch.stack([p.grad.norm()for p in model.decoder.parameters()if p.grad is not None]).norm())
        assert np.isfinite(gradient);opt.step()
        trace.append(dict(update=step+1,loss=float(np.mean(losses)),gradient_norm=gradient,lr=lr,
            per_window_order_sha256=hashlib.sha256(np.asarray(orders,np.float32).tobytes()).hexdigest()))
        if(step+1)%200==0:print(json.dumps(dict(event='TRAIN_PROGRESS',domain=domain,arm=arm,update=step+1,elapsed=time.monotonic()-start)),flush=True)
    final=state_hashes(model);frozen={k:v for k,v in initial.items()if not k.startswith('decoder.')}
    assert all(final[k]==v for k,v in frozen.items())and any(final[k]!=initial[k]for k in initial if k.startswith('decoder.'))
    assert all(not m.training for m in model.modules())
    checkpoint=out/'ENDPOINT.pth';torch.save(dict(model=model.state_dict(),update=2000,domain=domain,arm=arm),checkpoint)
    torch.save(opt.state_dict(),out/'OPTIMIZER.pth');dump(out/'TRACE.json',trace)
    dump(out/'TRAIN_RECEIPT.json',dict(status='COMPLETED',domain=domain,arm=arm,seed=2025,updates=2000,logical_batch=64,
        B1_forward_count=128000,exposures=128000,roles_read=roles,labels_read=False,CAL_QUERY_opened=False,
        checkpoint_relative=str(checkpoint.relative_to(w)),checkpoint_sha256=sha(checkpoint),optimizer_sha256=sha(out/'OPTIMIZER.pth'),
        plan_sha256=sha(p),scaler_sha256=task['scaler_sha256'],initial_state_hashes=initial,final_state_hashes=final,
        frozen_parameters_buffers_unchanged=True,all_modules_eval=True,trainable_names=[n for n,p in model.named_parameters()if p.requires_grad],
        trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),
        elapsed_GPU_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
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


def scoring_cases(w,root):
    import shutil
    frozen=read(root/'ALL_SIX_MODELS_FROZEN.json');assert frozen['status']=='COMPLETED'and len(frozen['trainings'])==6
    for arm in ['TT','SS','R']:
        case=root/'readout'/arm;case.mkdir(parents=True,exist_ok=False)
        dump(case/'ALL_SIX_MODELS_FROZEN.json',dict(status='COMPLETED',parent_relative=str((root/'ALL_SIX_MODELS_FROZEN.json').relative_to(w)),
            parent_sha256=sha(root/'ALL_SIX_MODELS_FROZEN.json')))
        for domain in ['SWAT','PSM']:
            target=case/domain;target.mkdir();shutil.copyfile(root/domain/'TASK.json',target/'TASK.json');shutil.copyfile(root/domain/'SCALER.npz',target/'SCALER.npz')
            receipt=root/'train'/domain/arm/'TRAIN_RECEIPT.json';selected=read(receipt)
            dump(target/'SCORE_BINDING.json',dict(domain=domain,arm=arm,checkpoint_relative=selected['checkpoint_relative'],
                actual_endpoint_sha256=selected['checkpoint_sha256'],train_receipt_relative=str(receipt.relative_to(w)),train_receipt_sha256=sha(receipt),
                actual_feature_window=100,feature_end='t inclusive',legacy512_predictor_not_executed=True))


def score_cache(w,root,domain,arm,role,part,parts):
    from .native_v_conditioning_runner import cache
    frozen=read(root/'ALL_SIX_MODELS_FROZEN.json')
    selected=next(r for r in frozen['trainings']if r['domain']==domain and r['arm']==arm)
    checkpoint=w/selected['checkpoint_relative'];assert sha(checkpoint)==selected['checkpoint_sha256']
    cache(w,root/'readout'/arm,domain,role,part,parts,native_checkpoint=checkpoint)


def queue_score(w,root):
    from .native_v_conditioning_queue import run_jobs
    from .native_v_conditioning_runner import merge_cache
    scoring_cases(w,root)
    tasks=[(domain+'_'+arm+'_'+role+'_'+str(part),['score_cache','--domain',domain,'--arm',arm,'--role',role,'--part',str(part),'--parts',str(parts)],30,7200)
        for arm in ['TT','SS','R']for domain in ['SWAT','PSM']for role,parts in [('CAL',1),('QUERY',4)]for part in range(parts)]
    jobs=run_jobs(w,root,'native_readout',tasks,'hybrid_tsad.native_v_replay');banks=[];reused=[]
    for domain in ['SWAT','PSM']:
        for arm in ['TT','SS','R']:
            for role in ['CAL','QUERY']:
                case=root/'readout'/arm;merge_cache(w,case,domain,role);p=case/'cache'/domain/role/'SEALED.json';sealed=read(p)
                banks.append(dict(domain=domain,method=arm,role=role,relative=sealed['relative'],sha256=sealed['sha256'],
                    receipt_relative=str(p.relative_to(w)),receipt_sha256=sha(p)))
        for role in ['CAL','QUERY']:
            p=w/'results'/OLD/'banks'/domain/'A'/role/'RECEIPT.json';r=read(p);assert sha(w/r['relative'])==r['sha256']
            reused.append(dict(domain=domain,method='noU',role=role,relative=r['relative'],sha256=r['sha256'],receipt_relative=str(p.relative_to(w)),receipt_sha256=sha(p)))
    alljobs=read(root/'GATES_COMPLETED.json')['jobs']+read(root/'ALL_SIX_MODELS_FROZEN.json')['jobs']+jobs
    trainings=[]
    for domain in ['SWAT','PSM']:
        for arm in ['TT','SS','R']:
            p=root/'train'/domain/arm/'TRAIN_RECEIPT.json';row=read(p)
            trainings.append(dict(row,train_receipt_relative=str(p.relative_to(w)),train_receipt_sha256=sha(p)))
    dump(root/'GLOBAL_RAW_BANK_SEAL.json',dict(status='COMPLETED',no_labels_decoded=True,labels_read=False,banks=banks,noU_reused=reused,
        trainings=trainings,jobs=alljobs,actual_GPU_hours=sum(j['actual_GPU_seconds']for j in alljobs)/3600))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='GLOBAL_RAW_BANK_SEAL',new_banks=12,reused_noU_banks=4)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','gate','train','queue_gate','queue_train','score_cache','queue_score']);p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--domain',choices=['SWAT','PSM']);p.add_argument('--arm',choices=['TT','SS','R'])
    p.add_argument('--role',choices=['CAL','QUERY']);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1)
    a=p.parse_args();root=a.output or a.workspace/'results'/EXPERIMENT
    if a.action=='prepare':prepare(a.workspace,root)
    elif a.action=='gate':gate(a.workspace,root,a.domain)
    elif a.action=='train':train(a.workspace,root,a.domain,a.arm)
    elif a.action=='score_cache':score_cache(a.workspace,root,a.domain,a.arm,a.role,a.part,a.parts)
    elif a.action=='queue_score':queue_score(a.workspace,root)
    else:
        from .native_v_conditioning_queue import run_jobs
        if a.action=='queue_gate':
            jobs=run_jobs(a.workspace,root,'engineering_gate',[(d,['gate','--domain',d],20,600)for d in ['SWAT','PSM']],'hybrid_tsad.native_v_replay')
            dump(root/'GATES_COMPLETED.json',dict(status='COMPLETED',jobs=jobs,labels_read=False))
        else:
            assert read(root/'GATES_COMPLETED.json')['status']=='COMPLETED'
            tasks=[]
            for d in ['SWAT','PSM']:
                seconds=read(root/d/'GATE.json')['estimated_train_seconds_per_arm']
                timeout=max(3600,int(seconds*1.8+600))
                tasks.extend((d+'_'+arm,['train','--domain',d,'--arm',arm],20,timeout)for arm in ['TT','SS','R'])
            jobs=run_jobs(a.workspace,root,'decoder_training',tasks,'hybrid_tsad.native_v_replay')
            dump(root/'ALL_SIX_MODELS_FROZEN.json',dict(status='COMPLETED',jobs=jobs,labels_read=False,trainings=[
                read(root/'train'/d/arm/'TRAIN_RECEIPT.json')for d in ['SWAT','PSM']for arm in ['TT','SS','R']]))
            print(json.dumps(dict(event='STAGE_COMPLETED',stage='ALL_SIX_MODELS_FROZEN')),flush=True)


if __name__=='__main__':main()
