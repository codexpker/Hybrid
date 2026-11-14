"""Matched normal-process R2 control: unchanged V1 losses/scoring, explicit policy.

Pure data/metadata paths do not import Torch. No query labels enter training.
The executor supplies resource ownership and SHA-pinned invocation wrappers.
"""
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np

NAMESPACE="R2_MATCHED_NORMAL_LEARNABILITY_20261105"
OPERATION="HYBRID_V1_R2_MATCHED_NORMAL_LEARNABILITY"
ARMS=("U","LAG_U","CROSS_OFF_LAG_U")
ALL_MODELS=("SOURCE",)+ARMS
U_PREFIXES=("fusion","tf_fusion_delta","ms_decoder","forecast_head")
HISTORY,HORIZON,N,BURN,SEED,ATTEMPTS,VALIDATE_EVERY=512,32,2560,4096,2025,1000,200
EVENT=(768,1536)
CASES=("CROSS_N","CROSS_LAG12_20","CROSS_BREAK","NO_DRIVER0_COUPLING","NORMAL_DRIFT")
ROUTE_ANCHORS=(512,640,767,768,1024,1279,1280,1408,1535,1536,1792,2047,2048,2304,2559)
GENERATOR_SHA="004b6c37e18b15c443f981d7a3249205e6113aa86b7844d47660211a4fcd1d10"
BUILDER_SHA="8a9756ae5dda71304ba45d91baddb9640c31121ba671bb615f6549fd571231e6"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def dump(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(clean(value),f,indent=2,allow_nan=False)
def append(path,value):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(clean(value),allow_nan=False)+'\n');f.flush()


def prefixes(arm):
    if arm not in ARMS:raise ValueError("Only the three fixed trained arms")
    return U_PREFIXES if arm=='U' else ('lag',)+U_PREFIXES


def fixed_plan(runs):
    """Only index draws; every normal role is a distinct independent full run."""
    by={r['role']:r for r in runs}
    if set(by)!={'FIT','SELECT','CAL'} or len({r['run'] for r in runs})!=3:
        raise ValueError('Three physically independent normal roles required')
    spec=dict(FIT=(SEED,ATTEMPTS,True),SELECT=(2718,32,False),CAL=(31415,512,False))
    plan=dict(history=HISTORY,horizon=HORIZON,seed=SEED)
    for role,(seed,count,repeated) in spec.items():
        targets=np.random.Generator(np.random.PCG64(seed)).choice(np.arange(HISTORY,N-HORIZON+1),size=count,replace=repeated)
        r=by[role]
        # The candidate grid retains a complete544 eligibility guard for every
        # role, but CAL consumes only the original arrived H+1 score footprint.
        # It neither reads nor scores the remaining unarrived31 targets.
        plan[role]=[dict(domain=r['domain'],entity=r['entity'],run=r['run'],role=role,index=int(t),footprint=[int(t)-HISTORY,int(t)+(1 if role=='CAL' else HORIZON)]) for t in targets]
    plan['CAL_grid_rule']='Targets512..2528 eligible under full544 guard; actual CAL footprint H+1, future GT not consumed'
    return plan


def build_query_inputs(*,n=N,burn=BURN,query_run=9204,independent_run=9205,role=NAMESPACE,event=EVENT):
    """Pure existing-generator composition; scaled geometries are fixtures only."""
    from . import controlled_mechanism as generator
    from . import controlled_wave as builder
    entries=[]
    for case,variant,span in (('CROSS_N','N',None),('CROSS_LAG12_20','A',list(event)),('CROSS_BREAK','BREAK',list(event))):
        entries.append(dict(id=case,family='cross_mixed_sign',variant=variant,C=3,seed_role=role,event=span,matched_interval=list(event)))
    protocol=dict(N=n,burn_in=burn,normal_fixed_delay=12,changed_delay=20,data_run_id=query_run,history=HISTORY,long_interval=list(event),cases=entries)
    cases,truth=builder.build_cases(protocol)
    independent,other=generator.generate_pair('cross_mixed_sign',run_id=independent_run,role=role,n=n,burn_in=burn,event=event,d0=12,d1=20)
    cases['NO_DRIVER0_COUPLING']=dict(values=np.column_stack((cases['CROSS_N']['values'][:,0],independent['N']['values'][:,1:])).astype(np.float32))
    cases['NORMAL_DRIFT']=dict(values=(cases['CROSS_N']['values'].astype(np.float64)+np.arange(n)[:,None]/(n-1)).astype(np.float32))
    return cases,truth,other


def prepare_data(output):
    """Generate once after the executor's explicit owned CPU dispatch."""
    from . import controlled_mechanism as generator
    from . import controlled_wave as builder
    if sha(generator.__file__)!=GENERATOR_SHA or sha(builder.__file__)!=BUILDER_SHA:
        raise PermissionError('Original 004b generator and8a builder required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);normal=out/'normal';query=out/'query';audit=out/'audit_only'
    for p in (normal,query,audit):p.mkdir()
    roles=[];seedbook={}
    def record(path,x,role,run):
        x=np.asarray(x,dtype=np.float32)
        if x.shape!=(N,3) or not np.isfinite(x).all():raise ValueError('Fixed finite C3 process')
        with path.open('xb') as f:np.savez_compressed(f,values=x,observed=np.ones(x.shape,bool))
        return dict(domain=NAMESPACE,entity='cross_mixed_sign',run=run,role=role,path=str(path.resolve()),
            format='npz',key='values',observed_key='observed',sha256=sha(path),expected_shape=[N,3],bounds=[0,N],original_start=0)
    for role,rid in (('FIT',9201),('SELECT',9202),('CAL',9203)):
        pair,truth=generator.generate_pair('cross_mixed_sign',run_id=rid,role=NAMESPACE,n=N,burn_in=BURN,event=EVENT,d0=12,d1=20)
        roles.append(record(normal/(role+'.npz'),pair['N']['values'],role,'normal_run'+str(rid)))
        seedbook[role]=dict(run_id=rid,seeds=truth['seeds'],variant='N_ONLY_NO_FAULT')
    cases,truth,other=build_query_inputs()
    query_roles=[record(query/(case+'.npz'),cases[case]['values'],'QUERY',case) for case in CASES]
    dump(out/'NORMAL_MANIFEST.json',dict(runs=roles));dump(out/'QUERY_MANIFEST.json',dict(runs=query_roles))
    dump(out/'COMMON_PLAN.json',fixed_plan(roles))
    dump(audit/'GENERATOR_AND_SEMANTICS.json',dict(namespace=NAMESPACE,normal_seedbook=seedbook,query_run=9204,
        query_seeds=truth['cross_mixed_sign/'+NAMESPACE]['seeds'],
        no_driver0_independent_latent_run=9205,no_driver0_seeds=other['seeds'],event=EVENT,
        normal_drift='All channels common ramp t/(N-1), amplitude1, declared normal operating variation',
        no_driver0='Independent source for response1/2: null mechanism control, not certified healthy real data',
        case_semantics={case:('NULL_MECHANISM_EVENT_NA_NOT_NORMAL_FPR' if case=='NO_DRIVER0_COUPLING' else
            ('DECLARED_GENERATED_NORMAL_FPR' if case in ('CROSS_N','NORMAL_DRIFT') else 'DECLARED_RELATION_CHANGE_EVENT')) for case in CASES},
        normal_FPR_cases=['CROSS_N','NORMAL_DRIFT'],null_event_metrics='NA_NO_TIMED_EVENT',
        SHAM=dict(alias='CROSS_N',extra_forward=False,independent_replication=False),
        all_roles_independent_run=True,H544_entire_footprint_required=True,labels_never_in_learner_manifest=True))
    with (audit/'QUERY_LABELS.npz').open('xb') as f:
        np.savez_compressed(f,**{case:(np.isin(np.arange(N),np.arange(*EVENT)).astype(np.int8)
            if case in ('CROSS_LAG12_20','CROSS_BREAK') else np.zeros(N,np.int8)) for case in CASES})
    result=dict(status='PREPARED_NOT_TRAINED',normal_runs=3,query_inputs=5,SHAM_alias=True,history=HISTORY,horizon=HORIZON,
        attempts=ATTEMPTS,seed=SEED,normal_manifest_sha256=sha(out/'NORMAL_MANIFEST.json'),
        query_manifest_sha256=sha(out/'QUERY_MANIFEST.json'),plan_sha256=sha(out/'COMMON_PLAN.json'),
        generator_sha256=GENERATOR_SHA,builder_sha256=BUILDER_SHA,query_labels_read=False,
        normal_FPR_cases=['CROSS_N','NORMAL_DRIFT'],null_control='NO_DRIVER0_COUPLING_NOT_CERTIFIED_NORMAL_EVENT_NA')
    dump(out/'RECEIPT.json',result);return result


def clone_for_arm(source,arm):
    model=copy.deepcopy(source);old=model.config
    cfg=replace(old,self_lag_enabled=True,cross_lag_enabled=arm!='CROSS_OFF_LAG_U')
    for module in model.modules():
        if getattr(module,'config',None) is old:module.config=cfg
    if model.config is not model.lag.config:raise ValueError('Root and lag configuration must agree')
    names=[];allow=prefixes(arm)
    for name,p in model.named_parameters():
        p.requires_grad_(name.split('.')[0] in allow)
        if p.requires_grad:names.append(name)
    return model,names


def training_mode(model,arm):
    model.train()
    for name,child in model.named_children():child.train(name in prefixes(arm))


def qk_signature(model):
    """CPU parameter observation; xy coefficient of the raw signed dot product."""
    q,k=model.lag.query,model.lag.key
    qw,kw=q.weight.detach().cpu().numpy(),k.weight.detach().cpu().numpy()
    return dict(query_weight=qw,query_bias=q.bias.detach().cpu().numpy(),key_weight=kw,key_bias=k.bias.detach().cpu().numpy(),
        raw_xy_coefficient=float(np.sum(qw.astype(np.float64)*kw.astype(np.float64))/np.sqrt(qw.size)),
        scope='Raw signed Q/K xy term; not a causal effect or guarantee of selected lag')


def state_change(before,after,arm):
    if before.keys()!=after.keys():raise ValueError('State names changed')
    changed=[n for n in before if before[n]!=after[n]]
    forbidden=[n for n in changed if n.split('.')[0] not in prefixes(arm)]
    return dict(changed=changed,forbidden=forbidden,passed=not forbidden,
        lag_changed=[n for n in changed if n.startswith('lag.')],
        lag_qk_changed=[n for n in changed if n.startswith(('lag.query.','lag.key.'))])


def lag_update_evidence(trace,changed,arm):
    """Training-exposure gate, never a claim that true lag relations were learnt."""
    groups=('query','key','value','self_out','cross_out')
    names=sorted({n for r in trace for n in r.get('lag_gradient_norms',{})}|{n for n in changed if n.startswith('lag.')})
    counts={n:sum(r.get('lag_gradient_norms',{}).get(n) is not None and r['lag_gradient_norms'][n]>0 for r in trace) for n in names}
    by={g:dict(parameters=[n for n in names if n.startswith('lag.'+g+'.')],
        nonzero_gradient_steps=sum(any(v is not None and v>0 for n,v in r.get('lag_gradient_norms',{}).items() if n.startswith('lag.'+g+'.')) for r in trace),
        changed_parameters=[n for n in changed if n.startswith('lag.'+g+'.')]) for g in groups}
    required=() if arm=='U' else groups if arm=='LAG_U' else groups[:-1]
    gate=(not any(n.startswith('lag.') for n in changed) if arm=='U' else
        all(by[g]['nonzero_gradient_steps']>0 and by[g]['changed_parameters'] for g in required))
    if arm=='CROSS_OFF_LAG_U' and by['cross_out']['changed_parameters']:gate=False
    return dict(per_parameter_nonzero_gradient_steps=counts,modules=by,required_modules=required,
        update_exposure_gate=bool(gate),cross_out_status='EXPECTED_NO_GRAD_OR_UPDATE' if arm=='CROSS_OFF_LAG_U' else 'ACTIVE' if arm=='LAG_U' else 'FROZEN',
        interpretation='QKV may update through Self in cross-off. Nonzero updates do not prove relation learning.')


def train_arm(source,fit_runs,select_runs,plan,*,arm,artifact_dir,technical=False):
    """Same optimizer/loss/masks/attempt clock; only the predeclared policy differs."""
    import torch
    from .stage2_u_matched import train_endpoint,resolve_u_plan,validate
    from .stage2_adaptation import state_identity
    from .training import passes,tensor_batch
    from .data import window
    from .objectives import ssl_objective
    from .readout import _rng,_rng_json
    out=Path(artifact_dir);out.mkdir(parents=True,exist_ok=False)
    count=2 if technical else ATTEMPTS;every=2 if technical else VALIDATE_EVERY
    if plan['seed']!=SEED or (plan['history'],plan['horizon'])!=(HISTORY,HORIZON) or len(plan['FIT'])!=count:
        raise ValueError('Fixed common finite plan required')
    model,names=clone_for_arm(source,arm);source_before=state_identity(source);before=state_identity(model)
    lag_initial={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('lag.')}
    torch.save(model.state_dict(),out/'initial.pt');dump(out/'STATE_BEFORE.json',before);dump(out/'QK_BEFORE.json',qk_signature(model))
    dump(out/'SOURCE_STATE_BEFORE.json',source_before);dump(out/'RNG_BEFORE.json',_rng_json(_rng()))
    dump(out/'ARM_CONFIG.json',dict(arm=arm,model_config=model.config.to_dict(),trainable_prefixes=prefixes(arm),
        source_weights_identical=before==source_before,self_enabled=True,cross_enabled=model.config.cross_lag_enabled))
    began=time.monotonic()
    if arm=='U':
        # Exact already-validated original normal adaptation path, not a new loop.
        model,report,opt=train_endpoint(model,fit_runs,select_runs,plan,history=HISTORY,horizon=HORIZON,seed=SEED,
            attempts=count,validate_every=every,lr=1e-4,weight_decay=.01,active_components=['r_tf','r_ms','r_pred'],
            scope='synthetic',method='plain',artifact_dir=out)
        report['lag_gradient_status']='FROZEN_BY_ORIGINAL_U_POLICY'
    else:
        points=resolve_u_plan(fit_runs,plan['FIT'],HISTORY,HORIZON,'FIT')
        resolve_u_plan(select_runs,plan['SELECT'],HISTORY,HORIZON,'SELECT')
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        trace=[];path=out/'training_trace.jsonl';path.open('x',encoding='utf-8').close()
        for attempt,(run,t) in enumerate(points):
            training_mode(model,arm);opt.zero_grad(set_to_none=True)
            b=tensor_batch(window(run,t,HISTORY,HORIZON),next(model.parameters()).device)
            append(path,dict(attempt=attempt,status='ATTEMPT_STARTED',index=b['index'],footprint=b['footprint'],actual_mask_seed=SEED+attempt))
            f,r=passes(model,b,SEED+attempt);obj=ssl_objective(f,r,b,'huber',['r_tf','r_ms','r_pred'])
            row=dict(attempt=attempt,index=b['index'],footprint=b['footprint'],mask_seed=SEED+attempt,updated=False,learning_rate=opt.param_groups[0]['lr'])
            if bool(obj['valid_windows'].all()):
                if not bool(torch.isfinite(obj['total'])):raise ValueError('Nonfinite fixed SSL objective')
                obj['total'].backward()
                grads={n:float(p.grad.detach().norm()) if p.grad is not None else None for n,p in model.named_parameters() if n.startswith('lag.')}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                row.update(updated=True,loss=float(obj['total'].detach()),gradient_norm=float(norm),lag_gradient_norms=grads,
                    parts={k:float(v.detach()) for k,v in obj['parts'].items()},counts={k:v.detach().cpu().tolist() for k,v in obj['valid_counts'].items()})
            else:row['reason']='NO_VALID_GT_NO_REDRAW'
            if (attempt+1)%every==0 or attempt==count-1:
                row['SELECT_diagnostic']=validate(model,select_runs,plan['SELECT'],HISTORY,HORIZON,SEED,['r_tf','r_ms','r_pred'])
            trace.append(row);append(path,dict(row,status='ATTEMPT_COMPLETED'))
        report=dict(attempts=count,updates=sum(r['updated'] for r in trace),endpoint_attempt=count-1,
            selection='FIXED_ATTEMPT_ENDPOINT_NO_BEST_CHECKPOINT',updated_names=names,trace=trace,
            lag_nonzero_gradient_steps=sum(any(v is not None and v>0 for v in r.get('lag_gradient_norms',{}).values()) for r in trace))
    after=state_identity(model);delta=state_change(before,after,arm)
    actual_modes={name:child.training for name,child in model.named_children()}
    if any(flag!=(name in prefixes(arm)) for name,flag in actual_modes.items()):
        raise ValueError('Actual final train/frozen child modes do not match the arm')
    dump(out/'MODE_POLICY.json',dict(root_training=model.training,children=actual_modes,
        trainable_parameters=[n for n,p in model.named_parameters() if p.requires_grad],
        note='Recorded after last SELECT, whose original context restores every prior module mode'))
    # Buffer mutations are never an allowed trainable-parameter update.
    parameter_names=set(dict(model.named_parameters()));buffer_changes=[n for n in delta['changed'] if n not in parameter_names]
    if not delta['passed'] or buffer_changes or state_identity(source)!=source_before:
        raise ValueError('Frozen parameter, buffer or source mutation')
    lag_valid=(not delta['lag_changed'] if arm=='U' else bool(delta['lag_changed']) and report['lag_nonzero_gradient_steps']>0)
    detailed_lag=lag_update_evidence(report['trace'],delta['changed'],arm)
    lag_deltas={n:float((p.detach()-lag_initial[n]).abs().max()) for n,p in model.named_parameters() if n.startswith('lag.')}
    torch.save(model.state_dict(),out/'adapted.pt')
    torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
        cuda_rng=[torch.cuda.get_rng_state(d) for d in range(torch.cuda.device_count())],numpy_rng=np.random.get_state(),
        resume_implemented=False),out/'final_training_state.pt')
    dump(out/'QK_AFTER.json',qk_signature(model));dump(out/'STATE_AFTER.json',after)
    dump(out/'SOURCE_STATE_AFTER.json',state_identity(source));dump(out/'RNG_AFTER.json',_rng_json(_rng()))
    report.update(arm=arm,policy_prefixes=prefixes(arm),state_change=delta,lag_policy_verified=lag_valid,
        lag_update_evidence=detailed_lag,lag_parameter_max_abs_changes=lag_deltas,
        source_unchanged=True,seconds=time.monotonic()-began,learning_rate=1e-4,weight_decay=.01,
        same_budget=not technical,technical_discard_only=technical,optimizer='AdamW',warmup_steps=0)
    dump(out/'training.json',report)
    dump(out/'RECEIPT.json',dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',arm=arm,attempts=count,updates=report['updates'],
        scientific_status='VALID_FINITE_ENDPOINT' if report['updates']==count and lag_valid and detailed_lag['update_exposure_gate'] else 'INCOMPLETE_NO_REDRAW_NO_EXTENSION',
        endpoint_attempt=count-1,lag_policy_verified=lag_valid,lag_qk_changed=delta['lag_qk_changed'],
        lag_update_evidence=detailed_lag,lag_parameter_max_abs_changes=lag_deltas,
        checkpoint_sha256=sha(out/'adapted.pt'),query_or_event_labels_read=False,technical_discard_only=technical))
    return model,report,opt


def load_original_controls(path):
    expected='5f08f9cbd033f3a97e8a7032aec2e52e941b207919339bf93cf0369046231666'
    if sha(path)!=expected:raise PermissionError('Original AR_SELF/ARX12 lambda1 math required')
    spec=importlib.util.spec_from_file_location('_r2_original_cpu_controls',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fit_controls(original,values,fit_times):
    """Only the two predeclared original h1 models, not ARX20 or a lag search."""
    x=np.asarray(values,dtype=np.float64);t=np.asarray(fit_times,dtype=np.int64)
    if x.shape!=(N,3) or len(t)!=ATTEMPTS or t.min()<HISTORY or t.max()+HORIZON>N:
        raise ValueError('The identical1000 FIT supervision anchors required')
    return {name:{c:original.fit(x,t,c,lag,1.) for c in (1,2)} for name,lag in [('AR_SELF',None),('ARX12',12)]}


def predict_controls(original,models,values,times):
    x=np.asarray(values,dtype=np.float64);t=np.asarray(times,dtype=np.int64)
    if len(t)==0 or t.min()<HISTORY or t.max()>=len(x):raise ValueError('Arrived original h1 axis')
    return {name:np.column_stack([original.predict(by[c],x,t,c,None if name=='AR_SELF' else 12) for c in (1,2)]) for name,by in models.items()}
