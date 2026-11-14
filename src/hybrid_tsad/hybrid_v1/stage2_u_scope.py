"""Matched target U endpoint: only HEAD versus FULL requires-grad scope differs.

The accepted U200 adapter remains immutable. Both scopes retain its exact
heterogeneous training-mode policy, normal SSL, attempt clock and diagnostics.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from . import stage2_u_matched as u

SCOPES=('HEAD','FULL')
MODE_POLICY='root_train_allowed_children_train_others_eval'

def scope_parameters(model,scope):
    if scope not in SCOPES:raise ValueError('Only HEAD and FULL')
    for name,p in model.named_parameters():
        p.requires_grad_(scope=='FULL' or name.split('.')[0] in u.ALLOWED)
    return sorted(n for n,p in model.named_parameters() if p.requires_grad)

def scope_state_difference(before,after,parameters,scope):
    if set(before)!=set(after):raise ValueError('State keys changed')
    allowed={n for n in parameters if scope=='FULL' or n.split('.')[0] in u.ALLOWED}
    changed=sorted(n for n in before if before[n]!=after[n])
    buffers=sorted(set(changed)-set(parameters))
    forbidden=sorted(set(changed)-allowed)
    return dict(PASS=not forbidden,changed_parameters=sorted(set(changed)&set(parameters)),
                changed_buffers=buffers,forbidden_changes=forbidden,allowed_parameter_names=sorted(allowed))

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def train_endpoint(source,fit_runs,select_runs,plan,*,scope,artifact_dir,technical=False):
    import torch
    from .training import tensor_batch,passes
    from .data import window
    from .objectives import ssl_objective
    from .stage2_adaptation import state_identity
    if scope not in SCOPES:raise ValueError('One paired parameter-scope contrast')
    attempts=2 if technical else 2000
    if (plan['history'],plan['horizon'],plan['seed'])!=(512,32,2025) or len(plan['FIT'])!=attempts:
        raise ValueError('Exact fixed attempt plan required')
    if len(plan['SELECT'])!=(2 if technical else 32):raise ValueError('SELECT diagnostic scope')
    u.no_labels(plan)
    resolved=u.resolve_u_plan(fit_runs,plan['FIT'],512,32,'FIT')
    u.resolve_u_plan(select_runs,plan['SELECT'],512,32,'SELECT')
    model=copy.deepcopy(source);initial=state_identity(source)
    allowed=scope_parameters(model,scope);u.adaptation_mode(model)
    modes={name:m.training for name,m in model.named_modules()}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    out=Path(artifact_dir);out.mkdir(parents=True,exist_ok=True)
    tracefile=out/'training_trace.jsonl';tracefile.open('x',encoding='utf-8').close()
    trace=[];gradient={n:dict(nonzero_steps=0,None_steps=0,zero_steps=0) for n,p in model.named_parameters() if p.requires_grad}
    initial_rng=dict(cpu=hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                     cuda=[hashlib.sha256(torch.cuda.get_rng_state(i).cpu().numpy().tobytes()).hexdigest() for i in u.rng_devices(model)])
    for attempt,(run,t) in enumerate(resolved):
        u.adaptation_mode(model)
        if {n:m.training for n,m in model.named_modules()}!=modes:raise ValueError('Module mode drift')
        optimizer.zero_grad(set_to_none=True);mask_seed=2025+attempt
        batch=tensor_batch(window(run,t,512,32),next(model.parameters()).device)
        row=dict(attempt=attempt,actual_mask_seed=mask_seed,domain=run.domain,entity=run.entity,run=run.run,
                 index=run.start+t,footprint=batch['footprint'],updated=False)
        u.append_trace(tracefile,dict(row,status='ATTEMPT_STARTED'))
        f,r=passes(model,batch,mask_seed);obj=ssl_objective(f,r,batch,'huber',None)
        if attempt==0:
            arrays={k:v.detach().cpu().numpy() for k,v in dict(prediction=f['prediction'],tf=r['tf_reconstruction'],ms=r['ms_reconstruction'],
                 current=batch['current_target'],observed=batch['current_observed'],hidden=r['hidden_mask'],mean=f['stats']['mean'],scale=f['stats']['scale']).items()}
            with (out/'FIRST_FORWARD.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
        if not bool(obj['valid_windows'].all()):
            u.append_trace(tracefile,dict(row,status='INVALID_NO_REDRAW',reason='NO_VALID_GT_NO_REDRAW'))
            raise ValueError('A fixed attempt was invalid; stop without replacement')
        if not bool(torch.isfinite(obj['total'])):raise ValueError('Nonfinite U objective')
        obj['total'].backward()
        for name,p in model.named_parameters():
            if not p.requires_grad:continue
            count=gradient[name]
            if p.grad is None:count['None_steps']+=1
            else:
                if not bool(torch.isfinite(p.grad).all()):raise ValueError('Nonfinite gradient '+name)
                count['nonzero_steps' if bool(p.grad.ne(0).any()) else 'zero_steps']+=1
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();row.update(updated=True,loss=float(obj['total'].detach()),gradient_norm=float(norm))
        if (attempt+1)%50==0 or attempt==attempts-1:
            row['SELECT_diagnostic']=u.validate(model,select_runs,plan['SELECT'],512,32,2025,None)
        trace.append(row);u.append_trace(tracefile,dict(row,status='ATTEMPT_COMPLETED'))
    difference=scope_state_difference(initial,state_identity(model),set(dict(model.named_parameters())),scope)
    if not difference['PASS'] or difference['changed_buffers'] or state_identity(source)!=initial:
        raise ValueError('Forbidden parameter/buffer/source mutation')
    if len(trace)!=attempts or not all(v['updated'] for v in trace):raise ValueError('Incomplete effective updates')
    if not difference['changed_parameters']:raise ValueError('No parameter changed')
    if scope=='FULL' and not any(n.split('.')[0] not in u.ALLOWED for n in difference['changed_parameters']):
        raise ValueError('FULL did not update any non-head parameter')
    return model,dict(scope=scope,attempts=attempts,updates=attempts,scientific_training_status='VALID_ALL_ATTEMPTS_UPDATED',
      selection='FIXED_ATTEMPT_ENDPOINT_NO_BEST_CHECKPOINT',endpoint_attempt=attempts-1,
      updated_names=allowed,actual_gradient_exposure=gradient,unused_trainable_names=[n for n,v in gradient.items() if v['None_steps']==attempts],
      trace=trace,state_difference=difference,source_unchanged=True,mode_policy=MODE_POLICY,module_modes=modes,
      initial_torch_rng=initial_rng,sample_mask_digest=digest([dict(axis=[x[k] for k in ('domain','entity','run','index')],seed=x['actual_mask_seed']) for x in trace]),
      SELECT_sequence_digest=digest(plan['SELECT']),FIT_sequence_digest=digest(plan['FIT'])),optimizer
