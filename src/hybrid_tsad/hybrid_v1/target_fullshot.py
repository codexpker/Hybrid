"""Fixed full-target-permission diagnostic; unchanged V1 objective and score."""
import copy,json,hashlib
from pathlib import Path
import numpy as np

def load_roles(manifest,roles):
    from .data import Run
    entries=json.loads(Path(manifest).read_text())['runs'];out=[]
    for i,a in enumerate(entries):
        for b in entries[i+1:]:
            if a['run']==b['run']:
                assert a['original_start']+a['expected_shape'][0]<=b['original_start'] or b['original_start']+b['expected_shape'][0]<=a['original_start']
        if a['role'] not in roles:continue
        p=Path(a['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==a['sha256']
        with np.load(p) as z:
            assert set(z.files)=={'values','observed'}
            out.append(Run(a['domain'],a['entity'],a['run'],a['role'],z['values'].copy(),z['observed'].copy(),a['original_start']))
    assert {r.role for r in out}==set(roles)
    return out

def train(source,runs,select,plan,out,technical=False):
    import torch
    from . import stage2_u_matched as u,cli
    from .training import tensor_batch,passes
    from .data import window
    from .objectives import ssl_objective
    from .stage2_adaptation import state_identity
    out=Path(out);model=copy.deepcopy(source);model.train()
    for p in model.parameters():p.requires_grad_(True)
    assert all(m.training for m in model.modules())
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    resolved=u.resolve_u_plan(runs,plan['FIT'][:2] if technical else plan['FIT'],512,32,'FIT')
    best=float('inf');selected=None;trace=[];initial=state_identity(source)
    for attempt,(run,t) in enumerate(resolved):
        model.train();optimizer.zero_grad(set_to_none=True)
        batch=tensor_batch(window(run,t,512,32),next(model.parameters()).device)
        f,r=passes(model,batch,2025+attempt);obj=ssl_objective(f,r,batch,'huber',None)
        assert bool(obj['valid_windows'].all()) and bool(torch.isfinite(obj['total']))
        obj['total'].backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        row=dict(attempt=attempt,updated=True,index=run.start+t,mask_seed=2025+attempt,loss=float(obj['total'].detach()),gradient_norm=float(norm),footprint=batch['footprint'])
        if technical or (attempt+1)%100==0:
            cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(0).clone() if torch.cuda.is_available() and next(model.parameters()).is_cuda else None
            val=u.validate(model,select,plan['SELECT'][:2] if technical else plan['SELECT'],512,32,2025)
            assert torch.equal(cpu,torch.get_rng_state()) and (cuda is None or torch.equal(cuda,torch.cuda.get_rng_state(0)))
            assert all(m.training for m in model.modules())
            row['SELECT_ssl']=val
            if val<best:
                best=val;selected=attempt;cli.save_model(out/'selected.pt',model)
        with (out/'training_trace.jsonl').open('a') as s:s.write(json.dumps(row)+'\n')
        trace.append(row)
    assert state_identity(source)==initial and len(trace)==(2 if technical else 8000)
    cli.save_model(out/'endpoint.pt',model);torch.save(optimizer.state_dict(),out/'optimizer.pt')
    report=dict(updates=len(trace),attempts=len(trace),selected_attempt=selected,SELECT_min=best,source_unchanged=True,all_modules_train=True,all_parameters_requires_grad=True,query_labels_read=False)
    cli.save_json(out/'training.json',report)
    return cli.load_model(dict(model.config.to_dict()),out/'selected.pt') if False else report
