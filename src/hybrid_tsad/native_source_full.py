"""Fixed original source objectives with all native parameters trainable."""
import sys
import time
from pathlib import Path
import numpy as np
import torch
from . import native_event_rank_training as old
from .native_position_aligned_training import datasets, PRIOR, PLAN_SHA
from .native_target_normal import parameter_groups
from .native_v_replay import decoder_model, optimizer
from .native_v_conditioning import state_hashes
from .native_v_conditioning_runner import setup,seed,read,dump,sha

OUT=Path("results/native_source_full_20261125_v1")
ARMS=("copy","point_weighted","point_weighted_e")


def check(w):
    setup();seed(2025)
    plan=old.load_plan(w/PRIOR)
    assert sha(w/PRIOR/"PLAN.npz")==PLAN_SHA
    fit,_=datasets(w)
    m=decoder_model(w,"cuda")
    x=next(old.windows(fit[1],plan["swat_fit"][0,:1],"cuda"))
    before=state_hashes(m)
    with torch.no_grad():
        reference={e:tuple(float(v) for v in old.native_endpoint_and_copy_loss(m,x,e)) for e in ("V","E")}
    m.requires_grad_(True)
    groups=parameter_groups(m)
    params=list(m.parameters())
    rows={}
    for e in ("V","E"):
        value,loss=old.native_endpoint_and_copy_loss(m,x,e)
        assert (float(value),float(loss))==reference[e]
        gradients=torch.autograd.grad(loss,params,allow_unused=True)
        rows["copy_"+e]={g:sum(float(grad.norm()) for p,grad in zip(params,gradients)
            if grad is not None and any(p is q for _,q in entries)) for g,entries in groups.items()}
        positive=torch.stack([v for v,_ in old.native_values(m,fit[0],plan["positive_fit"][0],"cuda",e)]).reshape(5,3)
        normal=torch.stack([v for v,_ in old.native_values(m,fit[0],plan["normal_fit"][0],"cuda",e)]).reshape(5,3)
        path=PRIOR if e=="V" else Path("results/native_e_vs_v_20261123_v1")
        scale=read(w/path/"SCALE.json")["value"]
        pair=old.event_pair_loss(positive,normal,scale,phase_sizes=plan["stratum_sizes"])
        grads=torch.autograd.grad(pair,params,allow_unused=True)
        rows["pair_"+e]={g:dict(norm=sum(float(grad.norm()) for p,grad in zip(params,grads)
            if grad is not None and any(p is q for _,q in entries)),reachable=sum(grad is not None for p,grad in zip(params,grads)
            if any(p is q for _,q in entries))) for g,entries in groups.items()}
        assert all(np.isfinite(r["norm"]) for r in rows["pair_"+e].values())
    assert before==state_hashes(m) and all(not a.training for a in m.modules())
    root=w/OUT;root.mkdir(parents=True,exist_ok=True)
    dump(root/"CHECK.json",dict(native_values=reference,gradients=rows,all_native_params=True,
        parameter_counts={g:sum(p.numel() for _,p in ps) for g,ps in groups.items()},
        plan_sha256=PLAN_SHA,test_read=False))


def train(w,arm):
    assert arm in ARMS and (w/OUT/"CHECK.json").exists()
    tick=time.monotonic();setup();seed(2025)
    plan=old.load_plan(w/PRIOR)
    assert sha(w/PRIOR/"PLAN.npz")==PLAN_SHA
    fit,select=datasets(w)
    ev="E" if arm.endswith("_e") else "V"
    source=PRIOR if ev=="V" else Path("results/native_e_vs_v_20261123_v1")
    record=read(w/source/"SCALE.json")
    assert sha(w/source/record["bank_file"])==record["bank_sha256"]
    m=decoder_model(w,"cuda");m.requires_grad_(True)
    before=state_hashes(m);opt=optimizer(m)
    rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state()
    root=w/OUT/arm;root.mkdir()
    trace=[];selection=[];best=None
    for step in range(241):
        if step:
            trace.append(old.train_step(m,opt,plan,fit,record["value"],arm,"cuda",step,ev))
        if step in old.SELECT_STEPS:
            val=old.validate(m,plan,select,record["value"],"cuda",step,ev)
            selection.append(val)
            if best is None or val["risk"]<best["risk"]:
                best=val;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            print(arm,step,val["risk"],best["update"],flush=True)
    after=state_hashes(m)
    groups=parameter_groups(m)
    changes={g:sum(before[k]!=after[k] for k,_ in ps) for g,ps in groups.items()}
    assert all(not v.training for v in m.modules()) and torch.equal(rng,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
    assert all(before[k]==after[k] for k,_ in m.named_buffers())
    torch.save(dict(model=state,update=best["update"],arm=arm),root/"selected.pt")
    dump(root/"TRAIN_TRACE.json",trace)
    dump(root/"SELECTION.json",dict(arm=arm,selected_step=best["update"],scale=record["value"],updates=240,
        source_select=selection,selected_checkpoint_sha256=sha(root/"selected.pt"),plan_sha256=PLAN_SHA,
        native_parameter_changes=changes,seconds=time.monotonic()-tick,peak_reserved=torch.cuda.max_memory_reserved(),
        nominal_exposures=7680,pair_exposures=0 if arm=="copy" else 3600,selected_nominal_exposures=best["update"]*32,
        selected_pair_exposures=0 if arm=="copy" else best["update"]*15,target_read=False))


if __name__=="__main__":
    w=Path.cwd()
    check(w) if sys.argv[1]=="check" else train(w,sys.argv[1])
