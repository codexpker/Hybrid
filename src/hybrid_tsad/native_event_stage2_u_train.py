"""One frozen-source checkpoint, two paired mixed adaptations and one shared source control."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from .native_event_rank import event_pair_loss, native_endpoint_and_copy_loss
from .native_event_rank_training import load_smd, nominal_swat_input, source_roles
from .native_pair_rank_data import require, SMD_ENTITIES, SMD_BINDING, SMD_SHA, task_binding
from .native_v_conditioning import sha, state_hashes
from .native_v_conditioning_queue import run_jobs
from .native_v_conditioning_runner import dump, read, role_values, seed, setup
from .native_v_replay import decoder_model, input_windows, optimizer, pool_starts

MODULE = "hybrid_tsad.native_event_stage2_u_train"
OUTPUT = Path("results/native_event_stage2_u_20261123_v1")
BASE = Path("results/native_event_rank_psm_20261122_v1")
CHECKPOINT = BASE / "train/point_weighted/selected.pt"
POINT_SHA = "ec572d31d5612ff0d7ebdbe58b133ec631a2430700e3bd449907307de798ddf1"
SCALE_SHA = "7e31a929d01a65bc52784b941c266324b5f883943bf6842bff2a3799b5069eb1"
TARGETS = ("PSM", "machine-3-2", "machine-3-3")
SOURCE = "SHARED"
ARMS = ("replay_copy", "replay_pair")
STEPS = 2000
WARMUP = 200
SEED = 2025
TARGET_SHA = {
    "PSM": ("fba30d9560eb35769d24d8649d592fe2e845813ae772395e845881a18cfaddc7",
            "4cd195ca11fa3873cb71e36f080fb01c8783e7313389136e163d850982cd2db0"),
    "machine-3-2": ("b1781413e5a94ff9d7a1884118c3a1f8bed090f2c7777aee5e8b7a3ba78ce9c2",
                    "b7f9975e4649d800d7935439d3bf9bdca666139e29fd3ae1990ed7631205cc1c"),
    "machine-3-3": ("85f41928b8c2d8bfd6eb979e52ea521015ea4ef73f05b87de6cc56863782e463",
                    "b7f9975e4649d800d7935439d3bf9bdca666139e29fd3ae1990ed7631205cc1c"),
}


def target_binding(w, domain):
    exp = "timeradar_native_v_replay_20261120_v1" if domain == "PSM" else "timeradar_target_mean_smd_20261120_v1"
    d = w / "results" / exp / domain
    task_sha, plan_sha = TARGET_SHA[domain]
    assert sha(d/"TASK.json") == task_sha and sha(d/"PLAN.npz") == plan_sha
    task = read(d/"TASK.json")
    with np.load(d/"PLAN.npz") as z:
        starts = z["target_starts"]
        np.testing.assert_array_equal(starts,pool_starts(task["roles"]["U_FIT"]))
        draws = z["T"]
        assert draws.shape == (STEPS,32) and draws.min() >= 0 and draws.max() < len(starts)
        target = starts[draws]
    return d, task, target


def prepare(w, out):
    _, swat, fit, _, records = source_roles(w)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence([SEED,20261123]).spawn(8)]
    pools = []
    for identifier in SMD_ENTITIES:
        fit_role, = [r for r in records[identifier]["roles"] if r["role"] == "FIT"]
        pools.append(pool_starts(fit_role["bounds_half_open"]))
    plan = {}
    for prefix, (sw, en, win) in zip(("S", "Sprime"),(rngs[:3],rngs[3:6])):
        plan[prefix+"_swat"] = sw.choice(pool_starts(swat["roles"]["SOURCE_FIT"]), size=(STEPS,16))
        ent = en.integers(len(pools),size=(STEPS,16))
        plan[prefix+"_smd_entity"] = ent
        starts = np.empty_like(ent)
        for j,pool in enumerate(pools):
            mask = ent == j
            starts[mask] = win.choice(pool,size=mask.sum())
        plan[prefix+"_smd"] = starts
    strata = [p for group in fit.positive_phases for p in group]
    plan["pair_pos"] = np.stack([rngs[6].choice(p,size=STEPS) for p in strata],axis=1)
    plan["pair_nom"] = rngs[7].choice(fit.normal_endpoints,size=(STEPS,15))
    plan["pair_weights"] = np.asarray([len(p) for p in strata]).reshape(5,3)
    assert sha(w/BASE/"PLAN.npz") == "df7e74047bb0fd49d25a2da0bb58e54a2fbe3327fd9f61b6e07ceca6422543e9"
    with np.load(w/BASE/"PLAN.npz") as z:
        np.testing.assert_array_equal(plan["pair_weights"],z["stratum_sizes"])
    assert all(np.all(np.isin(plan["pair_pos"][:,i],s)) for i,s in enumerate(strata))
    assert np.isin(plan["pair_nom"],fit.normal_endpoints).all()
    out.mkdir(exist_ok=False,parents=True)
    path = out/"SOURCE_PLAN.npz"
    np.savez_compressed(path,**plan)
    target_meta={}
    for domain in TARGETS:
        d,t,_ = target_binding(w,domain)
        target_meta[domain]=dict(old_dir=str(d.relative_to(w)),task_sha256=TARGET_SHA[domain][0],
                                 plan_sha256=TARGET_SHA[domain][1],u=t["roles"]["U_FIT"],
                                 scaler_sha256=t["scaler_sha256"])
    assert sha(w/CHECKPOINT)==POINT_SHA and sha(w/BASE/"SCALE.json")==SCALE_SHA
    dump(out/"CONFIG.json",dict(seed=SEED,initial_sha256=POINT_SHA,scale_sha256=SCALE_SHA,
        source_plan_sha256=sha(path),smd_source_binding_sha256=SMD_SHA,source_copy_swat=swat["roles"]["SOURCE_FIT"],
        swat_event_bounds=fit.bounds,source_scale=read(w/BASE/"SCALE.json")["value"],source_smd=SMD_ENTITIES,
        pair_stratum_sizes=plan["pair_weights"].tolist(),targets=target_meta,source_grid="role-anchored nonoverlapping 100",
        nominal_source_slots="16 SWaT and 16 uniformly sampled SMD machines/windows",
        plan_streams="SeedSequence([2025,20261123]).spawn(8); S, independent Sprime, shared SWaT pair",
        arms=[*ARMS,"source_pair"],update_limit=STEPS,checkpoint="fixed endpoint",update_range="decoder only",
        warmup=WARMUP,optimizer="fresh AdamW, lr=1e-4, betas 0.9/0.95, eps=1e-8, decay 0.1 except bias and LayerNorm",
        batch=64,source_pair_margin=1,source_pair_coefficient=1,source_labeled_windows_per_pair_update=30,
        recipe="32 target or extra source + 32 shared source reconstruction, copy MSE/64 per B1 window. Pair term over 5 events x 3 draws uses fixed positive-endpoint counts. Ten complementary masks, seed84551, norm0, terminal99, channel-mean sample variance."))


def current_model(w, device):
    assert sha(w/CHECKPOINT)==POINT_SHA
    m=decoder_model(w,device)
    official=state_hashes(m)
    state=torch.load(w/CHECKPOINT,weights_only=True,map_location="cpu")
    assert state["update"]==240 and state["arm"]=="point_weighted"
    m.load_state_dict(state["model"],strict=True)
    initial=state_hashes(m)
    assert all(initial[k]==v for k,v in official.items() if not k.startswith("decoder."))
    return m,initial


def source_slot(plan, prefix, row, swat, smd):
    for t in plan[prefix+"_swat"][row]:
        yield swat,t
    for j,t in zip(plan[prefix+"_smd_entity"][row],plan[prefix+"_smd"][row]):
        yield smd[j],t


def single(role,t,device):
    values,lo=role
    return next(input_windows(values,lo,[t],device))


def train(w, out, domain, arm):
    tic=time.monotonic()
    cfg=read(out/"CONFIG.json")
    assert ((domain in TARGETS and arm in ARMS) or (domain==SOURCE and arm=="source_pair"))
    assert cfg["source_plan_sha256"]==sha(out/"SOURCE_PLAN.npz") and sha(w/SMD_BINDING)==SMD_SHA
    assert sha(w/BASE/"SCALE.json")==SCALE_SHA
    with np.load(out/"SOURCE_PLAN.npz") as z:
        plan={k:z[k] for k in z.files}
    d,swat_task=task_binding(w,"SWAT")
    swat=nominal_swat_input(w,d,swat_task,"FIT")
    records={r["id"]:r for r in read(w/SMD_BINDING)["records"]}
    smd=[load_smd(w,records[e],"FIT") for e in SMD_ENTITIES]
    if arm!="replay_copy":
        event_task=dict(swat_task,roles={"EVENT_FIT":cfg["swat_event_bounds"]})
        values,_,lo,_=role_values(w,d,event_task,"EVENT_FIT")
        event=values,lo
    if domain!=SOURCE:
        folder,task,target_plan=target_binding(w,domain)
        values,_,lo,_=role_values(w,folder,task,"U_FIT")
        target=values,lo
    arm_dir=out/"train"/domain/arm
    arm_dir.mkdir(parents=True,exist_ok=False)
    device=setup()
    seed(SEED)
    m,initial=current_model(w,device)
    opt=optimizer(m)
    rng=[torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone()]
    trace=[]
    for i in range(STEPS):
        opt.zero_grad(set_to_none=True)
        lr=1e-4*min((i+1)/WARMUP,1.)
        for group in opt.param_groups:group["lr"]=lr
        slots=list(source_slot(plan,"Sprime",i,swat,smd) if domain==SOURCE else
                   ((target,t) for t in target_plan[i]))
        slots.extend(source_slot(plan,"S",i,swat,smd))
        assert len(slots)==64
        copy=[]
        for role,t in slots:
            x=single(role,t,device)
            _,loss=native_endpoint_and_copy_loss(m,x)
            (loss/64).backward()
            copy.append(loss.detach())
        row=dict(update=i+1,lr=lr,copy_MSE=float(torch.stack(copy).mean()))
        if arm!="replay_copy":
            scores=[]
            for key in ("pair_pos","pair_nom"):
                scores.append(torch.stack([native_endpoint_and_copy_loss(m,single(event,t-99,device))[0]
                                           for t in plan[key][i]]).reshape(5,3))
            pair=event_pair_loss(*scores,cfg["source_scale"],phase_sizes=plan["pair_weights"])
            pair.backward()
            row["pair_loss"]=float(pair.detach())
        norms=[p.grad.norm() for p in m.parameters() if p.requires_grad]
        grad=float(torch.linalg.vector_norm(torch.stack(norms)))
        assert all(np.isfinite(v) for v in row.values()) and np.isfinite(grad) and grad>0
        assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
        opt.step()
        trace.append(dict(row,decoder_gradient_norm=grad))
        if i==0 or (i+1)%100==0:
            print(domain,arm,i+1,round(time.monotonic()-tic,1),flush=True)
    final=state_hashes(m)
    assert all(initial[k]==v for k,v in final.items() if not k.startswith("decoder."))
    assert any(initial[k]!=v for k,v in final.items() if k.startswith("decoder."))
    assert all(not module.training for module in m.modules())
    assert torch.equal(rng[0],torch.get_rng_state()) and torch.equal(rng[1],torch.cuda.get_rng_state(device))
    checkpoint=arm_dir/"endpoint.pt"
    torch.save(dict(model=m.state_dict(),update=STEPS,arm=arm,task=domain),checkpoint)
    dump(arm_dir/"TRACE.json",trace)
    dump(arm_dir/"ENDPOINT.json",dict(status="COMPLETED",domain=domain,arm=arm,update=STEPS,
        checkpoint_sha256=sha(checkpoint),initial_sha256=POINT_SHA,source_plan_sha256=cfg["source_plan_sha256"],
        trace_sha256=sha(arm_dir/"TRACE.json"),config_sha256=sha(out/"CONFIG.json"),code_sha256=sha(__file__),
        target_reconstruction_windows=0 if domain==SOURCE else STEPS*32,
        source_reconstruction_windows=STEPS*(64 if domain==SOURCE else 32),
        additional_positive_windows=0 if arm=="replay_copy" else STEPS*15,
        additional_paired_normal_windows=0 if arm=="replay_copy" else STEPS*15,
        target_labels_read=False,target_values_read=domain!=SOURCE,source_pair_bounds=cfg["swat_event_bounds"] if arm!="replay_copy" else None,
        fixed_state_and_rng_verified=True,seed=SEED,elapsed_seconds=time.monotonic()-tic,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))


def queue(w,out):
    prepare(w,out)
    # Seven fixed endpoints; longest paired (38ch) fits get first access in each lane.
    cases=[(SOURCE,"source_pair"),("machine-3-2","replay_pair"),("machine-3-3","replay_pair"),
           ("PSM","replay_pair"),("PSM","replay_copy"),("machine-3-3","replay_copy"),("machine-3-2","replay_copy")]
    jobs=run_jobs(w,out,"training",[(d+"_"+a,["train","--domain",d,"--arm",a],20,21600) for d,a in cases],worker_module=MODULE)
    seal(w,out,jobs)


def seal(w,out,jobs):
    rows=[]
    for domain,arm in [(SOURCE,"source_pair")]+[(d,a) for d in TARGETS for a in ARMS]:
        p=out/"train"/domain/arm/"endpoint.pt"
        rec=read(p.with_name("ENDPOINT.json"))
        assert rec["status"]=="COMPLETED" and rec["update"]==2000 and sha(p)==rec["checkpoint_sha256"]
        rows.append(dict(domain=domain,arm=arm,checkpoint_relative=p.relative_to(out).as_posix(),sha256=sha(p)))
    dump(out/"ALL_MODELS_FROZEN.json",dict(status="COMPLETED",jobs=jobs,models=rows,
                                         actual_GPU_hours=sum(j["actual_GPU_seconds"] for j in jobs)/3600))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("train","queue"))
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--domain",choices=(*TARGETS,SOURCE))
    parser.add_argument("--arm",choices=(*ARMS,"source_pair"))
    args=parser.parse_args()
    if args.action=="queue":queue(args.workspace,args.output)
    else:train(args.workspace,args.output,args.domain,args.arm)
