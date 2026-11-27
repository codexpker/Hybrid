"""Repeat original LEFT adaptation with fixed inputs and new training RNG trajectories."""

import argparse
import csv
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import torch

from . import left_multisource as base
from . import left_reliability as weighting
from .left_memory_update import OUT as EXPOSURE, batches
from .left_multisource_eval import bank

SEEDS = (2026, 2027)
METHODS = ("R", "W")
CONDITIONS = "ABC"


def digest(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def pair_name(method, condition, seed):
    return f"{method}_{condition}_{seed}"


def verify(w, archive, out):
    """Bind plans, parents and frozen coefficients to existing 2025 receipts."""
    source_manifest = archive / "left_evidence_freeze_20261128/source_manifest.csv"
    with source_manifest.open(newline="", encoding="utf-8") as f:
        rows = {r["key"]: r for r in csv.DictReader(f)}
    old_weights = base.read(archive / "left_local_reliability_20261126/receipts.json")
    old_plans = base.read(archive / "left_memory_exposure_20261126/receipts.json")
    folds = {}
    for d in base.WIDTH:
        source = w / base.OUT / d / "source"
        parent = base.read(source / "receipt.json")
        assert base.sha(source / "selected.pt") == parent["sha256"]
        scaler_file = w / base.NATIVE / d / "native/SCALER.npz"
        scaler = base.sha(scaler_file)
        assert scaler == base.read(scaler_file.with_name("CONFIG.json"))["scaler_sha256"]
        plan = w / EXPOSURE / d / "plan"
        weight_root = w / weighting.OUT / d / "weights"
        teacher = old_weights[d]["teacher"]
        assert teacher["parent"] == parent["sha256"]
        for name, expected in teacher["hashes"].items():
            assert base.sha(weight_root / name) == expected, (d, name)
        _, _, sc, _ = base.target_inputs(w, d)
        del sc  # Only the archived identity/normal-value check is needed here.
        with np.load(plan/"A.npz") as p:
            original = p["first"]
            shared_second = p["second"]
        stream = np.random.default_rng(np.random.SeedSequence(2025).spawn(3)[0])
        np.testing.assert_array_equal(
            original, stream.integers(len(base.target(w, d, "U"))-base.WINDOW+1,
                                      size=(base.U_STEPS,16)))
        c_receipts = {}
        for c in CONDITIONS:
            path = plan / (c+".npz")
            plan_id = base.sha(path)
            assert plan_id==old_plans[d]["qualification"][c+"_sha256"]
            with np.load(path) as x:
                np.testing.assert_array_equal(x["second"], shared_second)
                replaced = int(np.count_nonzero(x["first"]!=original))
            for mid in "DE":
                r=rows[f"controlled_exposure/{d}/{mid}/{c}"]
                assert r["parent_sha256"]==parent["sha256"] and r["plan_sha256"]==plan_id
            w_id = rows[f"controlled_exposure/{d}/E/{c}"]["weights_sha256"]
            assert w_id == teacher["hashes"][c+".npy"]
            c_receipts[c]=dict(plan_sha256=plan_id,weights_sha256=w_id,changed_from_A=replaced,
                               archived_R=rows[f"controlled_exposure/{d}/D/{c}"]["checkpoint_sha256"],
                               archived_W=rows[f"controlled_exposure/{d}/E/{c}"]["checkpoint_sha256"])
        with np.load(plan/"B.npz") as b, np.load(plan/"C.npz") as c:
            slots=np.load(plan/"audit_slots.npy")
            changed_b=b["first"]!=original
            changed_c=c["first"]!=original
            # Audit slots store flattened coordinates into the shared first half.
            if slots.dtype==bool and slots.shape==original.shape:
                selected=slots
            elif slots.ndim==2 and slots.shape[1]==2:
                selected=np.zeros_like(original,dtype=bool)
                selected[slots[:,0],slots[:,1]]=True
            else:
                selected=np.zeros(original.size,dtype=bool)
                selected[slots.reshape(-1)]=True
                selected=selected.reshape(original.shape)
            assert selected.sum()==3200 and not ((changed_b|changed_c)&~selected).any()
        folds[d]=dict(parent=parent["sha256"],scaler=scaler,teacher=teacher["hashes"]["teacher.npz"],
                      audit_slots=base.sha(plan/"audit_slots.npy"), conditions=c_receipts)
    base.dump(out/"identity.json",dict(
        folds=folds, archive_commit="20331f57939693cc34d2ab7128504713752fd130",
        sampling_and_teacher_seed=2025, adaptation_seeds=SEEDS, budget_updates=2000,
        jobs=[dict(domain=d,method=m,condition=c,seed=s) for d in base.WIDTH
              for s in SEEDS for c in CONDITIONS for m in METHODS],
        query_labels_loaded=False))


def setup(w, d, method, seed):
    kwargs = dict(weighted_reductions=(weighting.smooth, weighting.multiscale)) if method=="W" else {}
    model = base.initialize(w, d, **kwargs)
    parent = base.restore(w, model, d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return model,opt,parent


def step(model, opt, x):
    previous=model.step
    opt.zero_grad(set_to_none=True)
    loss=model(x,None,None,None)[0]
    loss.backward()
    opt.step()
    assert model.step == previous+1
    return loss.detach()


def probe(w, d, output):
    start=time.monotonic()
    x=next(batches(w,d,"replay_A"))
    observations={}
    for seed in SEEDS:
        model,opt,_=setup(w,d,"R",seed)
        fired=[]
        handles=[m.register_forward_hook(lambda layer, _, result: fired.append(
            dict(p=layer.p, output=digest(result.detach().float().cpu().numpy()))))
                 for m in model.modules() if isinstance(m,torch.nn.Dropout) and m.p>0]
        first=digest(torch.cuda.get_rng_state().numpy())
        loss=step(model,opt,x).item()
        for h in handles:
            h.remove()
        assert fired and torch.isfinite(torch.tensor(loss))
        observations[str(seed)]=dict(
            rng_start=first,rng_end=digest(torch.cuda.get_rng_state().numpy()),
            loss=loss, dropout_calls=len(fired), first_dropout=fired[0],
            state_after=base.fingerprint(model.state_dict()))
    a,b=observations.values()
    assert a["first_dropout"]["output"]!=b["first_dropout"]["output"]
    assert a["state_after"]!=b["state_after"]
    base.dump(output/d/"rng_probe.json",dict(batch=digest(x.cpu().numpy()),domain=d,
              observations=observations,seconds=time.monotonic()-start,
              device=torch.cuda.get_device_name()))


def adapt(w, d, output, method, condition, seed):
    name = pair_name(method,condition,seed)
    dest = output/d/name
    receipt = dest/"receipt.json"
    if receipt.is_file():
        assert base.sha(dest/"selected.pt")==base.read(receipt)["sha256"]
        return
    binding = base.read(output/"identity.json")["folds"][d]
    c = binding["conditions"][condition]
    dest.mkdir(exist_ok=False)
    start = time.monotonic()
    # Planning, weight I/O and reconstruction of the source pool use no training RNG.
    weights=weighting.weights_file(w,d,condition) if method=="W" else None
    plan=w/EXPOSURE/d/"plan"/(condition+".npz")
    assert base.sha(plan)==c["plan_sha256"]
    model,opt,parent=setup(w,d,method,seed)
    assert parent["sha256"]==binding["parent"]
    assert (output/d/"rng_probe.json").is_file()
    initial=model.step
    curve, trace = [], []
    total=torch.zeros((),device="cuda")
    for i,x in enumerate(batches(w,d,"replay_"+condition),1):
        if weights is not None:
            weighting.set_weights(model,torch.as_tensor(np.array(weights[i-1]),device=x.device),False)
        if i<=4:
            fingerprint=digest(x.cpu().numpy())
            rng=digest(torch.cuda.get_rng_state().numpy())
        loss=step(model,opt,x)
        total+=loss
        if i<=4:
            row=dict(update=i,batch=fingerprint,rng_before=rng,
                     rng_after=digest(torch.cuda.get_rng_state().numpy()),loss=loss.item())
            trace.append(row)
            print(d,name,row,flush=True)
        if i%400==0:
            value=(total/400).item()
            assert np.isfinite(value)
            curve.append(dict(update=i,loss=value))
            total.zero_()
            print(d,name,curve[-1],flush=True)
    assert len(curve)==5 and model.step==initial+2000
    base.save(model,dest)
    base.dump(receipt,dict(
        domain=d,method=method,condition=condition,seed=seed,updates=2000,initial_step=initial,
        step=model.step, parent=parent["sha256"],scaler=binding["scaler"],
        numeric_plan=c["plan_sha256"],teacher_weights=c["weights_sha256"] if method=="W" else None,
        target_exposure=32000,source_exposure=32000,training_labels_read=False,
        sha256=base.sha(dest/"selected.pt"),device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,cuda_version=torch.version.cuda,
        seconds=time.monotonic()-start,peak_bytes=torch.cuda.max_memory_allocated(),
        visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),pid=os.getpid(),trace=trace,curve=curve))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("workspace",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("action",choices=("verify","probe","adapt","score"))
    p.add_argument("--archive",type=Path)
    p.add_argument("--domain",choices=base.WIDTH)
    p.add_argument("--method",choices=METHODS)
    p.add_argument("--condition",choices=CONDITIONS)
    p.add_argument("--seed",type=int,choices=SEEDS)
    a=p.parse_args()
    output=a.output.resolve()
    if a.action=="verify":
        for d in base.WIDTH:
            (output/d).mkdir(parents=True,exist_ok=True)
        verify(a.workspace,a.archive,output)
    elif a.action=="probe":
        probe(a.workspace,a.domain,output)
    elif a.action=="adapt":
        adapt(a.workspace,a.domain,output,a.method,a.condition,a.seed)
    else:
        bank(a.workspace,a.domain,pair_name(a.method,a.condition,a.seed),"QUERY",root=output)


if __name__=="__main__":
    main()
