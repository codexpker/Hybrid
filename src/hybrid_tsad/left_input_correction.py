"""Source-only development of bounded training-input correction for native LEFT."""

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from . import left_multisource as base
from . import left_reliability as weights
from .intervention_reference import edit
from .left_smd28_author import bindings

OUT = Path("results/left_input_correction_20261127_v1")
WINDOW, STEPS, SEED = 192, 2000, 2025
CONFIG = {"R0": (0., None), "R1": (.25, .95), "R2": (.5, .95),
          "R3": (.25, .99), "R4": (.5, .99), "R5": None, "R6": None, "R7": None}


def rng(stream, index=0):
    return np.random.default_rng(np.random.SeedSequence([SEED, 20261127, 427, stream, index]))


def roles(w):
    raw = base.source_records(w)
    names = [[f"TEP/run-{i}" for i in range(1, 21)], sorted(bindings(w))]
    select_names = [[f"TEP/run-{i}" for i in range(61, 66)], names[1]]
    fit = raw["fit"]
    query = raw["select"]
    assert [len(g) for g in fit] == [20, 28] and [len(g) for g in query] == [5, 28]
    manifest = {}
    limits = {"reference": ((0, 4), (0, 6)), "support": ((4, 12), (6, 17)),
              "replay": ((12, 20), (17, 28))}
    pools = {}
    for role, slices in limits.items():
        pools[role] = [fit[j][lo:hi] for j, (lo, hi) in enumerate(slices)]
        manifest[role] = [
            [dict(record=names[j][k], start=0, stop=len(fit[j][k])) for k in range(lo, hi)]
            for j, (lo, hi) in enumerate(slices)
        ]
    pools["query"] = query
    manifest["query"] = [
        [dict(record=name, start=(offset := len(fit[j][k]) if j else 0), stop=offset+len(x))
         for k, (name, x) in enumerate(zip(select_names[j], query[j]))]
        for j in range(2)
    ]
    return pools, manifest


def schedule(pool, shape, width, stream):
    random = rng(stream)
    plan = np.empty((*shape, width+2), dtype=np.int32)
    for i, row in enumerate(plan.reshape(-1, width+2)):
        group = pool[i%2]
        rec = random.integers(len(group))
        x = group[rec]
        row[:2] = rec, random.integers(len(x)-WINDOW+1)
        row[2:] = random.choice(x.shape[1], width, replace=False)
    return plan


def render(pool, plan):
    return np.stack([pool[i%2][r][s:s+WINDOW][:, cols]
                     for i, (r, s, *cols) in enumerate(plan)]).astype(np.float32)


def edited_window(x, kind, random):
    start = random.integers(6, WINDOW-20-6+1)
    return edit(x, kind, random, start=int(start)), np.arange(start, start+20)


def support_batch(pools, plan, index):
    x = render(pools["support"], plan["support"][index])
    random = rng(4, index)
    for j in range(2):
        slots = random.choice(np.arange(j, len(x), 2), 2, replace=False)
        for kind, k in enumerate(slots):
            x[k], _ = edited_window(x[k], kind, random)
    return x


@torch.no_grad()
def reconstruction(teacher, x):
    assert not teacher.training
    return teacher._forward(x, None, None, None)[9]


def correction(x, reference, method, stats):
    if method in ("R0", "R7"):
        return x
    if method == "R6":
        return torch.clamp(x, min=stats["low"], max=stats["high"])
    if method == "R5":
        return reference
    rho, q = CONFIG[method]
    residual = x-reference
    return x-rho*residual.sign()*(residual.abs()-stats[str(q)]).clamp_min(0)


def plan_files(root):
    r = base.read(root/"prepare.json")
    for k, digest in r["hashes"].items():
        assert base.sha(root/k) == digest
    return r, np.load(root/"plan.npz")


def torch_x(x):
    return torch.from_numpy(np.ascontiguousarray(x)).cuda()


@torch.no_grad()
def evaluate(model, clean, edits, positions):
    model.eval()
    native, normal_scores = [[], []], []
    for i in range(0, len(clean), 32):
        x = clean[i:i+32]
        native_batch = [model(torch_x(x[j::2]), None, None, None)[0].item() for j in range(2)]
        for group, value in zip(native, native_batch):
            group.append(value)
        normal_scores.append(model.infer(torch_x(x), None, None, None)[0].mean(-1).cpu().numpy())
    normal = np.concatenate(normal_scores)
    results = []
    for k, family in enumerate(("amplitude", "shift")):
        scores = np.concatenate([
            model.infer(torch_x(edits[k, i:i+32]), None, None, None)[0].mean(-1).cpu().numpy()
            for i in range(0, len(clean), 32)
        ])
        a = np.take_along_axis(normal, positions[k], axis=1)
        b = np.take_along_axis(scores, positions[k], axis=1)
        for j, source in enumerate(("TEP", "SMD")):
            s = np.r_[a[j::2].ravel(), b[j::2].ravel()]
            y = np.repeat([0, 1], len(s)//2)
            results.append(dict(family=family, source=source, AP=average_precision_score(y,s),
                                ROC_AUC=roc_auc_score(y,s), score_gap_mean=float((b[j::2]-a[j::2]).mean())))
    return dict(normal_risk={k: float(np.mean(x)) for k,x in zip(("TEP","SMD"),native)},
                panels=results, macro_AP=float(np.mean([v["AP"] for v in results])))


@torch.no_grad()
def prepare(w, d, root):
    if (root/"prepare.json").exists():
        plan_files(root)
        return
    t0 = time.monotonic()
    teacher = base.initialize(w,d).eval().requires_grad_(False)
    parent = base.restore(w,teacher,d)
    pools, manifest = roles(w)
    width = base.WIDTH[d]
    shapes = dict(reference=(512,), query=(256,), support=(STEPS,16), replay=(STEPS,16))
    plans = {k: schedule(pools[k], shape, width, i+10) for i,(k,shape) in enumerate(shapes.items())}
    np.savez_compressed(root/"plan.npz", **plans)
    base.dump(root/"roles.json", manifest)
    clean = render(pools["query"],plans["query"])
    edited, pos = [], []
    for k in range(2):
        for i,x in enumerate(clean):
            e,p = edited_window(x,k,rng(20+k,i))
            edited.append(e)
            pos.append(p)
    edited, pos = np.asarray(edited).reshape(2,*clean.shape), np.asarray(pos).reshape(2,len(clean),20)
    np.savez_compressed(root/"query.npz", clean=clean, edited=edited, positions=pos)
    ref = render(pools["reference"],plans["reference"])
    residual = np.concatenate([np.abs(x-reconstruction(teacher,torch_x(x)).cpu().numpy())
                               for x in np.split(ref,16)])
    stats = {str(q): float(np.quantile(residual,q)) for q in (.95,.99)}
    stats.update(low=float(np.quantile(ref,.005)), high=float(np.quantile(ref,.995)))
    print(d,"frozen bounds",stats,flush=True)
    # Exact existing window-weight definition on the new, paired proxy exposure.
    point = np.empty((STEPS,16,WINDOW), np.float32)
    for i in range(STEPS):
        x = torch_x(support_batch(pools,plans,i))
        point[i] = teacher.infer(x,None,None,None)[0].mean(-1).cpu().numpy()
    rank = (rankdata(point,method="average").reshape(point.shape)-.5)/point.size
    raw = 1.5-rank
    point_weights = (raw/raw.mean((1,2),keepdims=True)).astype(np.float32)
    np.testing.assert_allclose(point_weights.mean((1,2)),1.,atol=2e-7,rtol=0)
    np.save(root/"R7.npy",point_weights)
    assert teacher.step == parent["selected"] and base.fingerprint(teacher.state_dict()) == parent["fingerprint"]
    bench = evaluate(teacher,clean,edited,pos)
    base.dump(root/"prepare.json",dict(
        domain=d, parent=parent["sha256"], teacher_parameters=sum(p.numel() for p in teacher.parameters()),
        teacher_state_unchanged=True, stats=stats, frozen_query=bench, seconds=time.monotonic()-t0,
        hashes={name:base.sha(root/name) for name in ("plan.npz","roles.json","query.npz","R7.npy")},
        device=torch.cuda.get_device_name(), peak_bytes=torch.cuda.max_memory_allocated(), labels_read=False))


def models(w,d,arm):
    student = base.initialize(w,d,weighted_reductions=(weights.smooth,weights.multiscale)
                             if arm=="R7" else None)
    parent = base.restore(w,student,d)
    teacher = None
    if arm in ("R1","R2","R3","R4","R5"):
        teacher = base.initialize(w,d).eval().requires_grad_(False)
        assert base.restore(w,teacher,d)["sha256"] == parent["sha256"]
    opt = torch.optim.Adam(student.parameters(),lr=1e-4)
    if teacher is not None:
        assert not {id(p) for p in teacher.parameters()} & {id(p) for g in opt.param_groups for p in g["params"]}
    return student.train(), teacher,opt,parent


def update(student,opt,x):
    step = student.step
    opt.zero_grad(set_to_none=True)
    loss = student(x,None,None,None)[0]
    loss.backward()
    opt.step()
    assert student.step == step+1
    return loss.detach()


def check(w,d,root):
    if (root/"check.json").exists():
        return
    t0=time.monotonic()
    config, plan = plan_files(root)
    pools,_=roles(w)
    raw = support_batch(pools,plan,0)
    replay = render(pools["replay"],plan["replay"][0])
    a,b = torch_x(raw),torch_x(replay)
    proof = {}
    for arm in ("R0","R2","R7"):
        student,teacher,opt,parent=models(w,d,arm)
        if arm=="R7":
            weights.set_weights(student,a.new_ones((16,WINDOW)),False)
        cpu,cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
        reference=reconstruction(teacher,a) if teacher is not None else None
        y=correction(a,reference,arm,config["stats"])
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
        x=torch.cat((y,b))
        assert torch.equal(x[16:],b) and not x.requires_grad
        if arm in ("R0","R7"):
            assert y is a
        else:
            assert reference.shape==a.shape
            residual=(a-reference).abs()
            torch.testing.assert_close((a-y).abs(),.5*(residual-config["stats"]["0.95"]).clamp_min(0),
                                       atol=2e-6,rtol=2e-6)
        loss=float(update(student,opt,x))
        state=base.fingerprint(student.state_dict())
        if arm=="R0":
            original,_teacher,original_opt,_=models(w,d,"R0")
            native=base.update(original,original_opt,np.concatenate((raw,replay)))
            assert loss==native and state==base.fingerprint(original.state_dict())
            native_loss=loss
        if arm=="R7":
            assert abs(loss-native_loss)<2e-6
        if teacher is not None:
            assert not any(p.requires_grad or p.grad is not None for p in teacher.parameters())
            assert teacher.step==parent["selected"] and base.fingerprint(teacher.state_dict())==parent["fingerprint"]
        proof[arm]=dict(loss=loss, corrected_fraction=float((a!=y).float().mean()),step=student.step)
    # Native deployment consumes the original input with no correction or fitted weights.
    student.eval()
    before=base.fingerprint(student.state_dict()),student.step
    s=student.infer(a,None,None,None)[0]
    assert s.shape==a.shape and torch.isfinite(s).all() and before==(base.fingerprint(student.state_dict()),student.step)
    base.dump(root/"check.json",dict(proof=proof,raw_native_readout=True,source_unchanged=True,
              teacher_optimizer_disjoint=True,teacher_frozen=True,rng_equal=True,
              device=torch.cuda.get_device_name(),seconds=time.monotonic()-t0))


@torch.no_grad()
def diagnostics(teacher, clean, edited, pos, arm, stats):
    if arm in ("R0","R7"):
        return dict(clean_changed=0.,clean_rms=0.,eligible_edit_fraction=[0.,0.],
                    clean_relative_edit_rms=[1.,1.],off_edit_rms=[0.,0.])
    clean_t = torch_x(clean)
    def transform(x):
        if teacher is None:
            return correction(x,None,arm,stats), None
        r = torch.cat([reconstruction(teacher,z) for z in x.split(32)])
        return correction(x,r,arm,stats),r
    normal,_ = transform(clean_t)
    answer=dict(clean_changed=float((normal!=clean_t).float().mean()),
                clean_rms=float((normal-clean_t).square().mean().sqrt()),
                eligible_edit_fraction=[],clean_relative_edit_rms=[],off_edit_rms=[],teacher_copied_below_q99=[])
    for e,p in zip(edited,pos):
        x=torch_x(e)
        y,r=transform(x)
        changed=x!=clean_t
        answer["eligible_edit_fraction"].append(float((x[changed]!=y[changed]).float().mean()))
        answer["clean_relative_edit_rms"].append(float((y-clean_t)[changed].norm()/(x-clean_t)[changed].norm()))
        answer["off_edit_rms"].append(float((y-x)[~changed].square().mean().sqrt()))
        if r is not None:
            answer["teacher_copied_below_q99"].append(float(((x-r)[changed].abs()<=stats["0.99"]).float().mean()))
    return answer


def train(w,d,root,arm):
    job=root/arm
    if (job/"receipt.json").exists():
        assert base.sha(job/"selected.pt")==base.read(job/"receipt.json")["checkpoint_sha256"]
        return
    job.mkdir(exist_ok=False)
    t0=time.monotonic()
    config,plan=plan_files(root)
    assert (root/"check.json").is_file()
    # Materialize compact index plans once, never dense training windows.
    plan={k:plan[k] for k in ("support","replay")}
    pools,_=roles(w)
    model,teacher,opt,parent=models(w,d,arm)
    assert parent["sha256"]==config["parent"]
    r7=np.load(root/"R7.npy",mmap_mode="r") if arm=="R7" else None
    curve=[]
    total=torch.zeros(3,device="cuda")
    teacher_ms = []
    for i in range(STEPS):
        x=torch_x(support_batch(pools,plan,i))
        source=torch_x(render(pools["replay"],plan["replay"][i]))
        if teacher is None:
            ref=None
        else:
            e0,e1=(torch.cuda.Event(enable_timing=True) for _ in range(2))
            e0.record()
            ref=reconstruction(teacher,x)
            e1.record()
            teacher_ms.append((e0,e1))
        y=correction(x,ref,arm,config["stats"])
        if r7 is not None:
            weights.set_weights(model,torch_x(np.array(r7[i])),False)
        loss=update(model,opt,torch.cat((y,source)))
        total+=torch.stack((loss,(y-x).square().mean(),(y!=x).float().mean()))
        if (i+1)%400==0:
            loss,mse,changed=(total/400).tolist()
            assert np.isfinite([loss,mse,changed]).all()
            curve.append(dict(updates=i+1,loss=loss,correction_rms=mse**.5,changed=changed))
            print(d,arm,curve[-1],flush=True)
            total.zero_()
    assert model.step==parent["selected"]+STEPS
    if teacher is not None:
        assert teacher.step==parent["selected"] and base.fingerprint(teacher.state_dict())==parent["fingerprint"]
    base.save(model,job)
    fit_seconds=time.monotonic()-t0
    with np.load(root/"query.npz") as q:
        pane=evaluate(model,q["clean"],q["edited"],q["positions"])
        diag=diagnostics(teacher,q["clean"],q["edited"],q["positions"],arm,config["stats"])
    base.dump(job/"receipt.json",dict(
        domain=d,arm=arm,updates=STEPS,seed=SEED,parent=parent["sha256"],plan=config["hashes"]["plan.npz"],
        query=config["hashes"]["query.npz"],checkpoint_sha256=base.sha(job/"selected.pt"),
        target_exposure=STEPS*16,source_exposure=STEPS*16,edited_proxy_exposure=STEPS*4,
        model_step=model.step,curve=curve,proxy=pane,diagnostics=diag,
        device=torch.cuda.get_device_name(),peak_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.monotonic()-t0,fit_seconds=fit_seconds,
        teacher_train_seconds=sum(a.elapsed_time(b) for a,b in teacher_ms)/1000,
        labels_read=False,pid=os.getpid()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("workspace",type=Path)
    p.add_argument("domain",choices=base.WIDTH)
    p.add_argument("task",choices=("prepare","check",*CONFIG))
    args=p.parse_args()
    root=args.workspace/OUT/args.domain
    root.mkdir(parents=True,exist_ok=True)
    if args.task=="prepare":
        prepare(args.workspace,args.domain,root)
    elif args.task=="check":
        check(args.workspace,args.domain,root)
    else:
        train(args.workspace,args.domain,root,args.task)


if __name__=="__main__":
    main()
