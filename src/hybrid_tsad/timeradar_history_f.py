"""Offline public-support qualification and executable matched F, no QUERY selection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .timeradar_history_experiment import model, save
from .timeradar_history_context import record_window
from .native_v_conditioning import sha
from .native_v_replay import optimizer
from .trajectory_experiment import ROLES, target, dump
from .trajectory_evaluation import labels
from .left_multisource import target_inputs


def intervals(y):
    change=np.diff(np.r_[0,y,0])
    return list(zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)))


def numeric_intervals(ends,origin):
    """Exact union of legal numeric footprints, distinct from annotation budget."""
    spans=[]
    for end in np.sort(ends):
        start,stop=int(end)-611+origin,int(end)+1+origin
        if spans and start<=spans[-1][1]:
            spans[-1][1]=max(spans[-1][1],stop)
        else:
            spans.append([start,stop])
    return spans


def support_directory(out, domain, contract):
    return out/domain/("F_support" if contract=="original" else "F_support_full_normal_block")


def prepare(w,out,contract="original"):
    for d in ("PSM","SWAT"):
        dest=support_directory(out,d,contract)
        if contract!="original" and dest.exists():
            raise FileExistsError(dest)
        dest.mkdir(parents=True,exist_ok=True)
        lo,hi=ROLES[d]["U"]
        y=labels(w,d)[lo:hi]
        positive=None
        for a,b in intervals(y):
            # Require an entire labeled episode and at least one legal current W100.
            if a==0 or b==len(y): continue
            ends=np.arange(max(a+99,611),b)
            if len(ends):
                positive=(a,b,ends);break
        normal=None
        for a,b in intervals(1-y):
            # For the revised contract, even the first current W100 in the
            # normal block must have 512 preceding numeric U-FIT points.
            start=max(a,512 if contract=="full-normal-block" else 100)
            if start+512<=b:
                ends=np.arange(max(start+99,611),start+512)
                if len(ends):
                    normal=(start,start+512,ends);break
        if positive is None or normal is None:
            dump(dest/"support.json",dict(status="NA",reason="No complete labeled support event or qualified normal block"))
            continue
        rngs=[np.random.default_rng(r) for r in np.random.SeedSequence(2025).spawn(3)]
        pe,ne=positive[2],normal[2]
        if contract=="full-normal-block":
            assert len(ne)==413
        strata=np.array_split(pe,3)
        plan=dict(copy=rngs[0].choice(ne,(240,32)),
                  positive=np.column_stack([rngs[1].choice(g,240*5).reshape(240,5) for g in strata if len(g)]),
                  negative=rngs[2].choice(ne,(240,5*sum(bool(len(g)) for g in strata))),
                  weights=np.repeat([len(g) for g in strata if len(g)],5).astype(np.float64))
        if any(len(g)==0 for g in strata):
            dump(dest/"support.json",dict(status="NA",reason="Three nonempty positive time strata unavailable"))
            continue
        np.savez_compressed(dest/"plan.npz",**plan)
        # Only these exact numeric footprints are passed to fitting.
        ends=np.unique(np.concatenate([ne,pe]))
        values=target(w,d,"U")
        _,_,sc,_=target_inputs(w,d)
        footprints=np.stack([sc.transform(values[t-611:t+1]).astype(np.float32) for t in ends])
        assert all(t-611>=0 and t+1<=len(y) for t in ends)
        np.savez_compressed(dest/"inputs.npz",ends=ends,footprints=footprints,normal_ends=ne)
        spans=numeric_intervals(ends,lo)
        dump(dest/"support.json",dict(status="ready",domain=d,U_bounds=[lo,hi],support_contract=contract,
             abnormal_event=[int(positive[0]+lo),int(positive[1]+lo)],
             normal_support=[int(normal[0]+lo),int(normal[1]+lo)],
             anomaly_label_points=int(positive[1]-positive[0]),normal_label_points=512,
             earliest_history=int(ends.min()-611+lo),latest_current_stop=int(ends.max()+1+lo),
             numeric_footprint_intervals=spans,numeric_union_points=sum(b-a for a,b in spans),
             additional_unlabeled_history_points=int(sum(b-a for a,b in spans)-512-(positive[1]-positive[0])),
             history_labels_used=False,history_permission="numeric U-FIT, may include unlabeled abnormal points",
             positive_endpoints=len(pe),normal_endpoints=len(ne),plan_sha256=sha(dest/"plan.npz"),
             inputs_sha256=sha(dest/"inputs.npz"),steps=240,warmup=24,copy_batch=32,pair_batch=15,
             pair_coefficient=1,margin=1,loss="copy normal + weighted softplus(1-log1p(Vpos/aF)+log1p(Vneg/aF))",
             scale="official native V median on all legal normal support ends, floor1e-8, shared between arms",
             initial="selected source endpoint directly, never U",
             checkpoint="fixed240, step0 diagnostic, no selection",
             teacher_and_sample_rng=2025,source_replay=False))
        print(d,positive[:2],normal[:2],len(pe),len(ne),flush=True)


def fit(w,out,d,arm,initial="source",support_contract="original"):
    root=support_directory(out,d,support_contract)
    meta=json.loads((root/"support.json").read_text())
    assert meta["status"]=="ready"
    assert sha(root/"inputs.npz")==meta["inputs_sha256"] and sha(root/"plan.npz")==meta["plan_sha256"]
    with np.load(root/"inputs.npz") as z:
        ends=z["ends"];data=z["footprints"];normal=z["normal_ends"]
    with np.load(root/"plan.npz") as z: plan={k:z[k] for k in z.files}
    def window(end):
        values=torch.as_tensor(data[np.searchsorted(ends,end)],device="cuda")
        return record_window(values,512,(0,612),512)
    official=model(w,"native")
    with torch.no_grad():
        v=[official(*window(e)[:2],**window(e)[2],bypass=True)[1].var(0,correction=1).mean(-1)[0,-1].item()
           for e in normal]
    scale=max(1e-8,float(np.median(v)))
    del official
    m=model(w,arm)
    source=out/"source"/arm/"selected.pt"
    if initial=="source":
        parent=json.loads(source.with_name("receipt.json").read_text())
        assert sha(source)==parent["sha256"]
        m.load_state_dict(torch.load(source,map_location="cpu",weights_only=True)["model"])
    opt=optimizer(m)
    dest=out/d/("F_"+arm+("_official_initialization" if initial=="official" else ""))
    if support_contract!="original":
        dest=dest.with_name(dest.name+"_full_normal_block")
    dest.mkdir(exist_ok=False)
    started=time.monotonic();curve=[]
    def one(end):
        h,x,footprint=window(end)
        y=m(h,x,**footprint,bypass=arm!="history")[1]
        return y.var(0,correction=1).mean(-1)[0,-1],(y-x[None]).square().mean()
    for step in range(240):
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group["lr"]=1e-4*min(1,(step+1)/24)
        copy=0.
        for end in plan["copy"][step]:
            _,loss=one(end)
            (loss/32).backward()
            copy+=loss.detach().item()/32
        positive=torch.stack([one(e)[0] for e in plan["positive"][step]])
        negative=torch.stack([one(e)[0] for e in plan["negative"][step]])
        risk=F.softplus(1-torch.log1p(positive/scale)+torch.log1p(negative/scale))
        weight=risk.new_tensor(plan["weights"])
        pair=(risk*weight).sum()/weight.sum()
        pair.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        opt.step()
        curve.append(dict(step=step+1,copy=copy,pair=pair.item()))
    save(dest/"selected.pt",dict(model=m.state_dict(),step=240,arm=arm))
    dump(dest/"receipt.json",dict(steps=240,initial=initial,
         source=sha(source) if initial=="source" else None,
         full_official_and_initial_history_state=initial=="official",
         sha256=sha(dest/"selected.pt"),scale=scale,seconds=time.monotonic()-started,
         support_plan=meta["plan_sha256"],support_values=meta["inputs_sha256"]))
    dump(dest/"curve.json",curve)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","fit"))
    p.add_argument("workspace",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--domain",choices=("PSM","SWAT"))
    p.add_argument("--arm",choices=("native","history"))
    p.add_argument("--initial",choices=("source","official"),default="source")
    p.add_argument("--support-contract",choices=("original","full-normal-block"),default="original")
    a=p.parse_args()
    if a.action=="prepare":prepare(a.workspace,a.output,a.support_contract)
    else:fit(a.workspace,a.output,a.domain,a.arm,a.initial,a.support_contract)
