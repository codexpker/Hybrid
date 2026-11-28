"""Fixed paired source training and native-V evaluation for history TimeRadar."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .native_v_conditioning import load_native, sha, WEIGHT_SHA
from .native_v_replay import optimizer
from .timeradar_history_context import HistoryConditionedTimeRadar, record_window
from .intervention_reference import source_records
from .trajectory_experiment import ROLES, target, dump
from .left_multisource import target_inputs

SEED = 2025


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def model(w, arm, device="cuda"):
    torch.manual_seed(SEED)
    m = HistoryConditionedTimeRadar(load_native(w, device)).to(device).eval()
    if arm != "history":
        for n, p in m.named_parameters():
            p.requires_grad_(n.startswith("native.decoder."))
    return m


def pools(w):
    # Existing source arrays have FIT-only scales, with record boundaries intact.
    return source_records(w)


def source_draws(records, rng, count):
    draws = []
    for j in range(count):
        domain = j % 2
        rid = int(rng.integers(len(records[domain])))
        x = records[domain][rid]
        h = 256 if domain == 0 else 512
        start = int(rng.integers(h, len(x)-99))
        draws.append((domain, rid, start))
    return draws


def get_source(records, row, device):
    domain, rid, start = map(int,row)
    h = 256 if domain == 0 else 512
    record = records[domain][rid]
    assert h <= start and start+100 <= len(record)
    values = torch.as_tensor(np.array(record[start-h:start+100],copy=True),device=device)
    return values[:h][None], values[h:][None], dict(history_start=start-h,current_start=start)


def prepare(w, out):
    out.mkdir(parents=True,exist_ok=False)
    data = pools(w)
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(4)]
    select = np.array(source_draws(data["select"],rngs[1],32))
    source = np.array([source_draws(data["fit"],rngs[0],32) for _ in range(240)])
    replay = np.array([source_draws(data["fit"],rngs[2],16) for _ in range(2000)])
    plans = dict(source=source,select=select,replay=replay)
    for d in ("PSM","SWAT"):
        plans[d] = rngs[3].integers(512,len(target(w,d,"U"))-99,size=(2000,16))
    np.savez_compressed(out/"plans.npz",**plans)
    # Numerical record identities certify the shared parent source training.
    ids = {role:[[dict(shape=list(x.shape),sha256=__import__("hashlib").sha256(
        np.ascontiguousarray(x).tobytes()).hexdigest()) for x in group] for group in data[role]]
           for role in ("fit","select")}
    dump(out/"contract.json",dict(seed=SEED,parent=WEIGHT_SHA,source_shared_across_targets=True,
         source_records=ids,plan_sha256=sha(out/"plans.npz"),source_steps=240,logical_batch=32,
         source_select_steps=list(range(0,241,40)),source_warmup=24,U_steps=2000,U_warmup=200,
         U_batch=32,target_slots=16,TEP_slots=8,SMD_slots=8,
         loss="native copy mean over ten copies, all100 positions, all variables",
         score="native endpoint V, correction1, channel mean, B1 fixed complementary masks",
         source_labels=False,target_labels=False))
    print("shared source contract prepared",flush=True)


def forward(m, arm, window):
    h,x,meta=window
    return m(h,x,**meta,bypass=arm!="history")[1]


@torch.no_grad()
def validate(m,arm,records,plan):
    m.eval()
    rows=[]
    for draw in plan:
        h,x,meta=get_source(records,draw,"cuda")
        y=forward(m,arm,(h,x,meta))
        mse=(y-x[None]).square().mean()
        e=(y.mean(0)-x).square().mean()
        v=y.var(0,correction=1).mean()
        rows.append([mse.item(),e.item(),v.item()])
    return np.mean(rows,axis=0).tolist()


def fit(w,out,arm,domain=None):
    dest=out/("source" if domain is None else domain)/arm
    if (dest/"receipt.json").exists():
        assert sha(dest/"selected.pt")==json.loads((dest/"receipt.json").read_text())["sha256"]
        return
    dest.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    torch.set_num_threads(1)
    m=model(w,arm)
    if domain is not None:
        p=out/"source"/arm/"selected.pt"
        parent_receipt=json.loads(p.with_name("receipt.json").read_text())
        assert sha(p)==parent_receipt["sha256"]
        m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True)["model"])
    opt=optimizer(m)
    data=pools(w)
    with np.load(out/"plans.npz") as z: plan={k:z[k] for k in z.files}
    target_values=None
    if domain:
        _,_,sc,_=target_inputs(w,domain)
        target_values=torch.as_tensor(sc.transform(target(w,domain,"U")).astype(np.float32),device="cuda")
    steps=2000 if domain else 240
    warmup=200 if domain else 24
    curve=[]; select=[]; best=float("inf"); selected=0
    first_step=0
    if (dest/"resume.pt").exists():
        resumed=torch.load(dest/"resume.pt",map_location="cuda",weights_only=True)
        m.load_state_dict(resumed["model"])
        opt.load_state_dict(resumed["optimizer"])
        first_step=resumed["step"]
        curve,select=resumed["curve"],resumed["select"]
        if select:
            choice=min(select,key=lambda row:(row["copy"],row["step"]))
            best,selected=choice["copy"],choice["step"]
    initial={k:v.clone() for k,v in m.native.state_dict().items() if not k.startswith("decoder.")}
    for step in range(first_step,steps+1):
        if domain is None and step%40==0:
            values=validate(m,arm,data["select"],plan["select"])
            select.append(dict(step=step,copy=values[0],E_full=values[1],V_full=values[2]))
            if values[0]<best:
                best,selected=values[0],step
                save(dest/"selected.pt",dict(model=m.state_dict(),step=step,arm=arm))
            print("SELECT",arm,select[-1],flush=True)
        if step==steps: break
        tick=time.monotonic()
        opt.zero_grad(set_to_none=True)
        for g in opt.param_groups: g["lr"]=1e-4*min(1,(step+1)/warmup)
        rows=plan["source"][step] if domain is None else plan["replay"][step]
        loss_total=0.
        for row in rows:
            window=get_source(data["fit"],row,"cuda")
            y=forward(m,arm,window)
            loss=(y-window[1][None]).square().mean()
            (loss/32).backward()
            loss_total+=loss.detach().item()/32
        if domain:
            for start in plan[domain][step]:
                window=record_window(target_values,int(start),(0,len(target_values)),512)
                y=forward(m,arm,window)
                loss=(y-window[1][None]).square().mean()
                (loss/32).backward()
                loss_total+=loss.detach().item()/32
        assert np.isfinite(loss_total)
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
        grad={name:float(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None))
              for name,module in (("history",m.history),("cross",m.cross),("output",m.output),("decoder",m.native.decoder))}
        opt.step()
        torch.cuda.synchronize()
        curve.append(dict(step=step+1,loss=loss_total,seconds=time.monotonic()-tick))
        if step<2 or (step+1)%40==0:
            print(domain or "source",arm,curve[-1],"grad",grad,flush=True)
        if step==0:
            with torch.no_grad():
                h,x,meta=window
                outputs={}
                for kind in ("correct","replacement","bypass"):
                    hnew=h.flip(1) if kind=="replacement" else h
                    y=m(hnew,x,**meta,bypass=kind=="bypass")[1]
                    outputs[kind]=y.var(0,correction=1).mean(-1)[0,-1].item()
            dump(dest/"first_update.json",dict(loss=loss_total,gradient=grad,endpoint_V=outputs,
                 seconds=curve[-1]["seconds"],peak_bytes=torch.cuda.max_memory_allocated(),
                 replacement_is_engineering_time_reversal=True,planned_nonoverlap_diagnostic_separate=True))
        if (step+1)%40==0:
            save(dest/"resume.pt",dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step+1,
                                       curve=curve,select=select))
    if domain:
        selected=steps
        save(dest/"selected.pt",dict(model=m.state_dict(),step=steps,arm=arm))
    assert all(torch.equal(initial[k],m.native.state_dict()[k]) for k in initial)
    dump(dest/"curve.json",curve)
    dump(dest/"select.json",select)
    dump(dest/"receipt.json",dict(arm=arm,domain=domain,steps=steps,selected_step=selected,
         sha256=sha(dest/"selected.pt"),plan_sha256=sha(out/"plans.npz"),parent=WEIGHT_SHA,
         source_checkpoint_sha256=sha(out/"source"/arm/"selected.pt") if domain else None,
         seconds=time.monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated(),
         input_exposures=32*steps,frozen_native_encoder=True,checkpoint_selection="source normal copy MSE; earliest minimum" if not domain else "fixed endpoint",
         GPU=torch.cuda.get_device_name()))
    print("COMPLETE",domain,arm,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","source","U"))
    p.add_argument("workspace",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--arm",choices=("native","history"))
    p.add_argument("--domain",choices=("PSM","SWAT"))
    a=p.parse_args()
    if a.action=="prepare": prepare(a.workspace,a.output)
    else: fit(a.workspace,a.output,a.arm,a.domain if a.action=="U" else None)


if __name__=="__main__":
    main()
