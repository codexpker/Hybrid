"""Fixed paired source training and native-V evaluation for history TimeRadar."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .native_v_conditioning import load_native, sha, state_hashes, WEIGHT_SHA
from .native_v_replay import optimizer
from .timeradar_history_context import HistoryConditionedTimeRadar, record_window
from .intervention_reference import source_records
from .trajectory_experiment import ROLES, target, dump
from .left_multisource import target_inputs

SEED = 2025
FROZEN_HISTORY = "frozen_history"


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
    return m(h,x,**meta,bypass=arm=="native")[1]


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


def fit(w,out,arm,domain=None,reference=None,*,smd_fold=False):
    frozen_history = arm == FROZEN_HISTORY
    if frozen_history and (domain is None or (reference is None and not smd_fold)):
        raise ValueError("Frozen-History U requires a target and the sealed A run")
    dest=out/("source" if domain is None else domain)/arm
    if (dest/"receipt.json").exists():
        assert sha(dest/"selected.pt")==json.loads((dest/"receipt.json").read_text())["sha256"]
        return
    dest.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    torch.set_num_threads(1)
    m=model(w,arm)
    if domain is not None:
        p=out/"source"/("history" if frozen_history else arm)/"selected.pt"
        parent_receipt=json.loads(p.with_name("receipt.json").read_text())
        assert sha(p)==parent_receipt["sha256"]
        m.load_state_dict(torch.load(p,map_location="cpu",weights_only=True)["model"])
    opt=optimizer(m)
    if frozen_history:
        source_hashes = state_hashes(m)
        trainable = {name for name, p in m.named_parameters() if p.requires_grad}
        decoder = {name for name, _ in m.named_parameters() if name.startswith("native.decoder.")}
        optimizer_params = [id(p) for group in opt.param_groups for p in group["params"]]
        assert trainable == decoder and len(optimizer_params) == len(decoder)
        assert set(optimizer_params) == {id(p) for name, p in m.named_parameters() if name in decoder}
        assert not opt.state and all(not module.training for module in m.modules())
        if not smd_fold:
            a_receipt = json.loads((reference/domain/"history"/"receipt.json").read_text())
            a_part = json.loads((reference/domain/"scores"/"U_history"/"part0.json").read_text())
            assert sha(out/"plans.npz") == sha(reference/"plans.npz") == a_receipt["plan_sha256"]
            assert sha(p) == sha(reference/"source"/"history"/"selected.pt")
            assert a_receipt["steps"] == a_receipt["selected_step"] == 2000
    if smd_fold:
        from .timeradar_history_smd import load_fold, smd_values
        data, contract = load_fold(w,out)
    else:
        data=pools(w)
    with np.load(out/"plans.npz") as z: plan={k:z[k] for k in z.files}
    target_values=None
    if domain:
        if smd_fold:
            x, scaler_binding = smd_values(w,contract,domain,"U")
        else:
            _,_,sc,scaler_binding=target_inputs(w,domain)
            x=sc.transform(target(w,domain,"U")).astype(np.float32)
        target_values=torch.as_tensor(x,device="cuda")
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
    if frozen_history:
        if not smd_fold:
            assert scaler_binding == a_part["normal_values_sha256"]
        assert plan["replay"].shape == (2000, 16, 3) and plan[domain].shape == (2000, 16)
        assert (plan[domain]>=512).all() and (plan[domain]+100<=len(target_values)).all()
        if first_step:
            assert all(source_hashes[k] == v for k, v in state_hashes(m).items() if not k.startswith("native.decoder."))
        call_counts = dict.fromkeys(("past_context", "query", "cross", "output"), 0)
        hooks = []
        if not first_step:
            def count_forward(name):
                def record(*_):
                    call_counts[name] += 1
                return record
            for name, module in (("past_context", m.history), ("query", m.query), ("cross", m.cross), ("output", m.output)):
                hooks.append(module.register_forward_hook(count_forward(name)))
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
            if frozen_history:
                for hook in hooks:
                    hook.remove()
                assert all(count == 32 for count in call_counts.values())
                assert grad["decoder"] > 0
                assert all(p.grad is None for n,p in m.named_parameters() if n not in decoder)
                updated = state_hashes(m)
                assert all(updated[k] == v for k,v in source_hashes.items() if not k.startswith("native.decoder."))
                assert any(updated[k] != v for k,v in source_hashes.items() if k.startswith("native.decoder."))
                a_first = json.loads(((out if smd_fold else reference)/domain/"history"/"first_update.json").read_text())
                # Differentiable and frozen paths may round differently at float32 precision.
                relative_tolerance = float(np.finfo(np.float32).eps) if smd_fold else 0.0
                np.testing.assert_allclose(loss_total, a_first["loss"], atol=1e-7, rtol=relative_tolerance)
                dump(dest/"update_scope_check.json", dict(
                    trained_names=sorted(decoder), trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
                    optimizer_decoder_only=True, fresh_adamw=True, frozen_gradients_absent=True,
                    post_first_update_frozen_hashes_equal=True, decoder_changed=True, history_forward_calls=call_counts,
                    copy_loss=loss_total, archived_A_copy_loss=a_first["loss"], native_eval=True,
                    first_copy_loss_absolute_difference=abs(loss_total-a_first["loss"]),
                    first_copy_loss_atol=1e-7, first_copy_loss_rtol=relative_tolerance,
                    target_normal_prefix_sha256=scaler_binding, plan_sha256=sha(out/"plans.npz"),
                    mask_seed=84551, sampling_seed=SEED))
                print("UPDATE_SCOPE_OK", domain, call_counts, flush=True)
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
    if frozen_history:
        final_hashes = state_hashes(m)
        fixed_names = sorted(k for k in source_hashes if not k.startswith("native.decoder."))
        assert all(source_hashes[k] == final_hashes[k] for k in fixed_names)
        assert any(source_hashes[k] != final_hashes[k] for k in decoder)
        source_state = torch.load(p,map_location="cpu",weights_only=True)["model"]
        final_state = m.state_dict()
        decoder_l2 = sum((final_state[k].cpu()-source_state[k]).abs().square().sum().item() for k in decoder)**.5
        values = validate(m,arm,data["select"],plan["select"])
        select.append(dict(step=steps,copy=values[0],E_full=values[1],V_full=values[2],diagnostic_only=True))
        dump(dest/"parameter_audit.json", dict(
            frozen_state_before={k:source_hashes[k] for k in fixed_names},
            frozen_state_after={k:final_hashes[k] for k in fixed_names},
            decoder_changed_tensor_count=sum(source_hashes[k]!=final_hashes[k] for k in decoder),
            decoder_L2_displacement=decoder_l2, history_L2_displacement=0.0,
            history_forward_active=True, native_eval=True))
    dump(dest/"curve.json",curve)
    dump(dest/"select.json",select)
    dump(dest/"receipt.json",dict(arm=arm,domain=domain,steps=steps,selected_step=selected,
         sha256=sha(dest/"selected.pt"),plan_sha256=sha(out/"plans.npz"),parent=WEIGHT_SHA,
         source_checkpoint_sha256=sha(p) if domain else None,
         seconds=time.monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated(),
         input_exposures=32*steps,frozen_native_encoder=True,checkpoint_selection="source normal copy MSE; earliest minimum" if not domain else "fixed endpoint",
         GPU=torch.cuda.get_device_name()))
    print("COMPLETE",domain,arm,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","source","U"))
    p.add_argument("workspace",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--arm",choices=("native","history",FROZEN_HISTORY))
    p.add_argument("--reference",type=Path)
    p.add_argument("--domain",choices=("PSM","SWAT"))
    a=p.parse_args()
    if a.action=="prepare": prepare(a.workspace,a.output)
    else: fit(a.workspace,a.output,a.arm,a.domain if a.action=="U" else None,a.reference)


if __name__=="__main__":
    main()
