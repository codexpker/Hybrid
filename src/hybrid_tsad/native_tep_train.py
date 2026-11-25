"""One source checkpoint per arm with identical mixed-domain selection."""
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import native_event_rank_training as old
from .native_position_aligned_training import datasets, PRIOR, PLAN_SHA
from .native_tep_support import OUT, ARMS
from .native_v_conditioning import state_hashes, WEIGHT_SHA
from .native_v_conditioning_runner import read, dump, setup, seed, sha
from .native_v_replay import decoder_model, optimizer


def tep_score(model, array, run, end, device, evidence):
    return old.native_values(model,(array[run],0),[end],device,evidence)[0]


def train(w,arm):
    assert arm in ARMS
    start = time.monotonic()
    root = w/OUT
    meta = read(root/"PLAN.json")
    assert meta["plan_sha256"] == sha(root/"SUPPORT_PLAN.npz") and sha(w/PRIOR/"PLAN.npz")==PLAN_SHA
    plan = old.load_plan(w/PRIOR)
    with np.load(root/"SUPPORT_PLAN.npz") as z:
        draws = dict(z)
    inputs = read(root/"INPUT.json")
    for record in inputs["files"]:
        assert sha(root/(record["object"]+".npy"))==record["derived_sha256"]
    normal = np.load(root/"fault_free_training.npy",mmap_mode="r")[0]
    faulty = np.load(root/"faulty_training.npy",mmap_mode="r")
    fit,select = datasets(w)
    ev = arm[-1]
    scale_dir = w/(PRIOR if ev=="V" else Path("results/native_e_vs_v_20261123_v1"))
    scale_info = read(scale_dir/"SCALE.json")
    assert sha(scale_dir/scale_info["bank_file"]) == scale_info["bank_sha256"]
    assert scale_info["checkpoint_sha256"]==WEIGHT_SHA and scale_info["plan_sha256"]==PLAN_SHA
    scale = scale_info["value"]
    device = setup()
    seed(2025)
    model = decoder_model(w,device)
    frozen = state_hashes(model)
    opt = optimizer(model)
    mass = plan["stratum_sizes"].ravel().astype(float)
    mass /= mass.sum()
    out = root/arm
    out.mkdir()
    best = float("inf")
    trace, validations = [], []

    def pair(p,n):
        return F.softplus(1-torch.log1p(p/scale)+torch.log1p(n/scale))

    def tep_pair(row):
        f,r,p,nr,n = row
        pos,_ = tep_score(model,faulty[f],r,p,device,ev)
        neg,_ = tep_score(model,normal,nr,n,device,ev)
        return pair(pos,neg)

    for step in range(241):
        if step:
            row = step-1
            opt.zero_grad(set_to_none=True)
            for group in opt.param_groups:
                group["lr"] = 1e-4*min(step/24,1)
            counts = np.bincount(draws["copy_domain"][row],minlength=3)
            roles = [(fit[1],plan["swat_fit"][row,:counts[0]])]
            for k in range(counts[1]):
                roles.append((fit[2][plan["smd_entity"][row,k]],[plan["smd_fit"][row,k]]))
            for k in range(counts[2]):
                roles.append(((normal[draws["normal_run"][row,k]],0),[draws["normal_end"][row,k]]))
            copies = []
            for role,endpoints in roles:
                for window in old.windows(role,endpoints,device):
                    _, loss = old.native_endpoint_and_copy_loss(model,window,ev)
                    (loss/32).backward()
                    copies.append(loss.detach())
            shared = draws["shared"][row]
            terms = []
            for k in range(15):
                if arm=="single_V" or shared[k]:
                    p,_ = old.native_values(model,fit[0],[plan["positive_fit"][row,k]],device,ev)[0]
                    n,_ = old.native_values(model,fit[0],[plan["normal_fit"][row,k]],device,ev)[0]
                    coeff = mass[k] if arm=="single_V" else 0.5*mass[k]*15/shared.sum()
                    term = pair(p,n)*coeff
                else:
                    term = tep_pair(draws["tep_pair"][row,k])*(0.5/(~shared).sum())
                term.backward()
                terms.append(term.detach())
            grad = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
            assert np.isfinite(float(grad)) and float(grad)>0
            opt.step()
            trace.append(dict(step=step,copy=float(torch.stack(copies).mean()),pair=float(torch.stack(terms).sum()),grad=float(grad)))
        if step in old.SELECT_STEPS:
            with torch.inference_mode():
                sw = old.validate(model,plan,select,scale,device,step,ev)
                losses = torch.stack([tep_pair(row) for row in draws["select"]]).cpu().numpy()
                risk = (sw["risk"]+losses.astype(float).mean())/2
            validations.append(dict(step=step,risk=risk,swat=sw,tep_mean=float(losses.mean()),
                                    tep_by_fault=losses.reshape(20,-1).mean(1).tolist()))
            if risk < best:
                best = risk
                best_step = step
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(arm,step,risk,best_step,flush=True)
    final = state_hashes(model)
    assert all(final[k]==v for k,v in frozen.items() if not k.startswith("decoder."))
    assert any(final[k]!=v for k,v in frozen.items() if k.startswith("decoder."))
    checkpoint = out/"selected.pt"
    torch.save(dict(model=state,update=best_step,arm=arm),checkpoint)
    dump(out/"TRAIN_TRACE.json",trace)
    dump(out/"SELECTION.json",dict(arm=arm,selected_step=best_step,scale=scale,updates=240,selection=validations,
        selected_checkpoint_sha256=sha(checkpoint),plan_sha256=meta["plan_sha256"],trainable="decoder",
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-start,target_read=False))


if __name__=="__main__":
    train(Path.cwd(),sys.argv[1])
