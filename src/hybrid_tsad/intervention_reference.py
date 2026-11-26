"""Synthetic source pairs and matched absolute/local target reference learning.

All learner inputs are public numeric records. No evaluation labels are loaded here.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch_frft.frft_module import ifrft

from .left_smd28_author import bindings, checked_data
from .native_v_conditioning import fixed_native_rng, state_hashes, WEIGHT_SHA
from .native_v_conditioning_runner import setup
from .native_v_replay import decoder_model, optimizer
from .trajectory_experiment import TEP, ROLES, sha, checked, dump, target, standardizer

ROOT = Path("results/intervention_reference_20261126_v1")
STEPS = 120
BATCH = 4
SUPPORT = slice(40, 60)
SOURCE_ARMS = ("reconstruction", "pair")
U_ARMS = ("replay", "absolute", "relative")


def source_records(w):
    tep = checked(w/TEP, "2c4e24dc41b14a7a521de71a8bbac87cef3788fcf46c9dd321aab5a14f38c698")[0]
    pools = {"fit": [list(tep[:20]), []], "select": [list(tep[20:]), []]}
    for _, record in sorted(bindings(w).items()):
        x = checked_data(w, record, "train")
        k = len(x)*4//5
        mu, sd = standardizer(x[:k])
        for role, part in (("fit", x[:k]), ("select", x[k:])):
            pools[role][1].append(((part-mu)/sd).astype(np.float32))
    return pools


def sample(records, rng):
    record = records[rng.integers(len(records))]
    start = rng.integers(len(record)-99)
    return np.array(record[start:start+100], dtype=np.float32, copy=True)


def edit(x, kind, rng):
    y = x.copy()
    cols = rng.choice(x.shape[1], size=max(1, (x.shape[1]+3)//4), replace=False)
    sign = rng.choice([-1, 1])
    envelope = np.sin(np.pi*(np.arange(20)+0.5)/20)**2
    if kind == 0:
        change = sign * rng.uniform(1, 2) * envelope[:, None]
    else:
        lag = int(rng.choice([3, 6]))
        change = envelope[:, None]*(x[40+sign*lag:60+sign*lag, cols]-x[SUPPORT, cols])
    y[SUPPORT, cols] += change
    return y


def pairs(records, stream, step, batch=BATCH, source=True):
    """One independent reproducible stream per stage/step. Balance type within each source."""
    rng = np.random.default_rng(np.random.SeedSequence([2025, stream, step]))
    for j in range(batch):
        x = sample(records[j % 2] if source else records, rng)
        y = edit(x, (j//2+step) % 2, rng)
        yield x, y


def tensor(x):
    return torch.as_tensor(x, device="cuda").unsqueeze(0)


def forward(model, x):
    with fixed_native_rng(x.device):
        _, copies = model(x, norm=0, mask_mode="c", copies=10)
    score = model.cal_anomaly_score(x, copies)[0]
    return score, (copies-x.unsqueeze(0)).square().mean()


def target_term(v, edited, frozen, scale, reference):
    """A uniform raw V shift cancels before the nonlinear training transform."""
    corrected = (edited-v+frozen)/scale
    rank = F.softplus(1+torch.asinh(reference/scale)-torch.asinh(corrected)).mean()
    anchor = F.smooth_l1_loss(torch.asinh(v/scale), torch.asinh(frozen/scale))
    return rank, anchor


def source_loss(model, data, scale, use_pair):
    x, y = data
    v, mse = forward(model, tensor(x))
    if not use_pair:
        return mse, (float(mse.detach()), 0.)
    edited, _ = forward(model, tensor(y))
    pair = F.softplus(1+torch.asinh(v[SUPPORT]/scale)-torch.asinh(edited[SUPPORT]/scale)).mean()
    return mse+pair, (float(mse.detach()), float(pair.detach()))


@torch.no_grad()
def source_median(model, pools):
    values = [forward(model, tensor(x))[0][SUPPORT].cpu().numpy()
              for x, _ in pairs(pools, 10, 0, 64)]
    return max(float(np.median(values)), 1e-8)


def load_decoder(model, path):
    model.decoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def learn(w, arm, domain):
    started = time.monotonic()
    out = w/ROOT
    source = arm in SOURCE_ARMS
    dest = out/("source" if source else domain)/arm
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/"receipt.json").exists():
        raise FileExistsError(dest)
    device = setup()
    model = decoder_model(w, device)
    pools = source_records(w)
    scale = source_median(model, pools["fit"])
    val = list(pairs(pools["select"], 11, 0, 56))
    teacher = None
    if not source:
        checkpoint = out/"source/pair/selected.pt"
        load_decoder(model, checkpoint)
        teacher = copy.deepcopy(model).requires_grad_(False)
        fixed = source_median(teacher, pools["fit"])
        mu, sd = standardizer(target(w, domain, "FIT"))
        mixed = [((target(w, domain, "U")-mu)/sd).astype(np.float32)]
    initial = state_hashes(model)
    opt = optimizer(model)
    best, selected = float("inf"), 0
    history, validation = [], []
    for step in range(STEPS+1):
        if source and step % 20 == 0:
            with torch.no_grad():
                loss = np.mean([source_loss(model, xy, scale, True)[1] for xy in val], axis=0)
            risk = float(loss[1])
            validation.append(dict(step=step, mse=float(loss[0]), risk=risk))
            if risk < best:
                best, selected = risk, step
                torch.save(model.decoder.state_dict(), dest/"selected.pt")
        if step == STEPS:
            break
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["lr"] = 1e-4*min(1., (step+1)/12)
        stats = []
        for xy in pairs(pools["fit"], 12 if source else 13, step):
            # Ordinary replay uses only copy reconstruction on each half.
            loss, components = source_loss(model, xy, scale, arm != "reconstruction" and arm != "replay")
            (loss/(BATCH if source else 2*BATCH)).backward()
            stats.append([*components, float(np.abs(xy[1]-xy[0]).mean())])
        diagnostics = []
        if not source:
            for x, y in pairs(mixed, 21, step, source=False):
                v, mse = forward(model, tensor(x))
                if arm == "replay":
                    loss = mse
                else:
                    with torch.no_grad():
                        ref, _ = forward(teacher, tensor(x))
                    ev, _ = forward(model, tensor(y))
                    r = ref[SUPPORT] if arm == "relative" else torch.full_like(ref[SUPPORT], fixed)
                    pair, anchor = target_term(v[SUPPORT], ev[SUPPORT], ref[SUPPORT], scale, r)
                    # Anchor all positions, use only edited support for the ranking term.
                    anchor = F.smooth_l1_loss(torch.asinh(v/scale), torch.asinh(ref/scale))
                    loss = pair+anchor
                    diagnostics.append([float(t.detach()) for t in
                        (v.mean(), ref.mean(), ev[SUPPORT].mean(), v[SUPPORT].mean(), pair, anchor)]
                        +[float(np.abs(x-y).mean())])
                (loss/(2*BATCH)).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), float("inf"), error_if_nonfinite=True)
        opt.step()
        history.append(dict(step=step+1, source_mse_pair_edit=np.mean(stats,0).tolist(),
                            target_v_teacher_edited_original_pair_anchor_edit=np.mean(diagnostics,0).tolist() if diagnostics else [],
                            grad=float(norm)))
    if not source:
        selected = STEPS
        torch.save(model.decoder.state_dict(),dest/"selected.pt")
    final = state_hashes(model)
    assert all(initial[k]==final[k] for k in initial if not k.startswith("decoder."))
    dump(dest/"trace.json", dict(training=history, source_select=validation))
    dump(dest/"receipt.json",dict(arm=arm,domain=domain,selected=selected,steps=STEPS,
         scale=scale,absolute_reference=None if source else fixed,seconds=time.monotonic()-started,
         device=torch.cuda.get_device_name(),peak_bytes=torch.cuda.max_memory_allocated(),
         checkpoint_sha256=sha(dest/"selected.pt"),official_sha256=WEIGHT_SHA,
         parent_sha256=WEIGHT_SHA if source else sha(checkpoint),labels_read=False))
    print(dest,selected,flush=True)


def score(w, domain, role):
    """Share one actual native encoder call across the frozen decoder states."""
    start = time.monotonic()
    model = decoder_model(w, setup())
    out = w/ROOT
    arms = ["official", *SOURCE_ARMS] if role == "TEST" else ["pair", *U_ARMS]
    decoders = {}
    for arm in arms:
        if arm != "official":
            load_decoder(model,out/("source" if arm in SOURCE_ARMS else domain)/arm/"selected.pt")
        decoders[arm] = copy.deepcopy(model.decoder).requires_grad_(False)
    model.decoder = decoders[arms[0]]
    orders = []
    handle = model.net.register_forward_hook(lambda m,args,y: orders.append(y.squeeze()))
    mu,sd = standardizer(target(w,domain,"FIT"))
    raw = target(w,domain,role)
    lo,hi = ROLES[domain][role]
    first = ((lo+99)//100)*100
    starts = np.arange(first, hi-99, 100)
    scores = {a:[] for a in arms}
    with torch.no_grad():
        for i,t in enumerate(starts):
            x = tensor(((raw[t-lo:t-lo+100]-mu)/sd).astype(np.float32))
            orders.clear()
            with fixed_native_rng(x.device):
                h, base = model(x,norm=0,mask_mode="c",copies=10)
            for arm, decoder in decoders.items():
                if arm == arms[0]:
                    outputs = base
                else:
                    outputs = ifrft(decoder(h),orders[0],dim=-1).abs().reshape(10,1,-1,100).transpose(-1,-2)
                v = model.cal_anomaly_score(x,outputs)[0]
                if i == 0:
                    torch.testing.assert_close(v, outputs.var(0,correction=1)[0].mean(-1))
                    if arm != arms[0]:
                        model.decoder = decoder
                        direct,_ = forward(model,x)
                        torch.testing.assert_close(v,direct,rtol=1e-5,atol=1e-6)
                        model.decoder = decoders[arms[0]]
                scores[arm].append(v.cpu().numpy())
    handle.remove()
    dest = out/domain/"banks"
    dest.mkdir(parents=True,exist_ok=True)
    bank = dest/(role+".npz")
    index = (starts[:,None]+np.arange(100)).ravel()
    np.savez_compressed(bank,index=index,available=np.repeat(starts+99,100),
        **{k:np.concatenate(v) for k,v in scores.items()})
    dump(bank.with_suffix(".json"),dict(sha256=sha(bank),seconds=time.monotonic()-start,
         arms=arms,models={k:sha(out/("source" if k in SOURCE_ARMS else domain)/k/"selected.pt")
         if k!="official" else WEIGHT_SHA for k in arms},points=len(index)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace",type=Path)
    parser.add_argument("action",choices=("train","score"))
    parser.add_argument("arm_or_role",choices=(*SOURCE_ARMS,*U_ARMS,"TEST","CAL","QUERY"))
    parser.add_argument("domain",choices=("source","PSM","SWAT"))
    args = parser.parse_args()
    if args.action=="train":
        learn(args.workspace,args.arm_or_role,args.domain)
    else:
        score(args.workspace,args.domain,args.arm_or_role)
