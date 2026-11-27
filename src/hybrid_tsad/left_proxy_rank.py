"""Matched normal continuation and static proxy ranking on the native LEFT score."""

import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import left_multisource as base
from .intervention_reference import edit

OUT = Path("results/left_proxy_rank_20261127_v1")
SEED = 2025
STEPS = 2000
ARMS = ("C", "M1")
PAIR_WEIGHT = 0.1
EDIT_LENGTH = 20
MAX_SHIFT = 6
LR = 1e-4


def stream(purpose):
    return np.random.default_rng(np.random.SeedSequence([SEED, 20261127, purpose]))


def score(model, x):
    # Preserve the official score while permitting backward through its inputs.
    assert not model.training
    ms_gt, ms_dec, lengths, from_freq, from_time, x, _, _, _, ms, cache = model._forward(
        x, None, None, None
    )
    return model._ms_anomaly_score(ms_dec, ms_gt, lengths, from_freq, from_time, x, ms, cache)


def pairs(clean, rng, step):
    starts = rng.integers(MAX_SHIFT, base.WINDOW - EDIT_LENGTH - MAX_SHIFT + 1, len(clean))
    kinds = (np.arange(len(clean)) // 2 + step) % 2
    edited = np.stack([
        edit(x, int(kind), rng, start=int(start))
        for x, kind, start in zip(clean, kinds, starts)
    ])
    positions = starts[:, None] + np.arange(EDIT_LENGTH)
    return edited, positions, kinds


def pair_loss(model, clean, edited, positions, scale):
    pairs_in = torch.cat((clean, edited))
    point = score(model, pairs_in).mean(dim=-1)
    a, b = (part.gather(1, positions) for part in point.chunk(2))
    gap = torch.asinh(b / scale) - torch.asinh(a / scale)
    return F.softplus(1.0 - gap).mean(), gap.detach().mean()


def start_model(w, d, device):
    model = base.initialize(w, d, device=device)
    parent = base.restore(w, model, d)
    return model, parent["sha256"]


def calibration_scale(model, fit, width, device):
    windows = base.source_batch(fit, width, 64, stream(2))
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(windows).to(device)
        points = model.infer(x, None, None, None)[0].mean(dim=-1)
    median = float(np.median(points.cpu().numpy()))
    assert np.isfinite(median)
    return max(median, 1e-6), hashlib.sha256(windows.tobytes()).hexdigest()


def select_loss(model, select, width, device):
    rng = np.random.default_rng(SEED + 1)
    model.eval()
    with torch.no_grad():
        values = [
            model(torch.from_numpy(base.source_batch(select, width, base.BATCH, rng)).to(device),
                  None, None, None)[0].item()
            for _ in range(32)
        ]
    return float(np.mean(values))


def rng_devices(device):
    return [torch.cuda.current_device()] if device.type == "cuda" else []


def step(model, opt, clean, device, *, edit_rng=None, iteration=0, scale=None, timing=None):
    count = model.step
    model.train()
    opt.zero_grad(set_to_none=True)
    x = torch.from_numpy(clean).to(device)
    native = model(x, None, None, None)[0]
    native.backward()
    assert model.step == count + 1
    pair = gap = None
    if edit_rng is not None:
        if timing is not None:
            t_edit = time.monotonic()
        edited, positions, _ = pairs(clean, edit_rng, iteration)
        if timing is not None:
            timing["edit_seconds"] += time.monotonic() - t_edit
            events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            events[0].record()
        # Read gradients from the post-EMA parameters and eval score without
        # another curriculum tick, EMA write or change to native dropout RNG.
        with torch.random.fork_rng(devices=rng_devices(device)):
            model.eval()
            pair, gap = pair_loss(model, x, torch.from_numpy(edited).to(device),
                                  torch.from_numpy(positions).to(device), scale)
            if timing is not None:
                events[1].record()
            (PAIR_WEIGHT * pair).backward()
            if timing is not None:
                events[2].record()
                timing["events"].append(events)
            model.train()
    opt.step()
    assert model.step == count + 1
    return [native.detach()] if pair is None else [native.detach(), pair.detach(), gap]


def check(w, d, device, dest):
    started = time.monotonic()
    model, parent = start_model(w, d, device)
    width = base.WIDTH[d]
    fit = base.source_records(w)["fit"]
    scale, scale_hash = calibration_scale(model, fit, width, device)
    clean = base.source_batch(fit, width, base.BATCH, stream(1))
    edited, positions, kinds = pairs(clean, stream(3), 1)
    support = (np.arange(base.WINDOW)[None, :] >= positions[:, :1]) & (
        np.arange(base.WINDOW)[None, :] <= positions[:, -1:]
    )
    np.testing.assert_array_equal(edited[~support], clean[~support])
    assert np.bincount(kinds).tolist() == [16, 16]
    for group in (kinds[::2], kinds[1::2]):
        assert np.bincount(group).tolist() == [8, 8]
    x = torch.from_numpy(clean).to(device)
    y = torch.from_numpy(edited).to(device)
    index = torch.from_numpy(positions).to(device)
    before = model.step
    model.train()
    native = model(x, None, None, None)[0]
    native.backward()
    assert model.step == before + 1
    params = dict(model.named_parameters())
    active = [name for name, value in params.items() if value.grad is not None]
    state, counter = base.fingerprint(model.state_dict()), model.step
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if device.type == "cuda" else None
    with torch.random.fork_rng(devices=rng_devices(device)):
        model.eval()
        xy = torch.cat((x, y))
        actual = score(model, xy)
        reference = model.infer(xy, None, None, None)[0]
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
        pair, gap = pair_loss(model, x, y, index, scale)
        grads = torch.autograd.grad(pair, tuple(params.values()), allow_unused=True)
        present = {name: grad for name, grad in zip(params, grads) if grad is not None}
        assert all(torch.isfinite(g).all() for g in present.values())
        grad_norm = torch.stack([g.norm() for g in present.values()]).norm().item()
        group_norms = {}
        for group in ("Et", "Ef", "Dt", "Df", "proto_t", "proto_f", "ms_decoder"):
            members = [grad.norm() for name, grad in present.items() if name.startswith(group + ".")]
            group_norms[group] = torch.stack(members).norm().item() if members else 0.0
        assert grad_norm > 0 and group_norms["ms_decoder"] > 0, group_norms
        if model.alpha_cycle_s > 0:
            assert all(n > 0 for n in group_norms.values()), group_norms
    assert model.step == counter and base.fingerprint(model.state_dict()) == state
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    result = dict(
        domain=d, device=str(device), parent=parent, scale=scale, scale_inputs_sha256=scale_hash,
        inference_max_abs=float((actual.detach() - reference).abs().max()),
        native_step_increment=counter - before, pair_step_increment=model.step - counter,
        pair_parameters_unchanged=True, persistent_state_unchanged=True, cpu_rng_equal=True,
        cuda_rng_equal=True if cuda_rng is not None else None, pair_grad_norm=grad_norm,
        pair_grad_group_norms=group_norms, native_grad_tensors=len(active),
        pair_grad_tensors=len(present), pair_loss=float(pair.detach()), pair_gap=float(gap),
        actual_edited_windows=int(np.any(edited != clean, axis=(1, 2)).sum()),
        alpha_cycle_s=model.alpha_cycle_s, labels_read=False,
    )
    # Validate the exact train step and profile each arm at the real batch size.
    if device.type == "cuda":
        del actual, reference, grads, present, pair, native
        result["gpu"] = torch.cuda.get_device_name()
        for arm in ARMS:
            model, _ = start_model(w, d, device)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            norm_rng, edit_rng = stream(1), stream(3) if arm == "M1" else None
            for i in range(2):
                batch = base.source_batch(fit, width, base.BATCH, norm_rng)
                step(model, opt, batch, device, edit_rng=edit_rng, iteration=i+1, scale=scale)
            torch.cuda.synchronize()
            t0 = time.monotonic()
            for i in range(8):
                batch = base.source_batch(fit, width, base.BATCH, norm_rng)
                vals = step(model, opt, batch, device, edit_rng=edit_rng, iteration=i+3, scale=scale)
                assert torch.isfinite(torch.stack(vals)).all()
            torch.cuda.synchronize()
            result[arm + "_seconds_per_update"] = (time.monotonic() - t0) / 8
        result["peak_bytes"] = torch.cuda.max_memory_allocated()
    result.update(seconds=time.monotonic()-started, pid=os.getpid(),
                  visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"))
    base.dump(dest, result)
    print(result, flush=True)


def train_source(w, d, arm, device, dest):
    t0 = time.monotonic()
    model, parent = start_model(w, d, device)
    initial_step = model.step
    width = base.WIDTH[d]
    pools = base.source_records(w)
    scale, scale_hash = (calibration_scale(model, pools["fit"], width, device)
                         if arm == "M1" else (None, None))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    norm_rng, edit_rng = stream(1), stream(3) if arm == "M1" else None
    digest = hashlib.sha256()
    curve = [dict(update=0, select=select_loss(model, pools["select"], width, device))]
    n_vals = 3 if edit_rng is not None else 1
    totals = torch.zeros(n_vals, device=device)
    timing = {"edit_seconds": 0.0, "events": []}
    for i in range(1, STEPS + 1):
        batch = base.source_batch(pools["fit"], width, base.BATCH, norm_rng)
        digest.update(batch.tobytes())
        vals = step(model, opt, batch, device, edit_rng=edit_rng, iteration=i, scale=scale,
                    timing=timing if edit_rng is not None else None)
        totals += torch.stack(vals)
        if i % 400 == 0:
            means = (totals / 400).tolist()
            if not np.isfinite(means).all():
                raise FloatingPointError(f"{d} {arm} {i}: {means}")
            row = dict(update=i, native=means[0], curriculum=model._curriculum_lambda(),
                       select=select_loss(model, pools["select"], width, device))
            if edit_rng is not None:
                row.update(pair=means[1], transformed_gap=means[2])
            assert np.isfinite(row["select"])
            curve.append(row)
            print(d, arm, row, flush=True)
            totals.zero_()
    assert model.step == initial_step + STEPS
    torch.cuda.synchronize()
    base.save(model, dest)
    extra = STEPS * base.BATCH if arm == "M1" else 0
    return dict(updates=STEPS, initial_step=initial_step, model_step=model.step, parent=parent,
                normal_plan_sha256=digest.hexdigest(), scale=scale, scale_inputs_sha256=scale_hash,
                normal_exposure=STEPS * base.BATCH, pair_clean_exposure=extra,
                pair_edited_exposure=extra, magnitude_pairs=extra//2, shift_pairs=extra//2,
                edit_seconds=timing["edit_seconds"], source_seconds=time.monotonic()-t0,
                pair_forward_seconds=sum(a.elapsed_time(b) for a, b, _ in timing["events"])/1000,
                pair_backward_seconds=sum(b.elapsed_time(c) for _, b, c in timing["events"])/1000,
                curve=curve, target_exposure=0)


def load_source(model, source):
    receipt = base.read(source / "receipt.json")
    path = source / "selected.pt"
    assert base.sha(path) == receipt["sha256"]
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["state"], strict=True)
    model.step = ck["step"]
    assert model.step == receipt["model_step"]
    return receipt


def run(w, d, arm, stage):
    root = w / OUT / d
    dest = root / f"{arm}_{stage}"
    if (dest / "receipt.json").exists():
        return
    dest.mkdir(exist_ok=False)
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name()
    t0 = time.monotonic()
    if stage == "Z":
        result = train_source(w, d, arm, device, dest)
    else:
        result = base.mixed(w, d, "replay", dest,
                            initial=lambda model: load_source(model, root / f"{arm}_Z"))
    base.dump(dest / "receipt.json", dict(
        domain=d, arm=arm, stage=stage, seed=SEED, gpu=gpu, pid=os.getpid(),
        visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"), torch_version=torch.__version__,
        peak_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-t0,
        sha256=base.sha(dest / "selected.pt"), labels_read=False, **result,
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("domain", choices=base.WIDTH)
    p.add_argument("stage", choices=("check", "Z", "U", "score"))
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--score-stage", choices=("Z", "U"))
    args = p.parse_args()
    w, d, stage, arm = args.workspace, args.domain, args.stage, args.arm
    root = w / OUT / d
    root.mkdir(parents=True, exist_ok=True)
    if stage == "check":
        dest = root / f"check_{args.device}.json"
        if not dest.exists():
            check(w, d, torch.device(args.device), dest)
    elif arm is None:
        p.error("--arm is required for Z, U, and score")
    elif stage == "score":
        from .left_multisource_eval import bank
        for checkpoint in ((args.score_stage,) if args.score_stage else ("Z", "U")):
            bank(w, d, f"{arm}_{checkpoint}", "QUERY", root=OUT)
    else:
        run(w, d, arm, stage)


if __name__ == "__main__":
    main()
