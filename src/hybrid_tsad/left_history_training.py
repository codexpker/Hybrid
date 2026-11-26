"""Matched continuation of archived multisource LEFT on whole 448-point records."""

import argparse
import hashlib
import time

import numpy as np
import torch

from . import left_history_relation as hr
from . import left_multisource as base
from .left_smd_source import source_pools
from .intervention_reference import source_records

STEPS = 2000
BATCH = 32
SELECT_BATCHES = 32


def sample(records, width, rng, digest):
    x = np.empty((BATCH, 448, width), np.float32)
    for k in range(BATCH):
        domain = k % 2
        group = records[domain]
        i = int(rng.integers(len(group)))
        record = group[i]
        start = int(rng.integers(len(record) - 448 + 1))
        channels = rng.choice(record.shape[1], width, replace=False)
        x[k] = record[start : start + 448, channels]
        digest.update(np.array([domain, i, start, *channels], dtype="<i4").tobytes())
    return torch.from_numpy(x).cuda()


def loss_mean(array, valid):
    return (array * valid).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def scales(model, panel):
    model.eval()
    state = base.fingerprint(model.state_dict())
    step = model.left.step
    sums = {}
    counts = {}
    for x in panel:
        native, pieces = hr.raw_batch(model, x)
        sums["native"] = sums.get("native", 0.0) + native.double().sum().item()
        counts["native"] = counts.get("native", 0) + native.numel()
        for key, (err, valid) in pieces.items():
            sums[key] = sums.get(key, 0.0) + (err.double() * valid).sum().item()
            counts[key] = counts.get(key, 0) + valid.sum().item()
    assert state == base.fingerprint(model.state_dict()) and model.left.step == step
    assert min(counts.values()) > 0
    scale = {k: max(sums[k] / counts[k], 1e-6) for k in sums}
    assert np.isfinite(list(scale.values())).all()
    return scale, counts


def check(model, x):
    model.eval()
    x = x[:2]
    with torch.no_grad():
        native = model.left.infer(x[:, 256:].contiguous(), None, None, None)[0].mean(-1)
        assert torch.isfinite(native).all()
        r, valid = hr.correlations(x, torch.tensor([287, 319], device=x.device))
        data = x.cpu().double().numpy()
        errors = []
        for block, start in enumerate((256, 288)):
            for k, lag in enumerate(hr.LAGS):
                i, j = 0, 1
                a, b = (
                    data[0, start : start + 32, i],
                    data[0, start - lag : start + 32 - lag, j],
                )
                if a.std() > hr.MIN_RMS and b.std() > hr.MIN_RMS:
                    assert valid[0, block, k, i, j]
                    errors.append(
                        abs(r[0, block, k, i, j].item() - np.corrcoef(a, b)[0, 1])
                    )
        assert errors and max(errors) < 1e-3, errors
        flat = x.clone()
        flat[:, :, 0] = 2.0
        _, mask = hr.correlations(flat, torch.tensor([350], device=x.device))
        assert not mask[:, :, :, 0, :].any() and not mask[:, :, :, :, 0].any()
    grads = {}
    if model.head:
        predicted = model.predict(x[:, :256])
        altered = x.clone()
        altered[:, 256:] = 0
        with torch.no_grad():
            torch.testing.assert_close(
                predicted, model.predict(altered[:, :256]), rtol=0, atol=0
            )
        del predicted
        err, mask = hr.components(model, x, False)["relation"]
        loss_mean(err, mask).backward()
        for name, module in [
            ("shared_patch", model.left.patch_embedding.value_embedding),
            ("shared_encoder", model.left.encoder),
            ("head", model.head),
        ]:
            g = (
                sum(
                    p.grad.square().sum().item()
                    for p in module.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            assert np.isfinite(g) and g > 0
            grads[name] = g
        model.zero_grad(set_to_none=True)
    return dict(
        reference_checks=len(errors),
        max_correlation_error=max(errors),
        gradient_norms=grads,
    )


def train(w, d, arm):
    start = time.monotonic()
    dest = w / hr.OUT / d / arm
    dest.mkdir(exist_ok=True, parents=True)
    if (dest / "receipt.json").exists():
        hr.load_model(w, d, arm)
        return
    torch.cuda.set_per_process_memory_fraction(
        20 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    model, parent = hr.initial(w, d, arm == "joint")
    records = (source_pools if d == "SMD" else source_records)(w)
    for role in records:
        assert all(len(x) >= 448 for domain in records[role] for x in domain)
    rng = np.random.default_rng(2025)
    val_rng = np.random.default_rng(2026)
    plan = hashlib.sha256()
    val_hash = hashlib.sha256()
    panel = [
        sample(records["select"], hr.WIDTHS[d], val_rng, val_hash)
        for _ in range(SELECT_BATCHES)
    ]
    checks = check(model, panel[0])
    parent_scales = scales(model, panel)[0] if arm == "continue" else None
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    trace = []
    losses = []
    initial_step = model.left.step
    initial_rng = torch.cuda.get_rng_state().clone()
    for step in range(1, STEPS + 1):
        x = sample(records["fit"], hr.WIDTHS[d], rng, plan)
        opt.zero_grad(set_to_none=True)
        left_loss = model.left(x[:, 256:].contiguous(), None, None, None)[0]
        total = left_loss
        extra = x.new_zeros(())
        if arm == "joint":
            # Isolate history-forward dropout draws; advancing LEFT training RNG
            # is reserved for the one current-window native forward each step.
            with torch.random.fork_rng(devices=[x.device]):
                err, valid = hr.components(model, x, False)["relation"]
                extra = loss_mean(err, valid)
            total = left_loss + extra
        total.backward()
        opt.step()
        losses.append((left_loss.item(), extra.item()))
        if step % 200 == 0:
            averages = np.mean(losses, axis=0)
            assert np.isfinite(averages).all()
            trace.append(
                dict(step=step, left=float(averages[0]), relation=float(averages[1]))
            )
            print(d, arm, trace[-1], flush=True)
            losses = []
    assert model.left.step == initial_step + STEPS
    torch.save(
        dict(state=model.state_dict(), step=model.left.step), dest / "selected.pt"
    )
    scale, counts = scales(model, panel)
    base.dump(
        dest / "receipt.json",
        dict(
            domain=d,
            arm=arm,
            seed=2025,
            steps=STEPS,
            initial_step=initial_step,
            endpoint=model.left.step,
            parent=parent["sha256"],
            sha256=base.sha(dest / "selected.pt"),
            checks=checks,
            curve=trace,
            optimizer="Adam lr=1e-4",
            selection="fixed endpoint",
            lambda_relation=1.0,
            training_windows=STEPS * BATCH,
            training_plan_sha256=plan.hexdigest(),
            source_select_plan_sha256=val_hash.hexdigest(),
            scales=scale,
            scale_valid_counts=counts,
            parent_scales=parent_scales,
            native_parameters=sum(p.numel() for p in model.left.parameters()),
            added_parameters=sum(p.numel() for p in model.head.parameters())
            if model.head
            else 0,
            source_domains=["TEP", "HAI" if d == "SMD" else "SMD"],
            device=torch.cuda.get_device_name(),
            peak_bytes=torch.cuda.max_memory_allocated(),
            seconds=time.monotonic() - start,
            native_rng_start=hashlib.sha256(initial_rng.numpy().tobytes()).hexdigest(),
            native_rng_end=hashlib.sha256(
                torch.cuda.get_rng_state().numpy().tobytes()
            ).hexdigest(),
        ),
    )


if __name__ == "__main__":
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("domain", choices=hr.WIDTHS)
    p.add_argument("arm", choices=("joint", "continue"))
    args = p.parse_args()
    train(args.workspace, args.domain, args.arm)
