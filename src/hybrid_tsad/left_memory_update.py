"""Native LEFT replay, with an optional source-only explicit EMA write rule."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import left_multisource as base

OUT = Path("results/left_memory_exposure_20261126_v1")
ARMS = (
    "restricted_A",
    "ordinary_B",
    "replay_B",
    "restricted_B",
    "ordinary_C",
    "replay_C",
    "restricted_C",
)


def selector(model, native, err, js, conf_t, conf_f, qerr, qjs):
    chosen = native
    if model.source_writes:
        qerr = torch.quantile(err[16:], model.update_q_err)
        qjs = torch.quantile(js[16:], model.update_q_ad)
        chosen = (
            (err <= qerr)
            & (js <= qjs)
            & (conf_t >= model.conf_threshold)
            & (conf_f >= model.conf_threshold)
        )
        chosen[:16] = False
    model.ema_observation = (native.clone(), chosen.clone(), qerr, qjs)
    return chosen


def setup(w, d, arm):
    model = base.initialize(w, d, selector)
    parent = base.restore(w, model, d)
    model.source_writes = arm.startswith("restricted")
    return model.train(), torch.optim.Adam(model.parameters(), lr=1e-4), parent


def batches(w, d, arm):
    method, condition = arm.split("_")
    _, _, scaler, _ = base.target_inputs(w, d)
    x = scaler.transform(base.target(w, d, "U")).astype(np.float32)
    with np.load(w / OUT / d / "plan" / (condition + ".npz")) as p:
        first, second = p["first"], p["second"]
    assert first.shape == second.shape == (2000, 16)
    ix = np.arange(base.WINDOW)
    if method != "ordinary":
        pool = base.source_records(w)["fit"]
        rs = np.random.default_rng(np.random.SeedSequence(2025).spawn(3)[2])
    for a, b in zip(first, second):
        other = (
            x[b[:, None] + ix]
            if method == "ordinary"
            else base.source_batch(pool, base.WIDTH[d], 16, rs)
        )
        yield torch.from_numpy(np.concatenate([x[a[:, None] + ix], other])).cuda()


def prototypes(model):
    return torch.stack(
        [model.proto_t.prototypes.detach(), model.proto_f.prototypes.detach()]
    )


def advance(model, opt, x):
    count = model.step
    opt.zero_grad(set_to_none=True)
    start = prototypes(model)
    loss = model(x, None, None, None)[0]
    middle = prototypes(model)
    loss.backward()
    grad = torch.stack(
        [bank.prototypes.grad.norm() for bank in (model.proto_t, model.proto_f)]
    )
    opt.step()
    end = prototypes(model)
    native, chosen, qerr, qjs = model.ema_observation
    assert model.step == count + 1
    shifts = (middle - start).norm(dim=-1)
    values = torch.cat(
        [
            torch.stack([loss.detach(), qerr, qjs]),
            shifts.norm(dim=-1),
            (end - middle).flatten(1).norm(dim=-1),
            grad,
            (shifts > 0).sum(dim=-1),
        ]
    )
    return (torch.stack([native, chosen]).cpu().numpy(), values.detach().cpu().numpy())


FIELDS = (
    "loss",
    "qerr",
    "qjs",
    "ema_t",
    "ema_f",
    "adam_t",
    "adam_f",
    "grad_t",
    "grad_f",
    "changed_rows_t",
    "changed_rows_f",
)


def check(w, d):
    batch = next(batches(w, d, "restricted_A"))
    original = base.initialize(w, d)
    base.restore(w, original, d)
    original.train()
    ref_loss = base.update(
        original, torch.optim.Adam(original.parameters(), lr=1e-4), batch.cpu().numpy()
    )
    rng = torch.cuda.get_rng_state()
    ordinary, opt, _ = setup(w, d, "replay_B")
    mask, metrics = advance(ordinary, opt, batch)
    assert np.array_equal(mask[0], mask[1]) and torch.equal(
        torch.cuda.get_rng_state(), rng
    )
    assert abs(ref_loss - metrics[0]) < 1e-6
    error = max(
        (p - q).abs().max().item()
        for p, q in zip(original.parameters(), ordinary.parameters())
    )
    assert error < 2e-5, error
    restricted, opt, _ = setup(w, d, "restricted_A")
    masks, v = advance(restricted, opt, batch)
    assert (
        not masks[1, :16].any()
        and (v[5:9] > 0).all()
        and torch.equal(torch.cuda.get_rng_state(), rng)
    )
    assert abs(ref_loss - v[0]) < 1e-6
    return dict(
        native_loss=ref_loss,
        native_max_abs_param_diff=error,
        native_selected=int(mask[0].sum()),
        source_only_selected=int(masks[1].sum()),
        prototype_t_grad=float(v[7]),
        prototype_f_grad=float(v[8]),
        step=restricted.step,
        confidence=restricted.conf_threshold,
        quantile_error=restricted.update_q_err,
        quantile_js=restricted.update_q_ad,
        momentum=restricted.ema_momentum,
    )


def run(w, d, arm):
    folder = w / OUT / d / arm
    folder.mkdir(parents=True, exist_ok=True)
    if arm == "check":
        base.dump(folder / "check.json", check(w, d))
        return
    if (folder / "receipt.json").exists():
        receipt = base.read(folder / "receipt.json")
        assert base.sha(folder / "selected.pt") == receipt["sha256"]
        assert base.sha(folder / "trace.npz") == receipt["trace_sha256"]
        return
    assert base.read(w / OUT / d / "check/check.json")["step"] == 4001
    t = time.monotonic()
    model, opt, parent = setup(w, d, arm)
    masks = np.empty((2000, 2, 32), bool)
    stats = np.empty((2000, len(FIELDS)), np.float32)
    for i, x in enumerate(batches(w, d, arm)):
        masks[i], stats[i] = advance(model, opt, x)
        if (i + 1) % 500 == 0:
            print(d, arm, i + 1, round(time.monotonic() - t, 1), flush=True)
    assert model.step == 6000 and np.isfinite(stats).all()
    if model.source_writes:
        assert not masks[:, 1, :16].any()
    np.savez_compressed(
        folder / "trace.npz",
        native=masks[:, 0],
        written=masks[:, 1],
        stats=stats,
        fields=FIELDS,
    )
    base.save(model, folder)
    base.dump(
        folder / "receipt.json",
        dict(
            domain=d,
            arm=arm,
            updates=2000,
            seed=2025,
            parent=parent["sha256"],
            initial_step=4000,
            step=model.step,
            sha256=base.sha(folder / "selected.pt"),
            numeric_plan=base.sha(w / OUT / d / "plan" / (arm[-1] + ".npz")),
            trace_sha256=base.sha(folder / "trace.npz"),
            seconds=time.monotonic() - t,
            peak_bytes=torch.cuda.max_memory_allocated(),
            device=torch.cuda.get_device_name(),
            source_windows=0 if arm.startswith("ordinary") else 32000,
            labels_read=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("domain", choices=base.WIDTH)
    p.add_argument("arm", choices=(*ARMS, "check"))
    a = p.parse_args()
    run(a.workspace, a.domain, a.arm)
