"""Fixed-source point-score reliability and matched local/window LEFT loss reductions."""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch
import torch.nn.functional as F

from . import left_multisource as base
from .left_memory_update import OUT as EXPOSURE, batches

OUT = Path("results/left_local_reliability_20261126_v1")
ARMS = tuple(m + "_" + k for m in ("window", "local") for k in "ABC")


def plan(w, d, k):
    path = w / EXPOSURE / d / "plan" / (k + ".npz")
    matched = w / EXPOSURE / d / ("restricted_" + k) / "receipt.json"
    assert base.sha(path) == base.read(matched)["numeric_plan"]
    with np.load(path) as p:
        return p["first"]


@torch.no_grad()
def prepare(w, d):
    dest = w / OUT / d / "weights"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "receipt.json").exists():
        for k, v in base.read(dest / "receipt.json")["hashes"].items():
            assert base.sha(dest / k) == v
        return
    start = time.monotonic()
    model = base.initialize(w, d).eval()
    parent = base.restore(w, model, d)
    model.requires_grad_(False)
    _, _, scaler, _ = base.target_inputs(w, d)
    x = scaler.transform(base.target(w, d, "U")).astype(np.float32)
    plans = {k: plan(w, d, k) for k in "ABC"}
    starts = np.unique(np.concatenate(list(plans.values())))
    scores = np.empty((len(starts), 192), np.float32)
    for i in range(0, len(starts), 128):
        b = x[starts[i : i + 128, None] + np.arange(192)]
        scores[i : i + len(b)] = (
            model.infer(torch.from_numpy(b).cuda(), None, None, None)[0]
            .mean(-1)
            .cpu()
            .numpy()
        )
    assert np.isfinite(scores).all() and model.step == parent["selected"]
    assert base.fingerprint(model.state_dict()) == parent["fingerprint"]
    np.savez_compressed(dest / "teacher.npz", starts=starts, scores=scores)
    hashes = {"teacher.npz": base.sha(dest / "teacher.npz")}
    budget = {}
    for k, p in plans.items():
        values = scores[np.searchsorted(starts, p)]
        # Every repeated exposure contributes to this condition's empirical CDF.
        rank = (
            rankdata(values, method="average").reshape(values.shape) - 0.5
        ) / values.size
        raw = 1.5 - rank
        weights = (raw / raw.mean(axis=(1, 2), keepdims=True)).astype(np.float32)
        np.testing.assert_allclose(weights.mean((1, 2)), 1, rtol=0, atol=2e-7)
        budget[k] = dict(
            points=int(values.size),
            unique_starts=int(np.unique(p).size),
            min=float(weights.min()),
            max=float(weights.max()),
            max_budget_error=float(
                np.max(np.abs(weights.mean((1, 2), dtype=np.float64) - 1))
            ),
        )
        filename = k + ".npy"
        np.save(dest / filename, weights)
        hashes[filename] = base.sha(dest / filename)
    base.dump(
        dest / "receipt.json",
        dict(
            parent=parent["sha256"],
            hashes=hashes,
            unique=len(starts),
            seconds=time.monotonic() - start,
            per_condition=budget,
            labels_read=False,
        ),
    )


def reduce_error(model, err, time_grid):
    if time_grid:
        assert err.ndim == 3 and err.shape[:2] == model.point_weights.shape
        weight = model.point_weights[:, :, None]
    else:
        weight = model.window_weights.reshape(-1, *([1] * (err.ndim - 1)))
    return (err * weight).mean()


def smooth(model, y, x, time_grid):
    return reduce_error(model, F.smooth_l1_loss(y, x, reduction="none"), time_grid)


def multiscale(model, y, x, seg_lens, weights):
    errors = F.smooth_l1_loss(y, x, reduction="none").split(seg_lens, dim=1)
    mix = y.new_tensor([1.0] * len(errors) if weights is None else weights)
    mix = mix / (mix.sum() + 1e-12)
    return (
        mix
        * torch.stack(
            [reduce_error(model, e, i == len(errors) - 1) for i, e in enumerate(errors)]
        )
    ).sum()


def set_weights(model, targets, local):
    pts = torch.cat((targets, torch.ones_like(targets)), dim=0)
    means = pts.mean(1)
    model.point_weights = pts if local else means[:, None].expand_as(pts)
    model.window_weights = means


def weights_file(w, d, k):
    folder = w / OUT / d / "weights"
    path = folder / (k + ".npy")
    assert base.sha(path) == base.read(folder / "receipt.json")["hashes"][path.name]
    return np.load(path, mmap_mode="r")


def check(w, d):
    # Use one real replay batch and replay training RNG for each comparison.
    x = next(batches(w, d, "replay_A"))
    results = {}
    saved = None
    for mode in ("native", "window", "local", "window_real", "local_real"):
        model = base.initialize(
            w, d, weighted_reductions=None if mode == "native" else (smooth, multiscale)
        )
        base.restore(w, model, d)
        model.train()
        if mode != "native":
            a = (
                torch.tensor(weights_file(w, d, "A")[0], device=x.device)
                if mode.endswith("real")
                else x.new_ones((16, 192))
            )
            set_weights(model, a, mode.startswith("local"))
            assert model.point_weights[16:].eq(1).all()
            assert abs(model.point_weights[:16].mean().item() - 1) < 2e-7
        loss = model(x, None, None, None)[0]
        loss.backward()
        grads = torch.cat(
            [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        )
        assert model.step == 4001
        if saved is None:
            saved = (loss.item(), grads, torch.cuda.get_rng_state())
        assert torch.equal(saved[2], torch.cuda.get_rng_state())
        err = float((grads - saved[1]).abs().max())
        rel = float((grads - saved[1]).norm() / saved[1].norm())
        if mode in ("window", "local"):
            assert abs(loss.item() - saved[0]) < 2e-6 and rel < 2e-5, (mode, err, rel)
        if mode == "window_real":
            real = grads
        if mode == "local_real":
            results["real_between_arm_relative_gradient_difference"] = float(
                (real - grads).norm() / real.norm()
            )
            assert results["real_between_arm_relative_gradient_difference"] > 0
        results[mode] = dict(
            loss=loss.item(), max_abs_grad_diff=err, relative_l2_grad_diff=rel
        )
    return results


PERMUTATION_OUT = Path("results/left_window_permutation_20261126_v1")


def permutation_plan():
    # Common permutation across domains/conditions, isolated from all learning RNG.
    rng = np.random.default_rng(np.random.SeedSequence([2025, 20261126, 16]))
    return np.stack([rng.permutation(16) for _ in range(2000)])


def run(w, d, arm):
    shuffled = arm.startswith("permuted_")
    folder = w / (PERMUTATION_OUT if shuffled else OUT) / d / arm
    folder.mkdir(parents=True, exist_ok=True)
    if arm == "prepare":
        prepare(w, d)
        base.dump(folder / "check.json", check(w, d))
        return
    if (folder / "receipt.json").exists():
        assert (
            base.sha(folder / "selected.pt")
            == base.read(folder / "receipt.json")["sha256"]
        )
        return
    assert (
        base.read(w / OUT / d / "prepare/check.json")[
            "real_between_arm_relative_gradient_difference"
        ]
        > 0
    )
    start = time.monotonic()
    mode, k = arm.split("_")
    model = base.initialize(w, d, weighted_reductions=(smooth, multiscale)).train()
    parent = base.restore(w, model, d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    weights = weights_file(w, d, k)
    perm = permutation_plan() if shuffled else None
    if shuffled:
        assert np.all(np.sort(perm, axis=1) == np.arange(16))
        np.save(folder / "permutation.npy", perm)
    curve = []
    for i, x in enumerate(batches(w, d, "replay_" + k)):
        target_weights = torch.tensor(weights[i], device=x.device)
        if shuffled:
            original = target_weights.mean(1)
            target_weights = target_weights[torch.as_tensor(perm[i], device=x.device)]
            assert torch.equal(
                original.sort().values, target_weights.mean(1).sort().values
            )
        set_weights(model, target_weights, mode == "local")
        opt.zero_grad(set_to_none=True)
        loss = model(x, None, None, None)[0]
        loss.backward()
        opt.step()
        curve.append(loss.detach().item())
        if (i + 1) % 500 == 0:
            print(d, arm, i + 1, round(time.monotonic() - start, 1), flush=True)
    assert model.step == 6000 and len(curve) == 2000 and np.isfinite(curve).all()
    np.save(folder / "fit_loss.npy", curve)
    base.save(model, folder)
    base.dump(
        folder / "receipt.json",
        dict(
            domain=d,
            arm=arm,
            seed=2025,
            step=6000,
            updates=2000,
            parent=parent["sha256"],
            sha256=base.sha(folder / "selected.pt"),
            target_windows=32000,
            source_windows=32000,
            plan_sha256=base.sha(w / EXPOSURE / d / "plan" / (k + ".npz")),
            weights_sha256=base.sha(w / OUT / d / "weights" / (k + ".npy")),
            **(
                {
                    "permutation_sha256": base.sha(folder / "permutation.npy"),
                    "unchanged_slots": int((perm == np.arange(16)).sum()),
                }
                if shuffled
                else {}
            ),
            seconds=time.monotonic() - start,
            peak_bytes=torch.cuda.max_memory_allocated(),
            device=torch.cuda.get_device_name(),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("domain", choices=base.WIDTH)
    parser.add_argument(
        "arm", choices=(*ARMS, *("permuted_" + k for k in "ABC"), "prepare")
    )
    args = parser.parse_args()
    run(args.workspace, args.domain, args.arm)
