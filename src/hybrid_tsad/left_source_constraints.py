"""Match native LEFT replay; constrain the post-EMA Adam parameter displacement."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from . import left_multisource as base

OUT = Path("results/left_source_constraints_20261126_v1")
ARMS = ("source_only", "mean", "separate")
VECTOR_STEPS = (1, 1000, 2000)


def unit(v):
    norm = torch.linalg.vector_norm(v)
    return v / norm if norm.item() else None


def projection(d, normals):
    """Euclidean projection onto at most two halfspaces through the origin, float64."""
    normals = [u for u in normals if u is not None]
    q = [torch.dot(d, u) for u in normals]
    if all(x.item() <= 0 for x in q):
        return d, False
    if len(q) == 1:
        return d - q[0] * normals[0], True
    u, v = normals
    # A feasible projection onto a single face is already the joint optimum.
    for i, (normal, other) in enumerate(((u, v), (v, u))):
        if q[i].item() > 0:
            z = d - q[i] * normal
            if torch.dot(z, other).item() <= 0:
                return z, True
    # Both faces active. Orthogonalize directly even for nearly parallel normals.
    orth = v - torch.dot(u, v) * u
    orth = orth - torch.dot(orth, u) * u
    z = d - q[0] * u
    if torch.linalg.vector_norm(orth).item() > 8 * torch.finfo(d.dtype).eps:
        direction = unit(orth)
        z = z - torch.dot(z, direction) * direction
    return z, True


@torch.no_grad()
def install(params, old, move):
    flat = (old.double() + move).to(old.dtype)
    for p, values in zip(params, flat.split([p.numel() for p in params])):
        p.copy_(values.view_as(p))


def realized(params, old):
    return (parameters_to_vector(params).detach() - old).double()


def reference(model, x, params=None):
    # Replay the exact eval RNG state both before and after the Adam displacement.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        with torch.set_grad_enabled(params is not None):
            loss = model(x, None, None, None)[0]
            if params is None:
                return loss.item()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            return loss.item(), torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for p, g in zip(params, grads)
                ]
            ).double()


def step(model, opt, x, arm, vector_path=None, check=False):
    model.train()
    count = model.step
    opt.zero_grad(set_to_none=True)
    loss = model(x, None, None, None)[0]
    loss.backward()
    assert model.step == count + 1
    params = [p for p in model.parameters() if p.requires_grad]
    model.eval()
    old = parameters_to_vector(params).detach()
    if check:
        state = base.fingerprint(model.state_dict())
        rng = torch.cuda.get_rng_state()
        main_grads = [p.grad.clone() if p.grad is not None else None for p in params]
    sources = (x[16::2], x[17::2])
    refs = [reference(model, s, params) for s in sources]
    if check:
        assert base.fingerprint(model.state_dict()) == state
        assert torch.equal(rng, torch.cuda.get_rng_state())
        assert all(
            (p.grad is None if g is None else torch.equal(p.grad, g))
            for p, g in zip(params, main_grads)
        )
    opt.step()
    raw = realized(params, old)
    gradients = [g for _, g in refs]
    normals = (
        [unit((gradients[0] + gradients[1]) * 0.5)]
        if arm == "mean"
        else [unit(g) for g in gradients]
    )
    normals = [u for u in normals if u is not None]
    intended, trigger = projection(raw, normals)
    repairs = 0
    if trigger:
        install(params, old, intended)
    actual = realized(params, old)
    # If parameter storage rounded a face outward, make the smallest tested inward
    # nudge along the common interior direction. Never modify a feasible proposal.
    if trigger and any(torch.dot(actual, u).item() > 0 for u in normals):
        interior = sum(normals)
        margin = min(torch.dot(interior, u).item() for u in normals)
        if margin > 1e-12:
            epsilon = torch.linalg.vector_norm(raw).item() * 1e-10
            for repairs in range(1, 11):
                violation = max(torch.dot(actual, u).item() for u in normals)
                epsilon = max(epsilon * 2, 2 * violation / margin)
                install(params, old, intended - epsilon * interior)
                actual = realized(params, old)
                if all(torch.dot(actual, u).item() <= 0 for u in normals):
                    break
    norm0 = raw.norm().item()
    max_res = max((torch.dot(actual, u).item() for u in normals), default=0.0)
    if max_res > 1e-8 * norm0:
        raise ArithmeticError(f"unresolved storage residual {max_res} / {norm0}")
    row = dict(
        loss=loss.item(),
        trigger=int(trigger),
        proposed=norm0,
        actual=actual.norm().item(),
        correction=(actual - raw).norm().item(),
        storage_error=(actual - intended).norm().item(),
        storage_nudges=repairs,
        enforced_residual=max_res,
    )
    for name, (before, g), s in zip(("tep", "smd"), refs, sources):
        after = reference(model, s)
        row.update(
            {
                name + "_norm": g.norm().item(),
                name + "_raw": torch.dot(g, raw).item(),
                name + "_actual": torch.dot(g, actual).item(),
                name + "_before": before,
                name + "_after": after,
                name + "_change": after - before,
            }
        )
    if not np.isfinite(list(row.values())).all():
        raise FloatingPointError(row)
    if vector_path:
        np.savez_compressed(
            vector_path,
            proposed=raw.float().cpu().numpy(),
            actual=actual.float().cpu().numpy(),
        )
    assert model.step == count + 1
    model.train()
    return row


def batches(w, domain, arm):
    fit = base.source_records(w)["fit"]
    r1, _, rs = [
        np.random.default_rng(s) for s in np.random.SeedSequence(2025).spawn(3)
    ]
    extra = np.random.default_rng(np.random.SeedSequence([2025, 1126, 4]))
    width = base.WIDTH[domain]
    if arm != "source_only":
        _, _, scaler, _ = base.target_inputs(w, domain)
        data = scaler.transform(base.target(w, domain, "U")).astype(np.float32)
    while True:
        if arm == "source_only":
            first = base.source_batch(fit, width, 16, extra)
        else:
            starts = r1.integers(len(data) - base.WINDOW + 1, size=16)
            first = data[starts[:, None] + np.arange(base.WINDOW)]
        yield np.concatenate([first, base.source_batch(fit, width, 16, rs)])


def checks(w, domain):
    for vector, ns, expected in [
        ([1, 2], [[0, 0]], [1, 2]),
        ([0, 0], [[1, 0]], [0, 0]),
        ([-3, -1], [[1, 0], [0, 1]], [-3, -1]),
        ([3, -1], [[1, 0], [0, 1]], [0, -1]),
        ([2, 3], [[1, 0], [0, 1]], [0, 0]),
        ([1, 2], [[1, 0], [2, 0]], [0, 2]),
        ([-1, 2], [[1, 0], [-1, 0]], [0, 2]),
        ([1, 2], [[1, 0], [-1, 1e-10]], [0, 0]),
    ]:
        d = torch.tensor(vector, dtype=torch.double)
        z, _ = projection(d, [unit(torch.tensor(n, dtype=torch.double)) for n in ns])
        torch.testing.assert_close(
            z, torch.tensor(expected, dtype=torch.double), atol=1e-12, rtol=0
        )
    batch = next(batches(w, domain, "separate"))

    def start():
        m = base.initialize(w, domain)
        base.restore(w, m, domain)
        return m, torch.optim.Adam(m.parameters(), lr=1e-4)

    m, o = start()
    old_step = m.step
    plain = base.update(m, o, batch)
    rng = torch.cuda.get_rng_state()
    plain_state = base.fingerprint(m.state_dict())
    plain_tensors = {k: v.clone() for k, v in m.state_dict().items()}
    m, o = start()
    row = step(m, o, torch.from_numpy(batch).cuda(), "separate", check=True)
    assert abs(plain - row["loss"]) < 1e-6 and m.step == old_step + 1
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert all(int(st["step"].item()) == 1 for st in o.state.values())
    if not row["trigger"]:
        assert row["correction"] == 0.0
    native_difference = max(
        (m.state_dict()[k] - v).abs().max().item() for k, v in plain_tensors.items()
    )
    assert native_difference < 2e-5, native_difference
    return dict(
        analytic_cases=8,
        plain_loss=plain,
        native_after=plain_state,
        native_max_abs_difference=native_difference,
        reference_step=row,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("domain", choices=base.WIDTH)
    parser.add_argument("arm", choices=(*ARMS, "check"))
    args = parser.parse_args()
    w, d, a = args.workspace, args.domain, args.arm
    dest = w / OUT / d / a
    dest.mkdir(parents=True, exist_ok=True)
    if a == "check":
        base.dump(dest / "check.json", checks(w, d))
        return
    if (dest / "receipt.json").exists():
        raise FileExistsError(dest)
    gate = base.read(w / OUT / d / "check/check.json")
    assert gate["analytic_cases"] == 8
    t = time.monotonic()
    model = base.initialize(w, d)
    parent = base.restore(w, model, d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    stream = batches(w, d, a)
    with (dest / "steps.csv").open("w", newline="") as f:
        for i in range(1, 2001):
            batch = next(stream)
            if a == "source_only":
                row = dict(loss=base.update(model, opt, batch))
            else:
                row = step(
                    model,
                    opt,
                    torch.from_numpy(batch).cuda(),
                    a,
                    dest / f"delta_{i}.npz" if i in VECTOR_STEPS else None,
                )
            row["step"] = i
            if i == 1:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if i % 100 == 0:
                f.flush()
                print(d, a, i, round(time.monotonic() - t, 1), flush=True)
    assert model.step == parent["selected"] + 2000
    base.save(model, dest)
    base.dump(
        dest / "receipt.json",
        dict(
            domain=d,
            arm=a,
            updates=2000,
            parent=parent["sha256"],
            seed=2025,
            step=model.step,
            sha256=base.sha(dest / "selected.pt"),
            seconds=time.monotonic() - t,
            peak_bytes=torch.cuda.max_memory_allocated(),
            device=torch.cuda.get_device_name(),
            target_windows=0 if a == "source_only" else 32000,
            windows_per_source=32000 if a == "source_only" else 16000,
            reference_windows_per_source=0 if a == "source_only" else 16000,
            step_log_sha256=base.sha(dest / "steps.csv"),
            vectors={p.name: base.sha(p) for p in dest.glob("delta_*.npz")},
        ),
    )


if __name__ == "__main__":
    main()
