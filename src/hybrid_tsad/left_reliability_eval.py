"""Native score banks, common-axis metrics and evaluation-only reliability attribution."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import left_multisource as base
from . import left_multisource_eval as ev
from .left_reliability import ARMS, OUT, plan, weights_file


def bank(w, d):
    ev.OUT = OUT
    for arm in ARMS:
        for role in ("CAL", "QUERY"):
            ev.bank(w, d, arm, role)


def evaluate(args):
    ev.OUT = OUT
    return ev.evaluate(args)


def mechanism(w, d, dest):
    lo, hi = ev.ROLES[d]["U"]
    y = ev.labels(w, d)[lo:hi].astype(bool)
    rows = []
    for k in "ABC":
        starts = plan(w, d, k).ravel()
        grid = weights_file(w, d, k).reshape(-1, 192)
        truth = y[starts[:, None] + np.arange(192)]
        count = truth.sum(1)
        mean = grid.mean(1, dtype=np.float64)
        flat = np.broadcast_to(mean[:, None], grid.shape)
        partial = (count > 0) & (count < 192)
        for category, mask in (
            ("all", np.ones_like(partial)),
            ("fully_normal", count == 0),
            ("partial", partial),
            ("fully_abnormal", count == 192),
        ):
            for kind, t in (
                ("all", np.ones_like(truth)),
                ("normal", ~truth),
                ("abnormal", truth),
            ):
                keep = mask[:, None] & t
                pts = int(keep.sum())
                row = dict(
                    domain=d,
                    condition=k,
                    category=category,
                    position=kind,
                    window_exposures=int(mask.sum()),
                    positions=pts,
                )
                if pts:
                    local = grid[keep].mean(dtype=np.float64)
                    whole = flat[keep].mean()
                    row.update(
                        local_weight=local,
                        window_weight=whole,
                        local_minus_window=local - whole,
                    )
                    relevant = keep.any(1)
                    within = (
                        np.sum(np.where(keep, grid, 0), axis=1, dtype=np.float64)[
                            relevant
                        ]
                        / keep.sum(1)[relevant]
                    )
                    row["window_fraction_local_above_uniform"] = float(
                        (within - mean[relevant] > 1e-7).mean()
                    )
                rows.append(row)
        if k == "A":
            mid = np.median(count[partial])
            loc = np.flatnonzero(partial)[np.argmin(abs(count[partial] - mid))]
            with np.load(dest / "weights/teacher.npz") as teacher:
                idx = np.searchsorted(teacher["starts"], starts[loc])
                np.savez_compressed(
                    dest / "example.npz",
                    index=lo + starts[loc] + np.arange(192),
                    step=int(loc // 16 + 1),
                    slot=int(loc % 16),
                    window_weight=mean[loc],
                    point_weight=grid[loc],
                    teacher_score=teacher["scores"][idx],
                    label=truth[loc],
                )
    ev.save_rows(dest / "mechanism.csv", rows)


def summarize(w, d):
    ev.OUT = OUT
    for domain in base.WIDTH:
        for arm in ARMS:
            for role in ("CAL", "QUERY"):
                path = w / OUT / domain / "banks" / f"{arm}_{role}.npz"
                assert base.sha(path) == base.read(path.with_suffix(".json"))["sha256"]
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(evaluate, [(w, d, a, "QUERY") for a in ARMS]))
    dest = w / OUT / d
    for name, rows in (
        ("mixed", [r for r, _, _ in results]),
        ("operations", [r for _, o, _ in results for r in o]),
        ("events", [r for _, _, e in results for r in e]),
    ):
        ev.save_rows(dest / (name + ".csv"), rows)
    mechanism(w, d, dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("domain", choices=base.WIDTH)
    p.add_argument("mode", choices=("bank", "metrics"))
    a = p.parse_args()
    (bank if a.mode == "bank" else summarize)(a.workspace, a.domain)
