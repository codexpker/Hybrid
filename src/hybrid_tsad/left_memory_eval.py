"""Sealed bank evaluation and post-training attribution of explicit EMA writes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import left_multisource as base
from . import left_multisource_eval as ev
from .left_memory_update import ARMS, OUT


def evaluate(args):
    ev.OUT = OUT
    return ev.evaluate(args)


def banks(w, d):
    ev.OUT = OUT
    for arm in ARMS:
        for role in ("CAL", "QUERY"):
            ev.bank(w, d, arm, role)


def composition(domain, arm, stage, group, times, counts, boundaries):
    hits = sum(
        np.any((times < end) & (times + 192 > begin)) for begin, end in boundaries
    )
    return dict(
        domain=domain,
        arm=arm,
        stage=stage,
        group=group,
        presentations=len(counts),
        unique_starts=len(np.unique(times)),
        fully_anomalous=int((counts == 192).sum()),
        partially_anomalous=int(((counts > 0) & (counts < 192)).sum()),
        fully_normal=int((counts == 0).sum()),
        overlapping_U_events=hits,
        attack_point_presentations=int(counts.sum()),
    )


def summarize(w, d):
    ev.OUT = OUT
    dest = w / OUT / d
    for domain in base.WIDTH:
        for arm in ARMS:
            for role in ("CAL", "QUERY"):
                path = w / OUT / domain / "banks" / f"{arm}_{role}.npz"
                assert base.sha(path) == base.read(path.with_suffix(".json"))["sha256"]
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(evaluate, [(w, d, a, "QUERY") for a in ARMS]))
    for name, rs in (
        ("mixed", [r for r, _, _ in rows]),
        ("operations", [r for _, o, _ in rows for r in o]),
        ("events", [r for _, _, e in rows for r in e]),
    ):
        ev.save_rows(dest / (name + ".csv"), rs)
    lo, hi = ev.ROLES[d]["U"]
    y = ev.labels(w, d)[lo:hi]
    cs = np.r_[0, np.cumsum(y)]
    edges = np.diff(np.r_[0, y, 0])
    boundaries = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    slots = np.load(dest / "plan" / "audit_slots.npy")
    mask_donor = np.zeros((2000, 32), bool)
    mask_donor[np.unravel_index(slots, (2000, 16))] = True
    init = torch.load(
        w / base.OUT / d / "source/selected.pt", weights_only=True, map_location="cpu"
    )["state"]
    mechanisms = []
    compositions = []
    for arm in ARMS:
        method, condition = arm.split("_")
        folder = dest / arm
        data = np.load(folder / "trace.npz")
        plan = np.load(dest / "plan" / (condition + ".npz"))
        starts = np.c_[
            plan["first"],
            plan["second"] if method == "ordinary" else np.full((2000, 16), -1),
        ]
        target = starts >= 0
        donor = mask_donor if condition != "A" else np.zeros_like(mask_donor)
        counts = np.where(target, cs[starts + 192] - cs[starts], 0)
        if condition != "A":
            assert np.all(counts[donor] == (192 if condition == "B" else 0))
        stat = dict(zip(data["fields"].tolist(), data["stats"].T))
        native, written = data["native"], data["written"]
        state = torch.load(
            folder / "selected.pt", map_location="cpu", weights_only=True
        )["state"]
        m = dict(
            domain=d,
            arm=arm,
            method=method,
            condition=condition,
            target_exposure=int(target.sum()),
            source_exposure=int((~target).sum()),
            label_point_exposure=int(counts.sum()),
            label_point_fraction=float(counts.sum() / (64000 * 192)),
            native_target=int((native & target).sum()),
            native_source=int((native & ~target).sum()),
            written_target=int((written & target).sum()),
            written_source=int((written & ~target).sum()),
            no_write_steps=int((~written.any(1)).sum()),
            ema_write_steps=int(written.any(1).sum()),
            bank_write_calls=2 * int(written.any(1).sum()),
        )
        for view in ("t", "f"):
            key = "proto_" + view + ".prototypes"
            m[view + "_net_drift"] = (state[key] - init[key]).norm().item()
            for k in ("ema", "adam", "changed_rows"):
                m[view + "_" + k + "_total"] = float(
                    stat[k + "_" + view].sum(dtype=np.float64)
                )
            assert not np.any(stat["changed_rows_" + view][~written.any(1)])
        mechanisms.append(m)
        for kind, selected in (
            ("presented", np.ones_like(written)),
            ("native_filter", native),
            ("explicit_write", written),
        ):
            for group, included in (("all_target", target), ("replacement", donor)):
                take = selected & included
                times = starts[take]
                n = counts[take]
                compositions.append(
                    composition(d, arm, kind, group, times, n, boundaries)
                )
    # The plans determine A exposure. Its historical checkpoints do not encode
    # window-level write decisions, so only the presented stage is recoverable.
    a = np.load(dest / "plan/A.npz")
    for arm, starts in (
        ("ordinary_A", np.c_[a["first"], a["second"]]),
        ("replay_A", a["first"]),
    ):
        starts = starts.ravel()
        n = cs[starts + 192] - cs[starts]
        compositions.append(
            composition(d, arm, "presented", "all_target", starts, n, boundaries)
        )
    ev.save_rows(dest / "mechanism.csv", mechanisms)
    ev.save_rows(dest / "composition.csv", compositions)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("domain", choices=base.WIDTH)
    p.add_argument("mode", choices=("bank", "metrics"))
    a = p.parse_args()
    (banks if a.mode == "bank" else summarize)(a.workspace, a.domain)
