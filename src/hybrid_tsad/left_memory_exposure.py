"""Offline intervention constructor. The learner receives only numeric window starts."""

import argparse
from pathlib import Path

import numpy as np

from .left_exposure_bundle import legal_windows
from .left_multisource import dump, sha, WINDOW
from .trajectory_evaluation import labels
from .trajectory_experiment import ROLES

OUT = Path("results/left_memory_exposure_20261126_v1")


def construct(w, d):
    dest = w / OUT / d / "plan"
    dest.mkdir(parents=True, exist_ok=False)
    lo, hi = ROLES[d]["U"]
    y = labels(w, d)[lo:hi]
    abnormal, normal, _ = legal_windows(y, WINDOW)
    assert len(abnormal) and len(normal)
    rng, rp, _ = [
        np.random.default_rng(s) for s in np.random.SeedSequence(2025).spawn(3)
    ]
    shape = (2000, 16)
    first = rng.integers(len(y) - WINDOW + 1, size=shape)
    second = rp.integers(len(y) - WINDOW + 1, size=shape)
    slot, b, c = [
        np.random.default_rng(s)
        for s in np.random.SeedSequence([2025, 20261118, 10]).spawn(3)
    ]
    slots = np.sort(slot.choice(first.size, 3200, replace=False))
    events = np.flatnonzero(np.diff(np.r_[0, y, 0]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[0, y, 0]) == -1)
    assert len(events) == len(ends)
    audit = dict(
        U_bounds=[lo, hi],
        eligible_B=len(abnormal),
        eligible_C=len(normal),
        eligible_B_events=int(np.sum(ends - events >= WINDOW)),
        U_events=len(events),
        U_attack_points=int(y.sum()),
    )
    for condition, donors in (
        ("A", None),
        ("B", b.choice(abnormal, len(slots))),
        ("C", c.choice(normal, len(slots))),
    ):
        starts = first.copy()
        if donors is not None:
            starts.flat[slots] = donors
            unique, counts = np.unique(donors, return_counts=True)
            distribution = {}
            if condition == "B":
                ids = np.searchsorted(events, donors, side="right") - 1
                assert (donors + WINDOW <= ends[ids]).all()
                distribution = {
                    str((lo + int(events[i]), lo + int(ends[i]))): int((ids == i).sum())
                    for i in np.unique(ids)
                }
            audit[condition] = dict(
                distinct=int(len(unique)),
                repeated=int(len(slots) - len(unique)),
                max_exposures=int(max(counts)),
                donor_events=distribution,
            )
        path = dest / (condition + ".npz")
        np.savez_compressed(path, first=starts, second=second)
        audit[condition + "_sha256"] = sha(path)
    np.save(dest / "audit_slots.npy", slots)
    dump(dest / "audit.json", audit)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("domain", choices=ROLES)
    a = p.parse_args()
    construct(a.workspace, a.domain)
