"""Sealed point-score banks and retrospective evaluation for shuffled window weights."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import left_multisource_eval as ev
from .left_reliability import PERMUTATION_OUT

DOMAINS = ("PSM", "SWAT")
ARMS = tuple("permuted_" + condition for condition in "ABC")


def evaluate(args):
    ev.OUT = PERMUTATION_OUT
    return ev.evaluate(args)


def run(workspace, mode):
    ev.OUT = PERMUTATION_OUT
    for domain in DOMAINS:
        for arm in ARMS:
            for role in ("CAL", "QUERY"):
                if mode == "bank":
                    ev.bank(workspace, domain, arm, role)
                else:
                    path = (
                        workspace
                        / PERMUTATION_OUT
                        / domain
                        / "banks"
                        / f"{arm}_{role}.npz"
                    )
                    assert ev.sha(path) == ev.read(path.with_suffix(".json"))["sha256"]
    if mode == "metrics":
        for domain in DOMAINS:
            with ProcessPoolExecutor(max_workers=3) as pool:
                results = list(
                    pool.map(
                        evaluate, [(workspace, domain, arm, "QUERY") for arm in ARMS]
                    )
                )
            folder = workspace / PERMUTATION_OUT / domain
            ev.save_rows(folder / "mixed.csv", [rank for rank, _, _ in results])
            ev.save_rows(
                folder / "operations.csv", [row for _, op, _ in results for row in op]
            )
            ev.save_rows(
                folder / "events.csv",
                [row for _, _, events in results for row in events],
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("mode", choices=("bank", "metrics"))
    args = p.parse_args()
    run(args.workspace, args.mode)
