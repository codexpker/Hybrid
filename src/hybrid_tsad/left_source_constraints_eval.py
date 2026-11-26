"""Evaluate sealed constrained LEFT models on the established U suffix and CAL."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import left_multisource_eval as ev
from .left_source_constraints import ARMS, OUT


def metrics(args):
    ev.OUT = OUT
    return ev.evaluate(args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("domain", choices=ev.WIDTH)
    a = p.parse_args()
    w, d = a.workspace, a.domain
    ev.OUT = OUT
    for arm in ARMS:
        for role in ("CAL", "QUERY"):
            ev.bank(w, d, arm, role)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(metrics, [(w, d, arm, "QUERY") for arm in ARMS]))
    dest = w / OUT / d
    ev.save_rows(dest / "mixed.csv", [r[0] for r in rows])
    ev.save_rows(dest / "operations.csv", [op for _, ops, _ in rows for op in ops])
    ev.save_rows(dest / "events.csv", [e for _, _, events in rows for e in events])


if __name__ == "__main__":
    main()
