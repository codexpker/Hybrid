"""Finish the fixed queued experiment; never schedule another training variant."""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from hybrid_tsad.left_timesfm_evaluation import (
    OUT, ARMS, previous, measure, controlled_metrics
)
from hybrid_tsad.left_timesfm_report import summarize


def score_remaining(w, previous_pid):
    root = w / OUT
    while Path(f"/proc/{previous_pid}").exists():
        time.sleep(20)
    for d in ("PSM", "SWAT", "SMD"):
        for entity in previous.members(w, d):
            for a in ARMS:
                assert (root / "public" / d / entity / f"{a}_TEST.json").exists(), (
                    "Previous Z queue exited without complete scores", d, entity, a
                )
    pending = [(d, e) for d in ("PSM", "SWAT", "SMD") for e in previous.members(w, d)]
    while pending:
        ready = [(d, e) for d, e in pending if all(
            (root / d / "U" / e / a / "receipt.json").exists() for a in ("A", "C")
        )]
        if not ready:
            time.sleep(20)
            continue
        for d, e in ready:
            # Subprocess lifetime releases CUDA and host buffers after each entity.
            for role in ("QUERY", "CAL"):
                command = [sys.executable, "-u", "-m", "hybrid_tsad.left_timesfm_evaluation",
                           str(w), "score", d, "--entities", e, "--arms", "A_U", "C_U",
                           "--role", role]
                begin = time.time()
                subprocess.run(command, check=True)
                print(json.dumps(dict(event="U_SCORED", domain=d, entity=e, role=role,
                                      process_wall_seconds=time.time()-begin)), flush=True)
            pending.remove((d, e))
    (root / "U_READOUT_COMPLETE.json").write_text(json.dumps(dict(entities=30)))


def metrics_remaining(w):
    root = w / OUT
    pending = [
        (w, d, e, a, role)
        for d in ("PSM", "SWAT", "SMD")
        for e in previous.members(w, d)
        for role, arms in (("TEST", ARMS), ("QUERY", ("A", "C", "A_U", "C_U")))
        for a in arms
    ]
    futures = {}
    with ProcessPoolExecutor(4) as pool:
        while pending or futures:
            for task in list(pending):
                _, d, e, a, role = task
                dest = root / "metrics" / d / e / f"{a}_{role}.json"
                if dest.exists():
                    pending.remove(task)
                    continue
                banks = (role, "CAL") if role == "QUERY" else (role,)
                if len(futures) < 8 and all(
                    (root / "public" / d / e / f"{a}_{r}.json").exists() for r in banks
                ):
                    futures[pool.submit(measure, task)] = task
                    pending.remove(task)
            for future in list(futures):
                if future.done():
                    path = future.result()
                    print("METRICS_SEALED", path, flush=True)
                    del futures[future]
            if pending or futures:
                time.sleep(10)
    controls = [root / "controlled" / d / f"{a}_{group}.json"
                for d in ("PSM", "SWAT") for a in ARMS
                for group in ("coupled_AR", "normal_SELECT")]
    while not all(p.exists() for p in controls):
        time.sleep(20)
    controlled_metrics(w, ("PSM", "SWAT"))
    output = summarize(w)
    print("FIXED_MATRIX_COMPLETE", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("action", choices=("score", "metrics"))
    p.add_argument("--previous-pid", type=int, default=3398448)
    a = p.parse_args()
    if a.action == "score":
        score_remaining(a.workspace, a.previous_pid)
    else:
        metrics_remaining(a.workspace)
