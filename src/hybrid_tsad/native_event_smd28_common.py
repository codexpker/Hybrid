"""Intersect sealed SMD endpoint and native LEFT whole-window outputs by machine."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_tsad.left_native_swat_metrics import continuous_metrics, load_author_metric, sha


METRICS = ["AP", "ROC_AUC", "VUS_PR", "VUS_ROC"]
VROOT = Path("results/native_event_smd28_20261123_v1")
LROOT = Path("results/left_smd28_author_20261123_v1")
LABELS = Path("preparations/timeradar_smd_full_20261116_v1/evaluation_manifest_full28.json")
AUTHOR = Path("preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py")


def intersect_one(job):
    workspace, record = job
    machine, length = record["id"], record["n"]
    vp = workspace / VROOT / f"{machine}.npz"
    lp = workspace / LROOT / machine / "scores.npz"
    target = workspace / record["path"]
    native_manifest = json.loads(vp.with_suffix(".json").read_text())
    left_manifest = json.loads(lp.with_name("SCORES.json").read_text())
    if sha(vp) != native_manifest["sha256"] or sha(lp) != left_manifest["bank_sha256"]:
        raise ValueError(f"Score bank identity mismatch for {machine}")
    with np.load(vp, allow_pickle=False) as v, np.load(lp, allow_pickle=False) as left:
        vi, li = v["query_index"], left["index"]
        np.testing.assert_array_equal(vi, np.arange(99, length))
        np.testing.assert_array_equal(li, np.arange(length // 192 * 192))
        np.testing.assert_array_equal(left["available_index"], (li // 192 + 1) * 192 - 1)
        axis, vpos, lpos = np.intersect1d(vi, li, assume_unique=True, return_indices=True)
        np.testing.assert_array_equal(axis, np.arange(99, length // 192 * 192))
        names = v["keys"].tolist()
        expected = ["frozen_native_V"] + [f"{s}_{a}" for s in [2025, 2026, 2027]
                                         for a in ["copy", "point_weighted"]]
        if set(names) != set(expected) or len(names) != len(expected):
            raise ValueError(f"Incomplete checkpoint panel for {machine}: {names}")
        scores = {k: v["query"][vpos, j] for j, k in enumerate(names)}
        scores["LEFT"] = left["score"].reshape(-1)[lpos]
    if sha(target) != record["sha256"]:
        raise ValueError(f"Label identity mismatch for {machine}")
    with np.load(target, allow_pickle=False) as truth:
        np.testing.assert_array_equal(truth["index"], np.arange(length))
        y = truth["label"].reshape(-1)[axis]
    metric = load_author_metric(workspace / AUTHOR)
    rows = []
    for name, score in scores.items():
        assert score.shape == y.shape and np.isfinite(score).all()
        rank, _ = continuous_metrics(y, score, metric)
        rows.append(dict(entity=machine, model=name, first=int(axis[0]), stop=int(axis[-1]+1),
                         points=len(axis), positives=int(y.sum()), **{k: rank[k] for k in METRICS}))
    return rows


def run(workspace, out):
    records = json.loads((workspace / LABELS).read_text())["records"]
    assert len(records) == len({r["id"] for r in records}) == 28
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for result in pool.map(intersect_one, [(workspace, r) for r in records]):
            rows.extend(result)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "common_entity.csv", index=False)
    macro = frame.groupby("model", sort=False)[METRICS].mean()
    macro.to_csv(out / "common_macro_by_checkpoint.csv")
    groups = []
    for family in ["frozen_native_V", "copy", "point_weighted", "LEFT"]:
        sample = macro.loc[[family]] if family in macro.index else macro.loc[
            [f"{s}_{family}" for s in (2025, 2026, 2027)]]
        groups.append(dict(method=family, checkpoints=len(sample), entities=28,
                           points=sum(f["points"] for f in rows if f["model"]=="LEFT"),
                           **{k: sample[k].mean() for k in METRICS},
                           **{k+"_sd": sample[k].std(ddof=1) for k in METRICS}))
    pd.DataFrame(groups).to_csv(out / "common_methods.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workspace", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    run(args.workspace, args.output)
