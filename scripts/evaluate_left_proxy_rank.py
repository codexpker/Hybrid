"""Evaluate sealed proxy-ranking banks on the original LEFT mixed suffixes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import time

import numpy as np

from hybrid_tsad import left_multisource as base
from hybrid_tsad.left_multisource_eval import AUTHOR, MIXED, read_score
from hybrid_tsad.left_proxy_rank import ARMS, OUT, STEPS
from hybrid_tsad.trajectory_evaluation import (
    continuous_metrics, independent_check, labels, load_author_metric, save_rows,
)

METRICS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")
STAGES = ("Z", "U")
M0_NAMES = {"Z": "source", "U": "replay"}


def sealed(w):
    for d in MIXED:
        root = w / OUT / d
        normal = {}
        for arm in ARMS:
            parent = None
            for stage in STAGES:
                key = f"{arm}_{stage}"
                dest = root / key
                receipt = base.read(dest / "receipt.json")
                bank_path = root / "banks" / f"{key}_QUERY.npz"
                bank = base.read(bank_path.with_suffix(".json"))
                if stage == "Z":
                    parent = base.read(w / base.OUT / d / "source" / "receipt.json")["sha256"]
                    normal[arm] = receipt["normal_plan_sha256"]
                assert receipt["parent"] == parent
                assert receipt["updates"] == STEPS and not receipt["labels_read"]
                assert receipt["model_step"] == receipt["initial_step"] + STEPS
                assert receipt["sha256"] == bank["checkpoint_sha256"] == base.sha(dest/"selected.pt")
                assert bank["step"] == receipt["model_step"]
                assert base.sha(bank_path) == bank["sha256"]
                parent = receipt["sha256"]
        assert normal["C"] == normal["M1"]


def evaluate(job):
    w, d, arm, stage, baseline = job
    start = time.monotonic()
    first, stop = MIXED[d]
    index, old, available = read_score(w, d, M0_NAMES[stage], "QUERY")
    if arm == "M0":
        with (baseline / f"{d}.csv").open(newline="") as file:
            row = next(r for r in csv.DictReader(file) if r["arm"] == M0_NAMES[stage])
        assert (int(row["start"]), int(row["stop"])) == (first, stop)
        score = old
        values = {metric: float(row[metric]) for metric in METRICS}
    else:
        i, score, v = read_score(w, d, f"{arm}_{stage}", "QUERY", root=OUT)
        np.testing.assert_array_equal(i, index)
        np.testing.assert_array_equal(v, available)
    take = (index >= first) & (index < stop)
    np.testing.assert_array_equal(index[take], np.arange(first, stop))
    points, scores = labels(w, d)[index[take]], score[take]
    if arm != "M0":
        computed, _ = continuous_metrics(points, scores, load_author_metric(w / AUTHOR))
        values = {metric: float(computed[metric]) for metric in METRICS}
    ap, roc = independent_check(points, scores)
    np.testing.assert_allclose([ap, roc], [values["AP"], values["ROC_AUC"]], atol=1e-11, rtol=0)
    return dict(domain=d, arm=arm, stage=stage, start=first, stop=stop,
                points=len(points), source="archived" if arm == "M0" else "new",
                **values, independent_AP=ap, independent_ROC_AUC=roc,
                evaluation_seconds=time.monotonic()-start)


def comparisons(rows):
    lookup = {(r["domain"], r["arm"], r["stage"]): r for r in rows}
    delta_rows = []
    for d in MIXED:
        for stage in STAGES:
            for reference in ("C", "M0"):
                a, b = (lookup[d, arm, stage] for arm in ("M1", reference))
                delta_rows.append(dict(domain=d, stage=stage, contrast=f"M1-{reference}",
                                       **{m: a[m]-b[m] for m in METRICS}))
        for arm in ("M0", *ARMS):
            a, b = (lookup[d, arm, stage] for stage in ("U", "Z"))
            delta_rows.append(dict(domain=d, stage="U-Z", contrast=arm,
                                   **{m: a[m]-b[m] for m in METRICS}))
    u = {(r["domain"], r["contrast"]): r for r in delta_rows if r["stage"] == "U"}
    contrasts = ("M1-C", "M1-M0")
    averages = {c: float(np.mean([u[d, c]["AP"] for d in MIXED])) for c in contrasts}
    dual_gain = [d for d in MIXED if all(u[d, c]["AP"] >= .005 for c in contrasts)]
    all_floor = all(r["AP"] >= -.005 for r in u.values())
    costs = [dict(domain=d, contrast=c, metric=m, difference=r[m])
             for (d, c), r in u.items() for m in METRICS[1:] if r[m] < -.01]
    decision = dict(continue_fixed_candidate=bool(dual_gain and all_floor and
                    all(value > 0 for value in averages.values()) and not costs),
                    both_AP_gains_at_least_005=dual_gain, all_AP_differences_at_least_minus_005=all_floor,
                    mean_AP_differences=averages, other_metric_regressions_exceeding_01=costs)
    return delta_rows, decision


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("archived_metrics", type=Path)
    args = p.parse_args()
    w = args.workspace
    output = w / OUT / "metrics"
    if (output / "decision.json").exists():
        raise FileExistsError(output / "decision.json")
    sealed(w)
    output.mkdir(exist_ok=True)
    jobs = [(w, d, a, s, args.archived_metrics)
            for d in MIXED for a in ("M0", *ARMS) for s in STAGES]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(evaluate, jobs))
    delta, decision = comparisons(rows)
    save_rows(output / "methods.csv", rows)
    save_rows(output / "comparisons.csv", delta)
    base.dump(output / "decision.json", decision)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
