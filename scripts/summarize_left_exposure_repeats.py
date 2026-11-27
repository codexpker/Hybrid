"""Evaluate new sealed LEFT banks, then rebuild paired tables from archived metrics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import statistics
import time

DOMAINS = ("PSM", "SWAT")
SEEDS = (2025, 2026, 2027)
METRICS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")
FREEZE = "docs/experiments/left_evidence_freeze_20261128"
REPEAT = "docs/experiments/left_fixed_exposure_repeats_20261128"


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one(job):
    import numpy as np
    from hybrid_tsad.left_multisource_eval import (
        AUTHOR, MIXED, OUT, continuous_metrics, independent_check, labels, load_author_metric, read_score,
    )

    workspace, output, domain, method, condition, seed = job
    arm = f"{method}_{condition}_{seed}"
    idx, score, available = read_score(workspace, domain, arm, "QUERY", root=output)
    old_idx, _, old_available = read_score(workspace, domain, "replay", "QUERY", root=OUT)
    np.testing.assert_array_equal(idx, old_idx)
    np.testing.assert_array_equal(available, old_available)
    lo, hi = MIXED[domain]
    take = (lo <= idx) & (idx < hi)
    np.testing.assert_array_equal(idx[take], np.arange(lo, hi))
    y = labels(workspace, domain)[idx[take]]
    scores = score[take]
    started = time.monotonic()
    metrics, _ = continuous_metrics(y, scores, load_author_metric(workspace / AUTHOR))
    error = float(max(abs(np.array(independent_check(y, scores)) - [metrics[k] for k in METRICS[:2]])))
    assert error < 1e-11, error
    return dict(domain=domain, method=method, condition=condition, seed=seed,
                start=lo, stop=hi, points=len(y), positives=int(y.sum()),
                **{k: metrics[k] for k in METRICS}, independent_AP_ROC_max_error=error,
                metric_seconds=time.monotonic()-started)


def evaluate(workspace, output):
    from hybrid_tsad.left_multisource import read, sha

    started = time.monotonic()
    identity = read(output/"identity.json")
    jobs, receipts, traces = [], [], {}
    for spec in identity["jobs"]:
        d, m, c, s = (spec[k] for k in ("domain", "method", "condition", "seed"))
        arm = f"{m}_{c}_{s}"
        model = output/d/arm
        raw = output/d/"banks"/f"{arm}_QUERY.npz"
        train = read(model/"receipt.json")
        bank = read(raw.with_suffix(".json"))
        fold = identity["folds"][d]
        condition = fold["conditions"][c]
        assert sha(model/"selected.pt") == train["sha256"] == bank["checkpoint_sha256"]
        assert sha(raw) == bank["sha256"]
        assert train["numeric_plan"] == condition["plan_sha256"]
        assert train["teacher_weights"] == (condition["weights_sha256"] if m == "W" else None)
        assert train["parent"] == fold["parent"] and train["scaler"] == fold["scaler"]
        assert train["step"] == bank["step"] and train["updates"] == 2000
        assert (train["domain"], train["method"], train["condition"], train["seed"]) == (d,m,c,s)
        assert not train["training_labels_read"]
        traces[d,m,c,s] = train["trace"]
        jobs.append((workspace,output,d,m,c,s))
        receipts.append(dict(training=train, bank=bank))
    for d in DOMAINS:
        for c in "ABC":
            first, second = (traces[d,"R",c,s] for s in SEEDS[1:])
            assert [x["batch"] for x in first] == [x["batch"] for x in second]
            assert first[0]["rng_before"] != second[0]["rng_before"]
            for s in SEEDS[1:]:
                r, w = (traces[d,m,c,s] for m in "RW")
                assert len(r) == len(w) == 4
                for key in ("batch", "rng_before", "rng_after"):
                    assert [x[key] for x in r] == [x[key] for x in w], (d,c,s,key)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(evaluate_one, jobs))
    for row, receipt in zip(rows, receipts):
        row.update(checkpoint_sha256=receipt["training"]["sha256"], bank_sha256=receipt["bank"]["sha256"])
    write_rows(output/"new_metrics.csv", rows)
    (output/"receipts.json").write_text(json.dumps(dict(jobs=receipts, CPU_wall_seconds=time.monotonic()-started,
        independent_AP_ROC_max_error=max(r["independent_AP_ROC_max_error"] for r in rows),
        VUS_verification="Pinned implementation only; no second independent implementation"),indent=2)+"\n")


def tables(repo, out):
    keys = ("domain", "seed", "method", "condition", "start", "stop")
    legacy = read_rows(repo/FREEZE/"abc_metrics.csv")
    endpoints = [
        dict(zip(keys, [row["domain"], "2025", {"D": "R", "E": "W"}[row["method_id"]],
                        row["condition"], row["start"], row["stop"]]),
             **{k: row[k] for k in METRICS}, source=row["key"])
        for row in legacy if row["method_id"] in ("D","E")
    ]
    new = read_rows(out/"new_metrics.csv")
    endpoints.extend(dict(**{k: row[k] for k in (*keys,*METRICS)}, source=row["bank_sha256"]) for row in new)
    endpoints.sort(key=lambda r: (r["domain"], r["seed"], r["condition"], r["method"]))
    lookup = {(r["domain"], int(r["seed"]), r["method"], r["condition"]): r for r in endpoints}
    assert len(endpoints) == len(lookup) == 36
    paired = []
    for d in DOMAINS:
        for seed in SEEDS:
            rows = {(m,c): lookup[d,seed,m,c] for m in "RW" for c in "ABC"}
            assert len({(r["start"], r["stop"]) for r in rows.values()}) == 1
            for metric in METRICS:
                values = {m+c: float(r[metric]) for (m,c), r in rows.items()}
                direct = {f"W_minus_R_{c}": values[f"W{c}"]-values[f"R{c}"] for c in "ABC"}
                r = values["RB"]-values["RC"]
                w = values["WB"]-values["WC"]
                paired.append(dict(domain=d,seed=seed,metric=metric,**values,**direct,
                                   R_B_minus_C=r,W_B_minus_C=w,interaction=w-r))
    summaries = []
    for d in DOMAINS:
        for metric in METRICS:
            subset = [r for r in paired if r["domain"] == d and r["metric"] == metric]
            for contrast in ("W_minus_R_A","W_minus_R_B","W_minus_R_C","R_B_minus_C","W_B_minus_C","interaction"):
                delta = [r[contrast] for r in subset]
                summaries.append(dict(domain=d,metric=metric,contrast=contrast,n=len(delta),
                                      mean=statistics.mean(delta),sample_sd=statistics.stdev(delta),
                                      minimum=min(delta),maximum=max(delta),positive=sum(x>0 for x in delta),
                                      zero=delta.count(0),negative=sum(x<0 for x in delta)))
    for name, rows in (("endpoints",endpoints),("paired",paired),("paired_summary",summaries)):
        write_rows(out/(name+".csv"),rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate","tables"))
    parser.add_argument("repository_or_workspace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "evaluate":
        evaluate(args.repository_or_workspace.resolve(), args.output.resolve())
    else:
        tables(args.repository_or_workspace.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
