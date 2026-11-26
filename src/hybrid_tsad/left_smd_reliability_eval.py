"""CPU-only evaluation of sealed SMD38 LEFT score banks on fixed test-record grids."""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .left_decoder_context_metrics import AUTHOR
from .left_decoder_context_run import PUBLIC
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_smd28_author import bindings
from .left_swat_author import read, sha
from .trajectory_evaluation import independent_check
from .timercd_swat_evaluation import evaluate_score

OUT = Path("results/left_smd_reliability_20261126_v1")
METRICS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")
PUBLIC_ARMS = ("source", "N", "scratch_N")
MIXED_ARMS = ("source", "random_u", "ordinary", "replay", "window", "source_only")


def bank(w, entity, arm, role):
    if arm == "scratch_N":
        root = w / PUBLIC / "SMD" / entity / "native"
        path = root / "scores.npz"
        digest = read(root / "SCORES.json")["bank_sha256"]
        avail_key = "available_index"
    else:
        path = w / OUT / entity / "banks" / f"{arm}_{role}.npz"
        digest = read(path.with_suffix(".json"))["sha256"]
        avail_key = "available"
    assert sha(path) == digest
    with np.load(path) as z:
        idx, scores, avail = z["index"], z["score"], z[avail_key]
    assert idx.shape == scores.shape == avail.shape and np.isfinite(scores).all()
    np.testing.assert_array_equal(avail, (idx // 192 + 1) * 192 - 1)
    return idx, scores, avail, digest


def ranked(y, s, metric):
    if len(np.unique(y)) < 2:
        return dict.fromkeys(METRICS), None
    scores, _ = continuous_metrics(y, s, metric)
    independent = independent_check(y, s)
    err = float(
        np.max(np.abs(independent - np.array([scores["AP"], scores["ROC_AUC"]])))
    )
    assert err < 1e-10
    return {k: scores[k] for k in METRICS}, err


def measure(task):
    w, entity, arm, role = task
    rec = bindings(w)[entity]
    manifest = (
        w
        / "preparations/timeradar_smd_full_20261116_v1/evaluation_manifest_full28.json"
    )
    assert (
        sha(manifest)
        == "aee4c628d138cc2355535fb5e0bf24fa103d88fd44a50953b8ed2d4e2a05132b"
    )
    record = next(r for r in read(manifest)["records"] if r["id"] == entity)
    label_path = w / record["path"]
    assert sha(label_path) == record["sha256"]
    with np.load(label_path) as z:
        y = z["label"].reshape(-1).astype(int)
        np.testing.assert_array_equal(z["index"], np.arange(len(y)))
    assert len(y) == rec["query_points"] and np.isin(y, [0, 1]).all()
    idx, s, avail, digest = bank(w, entity, arm, role)
    first = 0 if role == "TEST" else ((len(y) // 3 + 544 + 191) // 192) * 192
    np.testing.assert_array_equal(idx, np.arange(first, len(y) // 192 * 192))
    metric = load_author_metric(w / AUTHOR)
    rankings = []
    axes = [("public" if role == "TEST" else "mixed", first, int(idx[-1]) + 1)]
    if role == "TEST":
        axes.append(
            ("external_common", 99, min(len(y) // 100 * 100, len(y) // 192 * 192))
        )
    for panel, lo, hi in axes:
        sl = slice(lo - first, hi - first)
        scores, err = ranked(y[lo:hi], s[sl], metric)
        rankings.append(
            dict(
                entity=entity,
                arm=arm,
                panel=panel,
                start=lo,
                stop=hi,
                points=hi - lo,
                positives=int(y[lo:hi].sum()),
                **scores,
                bank_sha256=digest,
                check_ap_roc_maxerr=err,
            )
        )
    ops, events = [], []
    if role == "QUERY":
        ci, cs, _, cal_hash = bank(w, entity, arm, "CAL")
        ntrain = rec["train_points"]
        cal_first = ((ntrain * 4 // 5 + 544 + 191) // 192) * 192
        np.testing.assert_array_equal(
            ci, np.linspace(cal_first, ntrain // 192 * 192 - 1, 512, dtype=np.int64)
        )
        cutoff = np.quantile(cs, 0.99, method="higher")
        assert np.sort(cs)[int(np.ceil(0.99 * (len(cs) - 1)))] == cutoff
        ops, events, _, _ = evaluate_score(
            idx,
            s,
            y,
            float(cutoff),
            avail,
            query_start=first,
            query_stop=int(idx[-1]) + 1,
            original_length=len(y),
            expected_events=None,
            ranking=rankings[0],
        )
        for row in ops:
            row.update(cal_points=len(cs), cal_hash=cal_hash)
            if row["Ntrue"] == 0:
                row["one_to_one_eventF1"] = None
            if not y[idx].any():
                row["Recall"] = row["F1"] = None
            bits = s > cutoff
            if row["stream"] == "confirmed3":
                bits = np.r_[False, False, bits[2:] & bits[1:-1] & bits[:-2]]
            assert (
                row["FP"] == (bits & (y[idx] == 0)).sum()
                and row["TP"] == (bits & (y[idx] == 1)).sum()
            )
        for row in ops + events:
            row.update(entity=entity, arm=arm)
    dest = w / OUT / "metrics"
    dest.mkdir(exist_ok=True)
    dst = dest / f"{entity}_{arm}_{role}.json"
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            dict(rankings=rankings, operations=ops, events=events), allow_nan=False
        )
    )
    os.replace(tmp, dst)


def ready(w, entity, arm, role):
    if arm == "scratch_N":
        return (w / PUBLIC / "SMD" / entity / "native" / "SCORES.json").is_file()
    root = w / OUT / entity / "banks"
    return all(
        (root / f"{arm}_{key}.json").is_file()
        for key in (("QUERY", "CAL") if role == "QUERY" else ("TEST",))
    )


def psm_external(w):
    from .left_multisource_eval import NATIVE, labels, read_score
    from .trajectory_evaluation import save_rows

    data = labels(w, "PSM")[99:87744]
    output = []
    for arm in PUBLIC_ARMS:
        if arm == "scratch_N":
            root = w / NATIVE / "PSM" / "native"
            path = root / "raw_scores.npz"
            assert sha(path) == read(root / "SCORE_RECEIPT.json")["bank_sha256"]
            with np.load(path) as z:
                idx = z["absolute_time_index"] - 132481
                score = z["canonical_score"]
        else:
            idx, score, _ = read_score(w, "PSM", arm, "TEST")
        np.testing.assert_array_equal(idx, np.arange(87744))
        result, err = ranked(data, score[99:87744], load_author_metric(w / AUTHOR))
        output.append(
            dict(arm=arm, start=99, stop=87744, **result, check_ap_roc_maxerr=err)
        )
    save_rows(w / OUT / "psm_external.csv", output)


def run(w, workers):
    if not (w / OUT / "psm_external.csv").exists():
        psm_external(w)
    tasks = [
        (w, e, a, r)
        for e in sorted(bindings(w))
        for r, arms in (("TEST", PUBLIC_ARMS), ("QUERY", MIXED_ARMS))
        for a in arms
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while tasks:
            pending = []
            work = []
            for task in tasks:
                _, e, a, r = task
                if (w / OUT / "metrics" / f"{e}_{a}_{r}.json").exists():
                    continue
                if ready(*task):
                    work.append(task)
                else:
                    pending.append(task)
            if work:
                list(pool.map(measure, work))
            elif pending:
                time.sleep(15)
            tasks = pending


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.workspace, args.workers)
