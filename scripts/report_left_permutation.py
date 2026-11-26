"""Compare sealed shuffled-weight endpoints with archived matched LEFT runs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from hybrid_tsad import left_multisource_eval as ev
from hybrid_tsad.left_reliability import (
    OUT as WEIGHTS,
    PERMUTATION_OUT,
    permutation_plan,
)
from report_left_memory_update import METRICS, OPS, markdown
from sklearn.metrics import average_precision_score, roc_auc_score

OLD = Path("docs/experiments/left_local_reliability_20261126")
DOC = Path("docs/experiments/left_smd_reliability_20261126")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(w, d, condition, rank, ops):
    root = w / PERMUTATION_OUT / d
    arm = "permuted_" + condition
    folder = root / arm
    receipt = read(folder / "receipt.json")
    original = read(w / WEIGHTS / d / ("window_" + condition) / "receipt.json")
    for key in (
        "seed",
        "step",
        "updates",
        "parent",
        "plan_sha256",
        "weights_sha256",
        "target_windows",
        "source_windows",
    ):
        assert receipt[key] == original[key]
    assert digest(folder / "selected.pt") == receipt["sha256"]
    perms = np.load(folder / "permutation.npy")
    assert digest(folder / "permutation.npy") == receipt["permutation_sha256"]
    np.testing.assert_array_equal(perms, permutation_plan())
    np.testing.assert_array_equal(
        np.sort(perms, axis=1), np.tile(np.arange(16), (2000, 1))
    )
    weights = np.load(w / WEIGHTS / d / "weights" / (condition + ".npy"), mmap_mode="r")
    mean = weights.mean(2)
    shuffled = np.take_along_axis(mean, perms, axis=1)
    np.testing.assert_array_equal(np.sort(shuffled), np.sort(mean))
    order_error = np.abs(mean - shuffled)
    np.testing.assert_allclose(shuffled.mean(1), 1, atol=3e-7, rtol=0)
    banks = {}
    scoring_seconds = 0
    for role in ("QUERY", "CAL"):
        path = root / "banks" / f"{arm}_{role}.npz"
        meta = read(path.with_suffix(".json"))
        assert (
            digest(path) == meta["sha256"]
            and meta["checkpoint_sha256"] == receipt["sha256"]
        )
        scoring_seconds += meta["seconds"]
        with np.load(path) as f:
            banks[role] = (f["index"], f["score"], f["available"])
    axis = np.arange(rank.start, rank.stop)
    idx, scores, av = banks["QUERY"]
    at = np.searchsorted(idx, axis)
    np.testing.assert_array_equal(idx[at], axis)
    assert np.all((av[at] >= axis) & (av[at] - axis < 192))
    scores = scores[at]
    y = ev.labels(w, d)[axis].astype(bool)
    ap, auc = average_precision_score(y, scores), roc_auc_score(y, scores)
    np.testing.assert_allclose([ap, auc], [rank.AP, rank.ROC_AUC], rtol=0, atol=1e-12)
    ci, cs, _ = banks["CAL"]
    cal = ev.PSM_CAL if d == "PSM" else ev.SW_CAL
    ca = np.searchsorted(ci, cal)
    np.testing.assert_array_equal(ci[ca], cal)
    tau = np.sort(cs[ca])[int(np.ceil(0.99 * (len(ca) - 1)))]
    strict = scores > tau
    for stream, mask in (
        ("strict", strict),
        ("confirmed3", np.r_[False, False, strict[:-2] & strict[1:-1] & strict[2:]]),
    ):
        row = ops.loc[stream]
        np.testing.assert_allclose(tau, row.threshold, atol=1e-12, rtol=0)
        assert np.sum(mask & y) == row.TP and np.sum(mask & ~y) == row.FP
    return dict(
        domain=d,
        arm=arm,
        permutation=receipt["permutation_sha256"],
        checkpoint=receipt["sha256"],
        input_plan=receipt["plan_sha256"],
        weights=receipt["weights_sha256"],
        source=receipt["parent"],
        fixed_slots=int(np.sum(perms == np.arange(16))),
        slots=perms.size,
        mean_abs_weight_difference=float(order_error.mean()),
        changed_weights=int(np.count_nonzero(order_error)),
        AP_checked=ap,
        ROC_AUC_checked=auc,
        threshold_checked=float(tau),
        training_seconds=receipt["seconds"],
        scoring_seconds=scoring_seconds,
        peak_bytes=receipt["peak_bytes"],
    )


def collect(w):
    DOC.mkdir(parents=True, exist_ok=True)

    def load(file, previous):
        df = pd.read_csv(OLD / previous)
        parts = [df[df.method.isin(["window", "replay"])]]
        for d in ("PSM", "SWAT"):
            rows = pd.read_csv(w / PERMUTATION_OUT / d / file)
            rows[["method", "condition"]] = rows.arm.str.split("_", expand=True)
            parts.append(rows)
        return pd.concat(parts, ignore_index=True)

    ranks = load("mixed.csv", "methods.csv")
    ops = load("operations.csv", "operations.csv")
    events = load("events.csv", "events.csv")
    reviews = []
    for _, row in ranks[ranks.method == "permuted"].iterrows():
        subset = ops[(ops.domain == row.domain) & (ops.arm == row.arm)].set_index(
            "stream"
        )
        reviews.append(verify(w, row.domain, row.condition, row, subset))

    def differences(df, cols):
        contrasts = []
        for d, group in df.groupby("domain"):
            wide = group.set_index(["method", "condition"])[cols]
            for high, low in (
                ("window", "permuted"),
                ("permuted", "replay"),
                ("window", "replay"),
            ):
                diff = wide.loc[high] - wide.loc[low]
                for k in "ABC":
                    contrasts.append(
                        dict(
                            domain=d,
                            comparison=high + " minus " + low,
                            condition=k,
                            **diff.loc[k],
                        )
                    )
                contrasts.append(
                    dict(
                        domain=d,
                        comparison=high + " minus " + low,
                        condition="B-C",
                        **(diff.loc["B"] - diff.loc["C"]),
                    )
                )
        return pd.DataFrame(contrasts)

    delta = differences(ranks, METRICS)
    opdelta = pd.concat(
        [differences(sub, OPS).assign(stream=s) for s, sub in ops.groupby("stream")]
    )
    review = pd.DataFrame(reviews)
    for name, df in (
        ("methods", ranks),
        ("contrasts", delta),
        ("operations", ops),
        ("operating_contrasts", opdelta),
        ("events", events),
        ("review", review),
    ):
        df.to_csv(DOC / ("permutation_" + name + ".csv"), index=False)
    wide = ranks.pivot(index=["condition", "method"], columns="domain", values=METRICS)
    wide = wide.loc[:, [(m, d) for d in ("PSM", "SWAT") for m in METRICS]]
    wide.columns = [d + " " + m for m, d in wide.columns]
    pairs = delta[delta.comparison == "window minus permuted"].drop(
        columns="comparison"
    )
    fixed = int(review.fixed_slots.iloc[0])
    text = [
        "# Window reliability and its shuffled assignment",
        "Each of six runs used the matching source checkpoint, replay batches and frozen coefficient bank. A separate NumPy generator shuffled the 16 window coefficients inside each batch. The same permutation sequence was used for both datasets and all three conditions. Ordinary random permutations retained "
        + str(fixed)
        + " of the 32,000 assignments in their original slot per run. Each batch retained the exact coefficient multiset. The 16 source coefficients remained one.",
        "## Mixed-target detection",
        "A is natural exposure. B and C replace the same 3,200 common target slots with fully anomalous and fully nominal 192-point windows. All methods use native retrospective LEFT scores; PSM uses [22656,87744) and SWaT uses [608535,944856).",
        markdown(wide.reset_index()),
        "## Correct assignment minus shuffled assignment",
        markdown(pairs),
        "Correctly assigned coefficients improve all four PSM ranking measures over shuffled coefficients in A, B and C. The largest AP difference (+0.0180) occurs under additional anomalous exposure. The original mixed CAL gives fewer confirmed false alarms alongside lower recall and fewer one-to-one event matches. On SWaT the AP advantage is smaller (+0.0009 in A, +0.0034 in B); VUS-ROC is lower in all conditions. In B, correct assignment gains 0.210 in long-event macro coverage but adds 0.054 in confirmed false-positive rate and matches two fewer events. These comparisons support a task-dependent contribution from matching weights to windows. They do not establish a general advantage at the original operating points.",
        "B−C in this table is the difference of the two methods’ within-method exposure contrasts. Positive ranking differences favor correct assignment. One development seed is reported.",
        "Alarm evaluation used each endpoint’s own fixed CAL q99/higher threshold and strict exceedance. The 512 selected CAL points contain 512 attack-labeled points on PSM and 38 on SWaT. [Operating results](permutation_operations.csv) · [paired operating differences](permutation_operating_contrasts.csv) · [events](permutation_events.csv).",
        f"Six new training runs used {review.training_seconds.sum():.2f} A40 seconds in total; twelve score banks used {review.scoring_seconds.sum():.2f} seconds. Independent CPU checks reproduced the assignments, per-batch coefficient multisets, model and bank hashes, AP, ROC-AUC, CAL thresholds and strict/three-confirmation counts. VUS and event matching used the existing evaluator. [Checks and costs](permutation_review.csv).",
    ]
    (DOC / "permutation.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    collect(p.parse_args().workspace)
