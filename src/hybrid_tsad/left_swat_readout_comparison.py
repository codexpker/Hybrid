"""Compare frozen LEFT endpoint and native full-window readouts on one SWaT axis."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .left_native_swat_metrics import (
    AUTHOR_METRIC_SHA, LABEL_SHA, SCORED, START, WINDOW, continuous_metrics, load_author_metric, sha,
)
from .timercd_swat_evaluation import (
    BASELINE_SEAL_SHA, CAL_END, CAL_START, OLD, PREP, QUERY_END, QUERY_START, threshold,
)


MODELS = ("noU", "U_A", "U_B", "U_C")
ROLE_BOUNDS = {"CAL": (445661, CAL_END), "QUERY": (608023, QUERY_END)}
FULL_AXES = {"CAL": (CAL_START, CAL_END), "QUERY": (QUERY_START, START + SCORED)}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def full_bank(path, role):
    lo, hi = FULL_AXES[role]
    first, stop = ROLE_BOUNDS[role]
    index = np.arange(lo, hi, dtype=np.int64)
    start = START + ((index - START) // WINDOW) * WINDOW
    end = start + WINDOW - 1
    with np.load(path, allow_pickle=False) as bank:
        if (not np.array_equal(bank["absolute_time_index"], index)
                or not np.array_equal(bank["available_index"], end)
                or not np.array_equal(bank["window_start"], np.unique(start))
                or not np.array_equal(bank["window_end_exclusive"], np.unique(end) + 1)
                or (int(bank["role_input_start"]), int(bank["role_input_end"])) != (first, stop)
                or np.any(start < first) or np.any(end >= stop)):
            raise ValueError(f"Full-window {role} footprint or point axis changed")
        score, per_channel = bank["canonical_score"], bank["per_channel"]
        if (per_channel.shape != (len(index), 31) or per_channel.dtype != np.float32
                or score.shape != index.shape or score.dtype != np.float32
                or not np.array_equal(bank["valid"], np.ones(len(index), dtype=bool))
                or not np.isfinite(score).all() or not np.array_equal(score, per_channel.mean(axis=-1))):
            raise ValueError(f"Invalid native channel mean for {role}")
    return index, score, end


def strict_points(truth, score, tau):
    alarms = score > tau
    false_positive = int(np.count_nonzero(alarms & ~truth))
    true_positive = int(np.count_nonzero(alarms & truth))
    return dict(FP=false_positive, TP=true_positive,
                normal_FPR=false_positive / int(np.count_nonzero(~truth)),
                recall=true_positive / int(np.count_nonzero(truth)))


def run(workspace, full_root, full_seal, author_path, out):
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError(out)
    old_root = workspace / OLD / "fixed_scores"
    old_seal = old_root / "ALL_SCORES_THRESHOLDS_MASKS_SEALED.json"
    if sha(old_seal) != BASELINE_SEAL_SHA:
        raise ValueError("Historical endpoint seal mismatch")
    old_hashes = {item["method"]: item["sha256"] for item in read_json(old_seal)["descriptors"]}
    native_seal = read_json(full_seal)
    full_hashes = {item["relative"]: item["sha256"] for item in native_seal["banks"]}
    expected = {f"{model}/{role}.npz" for model in MODELS for role in FULL_AXES}
    if (native_seal["status"] != "COMPLETED" or not native_seal["no_labels_decoded"]
            or set(full_hashes) != expected or len(native_seal["banks"]) != len(expected)):
        raise ValueError("Require the complete four-model, two-role native score seal")

    banks, model_ids = {}, {}
    for model in MODELS:
        old_run = workspace / OLD
        original = read_json(old_run / ("left_score/RECEIPT.json" if model == "noU" else
                                       f"U/{model[-1]}/score/RECEIPT.json"))
        receipt = read_json(full_root / model / "RECEIPT.json")
        old_step = original.get("python_step_restored", original.get("python_step"))
        if (receipt["arm"] != model or receipt["status"] != "COMPLETED" or receipt["labels_read"]
                or receipt["checkpoint_selection_changed"] or receipt["training_updates"] != 0
                or not receipt["model_state_step_RNG_unchanged"]
                or receipt["checkpoint_sha256"] != original["checkpoint_sha256"]
                or receipt["python_step"] != old_step
                or {item["role"]: item["sha256"] for item in receipt["roles"]}
                != {item["role"]: item["sha256"] for item in original["roles"]}):
            raise ValueError(f"Frozen scoring state differs from the historical run for {model}")
        model_ids[model] = dict(checkpoint_sha256=receipt["checkpoint_sha256"], python_step=old_step,
                                scaler_sha256=receipt["scaler_sha256"])
        native = {}
        for role in FULL_AXES:
            key = f"{model}/{role}.npz"
            path = full_root / key
            if sha(path) != full_hashes[key] or full_hashes[key] != next(
                    item["sha256"] for item in receipt["banks"] if item["role"] == role):
                raise ValueError(f"Native score seal mismatch for {key}")
            native[role] = full_bank(path, role)
        old_path = old_root / f"{model}.npz"
        if sha(old_path) != old_hashes[model]:
            raise ValueError(f"Historical endpoint bank mismatch for {model}")
        with np.load(old_path, allow_pickle=False) as old:
            index = old["index"]
            if not np.array_equal(index, np.arange(QUERY_START, QUERY_END)):
                raise ValueError(f"Historical endpoint axis changed for {model}")
            endpoint_score = old["score"][:len(native["QUERY"][0])]
            old_tau, old_points = threshold(old["CAL_index"], old["CAL_score"])
            if not np.array_equal(old["selected_CAL"], old_points):
                raise ValueError(f"Historical selected CAL points changed for {model}")
        native_tau, native_points = threshold(*native["CAL"][:2])
        if not np.isfinite(endpoint_score).all() or not np.array_equal(native_points, old_points):
            raise ValueError(f"The two readouts must use the same frozen CAL points for {model}")
        banks[model] = native, endpoint_score, old_tau, native_tau

    label_path = workspace / PREP / "SWAT_LABEL_METADATA.npz"
    if sha(label_path) != LABEL_SHA:
        raise ValueError("SWaT label asset changed")
    with np.load(label_path, allow_pickle=False) as label_bank:
        labels = label_bank["label"]
    if labels.shape != (QUERY_END,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("SWaT label axis changed")
    cal_points = np.linspace(CAL_START, CAL_END - 1, 512, dtype=np.int64)
    if labels[cal_points].any():
        raise ValueError("The original CAL points must be normal")
    query_lo, query_hi = FULL_AXES["QUERY"]
    truth = labels[query_lo:query_hi].astype(bool)
    metric = load_author_metric(author_path)
    rows, surfaces, end_checks = [], {}, {}
    for model, (native, endpoint_score, old_tau, native_tau) in banks.items():
        idx, full_score, available = native["QUERY"]
        at_window_end = idx == available
        end_checks[model] = dict(points=int(at_window_end.sum()), max_abs=float(
            np.max(np.abs(full_score[at_window_end] - endpoint_score[at_window_end]))))
        for name, score, tau in (("endpoint", endpoint_score, old_tau), ("full_window", full_score, native_tau)):
            row, curves = continuous_metrics(truth, score, metric)
            rows.append(dict(model=model, readout=name, threshold=tau, **row, **strict_points(truth, score, tau)))
            for field, array in zip(("TPR", "FPR", "precision", "windows"), curves):
                surfaces[f"{model}_{name}_{field}"] = array

    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "VUS_SURFACES.npz", **surfaces)
    result = dict(query_axis=[query_lo, query_hi], query_points=len(truth),
                  positive=int(truth.sum()), negative=int((~truth).sum()), fixed_cal_points=len(cal_points),
                  original_unscored_tail_positive=int(labels[query_hi:].sum()),
                  frozen_model_identities=model_ids,
                  original_frozen_endpoint_bank_sha256={m: old_hashes[m] for m in MODELS},
                  full_window_bank_sha256=full_hashes, full_window_seal_sha256=sha(full_seal),
                  label_sha256=LABEL_SHA, vus_source_sha256=AUTHOR_METRIC_SHA,
                  vus_window_size=100, vus_threshold_grid=250, query_shared_window_ends=end_checks,
                  rows=rows, cpu_wall_seconds=time.perf_counter() - started)
    with (out / "METRICS.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("workspace", "full-root", "full-seal", "author-metric", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.workspace, args.full_root, args.full_seal, args.author_metric, args.out), indent=2))


if __name__ == "__main__":
    main()
