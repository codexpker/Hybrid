"""Evaluate fixed LEFT replay controls and draw their CAL operating points."""

import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from .left_author_u_transfer import make_replay_plan
from .left_native_swat_metrics import (
    AUTHOR_METRIC_SHA, LABEL_SHA, START, WINDOW, continuous_metrics, load_author_metric, sha,
)
from .left_swat_natural_u_evaluation import (
    AXES, CAL_POINTS, INIT_CHECKPOINT_SHA, INIT_STEP, METRIC_PATH, SCALER_SHA, UPDATES,
    checked, q99, read_json, read_scores, role_axis, truth, write_json,
)
from .timercd_swat_evaluation import evaluate_score


ARMS = ("TT", "SS", "R")
METHODS = ("noU", "U", *ARMS)
RANK_KEYS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "AUC_PR_trapezoidal")
BASELINE_METRICS_SHA = "ce9fb0394410701321153744caa405f3f63be87ba057ddf78b34076d075057da"
BASELINE_SEAL_SHA = "68507716af0e4ce7711b6b7a68339ea9dfc324abd15ff82cdfa31db35c78a264"
CACHE_SEAL_SHA = "7d258293907f6db2994f9d9aa5969242c6cfea8ebbcf3bc75c7c4c52d4a9ece4"
TARGET_PLAN_SHA = "1d4dbd9cfbeb8dcbcf218f7a3bb46176c290a29406392b5eb8a5835f37afa966"


def banks(root, descriptors, per_channel):
    if set(descriptors) != set(AXES):
        raise ValueError("Require the CAL and QUERY banks")
    return {role: read_scores(checked(root / f"{role}.npz", descriptors[role]), role, per_channel)
            for role in AXES}


def plot_curves(output, truth_query, query_scores, rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    palette = ("#374151", "#0072B2", "#D55E00", "#8C4E96", "#008661")
    styles = ("-", "--", ":", "-.", "-")
    legends = ("noU", "Ordinary U", "TT", "SS", "R (source replay)")
    low_roc_limit = .05
    fig, (pr, roc) = plt.subplots(1, 2, figsize=(11.0, 4.3))
    handles, data = [], {}
    op_outside = []
    for model, label, color, style in zip(METHODS, legends, palette, styles):
        score = query_scores[model]
        precision, recall, _ = precision_recall_curve(truth_query, score)
        fpr, tpr, _ = roc_curve(truth_query, score)
        data.update({f"{model}_precision": precision, f"{model}_recall": recall,
                     f"{model}_fpr": fpr, f"{model}_tpr": tpr})
        pr_line, = pr.plot(recall, precision, color=color, ls=style, lw=1.8, label=label)
        handles.append(pr_line)
        roc.plot(fpr, tpr, color=color, ls=style, lw=1.8)
        point = rows[model]
        if point["Precision"] is not None:
            pr.scatter(point["Recall"], point["Precision"], color=color, facecolors="white", zorder=4, s=32)
        if point["normal_FPR"] <= low_roc_limit:
            roc.scatter(point["normal_FPR"], point["Recall"], color=color, facecolors="white", zorder=4, s=32)
        else:
            op_outside.append(f"{model} {point['normal_FPR']:.1%}")

    pr.axhline(np.mean(truth_query), lw=.8, color="#a1a1aa", linestyle="--")
    pr.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Point recall", ylabel="Point precision", title="Precision–recall")
    roc.set(xlim=(0, low_roc_limit), ylim=(0, 1.02), xlabel="Normal-point false positive rate",
            ylabel="Point recall", title="ROC at low false positive rates")
    for axis in (pr, roc):
        axis.xaxis.set_major_formatter(PercentFormatter(1))
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(alpha=.17)
        axis.spines[["top", "right"]].set_visible(False)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, 1.01), ncol=5, frameon=False)
    caption = "Open circles mark each model’s fixed mixed-CAL q99 operating point on QUERY."
    if op_outside:
        caption += "  Outside ROC range: " + ", ".join(op_outside) + "."
    fig.text(.5, .025, caption, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93))
    fig.savefig(output / "pr_low_fpr_roc.png", dpi=240)
    fig.savefig(output / "pr_low_fpr_roc.pdf")
    plt.close(fig)
    np.savez_compressed(output / "PR_ROC_ARRAYS.npz", **data)


def baseline_inputs(cache, natural, metrics_path):
    cache_seal = read_json(checked(cache / "CACHE_SEAL.json", CACHE_SEAL_SHA))
    natural_seal = read_json(checked(natural / "RAW_BANK_SEAL.json", BASELINE_SEAL_SHA))
    metrics = read_json(checked(metrics_path, BASELINE_METRICS_SHA))
    natural_banks = {item["role"]: item["sha256"] for item in natural_seal["banks"]}
    if (metrics["query_axis"] != list(AXES["QUERY"]) or metrics["cal_axis"] != list(AXES["CAL"])
            or metrics["source_checkpoint_sha256"] != INIT_CHECKPOINT_SHA or metrics["source_scaler_sha256"] != SCALER_SHA
            or not np.array_equal(cache_seal["calibration_points"], CAL_POINTS)
            or metrics["adapted_score_bank_sha256"] != natural_banks):
        raise ValueError("Historical noU and ordinary U must match the already accepted common axis and scaler")
    scores = {"noU": banks(cache, cache_seal["banks"], False), "U": banks(natural / "U", natural_banks, True)}
    previous = {model: row for model, row in zip(("noU", "U"), metrics["rows"], strict=True)}
    for model, row in previous.items():
        if q99(scores[model]["CAL"]) != (row["threshold"], row["CAL_exceedances"]):
            raise ValueError(f"Archived {model} CAL threshold differs")
    return scores, previous


def replay_inputs(natural, replay, seed=2025):
    if seed == 2025:
        target_path = checked(natural / "U/U_PLAN.npz", TARGET_PLAN_SHA)
        with np.load(target_path, allow_pickle=False) as archive:
            target = archive["base_starts"]
    else:
        target = np.random.default_rng(seed).integers(0, 90000 - WINDOW + 1, size=(UPDATES, 128))
        target_path = replay / "TARGET_PLAN.npz"
        with np.load(target_path, allow_pickle=False) as archive:
            if not np.array_equal(archive["base_starts"], target):
                raise ValueError("New seed target plan must follow the predetermined sampling rule")
    target_sha = sha(target_path)
    expected_plan = make_replay_plan(target, seed=seed)
    plan_path = replay / "SAMPLING_PLAN.npz"
    plan_sha = sha(plan_path)
    frozen_plan = read_json(replay / "PLAN_FREEZE.json")
    if (frozen_plan["status"] != "FROZEN_BEFORE_TRAINING_OR_QUERY" or frozen_plan["labels_read"]
            or frozen_plan["plan_sha256"] != plan_sha or frozen_plan["seed"] != seed
            or frozen_plan["target_plan_sha256"] != target_sha):
        raise ValueError("Replay plan must be frozen before training or QUERY access")
    with np.load(plan_path, allow_pickle=False) as plan:
        if set(plan.files) != set(expected_plan) or any(
                not np.array_equal(plan[key], draws) for key, draws in expected_plan.items()):
            raise ValueError("Replay sampling differs from the shared mixture and independent half-plans")

    seal_path = replay / "RAW_BANK_SEAL.json"
    seal = read_json(seal_path)
    items = {(item["arm"], item["role"]): item["sha256"] for item in seal["banks"]}
    if (seal["status"] != "COMPLETED" or not seal["no_labels_decoded"] or seal["sampling_plan_sha256"] != plan_sha
            or set(items) != {(model, role) for model in ARMS for role in AXES}
            or len(seal["banks"]) != len(items) or len(seal["jobs"]) != len(ARMS)
            or any(job["OS_exit_code"] != 0 for job in seal["jobs"])):
        raise ValueError("All three completed, label-free train-and-score arms are required")

    pairings = {"R": ["T", "S"], "TT": ["T", "T_prime"], "SS": ["S_prime", "S"]}
    target_role, source_role = [495000, 585000], [0, 396000]
    roles = {"R": [target_role, source_role], "TT": target_role, "SS": source_role}
    coverage = {"CAL": [585544, 607479], "QUERY": [608023, 944919]}
    scores, identities = {}, {}
    for model in ARMS:
        root = replay / model
        trained = read_json(root / "train/RECEIPT.json")
        binding = read_json(root / "REPLAY_BINDING.json")
        exposure = read_json(root / "train/EXPOSURE_RECEIPT.json")
        scored = read_json(root / "RECEIPT.json")
        if (trained["status"] != "COMPLETED" or trained["initial_python_step"] != INIT_STEP
                or trained["final_python_step"] != INIT_STEP + UPDATES or trained["updates"] != UPDATES
                or trained["labels_read"] or trained["CAL_QUERY_read"] or not trained["endpoint_only"]
                or trained["optimizer"] != "Adam lr1e-4" or trained["seed"] != seed
                or trained["train_role"] != roles[model] or trained["plan_sha256"] != plan_sha
                or binding["arm"] != model or binding["pairing"] != pairings[model] or binding["batch"] != 256
                or binding["labels_read"] or not binding["single_full_native_forward_per_update"]
                or binding["plan_sha256"] != plan_sha or binding["original_target_plan_sha256"] != target_sha
                or exposure["batch"] != 256 or not exposure["full_batch_native_loss"]
                or exposure["optimizer_steps"] != UPDATES or exposure["native_forward_calls"] != UPDATES
                or exposure["window_exposures"] != UPDATES * 256):
            raise ValueError(f"{model} training does not match the frozen ordinary-U control")
        if (scored["status"] != "COMPLETED" or scored["labels_read"] or scored["python_step"] != INIT_STEP + UPDATES
                or not scored["readout_state_step_RNG_immutable"] or not scored["source_checkpoint_unchanged"]
                or scored["worker_sha256"] != frozen_plan["worker_sha256"]
                or (scored["anchor"], scored["window"], scored["stride"]) != (START, WINDOW, WINDOW)
                or scored["checkpoint_sha256"] != trained["checkpoint_sha256"]
                or {item["role"]: item["input"] for item in scored["coverage"]} != coverage):
            raise ValueError(f"{model} scoring does not use the fixed role windows and training endpoint")
        if any(r["source_checkpoint_sha256"] != INIT_CHECKPOINT_SHA or r["scaler_sha256"] != SCALER_SHA
               for r in (trained, scored)):
            raise ValueError(f"{model} checkpoint or scaler initialization differs")
        checked(root / "train/endpoint2000.pth", trained["checkpoint_sha256"])
        bank_ids = {role: items[(model, role)] for role in AXES}
        if bank_ids != {item["role"]: item["sha256"] for item in scored["banks"]}:
            raise ValueError(f"{model} bank receipts differ from the global seal")
        scores[model] = banks(root, bank_ids, True)
        identities[model] = dict(checkpoint_sha256=trained["checkpoint_sha256"], score_sha256=bank_ids,
                                 training_seconds=trained["training_seconds"])
    return scores, dict(seed=seed, plan_sha256=plan_sha,
                        original_target_plan_sha256=TARGET_PLAN_SHA if seed == 2025 else None,
                        seal_sha256=sha(seal_path), arms=identities, actual_GPU_hours=seal["actual_GPU_hours"])


def evaluate(workspace, cache, natural, natural_metrics, replay, output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    scores, previous = baseline_inputs(cache, natural, natural_metrics)
    new_scores, replay_identity = replay_inputs(natural, replay)
    scores.update(new_scores)
    calibration = {model: q99(scores[model]["CAL"]) for model in METHODS}

    labels = truth(workspace)
    index, _, available = role_axis("QUERY")
    query_truth = labels[index]
    author_metric = load_author_metric(workspace / METRIC_PATH)
    rows, points, details, surfaces = [], {}, [], {}
    for model in METHODS:
        score = scores[model]["QUERY"]
        if model in previous:
            rank = {key: previous[model][key] for key in RANK_KEYS}
        else:
            rank, curves = continuous_metrics(query_truth, score, author_metric)
            for name, array in zip(("TPR", "FPR", "precision", "windows"), curves):
                surfaces[f"{model}_{name}"] = array
        threshold, exceedances = calibration[model]
        streams, event_detail, _, _ = evaluate_score(index, score, labels, threshold, available,
                                                     query_stop=AXES["QUERY"][1], ranking=rank)
        for stream in streams:
            stream.update(rank, method=model, CAL_exceedances=exceedances)
        strict = streams[0]
        if model in previous and any(strict[key] != previous[model][key]
                                     for key in ("FP", "TP", "normal_FPR")):
            raise ValueError(f"Event evaluation changed previously accepted {model} point counts")
        points[model] = strict
        rows.extend(streams)
        details.extend(dict(method=model, **event) for event in event_detail)

    compare_keys = (*RANK_KEYS, "FP", "normal_FPR", "Recall", "one_to_one_eventF1", "auxiliary_any_hit_events",
                    "event_point_coverage", "event_macro_coverage", "long_steady_macro_coverage",
                    "long_steady_point_coverage", "max_event_miss_points")
    by_method = {(row["method"], row["stream"]): row for row in rows}
    contrasts = []
    for model in ("U", "TT", "SS", "noU"):
        for stream in ("strict", "confirmed3"):
            left, right = by_method[("R", stream)], by_method[(model, stream)]
            contrasts.append(dict(contrast=f"R−{model}", stream=stream,
                                  **{key: left[key] - right[key] for key in compare_keys}))

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "VUS_SURFACES.npz", **surfaces)
    plot_curves(output, query_truth, {model: scores[model]["QUERY"] for model in METHODS}, points)
    write_json(output / "EVENTS.json", details)
    result = dict(query_axis=AXES["QUERY"], query_points=len(index), query_attack_points=int(query_truth.sum()),
                  cal_axis=AXES["CAL"], cal_points=CAL_POINTS.tolist(), CAL_rule="q99/higher/strictly greater",
                  source_checkpoint_sha256=INIT_CHECKPOINT_SHA, scaler_sha256=SCALER_SHA, label_sha256=LABEL_SHA,
                  vus_source_sha256=AUTHOR_METRIC_SHA, vus_window_size=100, vus_threshold_grid=250,
                  baseline_metrics_sha256=BASELINE_METRICS_SHA, replay=replay_identity, rows=rows,
                  paired_contrasts=contrasts, cpu_wall_seconds=time.perf_counter() - started)
    write_json(output / "METRICS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("workspace", "cache", "natural", "natural-metrics", "replay", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.workspace, args.cache, args.natural, args.natural_metrics, args.replay, args.out)
    print(f"Saved {len(result['rows'])} rows and both curves to {args.out}")


if __name__ == "__main__":
    main()
