"""Fixed-initialization, three-seed LEFT source-replay comparison."""

import argparse
import csv
from pathlib import Path
import time

import numpy as np

from .left_native_swat_metrics import continuous_metrics, load_author_metric, sha
from .left_swat_natural_u_evaluation import AXES, METRIC_PATH, q99, read_json, role_axis, truth, write_json
from .left_swat_replay_evaluation import ARMS, RANK_KEYS, baseline_inputs, replay_inputs
from .timercd_swat_evaluation import evaluate_score


SEEDS = (2025, 2026, 2027)
OLD_METRICS_SHA = "353b96f7493bf7bb22ed4d44274116b54cfa7044238186bb3a82ebd173f0b885"
KEYS = (*RANK_KEYS, "FP", "normal_FPR", "Recall", "one_to_one_eventF1",
        "auxiliary_any_hit_events", "event_point_coverage", "event_macro_coverage",
        "long_steady_macro_coverage", "long_steady_point_coverage", "max_event_miss_points")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_statistics(rows, no_update):
    lookup = {(r["seed"], r["method"], r["stream"]): r for r in rows}
    frozen = {r["stream"]: r for r in no_update}
    pairs, summaries = [], []
    for control in ("TT", "SS", "noU"):
        for stream in ("strict", "confirmed3"):
            differences = []
            for seed in SEEDS:
                left = lookup[(seed, "R", stream)]
                right = frozen[stream] if control == "noU" else lookup[(seed, control, stream)]
                diff = dict(seed=seed, contrast=f"R−{control}", stream=stream,
                            **{key: left[key] - right[key] for key in KEYS})
                differences.append(diff)
                pairs.append(diff)
            for key in KEYS:
                values = np.array([r[key] for r in differences], dtype=np.float64)
                summaries.append(dict(contrast=f"R−{control}", stream=stream, metric=key, n=3,
                                      mean=float(values.mean()), sample_sd=float(values.std(ddof=1))))
    return pairs, summaries


def evaluate(workspace, replay_root, output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    old_path = workspace / "results/left_author_swat_source_replay_metrics_20261118_v1/METRICS.json"
    if sha(old_path) != OLD_METRICS_SHA:
        raise ValueError("Original seed2025 accepted results changed")
    old = read_json(old_path)
    cache = workspace / "results/left_author_swat_u_cache_20261118_v1"
    natural = workspace / "results/left_author_swat_u_transfer_20261118_v1"
    baseline_inputs(cache, natural, workspace / "results/left_author_swat_u_metrics_20261118_v1/METRICS.json")
    # Validate and load both new seeds before opening labels for this evaluation.
    scores, identities = {}, {}
    global_seal = read_json(replay_root / "GLOBAL_RAW_BANK_SEAL.json")
    if (global_seal["status"] != "COMPLETED" or not global_seal["no_labels_decoded"]
            or len(global_seal["banks"]) != 12 or len(global_seal["jobs"]) != 6
            or any(job["OS_exit_code"] != 0 for job in global_seal["jobs"])):
        raise ValueError("Require all six completed fixed jobs before this evaluation")
    for seed in SEEDS[1:]:
        scores[seed], identities[seed] = replay_inputs(natural, replay_root / f"seed{seed}", seed=seed)
    index, _, available = role_axis("QUERY")
    labels = truth(workspace)
    metric = load_author_metric(workspace / METRIC_PATH)
    rows = [dict(seed=2025, **row) for row in old["rows"] if row["method"] in ARMS]
    frozen = [dict(seed=None, reference_scope="single fixed initialization", **row)
              for row in old["rows"] if row["method"] == "noU"]
    ordinary = [dict(seed=2025, reference_scope="ordinary U128 seed2025 only", **row)
                for row in old["rows"] if row["method"] == "U"]
    details, curves = [], {}
    for seed in SEEDS[1:]:
        for arm in ARMS:
            banks = scores[seed][arm]
            tau, exceedances = q99(banks["CAL"])
            rank, surface = continuous_metrics(labels[index], banks["QUERY"], metric)
            streams, events, _, _ = evaluate_score(index, banks["QUERY"], labels, tau, available,
                                                  query_stop=AXES["QUERY"][1], ranking=rank)
            for row in streams:
                row.update(rank, seed=seed, method=arm, CAL_exceedances=exceedances)
            rows.extend(streams)
            details.extend(dict(seed=seed, method=arm, **event) for event in events)
            for name, array in zip(("TPR", "FPR", "precision", "windows"), surface):
                curves[f"{seed}_{arm}_{name}"] = array
    paired, summaries = paired_statistics(rows, frozen)
    arm_summary = []
    for arm in ARMS:
        for stream in ("strict", "confirmed3"):
            chosen = [r for r in rows if r["method"] == arm and r["stream"] == stream]
            for key in KEYS:
                values = np.asarray([r[key] for r in chosen], dtype=np.float64)
                arm_summary.append(dict(method=arm, stream=stream, metric=key, n=3,
                                        mean=float(values.mean()), sample_sd=float(values.std(ddof=1))))
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (("per_seed.csv", rows), ("fixed_references.csv", frozen + ordinary),
                        ("paired_differences.csv", paired), ("paired_summary.csv", summaries),
                        ("method_summary.csv", arm_summary)):
        write_csv(output / name, table)
    np.savez_compressed(output / "VUS_SURFACES.npz", **curves)
    write_json(output / "NEW_EVENTS.json", details)
    result = dict(rows=rows, references=frozen + ordinary, paired=paired, paired_summary=summaries,
                  method_summary=arm_summary, new_identities=identities, reused_2025_sha256=OLD_METRICS_SHA,
                  global_seal_sha256=sha(replay_root / "GLOBAL_RAW_BANK_SEAL.json"),
                  new_actual_GPU_hours=global_seal["actual_GPU_hours"],
                  query_axis=AXES["QUERY"], query_points=len(index), n_seeds=3, sample_sd_ddof=1,
                  note="Seeds quantify adaptation repeatability on one fixed task and initialization.",
                  cpu_wall_seconds=time.perf_counter() - started)
    write_json(output / "METRICS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("workspace", "replay-root", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.workspace, args.replay_root, args.out)
    print(f"Completed {len(result['rows'])} seed/arm/stream rows")


if __name__ == "__main__":
    main()
