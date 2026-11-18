"""Evaluate natural-mixture ordinary U and the cached LEFT author checkpoint."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .left_native_swat_metrics import (
    AUTHOR_METRIC_SHA, LABEL_SHA, SCORED, START, TEST_POINTS, WINDOW,
    continuous_metrics, load_author_metric, sha,
)
from .left_swat_readout_comparison import read_json, strict_points


AUTHOR_RUN = Path("results/left_swat_author_20261118_v1")
LABEL_PATH = Path("preparations/left_multievent_mixed_u_20261111_v1/SWAT_LABEL_METADATA.npz")
METRIC_PATH = Path("literature/novelty_review/left_source_20261031/ts_ad_evaluation/auc_vus/metrics.py")
NATIVE_BANK_SHA = "e8b6e6b8e7b978a6a3757576ef60ccdcea6bf86d7b4ecadbe734f4123dec22a1"
INIT_CHECKPOINT_SHA = "064747db9b1c24ce10d8678b00b2f72e504b4bcb4bec32b93399747234c5630d"
SCALER_SHA = "f46e9aa07179c7dd3a215846b502d4a1de11d7351369970ce45d0717433cccf2"
INIT_STEP = 52581
UPDATES = 2000
U_FIT = (495000, 585000)
CAL_ROLE = (585544, 607479)
QUERY_ROLE = (608023, 944919)
QUERY_AXIS = (608535, START + SCORED)


def grid_start(index):
    return START + ((index - START) // WINDOW) * WINDOW


CAL_AXIS = (int(grid_start(CAL_ROLE[0] + WINDOW - 1)), int(grid_start(CAL_ROLE[1])))
AXES = {"CAL": CAL_AXIS, "QUERY": QUERY_AXIS}
CAL_POINTS = np.linspace(CAL_AXIS[0], CAL_AXIS[1] - 1, 512, dtype=np.int64)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Unexpected asset identity for {path.name}")
    return path


def role_axis(role):
    index = np.arange(*AXES[role], dtype=np.int64)
    starts = grid_start(index)
    ends = starts + WINDOW - 1
    lo, hi = CAL_ROLE if role == "CAL" else QUERY_ROLE
    if starts[0] < lo or ends[-1] >= hi:
        raise ValueError(f"Full window crosses the {role} role boundary")
    return index, starts, ends


def q99(cal_scores):
    if cal_scores.shape != (CAL_AXIS[1] - CAL_AXIS[0],):
        raise ValueError("CAL needs the exact complete-window scoring axis")
    chosen = cal_scores[CAL_POINTS - CAL_AXIS[0]]
    tau = float(np.quantile(chosen, .99, method="higher"))
    return tau, int(np.count_nonzero(chosen > tau))


def truth(workspace):
    with np.load(checked(workspace / LABEL_PATH, LABEL_SHA), allow_pickle=False) as bank:
        labels = bank["label"]
    if labels.shape != (START + TEST_POINTS,) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected original binary SWaT labels")
    return labels


def cache_baseline(workspace, output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    native = workspace / AUTHOR_RUN
    receipt = read_json(native / "RAW_SCORE_RECEIPT.json")
    if (receipt["status"] != "COMPLETED" or receipt["bank_sha256"] != NATIVE_BANK_SHA
            or receipt["checkpoint_sha256"] != INIT_CHECKPOINT_SHA or receipt["final_python_step"] != INIT_STEP
            or receipt["seq_len"] != WINDOW or receipt["stride"] != WINDOW or receipt["labels_read"]):
        raise ValueError("Author baseline was not produced by the designated initialization")
    checked(native / "SCALER.npz", SCALER_SHA)
    with np.load(checked(native / "raw_scores.npz", NATIVE_BANK_SHA), allow_pickle=False) as bank:
        original_index = bank["absolute_time_index"]
        score, channel = bank["canonical_score"], bank["per_channel"]
        if (not np.array_equal(original_index, np.arange(START, START + SCORED))
                or not bank["valid"].all() or score.dtype != np.float32 or not np.isfinite(score).all()
                or channel.shape != (SCORED, 31) or not np.array_equal(score, channel.mean(axis=-1))):
            raise ValueError("Author score bank no longer agrees with the native point axis and channel mean")

    output.mkdir(parents=True, exist_ok=False)
    paths, sliced = {}, {}
    for role in AXES:
        index, starts, ends = role_axis(role)
        sliced[role] = score[index - START]
        path = output / f"{role}.npz"
        # The original archive's availability field predates the known metadata correction.
        np.savez_compressed(path, absolute_time_index=index, available_index=ends,
                            window_start=np.unique(starts), window_end_exclusive=np.unique(ends) + 1,
                            canonical_score=sliced[role])
        paths[role] = sha(path)
    tau, cal_exceedances = q99(sliced["CAL"])
    write_json(output / "CACHE_SEAL.json", dict(source_sha256=NATIVE_BANK_SHA, banks=paths,
               checkpoint_sha256=INIT_CHECKPOINT_SHA, scaler_sha256=SCALER_SHA, step=INIT_STEP,
               threshold=tau, calibration_exceedances=cal_exceedances, calibration_points=CAL_POINTS.tolist()))
    index = role_axis("QUERY")[0]
    labels = truth(workspace)[index]
    values, curves = continuous_metrics(labels, sliced["QUERY"], load_author_metric(workspace / METRIC_PATH))
    np.savez_compressed(output / "NO_U_VUS.npz", TPR=curves[0], FPR=curves[1],
                        precision=curves[2], windows=curves[3])
    write_json(output / "NO_U_METRICS.json", dict(**values, cache_seal_sha256=sha(output / "CACHE_SEAL.json"),
               label_sha256=LABEL_SHA, metric_sha256=AUTHOR_METRIC_SHA,
               cpu_wall_seconds=time.perf_counter() - started))
    return values


def read_scores(path, role, check_channel):
    index, starts, ends = role_axis(role)
    with np.load(path, allow_pickle=False) as bank:
        score = bank["canonical_score"]
        if (not np.array_equal(bank["absolute_time_index"], index)
                or not np.array_equal(bank["available_index"], ends)
                or not np.array_equal(bank["window_start"], np.unique(starts))
                or not np.array_equal(bank["window_end_exclusive"], np.unique(ends) + 1)
                or score.shape != index.shape or score.dtype != np.float32 or not np.isfinite(score).all()):
            raise ValueError(f"Invalid {role} time axis, availability or score")
        if check_channel:
            per_channel = bank["per_channel"]
            if (per_channel.shape != (len(index), 31) or per_channel.dtype != np.float32
                    or not np.array_equal(bank["valid"], np.ones(len(index), dtype=bool))
                    or not np.array_equal(score, per_channel.mean(axis=-1))):
                raise ValueError(f"Invalid {role} native channel reduction")
    return score


def validate_update(adapted):
    train_dir = adapted / "train"
    trained = read_json(train_dir / "RECEIPT.json")
    scored = read_json(adapted / "RECEIPT.json")
    final_step = INIT_STEP + UPDATES
    if (trained["status"] != "COMPLETED" or trained["updates"] != UPDATES
            or trained["initial_python_step"] != INIT_STEP or trained["final_python_step"] != final_step
            or trained["labels_read"] or trained["CAL_QUERY_read"] or not trained["endpoint_only"]
            or trained["seed"] != 2025 or tuple(trained["train_role"]) != U_FIT
            or trained["optimizer"] != "Adam lr1e-4"
            or scored["status"] != "COMPLETED" or scored["labels_read"] or scored["python_step"] != final_step
            or not scored["readout_state_step_RNG_immutable"] or not scored["source_checkpoint_unchanged"]
            or (scored["anchor"], scored["window"], scored["stride"]) != (START, WINDOW, WINDOW)
            or scored["checkpoint_sha256"] != trained["checkpoint_sha256"]):
        raise ValueError("Natural U must score its fixed ordinary-U endpoint without label or role access in training")
    for receipt in (trained, scored):
        if receipt["source_checkpoint_sha256"] != INIT_CHECKPOINT_SHA or receipt["scaler_sha256"] != SCALER_SHA:
            raise ValueError("Natural U and no-U must share the same native checkpoint and scaler")
    checked(train_dir / "endpoint2000.pth", trained["checkpoint_sha256"])
    plan_path = checked(adapted / "U_PLAN.npz", trained["plan_sha256"])
    with np.load(plan_path, allow_pickle=False) as archive:
        draws = archive["base_starts"]
    expected = np.random.default_rng(2025).integers(0, U_FIT[1] - U_FIT[0] - WINDOW + 1, size=(UPDATES, 128))
    if not np.array_equal(draws, expected):
        raise ValueError("Natural U training windows must match the frozen uniform sampling plan")
    return trained, scored


def compare(workspace, cache_root, adapted, out):
    started = time.perf_counter()
    if out.exists():
        raise FileExistsError(out)
    cache_seal_path = cache_root / "CACHE_SEAL.json"
    cache_seal = read_json(cache_seal_path)
    cached_rank = read_json(cache_root / "NO_U_METRICS.json")
    if (cached_rank["cache_seal_sha256"] != sha(cache_seal_path)
            or cache_seal["source_sha256"] != NATIVE_BANK_SHA
            or cache_seal["checkpoint_sha256"] != INIT_CHECKPOINT_SHA
            or cache_seal["scaler_sha256"] != SCALER_SHA or cache_seal["step"] != INIT_STEP
            or not np.array_equal(cache_seal["calibration_points"], CAL_POINTS)
            or cached_rank["label_sha256"] != LABEL_SHA or cached_rank["metric_sha256"] != AUTHOR_METRIC_SHA):
        raise ValueError("No-U metrics do not match the selected author cache and exact CAL points")

    trained, scored = validate_update(adapted)
    seal_path = adapted.parent / "RAW_BANK_SEAL.json"
    seal = read_json(seal_path)
    if seal["status"] != "COMPLETED" or not seal["no_labels_decoded"]:
        raise ValueError("Require the sealed label-free adapted score banks")
    bank_ids = {item["role"]: item["sha256"] for item in seal["banks"]}
    if (set(bank_ids) != set(AXES) or len(seal["banks"]) != len(AXES)
            or {item["role"]: item["sha256"] for item in scored["banks"]} != bank_ids):
        raise ValueError("Require exactly one adapted bank per role")
    baseline, updated = {}, {}
    for role in AXES:
        baseline[role] = read_scores(checked(cache_root / f"{role}.npz", cache_seal["banks"][role]), role, False)
        updated[role] = read_scores(checked(adapted / f"{role}.npz", bank_ids[role]), role, True)
    baseline_tau, baseline_exceedances = q99(baseline["CAL"])
    updated_tau, updated_exceedances = q99(updated["CAL"])
    if baseline_tau != cache_seal["threshold"] or baseline_exceedances != cache_seal["calibration_exceedances"]:
        raise ValueError("No-U calibration does not reproduce from its own score slice")

    labels = truth(workspace)
    y = labels[slice(*QUERY_AXIS)].astype(bool)
    rank, curves = continuous_metrics(y, updated["QUERY"], load_author_metric(workspace / METRIC_PATH))
    rows = [dict(method="noU_author", threshold=baseline_tau, CAL_exceedances=baseline_exceedances,
                 **{key: cached_rank[key] for key in rank}, **strict_points(y, baseline["QUERY"], baseline_tau)),
            dict(method="ordinary_U_natural", threshold=updated_tau, CAL_exceedances=updated_exceedances,
                 **rank, **strict_points(y, updated["QUERY"], updated_tau))]
    composition = {}
    for name, axis in (("U_FIT", U_FIT), ("CAL_role", CAL_ROLE), ("CAL_scored", CAL_AXIS)):
        composition[name] = dict(points=axis[1] - axis[0], labeled_attack_points=int(labels[slice(*axis)].sum()))
    composition["CAL_selected"] = dict(points=len(CAL_POINTS), labeled_attack_points=int(labels[CAL_POINTS].sum()))
    result = dict(query_axis=QUERY_AXIS, query_points=len(y), query_attack_points=int(y.sum()),
                  cal_role=CAL_ROLE, cal_axis=CAL_AXIS, cal_points=len(CAL_POINTS), cal_quantile=.99,
                  cal_quantile_method="higher", alarm_comparison=">", posthoc_label_composition=composition,
                  rows=rows, source_checkpoint_sha256=INIT_CHECKPOINT_SHA, source_scaler_sha256=SCALER_SHA,
                  adapted_checkpoint_sha256=trained["checkpoint_sha256"], training_plan_sha256=trained["plan_sha256"],
                  source_bank_sha256=NATIVE_BANK_SHA, adapted_seal_sha256=sha(seal_path),
                  adapted_score_bank_sha256=bank_ids, label_sha256=LABEL_SHA, vus_source_sha256=AUTHOR_METRIC_SHA,
                  vus_window_size=100, vus_threshold_grid=250,
                  cache_cpu_wall_seconds=cached_rank["cpu_wall_seconds"], comparison_cpu_wall_seconds=time.perf_counter()-started)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "U_VUS.npz", TPR=curves[0], FPR=curves[1], precision=curves[2], windows=curves[3])
    write_json(out / "METRICS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cache", "compare"))
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--adapted", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.action == "cache":
        result = cache_baseline(args.workspace, args.cache)
    else:
        if args.adapted is None or args.out is None:
            parser.error("compare requires --adapted and --out")
        result = compare(args.workspace, args.cache, args.adapted, args.out)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
