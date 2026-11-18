"""Three-seed replay evaluation on a fixed PSM natural-mixture task."""

import argparse
from pathlib import Path
import time

import numpy as np

from .left_native_swat_metrics import continuous_metrics, load_author_metric, sha
from .left_swat_natural_u_evaluation import checked, read_json, write_json
from .left_swat_replay_repeats import ARMS, KEYS, SEEDS, paired_statistics, write_csv
from .timercd_swat_evaluation import evaluate_score


N, WINDOW, CHANNELS = 87841, 192, 25
ROLES = {"U_FIT": (0, 17568), "CAL": (18112, 21960), "QUERY": (22504, N)}
AXES = {"CAL": (18240, 21888), "QUERY": (22656, 87744)}
CAL_POINTS = np.linspace(*[AXES["CAL"][0], AXES["CAL"][1] - 1], 512, dtype=np.int64)
METRIC = Path("literature/novelty_review/left_source_20261031/ts_ad_evaluation/auc_vus/metrics.py")


def read_labels(path):
    return np.load(path, allow_pickle=False).reshape(-1)


def inputs(root, task):
    if (task["domain"] != "PSM" or task["seeds"] != list(SEEDS)
            or task["roles"] != {"SOURCE_FIT": [0, 105984], **{k: list(v) for k, v in ROLES.items()}}
            or task["score_axes"] != {k: list(v) for k, v in AXES.items()}
            or task["training"]["updates"] != 2000 or task["training"]["batch"] != 256
            or task["source"]["python_step"] != 16540):
        raise ValueError("PSM task differs from the fixed role and budget definition")
    seal = read_json(root / "GLOBAL_RAW_BANK_SEAL.json")
    if (seal["status"] != "COMPLETED" or not seal["no_labels_decoded"]
            or len(seal["banks"]) != 20 or len(seal["jobs"]) != 10
            or any(job["OS_exit_code"] != 0 for job in seal["jobs"])):
        raise ValueError("Require noU readout and all nine fixed adaptations before evaluation")
    scores, identities = {}, {}
    task_sha = sha(root / "TASK.json")
    for seed in SEEDS:
        seedroot = root / f"seed{seed}"
        frozen_plan = read_json(seedroot / "PLAN_FREEZE.json")
        plan_path = checked(seedroot / "SAMPLING_PLAN.npz", frozen_plan["plan_sha256"])
        children = np.random.SeedSequence(seed).spawn(3)
        expected = dict(T_starts=np.random.default_rng(seed).integers(0, 17568-192+1, size=(2000,128)),
                        S_starts=np.random.default_rng(children[0]).integers(0,105984-192+1,size=(2000,128)),
                        T_prime_starts=np.random.default_rng(children[1]).integers(0,17568-192+1,size=(2000,128)),
                        S_prime_starts=np.random.default_rng(children[2]).integers(0,105984-192+1,size=(2000,128)))
        with np.load(plan_path, allow_pickle=False) as plan:
            if set(plan.files) != set(expected) or any(not np.array_equal(plan[k], v) for k,v in expected.items()):
                raise ValueError("PSM paired source/target draws differ from the fixed seed streams")
        if frozen_plan["task_sha256"] != task_sha or frozen_plan["seed"] != seed:
            raise ValueError("PSM plan was frozen against a different task")
    frozen = {(item["seed"], item["arm"], item["role"]): item["sha256"] for item in seal["banks"]}
    if set(frozen) != {(seed, arm, role) for seed in SEEDS for arm in ARMS for role in AXES} | {
            (None, "noU", role) for role in AXES}:
        raise ValueError("PSM score seal has missing or duplicate models")
    for seed, arm in [(None, "noU")] + [(s, a) for s in SEEDS for a in ARMS]:
        folder = root / "noU" if seed is None else root / f"seed{seed}" / arm
        receipt = read_json(folder / "RECEIPT.json")
        if (receipt["status"] != "COMPLETED" or receipt["labels_read"]
                or receipt["scaler_sha256"] != task["source"]["scaler"]["sha256"]
                or receipt["source_checkpoint_sha256"] != task["source"]["checkpoint"]["sha256"]
                or receipt["python_step"] != 16540 + (2000 if seed is not None else 0)
                or not receipt["readout_state_step_RNG_immutable"] or receipt["task_sha256"] != task_sha
                or not receipt["source_checkpoint_unchanged"]):
            raise ValueError(f"Invalid PSM initialization or readout state for {seed}/{arm}")
        descriptors = {role: frozen[(seed, arm, role)] for role in AXES}
        if descriptors != {item["role"]: item["sha256"] for item in receipt["banks"]}:
            raise ValueError("PSM model bank receipt differs from the global seal")
        if seed is not None:
            trained = read_json(folder / "train/RECEIPT.json")
            permitted = {"SOURCE_FIT", "U_FIT"} if arm == "R" else {"U_FIT"} if arm == "TT" else {"SOURCE_FIT"}
            actual = trained["training_roles"]
            if (trained["updates"] != 2000 or trained["seed"] != seed or trained["labels_read"]
                    or trained["CAL_QUERY_read"] or trained["initial_python_step"] != 16540
                    or trained["final_python_step"] != 18540
                    or trained["batch"] != 256 or trained["native_forward_calls"] != 2000
                    or trained["optimizer"] != "fresh Adam lr1e-4" or not trained["endpoint_only"]
                    or trained["plan_sha256"] != sha(root / f"seed{seed}/SAMPLING_PLAN.npz")
                    or trained["checkpoint_sha256"] != receipt["checkpoint_sha256"]
                    or {r["role"] for r in actual} != permitted
                    or any(r["input"] != task["roles"][r["role"]] for r in actual)):
                raise ValueError("PSM training endpoint or role permissions differ")
            checked(folder / "train/endpoint2000.pth", trained["checkpoint_sha256"])
        scores[(seed, arm)] = read_model(folder, descriptors)
        identities[f"{seed}/{arm}"] = dict(checkpoint_sha256=receipt["checkpoint_sha256"], banks=descriptors)
    return scores, identities, seal


def bank(path, role):
    expected = np.arange(*AXES[role], dtype=np.int64)
    starts = expected // WINDOW * WINDOW
    with np.load(path, allow_pickle=False) as z:
        index, score, available = z["absolute_time_index"], z["canonical_score"], z["available_index"]
        per_channel = z["per_channel"]
        if (not np.array_equal(index, expected) or not np.array_equal(available, starts + WINDOW - 1)
                or score.dtype != np.float32 or score.shape != expected.shape or not np.isfinite(score).all()
                or per_channel.shape != (len(index), CHANNELS) or not np.array_equal(score, per_channel.mean(-1))
                or not np.array_equal(z["window_start"], np.unique(starts))
                or not np.array_equal(z["window_end_exclusive"], np.unique(starts)+WINDOW)
                or starts[0] < ROLES[role][0] or starts[-1] + WINDOW > ROLES[role][1]
                or not z["valid"].all()):
            raise ValueError(f"Invalid PSM {role} window/axis/channel-mean score")
    return score


def read_model(folder, descriptors):
    if set(descriptors) != set(AXES):
        raise ValueError("Require both label-free CAL and QUERY banks")
    return {role: bank(checked(folder / f"{role}.npz", descriptors[role]), role) for role in AXES}


def evaluate(workspace, root, label, label_sha, output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError(output)
    task = read_json(root / "TASK.json")
    if label_sha != task["labels"]["sha256"]:
        raise ValueError("Evaluation labels must match the task's frozen label asset")
    scores, identities, seal = inputs(root, task)
    calibration = {}
    for model, banks in scores.items():
        values = banks["CAL"][CAL_POINTS - AXES["CAL"][0]]
        tau = float(np.quantile(values, .99, method="higher"))
        calibration[model] = (tau, int((values > tau).sum()))
    labels = read_labels(checked(label, label_sha))
    if labels.shape != (N,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Original PSM test label sequence changed")
    index = np.arange(*AXES["QUERY"])
    available = index // WINDOW * WINDOW + WINDOW - 1
    metric = load_author_metric(workspace / METRIC)
    rows, frozen, details, curves = [], [], [], {}
    for (seed, arm), banks in scores.items():
        rank, surface = continuous_metrics(labels[index], banks["QUERY"], metric)
        tau, exceedances = calibration[(seed, arm)]
        streams, event_rows, _, events = evaluate_score(index, banks["QUERY"], labels, tau, available,
            query_start=AXES["QUERY"][0], query_stop=AXES["QUERY"][1], original_length=N,
            expected_events=None, ranking=rank)
        for row in streams:
            row.update(rank, method=arm, seed=seed, CAL_exceedances=exceedances)
        if arm == "noU":
            frozen.extend(streams)
        else:
            rows.extend(streams)
        details.extend(dict(seed=seed, method=arm, **event) for event in event_rows)
        for key, values in zip(("TPR", "FPR", "precision", "windows"), surface):
            curves[f"{seed}_{arm}_{key}"] = values
    paired, summary = paired_statistics(rows, frozen)
    arm_summary = []
    for arm in ARMS:
        for stream in ("strict", "confirmed3"):
            chosen = [r for r in rows if r["method"] == arm and r["stream"] == stream]
            for key in KEYS:
                values = np.asarray([r[key] for r in chosen], dtype=np.float64)
                arm_summary.append(dict(method=arm, stream=stream, metric=key, n=3,
                                        mean=float(values.mean()), sample_sd=float(values.std(ddof=1))))
    composition = {name: dict(points=hi-lo, labeled_attack_points=int(labels[lo:hi].sum()))
                   for name, (lo, hi) in ROLES.items()}
    composition["CAL_selected"] = dict(points=512, labeled_attack_points=int(labels[CAL_POINTS].sum()))
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (("per_seed.csv", rows), ("fixed_references.csv", frozen),
                        ("paired_differences.csv", paired), ("paired_summary.csv", summary),
                        ("method_summary.csv", arm_summary)):
        write_csv(output / name, table)
    np.savez_compressed(output / "VUS_SURFACES.npz", **curves)
    write_json(output / "EVENTS.json", details)
    result = dict(rows=rows, references=frozen, paired=paired, paired_summary=summary, method_summary=arm_summary,
                  query_axis=AXES["QUERY"], query_points=len(index), cal_points=CAL_POINTS.tolist(),
                  event_coverage=events, posthoc_composition=composition, identities=identities,
                  label_sha256=label_sha, task_sha256=sha(root / "TASK.json"),
                  seal_sha256=sha(root / "GLOBAL_RAW_BANK_SEAL.json"), actual_GPU_hours=seal["actual_GPU_hours"],
                  n_seeds=3, sample_sd_ddof=1, cpu_wall_seconds=time.perf_counter()-started)
    write_json(output / "METRICS.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("workspace", "root", "label", "out"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--label-sha", required=True)
    args = parser.parse_args()
    result = evaluate(args.workspace, args.root, args.label, args.label_sha, args.out)
    print(f"Completed {len(result['rows'])} PSM seed/arm/stream rows")


if __name__ == "__main__":
    main()
