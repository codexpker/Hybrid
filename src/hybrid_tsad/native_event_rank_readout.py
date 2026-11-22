"""Frozen causal readout for the fixed SWaT/SMD-to-PSM source event fold."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import time

import numpy as np

from .native_v_conditioning_evaluation import (
    checked, evaluate_one, load_bank, read, sha,
)


ARMS = ("copy", "point_weighted", "event_balanced")
DOMAIN = "PSM"
BASE = Path("results/native_v_conditioning_20261119_v1")
ASSETS = Path("preparations/native_event_rank_readout_20261122_v1")
TASK_SHA = "f65df51946b65518a8e0ce8d40f58726a4a02e7a56d522ebbd07a9fc64df5b38"
TARGET_SHA = "6592518615d690ab153592b6dd0220c424441a0d4ee7bac87c3f89fe942893f5"
LABEL_SHA = "309449a288d65de43ff5e58ca2bc06b9913556e46024f5aa0b04153a40b30a69"
METRIC_SHA = "808a07460a84d31c87edce1d803e1df231111ec53ffa702ee32eb54a79897dfc"
AXIS = (23016, 87744)
ROLES = ("CAL", "QUERY")
COPY = "Reconstruction gradients only; source-event validation selection"
TRAINING = {
    "copy": COPY,
    "point_weighted": "Real public source event supervision, point-weighted",
    "event_balanced": "Real public source event supervision, event-balanced",
}
FIELDS = (
    "AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "Recall", "normal_FPR",
    "normal_alarm_fragments", "M", "Ntrue", "auxiliary_any_hit_events",
    "one_to_one_eventPrecision", "one_to_one_eventRecall", "one_to_one_eventF1",
    "event_macro_coverage", "event_point_coverage", "long_steady_macro_coverage",
    "long_steady_point_coverage", "max_event_miss_points", "delay_hit_only_median",
)
EXTERNAL_CSV_SHA = "c64a307a26663d240383a4581c05480da12e2551d047258f3f5510317c17c3e3"
EVENT_CSV_SHA = "cf228476d8d04f57d2ce0d0adac030de5ae6ecc9732a003b9c20ba43734b8de1"
WORKER_MODULE = "hybrid_tsad.native_event_rank_readout"


def asset(root, relative, digest):
    return checked(root, {"relative": str(relative), "sha256": digest})


def models(workspace, root):
    seal = read(root / "MODELS_FROZEN.json")
    if seal["status"] != "COMPLETED" or set(seal["arms"]) != set(ARMS):
        raise ValueError("All three source-only selections must be sealed")
    for arm in seal["arms"].values():
        for kind in ("selection", "checkpoint"):
            asset(root, arm[f"{kind}_relative"], arm[f"{kind}_sha256"])
    return seal


def verify_step0(workspace, root, arms):
    """Reuse the original native bank only after checking the complete state."""
    if not any(arm["selected_step"] == 0 for arm in arms.values()):
        return []
    import torch
    from .native_v_conditioning import load_native
    initial = load_native(workspace, torch.device("cpu")).state_dict()
    reused = []
    for name, arm in arms.items():
        if arm["selected_step"] == 0:
            saved = torch.load(root / arm["checkpoint_relative"], map_location="cpu",
                               weights_only=True)
            model = saved["model"]
            if (saved["update"] != 0 or set(model) != set(initial)
                    or any(not torch.equal(model[key], value) for key, value in initial.items())):
                raise ValueError("Step-zero checkpoint does not match original native state")
            reused.append(name)
    return reused


def shard(workspace, root, arm, role, part, parts):
    from .native_v_conditioning_runner import cache, record_for_role
    publication = models(workspace, root)
    case = root / "readout" / arm
    source_sha = sha(root / "MODELS_FROZEN.json")
    binding = read(case / "SOURCE_MODELS_BOUND.json")
    if binding["source_freeze_sha256"] != source_sha:
        raise ValueError("Source selection seal changed")
    task = read(asset(case / DOMAIN, "TASK.json", TASK_SHA))
    record = record_for_role(task, role)
    asset(workspace, record["relative"], TARGET_SHA)
    checkpoint = root / publication["arms"][arm]["checkpoint_relative"]
    cache(workspace, case, DOMAIN, role, part, parts, native_checkpoint=checkpoint,
          freeze_name="SOURCE_MODELS_BOUND.json", normal_fixture_reuse=False,
          metadata={"arm": arm, "source_freeze_sha256": source_sha})


def queue(workspace, root, gpus, stage):
    from .native_v_conditioning_queue import run_jobs
    from .native_v_conditioning_runner import dump, merge_cache
    publication = models(workspace, root)
    source_sha = sha(root / "MODELS_FROZEN.json")
    reused = verify_step0(workspace, root, publication["arms"])
    tasks = []
    for arm in ARMS:
        if arm in reused:
            continue
        case = root / "readout" / arm
        domain = case / DOMAIN
        domain.mkdir(parents=True, exist_ok=True)
        for name, digest in (
            ("TASK.json", TASK_SHA),
            ("SCALER.npz", "aabe7e305dd2a0bfb40142e27029c8f6e971dd5062fb44c1ce68c6bef32c3ae0"),
        ):
            source = asset(workspace / BASE / DOMAIN, name, digest)
            if not (domain / name).exists():
                shutil.copyfile(source, domain / name)
            asset(domain, name, digest)
        binding = {"status": "COMPLETED", "source_freeze_sha256": source_sha,
                   "checkpoint_sha256": publication["arms"][arm]["checkpoint_sha256"]}
        if not (case / "SOURCE_MODELS_BOUND.json").exists():
            dump(case / "SOURCE_MODELS_BOUND.json", binding)
        elif read(case / "SOURCE_MODELS_BOUND.json") != binding:
            raise ValueError("Existing readout belongs to a different checkpoint")
        for role, parts in (("QUERY", 4), ("CAL", 1)):
            for part in range(parts):
                path = case / "cache" / DOMAIN / role / f"part{part}"
                if (path / "RECEIPT.json").exists():
                    continue
                if path.exists():
                    raise FileExistsError(f"Inspect incomplete shard before retry: {path}")
                tasks.append((f"{arm}_{role}_{part}", [
                    "shard", "--arm", arm, "--role", role, "--part", str(part),
                    "--parts", str(parts)], 30, 2700))
    tasks.sort(key=lambda job: ("_CAL_" in job[0], job[0]))
    if tasks:
        run_jobs(workspace, root, stage, tasks, WORKER_MODULE, gpus)
    banks = {}
    for arm in ARMS:
        banks[arm] = {}
        expected_sha = publication["arms"][arm]["checkpoint_sha256"]
        for role in ROLES:
            if arm in reused:
                receipt_path = workspace / BASE / "banks" / DOMAIN / "A" / role / "RECEIPT.json"
                banks[arm][role] = dict(relative=str(receipt_path.relative_to(workspace)),
                                        sha256=sha(receipt_path))
                continue
            case = root / "readout" / arm
            folder = case / "cache" / DOMAIN / role
            if not (folder / "SEALED.json").exists():
                merge_cache(workspace, case, DOMAIN, role,
                            metadata={"checkpoint_sha256": expected_sha,
                                      "source_freeze_sha256": source_sha})
            receipt_path = folder / "SEALED.json"
            receipt = read(receipt_path)
            for part in receipt["parts"]:
                item = read(checked(workspace, part))
                if (item["checkpoint_sha256"] != expected_sha or item["labels_read"]
                        or item["source_freeze_sha256"] != source_sha):
                    raise ValueError("Readout shard does not match the frozen model")
            banks[arm][role] = dict(relative=str(receipt_path.relative_to(workspace)),
                                    sha256=sha(receipt_path))
    if sha(root / "MODELS_FROZEN.json") != source_sha:
        raise ValueError("Source seal changed during readout")
    dump(root / "READOUT_FROZEN.json", {
        "status": "COMPLETED", "source_freeze_sha256": source_sha, "banks": banks,
        "reused_verified_step0": reused,
    })


def evaluate_arm(args):
    arm, cal_path, query_path, indices, labels, metric = args
    cal, query = (load_bank(path) for path in (cal_path, query_path))
    for bank in (cal, query):
        index = bank["absolute_time_index"]
        np.testing.assert_array_equal(bank["window_start"], index - 99)
        np.testing.assert_array_equal(bank["window_end_exclusive"], index + 1)
    rows, events, _, _ = evaluate_one(DOMAIN, "A", cal, query, indices, labels, metric)
    for row in rows:
        row.update(method=arm, arm=arm, training=TRAINING[arm],
                   bank_CAL_sha256=sha(cal_path), bank_QUERY_sha256=sha(query_path))
    for event in events:
        event.update(method=arm)
    return rows, events


def cached_rows(workspace):
    path = asset(workspace / ASSETS, "timesfm_methods.csv", EXTERNAL_CSV_SHA)
    with path.open(newline="", encoding="utf-8") as handle:
        fields = FIELDS + ("long_event_count", "CAL_selected_attack_points", "threshold",
                           "CAL_exceedances", "FP", "TP", "FN", "TN", "Precision", "F1")
        integers = ("M", "Ntrue", "auxiliary_any_hit_events", "normal_alarm_fragments",
                    "max_event_miss_points", "long_event_count", "CAL_selected_attack_points",
                    "CAL_exceedances", "FP", "TP", "FN", "TN")
        names = {"TimesFM3_frozen", "TimesFM_RAW_D", "HistoryMean512",
                 "MLP_noU", "MLP_R", "CANDI_A", "TimeRadar_V_A"}
        for row in csv.DictReader(handle):
            if (row["domain"] == DOMAIN and row["method"] in names and row["mode"] == "CAUSAL_ENDPOINT"
                    and row["QUERY_axis"] == str(list(AXIS)) and row["scored_points"] == "64728"):
                values = {k: (int(row[k]) if k in integers else float(row[k])) if row[k] else None
                          for k in fields}
                yield dict(values, domain=DOMAIN, method=row["method"], stream=row["stream"],
                           source="archived causal metrics", archive_sha256=EXTERNAL_CSV_SHA,
                           training=row["input_permission"] or row["input_access"] or row["training"],
                           bank_CAL_sha256=row["bank_CAL_sha256"], bank_QUERY_sha256=row["bank_QUERY_sha256"],
                           scored_points=64728)


def left_endpoint_row(workspace):
    panel_path = asset(workspace / ASSETS, "common_panels.json",
                       "598460210c3a330bf9138dc5c3fbf2e38b7bd7ba2d02e2fdb22a404d9c296b3d")
    panel = next(p for p in read(panel_path)["panels"]
                 if p["dataset"] == DOMAIN and p["track"] == "causal-endpoint")
    if panel["axis"] != list(AXIS):
        raise ValueError("External common endpoint axis changed")
    row = next(r for r in panel["rows"] if r["method"] == "LEFT W192 endpoint")
    return dict(row, method="LEFT W192 endpoint", stream="ranking_only",
                training=row["permission"], source="archived causal ranking")


def evaluate(workspace, root, output):
    from .left_swat_replay_repeats import write_csv
    from .native_v_conditioning_runner import dump, role_index
    start = time.perf_counter()
    publication = models(workspace, root)
    raw = read(root / "READOUT_FROZEN.json")
    source_sha = sha(root / "MODELS_FROZEN.json")
    if raw["status"] != "COMPLETED" or raw["source_freeze_sha256"] != source_sha:
        raise ValueError("All target banks must match the frozen source selections")
    task_dir = workspace / BASE / DOMAIN
    task = read(asset(task_dir, "TASK.json", TASK_SHA))
    indices = role_index(workspace, task_dir, task, "CAL")
    paths = {}
    for arm, roles in raw["banks"].items():
        paths[arm] = {}
        for role in ROLES:
            receipt = read(checked(workspace, roles[role]))
            if (receipt["scaler_sha256"] != task["scaler_sha256"] or receipt["labels_read"]
                    or (arm not in raw["reused_verified_step0"] and
                        (receipt["checkpoint_sha256"] != publication["arms"][arm]["checkpoint_sha256"]
                         or receipt["source_freeze_sha256"] != source_sha))):
                raise ValueError("Bank checkpoint, source freeze or scaler mismatch")
            paths[arm][role] = checked(workspace, receipt)
    archived = list(cached_rows(workspace))
    rows = [dict(row, method="frozen_native_V", arm="frozen_native_V", seed=None)
            for row in archived if row["method"] == "TimeRadar_V_A"]
    frozen_rows = rows.copy()
    for role, index in (("CAL", indices), ("QUERY", np.arange(*AXIS))):
        receipt = read(workspace / BASE / "banks" / DOMAIN / "A" / role / "RECEIPT.json")
        bank = load_bank(checked(workspace, receipt))
        if len(rows) != 2 or any(r[f"bank_{role}_sha256"] != receipt["sha256"] for r in rows):
            raise ValueError("Archived baseline metrics do not match the frozen bank")
        np.testing.assert_array_equal(bank["absolute_time_index"], index)
        np.testing.assert_array_equal(bank["available_index"], index)
    path = asset(workspace / ASSETS, "timesfm_events.csv", EVENT_CSV_SHA)
    with path.open(newline="", encoding="utf-8") as handle:
        frozen_events = [dict(e, method="frozen_native_V") for e in csv.DictReader(handle)
                         if e["domain"] == DOMAIN and e["method"] == "TimeRadar_V_A"]
    events = frozen_events.copy()
    # Labels first become available after all score banks and their identity checks.
    labels = np.load(asset(workspace, "data/common_psm_20261028_v1/test_label.npy", LABEL_SHA),
                     allow_pickle=False).reshape(-1)
    metric = asset(workspace, "preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py",
                   METRIC_SHA)
    inputs = [(a, paths[a]["CAL"], paths[a]["QUERY"], indices, labels, metric)
              for a in ARMS if a not in raw["reused_verified_step0"]]
    if inputs:
        with ProcessPoolExecutor(max_workers=len(inputs)) as pool:
            outcomes = dict(zip((i[0] for i in inputs), pool.map(evaluate_arm, inputs)))
    else:
        outcomes = {}
    for arm in ARMS:
        if arm in outcomes:
            new_rows, new_events = outcomes[arm]
        else:
            new_rows = [dict(row, arm=arm, method=arm, training=TRAINING[arm], seed=2025,
                             source="step0 state equal to official frozen native V") for row in frozen_rows]
            new_events = [dict(event, method=arm) for event in frozen_events]
        for row in new_rows:
            row.update(selected_step=publication["arms"][arm]["selected_step"],
                       checkpoint_sha256=publication["arms"][arm]["checkpoint_sha256"])
        rows.extend(new_rows)
        events.extend(new_events)
    pairs = []
    for stream in ("strict", "confirmed3"):
        lookup = {row["arm"]: row for row in rows if row["stream"] == stream}
        for left, right in (
            ("event_balanced", "point_weighted"), ("event_balanced", "copy"),
            ("point_weighted", "copy"), ("copy", "frozen_native_V"),
            ("point_weighted", "frozen_native_V"), ("event_balanced", "frozen_native_V"),
        ):
            a, b = lookup[left], lookup[right]
            if any(a[k] != b[k] for k in ("Ntrue", "long_event_count", "CAL_selected_attack_points")):
                raise ValueError("Fixed evaluation eligibility has changed")
            pairs.append(dict(stream=stream, left=left, right=right, n=1, **{
                k: a[k] - b[k] if a[k] is not None and b[k] is not None else None for k in FIELDS}))
    rows.extend(row for row in archived if row["method"] != "TimeRadar_V_A")
    rows.append(left_endpoint_row(workspace))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "methods.csv", rows)
    write_csv(output / "paired.csv", pairs)
    dump(output / "EVENTS.json", events)
    dump(output / "METRICS.json", dict(rows=rows, paired=pairs, source_freeze_sha256=source_sha,
         readout_freeze_sha256=sha(root / "READOUT_FROZEN.json"),
         cpu_wall_seconds=time.perf_counter() - start, query_axis=list(AXIS), synthetic=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("shard", "queue", "evaluate"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--stage", default="target_readout")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--part", type=int)
    parser.add_argument("--parts", type=int)
    args = parser.parse_args()
    workspace, root = args.workspace.resolve(), args.output.resolve()
    if args.action == "shard":
        shard(workspace, root, args.arm, args.role, args.part, args.parts)
    elif args.action == "queue":
        queue(workspace, root, args.gpus, args.stage)
    else:
        evaluate(workspace, root, args.metrics_output.resolve())


if __name__ == "__main__":
    main()
