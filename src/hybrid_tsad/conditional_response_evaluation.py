"""Fixed-axis evaluation for conditional-response energy and cached comparators.

This module neither loads models nor fits scores. Labels may be joined only
after the new A/B, noU/R banks are sealed. Old native axes remain untouched.
"""
import argparse
import concurrent.futures
from pathlib import Path
import json
import time

import numpy as np

from .left_native_swat_metrics import continuous_metrics, load_author_metric, sha
from .timercd_swat_evaluation import evaluate_score


AXES = {
    "SWAT": {"CAL": (586056, 607320), "QUERY": (608535, 944856)},
    "PSM": {"CAL": (18624, 21888), "QUERY": (23016, 87744)},
}


def select_bank(path, axis, *, expected_sha=None, causal=False,
                index_key="absolute_time_index", score_key="canonical_score",
                available_key="available_index", valid_key=None):
    """Select saved actual points, retaining each point's real availability."""
    path = Path(path)
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError("Score-bank identity changed")
    with np.load(path, allow_pickle=False) as bank:
        index = bank[index_key].copy()
        score = bank[score_key].copy()
        available = bank[available_key].copy() if available_key else index.copy()
        if valid_key:
            valid = bank[valid_key]
            if valid.shape != index.shape or valid.dtype != np.bool_:
                raise ValueError("Saved validity mask changed")
            index, score, available = index[valid], score[valid], available[valid]
    if (index.ndim != 1 or index.dtype.kind not in "iu" or not len(index)
            or np.any(np.diff(index) <= 0) or score.shape != index.shape
            or not np.isfinite(score).all() or available.shape != index.shape
            or available.dtype.kind not in "iu" or np.any(available < index)):
        raise ValueError("Require finite saved scores with ordered actual indices")
    wanted = np.arange(*axis, dtype=np.int64)
    pos = np.searchsorted(index, wanted)
    if np.any(pos >= len(index)) or not np.array_equal(index[pos], wanted):
        raise ValueError("Requested common axis is not fully present in cache")
    if causal and not np.array_equal(available[pos], wanted):
        raise ValueError("Current-observation energy must be available at t")
    return wanted, score[pos], available[pos]


def calibrated_threshold(index, score, domain):
    """Same predeclared512 legal mixed-CAL points, not historical tau units."""
    lo, hi = AXES[domain]["CAL"]
    points = np.linspace(lo, hi - 1, 512, dtype=np.int64)
    pos = np.searchsorted(index, points)
    if np.any(pos >= len(index)) or not np.array_equal(index[pos], points):
        raise ValueError("All exact legal CAL points must exist")
    values = np.asarray(score)[pos]
    if not np.isfinite(values).all():
        raise ValueError("CAL must be finite")
    return float(np.quantile(values, .99, method="higher")), points


def evaluate_saved(domain, name, cal_path, query_path, labels, metric_path,
                   *, causal, cal_sha=None, query_sha=None, permission=None,
                   bank_keys=None):
    """Evaluate one immutable model on the common new footprint, without fitting."""
    keys = bank_keys or {}
    ci, cs, _ = select_bank(cal_path, AXES[domain]["CAL"], expected_sha=cal_sha, causal=causal, **keys)
    index, score, available = select_bank(query_path, AXES[domain]["QUERY"], expected_sha=query_sha, causal=causal, **keys)
    tau, points = calibrated_threshold(ci, cs, domain)
    rank, surface = continuous_metrics(labels[index], score, load_author_metric(metric_path))
    rows, events, _, _ = evaluate_score(index, score, labels, tau, available,
        query_start=int(index[0]), query_stop=int(index[-1]) + 1,
        original_length=len(labels), expected_events=None, ranking=rank)
    for row in rows:
        row.update(rank, domain=domain, method=name, seed=2025,
            CAL_axis=list(AXES[domain]["CAL"]), QUERY_axis=list(AXES[domain]["QUERY"]),
            CAL_exceedances=int(np.sum(cs[np.searchsorted(ci, points)] > tau)),
            CAL_selected_attack_points=int(labels[points].sum()),
            calibration="mixed-CAL512/q99/higher/strict>; not normal-FPR certification",
            input_permission=permission, bank_CAL_sha256=sha(Path(cal_path)),
            bank_QUERY_sha256=sha(Path(query_path)),
            mode="CAUSAL_ENDPOINT" if causal else "OFFLINE_CONTEXT")
        if causal:
            hit = [e["offline_availability_offset"] for e in events
                   if e["stream"] == row["stream"] and e["main_eligible"]
                   and e["offline_availability_offset"] is not None]
            row["delay_hit_only_median"] = float(np.median(hit)) if hit else None
            row["delay_definition"] = "causal first available hit inside complete event; nonhits separately censored"
            row["offline_first_hit_offset_median"] = None
            row["offline_availability_offset_median"] = None
    for event in events:
        event.update(domain=domain, method=name)
        if causal:
            event["delay_points"] = event["offline_availability_offset"]
            event["offline_availability_offset"] = None
            event["offline_first_hit_offset"] = None
            event["timing_status"] = "CAUSAL_ENDPOINT"
    return rows, events, surface


def paired_rows(rows):
    """Single-seed descriptive deltas; do not fabricate uncertainty estimates."""
    lookup = {(r["domain"], r["method"], r["stream"]): r for r in rows}
    names = (("A_noU", "B_noU"), ("A_R", "B_R"),
             ("A_R", "A_noU"), ("B_R", "B_noU"))
    keys = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "normal_FPR", "Recall", "F1",
        "M", "auxiliary_any_hit_events", "one_to_one_eventF1", "event_point_coverage",
        "long_steady_macro_coverage", "long_steady_point_coverage", "max_event_miss_points")
    result = []
    for domain in AXES:
        for stream in ("strict", "confirmed3"):
            for left, right in names:
                a, b = lookup[domain, left, stream], lookup[domain, right, stream]
                result.append(dict(domain=domain, stream=stream, contrast=f"{left}−{right}",
                    n=1, sample_SD=None, **{k: None if a.get(k) is None or b.get(k) is None
                        else a[k] - b[k] for k in keys}))
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checked(path, expected):
    if sha(Path(path)) != expected:
        raise ValueError(f"Frozen asset changed: {path}")
    return Path(path)


def cache_manifest(workspace):
    """Use existing accepted cache identities, without reading labels/scores."""
    models = []
    def add(domain, method, root, hashes, causal, permission, bank_keys=None):
        banks = {role: dict(path=str((root / f"{role}.npz").relative_to(workspace)).replace("\\", "/"),
                           sha256=hashes[role]) for role in ("CAL", "QUERY")}
        for item in banks.values():
            checked(workspace / item["path"], item["sha256"])
        models.append(dict(domain=domain, method=method, banks=banks, causal=causal,
            permission=permission, **({"bank_keys": bank_keys} if bank_keys else {})))
    pinned = {
        "candi": ("results/candi_native_a_20261118_v1", "61670b71f5c81d3d60be1ec117d328fa18ffc7948c53ad8ab567e9070dbcfe65"),
        "mlp": ("results/mlp_replay_20261119_v1", "5d806cec1c4171f8527aee75a2720ddfb84b09bcc925aa40a9a58634b0f482fe"),
    }
    seals = {}
    for kind, (root, expected) in pinned.items():
        seals[kind] = read_json(checked(workspace / root / "GLOBAL_RAW_BANK_SEAL.json", expected))
    for domain in AXES:
        for original, name in (("noU", "MLP_noU"), ("CANDI", "CANDI_A")):
            hashes = {b["role"]: b["sha256"] for b in seals["candi"]["banks"]
                      if b["relative"].replace("\\", "/").startswith(f"{domain}/{original}/")}
            add(domain, name, workspace / pinned["candi"][0] / domain / original, hashes, True,
                "native10-point whole-window MSE at endpoint; SOURCE_FIT-only scaler; normal source MLP; CANDI U-only FPM/SANA then frozen" if original == "CANDI"
                else "native10-point MLP normal source initialization, FIT-only scaler, author normal SELECT; no target updates")
        for arm in ("TT", "R"):
            hashes = {b["role"]: b["sha256"] for b in seals["mlp"]["banks"]
                      if b["relative"].replace("\\", "/").startswith(f"{domain}/{arm}/")}
            add(domain, "MLP_" + arm, workspace / pinned["mlp"][0] / domain / arm, hashes, True,
                "native10-point MLP; FIT-only scaler; existing2000xB256 replay/TT; different context/objective/exposure from candidate")
    sw = read_json(checked(workspace / "results/left_author_swat_source_replay_metrics_20261118_v1/METRICS.json",
        "353b96f7493bf7bb22ed4d44274116b54cfa7044238186bb3a82ebd173f0b885"))
    ps = read_json(checked(workspace / "results/left_author_psm_source_replay_metrics_20261118_v1/METRICS.json",
        "0ec57ff0474f5bcd88c60118a061bbe65cc220a54115e30c8957e4ee6ed10716"))
    swcache = workspace / "results/left_author_swat_u_cache_20261118_v1"
    sc = read_json(checked(swcache / "CACHE_SEAL.json",
        "7d258293907f6db2994f9d9aa5969242c6cfea8ebbcf3bc75c7c4c52d4a9ece4"))
    for domain in AXES:
        for arm in ("noU", "TT", "R"):
            if domain == "SWAT":
                root = swcache if arm == "noU" else workspace / "results/left_author_swat_source_replay_20261118_v1" / arm
                hashes = sc["banks"] if arm == "noU" else sw["replay"]["arms"][arm]["score_sha256"]
            else:
                root = workspace / "results/left_author_psm_source_replay_20261118_v1" / ("noU" if arm == "noU" else f"seed2025/{arm}")
                hashes = ps["identities"]["None/noU" if arm == "noU" else "2025/" + arm]["banks"]
            add(domain, "LEFT_" + arm, root, hashes, False,
                "native192-point nonoverlapping whole-window retrospective scores; normal-prefix scaler includes SOURCE_SELECT; existing noU or2000xB256 native updates; not same-context/compute ablation")
    # PSM's accepted causal history bank includes the new CAL role. Its fixed
    # source residual scaling is retained; the new threshold uses mixed CAL.
    hp = workspace / "results/hybrid_v1_psm_Z1_fixed_evaluation_20261102_v1/history_mean512/calculation_inputs.npz"
    hsha = "f1c792db8c65d95783c6ac183f5bbca22de3621164384d70c0aac4e40c6a423a"
    checked(hp, hsha)
    models.append(dict(domain="PSM", method="HistoryMean512", causal=True,
        banks={r: dict(path=str(hp.relative_to(workspace)).replace("\\", "/"), sha256=hsha) for r in ("CAL", "QUERY")},
        bank_keys=dict(index_key="index", score_key="score", available_key="available_index", valid_key="valid"),
        permission="cached causal history512 with unchanged legacy source-calibrated residual scaling; new own mixed-CAL threshold; not FIT-only candidate normalization"))
    labels = {}
    for domain in AXES:
        task = read_json(workspace / pinned["candi"][0] / domain / "TASK.json")
        item = task["labels"]
        labels[domain] = dict(path=item["relative"], sha256=item["sha256"], key="label", length=item["shape"][0])
    timedir = workspace / "results/timeradar_cross_variable_20261116_v1/PSM/zero/score"
    native_receipt = read_json(timedir / "SCORE_RECEIPT.json")
    native_hash = next(b["sha256"] for b in native_receipt["banks"] if Path(b["path"]).name == "QUERY.npz")
    checked(timedir / "QUERY.npz", native_hash)
    history = workspace / "results/left_multievent_mixed_u_20261111_v1/fixed_scores/HistoryMean512.npz"
    history_hash = "2566b41f8550bcebcbd24919362274aecdc26688e017e8b878aaa6183e6ac7f5"
    checked(history, history_hash)
    rank_only = [dict(domain="PSM", method="TimeRadar_native", causal=False,
        path=str((timedir / "QUERY.npz").relative_to(workspace)).replace("\\", "/"), sha256=native_hash,
        bank_keys=dict(index_key="index", score_key="score", available_key="available_index"),
        reason="Original full native offline score; batch-shared order may cross the new CAL role, so no new legal threshold",
        permission="Official pretrained zero-update TimeRadar; original100-window/batched offline L1; old FIT/scoring permissions"),
        dict(domain="SWAT", method="HistoryMean512", causal=True,
            path=str(history.relative_to(workspace)).replace("\\", "/"), sha256=history_hash,
            bank_keys=dict(index_key="index", score_key="score", available_key=None),
            reason="Cached CAL is old normal-prefix role, not new mixed CAL; ranking only without new computation",
            permission="Accepted strictly-past512 historical predictor; original source residual scaling retained")]
    return dict(compatible_models=models, ranking_only_models=rank_only, labels=labels,
        author_metric="literature/novelty_review/left_source_20261031/ts_ad_evaluation/auc_vus/metrics.py",
        incompatible_models=[dict(domain="SWAT", method="TimeRadar", status="NA_NO_VERIFIED_COMPATIBLE_ROLE_BANK"),
            dict(domain="PSM", method="TimeRadar", status="RANK_ONLY_OFFLINE_HISTORICAL_CONTEXT", reason="Old batch-shared order used test contexts outside new CAL; cannot use it as a legal new CAL readout"),
            dict(domain="SWAT", method="HistoryMean512", status="RANK_ONLY_NO_CURRENT_MIXED_CAL", reason="Accepted CAL belongs to old normal-prefix role; never substitute its threshold for the new mixed role")],
        created_without_opening_labels_or_score_arrays=True,
        original_metric_axes_and_thresholds_preserved=True)


def new_descriptors(workspace, root):
    """Seal/recipe/selection gates do not read any label array."""
    seal = read_json(root / "GLOBAL_RAW_BANK_SEAL.json")
    expected = {(d, a, p, r) for d in AXES for a in ("A", "B")
                for p in ("source", "R") for r in ("CAL", "QUERY")}
    records = {(b["domain"], b["arm"], b["phase"], b["role"]): b for b in seal["banks"]}
    if (seal["status"] != "COMPLETED" or not seal["no_labels_decoded"]
            or len(seal["banks"]) != 16 or set(records) != expected):
        raise ValueError("All16 fixed A/B noU/R banks must seal before label access")
    desc, receipts = [], {}
    for domain in AXES:
        folder = root / domain
        task = read_json(folder / "TASK.json")
        if (task["seed"] != 2025 or task["K"] != 4 or task["copies"] != 10
                or task["window"] != 512 or task["response_window"] != 100
                or task["labels_accessed"] or task["training"]["updates"] != 2000
                or task["training"]["logical_batch"] != 16):
            raise ValueError("Fixed A/B recipe changed")
        checked(folder / "PLAN.npz", task["plan_sha256"])
        checked(folder / "SCALER.npz", task["scaler_sha256"])
        for role in AXES[domain]:
            if task["score_axes"][role] != list(AXES[domain][role]):
                raise ValueError("Common scored axes changed")
        initials = []
        for arm in ("A", "B"):
            for phase in ("source", "R"):
                stage = folder / arm / phase
                receipt = read_json(stage / "TRAIN_RECEIPT.json")
                if (receipt["status"] != "COMPLETED" or receipt["labels_read"]
                        or receipt["seed"] != 2025 or receipt["updates"] != 2000
                        or receipt["exposure"] != 32000 or not receipt["teacher_unchanged"]
                        or receipt["plan_sha256"] != task["plan_sha256"]
                        or receipt["scaler_sha256"] != task["scaler_sha256"]):
                    raise ValueError("Source/replay training recipe or role binding changed")
                allowed = ["SOURCE_FIT", "SOURCE_SELECT"] if phase == "source" else ["SOURCE_FIT", "U_FIT"]
                if receipt["roles_read"] != allowed:
                    raise ValueError("Unexpected training data role")
                checkpoint = checked(workspace / receipt["checkpoint_relative"], receipt["checkpoint_sha256"])
                trace = read_json(stage / "TRACE.json")
                if ([r["step"] for r in trace] != list(range(1, 2001))
                        or not all(np.isfinite(r["loss"]) and np.isfinite(r["gradient_norm"]) for r in trace)):
                    raise ValueError("Need2000 actual finite updates")
                if phase == "source":
                    selection = read_json(stage / "SELECT.json")
                    if ([r["step"] for r in selection] != list(range(0, 2001, 100))
                            or min(selection, key=lambda r: r["loss"])["step"] != receipt["selected_step"]):
                        raise ValueError("Normal SELECT including step0 must select earlier tied minimum")
                    initials.append(receipt["initial_state_hashes"])
                elif receipt["origin_checkpoint_sha256"] != receipts[domain, arm, "source"]["checkpoint_sha256"]:
                    raise ValueError("R did not start from its own selected source state")
                receipts[domain, arm, phase] = receipt
                banks = {}
                for role in ("CAL", "QUERY"):
                    b = records[domain, arm, phase, role]
                    banks[role] = dict(path=b["relative"], sha256=b["sha256"])
                    checked(workspace / b["relative"], b["sha256"])
                desc.append(dict(domain=domain, method=arm + ("_noU" if phase == "source" else "_R"),
                    causal=True, banks=banks, checkpoint_sha256=sha(checkpoint),
                    permission="FIT-only scaler; current-value conditional generation from strictly past512; own same-domain source init; logicalB16/2000updates"))
        if initials[0] != initials[1]:
            raise ValueError("A/B generators must have identical actual initial tensors")
    return desc, seal


def run_item(item):
    workspace, descriptor, labels, metric_path = item
    d = descriptor
    return evaluate_saved(d["domain"], d["method"], workspace / d["banks"]["CAL"]["path"],
        workspace / d["banks"]["QUERY"]["path"], labels, metric_path,
        causal=d["causal"], cal_sha=d["banks"]["CAL"]["sha256"],
        query_sha=d["banks"]["QUERY"]["sha256"], permission=d["permission"],
        bank_keys=d.get("bank_keys"))


def rank_only_item(item):
    workspace, d, labels, metric_path = item
    index, score, available = select_bank(workspace / d["path"], AXES[d["domain"]]["QUERY"],
        expected_sha=d["sha256"], causal=d["causal"], **d["bank_keys"])
    rank, surface = continuous_metrics(labels[index], score, load_author_metric(metric_path))
    return dict(domain=d["domain"], method=d["method"], stream="ranking_only", seed=None,
        **rank, mode="CAUSAL_ENDPOINT" if d["causal"] else "OFFLINE_CONTEXT",
        QUERY_axis=list(AXES[d["domain"]]["QUERY"]), scored_points=len(index),
        threshold=None, normal_FPR=None, Recall=None, one_to_one_eventF1=None,
        long_steady_macro_coverage=None, long_steady_point_coverage=None,
        operating_metrics_status="NA_NO_COMPATIBLE_MIXED_CAL_BANK", reason=d["reason"],
        input_permission=d["permission"], bank_QUERY_sha256=d["sha256"]), surface


def run(workspace, root, manifest_path, out, workers=2):
    started = time.monotonic()
    if out.exists():
        raise FileExistsError(out)
    new, seal = new_descriptors(workspace, root)
    manifest = read_json(manifest_path)
    external = manifest["compatible_models"]
    # Only now are labels opened, after all new training/scoring is complete.
    labels = {}
    for domain, spec in manifest["labels"].items():
        path = checked(workspace / spec["path"], spec["sha256"])
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as bank:
                values = bank[spec["key"]].copy()
        else:
            values = np.load(path, allow_pickle=False).reshape(-1)
        if values.shape != (spec["length"],) or not np.isin(values, [0, 1]).all():
            raise ValueError("Complete original point label axis required")
        labels[domain] = values
    metric = workspace / manifest["author_metric"]
    out.mkdir(parents=True, exist_ok=False)
    items = [(workspace, d, labels[d["domain"]], metric) for d in new + external]
    rows, events, surfaces = [], [], {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for item, result in zip(items, pool.map(run_item, items)):
            rr, ee, surface = result
            rows.extend(rr); events.extend(ee)
            key = item[1]["domain"] + "_" + item[1]["method"]
            for name, value in zip(("TPR", "FPR", "precision", "windows"), surface):
                surfaces[key + "_" + name] = value
        rank_items = [(workspace, d, labels[d["domain"]], metric) for d in manifest.get("ranking_only_models", [])]
        for item, result in zip(rank_items, pool.map(rank_only_item, rank_items)):
            row, surface = result
            rows.append(row)
            key = item[1]["domain"] + "_" + item[1]["method"]
            for name, value in zip(("TPR", "FPR", "precision", "windows"), surface):
                surfaces[key + "_" + name] = value
    from .left_swat_replay_repeats import write_csv
    paired = paired_rows(rows)
    write_csv(out / "methods.csv", rows)
    write_csv(out / "paired.csv", paired)
    np.savez_compressed(out / "VUS_SURFACES.npz", **surfaces)
    (out / "EVENTS.json").write_text(json.dumps(events, indent=2, allow_nan=False), encoding="utf-8")
    result = dict(status="COMPLETED", rows=rows, paired=paired,
        incompatible_models=manifest.get("incompatible_models", []),
        seal_sha256=sha(root / "GLOBAL_RAW_BANK_SEAL.json"),
        external_manifest_sha256=sha(manifest_path), actual_GPU_hours=seal["actual_GPU_hours"],
        cpu_seconds=time.monotonic() - started,
        limitations=["single fixed training seed2025; no confidence interval or cross-domain confirmation",
            "same-domain normal initialization and natural mixed adaptation, not new heterogeneous pretraining",
            "mixed CAL and all-attack PSM CAL cannot certify normal FPR",
            "offline LEFT context and its SELECT-inclusive scaler/exposure differ from causal candidate",
            "mature replay, native TimeRadar and TimesFM3 modules retain their sources"])
    (out / "METRICS.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("workspace", "root", "external-manifest", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = run(args.workspace, args.root, args.external_manifest, args.out, args.workers)
    print(f"Completed {len(result['rows'])} method/stream rows")
