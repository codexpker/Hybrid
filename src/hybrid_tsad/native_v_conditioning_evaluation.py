"""Read-only fixed-axis native-V evaluation; never import a neural model here.

New labels are joined only after six normal-source selections and all16 final
banks have sealed. Existing external comparisons retain their original axes,
permissions and calibrated operating points; they are not retrained.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import numpy as np


AXES = {"SWAT": (608535, 944856), "PSM": (23016, 87744)}
METHODS = {"A": "TimeRadar_V_A", "B": "MLP_Q_B",
           "C": "TimesFM_Q_C", "D": "TimesFM_RAW_D"}
EXTERNAL_METRICS = "results/tf_conditional_energy_metrics_20261119_v1/METRICS.json"
EXTERNAL_SHA = "79f56bdd4005149dd506c774a328c5bfe033f62a84f11d3a35d2c8f6a90b25fa"
EXTERNAL_MANIFEST = "results/tf_conditional_energy_20261119_v1/EXTERNAL_CACHE_MANIFEST_V2.json"
EXTERNAL_MANIFEST_SHA = "420ae8fd09a1847ab523c0b4dfa15b6e6d80f087e7cd4efef27b9d74803189d8"
EXTERNAL_EVENTS = "results/tf_conditional_energy_metrics_20261119_v1/EVENTS.json"
EXTERNAL_EVENTS_SHA = "c1160f7928d05a279f7a30eb95ae0669d2a046c91d6ca0bb0871ccbef0d4ed91"
EXTERNAL_NAMES = {"LEFT_noU", "LEFT_TT", "LEFT_R", "MLP_noU", "MLP_TT", "MLP_R",
                  "CANDI_A", "HistoryMean512", "TimeRadar_native"}
NATIVE_PARAMETERS = 1636043
BASE_PROTOCOL = "1be5e0001ae80b787302c2fc62f58860e101cd11"
REPLAY_ARMS = ("noU", "TT", "SS", "R")
ACCEPTED_NATIVE_METRICS = "results/native_v_conditioning_metrics_20261119_v1/METRICS.json"
ACCEPTED_NATIVE_METRICS_SHA = "a3c31f2325174a02fb76bd76486013064c71e83e5909ed90368f9b46e3ce9ccb"
ACCEPTED_NATIVE_EVENTS = "results/native_v_conditioning_metrics_20261119_v1/EVENTS.json"
ACCEPTED_NATIVE_EVENTS_SHA = "6d54b9be99457c42f1c873fe345a30fafbfc09991e7d9983e8df8bf6d96b8789"
ACCEPTED_HEAD_ROOT = "results/timeradar_native_v_replay_20261120_v1"
ACCEPTED_HEAD_SEAL_SHA = "6220f28d1b865b83629ad9426933d64c164d5968b26f4c180ae8c0f88686d51e"
ACCEPTED_HEAD_METRICS = "results/timeradar_native_v_replay_metrics_20261120_v1/METRICS.json"
ACCEPTED_HEAD_METRICS_SHA = "2dfd4d51a434b90308db120905c9e709907f1b0d3abb49f30182e75c2f6f67df"
ACCEPTED_HEAD_EVENTS = "results/timeradar_native_v_replay_metrics_20261120_v1/EVENTS.json"
ACCEPTED_HEAD_EVENTS_SHA = "1d56c8ef4f1b59bb4bf148c4c1beb893893e534f38c37966317277bb10129f51"
TARGET_MEAN_ARM = "R_target_mean"
ACCEPTED_TARGET_MEAN_METRICS = "results/timeradar_target_mean_metrics_20261120_v1/METRICS.json"
ACCEPTED_TARGET_MEAN_METRICS_SHA = "daa93070e72348ef8d304bcb184f85222b7c14228097fa1fef2db6d1d8346133"
ACCEPTED_TARGET_MEAN_SEAL = "results/timeradar_target_mean_20261120_v1/GLOBAL_RAW_BANK_SEAL.json"
ACCEPTED_TARGET_MEAN_SEAL_SHA = "a54a115c3f007c761e38d93b6d95b68211d527b88a90a6bef789de0b51889d31"
ACCEPTED_TARGET_MEAN_EVENTS_SHA = "cd4455af9cc42aa3b309dd9965c2f79d23577cef7cc631efc2dfe87a02c282a5"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def checked(workspace, record):
    path = (Path(workspace) / record["relative"]).resolve()
    if sha(path) != record["sha256"]:
        raise ValueError(f"Frozen asset identity changed: {record['relative']}")
    return path


def load_bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def quantile_score(v, q_normalized, scale):
    """Frozen formula, FP64 arithmetic, with negative scores and tied q allowed."""
    v = np.asarray(v, dtype=np.float64)
    q = np.asarray(q_normalized, dtype=np.float64)
    if (v.ndim != 1 or q.shape != (len(v), 3) or not np.isfinite(v).all()
            or not np.isfinite(q).all() or not np.isfinite(scale) or scale <= 0
            or np.any(q[:, 1:] < q[:, :-1])):
        raise ValueError("Require finite pointwise V and ordered q05/q50/q95 in normalized units")
    return (v / scale - q[:, 2]) / np.maximum(q[:, 2] - q[:, 0], 1e-3)


def raw_residual(prediction, truth):
    """Raw-variable D reference, per channel then per point, without V."""
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if (pred.ndim != 2 or pred.shape != truth.shape or not pred.shape[1]
            or not np.isfinite(pred).all() or not np.isfinite(truth).all()):
        raise ValueError("Require finite matching N,C normalized raw values")
    return np.square(pred - truth)


def exact_cal(index, score, declared_points):
    """Only the declared512 real observations; no interpolation or dense-CAL fiction."""
    index, score = np.asarray(index), np.asarray(score)
    wanted = np.asarray(declared_points, dtype=np.int64)
    if (wanted.shape != (512,) or not np.array_equal(index, wanted)
            or np.any(np.diff(index) <= 0) or score.shape != wanted.shape
            or not np.isfinite(score).all()):
        raise ValueError("Keep the exact512 fixed CAL points and order")
    return float(np.quantile(score.astype(np.float64), .99, method="higher"))


def step0_equivalence(cal_v, query_v, scale, actual_bias_f32):
    """Analytic reference from real V and the actual initialized head constants.

    No extra model inference and no label reading. Floating ties are recorded,
    not fixed by changing the criterion or selecting another expression.
    """
    cal_v = np.asarray(cal_v, dtype=np.float64)
    query_v = np.asarray(query_v, dtype=np.float64)
    bias = np.asarray(actual_bias_f32, dtype=np.float32).astype(np.float64)
    if bias.shape != (3,):
        raise ValueError("Actual three float32 initialization constants required")
    cal_score = quantile_score(cal_v, np.tile(bias, (len(cal_v), 1)), scale)
    score = quantile_score(query_v, np.tile(bias, (len(query_v), 1)), scale)
    order = np.argsort(query_v, kind="stable")
    dv, ds = np.diff(query_v[order]), np.diff(score[order])
    tau_v = float(np.quantile(cal_v, .99, method="higher"))
    tau_s = float(np.quantile(cal_score, .99, method="higher"))
    record = dict(reference="analytic zero-head initialization; not another trained model",
        input_bias_float32=bias.tolist(), s_V=float(scale), tau_native=tau_v, tau_step0=tau_s,
        order_reversals=int(np.sum(ds < 0)), newly_compressed_adjacent_values=int(np.sum((dv > 0) & (ds == 0))),
        native_tie_pairs_split=int(np.sum((dv == 0) & (ds != 0))),
        strict_alarm_disagreements=int(np.sum((query_v > tau_v) != (score > tau_s))),
        labels_used=False, inference=False)
    return record, score


def point_formula_gate(method, bank):
    """Recompute only already archived pointwise evidence, never a forward pass."""
    score = np.asarray(bank["canonical_score"], dtype=np.float64)
    if method in ("A", "B", "C") and np.any(np.asarray(bank["V"]) < 0):
        raise ValueError("Native reconstruction variance cannot be negative")
    if method == "A":
        expected = np.asarray(bank["V"], dtype=np.float64)
    elif method in ("B", "C"):
        s = np.asarray(bank["s_V"], dtype=np.float64)
        if s.size != 1:
            raise ValueError("One frozen SOURCE_FIT scale per model/domain")
        scale = float(s.reshape(-1)[0])
        q = np.asarray(bank["q_normalized"], dtype=np.float64)
        expected = quantile_score(bank["V"], q, scale)
        np.testing.assert_allclose(bank["q_physical"], scale * q, rtol=1e-12, atol=1e-12)
        width = q[:, 2] - q[:, 0]
        np.testing.assert_allclose(bank["width_normalized"], width, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(bank["width_physical"], scale * width, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(bank["denominator_normalized"], np.maximum(width, 1e-3), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(bank["denominator_physical"], scale * np.maximum(width, 1e-3), rtol=1e-12, atol=1e-12)
        if not np.array_equal(bank["floor_active"], width < 1e-3):
            raise ValueError("Floor flag must describe the declared normalized width")
    elif method == "D":
        residual = raw_residual(bank["yhat_normalized"], bank["actual_normalized"])
        np.testing.assert_allclose(bank["residual_per_channel"], residual, rtol=1e-12, atol=1e-12)
        expected = residual.mean(-1)
    else:
        raise ValueError(method)
    if score.ndim != 1 or expected.shape != score.shape or not np.isfinite(score).all():
        raise ValueError("Per-point finite scores required; scalar batch losses forbidden")
    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
    return dict(points=len(score), max_score_formula_abs_difference=float(np.max(np.abs(score - expected))))


def causal_timing(rows, events):
    """Translate the existing retrospective-named fields for causal endpoint rows."""
    for row in rows:
        offsets = [e["offline_availability_offset"] for e in events
                   if e["stream"] == row["stream"] and e["main_eligible"]
                   and e["offline_availability_offset"] is not None]
        row.update(mode="CAUSAL_ENDPOINT",
            delay_hit_only_median=float(np.median(offsets)) if offsets else None,
            delay_definition="causal first available alarm in complete event; nonhits separately censored",
            offline_first_hit_offset_median=None, offline_availability_offset_median=None)
    for event in events:
        event.update(delay_points=event["offline_availability_offset"],
            offline_first_hit_offset=None, offline_availability_offset=None, timing_status="CAUSAL_ENDPOINT")
    return rows, events


def paired_rows(rows):
    lookup = {(r["domain"], r["method"], r["stream"]): r for r in rows}
    pairs = (("C", "A"), ("C", "B"), ("C", "D"), ("B", "A"), ("D", "A"))
    keys = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "FP", "normal_FPR", "Recall", "F1",
            "M", "auxiliary_any_hit_events", "one_to_one_eventPrecision", "one_to_one_eventRecall",
            "one_to_one_eventF1", "event_point_coverage", "event_macro_coverage",
            "long_steady_macro_coverage", "long_steady_point_coverage", "max_event_miss_points")
    result = []
    for domain in AXES:
        for stream in ("strict", "confirmed3"):
            for left, right in pairs:
                a, b = lookup[domain, METHODS[left], stream], lookup[domain, METHODS[right], stream]
                result.append(dict(domain=domain, stream=stream, contrast=f"{left}−{right}",
                    n=1, sample_SD=None, **{k: None if a.get(k) is None or b.get(k) is None
                        else a[k] - b[k] for k in keys}))
    return result


def evaluate_one(domain, method, cal, query, cal_points, labels, metric_path):
    from .left_native_swat_metrics import continuous_metrics, load_author_metric
    from .timercd_swat_evaluation import evaluate_score
    ci, qi = cal["absolute_time_index"], query["absolute_time_index"]
    tau = exact_cal(ci, cal["canonical_score"], cal_points)
    if (not np.array_equal(qi, np.arange(*AXES[domain]))
            or not np.array_equal(query["available_index"], qi)
            or not np.array_equal(cal["available_index"], ci)):
        raise ValueError("Complete fixed causal QUERY axis required")
    cal_check = point_formula_gate(method, cal)
    query_check = point_formula_gate(method, query)
    ranking, surface = continuous_metrics(labels[qi], query["canonical_score"], load_author_metric(metric_path))
    rows, events, _, _ = evaluate_score(qi, query["canonical_score"], labels, tau, qi,
        query_start=AXES[domain][0], query_stop=AXES[domain][1], original_length=len(labels),
        expected_events=None, ranking=ranking)
    rows, events = causal_timing(rows, events)
    for row in rows:
        row.update(ranking, domain=domain, method=METHODS[method], arm=method, seed=2025,
            QUERY_axis=list(AXES[domain]), CAL_points=512,
            CAL_exceedances=int(np.sum(cal["canonical_score"] > tau)),
            CAL_selected_attack_points=int(labels[ci].sum()),
            calibration="same512 mixed CAL/q99/higher/strict>; not normal-FPR certification",
            training="official frozen TimeRadar" if method == "A" else "same-domain normal SOURCE_FIT; normal SOURCE_SELECT; no target U updates",
            timesfm_weights_used=False, original_protocol_commit=BASE_PROTOCOL)
    for event in events:
        event.update(domain=domain, method=METHODS[method])
    return rows, events, surface, dict(CAL=cal_check, QUERY=query_check)


def sealed_inputs(workspace, root):
    """Every identity gate precedes label loading; failed starts remain in jobs."""
    seal = read(root / "GLOBAL_RAW_BANK_SEAL.json")
    if (seal.get("status") != "COMPLETED"
            or seal.get("labels_read") is True or seal.get("no_labels_decoded") is False
            or not (seal.get("no_labels_decoded") is True or seal.get("labels_read") is False)):
        raise ValueError("Need completed label-free bank seal")
    expected = {(d, m, r) for d in AXES for m in METHODS for r in ("CAL", "QUERY")}
    records = {(b["domain"], b["method"], b["role"]): b for b in seal["banks"]}
    if len(seal["banks"]) != 16 or set(records) != expected:
        raise ValueError("Exactly16 distinct fixed A/B/C/D CAL/QUERY banks required")
    models = seal["trainings"]
    if (len(models) != 6 or {(m["domain"], m["method"]) for m in models}
            != {(d, m) for d in AXES for m in ("B", "C", "D")}):
        raise ValueError("Exactly six normal-source model selections must freeze")
    for model in models:
        p = (workspace / model["checkpoint_relative"]).resolve()
        if not p.is_relative_to(root) or sha(p) != model["checkpoint_sha256"]:
            raise ValueError("Selected source checkpoint changed or outside this new run")
        receipt_path = checked(workspace, dict(relative=model["train_receipt_relative"], sha256=model["train_receipt_sha256"]))
        receipt = read(receipt_path)
        d, m = model["domain"], model["method"]
        task = read(root / d / "TASK.json")
        fit_count = task["indexes"]["SOURCE_FIT"]["count"]
        if (receipt["status"] != "COMPLETED" or receipt["domain"] != d or receipt["method"] != m
                or receipt["seed"] != 2025 or receipt["labels_read"] or receipt["CAL_QUERY_opened"]
                or receipt["roles_read"] != ["SOURCE_FIT", "SOURCE_SELECT"]
                or receipt["updates"] != task["training"]["actual_max_updates"]
                or receipt["epochs"] != 20 or receipt["exposure"] != 20 * fit_count
                or receipt["checkpoint_sha256"] != model["checkpoint_sha256"]
                or receipt["scaler_sha256"] != task["scaler_sha256"]
                or receipt["source_permutations_sha256"] != task["source_permutation_sha256"]):
            raise ValueError("Normal-source training identity/permissions/budget changed")
        selection = read(receipt_path.parent / "SELECT.json")
        if [s["epoch"] for s in selection] != list(range(21)):
            raise ValueError("Need step0 and every fixed epoch SELECT")
        best = min(selection, key=lambda s: s["loss"])
        if (not all(np.isfinite(s["loss"]) and s["points"] == task["indexes"]["SOURCE_SELECT"]["count"] for s in selection)
                or best["epoch"] != receipt["selected_epoch"] or best["loss"] != receipt["selected_loss"]):
            raise ValueError("Checkpoint must be earliest minimum legal SOURCE_SELECT")
    freeze = read(root / "ALL_SIX_MODELS_FROZEN.json")
    if (freeze["status"] != "COMPLETED" or freeze["labels_read"]
            or not freeze["no_CAL_QUERY_opened"] or freeze["trainings"] != models):
        raise ValueError("All six selections must precede target readout")
    for d in AXES:
        c = read(root / "train" / d / "C" / "TRAIN_RECEIPT.json")
        dd = read(root / "train" / d / "D" / "TRAIN_RECEIPT.json")
        if not c["encoder_initial_state_hashes"] or c["encoder_initial_state_hashes"] != dd["encoder_initial_state_hashes"]:
            raise ValueError("C and D must share actual initial encoder tensors")
    paths = {}
    for key, descriptor in records.items():
        path = checked(workspace, descriptor)
        if not path.is_relative_to(root):
            raise ValueError("New scientific banks must not point to old candidates")
        receipt = read(checked(workspace, dict(relative=descriptor["receipt_relative"], sha256=descriptor["receipt_sha256"])))
        d, m, role = key
        if (receipt["status"] != "COMPLETED" or receipt["domain"] != d or receipt["method"] != m
                or receipt["role"] != role or receipt["labels_read"] or receipt["optimizer_updates"] != 0
                or not receipt["model_state_RNG_immutable"] or receipt["sha256"] != descriptor["sha256"]):
            raise ValueError("Readout identity/state/permissions changed")
        checked(workspace, dict(relative=receipt["source_cache_relative"], sha256=receipt["source_cache_sha256"]))
        paths[key] = path
    tasks = {d: read(root / d / "TASK.json") for d in AXES}
    for d, task in tasks.items():
        if (task["seed"] != 2025 or task["history"] != 512 or task["native_window"] != 100
                or task["labels_read"] or task["training"]["methods"] != ["B", "C", "D"]):
            raise ValueError("Frozen protocol task identity changed")
        for role in ("CAL", "QUERY"):
            descriptor = task["indexes"][role]
            index = np.load(checked(workspace, descriptor), allow_pickle=False)
            if role == "QUERY" and not np.array_equal(index, np.arange(*AXES[d])):
                raise ValueError("No QUERY cropping or entity selection permitted")
            if role == "CAL" and index.shape != (512,):
                raise ValueError("Keep all512 predeclared mixed-CAL observations")
    return seal, paths, tasks


def evaluate_item(item):
    d, m, cal_path, query_path, points, labels, metric = item
    result = evaluate_one(d, m, load_bank(cal_path), load_bank(query_path), points, labels, metric)
    rows, events, surface, checks = result
    for row in rows:
        row.update(bank_CAL_sha256=sha(cal_path), bank_QUERY_sha256=sha(query_path))
    return rows, events, surface, checks


def shared_evidence_gate(paths, tasks):
    """Verify all arms really read the same native responses and observations."""
    result = {}
    for d in AXES:
        for role in ("CAL", "QUERY"):
            a = load_bank(paths[d, "A", role])
            index = a["absolute_time_index"]
            for field, expected in (("available_index", index), ("history_start", index - 512),
                    ("history_end_exclusive", index), ("response_window_start", index - 99),
                    ("response_window_end_exclusive", index + 1)):
                np.testing.assert_array_equal(a[field], expected)
            for m in ("B", "C", "D"):
                bank = load_bank(paths[d, m, role])
                for field in ("absolute_time_index", "available_index", "V", "actual_raw", "actual_normalized",
                        "history_start", "history_end_exclusive", "response_window_start", "response_window_end_exclusive"):
                    np.testing.assert_array_equal(bank[field], a[field])
                point_formula_gate(m, bank)
            result[d + "_" + role] = dict(points=len(index), shared_arrays_exact=True,
                record_id=tasks[d]["indexes"][role]["record_id"], source_only_selection=True)
    return result


def model_costs(workspace, root, seal):
    """Actual owned occupancy and model-body timing are distinct quantities."""
    result = []
    for d in AXES:
        result.append(dict(domain=d, arm="A", trainable_parameters=0, frozen_TimeRadar_parameters=NATIVE_PARAMETERS,
            training_updates=0, shared_TimeRadar_cache=True))
        for m in ("B", "C", "D"):
            r = read(root / "train" / d / m / "TRAIN_RECEIPT.json")
            owned = next(j for j in seal["jobs"] if j["stage"] == "source_training" and j["identity"] == d + "_" + m)
            rr = {role: read(root / "banks" / d / m / role / "RECEIPT.json") for role in ("CAL", "QUERY")}
            result.append(dict(domain=d, arm=m, trainable_parameters=r["parameters"],
                frozen_TimeRadar_parameters=NATIVE_PARAMETERS if m in ("B", "C") else 0,
                training_updates=r["updates"], training_epochs=r["epochs"], source_window_exposures=r["exposure"],
                selected_epoch=r["selected_epoch"], selected_loss=r["selected_loss"],
                train_owned_GPU_seconds=owned["actual_GPU_seconds"], train_body_GPU_seconds=r["elapsed_GPU_seconds"],
                peak_train_allocated_GiB=r["peak_allocated_bytes"] / 2**30,
                CAL_readout_body_seconds=rr["CAL"]["elapsed_seconds"], QUERY_readout_body_seconds=rr["QUERY"]["elapsed_seconds"],
                QUERY_readout_points=rr["QUERY"]["count"],
                shared_TimeRadar_cache=m in ("B", "C"),
                checkpoint_sha256=r["checkpoint_sha256"]))
    return result


def event_contrasts(events):
    """All eligible events, not selected examples; retain nonhits as censored."""
    lookup = {(e["domain"], e["method"], e["stream"], e["event_id"]): e for e in events}
    result = []
    for e in events:
        if e["method"] != METHODS["C"] or not e["main_eligible"]:
            continue
        row = {k: e.get(k) for k in ("domain", "stream", "event_id", "start", "end_exclusive", "total_points")}
        for m in METHODS.values():
            v = lookup[e["domain"], m, e["stream"], e["event_id"]]
            for k in ("coverage", "steady_after512_coverage", "longest_miss_points", "steady_longest_miss_points", "delay_points"):
                row[m + "_" + k] = v.get(k)
        for m in ("A", "B", "D"):
            for k in ("coverage", "steady_after512_coverage", "longest_miss_points"):
                a, b = row[METHODS["C"] + "_" + k], row[METHODS[m] + "_" + k]
                row[f"C_minus_{m}_{k}"] = None if a is None or b is None else a - b
        result.append(row)
    return result


def replay_pairs(rows, contrasts=(("R", "TT"), ("R", "noU"), ("R", "SS"), ("TT", "noU"))):
    """Same-backbone/criterion learning contrasts; one fixed seed, not a CI."""
    lookup = {(r["domain"], r["method"], r["stream"]): r for r in rows}
    fields = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "TP", "FP", "normal_FPR", "Recall", "F1",
        "M", "auxiliary_any_hit_events", "one_to_one_eventPrecision", "one_to_one_eventRecall",
        "one_to_one_eventF1", "event_macro_coverage", "event_point_coverage", "long_steady_macro_coverage",
        "long_steady_point_coverage", "max_event_miss_points", "normal_alarm_fragments")
    result = []
    for d in AXES:
        for stream in ("strict", "confirmed3"):
            for a, b in contrasts:
                x, y = lookup[d, a, stream], lookup[d, b, stream]
                result.append(dict(domain=d, stream=stream, contrast=a + "−" + b, n=1, sample_SD=None,
                    **{k: None if x.get(k) is None or y.get(k) is None else x[k] - y[k] for k in fields}))
    return result


def replay_sealed_inputs(workspace, root):
    """Endpoint, role and score identities checked before label decoding."""
    seal = read(root / "GLOBAL_RAW_BANK_SEAL.json")
    if seal["status"] != "COMPLETED" or seal["labels_read"] or not seal["no_labels_decoded"]:
        raise ValueError("Complete label-free head-replay seal required")
    expected = {(d, a, r) for d in AXES for a in REPLAY_ARMS[1:] for r in ("CAL", "QUERY")}
    records = {(b["domain"], b["method"], b["role"]): b for b in seal["banks"]}
    if len(seal["banks"]) != 12 or set(records) != expected or len(seal["trainings"]) != 6:
        raise ValueError("Exactly six matched endpoint models and12 new banks required")
    trainings = {(r["domain"], r["arm"]): r for r in seal["trainings"]}
    if set(trainings) != {(d, a) for d in AXES for a in REPLAY_ARMS[1:]}:
        raise ValueError("Fixed domain/arm matrix changed")
    tasks = {d: read(root / d / "TASK.json") for d in AXES}
    for (d, a), r in trainings.items():
        if "train_receipt_relative" in r:
            checked(workspace, dict(relative=r["train_receipt_relative"], sha256=r["train_receipt_sha256"]))
        allowed = {"SS": ["SOURCE_FIT"], "TT": ["U_FIT"], "R": ["SOURCE_FIT", "U_FIT"]}[a]
        if (r["status"] != "COMPLETED" or r["seed"] != 2025 or r["updates"] != 2000 or r["logical_batch"] != 64
                or r["B1_forward_count"] != 128000 or r["exposures"] != 128000 or r["roles_read"] != allowed
                or r["labels_read"] or r["CAL_QUERY_opened"] or not r["all_modules_eval"]
                or not r["frozen_parameters_buffers_unchanged"] or r["scaler_sha256"] != tasks[d]["scaler_sha256"]
                or not r["trainable_names"] or not all(n.startswith("decoder.") for n in r["trainable_names"])):
            raise ValueError("Head-only training budget, role or state changed")
        p = checked(workspace, dict(relative=r["checkpoint_relative"], sha256=r["checkpoint_sha256"]))
        if not p.is_relative_to(root):
            raise ValueError("New endpoint must belong to this run")
        before, after = r["initial_state_hashes"], r["final_state_hashes"]
        if any(after[k] != v for k, v in before.items() if not k.startswith("decoder.")):
            raise ValueError("Frozen native parameters/buffers changed")
        if r["plan_sha256"] != sha(root / d / "PLAN.npz"):
            raise ValueError("Paired sampling plan changed")
    paths = {}
    for (d, a, role), record in records.items():
        p = checked(workspace, record)
        if not p.is_relative_to(root):
            raise ValueError("New score bank cannot be an old candidate")
        receipt = read(checked(workspace, dict(relative=record["receipt_relative"], sha256=record["receipt_sha256"])))
        if receipt["labels_read"] or receipt["scaler_sha256"] != tasks[d]["scaler_sha256"]:
            raise ValueError("Readout label/scaler identity changed")
        for part in receipt["parts"]:
            r = read(checked(workspace, part))
            if (r["actual_endpoint_sha256"] != trainings[d, a]["checkpoint_sha256"]
                    or not r["B1_only"] or not r["native_immutable"] or not r["CPU_CUDA_RNG_restored"]
                    or r["mask_seed"] != 84551 or r["labels_read"]):
                raise ValueError("Native readout must bind this endpoint and fixed B1 masks")
        paths[d, a, role] = p
    expected_old = {(d, "noU", r) for d in AXES for r in ("CAL", "QUERY")}
    if len(seal["noU_reused"]) != 4 or {(r["domain"], r["method"], r["role"]) for r in seal["noU_reused"]} != expected_old:
        raise ValueError("Exactly four immutable A/noU bank references required")
    for r in seal["noU_reused"]:
        paths[r["domain"], "noU", r["role"]] = checked(workspace, r)
    for d in AXES:
        for role in ("CAL", "QUERY"):
            with np.load(paths[d, "noU", role], allow_pickle=False) as old:
                index = old["absolute_time_index"]
                actual, normalized = old["actual_raw"], old["actual_normalized"]
                for arm in REPLAY_ARMS[1:]:
                    with np.load(paths[d, arm, role], allow_pickle=False) as new:
                        np.testing.assert_array_equal(new["absolute_time_index"], index)
                        np.testing.assert_array_equal(new["current_raw"], actual)
                        np.testing.assert_array_equal(new["current_normalized"], normalized)
                        np.testing.assert_array_equal(new["available_index"], index)
    return seal, paths, tasks


def run_head_replay(workspace, root, output, workers=2):
    """Native-V head probing reuses the fixed evaluator, without old B/C models."""
    started = time.perf_counter()
    workspace, root, output = map(lambda p: Path(p).resolve(), (workspace, root, output))
    if output.exists():
        raise FileExistsError(output)
    seal, paths, tasks = replay_sealed_inputs(workspace, root)
    accepted = read(checked(workspace, dict(relative=ACCEPTED_NATIVE_METRICS, sha256=ACCEPTED_NATIVE_METRICS_SHA)))
    manifest = read(checked(workspace, dict(relative=EXTERNAL_MANIFEST, sha256=EXTERNAL_MANIFEST_SHA)))
    labels, items = {}, []
    for d, task in tasks.items():
        spec = manifest["labels"][d]
        p = checked(workspace, dict(relative=spec["path"], sha256=spec["sha256"]))
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                labels[d] = z[spec["key"]].copy()
        else:
            labels[d] = np.load(p, allow_pickle=False).reshape(-1)
        cal_points = np.load(checked(workspace, task["indexes"]["CAL"]), allow_pickle=False)
        for arm in REPLAY_ARMS[1:]:
            items.append((d, arm, paths[d, arm, "CAL"], paths[d, arm, "QUERY"], cal_points,
                labels[d], workspace / manifest["author_metric"]))
    rows, events, surfaces, checks = [], [], {}, {}
    # Only new six states run VUS; noU and compatible external metrics are reused.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for item, (rr, ee, surface, gate) in zip(items, pool.map(evaluate_native_replay_item, items)):
            d, arm = item[:2]
            rows.extend(rr); events.extend(ee); checks[d + "_" + arm] = gate
            for k, v in zip(("TPR", "FPR", "precision", "windows"), surface):
                surfaces[d + "_" + arm + "_" + k] = v
    for old in accepted["rows"]:
        if old["method"] == METHODS["A"]:
            row = dict(old, method="noU", arm="noU", seed=None, reference_trial_seed=2025,
                updates=0, logical_batch=None, trainable_parameters=0, model_parameters=NATIVE_PARAMETERS,
                native_mask_seed=84551, reused_from_METRICS_sha256=ACCEPTED_NATIVE_METRICS_SHA)
            if sha(paths[row["domain"], "noU", "QUERY"]) != row["bank_QUERY_sha256"]:
                raise ValueError("Frozen A/noU QUERY identity changed")
            if sha(paths[row["domain"], "noU", "CAL"]) != row["bank_CAL_sha256"]:
                raise ValueError("Frozen A/noU CAL identity changed")
            rows.append(row)
    old_events = read(checked(workspace, dict(relative=ACCEPTED_NATIVE_EVENTS, sha256=ACCEPTED_NATIVE_EVENTS_SHA)))
    events.extend(dict(e, method="noU") for e in old_events if e["method"] == METHODS["A"])
    for row in rows:
        if row["method"] != "noU":
            training = next(r for r in seal["trainings"] if r["domain"] == row["domain"] and r["arm"] == row["method"])
            row.update(trainable_parameters=training["trainable_parameters"], model_parameters=NATIVE_PARAMETERS,
                native_mask_seed=84551, training_roles=training["roles_read"])
            baseline = next(r for r in rows if r["domain"] == row["domain"] and r["method"] == "noU" and r["stream"] == row["stream"])
            if any(row[k] != baseline[k] for k in ("Ntrue", "long_event_count", "CAL_selected_attack_points")):
                raise ValueError("Event qualification/CAL observation identity must match noU")
    pair = replay_pairs(rows)
    output.mkdir(parents=True)
    from .left_swat_replay_repeats import write_csv
    write_csv(output / "methods.csv", rows); write_csv(output / "learning_pairs.csv", pair)
    (output / "EVENTS.json").write_text(json.dumps(events, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "NEW_VUS_SURFACES.npz", **surfaces)
    result = dict(status="COMPLETED", mode="native_head_replay", rows=rows, paired=pair, formula_checks=checks,
        global_seal_sha256=sha(root / "GLOBAL_RAW_BANK_SEAL.json"), actual_GPU_hours=seal["actual_GPU_hours"],
        prior_native_metrics_sha256=ACCEPTED_NATIVE_METRICS_SHA, old_VUS_recomputed=False,
        cpu_wall_seconds=time.perf_counter() - started,
        interpretation="Natural mixed adaptation, not a contamination-causal experiment or Hybrid originality claim")
    (output / "METRICS.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def evaluate_native_replay_item(item):
    d, arm, cal_path, query_path, points, labels, metric = item
    cal, query = load_bank(cal_path), load_bank(query_path)
    for bank in (cal, query):
        index = bank["absolute_time_index"]
        np.testing.assert_array_equal(bank["window_start"], index - 99)
        np.testing.assert_array_equal(bank["window_end_exclusive"], index + 1)
    rows, events, surface, checks = evaluate_one(d, "A", cal, query, points, labels, metric)
    for row in rows:
        row.update(method=arm, arm=arm, training="native reconstruction head only; paired normal-source/natural-U slots",
            updates=2000, logical_batch=64, trainable_scope="decoder only; frozen backbone/order, all modules eval",
            endpoint_selection="fixed2000, no SELECT or QUERY selection", original_protocol_commit="f1f593aa69b61ca66bd2dcb648b90e0f66e4dd12",
            bank_CAL_sha256=sha(cal_path), bank_QUERY_sha256=sha(query_path))
    for event in events:
        event.update(method=arm)
    return rows, events, surface, checks


def target_mean_inputs(workspace, root):
    """Two new endpoints only; immutable old scores are references, never rerun."""
    seal = read(root / "GLOBAL_RAW_BANK_SEAL.json")
    if seal["status"] != "COMPLETED" or seal["labels_read"] or not seal["no_labels_decoded"]:
        raise ValueError("Complete label-free target-mean seal required")
    old = read(checked(workspace, dict(relative=ACCEPTED_HEAD_ROOT + "/GLOBAL_RAW_BANK_SEAL.json",
                                     sha256=ACCEPTED_HEAD_SEAL_SHA)))
    previous = read(checked(workspace, dict(relative=ACCEPTED_HEAD_METRICS, sha256=ACCEPTED_HEAD_METRICS_SHA)))
    records = {(b["domain"], b["method"], b["role"]): b for b in seal["banks"]}
    expected = {(d, TARGET_MEAN_ARM, role) for d in AXES for role in ("CAL", "QUERY")}
    if len(seal["banks"]) != 4 or set(records) != expected or len(seal["trainings"]) != 2:
        raise ValueError("Exactly two target-mean models and four new banks required")
    trainings = {(r["domain"], r["arm"]): r for r in seal["trainings"]}
    if set(trainings) != {(d, TARGET_MEAN_ARM) for d in AXES}:
        raise ValueError("No additional arm, seed or domain permitted")
    tasks, paths, old_paths = {}, {}, {}
    for d in AXES:
        tasks[d] = read(root / d / "TASK.json")
        new, prior = trainings[d, TARGET_MEAN_ARM], next(r for r in old["trainings"] if r["domain"] == d and r["arm"] == "R")
        if (new["status"] != "COMPLETED" or new["seed"] != 2025 or new["updates"] != 2000
                or new["logical_batch"] != 64 or new["B1_forward_count"] != 128000
                or new["exposures"] != 128000 or new["roles_read"] != ["SOURCE_FIT", "U_FIT"]
                or new["labels_read"] or new["CAL_QUERY_opened"] or not new["all_modules_eval"]
                or not new["frozen_parameters_buffers_unchanged"]
                or new["loss_profile"] != "target_mean_source_copy"):
            raise ValueError("New training must change only the target reconstruction objective")
        for k in ("plan_sha256", "scaler_sha256", "initial_state_hashes", "trainable_names", "trainable_parameters"):
            if new[k] != prior[k]:
                raise ValueError("Unmatched initialization/sampling/frozen scope: " + k)
        if sha(root / d / "PLAN.npz") != prior["plan_sha256"] or sha(root / d / "SCALER.npz") != prior["scaler_sha256"]:
            raise ValueError("Copy original PLAN/SCALER bytes without regenerating them")
        checked(workspace, dict(relative=new["train_receipt_relative"], sha256=new["train_receipt_sha256"]))
        if not checked(workspace, dict(relative=new["checkpoint_relative"], sha256=new["checkpoint_sha256"])).is_relative_to(root):
            raise ValueError("New endpoint must reside in its independent experiment")
        if any(new["final_state_hashes"][k] != v for k, v in new["initial_state_hashes"].items() if not k.startswith("decoder.")):
            raise ValueError("Frozen official tensors changed")
        for role in ("CAL", "QUERY"):
            new_index = np.load(checked(workspace, tasks[d]["indexes"][role]), allow_pickle=False)
            old_task = read(workspace / ACCEPTED_HEAD_ROOT / d / "TASK.json")
            np.testing.assert_array_equal(new_index, np.load(checked(workspace, old_task["indexes"][role]), allow_pickle=False))
            if role == "QUERY":
                np.testing.assert_array_equal(new_index, np.arange(*AXES[d]))
            elif len(new_index) != 512:
                raise ValueError("Keep original512 CAL points")
    for key, descriptor in records.items():
        d, arm, role = key
        p = checked(workspace, descriptor)
        if not p.is_relative_to(root):
            raise ValueError("New score cannot substitute an old model")
        receipt = read(checked(workspace, dict(relative=descriptor["receipt_relative"], sha256=descriptor["receipt_sha256"])))
        if receipt["labels_read"] or receipt["scaler_sha256"] != trainings[d, arm]["scaler_sha256"]:
            raise ValueError("Score data permissions changed")
        for part in receipt["parts"]:
            r = read(checked(workspace, part))
            if (r["actual_endpoint_sha256"] != trainings[d, arm]["checkpoint_sha256"] or not r["B1_only"]
                    or not r["native_immutable"] or not r["CPU_CUDA_RNG_restored"]
                    or r["mask_seed"] != 84551 or r["labels_read"]):
                raise ValueError("Readout identity/randomness changed")
        paths[key] = p
    for b in old["banks"] + old["noU_reused"]:
        old_paths[b["domain"], b["method"], b["role"]] = checked(workspace, b)
    for d in AXES:
        for role in ("CAL", "QUERY"):
            new, baseline = load_bank(paths[d, TARGET_MEAN_ARM, role]), load_bank(old_paths[d, "R", role])
            for field in ("absolute_time_index", "available_index", "window_start", "window_end_exclusive",
                          "feature_start", "feature_end_exclusive", "current_raw", "current_normalized", "actual_order"):
                np.testing.assert_array_equal(new[field], baseline[field])
    return seal, paths, old_paths, tasks, previous


def run_target_mean(workspace, root, output, workers=2):
    """Evaluate only two new V states; retain all four old learning comparisons."""
    start = time.perf_counter()
    workspace, root, output = (Path(p).resolve() for p in (workspace, root, output))
    if output.exists():
        raise FileExistsError(output)
    seal, paths, old_paths, tasks, previous = target_mean_inputs(workspace, root)
    manifest = read(checked(workspace, dict(relative=EXTERNAL_MANIFEST, sha256=EXTERNAL_MANIFEST_SHA)))
    items = []
    for d in AXES:
        label = manifest["labels"][d]
        p = checked(workspace, dict(relative=label["path"], sha256=label["sha256"]))
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                y = z[label["key"]].copy()
        else:
            y = np.load(p, allow_pickle=False).reshape(-1)
        points = np.load(checked(workspace, tasks[d]["indexes"]["CAL"]), allow_pickle=False)
        items.append((d, TARGET_MEAN_ARM, paths[d, TARGET_MEAN_ARM, "CAL"], paths[d, TARGET_MEAN_ARM, "QUERY"],
                      points, y, workspace / manifest["author_metric"]))
    rows, events, surfaces, checks = [], [], {}, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for item, (rr, ee, surface, check) in zip(items, pool.map(evaluate_native_replay_item, items)):
            d = item[0]
            receipt = next(t for t in seal["trainings"] if t["domain"] == d)
            for row in rr:
                row.update(training="R target mean reconstruction MSE; source original copy MSE", loss_profile="target_mean_source_copy",
                           trainable_parameters=receipt["trainable_parameters"], model_parameters=NATIVE_PARAMETERS,
                           native_mask_seed=84551, training_roles=receipt["roles_read"])
                baseline = next(r for r in previous["rows"] if r["domain"] == d and r["method"] == "R" and r["stream"] == row["stream"])
                if any(row[k] != baseline[k] for k in ("Ntrue", "long_event_count", "CAL_selected_attack_points")):
                    raise ValueError("Event qualification/CAL identity changed")
            rows.extend(rr); events.extend(ee); checks[d] = check
            for k, v in zip(("TPR", "FPR", "precision", "windows"), surface):
                surfaces[d + "_" + k] = v
    for r in previous["rows"]:
        for role in ("CAL", "QUERY"):
            if sha(old_paths[r["domain"], r["method"], role]) != r["bank_" + role + "_sha256"]:
                raise ValueError("Accepted baseline score identity changed")
        rows.append(dict(r, reused_from_head_METRICS_sha256=ACCEPTED_HEAD_METRICS_SHA))
    events.extend(read(checked(workspace, dict(relative=ACCEPTED_HEAD_EVENTS, sha256=ACCEPTED_HEAD_EVENTS_SHA))))
    external = read(checked(workspace, dict(relative=ACCEPTED_NATIVE_METRICS, sha256=ACCEPTED_NATIVE_METRICS_SHA)))
    for r in external["rows"]:
        if r["method"] in EXTERNAL_NAMES:
            if r["QUERY_axis"] != list(AXES[r["domain"]]):
                raise ValueError("No cross-axis external comparison")
            rows.append(dict(r, external=True, reused_from_native_METRICS_sha256=ACCEPTED_NATIVE_METRICS_SHA))
    pair = replay_pairs(rows, tuple((TARGET_MEAN_ARM, m) for m in ("R", "noU", "TT", "SS")))
    output.mkdir(parents=True)
    from .left_swat_replay_repeats import write_csv
    write_csv(output / "methods.csv", rows); write_csv(output / "single_factor_pairs.csv", pair)
    (output / "EVENTS.json").write_text(json.dumps(events, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "NEW_VUS_SURFACES.npz", **surfaces)
    banks = [dict(domain=d, method=m, role=r, relative=str(p.relative_to(workspace)), sha256=sha(p))
             for (d, m, r), p in {**old_paths, **paths}.items()]
    result = dict(status="COMPLETED", mode="native_target_mean", rows=rows, paired=pair, formula_checks=checks,
                  bank_manifest=banks, global_seal_sha256=sha(root / "GLOBAL_RAW_BANK_SEAL.json"),
                  previous_metrics_sha256=ACCEPTED_HEAD_METRICS_SHA, previous_seal_sha256=ACCEPTED_HEAD_SEAL_SHA,
                  external_metrics_sha256=ACCEPTED_NATIVE_METRICS_SHA, old_VUS_recomputed=False,
                  actual_GPU_hours=seal["actual_GPU_hours"], cpu_wall_seconds=time.perf_counter() - start,
                  interpretation="Fixed source/target reconstruction constraint experiment; no contamination-causal or originality claim established by training alone")
    (output / "METRICS.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def adaptation_pair_rows(rows, fields=None):
    """Three adaptation seeds share one frozen noU, not three pretrainings."""
    if fields is None:
        fields = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "FP", "TP", "normal_FPR", "Recall", "F1",
                  "M", "auxiliary_any_hit_events", "one_to_one_eventPrecision", "one_to_one_eventRecall",
                  "one_to_one_eventF1", "event_macro_coverage", "event_point_coverage",
                  "long_steady_macro_coverage", "long_steady_point_coverage", "max_event_miss_points",
                  "normal_alarm_fragments", "delay_hit_only_median")
    lookup = {(r["domain"], r["method"], r.get("seed"), r["stream"]): r for r in rows}
    pairs, summaries = [], []
    for d in AXES:
        for stream in ("strict", "confirmed3"):
            for control in ("R", "noU"):
                group = []
                for seed_value in (2025, 2026, 2027):
                    a = lookup[d, TARGET_MEAN_ARM, seed_value, stream]
                    b = lookup[d, control, seed_value if control == "R" else None, stream]
                    diff = dict(domain=d, stream=stream, seed=seed_value, contrast=TARGET_MEAN_ARM + "−" + control,
                                shared_pretrained_initial=True, noU_is_single_fixed_reference=control == "noU",
                                **{k: None if a.get(k) is None or b.get(k) is None else a[k] - b[k] for k in fields})
                    group.append(diff); pairs.append(diff)
                for key in fields:
                    values = [r[key] for r in group if r[key] is not None]
                    summaries.append(dict(domain=d, stream=stream, contrast=TARGET_MEAN_ARM + "−" + control,
                        metric=key, n=len(values), mean=float(np.mean(values)) if values else None,
                        sample_SD=float(np.std(values, ddof=1)) if len(values) > 1 else None, ddof=1,
                        scope="adaptation sampling repeat at fixed official pretrained state; not independent datasets"))
    return pairs, summaries


def repeat_inputs(workspace, root):
    """Seed/endpoint identity barriers precede access to the new QUERY labels."""
    seal = read(root / "GLOBAL_RAW_BANK_SEAL.json")
    if seal.get("status") != "COMPLETED" or seal.get("labels_read") is not False or seal.get("no_labels_decoded") is not True:
        raise ValueError("Need complete label-free repetition seal")
    expected_models = {(d, seed_value, arm) for d in AXES for seed_value in (2026, 2027) for arm in ("R", TARGET_MEAN_ARM)}
    expected_banks = {(*key, role) for key in expected_models for role in ("CAL", "QUERY")}
    models = {(r["domain"], r["seed"], r["arm"]): r for r in seal["trainings"]}
    records = {(b["domain"], b["seed"], b["method"], b["role"]): b for b in seal["banks"]}
    if len(seal["trainings"]) != 8 or set(models) != expected_models or len(seal["banks"]) != 16 or set(records) != expected_banks:
        raise ValueError("Exactly eight new seed/domain/arm models and sixteen banks required")
    prior_seal = read(checked(workspace, dict(relative=ACCEPTED_HEAD_ROOT + "/GLOBAL_RAW_BANK_SEAL.json", sha256=ACCEPTED_HEAD_SEAL_SHA)))
    checked(workspace, dict(relative=ACCEPTED_TARGET_MEAN_SEAL, sha256=ACCEPTED_TARGET_MEAN_SEAL_SHA))
    prior = read(checked(workspace, dict(relative=ACCEPTED_TARGET_MEAN_METRICS, sha256=ACCEPTED_TARGET_MEAN_METRICS_SHA)))
    tasks, paths = {}, {}
    for (d, seed_value, arm), rec in models.items():
        case = root / ("seed" + str(seed_value))
        task = read(case / d / "TASK.json"); tasks[d, seed_value] = task
        baseline = next(r for r in prior_seal["trainings"] if r["domain"] == d and r["arm"] == "R")
        if (rec["status"] != "COMPLETED" or rec["updates"] != 2000 or rec["logical_batch"] != 64
                or rec["B1_forward_count"] != 128000 or rec["exposures"] != 128000
                or rec["roles_read"] != ["SOURCE_FIT", "U_FIT"] or rec["labels_read"] or rec["CAL_QUERY_opened"]
                or not rec["all_modules_eval"] or not rec["frozen_parameters_buffers_unchanged"]
                or rec.get("loss_profile", "native") != ("native" if arm == "R" else "target_mean_source_copy")):
            raise ValueError("Fixed training budget, roles or loss profile changed")
        for k in ("scaler_sha256", "initial_state_hashes", "trainable_names", "trainable_parameters"):
            if rec[k] != baseline[k]:
                raise ValueError("Shared official initial/frozen scope changed: " + k)
        if (task["replay_training"]["seed"] != seed_value or sha(case / d / "PLAN.npz") != rec["plan_sha256"]
                or task["replay_training"]["sampling_plan_sha256"] != rec["plan_sha256"]
                or sha(case / d / "SCALER.npz") != baseline["scaler_sha256"]):
            raise ValueError("Actual seed, sampling plan or scaler changed")
        counterpart = models[d, seed_value, TARGET_MEAN_ARM if arm == "R" else "R"]
        if rec["plan_sha256"] != counterpart["plan_sha256"]:
            raise ValueError("Within-seed R and R-MeanT must share one plan")
        # Rebuild only the canonical RNG plan, never train or forward a model.
        with np.load(case / d / "PLAN.npz", allow_pickle=False) as p:
            base_task = read(workspace / ACCEPTED_HEAD_ROOT / d / "TASK.json")
            source = np.arange(base_task["roles"]["SOURCE_FIT"][0], base_task["roles"]["SOURCE_FIT"][1] - 99, 100)
            target = np.arange(base_task["roles"]["U_FIT"][0], base_task["roles"]["U_FIT"][1] - 99, 100)
            np.testing.assert_array_equal(p["source_starts"], source); np.testing.assert_array_equal(p["target_starts"], target)
            streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed_value).spawn(3)]
            expected = dict(T=np.random.default_rng(seed_value).integers(len(target), size=(2000, 32)),
                            S=streams[0].integers(len(source), size=(2000, 32)),
                            T_prime=streams[1].integers(len(target), size=(2000, 32)),
                            S_prime=streams[2].integers(len(source), size=(2000, 32)))
            for k, v in expected.items(): np.testing.assert_array_equal(p[k], v)
        before, after = rec["initial_state_hashes"], rec["final_state_hashes"]
        if any(after[k] != v for k, v in before.items() if not k.startswith("decoder.")):
            raise ValueError("Frozen official tensor changed")
        if not checked(workspace, dict(relative=rec["checkpoint_relative"], sha256=rec["checkpoint_sha256"])).is_relative_to(case):
            raise ValueError("New seed endpoint identity belongs to another run")
        checked(workspace, dict(relative=rec["train_receipt_relative"], sha256=rec["train_receipt_sha256"]))
    for (d, seed_value, arm, role), desc in records.items():
        p = checked(workspace, desc)
        if not p.is_relative_to(root / ("seed" + str(seed_value))):
            raise ValueError("Score belongs to another adaptation seed")
        receipt = read(checked(workspace, dict(relative=desc["receipt_relative"], sha256=desc["receipt_sha256"])))
        if receipt["labels_read"] or receipt["scaler_sha256"] != models[d, seed_value, arm]["scaler_sha256"]:
            raise ValueError("Readout label/scaler identity changed")
        for part in receipt["parts"]:
            r = read(checked(workspace, part))
            if (r["actual_endpoint_sha256"] != models[d, seed_value, arm]["checkpoint_sha256"]
                    or not r["B1_only"] or not r["native_immutable"] or not r["CPU_CUDA_RNG_restored"]
                    or r["mask_seed"] != 84551 or r["labels_read"]):
                raise ValueError("Scoring must keep fixed-mask B1 native V")
        old = next(b for b in prior["bank_manifest"] if b["domain"] == d and b["method"] == "R" and b["role"] == role)
        previous = load_bank(checked(workspace, old)); new = load_bank(p)
        for k in ("absolute_time_index", "available_index", "current_raw", "current_normalized", "actual_order",
                  "window_start", "window_end_exclusive", "feature_start", "feature_end_exclusive"):
            np.testing.assert_array_equal(previous[k], new[k])
        task = tasks[d, seed_value]
        np.testing.assert_array_equal(new["absolute_time_index"], np.load(checked(workspace, task["indexes"][role]), allow_pickle=False))
        paths[d, seed_value, arm, role] = p
    return seal, paths, tasks, prior


def evaluate_seeded_replay_item(item):
    domain, arm, seed_value, *rest = item
    rows, events, surface, check = evaluate_native_replay_item((domain, arm, *rest))
    for r in rows:
        r.update(seed=seed_value, shared_official_pretrained_initial=True,
                 loss_profile="native" if arm == "R" else "target_mean_source_copy")
    for e in events: e.update(seed=seed_value)
    return rows, events, surface, check


def run_target_mean_repeat(workspace, root, output, workers=2):
    start = time.perf_counter()
    workspace, root, output = (Path(p).resolve() for p in (workspace, root, output))
    if output.exists(): raise FileExistsError(output)
    seal, paths, tasks, prior = repeat_inputs(workspace, root)
    manifest = read(checked(workspace, dict(relative=EXTERNAL_MANIFEST, sha256=EXTERNAL_MANIFEST_SHA)))
    items = []
    for d in AXES:
        descriptor = manifest["labels"][d]
        p = checked(workspace, dict(relative=descriptor["path"], sha256=descriptor["sha256"]))
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z: labels = z[descriptor["key"]].copy()
        else: labels = np.load(p, allow_pickle=False).reshape(-1)
        for seed_value in (2026, 2027):
            task = tasks[d, seed_value]
            points = np.load(checked(workspace, task["indexes"]["CAL"]), allow_pickle=False)
            for arm in ("R", TARGET_MEAN_ARM):
                items.append((d, arm, seed_value, paths[d, seed_value, arm, "CAL"], paths[d, seed_value, arm, "QUERY"],
                              points, labels, workspace / manifest["author_metric"]))
    rows, events, surfaces, checks = [], [], {}, {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for item, (rr, ee, surface, check) in zip(items, pool.map(evaluate_seeded_replay_item, items)):
            d, arm, seed_value = item[:3]; training = next(r for r in seal["trainings"] if (r["domain"], r["arm"], r["seed"]) == (d, arm, seed_value))
            for r in rr:
                baseline = next(v for v in prior["rows"] if v["domain"] == d and v["method"] == arm and v["stream"] == r["stream"])
                if any(r[k] != baseline[k] for k in ("Ntrue", "long_event_count", "CAL_selected_attack_points")):
                    raise ValueError("Event qualification/CAL points changed")
                r.update(trainable_parameters=training["trainable_parameters"], model_parameters=NATIVE_PARAMETERS,
                         native_mask_seed=84551, training_roles=training["roles_read"])
            rows.extend(rr); events.extend(ee); checks[f"{d}_{seed_value}_{arm}"] = check
            for k, v in zip(("TPR", "FPR", "precision", "windows"), surface): surfaces[f"{d}_{seed_value}_{arm}_{k}"] = v
    rows.extend(dict(r, reused_from_METRICS_sha256=ACCEPTED_TARGET_MEAN_METRICS_SHA) for r in prior["rows"])
    event_path = workspace / "results/timeradar_target_mean_metrics_20261120_v1/EVENTS.json"
    if sha(event_path) != ACCEPTED_TARGET_MEAN_EVENTS_SHA:
        raise ValueError("Accepted seed2025 event identity changed")
    events.extend(dict(e, seed=None if e["method"] == "noU" else 2025) for e in read(event_path))
    pairs, summary = adaptation_pair_rows(rows)
    bank_manifest = []
    for b in prior["bank_manifest"]:
        p = checked(workspace, b)
        bank_manifest.append(dict(b, seed=None if b["method"] == "noU" else 2025))
    bank_manifest.extend(dict(domain=d, method=a, seed=s, role=r, relative=str(p.relative_to(workspace)), sha256=sha(p))
                         for (d, s, a, r), p in paths.items())
    from .left_swat_replay_repeats import write_csv
    output.mkdir(parents=True)
    write_csv(output / "methods.csv", rows); write_csv(output / "paired_seeds.csv", pairs); write_csv(output / "paired_summary.csv", summary)
    np.savez_compressed(output / "NEW_VUS_SURFACES.npz", **surfaces)
    (output / "EVENTS.json").write_text(json.dumps(events, indent=2, allow_nan=False), encoding="utf-8")
    result = dict(status="COMPLETED", mode="native_target_mean_repeat", rows=rows, paired=pairs, paired_summary=summary,
                  bank_manifest=bank_manifest, formula_checks=checks, global_seal_sha256=sha(root / "GLOBAL_RAW_BANK_SEAL.json"),
                  seed2025_METRICS_sha256=ACCEPTED_TARGET_MEAN_METRICS_SHA, old_VUS_recomputed=False,
                  actual_GPU_hours=seal["actual_GPU_hours"], cpu_wall_seconds=time.perf_counter() - start,
                  interpretation="Three adaptation seeds at one shared official pretrained state; two exposed tasks, not independent datasets or pretrains")
    (output / "METRICS.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


SMD_FIXED_ENTITIES = ("machine-3-2", "machine-3-3")
SMD_FIXED_QUERY = (8957, 23703)


def evaluate_smd_arrays(entity, arm, cal, query, points, labels, metric_path):
    """Same native V, CAL and event definitions; train CAL has no test labels.

    Entity identity never becomes a claim of a separate data domain. The old
    H512-qualified QUERY axis remains fixed although native V needs100 points.
    """
    from .left_native_swat_metrics import continuous_metrics, load_author_metric
    from .timercd_swat_evaluation import evaluate_score
    if entity not in SMD_FIXED_ENTITIES or arm not in ("noU", "R", TARGET_MEAN_ARM):
        raise ValueError("Only the two predeclared entities and three methods")
    index = query["absolute_time_index"]
    if not np.array_equal(index, np.arange(*SMD_FIXED_QUERY)):
        raise ValueError("Keep the previously fixed full QUERY axis")
    for bank in (cal, query):
        t = bank["absolute_time_index"]
        for key, value in (("available_index", t), ("window_start", t-99), ("window_end_exclusive", t+1)):
            np.testing.assert_array_equal(bank[key], value)
        point_formula_gate("A", bank)
    tau = exact_cal(cal["absolute_time_index"], cal["canonical_score"], points)
    if np.asarray(labels).shape != (SMD_FIXED_QUERY[1],):
        raise ValueError("Original test label record, not train CAL or window labels")
    ranking, surface = continuous_metrics(labels[index], query["canonical_score"], load_author_metric(metric_path))
    rows, events, _, _ = evaluate_score(index, query["canonical_score"], labels, tau, index,
        query_start=SMD_FIXED_QUERY[0], query_stop=SMD_FIXED_QUERY[1], original_length=len(labels),
        expected_events=None, ranking=ranking)
    rows, events = causal_timing(rows, events)
    for row in rows:
        row.update(ranking, domain="SMD", entity=entity, method=arm, arm=arm,
            seed=None if arm == "noU" else 2025, reference_trial_seed=2025,
            QUERY_axis=list(SMD_FIXED_QUERY), CAL_points=512,
            CAL_exceedances=int(np.sum(cal["canonical_score"] > tau)),
            CAL_selected_attack_points=None,
            CAL_label_NA_reason="CAL belongs to published train record; no independent point labels; never index test labels by train times",
            calibration="same512 source-train CAL points; own q99/higher/strict>; nominal normal not independent healthy certification",
            training="official frozen TimeRadar" if arm == "noU" else "paired source-FIT/natural mixed target-U; existing decoder only",
            loss_profile="none" if arm == "noU" else "native" if arm == "R" else "target_mean_source_copy",
            updates=0 if arm == "noU" else 2000, logical_batch=None if arm == "noU" else 64,
            native_mask_seed=84551, trainable_parameters=0 if arm == "noU" else 793610,
            model_parameters=NATIVE_PARAMETERS, historically_exposed=True, blind=False,
            endpoint_selection="official fixed weights" if arm == "noU" else "fixed2000; no SELECT/QUERY checkpoint selection")
    for event in events:
        event.update(domain="SMD", entity=entity, method=arm, seed=None if arm == "noU" else 2025)
    return rows, events, surface


def smd_learning_pairs(rows):
    """Fixed within-entity contrasts; n1/SDNA, never pool the two machines."""
    fields = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "FP", "TP", "normal_FPR", "Recall", "F1",
              "M", "auxiliary_any_hit_events", "one_to_one_eventF1", "normal_alarm_fragments",
              "event_macro_coverage", "event_point_coverage", "long_steady_macro_coverage",
              "long_steady_point_coverage", "max_event_miss_points", "delay_hit_only_median")
    lookup = {(r["entity"], r["method"], r["stream"]): r for r in rows}
    pairs = []
    for entity in SMD_FIXED_ENTITIES:
        for stream in ("strict", "confirmed3"):
            for control in ("R", "noU"):
                a, b = lookup[entity, TARGET_MEAN_ARM, stream], lookup[entity, control, stream]
                pairs.append(dict(domain="SMD", entity=entity, seed=2025, stream=stream,
                    contrast=TARGET_MEAN_ARM+"−"+control, n=1, sample_SD=None,
                    **{k: None if a.get(k) is None or b.get(k) is None else a[k]-b[k] for k in fields}))
    return pairs


def smd_inputs(workspace, root):
    """Twelve bank/four endpoint barrier before opening any new test labels."""
    seal = read(root / "GLOBAL_RAW_BANK_SEAL.json")
    if seal.get("status") != "COMPLETED" or seal.get("labels_read") is not False or seal.get("no_labels_decoded") is not True:
        raise ValueError("Complete label-free SMD seal required")
    expected = {(e, a, r) for e in SMD_FIXED_ENTITIES for a in ("noU", "R", TARGET_MEAN_ARM) for r in ("CAL", "QUERY")}
    records = {(b["domain"], b["method"], b["role"]): b for b in seal["banks"]}
    models = {(r["domain"], r["arm"]): r for r in seal["trainings"]}
    if (len(seal["banks"]) != 12 or set(records) != expected or len(seal["trainings"]) != 4 or len(models) != 4
            or set(models) != {(e, a) for e in SMD_FIXED_ENTITIES for a in ("R", TARGET_MEAN_ARM)}):
        raise ValueError("Exactly two entities, three scores and four adaptations")
    tasks, paths = {}, {}
    for entity in SMD_FIXED_ENTITIES:
        task = read(root/entity/"TASK.json"); tasks[entity] = task
        if task["channels"] != 38 or task["seed"] != 2025 or task["native"]["norm"] != 0:
            raise ValueError("Keep seed2025/native norm0 and all38 channels")
        if (task["native"]["checkpoint_sha256"] != "528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c"
                or task["native"]["source_sha256"] != "881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317"):
            raise ValueError("Fixed official source/checkpoint required")
        cal = np.load(checked(workspace, task["indexes"]["CAL"]), allow_pickle=False)
        query = np.load(checked(workspace, task["indexes"]["QUERY"]), allow_pickle=False)
        if cal.shape != (512,) or np.any(np.diff(cal) <= 0) or not np.array_equal(query, np.arange(*SMD_FIXED_QUERY)):
            raise ValueError("Fixed legal CAL and complete common QUERY")
        if task["indexes"]["CAL"]["record_id"] == task["indexes"]["QUERY"]["record_id"]:
            raise ValueError("This task uses train CAL and separate test QUERY")
        rr, mm = models[entity,"R"], models[entity,TARGET_MEAN_ARM]
        for model in (rr, mm):
            if (model["status"] != "COMPLETED" or model["seed"] != 2025 or model["updates"] != 2000
                    or model["logical_batch"] != 64 or model["exposures"] != 128000
                    or model["B1_forward_count"] != 128000 or model["labels_read"]
                    or model["CAL_QUERY_opened"] or model["roles_read"] != ["SOURCE_FIT", "U_FIT"]
                    or not model["all_modules_eval"] or not model["frozen_parameters_buffers_unchanged"]):
                raise ValueError("Fixed training permissions, budget, mode or scope changed")
            checked(workspace, dict(relative=model["checkpoint_relative"], sha256=model["checkpoint_sha256"]))
        for key in ("scaler_sha256", "plan_sha256", "initial_state_hashes", "trainable_names", "trainable_parameters"):
            if rr[key] != mm[key]: raise ValueError("Paired initial, plan or scope differs: "+key)
        if rr.get("loss_profile", "native") != "native" or mm["loss_profile"] != "target_mean_source_copy":
            raise ValueError("Single target loss factor changed")
        for role in ("CAL", "QUERY"):
            reference = None
            for arm in ("noU", "R", TARGET_MEAN_ARM):
                descriptor = records[entity,arm,role]; p = checked(workspace, descriptor)
                receipt = read(checked(workspace, dict(relative=descriptor["receipt_relative"], sha256=descriptor["receipt_sha256"])))
                if receipt["labels_read"] or receipt["scaler_sha256"] != rr["scaler_sha256"]:
                    raise ValueError("Readout scaler/label permissions mismatch")
                for part in receipt["parts"]:
                    pr = read(checked(workspace, part))
                    expected_checkpoint = task["native"]["checkpoint_sha256"] if arm == "noU" else models[entity,arm]["checkpoint_sha256"]
                    if (pr["checkpoint_sha256"] != expected_checkpoint or not pr["B1_only"] or pr["mask_seed"] != 84551
                            or pr["labels_read"] or not pr["native_immutable"] or not pr["CPU_CUDA_RNG_restored"]):
                        raise ValueError("Fixed native checkpoint, B1 mask or immutable state changed")
                bank = load_bank(p)
                point_formula_gate("A", bank)
                np.testing.assert_array_equal(bank["absolute_time_index"], cal if role == "CAL" else query)
                if reference is not None:
                    for key in ("absolute_time_index", "available_index", "current_raw", "current_normalized", "window_start", "window_end_exclusive", "actual_order"):
                        np.testing.assert_array_equal(bank[key], reference[key])
                reference = bank; paths[entity,arm,role] = p
    return seal, tasks, paths


def run_target_mean_smd(workspace, root, output, workers=2):
    """Fixed two-machine extension; no other task's historical score is edited."""
    start = time.perf_counter()
    workspace, root, output = (Path(p).resolve() for p in (workspace, root, output))
    if output.exists(): raise FileExistsError(output)
    seal, tasks, paths = smd_inputs(workspace, root)
    metric = read(checked(workspace, dict(relative=EXTERNAL_MANIFEST, sha256=EXTERNAL_MANIFEST_SHA)))["author_metric"]
    rows, events, surfaces, low, checks = [], [], {}, [], {}
    from .native_v_conditioning_curves import low_fpr_entry
    for entity in SMD_FIXED_ENTITIES:
        task = tasks[entity]
        descriptor = read(root/entity/"EVALUATION_BINDING.json")
        p = checked(workspace, descriptor)
        labels = (np.loadtxt(p, dtype=np.int8) if descriptor.get("format") == "txt" else
                  np.load(p, allow_pickle=False) if p.suffix == ".npy" else load_bank(p)[descriptor["key"]])
        labels = np.asarray(labels).reshape(-1)
        points = np.load(checked(workspace, task["indexes"]["CAL"]), allow_pickle=False)
        for arm in ("noU", "R", TARGET_MEAN_ARM):
            cal, query = (load_bank(paths[entity,arm,r]) for r in ("CAL", "QUERY"))
            rr, ee, surface = evaluate_smd_arrays(entity, arm, cal, query, points, labels, workspace/metric)
            for row in rr:
                row.update(bank_CAL_sha256=sha(paths[entity,arm,"CAL"]), bank_QUERY_sha256=sha(paths[entity,arm,"QUERY"]))
            ll, arrays = low_fpr_entry(query["absolute_time_index"], query["canonical_score"], labels, rr,
                dict(domain="SMD", entity=entity, method=arm, seed=None if arm == "noU" else 2025))
            rows.extend(rr); events.extend(ee); low.extend(ll)
            for key, value in zip(("TPR", "FPR", "precision", "windows"), surface):
                surfaces[f"{entity}_{arm}_{key}"] = value
            for key, value in arrays.items(): surfaces[f"{entity}_{arm}_LOW_{key}"] = value
            checks[entity+"_"+arm] = dict(CAL=point_formula_gate("A", cal), QUERY=point_formula_gate("A", query))
    paired = smd_learning_pairs(rows)
    output.mkdir(parents=True)
    from .left_swat_replay_repeats import write_csv
    write_csv(output/"methods.csv", rows); write_csv(output/"learning_pairs.csv", paired)
    write_csv(output/"LOW_FPR_SUPPLEMENT.csv", low)
    np.savez_compressed(output/"VUS_AND_LOW_CURVES.npz", **surfaces)
    (output/"EVENTS.json").write_text(json.dumps(events,indent=2,allow_nan=False),encoding="utf-8")
    result = dict(status="COMPLETED", rows=rows, paired=paired, low_fpr=low,
        formula_checks=checks, global_seal_sha256=sha(root/"GLOBAL_RAW_BANK_SEAL.json"),
        actual_GPU_hours=seal["actual_GPU_hours"], cpu_wall_seconds=time.perf_counter()-start,
        interpretation="single-seed fixed historically exposed SMD subset, not fullSMD/blind/independent domains; normal train CAL not query labels",
        bank_manifest=seal["banks"])
    (output/"METRICS.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    return result


def run(workspace, root, output, workers=2):
    started = time.perf_counter()
    workspace, root, output = Path(workspace).resolve(), Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    seal, paths, tasks = sealed_inputs(workspace, root)
    shared_checks = shared_evidence_gate(paths, tasks)
    # Metadata-only references to accepted public-data caches; no old neural model.
    if sha(workspace / EXTERNAL_METRICS) != EXTERNAL_SHA or sha(workspace / EXTERNAL_MANIFEST) != EXTERNAL_MANIFEST_SHA:
        raise ValueError("Accepted external results changed")
    prior = read(workspace / EXTERNAL_METRICS)
    manifest = read(workspace / EXTERNAL_MANIFEST)
    # Only after all six weights and16 banks are identity-checked are labels joined.
    labels = {}
    for d, descriptor in manifest["labels"].items():
        path = checked(workspace, dict(relative=descriptor["path"], sha256=descriptor["sha256"]))
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                values = z[descriptor["key"]].copy()
        else:
            values = np.load(path, allow_pickle=False).reshape(-1)
        if values.shape != (descriptor["length"],) or not np.isin(values, [0, 1]).all():
            raise ValueError("Original complete public point-label axis required")
        labels[d] = values
    from sklearn.metrics import average_precision_score, roc_auc_score
    step0 = {}
    for d in AXES:
        cal, query = load_bank(paths[d, "A", "CAL"]), load_bank(paths[d, "A", "QUERY"])
        scale = read(root / d / "QUANTILE_SCALE.json")
        record, score = step0_equivalence(cal["V"], query["V"], scale["s_V"], scale["q_bias_f32"])
        y = labels[d][query["absolute_time_index"]]
        record.update(native_AP=float(average_precision_score(y, query["V"])),
            step0_AP=float(average_precision_score(y, score)), native_ROC_AUC=float(roc_auc_score(y, query["V"])),
            step0_ROC_AUC=float(roc_auc_score(y, score)), labels_used="post-seal descriptive check only")
        step0[d] = record
    output.mkdir(parents=True, exist_ok=False)
    rows, events, surfaces, formula_checks = [], [], {}, {}
    pending = []
    for d in AXES:
        points = np.load(workspace / tasks[d]["indexes"]["CAL"]["relative"], allow_pickle=False)
        for m in METHODS:
            pending.append((d, m, paths[d, m, "CAL"], paths[d, m, "QUERY"], points,
                            labels[d], workspace / manifest["author_metric"]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for item, (rr, ee, surface, check) in zip(pending, pool.map(evaluate_item, pending)):
            rows.extend(rr); events.extend(ee)
            key = item[0] + "_" + item[1]
            formula_checks[key] = check
            for k, value in zip(("TPR", "FPR", "precision", "windows"), surface):
                surfaces[key + "_" + k] = value
    for row in prior["rows"]:
        if row["method"] in EXTERNAL_NAMES:
            if row["QUERY_axis"] != list(AXES[row["domain"]]):
                raise ValueError("Cannot subtract or reuse metrics across QUERY axes")
            rows.append(dict(row, reused_from_METRICS_sha256=EXTERNAL_SHA, external=True))
    if sha(workspace / EXTERNAL_EVENTS) != EXTERNAL_EVENTS_SHA:
        raise ValueError("Accepted external event identities changed")
    events.extend(dict(e, external=True) for e in read(workspace / EXTERNAL_EVENTS) if e["method"] in EXTERNAL_NAMES)
    pair = paired_rows(rows)
    per_event = event_contrasts(events)
    from .left_swat_replay_repeats import write_csv
    write_csv(output / "methods.csv", rows)
    write_csv(output / "paired.csv", pair)
    write_csv(output / "EVENT_PAIRS.csv", per_event)
    costs = model_costs(workspace, root, seal)
    write_csv(output / "MODEL_COSTS.csv", costs)
    np.savez_compressed(output / "NEW_VUS_SURFACES.npz", **surfaces)
    (output / "EVENTS.json").write_text(json.dumps(events, indent=2, allow_nan=False), encoding="utf-8")
    result = dict(status="COMPLETED", rows=rows, paired=pair, formula_checks=formula_checks,
        shared_evidence_checks=shared_checks, step0_equivalence=step0, model_costs=costs,
        protocol_commit=BASE_PROTOCOL, raw_seal_sha256=sha(root / "GLOBAL_RAW_BANK_SEAL.json"),
        external_METRICS_sha256=EXTERNAL_SHA, external_VUS_recomputed=False,
        actual_GPU_hours=seal.get("actual_GPU_hours"), cpu_wall_seconds=time.perf_counter() - started,
        labels_joined_only_after_all_six_selected_and16_banks=True,
        limitations=["single seed2025, two historically exposed public development tasks",
            "same-domain normal source learning, no R or new heterogeneous pretraining",
            "TimeRadar official weights, TimesFM3 code blocks only; no official TimesFM3 weights",
            "mixed-CAL empirical operating points; PSM CAL all attack-labeled",
            "B/C not equal architecture or parameter count; D changes target/loss/head and teacher dependence",
            "external offline/context/scaler/source-adaptation exposure differences remain explicit"])
    (output / "METRICS.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("workspace", "root", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mode", choices=("conditioning", "head-replay", "target-mean", "target-mean-repeat", "target-mean-smd"), default="conditioning")
    args = parser.parse_args()
    fn = {"conditioning": run, "head-replay": run_head_replay, "target-mean": run_target_mean,
          "target-mean-repeat": run_target_mean_repeat, "target-mean-smd": run_target_mean_smd}[args.mode]
    result = fn(args.workspace, args.root, args.out, args.workers)
    print(f"Completed {len(result['rows'])} fixed method/stream rows")
