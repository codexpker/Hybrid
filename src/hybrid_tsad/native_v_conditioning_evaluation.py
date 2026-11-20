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


def replay_pairs(rows):
    """Same-backbone/criterion learning contrasts; one fixed seed, not a CI."""
    lookup = {(r["domain"], r["method"], r["stream"]): r for r in rows}
    fields = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "TP", "FP", "normal_FPR", "Recall", "F1",
        "M", "auxiliary_any_hit_events", "one_to_one_eventPrecision", "one_to_one_eventRecall",
        "one_to_one_eventF1", "event_macro_coverage", "event_point_coverage", "long_steady_macro_coverage",
        "long_steady_point_coverage", "max_event_miss_points", "normal_alarm_fragments")
    result = []
    for d in AXES:
        for stream in ("strict", "confirmed3"):
            for a, b in (("R", "TT"), ("R", "noU"), ("R", "SS"), ("TT", "noU")):
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
            row = dict(old, method="noU", arm="noU", reused_from_METRICS_sha256=ACCEPTED_NATIVE_METRICS_SHA)
            if sha(paths[row["domain"], "noU", "QUERY"]) != row["bank_QUERY_sha256"]:
                raise ValueError("Frozen A/noU QUERY identity changed")
            if sha(paths[row["domain"], "noU", "CAL"]) != row["bank_CAL_sha256"]:
                raise ValueError("Frozen A/noU CAL identity changed")
            rows.append(row)
    old_events = read(checked(workspace, dict(relative=ACCEPTED_NATIVE_EVENTS, sha256=ACCEPTED_NATIVE_EVENTS_SHA)))
    events.extend(dict(e, method="noU") for e in old_events if e["method"] == METHODS["A"])
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
    parser.add_argument("--mode", choices=("conditioning", "head-replay"), default="conditioning")
    args = parser.parse_args()
    fn = run_head_replay if args.mode == "head-replay" else run
    result = fn(args.workspace, args.root, args.out, args.workers)
    print(f"Completed {len(result['rows'])} fixed method/stream rows")
