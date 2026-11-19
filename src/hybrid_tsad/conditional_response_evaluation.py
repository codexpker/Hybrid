"""Fixed-axis evaluation for conditional-response energy and cached comparators.

This module neither loads models nor fits scores. Labels may be joined only
after the new A/B, noU/R banks are sealed. Old native axes remain untouched.
"""
from pathlib import Path
import json

import numpy as np

from .left_native_swat_metrics import continuous_metrics, load_author_metric, sha
from .timercd_swat_evaluation import evaluate_score


AXES = {
    "SWAT": {"CAL": (586056, 607320), "QUERY": (608535, 944856)},
    "PSM": {"CAL": (18624, 21888), "QUERY": (23016, 87744)},
}


def select_bank(path, axis, *, expected_sha=None, causal=False):
    """Select saved actual points, retaining each point's real availability."""
    path = Path(path)
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError("Score-bank identity changed")
    with np.load(path, allow_pickle=False) as bank:
        index = bank["absolute_time_index"].copy()
        score = bank["canonical_score"].copy()
        available = bank["available_index"].copy()
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
                   *, causal, cal_sha=None, query_sha=None, permission=None):
    """Evaluate one immutable model on the common new footprint, without fitting."""
    ci, cs, _ = select_bank(cal_path, AXES[domain]["CAL"], expected_sha=cal_sha, causal=causal)
    index, score, available = select_bank(query_path, AXES[domain]["QUERY"], expected_sha=query_sha, causal=causal)
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
