"""Fixed-score evaluation using the original event and AP helper contracts.

No score, threshold, model or seed selection. Affiliation requires an external
licensed callback; absent callback is explicitly unevaluated, never substituted.
"""
from types import SimpleNamespace

import numpy as np

from .events import binary_events, greedy_assign, real_blocks
from .statistics import ap_cache, weighted_ap


def evaluate_point_ranking(index, score, full_labels, scored_mask, *, query_start):
    """Fixed AP/ROC/PR-trapezoid plus original-label coverage, no threshold.

    Full labels are the complete provided test sequence, not a compressed mask.
    This function never selects a cutoff, adjusts points, or fits a model.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
    index, score = np.asarray(index), np.asarray(score)
    full_labels, mask = np.asarray(full_labels), np.asarray(scored_mask)
    if (index.ndim != 1 or score.shape != index.shape or full_labels.ndim != 1 or
            mask.shape != full_labels.shape or mask.dtype != np.bool_ or not len(index) or
            not np.issubdtype(index.dtype, np.integer) or np.any(np.diff(index) <= 0) or
            not np.isfinite(score).all() or not np.isfinite(full_labels).all() or
            not np.isin(full_labels, [0, 1]).all()):
        raise ValueError("Need finite fixed scores, binary full labels and real boolean coverage")
    full_index = query_start + np.arange(len(full_labels), dtype=np.int64)
    if not np.array_equal(index, full_index[mask]):
        raise ValueError("Frozen score indices and label/coverage alignment disagree; no offset repair")
    y = full_labels[mask]
    pi = float(y.mean())
    metrics = {"evaluated_points": len(index), "positive_points": int((y == 1).sum()),
               "negative_points": int((y == 0).sum()), "positive_fraction": pi,
               "unscored_points": int((~mask).sum())}
    precision = recall = np.array([], dtype=np.float64)
    if len(np.unique(y)) == 2:
        metrics.update(AP=float(average_precision_score(y, score)), ROC_AUC=float(roc_auc_score(y, score)))
        precision, recall, _ = precision_recall_curve(y, score)
        metrics["trapezoid_PR_AUC"] = float(auc(recall, precision))
        metrics["AP_divided_by_pi"] = metrics["AP"] / pi
        metrics["ranking_metric_NA_reason"] = None
    else:
        metrics.update(AP=None, ROC_AUC=None, trapezoid_PR_AUC=None, AP_divided_by_pi=None,
                       ranking_metric_NA_reason="SINGLE_CLASS_SCORED_LABELS")
    coverage = []
    for i, (lo, hi) in enumerate(binary_events(full_labels)):
        scored = int(mask[lo:hi].sum())
        coverage.append({"event_id": "PSM_test_E%03d" % (i+1),
                         "start": int(query_start+lo), "end_exclusive": int(query_start+hi),
                         "start_in_query": lo, "end_in_query_exclusive": hi,
                         "total_points": hi-lo, "scored_points": scored,
                         "coverage_ratio": scored/(hi-lo), "complete": scored == hi-lo,
                         "touches_query_start": lo == 0, "touches_query_end": hi == len(full_labels)})
    metrics.update(original_test_events=len(coverage),
                   complete_scoring_coverage_events=sum(e["complete"] for e in coverage),
                   incomplete_scoring_coverage_events=sum(not e["complete"] for e in coverage))
    return {"metrics": metrics, "event_coverage": coverage,
            "pr_precision": precision, "pr_recall": recall}


def validate_coverage(index, labels, coverage):
    """Reject contradictory event eligibility; never repair labels or boundaries."""
    if not isinstance(coverage, list):
        raise ValueError("Coverage must be a list")
    required = {"event_id", "start", "end_exclusive", "eligible_complete", "scored_points", "total_points"}
    seen, previous_end = set(), -1
    mapped = np.zeros(len(index), bool)
    owner = np.full(len(index), -1, dtype="int64")
    result = []
    for row in coverage:
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError("Coverage record is missing required fields")
        eid = row["event_id"]
        if not isinstance(eid, str) or not eid or len(eid) > 64 or eid in seen:
            raise ValueError("Coverage event IDs must be unique nonempty strings of at most 64 characters")
        seen.add(eid)
        for key in ("start", "end_exclusive", "scored_points", "total_points"):
            if type(row[key]) is not int:
                raise ValueError("Coverage boundaries and counts must be integers")
        start, end = row["start"], row["end_exclusive"]
        if start < 0 or end <= start or start < previous_end:
            raise ValueError("Coverage intervals must be ordered, positive-length and nonoverlapping")
        previous_end = end
        if type(row["eligible_complete"]) is not bool or row["total_points"] != end - start:
            raise ValueError("Coverage complete flag/total_points contradict the original interval")
        ix = np.flatnonzero((index >= start) & (index < end))
        if row["scored_points"] != len(ix) or not np.all(labels[ix] == 1):
            raise ValueError("Coverage scored count or positive-label mapping contradicts scored indices")
        actual_complete = len(ix) == end - start
        if row["eligible_complete"] != actual_complete:
            raise ValueError("Coverage complete eligibility contradicts full original-index coverage")
        if actual_complete and not np.array_equal(index[ix], np.arange(start, end)):
            raise ValueError("Complete event has missing original time indices")
        mapped[ix] = True
        owner[ix] = len(result)
        result.append(SimpleNamespace(**row))
    if not np.array_equal(mapped, labels == 1):
        raise ValueError("Positive scored labels are missing from coverage")
    contiguous_positive = (np.diff(index) == 1) & (labels[:-1] == 1) & (labels[1:] == 1)
    if np.any(contiguous_positive & (owner[:-1] != owner[1:])):
        raise ValueError("A contiguous scored positive run cannot be split across coverage event IDs")
    return result


def evaluate_fixed_scores(index, labels, score, threshold, coverage, *, affiliation=None):

    index, y, s = np.asarray(index), np.asarray(labels), np.asarray(score)
    if index.ndim != 1 or y.shape != index.shape or s.shape != index.shape:
        raise ValueError("Index, labels and score must be equal one-dimensional arrays")
    if not len(index) or not np.issubdtype(index.dtype, np.integer) or np.any(np.diff(index) <= 0) or not np.isfinite(s).all():
        raise ValueError("Need finite scores and strictly increasing original time indices")
    if not np.array_equal(np.unique(y), [0, 1]) or not np.isfinite(threshold):
        raise ValueError("Fixed evaluation requires both binary classes and a finite source threshold")
    coverage = validate_coverage(index, y, coverage)
    from sklearn.metrics import average_precision_score, roc_auc_score
    complete = [e for e in coverage if e.eligible_complete]
    event_id = np.full(len(index), "", dtype="<U64")
    for e in coverage:
        event_id[(index >= e.start) & (index < e.end_exclusive)] = e.event_id
    event_valid = (y == 0) | np.isin(event_id, [e.event_id for e in complete])
    pred = s > threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    tn = int((~pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    hits = matched_count = predicted_count = normal_fragments = 0
    events, alarms, affinity = [], [], []
    aff_f1 = float("nan")
    blocks = real_blocks(index, event_valid)
    for bi, ix in enumerate(blocks):
        pe = [(int(index[ix[lo]]), int(index[ix[hi - 1]]) + 1)
              for lo, hi in binary_events(pred[ix])]
        truth = [(e.event_id, e.start, e.end_exclusive) for e in complete
                 if e.start >= index[ix[0]] and e.end_exclusive <= index[ix[-1]] + 1]
        assignment = greedy_assign(pe, truth)
        matched_count += sum(hit is not None for hit in assignment)
        predicted_count += len(pe)
        for (lo, hi), hit in zip(pe, assignment):
            alarms.append(dict(block=bi, start=lo, end_exclusive=hi, matched_event=hit))
        normal_fragments += len(binary_events(pred[ix] & (y[ix] == 0)))
        if affiliation is not None:
            gt = binary_events(y[ix])
            p = r = block_f1 = float("nan")
            reason = "NO_GROUND_TRUTH_EVENTS"
            if gt:
                ar = affiliation(binary_events(pred[ix]), gt, (0, len(ix)))
                p, r = float(ar["precision"]), float(ar["recall"])
                block_f1 = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and p + r > 0 else float("nan")
                reason = None if np.isfinite(block_f1) else "OFFICIAL_UNDEFINED"
            affinity.append(dict(block=bi, start=int(index[ix[0]]), end_exclusive=int(index[ix[-1]]) + 1,
                                 Aff_P=p, Aff_R=r, Aff_F1=block_f1, reason=reason))
    aff_reason = "NOT_EVALUATED_EXTERNAL_CALLBACK_REQUIRED"
    if affiliation is not None:
        aff_reason = "MULTIPLE_REAL_BLOCKS_NO_GLOBAL_AGGREGATION" if len(blocks) > 1 else "NO_VALID_BLOCKS"
        if len(blocks) == 1:
            aff_f1, aff_reason = affinity[0]["Aff_F1"], affinity[0]["reason"]
    for e in coverage:
        eligible = bool(e.eligible_complete)
        ix = np.flatnonzero((event_id == e.event_id) & pred)
        hit = bool(len(ix)) if eligible else None
        if eligible:
            hits += int(hit)
        events.append(dict(event_id=e.event_id, complete=eligible, any_hit=hit,
                           delay_sample_points=int(index[ix[0]] - e.start) if eligible and hit else None,
                           scored_points=e.scored_points, total_points=e.total_points))
    ap, roc = float(average_precision_score(y, s)), float(roc_auc_score(y, s))
    metrics = dict(AP=ap, ROC=roc, positive_fraction=float(y.mean()),
                   AP_over_positive_fraction=ap / y.mean(), threshold=threshold,
                   normal_FPR=fp / (fp + tn), point_Precision=tp / (tp + fp) if tp + fp else float("nan"),
                   TP=tp, FP=fp, TN=tn, FN=fn, complete_event_hits=hits,
                   complete_events=len(complete), original_events=len(coverage),
                   complete_event_recall_any_hit_aux=hits / len(complete) if complete else float("nan"),
                   predicted_events=predicted_count,
                   one_to_one_event_precision=matched_count / predicted_count if predicted_count else float("nan"),
                   normal_alarm_fragments=normal_fragments, Aff_F1=aff_f1,
                   Aff_F1_reason=aff_reason,
                   affiliation_status="CALLBACK_USED" if affiliation is not None else "NOT_EVALUATED_EXTERNAL_CALLBACK_REQUIRED",
                   delay_units="original sample points; no reliable physical clock")
    return {"metrics": metrics, "events": events, "alarms": alarms, "affinity": affinity,
            "complete_event_valid": event_valid}


def paired_block_differences(labels, scores, block_id, *, methods=("temporal", "late", "history_mean"),
                             seeds=(0, 1, 2), n_replicates=2000, rng_seed=99173):
    """Original paired time-block AP calculation; not independent-domain inference."""
    y, block_id = np.asarray(labels), np.asarray(block_id)
    n_blocks = int(block_id.max()) + 1
    names = {f"{m}_s{i}" for m in ("hybrid",) + tuple(methods) for i in seeds}
    cache = {key: ap_cache(y, scores[key], block_id) for key in names}
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, n_blocks, size=(n_replicates, n_blocks))
    delta = np.full((n_replicates, len(methods)), np.nan)
    single_class = 0
    point_positive = np.bincount(block_id, weights=y, minlength=n_blocks)
    point_total = np.bincount(block_id, minlength=n_blocks)
    for j, draw in enumerate(draws):
        count = np.bincount(draw, minlength=n_blocks)
        if count @ point_positive == 0 or count @ (point_total - point_positive) == 0:
            single_class += 1
            continue
        ap = {key: weighted_ap(cache[key], count) for key in names}
        for ci, comp in enumerate(methods):
            delta[j, ci] = np.mean([ap[f"hybrid_s{i}"] - ap[f"{comp}_s{i}"] for i in seeds])
    return {"draws": draws, "AP_difference": delta, "single_class_NA": single_class,
            "comparisons": ["hybrid-" + m for m in methods],
            "scope": "paired time-block sensitivity; not independent-domain confirmation"}
