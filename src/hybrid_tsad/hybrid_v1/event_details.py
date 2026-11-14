"""Versioned UNKNOWN-aware event details for NEW exports only.

Does not modify archived fixed_metrics.py, thresholds, matching, or summaries.
Input alarms must already be defined by the caller's frozen policy.  Unknown
time points break observed miss runs; they are never counted as normal/missed.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from ..engine.events import binary_events

SCHEMA = 'event_observation_details_v2'


def _longest_observed_miss(scored, alarm):
    best = run = 0
    for seen, hit in zip(scored, alarm):
        run = run + 1 if seen and not hit else 0
        best = max(best, run)
    return best if scored.any() else None


def event_details(full_labels, index, alarm, *, original_start=0,
                  steady_after=192, long_min_length=512):
    """Keep full event bounds and return NA for ineligible full-event metrics.

    ``observed_*`` fields describe scored points only and are not full-event
    detection/coverage.  No event matching or any-hit recall is redefined here.
    """
    labels, index, alarm = np.asarray(full_labels), np.asarray(index), np.asarray(alarm)
    if (labels.ndim != 1 or not len(labels) or labels.dtype.kind not in 'biuf'
            or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all()
            or index.ndim != 1 or index.dtype.kind not in 'iu'
            or alarm.dtype != np.bool_ or alarm.shape != index.shape
            or np.any(index[1:] <= index[:-1]) or type(original_start) is not int or original_start < 0
            or np.any(index < original_start) or np.any(index >= original_start + len(labels))
            or type(steady_after) is not int or steady_after < 0
            or type(long_min_length) is not int or long_min_length < 0):
        raise ValueError('Complete original binary labels, unique time axis and boolean observed alarms required')
    index = index.astype(np.int64)
    scored = np.zeros(len(labels), bool); bits = np.zeros(len(labels), bool)
    scored[index-original_start] = True; bits[index-original_start] = alarm
    rows = []
    for number, (lo, hi) in enumerate(binary_events(labels), 1):
        seen, hits = scored[lo:hi], bits[lo:hi]
        count = int(seen.sum()); total = hi-lo
        complete = count == total
        boundary = lo == 0 or hi == len(labels)
        eligible = complete and not boundary
        miss = _longest_observed_miss(seen, hits)
        intervals = [[original_start+lo+a, original_start+lo+b] for a, b in binary_events(~seen)]
        row = dict(schema=SCHEMA, event_id='E%06d' % number,
            start=original_start+lo, end_exclusive=original_start+hi,
            total_points=total, scored_points=count, coverage_ratio=count/total,
            missing_points=total-count, missing_intervals=intervals,
            observation_status='COMPLETE' if complete else 'UNKNOWN_PARTIAL' if count else 'UNKNOWN_UNOBSERVED',
            boundary_censored=boundary, eligible_for_event_metric=eligible,
            metric_NA_reason=None if eligible else 'INCOMPLETE_SCORE_COVERAGE' if not complete else 'ORIGINAL_RUN_BOUNDARY_CENSORED',
            coverage=float(hits.mean()) if eligible else None,
            longest_miss_points=miss if eligible else None,
            observed_alarm_points=int(hits[seen].sum()),
            observed_only_coverage=float(hits[seen].mean()) if count else None,
            longest_observed_miss_points=miss,
            steady_coverage=None, steady_NA_reason='NOT_LONG_EVENT')
        if total > long_min_length:
            a = min(hi, lo+steady_after)
            s, h = scored[a:hi], bits[a:hi]
            row.update(steady_scored_points=int(s.sum()), steady_total_points=len(s),
                       steady_observed_only_coverage=float(h[s].mean()) if s.any() else None,
                       steady_coverage=float(h.mean()) if eligible and len(s) else None,
                       steady_NA_reason=('EMPTY_STEADY_INTERVAL' if not len(s) else row['metric_NA_reason']))
        rows.append(row)
    return rows


def write_new_details(path, rows):
    """Exclusive create: never overwrite an old CSV or silently change schema."""
    path = Path(path)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['schema']
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    return dict(schema=SCHEMA, rows=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
