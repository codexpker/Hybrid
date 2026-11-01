"""Pure fixed TEP evaluation. No I/O, models, fitting or real-data release.

Contract TEP-FIXED-SCORE-EVALUATION-20261102, including its three clarifications.
Sample numbers in event records are one-based; array endpoints are zero-based.
The future file loader/CLI is deliberately unavailable until separately reviewed.
"""
import argparse
import json
import numbers

import numpy as np

from .events import binary_events

SEEDS = (2025, 2026, 2027)
METHODS = ("LEFT", "SELF", "SYNC", "PEAK", "MULTI", "ZMAX")
READOUTS = ("last", "mean32")
FAULTS = tuple(range(1, 21))
RUNS = tuple(range(51, 71))
METRICS = ("AP", "trapezoid_PR_AUC", "ROC_AUC", "AP_divided_by_pi",
           "pointPrecision", "pointRecall", "pointF1", "pointFPR",
           "eventPrecision_full800", "eventRecall_60min", "eventRecall_full800",
           "eventF1_full800", "normal_false_alarms_per_hour_183min",
           "normal_false_alarms_per_hour_180min", "detection_delay_minutes")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(name + " must be an integer, not a coerced ID")
    return int(value)


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(name + " must be a finite real scalar")
    return float(value)


def _geometry(scores, end_index_0based, available_index_0based=None):
    scores, ends = np.asarray(scores), np.asarray(end_index_0based)
    if (scores.shape != (861,) or scores.dtype.kind not in 'fi' or
            not np.isfinite(scores).all()):
        raise ValueError("Exactly 861 finite real scores required; never fill missing scores")
    if (ends.dtype.kind not in 'iu' or ends.shape != (861,) or
            not np.array_equal(ends, np.arange(99, 960))):
        raise ValueError("Endpoints must be integer 99..959 without gaps/duplicates")
    if available_index_0based is not None:
        available = np.asarray(available_index_0based)
        if available.dtype.kind not in 'iu' or not np.array_equal(available, ends):
            raise ValueError("Scores must be available at their endpoint, never backfilled")
    return scores.astype(np.float64, copy=False)


def event_metrics(scores, end_index_0based, threshold):
    """Strict exceedance; only the third consecutive point confirms a segment.

    Boundary continuation includes short, unconfirmed crossing segments. They
    never become new faults. A segment confirmed before injection remains a
    normal false alarm even if it later crosses the boundary; count it once.
    """
    score = _geometry(scores, end_index_0based)
    threshold = _finite_scalar(threshold, "threshold")
    segments = []
    normal = new = boundary = 0
    first = None
    for lo, hi in binary_events(score > threshold):
        start, end = lo + 100, hi + 99  # inclusive original sample numbers
        confirm = start + 2 if hi - lo >= 3 else None
        crossing = start < 161 <= end
        normal += int(confirm is not None and confirm < 161)
        if crossing:
            kind = "boundary_continuation"
            boundary += 1
        elif start < 161:
            kind = "normal"
        else:
            kind = "new_fault_period"
            if confirm is not None:
                new += 1
                if first is None:
                    first = confirm
        segments.append(dict(start_sample=start, end_sample_inclusive=end,
                             confirm_sample=confirm, kind=kind,
                             boundary_confirmation=("before_injection" if confirm < 161 else "after_injection")
                             if crossing and confirm is not None else None))
    tp = int(first is not None)
    fp = normal + new - tp
    fn = 1 - tp
    alarms = normal + new
    delay = (first - 161) * 3 if first is not None else None
    metrics = dict(eventPrecision_full800=tp / alarms if alarms else None,
                   eventRecall_full800=float(tp), eventRecall_60min=float(first is not None and first <= 181),
                   eventF1_full800=2 * tp / (2 * tp + fp + fn),
                   normal_false_alarms_per_hour_183min=normal * 60 / 183,
                   normal_false_alarms_per_hour_180min=normal * 60 / 180,
                   detection_delay_minutes=delay)
    return dict(metrics=metrics, segments=segments, normal_confirmed_segments=normal,
                new_confirmed_segments=new, repeated_new_alarms=max(new - 1, 0),
                boundary_segments=boundary,
                boundary_confirmed_before=sum(s['boundary_confirmation'] == 'before_injection' for s in segments),
                boundary_confirmed_after=sum(s['boundary_confirmation'] == 'after_injection' for s in segments),
                first_eligible_confirmation_sample=first, delay_censored=first is None,
                censor_sample=960, censor_horizon_minutes=(960 - 161) * 3,
                normal_exposure_sample_minutes=183, normal_timestamp_span_minutes=180,
                event_TP=tp, event_FP=fp, event_FN=fn)


def evaluate_case(scores, end_index_0based, threshold, fault, run, method, readout, seed=None,
                  available_index_0based=None):
    """Evaluate one complete cell; labels are fixed nominal sample>=161 only.

    Ranking uses mature sklearn conventions (including ties); no synthetic-domain
    identity guard is imported into this pure numeric evaluator.
    """
    fault, run = _integer(fault, 'fault'), _integer(run, 'run')
    if fault not in FAULTS or run not in RUNS or method not in METHODS or readout not in READOUTS:
        raise ValueError("Unexpected fixed cell identity")
    if method == 'LEFT':
        if _integer(seed, 'seed') not in SEEDS:
            raise ValueError('LEFT requires the three predetermined training seeds')
    elif seed is not None:
        raise ValueError('Deterministic baseline must not be duplicated as training seeds')
    score = _geometry(scores, end_index_0based, available_index_0based)
    result = event_metrics(score, end_index_0based, threshold)
    # Import only after input validation; local event tests do not need sklearn.
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
    labels = np.arange(100, 961) >= 161
    positive = score > float(threshold)
    tp, fp = int((positive & labels).sum()), int((positive & ~labels).sum())
    fn, tn = 800 - tp, 61 - fp
    precision, recall, _ = precision_recall_curve(labels, score)
    ap = float(average_precision_score(labels, score))
    result['metrics'].update(AP=ap, ROC_AUC=float(roc_auc_score(labels, score)),
        trapezoid_PR_AUC=float(auc(recall, precision)), AP_divided_by_pi=ap / (800 / 861),
        pointPrecision=tp / (tp + fp) if tp + fp else None, pointRecall=tp / 800,
        pointF1=2 * tp / (2 * tp + fp + fn), pointFPR=fp / 61)
    result.update(fault=fault, run=run, method=method, readout=readout, seed=seed,
                  threshold=float(threshold), positive_points=800, normal_points=61,
                  point_TP=tp, point_FP=fp, point_FN=fn, point_TN=tn)
    return result


def validate_compact(arrays, family):
    """Validate already-loaded compact arrays, without opening any NPZ.

    Entity->fault/run identities must later come from an accepted sealed mapping;
    opaque entity numbers are never guessed as fault/run IDs here.
    """
    if family not in ('LEFT', 'RIDGE_ZMAX'):
        raise ValueError('Unknown compact family')
    base = {'entity_id', 'end_index_0based'}
    keys = base | ({'last', 'mean32', 'channel_last', 'channel_mean32', 'available_index_0based'}
                   if family == 'LEFT' else {m + '_' + r for m in METHODS[1:] for r in READOUTS})
    if set(arrays) != keys:
        raise ValueError('Compact keys differ from frozen query schema')
    entities = np.asarray(arrays['entity_id'])
    if entities.dtype.kind not in 'US' or entities.shape != (400,):
        raise ValueError('400 string entity IDs required')
    if entities.tolist() != ['entity_%04d' % i for i in range(1, 401)]:
        raise ValueError('Missing/reordered/duplicate entities')
    ends = np.asarray(arrays['end_index_0based'])
    if ends.shape != (400, 861):
        raise ValueError('Incomplete endpoint matrix')
    for i in range(400):
        _geometry(np.zeros(861), ends[i])
    for key in keys - base:
        a = np.asarray(arrays[key])
        if key == 'available_index_0based':
            if a.dtype.kind not in 'iu' or not np.array_equal(a, ends):
                raise ValueError('Availability differs from endpoints')
        elif (a.shape != ((400, 861, 3) if key.startswith('channel_') else (400, 861))
              or a.dtype.kind not in 'fi' or not np.isfinite(a).all()):
            raise ValueError('Incomplete/nonfinite score array: ' + key)
    return True


def bootstrap_draws():
    """One canonical matrix; save this array and reuse across every comparison."""
    return np.random.default_rng(20251101).integers(0, 20, size=(10000, 20), dtype=np.int64)


def _summary(values, ids):
    good = [float(v) for v in values if v is not None]
    return dict(mean=float(np.mean(good)) if good else None, total=len(values),
                valid=len(good), NA=len(values) - len(good),
                NA_ids=[i for v, i in zip(values, ids) if v is None])


def aggregate_cases(cases):
    """Strict full 6400-cell coverage; seed-average within original cells first.

    Missing seed values propagate NA into the seed-averaged cell. Descriptive
    means disclose all NA identities. No paired procedure drops any fixed cell.
    """
    cases = list(cases)
    expected = {(f, r, m, d, s) for f in FAULTS for r in RUNS for m in METHODS
                for d in READOUTS for s in (SEEDS if m == 'LEFT' else (None,))}
    indexed = {}
    for row in cases:
        f, r = _integer(row['fault'], 'fault'), _integer(row['run'], 'run')
        seed = row['seed']
        if seed is not None:
            seed = _integer(seed, 'seed')
        key = (f, r, row['method'], row['readout'], seed)
        if key not in expected or key in indexed or set(row['metrics']) != set(METRICS):
            raise ValueError('Unexpected/duplicate cell or incomplete metric set')
        for name, value in row['metrics'].items():
            if value is not None:
                _finite_scalar(value, name)
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('Incomplete fixed coverage; no missing cells may be filled')
    cells = []
    for f in FAULTS:
        for r in RUNS:
            for m in METHODS:
                for d in READOUTS:
                    rows = [indexed[f, r, m, d, s] for s in (SEEDS if m == 'LEFT' else (None,))]
                    metrics = {}
                    for metric in METRICS:
                        vals = [row['metrics'][metric] for row in rows]
                        metrics[metric] = None if any(v is None for v in vals) else float(np.mean(vals))
                    cells.append(dict(fault=f, run=r, method=m, readout=d, metrics=metrics))
    summaries = []
    for faults, group in [(FAULTS, 'all20'), ((4, 11, 14), 'related3')] + [((f,), 'fault%02d' % f) for f in FAULTS]:
        for m in METHODS:
            for d in READOUTS:
                selected = [c for c in cells if c['fault'] in faults and c['method'] == m and c['readout'] == d]
                ids = ['f%02d/run%d' % (c['fault'], c['run']) for c in selected]
                summaries.append(dict(group=group, method=m, readout=d,
                    metrics={k: _summary([c['metrics'][k] for c in selected], ids) for k in METRICS}))
    seed_summaries = []
    for seed in SEEDS:
        for group, faults in [('all20', FAULTS), ('related3', (4, 11, 14))] + [('fault%02d' % f, (f,)) for f in FAULTS]:
            for d in READOUTS:
                selected = [indexed[f, r, 'LEFT', d, seed] for f in faults for r in RUNS]
                ids = ['f%02d/run%d/seed%d' % (c['fault'], c['run'], seed) for c in selected]
                seed_summaries.append(dict(group=group, readout=d, seed=seed,
                    metrics={k: _summary([c['metrics'][k] for c in selected], ids) for k in METRICS}))
    # Keep originals and per-seed summaries, including every failing/NA cell.
    # Worst-case ranking direction varies by metric; no implicit best-seed selection.
    return dict(cases=cases, seed_averaged_cells=cells, summaries=summaries,
                LEFT_per_seed_summaries=seed_summaries)


def paired_intervals(aggregation, draws):
    """Paired run-cluster sensitivity CIs, not multiplicity-adjusted tests.

    Faults stay inside each run cluster; the SAME draws apply to both groups,
    every metric/method/readout. A single NA makes that paired comparison incomplete.
    """
    draws = np.asarray(draws)
    if draws.dtype != np.int64 or not np.array_equal(draws, bootstrap_draws()):
        raise ValueError('Use unchanged canonical seed20251101/10000x20 draw matrix')
    cells = aggregation['seed_averaged_cells']
    lookup = {(c['fault'], c['run'], c['method'], c['readout']): c['metrics'] for c in cells}
    expected = {(f, r, m, d) for f in FAULTS for r in RUNS for m in METHODS for d in READOUTS}
    if len(cells) != len(expected) or set(lookup) != expected:
        raise ValueError('Incomplete or duplicate seed-averaged cells')
    result = []
    for group, faults in [('all20', FAULTS), ('related3', (4, 11, 14))]:
        for d in READOUTS:
            for baseline in METHODS[1:]:
                for metric in METRICS:
                    bad, delta = [], []
                    for r in RUNS:
                        differences = []
                        for f in faults:
                            a, b = lookup[f, r, 'LEFT', d][metric], lookup[f, r, baseline, d][metric]
                            if a is None or b is None:
                                bad.append('f%02d/run%d' % (f, r))
                            else:
                                differences.append(_finite_scalar(a, metric) - _finite_scalar(b, metric))
                        delta.append(float(np.mean(differences)) if len(differences) == len(faults) else None)
                    record = dict(group=group, readout=d, comparison='LEFT-' + baseline, metric=metric,
                                  total_cells=20 * len(faults), NA_ids=bad, cluster_count=20,
                                  status='INCOMPLETE' if bad else 'COMPLETE', mean_difference=None,
                                  CI95=None, multiplicity_adjusted=False)
                    if not bad:
                        delta = np.asarray(delta)
                        values = delta[draws].mean(axis=1)
                        record.update(mean_difference=float(delta.mean()), CI95=np.percentile(values, [2.5, 97.5]).tolist())
                    result.append(record)
    return result


def require_future_release(config):
    """Fail closed BEFORE any score, identity mapping or nominal-label file I/O.

    No actual seal references or release schema have been accepted for this build.
    Even a forged non-null draft cannot enable a real evaluation through this CLI.
    """
    raise PermissionError('PENDING: real evaluation loader/release is not implemented or authorized')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as handle:
        config = json.load(handle)
    require_future_release(config)


if __name__ == '__main__':
    main()
