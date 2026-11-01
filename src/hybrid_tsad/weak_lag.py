"""Transparent WL1 mechanisms and small CPU regression controls.

No LEFT, private data, neural model, query-selected alpha or threshold. Truth is
returned separately from observations; fitting accepts only normal role arrays.
The CLI is metadata preview only. Formal pool generation needs separate release.
"""
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

H = 192
N = 24576
MODELS = ('AR191', 'SYNC191', 'PEAK191', 'MULTI191', 'SYNC64', 'MULTI64')
ALPHAS = (.01, .1, 1., 10., 100.)
ROLES = {'FIT': 4, 'SELECT': 2, 'CAL': 8, 'DEV': 8}
READOUTS = ('AR191.last', 'SYNC191.last', 'PEAK191.last', 'MULTI191.last',
            'SYNC64.last', 'SYNC64.mean128', 'MULTI64.last', 'MULTI64.mean128')


def derive_seed(scenario, role, run_id, stream):
    """Contract UTF-8 SHA256 first eight bytes, uint64 little endian."""
    text = f'WL1|{scenario}|{role}|{run_id}|{stream}'
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8], 'little')


def _rng(scenario, role, run_id, stream):
    return np.random.Generator(np.random.PCG64(derive_seed(scenario, role, run_id, stream)))


def _ar1(rng, length):
    q = np.empty(length, dtype=np.float64)
    q[0] = rng.standard_normal()
    innovation = rng.standard_normal(length - 1)
    for i in range(1, length):
        q[i] = .8*q[i-1] + .6*innovation[i-1]
    return q


def normal_lag(scenario, role, run_id, n=N, h=H):
    """Dwell starts at retained t=0; burn-in is for stationary latent streams."""
    if scenario == 'LIN':
        return np.full(n, 12, dtype=np.int64)
    if scenario != 'WEAK':
        raise ValueError('Generator scenarios are LIN or WEAK; IRR is a paired observation')
    rng = _rng(scenario, role, run_id, 'lag')
    current = int(rng.choice([12, 20])); t = 0
    lag = np.empty(n, dtype=np.int64)
    while t < n:
        dwell = int(rng.integers(4*h, 8*h+1))
        lag[t:min(n, t+dwell)] = current
        current = 32-current; t += dwell
    return lag


def generate_run(scenario, role, run_id, *, n=N, h=H):
    """Generate one run only. CONFIRM is deliberately refused, even in fixtures.

    Unit fixtures may shorten n/h explicitly; production config is fixed N/H.
    FIT/SELECT/CAL have only normal arrays and no event draw. DEV paired variants
    have the same full latent paths; no resetting or cross-run prefixes.
    """
    if role not in ROLES or scenario not in ('LIN', 'WEAK') or run_id < 0:
        raise ValueError('Only LIN/WEAK and current non-CONFIRM roles are permitted')
    if n < 105*h or h < 1:
        raise ValueError('Need room for prescribed event bounds and full windows')
    prefix = 2048+44
    streams = {s: _ar1(_rng(scenario, role, run_id, s), prefix+n) for s in ('u', 'z', 'v')}
    normal = normal_lag(scenario, role, run_id, n=n, h=h)
    tau = {'N': normal.copy()}
    start = end = None
    labels = {'N': np.zeros(n, dtype=np.int8)}
    if role == 'DEV':
        start = int(_rng(scenario, role, run_id, 'event').integers(24*h, 92*h+1))
        end = start+12*h
        tau['A'] = normal.copy(); tau['A'][start:end] += 24
        tau['SB'] = normal.copy(); tau['SB'][start:start+h] += 24; tau['SB'][end-h:end] += 24
        labels['A'] = np.zeros(n, np.int8); labels['A'][start:end] = 1
        labels['SB'] = np.zeros(n, np.int8); labels['SB'][start:start+h] = 1; labels['SB'][end-h:end] = 1
    a = .9 if scenario == 'LIN' else .3
    t = prefix+np.arange(n)
    arrays, irr = {}, {}
    for name, lag in tau.items():
        y = a*streams['u'][t-lag] + np.sqrt(1-a*a)*streams['z'][t]
        arrays[name] = np.column_stack((streams['u'][t], y))
        if scenario == 'WEAK':
            irr[name] = np.column_stack((streams['v'][t], y))
    if role == 'DEV':
        # Actual slice/rejoin sham; it must be byte-identical to N.
        for pool in [arrays] + ([irr] if irr else []):
            pool['S0'] = np.concatenate((pool['N'][:start], pool['N'][start:end], pool['N'][end:]))
            assert np.array_equal(pool['S0'], pool['N'])
        labels['S0'] = labels['N'].copy()
    dcf = a*(streams['u'][t-normal-24]-streams['u'][t-normal])
    truth = {'scenario': scenario, 'role': role, 'run_id': run_id, 'normal_lag': normal,
             'event_start': start, 'event_end': end, 'labels': labels,
             'seeds': {s: derive_seed(scenario, role, run_id, s) for s in ('u','z','v','lag','event')},
             'fit_counterfactual_delta': dcf if role == 'FIT' else None,
             'a': a, 'coupling_fraction': a*a, 'delta_over_H': 24/h}
    assert all(np.isfinite(v).all() for v in arrays.values())
    return arrays, irr, truth


@dataclass
class NormalRun:
    """Explicit source role; arrays only, no truth/event fields for the learner."""
    scenario: str
    role: str
    run_id: int
    values: np.ndarray


def check_normal_runs(runs, role):
    if not runs or any(r.role != role for r in runs):
        raise ValueError('Role-specific learner input required; DEV/CAL cannot select models')
    ids = [(r.scenario, r.run_id) for r in runs]
    if len(set(ids)) != len(ids) or len({r.scenario for r in runs}) != 1:
        raise ValueError('Run identities must be unique within one observation scenario')
    for r in runs:
        if r.values.ndim != 2 or r.values.shape[1] != 2 or not np.isfinite(r.values).all():
            raise ValueError('Finite two-channel run required')


def fit_scaler(fit_runs):
    check_normal_runs(fit_runs, 'FIT')
    values = np.concatenate([r.values for r in fit_runs])
    mean, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('FIT scales must be finite and positive')
    return {'mean': mean, 'scale': scale, 'ddof': 0,
            'fit_ids': [(r.scenario, r.role, r.run_id) for r in fit_runs]}


def standardize(values, scaler):
    return (values-scaler['mean'])/scaler['scale']


def feature_indices(model, target, times, *, peak=None):
    """Return channel/time arrays; target(t) never appears among its predictors."""
    if model not in MODELS or target not in (0, 1):
        raise ValueError('Unknown fixed model/target')
    times = np.asarray(times, dtype=np.int64)
    order = 64 if model.endswith('64') else 191
    own = np.arange(1, order+1)
    if model.startswith('AR'):
        other = np.array([], dtype=np.int64)
    elif model.startswith('SYNC'):
        other = np.array([0])
    elif model.startswith('PEAK'):
        if peak is None or not 1 <= peak <= 191:
            raise ValueError('PEAK lag must be fixed from FIT only')
        other = np.array([peak])
    else:
        other = np.arange(order+1)
    lag = np.r_[own, other]
    channel = np.r_[np.full(len(own), target), np.full(len(other), 1-target)]
    index = times[:, None]-lag[None, :]
    if np.any(index < 0) or np.any(index > times[:, None]):
        raise ValueError('Illegal history footprint')
    assert not np.any((channel == target) & (lag == 0))
    return index, channel


def design(values, model, target, times, *, peak=None):
    times = np.asarray(times, dtype=np.int64)
    if np.any(times >= len(values)):
        raise ValueError('Requested target outside this run')
    ix, channel = feature_indices(model, target, times, peak=peak)
    return values[ix, channel[None, :]], values[times, target]


def select_peak(fit_runs, scaler, target):
    """Pearson on exactly shared t>=191 endpoints; first exact tie -> lower lag."""
    check_normal_runs(fit_runs, 'FIT')
    yy, xx = [], []
    for r in fit_runs:
        a = standardize(r.values, scaler); t = np.arange(191, len(a))
        yy.append(a[t, target]); xx.append(a[t[:, None]-np.arange(1, 192), 1-target])
    y, x = np.concatenate(yy), np.concatenate(xx)
    yc = y-y.mean(); xc = x-x.mean(axis=0)
    den = np.sqrt((xc*xc).sum(axis=0)*np.dot(yc, yc))
    correlation = (xc.T@yc)/den
    if not np.isfinite(correlation).all():
        raise ValueError('Nonfinite FIT correlation')
    return int(np.argmax(np.abs(correlation))+1), correlation


def ridge_candidates(x, y, alphas=ALPHAS):
    """Centered float64 Gram/eigh; sum squared errors + alpha||beta||2.

    Each direction shares one Gram eigendecomposition across all five alpha.
    Intercept is explicitly unpenalized. Small tests compare sklearn svd Ridge.
    """
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite design and scalar target required')
    if any(a <= 0 or not np.isfinite(a) for a in alphas):
        raise ValueError('Positive alpha required')
    xm, ym = x.mean(axis=0), y.mean()
    xc, yc = x-xm, y-ym
    eigen, vectors = np.linalg.eigh(xc.T@xc)
    if eigen.min() < -1e-8*max(1., eigen.max()):
        raise ValueError('Gram lost positive semidefinite property')
    eigen = np.maximum(eigen, 0.)
    rhs = vectors.T@(xc.T@yc)
    result = {}
    for alpha in alphas:
        coef = vectors@(rhs/(eigen+alpha))
        result[float(alpha)] = {'coef': coef, 'intercept': float(ym-xm@coef)}
    return result


def choose_alpha(select_mse):
    """No labels, CAL, DEV or per-readout selection; larger alpha wins exact tie."""
    if set(select_mse) != set(ALPHAS) or not np.isfinite(list(select_mse.values())).all():
        raise ValueError('All five fixed SELECT losses required')
    return min(sorted(select_mse, reverse=True), key=lambda a: select_mse[a])


def fit_model(fit_runs, select_runs, scaler, model):
    check_normal_runs(fit_runs, 'FIT'); check_normal_runs(select_runs, 'SELECT')
    if fit_runs[0].scenario != select_runs[0].scenario:
        raise ValueError('Observation scenarios cannot share fitted model')
    if scaler['fit_ids'] != [(r.scenario, r.role, r.run_id) for r in fit_runs]:
        raise ValueError('Scaler must belong to this exact FIT pool')
    directions, peaks, curves = [], [], []
    candidate_losses = {a: [] for a in ALPHAS}
    for target in (0, 1):
        peak, curve = select_peak(fit_runs, scaler, target) if model == 'PEAK191' else (None, None)
        peaks.append(peak); curves.append(curve)
        fit_x, fit_y, select_x, select_y = [], [], [], []
        for pool, role, dst_x, dst_y in [(fit_runs, 'FIT', fit_x, fit_y),
                                        (select_runs, 'SELECT', select_x, select_y)]:
            for r in pool:
                a = standardize(r.values, scaler)
                t = np.arange(191, len(a)) if role == 'FIT' else np.arange(H-1, len(a), H)
                x, y = design(a, model, target, t, peak=peak)
                dst_x.append(x); dst_y.append(y)
        fits = ridge_candidates(np.concatenate(fit_x), np.concatenate(fit_y))
        sx, sy = np.concatenate(select_x), np.concatenate(select_y)
        for alpha, params in fits.items():
            candidate_losses[alpha].append(float(np.mean((sx@params['coef']+params['intercept']-sy)**2)))
        directions.append(fits)
    mse = {a: float(np.mean(loss)) for a, loss in candidate_losses.items()}
    alpha = choose_alpha(mse)
    return {'model': model, 'scenario': fit_runs[0].scenario, 'alpha': alpha, 'directions': [d[alpha] for d in directions],
            'peaks': peaks, 'peak_curves': curves, 'select_mse': mse,
            'solver': 'float64 centered Gram/eigh; unpenalized intercept; SSE+alpha L2',
            'fit_targets': sum(len(r.values)-191 for r in fit_runs),
            'select_targets': sum(len(r.values)//H for r in select_runs)}


def score_window(window, scaler, model):
    """One independent complete H window; no state/query-level stats."""
    if window.shape != (H, 2) or not np.isfinite(window).all():
        raise ValueError('One complete finite 192x2 window required')
    a = standardize(window, scaler)
    short = model['model'].endswith('64')
    t = np.arange(64, H) if short else np.array([H-1])
    predictions = np.empty((len(t), 2))
    for target, p in enumerate(model['directions']):
        x, _ = design(a, model['model'], target, t, peak=model['peaks'][target])
        predictions[:, target] = x@p['coef']+p['intercept']
    residual = np.abs(a[t]-predictions)
    result = {'times_in_window': t, 'prediction': predictions, 'target': a[t], 'residual': residual,
              'last': float(residual[-1].mean()), 'y_last': float(residual[-1, 1])}
    if short:
        result.update(mean128=float(residual.mean()), y_mean128=float(residual[:, 1].mean()))
    return result


def score_run(values, scaler, model):
    if len(values) % H:
        raise ValueError('Complete-window task requires N multiple of H')
    return [score_window(values[i:i+H], scaler, model) for i in range(0, len(values), H)]


def strata(normal_lag_path, start, end, h=H):
    n = len(normal_lag_path); endpoint = np.arange(h-1, n, h)
    left = endpoint-h+1
    interior = (left >= start+2*h) & (endpoint < end-2*h)
    far_normal = (endpoint < start-2*h) | (left >= end+2*h)
    # A switch is internal to the full input window: tau[j] != tau[j-1], j>left.
    switch = np.array([np.any(np.diff(normal_lag_path[l:t+1])) for l, t in zip(left, endpoint)])
    core = (interior | far_normal) & ~switch
    near = (endpoint >= start-2*h) & (left < end+2*h)
    return {'ALL': np.ones(len(endpoint), bool), 'CORE': core, 'NORMAL_SWITCH': switch,
            'NORMAL_SWITCH_NORMAL': switch & ~((endpoint >= start) & (endpoint < end)),
            'NORMAL_SWITCH_EVENT': switch & (endpoint >= start) & (endpoint < end),
            'EDGE': near & ~core, 'OTHER_GUARD': ~near & ~core,
            'endpoint': endpoint, 'input_start': left}


def assert_sham_equivalence(normal, sham, s0, normal_lag_path, start, end, scaler, models):
    """CORE full raw/scaled/design/target/prediction/residual equality, not labels only."""
    assert np.array_equal(normal, s0)
    layer = strata(normal_lag_path, start, end)
    count = 0
    for k in np.flatnonzero(layer['CORE']):
        lo, hi = k*H, (k+1)*H
        n, b = normal[lo:hi], sham[lo:hi]
        assert np.array_equal(n, b)
        ns, bs = standardize(n, scaler), standardize(b, scaler)
        assert np.array_equal(ns, bs)
        for model in models:
            t = np.arange(64, H) if model['model'].endswith('64') else np.array([H-1])
            for target in (0, 1):
                nx, ny = design(ns, model['model'], target, t, peak=model['peaks'][target])
                bx, by = design(bs, model['model'], target, t, peak=model['peaks'][target])
                assert np.array_equal(nx, bx) and np.array_equal(ny, by)
            p, q = score_window(n, scaler, model), score_window(b, scaler, model)
            for key in ('prediction', 'target', 'residual', 'last'):
                assert np.array_equal(p[key], q[key])
            if model['model'].endswith('64'):
                assert p['mean128'] == q['mean128']
        count += 1
    return count


def calibrate(cal_runs, scaler, model):
    check_normal_runs(cal_runs, 'CAL')
    if model.get('scenario', cal_runs[0].scenario) != cal_runs[0].scenario:
        raise ValueError('CAL pool must match observation scenario')
    pools = {k: [] for k in ('last', 'mean128') if k == 'last' or model['model'].endswith('64')}
    for r in cal_runs:
        for row in score_run(r.values, scaler, model):
            for key in pools:
                pools[key].append(row[key])
    result = {}
    for key, values in pools.items():
        values = np.asarray(values, np.float64)
        order = int(np.ceil((len(values)+1)*.99))
        result[key] = {'threshold': float(np.sort(values)[order-1]) if order <= len(values) else float('inf'),
                       'n': len(values), 'order_1based': order, 'scores': values, 'rule': 'score > threshold'}
    return result


def evaluate_readout(score, labels, layer, threshold, start, end, variant):
    """Fixed endpoint alarm rule; single-class AP NA; no threshold search/PA."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    score, labels = np.asarray(score), np.asarray(labels)
    t = layer['endpoint']; y = labels[t]
    if score.shape != t.shape or not np.isfinite(score).all() or np.isnan(threshold):
        raise ValueError('Finite endpoint scores and fixed CAL threshold required')
    alarm = score > threshold
    rows = []
    for name, mask in layer.items():
        if name in ('endpoint', 'input_start'):
            continue
        yy, ss = y[mask], score[mask]
        both = len(np.unique(yy)) == 2
        normal = mask & (y == 0)
        rows.append({'layer': name, 'points': int(mask.sum()), 'positive_points': int(yy.sum()),
                     'positive_fraction': float(yy.mean()) if len(yy) else None,
                     'AP': float(average_precision_score(yy, ss)) if both else None,
                     'ROC_AUC': float(roc_auc_score(yy, ss)) if both else None,
                     'NA_reason': None if both else 'EMPTY_OR_SINGLE_CLASS',
                     'normal_points': int(normal.sum()),
                     'normal_FPR': float(alarm[normal].mean()) if normal.any() else None})
    true_interval = (t >= start) & (t < end)
    # N/S0 retain a matched 12H interval, not an invented positive event.
    matched_hit = bool(alarm[true_interval].any()) if variant in ('N', 'S0') else None
    intervals = [(start, end)] if variant == 'A' else (
        [(start, start+H), (end-H, end)] if variant == 'SB' else [])
    events = []
    for lo, hi in intervals:
        hits = t[(t >= lo) & (t < hi) & alarm]
        events.append({'start': lo, 'end_exclusive': hi, 'hit': bool(len(hits)),
                       'first_alarm': int(hits[0]) if len(hits) else None,
                       'delay_points': int(hits[0]-lo) if len(hits) else None,
                       'right_censored': not bool(len(hits)), 'censoring_points': hi-lo})
    return {'layers': rows, 'events': events, 'matched_normal_interval_hit': matched_hit,
            'alarm': alarm, 'label_at_endpoint': y}


def paired_bootstrap(differences, *, repeats=10000):
    """One draw matrix shared by named comparisons; NA stays NA, not zero."""
    keys = list(differences)
    n = len(differences[keys[0]])
    if any(len(differences[k]) != n for k in keys) or not n:
        raise ValueError('Comparisons must retain the same independent run identities')
    rng = np.random.Generator(np.random.PCG64(derive_seed('CPU1', 'DEV', 0, 'bootstrap')))
    draws = rng.integers(0, n, size=(repeats, n))
    result = {}
    for key, values in differences.items():
        values = np.array([np.nan if x is None else x for x in values], dtype=np.float64)
        resampled = values[draws]
        counts = np.isfinite(resampled).sum(axis=1)
        means = np.divide(np.nansum(resampled, axis=1), counts,
                          out=np.full(repeats, np.nan), where=counts > 0)
        valid = np.isfinite(means)
        result[key] = {'run_differences': [float(v) if np.isfinite(v) else None for v in values],
                       'valid_pairs': int(np.isfinite(values).sum()),
                       'mean_difference': float(np.nanmean(values)) if np.isfinite(values).any() else None,
                       'CI95': np.percentile(means[valid], [2.5, 97.5]).tolist() if valid.any() else None,
                       'NA_bootstraps': int((~valid).sum()), 'repeats': repeats}
    return result, draws


def preview(config_path):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    fixed = {'H': H, 'N': N, 'roles': ROLES, 'models': list(MODELS), 'alphas': list(ALPHAS),
             'scenarios': ['LIN', 'WEAK', 'IRR'], 'bootstrap_repeats': 10000}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError('Config differs from fixed CPU first-wave contract')
    return {'status': 'PREVIEW_ONLY_NO_DATA_OR_FIT', 'config': config,
            'readouts': READOUTS, 'module_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'formal_execution': 'Not released; call reusable kernels only after root approval',
            'estimated_max_design_bytes': 97540*383*8,
            'estimated_design_plus_centered_bytes': 2*97540*383*8,
            'actual_cost': 'NOT_MEASURED_FOR_FORMAL_POOLS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(preview(args.config), indent=2))


if __name__ == '__main__':
    main()
