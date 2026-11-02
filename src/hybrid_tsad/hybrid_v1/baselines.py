"""Untrained, history-only h1 controls; NumPy only and no label/file I/O.

Default production history is512. Reductions use explicit float64, population
variance +1e-5, not a claim of bit-identical Torch reduction order.
"""
import numpy as np

from .data import arrived_targets, resolve_plan
from .scoring import fit_calibration

METHODS = ("persistence", "history_mean512")


def baseline_record(run, target, *, method, history=512):
    """Score one original arrived target using only [t-H,t) and current GT.

    A missing t-1 is never replaced by an older value. Current observations do
    not participate in predictions, history statistics or channel eligibility
    until the final residual mask. Missing GT never becomes an imputed zero.
    """
    if method not in METHODS or type(history) is not int or history < 2:
        raise ValueError("Known baseline and integer history>=2 required")
    values, observed = np.asarray(run.values), np.asarray(run.observed)
    if (values.ndim != 2 or values.shape[1] < 1 or observed.shape != values.shape
            or observed.dtype != np.bool_):
        raise ValueError("Two-dimensional values and matching boolean observed mask required")
    if not isinstance(target, (int, np.integer)) or target < history or target >= len(values):
        raise ValueError("Arrived H+1 footprint must stay within this role/run")
    target = int(target)
    past = values[target-history:target].astype(np.float64)
    visible = observed[target-history:target]
    current = values[target].astype(np.float64)
    current_observed = observed[target]
    if not np.isfinite(past[visible]).all() or not np.isfinite(current[current_observed]).all():
        raise ValueError("Visible past/current sensor values must be finite")
    safe = np.where(visible, past, 0.)
    count = visible.sum(axis=0)
    mean = safe.sum(axis=0)/np.maximum(count, 1)
    centered = np.where(visible, safe-mean, 0.)
    variance = np.square(centered).sum(axis=0)/np.maximum(count, 1)
    scale = np.sqrt(variance+1e-5)
    history_valid = count >= 2
    prediction_valid = history_valid.copy()
    if method == "persistence":
        prediction_valid &= visible[-1]
        prediction = np.where(prediction_valid, safe[-1], np.nan)
    else:
        prediction = np.where(prediction_valid, mean, np.nan)
    valid = prediction_valid & current_observed
    # Do not evaluate NaN arithmetic in invalid channels and then mask it away.
    residual = np.full(values.shape[1], np.nan, dtype=np.float64)
    residual[valid] = np.square((prediction[valid]-current[valid])/scale[valid])
    valid_count = int(valid.sum())
    index = int(run.start)+target
    return dict(method=method, domain=run.domain, entity=run.entity, run=run.run, role=run.role,
        index=index, available_index=index, forecast_issued_index=index-1,
        history=history, scored_horizon=1, reduction_dtype="float64",
        footprint=[index-history,index+1], forecast_history_bounds=[index-history,index],
        prediction=prediction[None].tolist(), prediction_valid=prediction_valid.tolist(),
        current_target=current.tolist(), current_observed=current_observed.tolist(),
        history_observed_count=count.tolist(), history_mean=mean.tolist(),
        history_variance=variance.tolist(), history_scale=scale.tolist(),
        history_valid=history_valid.tolist(), valid_channels=valid.tolist(),
        valid_channel_count=valid_count, total_channels=values.shape[1],
        per_channel={"r_pred":residual.tolist()},
        r_pred=float(residual[valid].mean()) if valid_count else float("nan"),
        valid=bool(valid_count), reason=None if valid_count else "NO_VALID_OBSERVED_CURRENT_CHANNELS",
        active_components=["r_pred"])


def iter_baselines(runs, *, method, history=512, index_plan=None):
    """Yield complete arrived coverage or the exact frozen source CAL positions.

    No training, query fit, P-1 tail exclusion, backward fill or window crossing.
    Original fixed-plan validation is reused; this function does not choose rows.
    """
    if method not in METHODS or type(history) is not int or history < 2:
        raise ValueError("Known baseline and integer history>=2 required")
    runs = list(runs)
    groups = ([(run,[t]) for run,t in resolve_plan(runs,index_plan,history,1,arrived=True)]
              if index_plan is not None else [(run,arrived_targets(run,history)) for run in runs])
    for run, targets in groups:
        for target in targets:
            yield baseline_record(run,int(target),method=method,history=history)


def fit_source_baseline_calibration(records, *, expected_domains):
    """Permission guard only; reuse the unchanged source r_pred scoring recipe.

    Caller binds the original source CAL manifest/positions and asset identities.
    This helper cannot authenticate a manifest or infer source-vs-target from names.
    """
    records = list(records)
    expected = set(expected_domains)
    if (not expected or not records or any(r.get("role") != "CAL" for r in records)
            or {r["domain"] for r in records} != expected
            or len({r.get("method") for r in records}) != 1
            or records[0].get("method") not in METHODS):
        raise ValueError("One named baseline and only bound source CAL domains/roles required")
    return fit_calibration(records,source=True,active_components=["r_pred"])
