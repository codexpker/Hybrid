"""One frozen source-only, arrived-score paired margin; no inference changes."""
from contextlib import contextmanager
import math
import numpy as np

COMPONENTS = ('r_tf', 'r_ms', 'r_pred')
WEIGHTS = (.25, .25, .5)


@contextmanager
def auxiliary_evaluation(model):
    """Eval-mode with gradients; all module flags and used-device RNG restored."""
    import torch
    flags = [(module, module.training) for module in model.modules()]
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        for module, flag in flags:
            module.training = flag


def fit_standard_deviations(runs):
    """Per-run population std from FIT observed cells only; no role fallback."""
    if any(run.role != 'FIT' for run in runs):
        raise PermissionError('Only source FIT may fit amplitude units')
    result = {}
    for run in runs:
        if run.domain != 'SMD':
            continue
        scales = []
        for channel in range(run.values.shape[1]):
            values = run.values[run.observed[:, channel], channel].astype(np.float64)
            scales.append(float(np.std(values, ddof=0)) if len(values) >= 2 else 0.)
        result[(run.domain, run.entity, run.run)] = np.asarray(scales, dtype=np.float64)
    if not result:
        raise ValueError('Source SMD FIT required')
    return result


def unit_origins(fit_runs, fit_plan):
    """First16 unique original replay origins per SMD FIT run, no value search."""
    if any(run.role != 'FIT' for run in fit_runs) or any(e['role'] != 'FIT' for e in fit_plan):
        raise PermissionError('Reference units forbid SELECT/CAL/QUERY')
    keys = [(r.domain, r.entity, r.run) for r in fit_runs if r.domain == 'SMD']
    if len(keys) != 3 or len(set(keys)) != 3:
        raise ValueError('Exactly three accepted source SMD FIT runs')
    chosen = {key: [] for key in keys}
    for entry in fit_plan:
        key = tuple(entry[k] for k in ('domain', 'entity', 'run'))
        if key in chosen and len(chosen[key]) < 16 and entry['index'] not in [e['index'] for e in chosen[key]]:
            chosen[key].append(dict(entry))
    if any(len(entries) != 16 for entries in chosen.values()):
        raise ValueError('Each source FIT run must supply16 original unique origins')
    return [dict(entry, footprint=[entry['index']-512, entry['index']+1],
                 original_training_footprint=entry['footprint']) for key in keys for entry in chosen[key]]


def component_score(scalars, fit_units):
    """Exact original differentiable per-channel pooling, fixed log1p weights."""
    import torch
    if set(fit_units) != set(COMPONENTS):
        raise ValueError('Three detached FIT-only score units required')
    values = []
    for key, weight in zip(COMPONENTS, WEIGHTS):
        unit = fit_units[key]
        if not isinstance(unit, (int, float)) or not math.isfinite(unit) or unit < 1e-6:
            raise ValueError('Positive finite constant FIT unit')
        scalar = scalars[key]
        if not bool(torch.isfinite(scalar).all()) or bool((scalar < 0).any()):
            raise ValueError('Incomplete arrived residual components; no replacement sampling')
        values.append(weight * torch.log1p(scalar / unit))
    return sum(values)


def _arrived_arrays(batch):
    import torch
    return (torch.cat((batch['history'], batch['current_target'][:, None]), dim=1),
            torch.cat((batch['observed'], batch['current_observed'][:, None]), dim=1))


def _batch_from_arrived(batch, values, observed):
    import torch
    horizon = batch['future_targets'].shape[1]
    future = torch.zeros_like(batch['future_targets'])
    future_observed = torch.zeros_like(batch['future_observed'])
    future[:, 0] = values[:, -1]
    future_observed[:, 0] = observed[:, -1]
    result = {key: batch[key] for key in ('domain', 'entity', 'run', 'role', 'index', 'available_index')}
    result.update(history=values[:, :-1], observed=observed[:, :-1],
                  reconstruction_context=values[:, 1:], reconstruction_observed=observed[:, 1:],
                  current_target=values[:, -1], current_observed=observed[:, -1],
                  future_targets=future, future_observed=future_observed,
                  footprint=[batch['index']-values.shape[1]+1, batch['index']+1],
                  unarrived_future_is_padding=True)
    assert horizon == future.shape[1] and not bool(future_observed[:, 1:].any())
    return result


def make_pair(batch, fit_std, smd_update_index):
    """Frozen ramp/step proxies with identical current values and observed mask."""
    import torch
    if batch['domain'] != 'SMD' or batch['role'] != 'FIT':
        raise PermissionError('Synthetic training pair is SMD FIT only')
    if batch['history'].shape[0:2] != (1, 512) or batch['future_targets'].shape[1] != 32:
        raise ValueError('Original B1/H512/P32 required')
    j = int(smd_update_index)
    if j < 0 or j != smd_update_index:
        raise ValueError('Nonnegative deterministic SMD update counter')
    std = torch.as_tensor(fit_std, dtype=batch['history'].dtype, device=batch['history'].device)
    if std.shape != batch['current_target'].shape[1:] or not bool(torch.isfinite(std).all()) or bool((std < 0).any()):
        raise ValueError('FIT observed per-channel std mismatch')
    active = torch.nonzero((std > 0) & batch['current_observed'][0], as_tuple=False).flatten()
    if not len(active):
        raise ValueError('No observed nonconstant channel; do not invent a pair')
    count = int(math.ceil(len(active) / 4))
    chosen = active[(torch.arange(count, device=active.device) + j) % len(active)]
    length, amplitude = (16, .25) if j % 2 == 0 else (256, .5)
    sign = 1 if (j // 2) % 2 == 0 else -1
    values, observed = _arrived_arrays(batch)
    ramp = values.clone()
    step = values.clone()
    offset = sign * amplitude * std[chosen]
    coefficient = torch.arange(1, length+1, dtype=values.dtype, device=values.device) / length
    # Only truly observed cells change; all missingness remains byte-identical.
    support = observed[:, -length:, chosen]
    step[:, -length:, chosen] = torch.where(support, values[:, -length:, chosen] + offset, values[:, -length:, chosen])
    ramp[:, -length:, chosen] = torch.where(support, values[:, -length:, chosen] + coefficient[None, :, None] * offset, values[:, -length:, chosen])
    normal = _batch_from_arrived(batch, ramp, observed)
    abnormal = _batch_from_arrived(batch, step, observed)
    assert torch.equal(normal['current_target'], abnormal['current_target'])
    assert torch.equal(normal['current_observed'], abnormal['current_observed'])
    return normal, abnormal, dict(smd_update_index=j, family='weak' if length == 16 else 'persistent',
        length=length, amplitude_std=amplitude, sign=sign, channels=chosen.detach().cpu().tolist(),
        channel_cycle_offset=j % len(active), future_observed_slots=[0], current_GT_pair_exact=True,
        synthetic_semantics='fixed smooth-ramp/step proxy, not certified natural health/fault')


class PairObjective:
    """Training callback; source SELECT/CAL cannot enter this object's methods."""
    def __init__(self, fit_units, fit_std_by_run, *, seed=2025, lambda_weight=.1):
        if seed != 2025 or lambda_weight not in (0, .1):
            raise ValueError('One fixed seed/lambda; zero is a technical equivalence fixture')
        self.fit_units = dict(fit_units)
        self.fit_std_by_run = fit_std_by_run
        self.seed = seed
        self.lambda_weight = lambda_weight
        self.smd_updates = 0

    def __call__(self, *, model, batch, forecast, reconstruction, normal_objective, attempt):
        import torch
        if batch['role'] != 'FIT' or batch['domain'] not in ('SMD', 'Sunspot'):
            raise PermissionError('Only source training FIT may call auxiliary objective')
        if batch['domain'] == 'Sunspot' or self.lambda_weight == 0:
            return dict(loss=normal_objective['total'] * 0,
                        report=dict(enabled=False, reason='Sunspot_normalSSL_only' if batch['domain'] == 'Sunspot' else 'lambda0_technical_equivalence'))
        key = tuple(batch[k] for k in ('domain', 'entity', 'run'))
        normal, abnormal, meta = make_pair(batch, self.fit_std_by_run[key], self.smd_updates)
        from .training import passes
        from .scoring import residual_components
        with auxiliary_evaluation(model):
            fn, rn = passes(model, normal, self.seed + int(attempt))
            fa, ra = passes(model, abnormal, self.seed + int(attempt))
            sn = component_score(residual_components(fn, rn, normal)['scalar'], self.fit_units)
            sa = component_score(residual_components(fa, ra, abnormal)['scalar'], self.fit_units)
            hinge = torch.relu(.1 + sn - sa).mean()
            loss = self.lambda_weight * hinge
        self.smd_updates += 1
        report = dict(meta, enabled=True, source_attempt=int(attempt), actual_mask_seed=self.seed+int(attempt),
                      Snormal=float(sn.detach().mean()), Sabnormal=float(sa.detach().mean()),
                      hinge=float(hinge.detach()), weighted_aux_loss=float(loss.detach()),
                      normal_current_values_not_future=True, additional_model_passes=4,
                      gradient_enabled=torch.is_grad_enabled())
        return dict(loss=loss, report=report)
