"""Fixed, source-domain-balanced calibration. No test fitting or fallback zeros."""
import numpy as np

COMPONENTS = ("r_tf", "r_ms", "r_pred")
WEIGHTS = np.array([.25, .25, .5])


def recipe(active_components=None):
    active = tuple(COMPONENTS if active_components is None else active_components)
    if not active or len(set(active)) != len(active) or not set(active) <= set(COMPONENTS):
        raise ValueError("Invalid fixed active-component recipe")
    rec = [k for k in active if k != "r_pred"]
    weights = {k: ((.5 if "r_pred" in active else 1.)/len(rec)) for k in rec}
    if "r_pred" in active:
        weights["r_pred"] = .5 if rec else 1.
    return active, weights


def valid_error(record, key):
    value = record.get(key)
    return value is not None and np.isfinite(value) and value >= 0


def weighted_quantile(values, weights, q):
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]); cumulative /= cumulative[-1]
    return float(values[order[min(np.searchsorted(cumulative, q, side="left"), len(order)-1)]])


def fit_calibration(records, *, source=True, active_components=None):
    # A record is one window, not one channel. Each domain contributes total1/D.
    active, component_weights = recipe(active_components)
    complete = [r for r in records if all(valid_error(r, k) for k in active)]
    expected_domains = sorted({r["domain"] for r in records})
    if source:
        unsupported = [d for d in expected_domains if not any(r["domain"] == d for r in complete)]
        if unsupported:
            raise ValueError("Source calibration domains lack valid support: " + ",".join(unsupported))
    if not complete:
        raise ValueError("No valid calibration windows")
    domains = sorted({r["domain"] for r in complete})
    from collections import Counter
    counts = Counter(r["domain"] for r in complete)
    weights = np.array([1/(len(domains)*counts[r["domain"]])
                        if source else 1/len(complete) for r in complete])
    scales = {k: max(weighted_quantile([r[k] for r in complete], weights, .5), 1e-6) for k in active}
    scores = [sum(component_weights[k]*np.log1p(r[k]/scales[k]) for k in active) for r in complete]
    return {"scales": scales, "threshold": weighted_quantile(scores, weights, .99),
            "source_domain_equal_weight": source, "domains": domains, "valid_windows": len(complete),
            "quantile_rule": "weighted inverse empirical CDF; first CDF>=q; strict score>threshold",
            "active_components": list(active), "weights": component_weights}


def apply_calibration(record, calibration):
    active, weights = recipe(calibration.get("active_components"))
    if calibration["weights"] != weights:
        raise ValueError("Calibration weights differ from the fixed registered recipe")
    if not all(valid_error(record, k) for k in active):
        return {"score": float("nan"), "valid": False, "reason": "MISSING_REQUIRED_COMPONENT"}
    score = sum(weights[k]*np.log1p(record[k]/calibration["scales"][k]) for k in active)
    return {"score": float(score), "valid": True, "alarm": bool(score > calibration["threshold"])}


def residual_components(forecast, reconstruction, batch, active_components=None):
    """Torch differentiable per-channel arrived-point errors; no query statistics."""
    import torch
    gt = batch["current_target"]
    obs = batch["current_observed"]
    hidden = reconstruction["hidden_mask"][:, :, -1] & obs[:, None]
    def average(error, valid, axes):
        count = valid.expand_as(error).sum(axes)
        value = torch.where(valid, error, torch.zeros_like(error)).sum(axes)/count.clamp_min(1)
        return torch.where(count > 0, value, torch.full_like(value, float("nan")))
    def squared_error(prediction, target, scale, valid):
        # Mask operands BEFORE subtraction/division/square. Masking NaN squared
        # errors afterwards gives 0*NaN in backward through event features.
        safe_prediction = torch.where(valid, prediction, torch.zeros_like(prediction))
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        safe_scale = torch.where(valid, scale, torch.ones_like(scale))
        return ((safe_prediction-safe_target)/safe_scale).square()
    scale = reconstruction["stats"]["scale"][:, :, 0]
    tf = reconstruction["tf_reconstruction"][:, :, :, -1]
    valid_tf = reconstruction["tf_valid"][:, :, :, -1] & hidden[:, :, None]
    rtf = average(squared_error(tf, gt[:, None, None], scale[:, :, None], valid_tf), valid_tf, (1, 2))
    ms = reconstruction["ms_reconstruction_by_scale"][:, :, :, :, -1]
    vm = reconstruction["ms_valid_by_scale"][:, :, :, :, -1] & hidden[:, :, None, None]
    each_scale = average(squared_error(ms, gt[:, None, None, None], scale[:, :, None, None], vm), vm, (1, 2))
    rms = average(each_scale, torch.isfinite(each_scale), (1,))
    variance = torch.where(valid_tf, tf, 0).var(2, unbiased=False)/scale.square()
    rv = average(variance, valid_tf.all(2), (1,))
    pv = forecast["valid"][:, 0] & obs
    rp = squared_error(forecast["prediction"][:, 0], gt, forecast["stats"]["scale"][:, 0], pv)
    rp = torch.where(pv, rp, torch.full_like(rp, float("nan")))
    channel = {"r_tf": rtf, "r_ms": rms, "r_var": rv, "r_pred": rp}
    scalar = {}
    for key, value in channel.items():
        valid = torch.isfinite(value)
        scalar[key] = torch.where(valid.any(1), torch.where(valid, value, 0).sum(1)/valid.sum(1).clamp_min(1),
                                  torch.full((len(value),), float("nan"), device=value.device))
    active, _ = recipe(active_components)
    channel_valid = torch.stack([torch.isfinite(channel[k]) for k in active]).all(0)
    def pool(value):
        summed = torch.where(channel_valid[..., None], value, torch.zeros_like(value)).sum(1)
        return torch.where(channel_valid.any(1)[:, None], summed/channel_valid.sum(1).clamp_min(1)[:, None],
                           torch.full_like(summed, float("nan")))
    features = torch.cat((pool(reconstruction["endpoint_t_features"]),
                          pool(reconstruction["endpoint_f_features"]),
                          torch.stack([torch.log1p(scalar[k]) if k in active else torch.zeros_like(scalar[k]) for k in COMPONENTS], -1)), -1)
    return {"scalar": scalar, "per_channel": channel, "event_features": features,
            "prediction": forecast["prediction"], "index": batch["index"],
            "available_index": batch["available_index"]}
