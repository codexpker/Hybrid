"""Elementwise robust SSL; true observed targets remain outside encoders."""


def elementwise(error, kind="half_mse"):
    import torch
    if kind == "half_mse":
        return .5*error.square()
    if kind == "huber":
        a = error.abs()
        return torch.where(a <= 1, .5*error.square(), a-.5)
    raise ValueError("Loss must be half_mse or Huber delta1")


def ssl_objective(forecast, reconstruction, batch, kind="half_mse", active_components=None):
    import torch
    observed = batch["reconstruction_observed"]
    y = torch.where(observed, batch["reconstruction_context"], 0)
    hidden = reconstruction["hidden_mask"] & observed[:, None]
    stats = reconstruction["stats"]
    z = (y[:, None]-stats["mean"])/stats["scale"]
    fstats = forecast["stats"]
    future = (torch.where(batch["future_observed"], batch["future_targets"], 0)-fstats["mean"])/fstats["scale"]
    tf = reconstruction["tf_normalized"]
    ms = reconstruction["ms_normalized_by_scale"]
    w = tf.shape[-2]
    tf_mask = hidden[:, :, None, -w:] & reconstruction["tf_valid"]
    ms_mask = hidden[:, :, None, None] & reconstruction["ms_valid_by_scale"]
    pred_mask = batch["future_observed"] & forecast["valid"]
    terms = {"tf": (tf-z[:, :, None, -w:], tf_mask),
             "ms": (ms-z[:, :, None, None], ms_mask),
             "pred": (forecast["prediction_normalized"]-future, pred_mask)}
    # Consistency only on the observed/artificially hidden overlapping GT positions.
    mean_ms = reconstruction["ms_normalized"][:, :, :, -w:]
    cmask = tf_mask & reconstruction["ms_valid"][:, :, :, -w:]
    terms["consistency"] = (tf-mean_ms, cmask)
    losses, counts, per_window = {}, {}, {}
    for key, (error, mask) in terms.items():
        mask = mask.expand_as(error)
        safe = torch.where(mask, error, torch.zeros_like(error))
        count = mask.flatten(1).sum(1)
        values = elementwise(safe, kind)*mask
        each = values.flatten(1).sum(1)/count.clamp_min(1)
        if key == "ms":
            scale_counts = mask.sum((1, 2, 4, 5))
            scale_losses = values.sum((1, 2, 4, 5))/scale_counts.clamp_min(1)
            each = (scale_losses*(scale_counts > 0)).sum(1)/(scale_counts > 0).sum(1).clamp_min(1)
        # Every scale is supervised; equal scale/copy/group dimensions have equal
        # valid support supplied by core, with explicit counts retained.
        losses[key] = each[count > 0].mean() if bool((count > 0).any()) else safe.sum()*0
        counts[key] = count
        per_window[key] = each
    from .scoring import recipe
    active, _ = recipe(active_components)
    keys = [key[2:] for key in active]
    valid = torch.stack([counts[key] > 0 for key in keys]).all(0)
    total = sum(losses[key] for key in keys)
    if "tf" in keys and "ms" in keys:
        total = total+.1*losses["consistency"]
    return {"total": total, "parts": losses, "valid_counts": counts,
            "per_window": per_window, "valid_windows": valid,
            "natural_missing_count": (~observed).sum(), "artificial_hidden_observed_count": hidden.sum(),
            "ms_scale_valid_counts": ms_mask.expand_as(ms).sum((0, 1, 2, 4, 5))}
