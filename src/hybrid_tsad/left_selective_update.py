"""Choose which window losses enter a native LEFT update.

The original full batch still drives LEFT's forward and prototype updates.
The pinned LEFT forward exposes all reconstruction tensors except the two
spectrograms, so capture those tensors from their original transform calls.
"""

import torch
from torch.nn import functional as F


def _window_mean(loss):
    return loss.flatten(1).mean(dim=1)


def selective_loss(model, windows, mode, *, generator=None, keep=None):
    """Return the differentiable update loss, retained batch indices and window losses."""
    if mode not in {"trim", "random"}:
        raise ValueError("Expected trim or random")
    batch = len(windows)
    keep = (4 * batch + 4) // 5 if keep is None else keep
    if not 0 < keep <= batch:
        raise ValueError("Invalid retained window count")
    if mode == "random" and (generator is None or generator.device.type != "cpu"):
        raise ValueError("Random retention needs a dedicated CPU generator")

    spectra = []

    def capture_spectrum(_module, _inputs, output):
        spectra.append(output)

    handle = model.tf.register_forward_hook(capture_spectrum)
    try:
        target_ms, decoded_ms, lengths, from_freq, _, observed, _, _, _, full_ms, _ = (
            model._forward(windows, None, None, None)
        )
    finally:
        handle.remove()
    if len(spectra) != 2:
        raise RuntimeError("Pinned LEFT must emit input and reconstructed spectra once each")
    input_spectrum, decoded_spectrum = spectra

    ms_error = F.smooth_l1_loss(decoded_ms, target_ms, reduction="none")
    segments = torch.split(ms_error, lengths[1:], dim=1)
    weights = getattr(model, "ms_seg_weights", None)
    if weights is None:
        weights = [1.0] * len(segments)
    weights = ms_error.new_tensor(weights)
    weights = weights / (weights.sum() + 1e-12)
    per_window_ms = torch.stack([_window_mean(segment) for segment in segments], dim=1)
    per_window_ms = (per_window_ms * weights).sum(dim=1)
    per_window_cycle = _window_mean(
        F.smooth_l1_loss(decoded_spectrum, input_spectrum, reduction="none")
    ) + _window_mean(F.smooth_l1_loss(from_freq, observed, reduction="none"))
    per_window_consistency = _window_mean(
        F.smooth_l1_loss(full_ms, from_freq.detach(), reduction="none")
    )
    per_window = (
        model.alpha_cycle_l * per_window_cycle
        + model.alpha_ms_l * per_window_ms
        + model.alpha_cons_l * per_window_consistency
    )

    if keep == batch:
        indices = torch.arange(batch, device=windows.device)
    elif mode == "trim":
        # The earlier batch slot wins exact ties. Sorting never enters autograd.
        indices = torch.argsort(per_window.detach(), stable=True)[:keep]
    else:
        indices = torch.randperm(batch, generator=generator, device="cpu")[:keep]
        indices = indices.to(windows.device)

    loss = per_window[indices].mean()
    band_weight = float(getattr(model, "alpha_band_reg", 0.0))
    if band_weight:
        # This pure, parameter-only term stays at its native batch-independent scale.
        loss = loss + model.alpha_ms_l * band_weight * model.ms_utils.band_regularizer()
    return loss, indices, per_window
