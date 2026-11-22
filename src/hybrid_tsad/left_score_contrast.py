"""Source-only probe of the deployed LEFT score and Adam displacement control."""

import math

import numpy as np
import torch

from hybrid_tsad.left_source_preservation import preserving_eval


WINDOW = 192
CORE = slice(48, 144)
SIGNS = ((1, 1), (-1, -1), (1, -1), (-1, 1))
ANGLE = math.pi / 12
ANCHORS = 8


def rotation_basis():
    positions = np.arange(CORE.stop - CORE.start, dtype=np.float64)
    taper = np.sin(np.pi * positions / (len(positions) - 1)) ** 2
    taper[[0, -1]] = 0.0
    phase = 2 * np.pi * 4 * positions / len(positions)
    carriers = np.stack((np.cos(phase), np.sin(phase)), axis=1)
    carriers -= np.average(carriers, axis=0, weights=taper)
    q, r = np.linalg.qr(taper[:, None] * carriers)
    basis = np.zeros((WINDOW, 2), dtype=np.float64)
    basis[CORE] = q * np.sign(np.diag(r))
    return basis


def source_windows(values):
    count = len(values) // WINDOW
    return values[:count * WINDOW].reshape(count, WINDOW, values.shape[1])


def select_pair(fit):
    coefficients = np.einsum("tk,btc->bkc", rotation_basis(), source_windows(fit))
    gram = np.einsum("bki,bkj->ij", coefficients, coefficients)
    denominators = np.sqrt(np.outer(gram.diagonal(), gram.diagonal()))
    pairs = np.triu_indices(fit.shape[1], k=1)
    valid = denominators[pairs] > 0
    strengths = np.full(len(valid), -np.inf)
    strengths[valid] = np.abs(gram[pairs][valid] / denominators[pairs][valid])
    winner = int(np.argmax(strengths))
    if not valid[winner] or strengths[winner] == 0:
        raise ValueError("No SOURCE_FIT pair has nonzero projected association")
    first, second = (int(axis[winner]) for axis in pairs)
    return (first, second), float(gram[first, second] / denominators[first, second])


def probe_windows(source, pair):
    """Return [four arms, anchors, time, channels] in the fixed order SIGNS."""
    if source.ndim != 3 or source.shape[1] != WINDOW or not 0 <= pair[0] < pair[1] < source.shape[2]:
        raise ValueError("Invalid native window or channel pair")
    basis = torch.as_tensor(rotation_basis(), device=source.device, dtype=source.dtype)
    original = source[:, :, list(pair)]
    coefficients = torch.einsum("tk,btc->bck", basis, original)
    cosine, sine = math.cos(ANGLE), math.sin(ANGLE)
    signs = source.new_tensor(SIGNS)
    first, second = coefficients.unbind(dim=-1)
    delta = torch.stack(((cosine - 1) * first - sine * signs[:, None, :] * second,
                         sine * signs[:, None, :] * first + (cosine - 1) * second), dim=-1)
    arms = source.unsqueeze(0).expand(len(SIGNS), -1, -1, -1).clone()
    arms[:, :, :, list(pair)] = original + torch.einsum("tk,abck->abtc", basis, delta)
    return arms


def native_score(model, windows):
    """Native infer body, with the caller deciding whether gradients are enabled."""
    ms_gt, ms_pred, lengths, from_freq, from_time, inputs, _, _, _, from_ms, cache = model._forward(
        windows, None, None, None
    )
    return model._ms_anomaly_score(ms_pred, ms_gt, lengths, from_freq, from_time, inputs, from_ms, cache)


def contrast_parts(scores):
    same = (scores[0] + scores[1]) / 2
    opposite = (scores[2] + scores[3]) / 2
    return same, opposite, opposite - same


def probe_scores(model, source, pair, *, grad):
    arms = probe_windows(source, pair)
    with preserving_eval(model), torch.set_grad_enabled(grad):
        per_channel = native_score(model, arms.flatten(0, 1))
        # First reduce channels as in the project's deployed native readout.
        core = per_channel[:, CORE].mean(dim=-1).mean(dim=-1)
    return core.reshape(len(SIGNS), len(source))


def reference_gradient(model, source, pair, parameters):
    same, opposite, contrast = contrast_parts(probe_scores(model, source, pair, grad=True))
    gradients = torch.autograd.grad(contrast.mean(), parameters, allow_unused=True)
    summary = torch.stack((same.mean(), opposite.mean(), contrast.mean())).detach().cpu().tolist()
    if not all(math.isfinite(v) for v in summary):
        raise FloatingPointError("Nonfinite native source contrast")
    return gradients, dict(zip(("same_score", "opposite_score", "contrast"), summary))


def dot64(first, second):
    terms = [torch.dot(a.reshape(-1).double(), b.reshape(-1).double())
             for a, b in zip(first, second) if a is not None and b is not None]
    return torch.stack(terms).sum().item() if terms else 0.0


@torch.no_grad()
def adam_displacement(optimizer, parameters, *, normal=None, ratio=None):
    """Advance Adam once, then optionally change the actual parameter displacement.

    Adam's moments are those of the native training gradient. Only the applied
    parameter increment can change. A no-op does not rewrite the parameters.
    """
    if (normal is None) == (ratio is None):
        raise ValueError("Choose exactly one displacement policy")
    if ratio is not None and not 0 <= ratio <= 1:
        raise ValueError("Invalid frozen relative step length")
    before = tuple(p.detach().clone() for p in parameters)
    optimizer.step()
    proposal = tuple(p - original for p, original in zip(parameters, before))
    proposal_sq = dot64(proposal, proposal)
    if not math.isfinite(proposal_sq):
        raise FloatingPointError("Nonfinite Adam proposal")
    if ratio is not None:
        if ratio != 1:
            for parameter, original, delta in zip(parameters, before, proposal):
                parameter.copy_(original + ratio * delta)
        applied = tuple(p - original for p, original in zip(parameters, before))
        return {"proposal_l2": math.sqrt(proposal_sq), "applied_l2": math.sqrt(dot64(applied, applied)),
                "replayed_length_ratio": ratio}

    before_dot = dot64(proposal, normal)
    normal_sq = dot64(normal, normal)
    if not math.isfinite(before_dot) or not math.isfinite(normal_sq):
        raise FloatingPointError("Nonfinite reference gradient")
    triggered = before_dot < 0
    if triggered:
        correction = -before_dot / normal_sq
        for parameter, gradient in zip(parameters, normal):
            if gradient is not None:
                parameter.add_(gradient, alpha=correction)
    applied = tuple(p - original for p, original in zip(parameters, before)) if triggered else proposal
    applied_sq = dot64(applied, applied) if triggered else proposal_sq
    applied_dot = dot64(applied, normal) if triggered else before_dot
    # Replay the length of the real stored update, with a cap at unity for rounding.
    measured = math.sqrt(applied_sq / proposal_sq) if proposal_sq else 1.0
    return {"proposal_l2": math.sqrt(proposal_sq), "applied_l2": math.sqrt(applied_sq),
            "dot_proposal": before_dot, "dot_applied": applied_dot, "reference_l2": math.sqrt(normal_sq),
            "triggered": triggered, "measured_ratio": measured, "length_ratio": min(1.0, measured)}
