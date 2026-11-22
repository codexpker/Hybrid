"""Source event-phase supervision applied directly to native TimeRadar variance."""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.nn import functional as F

from .native_v_conditioning import fixed_native_rng


WINDOW = 100
PHASE_COUNT = 3


@dataclass(frozen=True)
class SourceEventRole:
    bounds: tuple[int, int]
    events: tuple[tuple[int, int], ...]
    positive_phases: tuple[tuple[np.ndarray, ...], ...]
    normal_endpoints: np.ndarray


def _role(labels, origin, bounds, events):
    low, high = bounds
    prefix = np.concatenate(([0], np.cumsum(labels[low - origin:high - origin], dtype=np.int64)))
    event_count = prefix[WINDOW:] - prefix[:-WINDOW]
    normal = np.flatnonzero(event_count == 0) + low + WINDOW - 1
    if not len(normal):
        raise ValueError("Each source role requires a complete label-normal window")
    phases = tuple(tuple(np.array_split(np.arange(start + WINDOW - 1, stop, dtype=np.int64), PHASE_COUNT))
                   for start, stop in events)
    if any(not len(phase) for event in phases for phase in event):
        raise ValueError("Event needs a complete 100-point footprint in each of its three endpoint phases")
    return SourceEventRole(bounds, events, phases, normal)


def source_event_roles(labels, origin, *, fit_event_count=5):
    """Split a bounded source-label slice in the normal gap between complete events.

    The input represents one source-authorized record slice, in original order.
    A positive endpoint sees only its own complete annotated event. A normal
    endpoint sees only labeled-normal points from the same chronological role.
    """
    points = np.asarray(labels)
    if (points.ndim != 1 or len(points) < WINDOW or not np.isin(points, [0, 1]).all()
            or points[0] or points[-1]):
        raise ValueError("Require binary source labels with complete, internally bounded events")
    edges = np.flatnonzero(np.diff(points.astype(np.int8), prepend=0, append=0)) + origin
    events = tuple((int(lo), int(hi)) for lo, hi in edges.reshape(-1, 2))
    if not 0 < fit_event_count < len(events):
        raise ValueError("Both source fit and select must contain complete labeled events")
    before, after = events[:fit_event_count], events[fit_event_count:]
    cut = (before[-1][1] + after[0][0]) // 2
    return (_role(points, origin, (origin, cut), before),
            _role(points, origin, (cut, origin + len(points)), after))


def native_endpoint_and_copy_loss(model, window):
    """One native B1 pass. Returned variance retains the decoder gradient."""
    if window.ndim != 3 or window.shape[:2] != (1, WINDOW) or window.shape[2] == 0:
        raise ValueError("Native criterion requires one observed 100-point window")
    if any(module.training for module in model.modules()):
        raise ValueError("Keep native TimeRadar in eval mode; decoder gradients are still enabled")
    with fixed_native_rng(window.device):
        _, copies = model(window, norm=0, mask_mode="c", copies=10)
    if copies.shape != (10, *window.shape):
        raise ValueError("Native ten-copy shape changed")
    copy_loss = F.mse_loss(copies, window.unsqueeze(0).expand_as(copies))
    score = model.cal_anomaly_score(window, copies, anomaly_criterion=None, L=1)[0, WINDOW - 1]
    return score, copy_loss


def event_pair_loss(positive, normal, reference_scale, *, phase_sizes=None):
    """Paired native score margin, uniformly over events and phases by default.

    The labeled control uses the identical scores and positive exposure, with
    stratum weights proportional to the number of legal positive endpoints.
    """
    if positive.ndim != 2 or positive.shape != normal.shape or positive.shape[1] != PHASE_COUNT or not positive.shape[0]:
        raise ValueError("Require paired positive and normal native scores for each event x 3 phases")
    if not math.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("Require a positive, fixed nominal-source variance scale")
    gaps = torch.log1p(positive / reference_scale) - torch.log1p(normal / reference_scale)
    margin = F.softplus(1 - gaps)
    if phase_sizes is None:
        return margin.mean()
    weights = torch.as_tensor(phase_sizes, dtype=margin.dtype, device=margin.device)
    if weights.shape != margin.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Require positive legal-endpoint counts for every event and phase")
    return (weights * margin).sum() / weights.sum()
