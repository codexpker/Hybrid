"""Endpoint-aligned revision; archived shared-cycle training stays unchanged.

This opt-in factory does not change the historical factory or its scores.
Evidence is computed in FIT-standardized units, retaining sample/channel axes.
"""
import torch
from torch.nn import functional as F

from .left_predictive_cycle import predictive_cycle_class


def terminal_evidence(time, frequency, multiscale, target, *,
                      alpha_cycle, alpha_ms, alpha_cons, channels=None):
    """Return weighted per-channel evidence and unreduced SmoothL1 components.

    Inputs are terminal tensors [..., C], never windows. SmoothL1 beta=1
    and the detached consistency target exactly match the archived _terminal.
    Aggregate only the last (selected channel) axis to obtain point scores.
    """
    if target.ndim < 2 or any(x.shape != target.shape for x in (time, frequency, multiscale)):
        raise ValueError('Matching sample/channel terminal tensors required')
    if not all(bool(torch.isfinite(x).all()) for x in (time, frequency, multiscale, target)):
        raise ValueError('Nonfinite terminal evidence input')
    if channels is not None:
        if not channels or len(set(channels)) != len(channels):
            raise ValueError('Nonempty unique supervised channels required')
        time, frequency, multiscale, target = [x[..., channels] for x in (time, frequency, multiscale, target)]
    parts = dict(time=F.smooth_l1_loss(time, target, reduction='none'),
                 frequency=F.smooth_l1_loss(frequency, target, reduction='none'),
                 multiscale=F.smooth_l1_loss(multiscale, target, reduction='none'),
                 consistency=F.smooth_l1_loss(multiscale, frequency.detach(), reduction='none'))
    evidence = (alpha_cycle * (parts['time'] + parts['frequency'])
                + alpha_ms * parts['multiscale'] + alpha_cons * parts['consistency'])
    return evidence, parts


def terminal_aligned_cycle_class(native_class, mode):
    archived = predictive_cycle_class(native_class, mode)

    class TerminalAlignedLEFT(archived):
        @torch.no_grad()
        def selection_loss(self, arrived):
            if self.predictive_mode != 'native':
                return self.paired_losses(arrived)[2]
            _, current = self._split(arrived)
            loss = native_class.forward(self, current, None, None, None)[0]
            self.last_loss_parts = dict(native=float(loss.detach()))
            return loss

        def endpoint_evidence(self, prediction, current):
            return terminal_evidence(
                prediction[4][:, -1], prediction[3][:, -1], prediction[9][:, -1], current[:, -1],
                alpha_cycle=self.alpha_cycle_l, alpha_ms=self.alpha_ms_l,
                alpha_cons=self.alpha_cons_l, channels=self.prediction_channels)

        @torch.no_grad()
        def infer(self, arrived, x_mark_enc=None, x_dec=None, x_mark_dec=None):
            past, current = self._split(arrived)
            if self.predictive_mode == 'native':
                return native_class.infer(self, current, None, None, None)
            evidence, _ = self.endpoint_evidence(self.prediction_forward(past), current)
            # A single scored timestamp; do not broadcast across unsupervised H.
            return evidence[:, None, :], None

    return TerminalAlignedLEFT
