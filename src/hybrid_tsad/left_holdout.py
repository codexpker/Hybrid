"""Pre-view contiguous holdout reconstruction with the existing LEFT encoders."""

import torch
from torch.nn import functional as F

from . import left_timesfm_condition as history

PATCH, MASKS = 24, 4


def hidden_positions(mask_id, device):
    if not 0 <= mask_id < MASKS:
        raise ValueError(mask_id)
    return torch.arange(history.WINDOW, device=device) // PATCH % MASKS == mask_id


class HoldoutLEFT(history.ConditionalLEFT):
    def predict(self, visible, past, lengths, features=None, valid=None):
        self.set_history(past, lengths, features, valid)
        try:
            result = self.left._forward(visible, None, None, None)
            return result[4], result[3], result[9]
        finally:
            self.clear_history()

    def reconstruct(self, truth, mask_id, past, lengths, features=None, valid=None):
        """The native forward sees only visible observations and fixed zeros."""
        hidden = hidden_positions(mask_id, truth.device)
        visible = truth.masked_fill(hidden[None, :, None], 0.0)
        predictions = self.predict(visible, past, lengths, features, valid)
        return hidden, predictions

    def objective(self, truth, mask_id, *context):
        hidden, (time, freq, multiscale) = self.reconstruct(truth, mask_id, *context)
        target = truth.detach()
        # Preserve visible values on both sides of the frequency cycle. Its
        # prediction differs from the raw target only at the hidden positions.
        completed = torch.where(hidden[None, :, None], time, target)
        left = self.left
        cycle_tf = F.smooth_l1_loss(left.tf(completed), left.tf(target).detach())
        cycle_ft = F.smooth_l1_loss(freq[:, hidden], target[:, hidden])
        fine = F.smooth_l1_loss(multiscale[:, hidden], target[:, hidden])
        consistency = F.smooth_l1_loss(multiscale[:, hidden], freq[:, hidden].detach())
        band = float(getattr(left, "alpha_band_reg", 0.0))
        return (
            left.alpha_cycle_l * (cycle_tf + cycle_ft)
            + left.alpha_ms_l * (fine + band * left.ms_utils.band_regularizer())
            + left.alpha_cons_l * consistency
        )

    @torch.no_grad()
    def score(self, truth, *context):
        if self.training:
            raise RuntimeError("Holdout scoring requires frozen evaluation mode")
        left = self.left
        scores = truth.new_empty(truth.shape[:2])
        for mask_id in range(MASKS):
            hidden, (time, freq, multiscale) = self.reconstruct(
                truth, mask_id, *context
            )
            target = truth[:, hidden]
            residual = left.alpha_cycle_s * (
                left.alpha_time * (time[:, hidden] - target).abs()
                + left.alpha_freq * (freq[:, hidden] - target).abs()
            )
            residual += left.alpha_ms_s * F.smooth_l1_loss(
                multiscale[:, hidden], target, reduction="none"
            )
            scores[:, hidden] = residual.mean(-1)
        return scores


def initial(w, domain, arm):
    # The existing wrapper uses C for hidden embeddings, and E for the
    # identical architecture with random frozen hidden embeddings.
    prototype, parent = history.initial(w, domain, "A")
    kind = {"A": "A", "B": "C", "C": "E"}[arm]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        return HoldoutLEFT(prototype.left, kind).cuda(), parent
