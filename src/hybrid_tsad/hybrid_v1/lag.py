"""Project finite directional lag adapter; NOT the author circular Autoformer.

Explicit overlap-normalized Q/K correlation and V at endpoint-lag. Candidate
axes remain (target_i,source_j,lag). No cross-window/batch top-k sharing.
"""
import math
import torch
from torch import nn


class FiniteLag(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(1, config.lag_width)
        self.key = nn.Linear(1, config.lag_width)
        self.value = nn.Linear(1, config.width)
        self.self_out = nn.Linear(config.width, config.width, bias=False)
        self.cross_out = nn.Linear(config.width, config.width, bias=False)

    def correlations(self, student, visible):
        cfg = self.config
        x = student[:, -cfg.lag_window:].transpose(1, 2)
        m = visible[:, -cfg.lag_window:].transpose(1, 2)
        q, k = self.query(x[..., None]), self.key(x[..., None])
        scores, counts = [], []
        for lag in range(cfg.max_lag + 1):
            qa, ka = q[:, :, lag:], k[:, :, :x.shape[-1] - lag]
            qm, km = m[:, :, lag:], m[:, :, :x.shape[-1] - lag]
            count = torch.einsum("bis,bjs->bij", qm.float(), km.float())
            numerator = torch.einsum("bisd,bjsd->bij", qa * qm[..., None], ka * km[..., None])
            scores.append(numerator / (count.clamp_min(1) * math.sqrt(cfg.lag_width)))
            counts.append(count)
        correlation = torch.stack(scores, -1)
        count = torch.stack(counts, -1)
        # Endpoint value actually aggregated must itself be visible.
        endpoint_valid = torch.stack([m[..., -1 - lag] for lag in range(cfg.max_lag + 1)], -1)
        valid = (count >= cfg.lag_min_overlap) & endpoint_valid[:, None]
        values = self.value(x[..., None])
        endpoint_values = torch.stack([values[..., -1 - lag, :] for lag in range(cfg.max_lag + 1)], -2)
        return correlation, count, valid, endpoint_values

    def _aggregate(self, score, valid, values):
        b, c, _, l = score.shape
        flat, ok = score.reshape(b, c, -1), valid.reshape(b, c, -1)
        number = min(self.config.lag_top_k, flat.shape[-1])
        chosen_score, chosen = flat.masked_fill(~ok, -torch.inf).topk(number, -1)
        selected_valid = ok.gather(-1, chosen)
        any_valid = selected_valid.any(-1, keepdim=True)
        safe_score = torch.where(any_valid, chosen_score, torch.zeros_like(chosen_score))
        weights = torch.softmax(safe_score, -1).masked_fill(~selected_valid, 0)
        # source-lag grid, independently gathered for each target i.
        grid = values.reshape(b, c * l, -1)[:, None].expand(-1, c, -1, -1)
        selected_values = grid.gather(2, chosen[..., None].expand(-1, -1, -1, grid.shape[-1]))
        out = (selected_values * weights[..., None]).sum(-2)
        return out, {"source": chosen // l, "lag": chosen % l, "weights": weights,
                     "selected_valid": selected_valid, "valid": any_valid.squeeze(-1)}

    def forward(self, student, visible):
        score, count, valid, values = self.correlations(student, visible)
        c = student.shape[-1]
        eye = torch.eye(c, dtype=torch.bool, device=student.device)[None, :, :, None]
        lags = torch.arange(self.config.max_lag + 1, device=student.device)
        self_valid = valid & eye & (lags > 0)
        cross_valid = valid & ~eye
        if self.config.cross_synchronous_only:
            cross_valid = cross_valid & (lags == 0)
        own, own_meta = self._aggregate(score, self_valid, values)
        cross, cross_meta = self._aggregate(score, cross_valid, values)
        own = self.self_out(own) if self.config.self_lag_enabled else torch.zeros_like(own)
        cross = self.cross_out(cross) if self.config.cross_lag_enabled else torch.zeros_like(cross)
        return own, cross, {"correlation": score, "overlap_count": count,
            "self_candidate_valid": self_valid, "cross_candidate_valid": cross_valid,
            "self": own_meta, "cross": cross_meta,
            "source_sample_offset": student.shape[1] - 1 - lags,
            "target_axis": "i", "source_axis": "j", "lag_unit": "original_sample", "circular": False}
