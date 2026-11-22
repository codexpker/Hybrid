"""TimeRadar mask-dispersion and causal history ranking for mixed adaptation."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


WINDOW = 100
FEATURES = 7


def window_features(windows, terminal_variance):
    """Return channelwise native V and summaries of three strictly past patches."""
    values = np.asarray(windows, dtype=np.float32)
    variance = np.asarray(terminal_variance, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != WINDOW or variance.shape != values[:, -1].shape:
        raise ValueError("Expected [windows,100,channels] and matching native terminal variance")
    if not np.isfinite(values).all() or not np.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("Require finite standardized observations and nonnegative native variance")
    patches = values[:, 3:-1].reshape(len(values), 3, 32, values.shape[2])
    spread = patches.std(axis=2)
    movement = np.sqrt(np.mean(np.square(np.diff(patches, axis=2)), axis=2))
    combined = np.concatenate((np.log1p(variance)[:, None], np.log1p(spread), np.log1p(movement)), axis=1)
    if not np.isfinite(combined).all():
        raise FloatingPointError("Nonfinite native trajectory features")
    return np.ascontiguousarray(combined.transpose(0, 2, 1))


class NativePairRank(nn.Module):
    """Shared time–frequency channel scorer with permutation-invariant reduction."""

    def __init__(self):
        super().__init__()
        self.channel = nn.Sequential(nn.Linear(FEATURES, 24), nn.GELU(), nn.Linear(24, 16), nn.GELU())
        self.score = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, features):
        if features.ndim != 3 or features.shape[-1] != FEATURES or not features.shape[1]:
            raise ValueError("Expected [batch,channels,7] shared channel features")
        embedding = self.channel(features)
        pooled = torch.cat((embedding.mean(dim=1), embedding.amax(dim=1)), dim=-1)
        return self.score(pooled).squeeze(-1)


def rank_loss(positive, comparison, *, symmetric):
    """Both choices have slope -1/4 at equal logits; only one is symmetric."""
    if positive.shape != comparison.shape:
        raise ValueError("Rank loss requires one-to-one pairing")
    gap = positive - comparison
    return torch.sigmoid(-gap) if symmetric else 0.5 * F.softplus(-gap)
