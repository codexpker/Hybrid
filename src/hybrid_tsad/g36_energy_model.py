"""G36 normal response energy distribution with a paired feedback ablation.

The normal reference is generated from the strictly past 48-row context.
context_only masks response inputs; response_feedback differs only in that
mask. No current response or query update enters reference generation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .g36_model import PastConv


def spectrum(value):
    """Four response times; FIT standardized, with DC retained."""
    return torch.log1p(torch.abs(torch.fft.rfft(value, dim=-2, norm='ortho')))


def distance(a, b):
    """Broadcastable per-block temporal plus log-magnitude distance."""
    return (torch.linalg.vector_norm((a-b).flatten(-2), dim=-1)/math.sqrt(16)
            + torch.linalg.vector_norm((spectrum(a)-spectrum(b)).flatten(-2), dim=-1)/math.sqrt(12))


def fixed_noise(count, seed, device=None, dtype=torch.float32):
    # CPU draw makes the fixed banks independent of the selected GPU and batch.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(count, 16, generator=generator, dtype=dtype).to(device=device)


class ContextEnergyResponse(nn.Module):
    def __init__(self, variant='context_only', hidden=64, kernel=3):
        super().__init__()
        if variant not in ('context_only', 'response_feedback'):
            raise ValueError(variant)
        self.variant = variant
        self.input = nn.Conv1d(9, hidden, 1)
        self.layers = nn.ModuleList(PastConv(hidden, hidden, kernel, dilation=d) for d in (1, 2, 4, 8))
        self.generator = nn.Sequential(nn.Linear(hidden+16, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 16))

    def split(self, arrived):
        if arrived.ndim != 3 or arrived.shape[1:] != (52, 9):
            raise ValueError('Exactly 48 past + 4 arrived rows and nine channels required')
        if not bool(torch.isfinite(arrived).all()):
            raise ValueError('Finite FIT-scaled values required')
        return arrived[:, :48], arrived[:, 48:, :4]

    def context(self, past):
        if past.ndim != 3 or past.shape[1:] != (48, 9):
            raise ValueError('Wrong context shape')
        if self.variant == 'context_only':
            past = torch.cat((torch.zeros_like(past[:, :, :4]), past[:, :, 4:]), -1)
        h = F.gelu(self.input(past.transpose(1, 2)))
        for layer in self.layers:
            h = h + F.gelu(layer(h))/2
        return h[:, :, -1]

    def generate(self, context, noise):
        if noise.ndim == 2:
            noise = noise.unsqueeze(0).expand(context.shape[0], -1, -1)
        if noise.ndim != 3 or noise.shape[0] != context.shape[0] or noise.shape[-1] != 16:
            raise ValueError('Noise must be [draw,16] or [batch,draw,16]')
        h = context[:, None].expand(-1, noise.shape[1], -1)
        return self.generator(torch.cat((h, noise), -1)).reshape(context.shape[0], noise.shape[1], 4, 4)

    def energy(self, arrived, noise, pair_noise):
        past, target = self.split(arrived)
        h = self.context(past)
        a, b = self.generate(h, noise), self.generate(h, pair_noise)
        if a.shape != b.shape:
            raise ValueError('Independent paired draw banks must have equal sizes')
        return distance(target[:, None], a).mean(1) - .5*distance(a, b).mean(1)

    def loss(self, arrived, draws=16):
        shape = (len(arrived), draws, 16)
        a = torch.randn(shape, device=arrived.device, dtype=arrived.dtype)
        b = torch.randn(shape, device=arrived.device, dtype=arrived.dtype)
        return self.energy(arrived, a, b).mean()

    def components(self, arrived, reference_noise, pair_noise, probe_noise, probe_chunk=32):
        past, target = self.split(arrived)
        h = self.context(past)
        reference = self.generate(h, reference_noise)
        pair = self.generate(h, pair_noise)
        normal = self.generate(h, probe_noise)
        penalty = .5*distance(reference, pair).mean(1)
        observed_distance = distance(target[:, None], reference).mean(1)
        observed_energy = observed_distance-penalty
        normal_distance = []
        # Explicit norms avoid cdist's size-dependent matrix multiplication path.
        reference_spectral = spectrum(reference)
        for begin in range(0, normal.shape[1], probe_chunk):
            n = normal[:, begin:begin+probe_chunk]
            temporal = torch.linalg.vector_norm((n[:, :, None]-reference[:, None]).flatten(-2), dim=-1)/math.sqrt(16)
            spectral = torch.linalg.vector_norm((spectrum(n)[:, :, None]-reference_spectral[:, None]).flatten(-2), dim=-1)/math.sqrt(12)
            normal_distance.append((temporal+spectral).mean(-1))
        normal_distance = torch.cat(normal_distance, dim=1)
        # The same penalty occurs on each side and cancels exactly. Do not let
        # subtraction rounding change tail-count ties.
        count = (normal_distance >= observed_distance[:, None]).sum(1)
        probability = (count.to(arrived.dtype)+1)/(normal.shape[1]+1)
        return dict(score=-torch.log(probability), normal_tail_probability=probability,
                    normal_tail_count=count, observed_energy=observed_energy,
                    reference_pair_penalty=penalty, observed_distance=observed_distance,
                    normal_distance=normal_distance,
                    normal_energy=normal_distance-penalty[:, None],
                    target_response=target, predicted_response_mean=reference.mean(1))
