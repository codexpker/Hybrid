"""Native TimeRadar with a trainable past-only multiscale multivariate context.

The official model owns normalization, order, masks, inverse transform and score.
The context changes complex encoder outputs before the original decoder.
"""
import math

import torch
from torch import nn

from .native_v_conditioning import fixed_native_rng


def record_window(values, current_start, role_bounds, history_length):
    """Slice one already standardized record without crossing a role boundary."""
    lo, hi = role_bounds
    past_start = current_start - history_length
    if history_length not in (256, 512) or not 0 <= lo <= past_start < current_start < hi <= len(values):
        raise ValueError("Invalid history/role footprint")
    if current_start + 100 > hi:
        raise ValueError("Current window crosses role end")
    if values.ndim != 2:
        raise ValueError("Provide exactly one record, with time and variable axes")
    history = values[past_start:current_start].unsqueeze(0)
    current = values[current_start:current_start + 100].unsqueeze(0)
    return history, current, dict(history_start=past_start, current_start=current_start)


def reconstruction_update(model, optimizer, windows):
    """One logical update; windows share a sealed role and sampling plan."""
    if not windows:
        raise ValueError("Empty logical batch")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for history, current, footprint in windows:
        _, copies = model(history, current, **footprint)
        loss, _ = model.loss_and_score(copies, current)
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite loss before optimizer update")
        (loss / len(windows)).backward()
        total += loss.detach().item() / len(windows)
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Nonfinite gradient before optimizer update")
    optimizer.step()
    return total


class PastContext(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.patch = nn.Linear(32, width)
        self.statistics = nn.Linear(2, width)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0,
                                       batch_first=True, norm_first=True),
            2, enable_nested_tensor=False)
        self.variable = nn.MultiheadAttention(width, 4, dropout=0, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, history):
        # Only legal, unpadded histories enter normalization or attention.
        if history.ndim != 3 or history.shape[0] != 1:
            raise ValueError("One multivariate history per native B1 forward")
        if history.shape[1] not in (256, 512) or not torch.isfinite(history).all():
            raise ValueError("Require 256 or 512 finite past observations")
        h = history.transpose(1, 2)
        patches = h.unfold(-1, 32, 32)
        tokens = self.patch(patches)[0]  # C,P,D
        count = tokens.shape[1]
        position = torch.arange(count, device=h.device, dtype=h.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, self.width, 2, device=h.device,
                                           dtype=h.dtype) * (-math.log(10000) / self.width))
        angle = position * frequency
        pos = torch.stack((angle.sin(), angle.cos()), -1).flatten(-2)
        # Source-scaled level and spread remain visible, without RevIN.
        endpoints = torch.arange(31, h.shape[-1], 32, device=h.device)
        n = (endpoints + 1).to(h.dtype)
        mean = h.cumsum(-1)[..., endpoints] / n
        variance = (h.square().cumsum(-1)[..., endpoints] / n - mean.square()).clamp_min(0)
        stats = torch.stack((mean, (variance + 1e-6).sqrt()), -1)[0]
        base = tokens + pos[None] + self.statistics(stats)
        memories, own = [], []
        for scale in (1, 2, 4):
            x = base.reshape(base.shape[0], count // scale, scale, self.width).mean(2)
            causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
            x = self.temporal(x, mask=causal)
            # At each past patch endpoint, attend over variables of that patch.
            v = x.transpose(0, 1)
            mixed, _ = self.variable(v, v, v, need_weights=False)
            x = self.norm(v + mixed).transpose(0, 1)
            memories.append(x)
            own.append(x[:, -1])
        return torch.cat(memories, 1), torch.stack(own).mean(0)


class HistoryConditionedTimeRadar(nn.Module):
    def __init__(self, native, width=64):
        super().__init__()
        self.native = native
        self.history = PastContext(width)
        self.query = nn.Linear(512, width)
        self.cross = nn.MultiheadAttention(width, 4, dropout=0, batch_first=True)
        self.output = nn.Linear(width, 512)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.native.requires_grad_(False)
        self.native.decoder.requires_grad_(True)
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.native.eval()
        return self

    def forward(self, history, current, *, history_start, current_start, bypass=False):
        if current.ndim != 3 or current.shape[:2] != (1, 100):
            raise ValueError("Native W100 and B1 are required")
        if history.shape[2] != current.shape[2] or history_start + history.shape[1] != current_start:
            raise ValueError("History must end exactly before current window in one bound record")
        if not torch.isfinite(current).all():
            raise ValueError("Require finite source-scaled observations")
        # Caller binds record/role identity; offsets alone do not prove permission.
        if bypass:
            with fixed_native_rng(current.device):
                return self.native(current, norm=0, mask_mode="c", copies=10)
        memory, own = self.history(history)
        channels = current.shape[2]

        def condition(module, args, representation):
            expected = (10 * channels, 20, 256)
            if representation.shape != expected or not representation.is_complex():
                raise ValueError("Official encoder layout changed")
            ri = torch.cat((representation.real, representation.imag), -1)
            query = self.query(ri).reshape(10, channels, 20, -1)
            query = query + own[None, :, None]
            q = query.reshape(1, 10 * channels * 20, -1)
            kv = memory.reshape(1, -1, memory.shape[-1])
            context, _ = self.cross(q, kv, kv, need_weights=False)
            delta = self.output(context).reshape(10 * channels, 20, 512)
            real, imag = delta.chunk(2, -1)
            return representation + torch.complex(real, imag)

        handle = self.native.encoder.register_forward_hook(condition)
        try:
            with fixed_native_rng(current.device):
                return self.native(current, norm=0, mask_mode="c", copies=10)
        finally:
            handle.remove()

    @staticmethod
    def loss_and_score(copies, current):
        loss = (copies - current[None]).square().mean()
        score = copies.var(0, correction=1).mean(-1)
        return loss, score
