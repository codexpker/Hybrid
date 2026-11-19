"""Causal conditional generation with an immutable official response geometry.

The generator never receives the current target. Every response invocation is
one window (B=1); true and generated values share deterministic complementary
masks without consuming the training RNG. No historical adapter is installed.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import math

import torch
from torch import nn
from .hybrid_v1.temporal import TemporalEncoder


def energy_score(samples, truth):
    """Unbiased K-sample energy score, preserving each batch observation."""
    if samples.ndim != 3 or truth.shape != samples.shape[:1] + samples.shape[2:]:
        raise ValueError("Expected samples[B,K,D] and truth[B,D]")
    k = samples.shape[1]
    if k < 2:
        raise ValueError("At least two independent generated samples required")
    observed = torch.linalg.vector_norm(samples - truth[:, None], dim=-1).mean(1)
    # Explicit upper triangle avoids cdist's dimension-dependent GEMM path and
    # zero-distance sqrt derivatives; this is the ordered-pair formula / 2.
    spread = sum(torch.linalg.vector_norm(samples[:, i] - samples[:, j], dim=-1)
                 for i in range(k) for j in range(i + 1, k)) / (k * (k - 1))
    return observed - spread


class ConditionalGenerator(nn.Module):
    def __init__(self, sources):
        super().__init__()
        cfg = SimpleNamespace(width=128, temporal_patch=16, heads=4,
            temporal_layers=2, use_variate_attention=True, temporal_enabled=True)
        self.history = TemporalEncoder(cfg, sources)
        configs, dense, _ = sources.timesfm()
        self.output = dense.ResidualBlock(configs.ResidualBlockConfig(
            hidden_dims=256, output_dims=1, use_bias=True, activation="swish"))
        self.output.set_input_dims(128 + 8 + 8)

    def forward(self, history, shared_noise, variate_noise):
        if history.ndim != 3 or history.shape[1] != 512:
            raise ValueError("Exactly 512 strictly past observations required")
        b, _, c = history.shape
        k = shared_noise.shape[1]
        if shared_noise.shape != (b, k, 8) or variate_noise.shape != (b, k, c, 8):
            raise ValueError("Shared8 plus per-variate8 independent noise required")
        tokens, _, _ = self.history(history, torch.ones_like(history, dtype=torch.bool))
        summary = tokens[:, :, -1].unsqueeze(1).expand(b, k, c, 128)
        shared = shared_noise.unsqueeze(2).expand(b, k, c, 8)
        return self.output(torch.cat((summary, shared, variate_noise), -1)).squeeze(-1)


@contextmanager
def isolated_mask_rng(device, seed):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(seed)
        yield


class ResponseGeometry(nn.Module):
    def __init__(self, native, response_rms, mask_seed=84551):
        super().__init__()
        self.native = native
        self.register_buffer("response_rms", torch.as_tensor(float(response_rms)).clamp_min(1e-6))
        self.mask_seed = int(mask_seed)
        for p in native.parameters():
            p.requires_grad_(False)
        native.eval()

    def train(self, mode=True):
        self.training = mode
        self.native.eval()
        return self

    def response(self, window):
        if window.shape[0] != 1 or window.shape[1] != 100:
            raise ValueError("Official response must be invoked independently B1/W100")
        with isolated_mask_rng(window.device, self.mask_seed):
            _, reconstruction = self.native(window, norm=0, mask_mode="c", copies=10)
        if reconstruction.shape != (10, 1, 100, window.shape[2]):
            raise ValueError("Official reconstruction shape changed")
        return reconstruction[:, 0]

    def feature(self, history, value, use_response=True):
        c = value.numel()
        raw = value.flatten() / math.sqrt(c)
        if not use_response:
            return raw
        window = torch.cat((history[-99:], value[None]), 0)[None]
        response = self.response(window).flatten() / (self.response_rms * math.sqrt(10 * 100 * c))
        return torch.cat((raw, response))

    def score(self, history, generated, truth, use_response=True):
        # These loops intentionally preserve separate order/statistics per
        # observation AND per generated candidate; never flatten B*K into B.
        scores = []
        for h, values, y in zip(history, generated, truth):
            with torch.no_grad():
                gy = self.feature(h, y, use_response)
            gv = torch.stack([self.feature(h, v, use_response) for v in values])
            scores.append(energy_score(gv[None], gy[None])[0])
        return torch.stack(scores)


def fixed_noise(batch, channels, generator, *, device, copies=4):
    shared = torch.randn(batch, copies, 8, generator=generator)
    variate = torch.randn(batch, copies, channels, 8, generator=generator)
    return shared.to(device), variate.to(device)
