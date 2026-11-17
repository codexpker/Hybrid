"""Local patch residual for the pinned Time-RCD encoder.

No context, position, normalization, decoder or scoring changes. FRFT is the
discrete unitary fractional transform from torch-frft, along the patch axis.
Its coordinates are not physical time delays. Upstream code is not modified.
"""
import math

import torch
from torch import nn


class PatchResidual(nn.Module):
    def __init__(self, native, mode, hidden=32):
        super().__init__()
        if mode not in ('time', 'frft'):
            raise ValueError(mode)
        self.native = native
        self.mode = mode
        self.enabled = True
        self.raw_order = nn.Parameter(torch.tensor(math.log(1 / 3)))
        self.project = nn.Sequential(nn.Linear(2 * native.in_features, hidden),
                                     nn.GELU(), nn.Linear(hidden, native.out_features))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    @property
    def order(self):
        return 2 * torch.sigmoid(self.raw_order)

    def forward(self, patches):
        original = self.native(patches)
        if not self.enabled:
            return original
        if self.mode == 'frft':
            from torch_frft.dfrft_module import dfrft
            z = dfrft(patches.float(), self.order, dim=-1)
            features = torch.cat((z.real, z.imag), dim=-1)
        else:
            features = torch.cat((patches, torch.nn.functional.gelu(self.order * patches)), dim=-1)
        return original + self.project(features.to(original.dtype))


def install_residual(model, mode='native', hidden=32):
    """Install after loading the official checkpoint; return the same model."""
    if mode == 'native':
        return model
    layer = model.ts_encoder.embedding_layer
    if not isinstance(layer, nn.Linear):
        raise TypeError('Expected original Linear patch embedding; refuse double wrapping')
    model.ts_encoder.embedding_layer = PatchResidual(layer, mode, hidden).to(
        device=layer.weight.device, dtype=layer.weight.dtype)
    return model
