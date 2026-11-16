"""Receiver-specific other-variable context in the native complex patch domain.

The original z residual stays intact. No lag/causality or online claim is made;
the unchanged native FRFT order still shares statistics over its batch/channels.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .timeradar_cross_variable import ComplexChannelAdapter

ARMS = ('selective', 'capacity_self')


class SelectiveContextAdapter(nn.Module):
    def __init__(self, base_adapter, mode='selective'):
        super().__init__()
        if mode not in ARMS:
            raise ValueError(mode)
        self.dimension, self.rank, self.mode = base_adapter.dimension, base_adapter.rank, mode
        # Reuse the already-constructed historical down/up, including zero up.
        self.down, self.up = base_adapter.down, base_adapter.up
        self.query = nn.Linear(self.rank, self.rank, bias=False)
        self.key = nn.Linear(self.rank, self.rank, bias=False)
        self.force_uniform = False  # FIT-only mechanism intervention, never a scored arm.
        self.zero_donor_value = None
        self.capture_diagnostics = False
        self.last_diagnostics = None

    def other_weights(self, tokens):
        channels = tokens.shape[-2]
        q, k = self.query(tokens), self.key(tokens)
        logits = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(self.rank)
        if self.force_uniform:
            logits = torch.zeros_like(logits)
        diagonal = torch.eye(channels, device=tokens.device, dtype=torch.bool)
        return torch.softmax(logits.masked_fill(diagonal, float('-inf')), -1)

    def forward(self, representation, copies, batch, channels):
        if not torch.is_complex(representation) or representation.ndim != 3:
            raise ValueError('Native complex [K*B*C,P,D] required')
        n, patches, dimension = representation.shape
        if n != copies*batch*channels or dimension != self.dimension:
            raise ValueError('Explicit original K,B,C layout mismatch')
        z = representation.reshape(copies, batch, channels, patches, dimension)
        h = F.gelu(self.down(torch.cat((z.real, z.imag), -1)))
        weights = None
        if self.mode == 'capacity_self':
            context = h + self.query(F.gelu(self.key(h)))
        elif channels == 1:
            context = h
        else:
            # K/B/P never enter the attention reduction; only donor-variable j.
            tokens = h.permute(0, 1, 3, 2, 4)  # K,B,P,C,r
            weights = self.other_weights(tokens)
            values = tokens
            if self.zero_donor_value is not None:
                donor = self.zero_donor_value
                if type(donor) is not int or not 0 <= donor < channels:
                    raise ValueError('Invalid FIT diagnostic donor')
                # Q/K remain unchanged; this removes only the chosen donor value.
                values = values.clone(); values[..., donor, :] = 0
            context = torch.matmul(weights, values).permute(0, 1, 3, 2, 4)
        real, imag = self.up(context).chunk(2, -1)
        if self.capture_diagnostics:
            self.last_diagnostics = {'weights': None if weights is None else weights.detach().cpu(),
                'context': context.detach().cpu(), 'own_h': h.detach().cpu(),
                'axes': 'weights[K,B,P,receiver,donor]; context[K,B,C,P,r]',
                'force_uniform': self.force_uniform, 'zero_donor_value': self.zero_donor_value}
        return (z+torch.complex(real, imag)).reshape_as(representation)


def attach(wrapper, mode):
    """Keep the existing hook/native freeze policy; replace only its adapter."""
    if not isinstance(wrapper.adapter, ComplexChannelAdapter):
        raise ValueError('Construct the historical self adapter before extension')
    wrapper.adapter = SelectiveContextAdapter(wrapper.adapter, mode)
    wrapper.mode = mode
    wrapper.train(wrapper.training)
    return wrapper
