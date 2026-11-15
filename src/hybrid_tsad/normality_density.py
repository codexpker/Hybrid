"""Joint observed-window normality density; a new research model, not LEFT.

Orthogonal wavelet changes of coordinates and affine coupling follow mature
normalizing-flow constructions. This project tests their joint time/frequency
normality criterion, without claiming these primitives are new inventions.
All input observations are available at the scored endpoint. Internal flow
coordinates are not physical causal effects or conditional target forecasts.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def haar(x, inverse=False, levels=2):
    """Orthonormal multiresolution Haar, shape [...,time,channel], |det|=1."""
    n = x.shape[-2]
    if n % (2 ** levels):
        raise ValueError('Window must be divisible by 2**levels')
    result = x
    if not inverse:
        for level in range(levels):
            length = n // (2**level)
            active, rest = result[..., :length, :], result[..., length:, :]
            a, b = active[..., ::2, :], active[..., 1::2, :]
            result = torch.cat(((a+b)/math.sqrt(2), (a-b)/math.sqrt(2), rest), -2)
    else:
        for level in reversed(range(levels)):
            length = n // (2**level)
            low, high = result[..., :length//2, :], result[..., length//2:length, :]
            restored = torch.stack(((low+high)/math.sqrt(2), (low-high)/math.sqrt(2)), -2)
            restored = restored.flatten(-3,-2)
            result = torch.cat((restored, result[..., length:, :]), -2)
    return result


class CausalConv(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, (self.dilation[0]*(self.kernel_size[0]-1),0)))


class Conditioner(nn.Module):
    def __init__(self, channels, hidden, local=False):
        super().__init__()
        self.input = nn.Conv1d(2*channels,hidden,1)
        self.layers = nn.ModuleList(CausalConv(hidden,hidden,3,dilation=d)
                                    for d in ((1,1,1,1) if local else (1,4,16,64)))
        self.output = nn.Conv1d(hidden,2*channels,1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, observed, mask):
        x = torch.cat((observed, mask.expand_as(observed)),dim=-1).transpose(1,2)
        h = F.gelu(self.input(x))
        for layer in self.layers:
            h = h + F.gelu(layer(h)) / math.sqrt(len(self.layers))
        return self.output(h).transpose(1,2).chunk(2,-1)


class Coupling(nn.Module):
    def __init__(self, channels, history, parity, hidden, local):
        super().__init__()
        mask = ((torch.arange(history)[:,None]+torch.arange(channels)[None,:]+parity)%2).float()
        self.register_buffer('mask',mask[None])
        self.conditioner = Conditioner(channels,hidden,local)

    def forward(self, x, inverse=False):
        passive = x*self.mask
        logscale, shift = self.conditioner(passive,self.mask)
        logscale = 2*torch.tanh(logscale/2)*(1-self.mask)
        shift = shift*(1-self.mask)
        if inverse:
            return passive + (1-self.mask)*(x-shift)*torch.exp(-logscale), -logscale
        return passive+(1-self.mask)*(x*torch.exp(logscale)+shift), logscale


class NormalityDensity(nn.Module):
    """One normalized density and one endpoint-assigned joint NLL criterion.

    Variants full/time_only/local have equal trainable parameter counts.
    history+1 inputs are accepted only for compatibility: last history are used.
    """
    def __init__(self, channels, history, variant='full', hidden=64):
        super().__init__()
        if variant not in ('full','time_only','local'):
            raise ValueError(variant)
        if history % 4 or channels < 1:
            raise ValueError('Positive channels and history divisible by four required')
        self.channels, self.history, self.variant = channels,history,variant
        self.offset = nn.Parameter(torch.zeros(1,1,channels))
        self.loggain = nn.Parameter(torch.zeros(1,1,channels))
        self.steps = nn.ModuleList(Coupling(channels,history,i,hidden,variant=='local') for i in range(4))

    def _window(self, arrived):
        if arrived.ndim != 3 or arrived.shape[-1] != self.channels or arrived.shape[1] not in (self.history,self.history+1):
            raise ValueError('Expected batch,history[, +1],channels')
        if not bool(torch.isfinite(arrived).all()):
            raise ValueError('Finite FIT-standardized inputs required')
        return arrived[:,-self.history:]

    def transform(self, x, inverse=False):
        x = self._window(x)
        gain = 2*torch.tanh(self.loggain/2)
        if not inverse:
            z, ld = (x-self.offset)*torch.exp(gain), gain.expand_as(x)
            for i, step in enumerate(self.steps):
                if self.variant != 'time_only' and i:
                    z = haar(z, inverse=i%2==0)
                z, change = step(z)
                # Coordinatewise allocations need not be invariant under Haar;
                # their full sum is the exact scalar log-Jacobian determinant.
                ld = ld+change
            return z,ld
        z, ld = x,torch.zeros_like(x)
        for i in reversed(range(len(self.steps))):
            z, change = self.steps[i](z,inverse=True); ld=ld+change
            if self.variant != 'time_only' and i:
                z = haar(z,inverse=i%2==1)
        return z*torch.exp(-gain)+self.offset,ld-gain

    @staticmethod
    def prior_nll(z):
        nu = 5.
        constant = math.lgamma(nu/2)-math.lgamma((nu+1)/2)+.5*math.log(nu*math.pi)
        return constant + .5*(nu+1)*torch.log1p(z.square()/nu)

    def evidence(self, arrived):
        z, logdet = self.transform(arrived)
        return self.prior_nll(z)-logdet

    def loss(self, arrived):
        return self.evidence(arrived).mean()

    def score(self, arrived):
        # Only the average across ALL channels is the joint probability score.
        # Channel allocations are diagnostics, not conditional/marginal density.
        return self.evidence(arrived).mean(1)

    def forward(self, arrived):
        return self.loss(arrived)

    def identity_score(self, arrived):
        return self.prior_nll(self._window(arrived)).mean(1)
