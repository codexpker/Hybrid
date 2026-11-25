"""Matched uses of LEFT's registered multiscale modulation path."""

import inspect
from types import MethodType
from textwrap import dedent

import torch

ARMS = ('native', 'observation', 'prototype')
REVISION = '3fadb4811797075e233076b9faceb0d8ec59b697'


def channel_context(context, channels):
    """Match the decoder's sample-major [B*C, token, D] layout."""
    return context.repeat_interleave(channels, dim=0)


def mismatch_context(context):
    """Rotate window contexts by half the current batch, retaining a singleton."""
    return context.roll(context.shape[0]//2, dims=0)


def model_class(native, arm):
    if arm not in (*ARMS, 'mismatched'):
        raise ValueError(arm)
    # Reorder only the deterministic association/read block to make its
    # outputs available at the original decoder call. Preserve upstream RNG
    # order, decoder remainder, consistency, loss, and end-of-forward EMA.
    source = dedent(inspect.getsource(native._forward))
    left = source.index('    # Prototype association on fused tokens')
    right = source.index('    z_t_used =', left)
    read_block = source[left:right]
    assert 'z_t_mem = w_t @ Pt' in read_block and 'lam = self._curriculum_lambda()' in read_block
    source = source[:left] + source[right:]
    call = 'self.ms_decoder(Mtok_up, context=None)'
    assert source.count(call) == 1
    insertion = source.index('    y_flat, _, _ = '+call)
    source = source[:insertion] + read_block + source[insertion:]
    source = source.replace(call, 'self._condition_ms(Mtok_up, z_t, z_f, z_t_mem, z_f_mem, lam, B, C)')
    namespace = dict(native._forward.__globals__)
    exec(compile(source, '<LEFT fixed decoder context>', 'exec'), namespace)

    class ConditionedLeft(native):
        _forward = namespace['_forward']

        def __init__(self, config):
            super().__init__(config)
            if arm == 'mismatched':
                self.context_audit = None
                self.fit_singletons = 0
            for bank in (self.proto_t, self.proto_f):
                original_norm = bank._norm

                def snapshot_norm(module, value, norm=original_norm):
                    # Native EMA writes through .data before backward.
                    # A differentiable clone preserves pre-EMA storage for
                    # both the association and the value read in all arms.
                    if value is module.prototypes and torch.is_grad_enabled():
                        value = value.clone()
                    return norm(value)

                bank._norm = MethodType(snapshot_norm, bank)
            if arm != 'native':
                with torch.no_grad():
                    self.ms_decoder.to_gb[-1].weight.zero_()
                    self.ms_decoder.to_gb[-1].bias.zero_()
                if self.d_model == 128:
                    assert sum(p.numel() for p in self.ms_decoder.to_gb.parameters()) == 33280

        def _condition_ms(self, tokens, z_t, z_f, m_t, m_f, lam, batch, channels):
            assert tokens.shape[0] == batch * channels
            if arm == 'mismatched' and self.training and batch == 1:
                self.fit_singletons += 1
            if arm == 'native' or lam == 0:
                return self.ms_decoder(tokens, context=None)
            pair = (m_t, m_f) if arm == 'prototype' else (z_t, z_f)
            context = self.z_fuse(torch.cat(pair, dim=-1))
            assigned = mismatch_context(context) if arm == 'mismatched' else context
            # Project once per sample, then share the modulation across that
            # sample's channel tokens. Dropout and the native decoder remain
            # in their original locations.
            gb = channel_context(self.ms_decoder.to_gb(assigned), channels)
            gamma, beta = gb.unsqueeze(1).chunk(2, dim=-1)
            increment = lam * (gamma.tanh() * tokens + beta)
            if arm == 'mismatched' and self.context_audit is not None:
                self.context_audit(self.ms_decoder.to_gb, context, assigned, tokens, increment, lam, channels)
            return self.ms_decoder(tokens + increment, context=None)

    return ConditionedLeft
