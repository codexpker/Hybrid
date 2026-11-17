"""Historical complex adapter with the entire native model frozen and eval.

Decoder parameters are frozen, but its operations remain in the autograd graph
so the decoder-input adapter receives the actual reconstruction-loss gradient.
"""
from torch import nn
from .timeradar_cross_variable import TimeRadarCrossVariable, ComplexChannelAdapter


class AdapterOnlyTimeRadar(TimeRadarCrossVariable):
    def __init__(self, constructed):
        nn.Module.__init__(self)
        if not isinstance(constructed.adapter, ComplexChannelAdapter):
            raise ValueError('Historical self/cross adapter required')
        constructed.close()
        self.native, self.adapter, self.mode = constructed.native, constructed.adapter, constructed.mode
        self._layout = None; self.adapter_enabled = True
        self.native.requires_grad_(False)
        self.adapter.requires_grad_(True)
        self._hook = self.native.decoder.register_forward_pre_hook(self._adapt)
        self.train(False)

    def train(self, mode=True):
        self.training = mode
        self.native.eval()  # Includes decoder and all original metadata buffers.
        self.adapter.train(mode)
        return self
