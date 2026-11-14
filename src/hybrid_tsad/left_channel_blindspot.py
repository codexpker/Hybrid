"""Whole-channel LEFT conditional reconstruction, with no additional parameters.

Normal masked-channel reconstruction is a reused learning principle, not a
claim of novelty. Current other-channel observations are allowed: this is
causal conditional detection at t, not advance forecasting.
"""
import torch
from torch.nn import functional as F
from .left_predictive_cycle import predictive_cycle_class


def channel_blindspot_class(native_class, mode='blind'):
    if mode not in ('blind', 'native'):
        raise ValueError(mode)
    guarded_base = predictive_cycle_class(native_class, 'full')

    class ChannelBlindLEFT(guarded_base):
        def __init__(self, configs):
            super().__init__(configs)
            self.blind_mode = mode
            self.last_blind_parts = None

        def _validate(self, current, channels):
            if current.ndim != 3 or current.shape[1:] != (self.seq_len, self.enc_in):
                raise ValueError('Complete role-contained H-window required')
            if channels.shape != (len(current),) or channels.dtype != torch.long:
                raise ValueError('One predetermined integer channel per sample')
            if not bool(((channels >= 0) & (channels < self.enc_in)).all()):
                raise ValueError('Channel outside fixed sensor order')
            if not bool(torch.isfinite(current).all()):
                raise ValueError('No fabricated/invalid observations')

        @staticmethod
        def _pick(value, channels):
            # Channel-last time/multiscale tensors, preserve singleton channel.
            return value.gather(2, channels[:, None, None].expand(-1, value.shape[1], 1))

        @staticmethod
        def _pick_spectrum(value, channels):
            shape = [len(channels), 1] + list(value.shape[2:])
            idx = channels.reshape(len(channels), 1, *([1] * (value.ndim - 2))).expand(shape)
            return value.gather(1, idx)

        def masked_forward(self, current, channels):
            self._validate(current, channels)
            masked = current.clone()
            masked.scatter_(2, channels[:, None, None].expand(-1, self.seq_len, 1), 0.)
            # Uses pre-native prototype snapshot, independent dropout RNG and no
            # explicit EMA/step; no target statistics enter any encoder/cache.
            with self._past_state_guard(masked):
                return native_class._forward(self, masked, None, None, None)

        def masked_target_loss(self, prediction, truth, channels):
            scales = self.ms_utils(truth)
            target_ms = self.ms_utils.concat_sampling_list(scales[1:] + [truth])
            ms = self._ms_weighted_smoothl1(
                self._pick(prediction[1], channels), self._pick(target_ms, channels),
                seg_lens=prediction[2][1:], weights=getattr(self, 'ms_seg_weights', None))
            predicted_s = self.tf(prediction[4])
            target_s = self.tf(truth)
            cycle_tf = F.smooth_l1_loss(self._pick_spectrum(predicted_s, channels),
                                       self._pick_spectrum(target_s, channels))
            cycle_ft = F.smooth_l1_loss(self._pick(prediction[3], channels),
                                       self._pick(truth, channels))
            cons = F.smooth_l1_loss(self._pick(prediction[9], channels),
                                   self._pick(prediction[3].detach(), channels))
            loss = self.alpha_cycle_l * (cycle_tf + cycle_ft) + self.alpha_ms_l * ms + self.alpha_cons_l * cons
            return loss, dict(masked_ms=ms, masked_cycle_tf=cycle_tf,
                              masked_cycle_ft=cycle_ft, masked_consistency=cons)

        def paired_losses(self, current, channels):
            prediction = self.masked_forward(current, channels)
            mloss, parts = self.masked_target_loss(prediction, current, channels)
            native = native_class.forward(self, current, None, None, None)[0]
            joint = .5 * native + .5 * mloss
            self.last_blind_parts = {k: float(v.detach()) for k,v in
                dict(native=native, masked=mloss, joint=joint, **parts).items()}
            return native, mloss, joint

        def forward(self, current, channels, *unused):
            native, masked, joint = self.paired_losses(current, channels)
            return (joint if self.blind_mode == 'blind' else native), None

        @torch.no_grad()
        def selection_loss(self, current, channels):
            return self.paired_losses(current, channels)[2]

        @torch.no_grad()
        def score_one_channel(self, current, channels, return_reconstruction=False):
            prediction = self.masked_forward(current, channels)
            scales = self.ms_utils(current)
            target_ms = self.ms_utils.concat_sampling_list(scales[1:] + [current])
            score = self._ms_anomaly_score(prediction[1], target_ms, prediction[2],
                prediction[3], prediction[4], current, prediction[9], prediction[10])
            selected = self._pick(score, channels)[:, -1, 0]
            if return_reconstruction:
                return selected, tuple(self._pick(prediction[i], channels) for i in (3,4,9))
            return selected

        @torch.no_grad()
        def native_readout(self, current):
            return native_class.infer(self, current, None, None, None)[0]

    return ChannelBlindLEFT
