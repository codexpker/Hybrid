"""Shared complete LEFT learning/decoding with a strictly past prediction path.

The official native encoder/fusion/prototypes/three decoders and anomaly formula
are reused without an additional prediction head. This is a project mechanism,
not an official LEFT implementation or a claim of novelty. Old Et-history
injection, external Ridge scores, and all failed local candidates are untouched.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F

MODES = ('native', 'full', 'history_only')


def predictive_cycle_class(native_class, mode):
    """Return a parameter-compatible LEFT subclass; no new trainable tensors."""
    if mode not in MODES:
        raise ValueError(mode)

    class SharedPredictiveLEFT(native_class):
        def __init__(self, configs):
            super().__init__(configs)
            self.predictive_mode = mode
            self.prediction_channels = getattr(configs, 'prediction_channels', None)
            self.prediction_rng_seed = 84551
            self._past_lambda_override = None
            self.last_loss_parts = None

        def _curriculum_lambda(self):
            if self._past_lambda_override is not None:
                return self._past_lambda_override
            return super()._curriculum_lambda()

        def _split(self, arrived):
            if arrived.ndim != 3 or arrived.shape[1:] != (self.seq_len + 1, self.enc_in):
                raise ValueError('Exactly H+1 role-contained arrived observations required')
            if not bool(torch.isfinite(arrived).all()):
                raise ValueError('No fabricated native target/history observations allowed')
            return arrived[:, :-1], arrived[:, 1:]

        @contextmanager
        def _past_state_guard(self, past):
            """Disable only native top-level EMA/step, retain child train modes.

            LEFT checks self.training for explicit state updates and curriculum.
            Dropout lives in child modules, whose modes are never changed here.
            The already-effective current curriculum is retained explicitly.
            A forked per-update prediction RNG never advances native/drop-plan RNG.
            """
            training = self.training
            old_override = self._past_lambda_override
            # Read the next native optimizer update's curriculum without
            # advancing persistent state or exposing current observations.
            step = self.step
            try:
                if training:
                    self.step = step + 1
                lam = native_class._curriculum_lambda(self)
            finally:
                self.step = step
            devices = [past.device.index] if past.is_cuda else []
            normalizers = []
            # Native current EMA mutates prototype.data in place. Prediction
            # must use a differentiable snapshot, both for causal state and
            # for backward after the subsequent native EMA update.
            for bank in (self.proto_t, self.proto_f):
                original = bank._norm
                snapshot = bank.prototypes.clone()
                def snapshot_norm(x, bank=bank, original=original, snapshot=snapshot):
                    return original(snapshot if x is bank.prototypes else x)
                normalizers.append((bank, original))
                bank._norm = snapshot_norm
            self.training = False
            self._past_lambda_override = lam
            try:
                with torch.random.fork_rng(devices=devices, enabled=training):
                    if training:
                        seed = self.prediction_rng_seed + self.step + 1
                        torch.random.default_generator.manual_seed(seed)
                        if past.is_cuda:
                            torch.cuda.default_generators[past.device.index].manual_seed(seed)
                    yield
            finally:
                self.training = training
                self._past_lambda_override = old_override
                for bank, normalizer in normalizers:
                    bank._norm = normalizer

        def prediction_forward(self, past):
            if self.predictive_mode == 'history_only':
                if self.prediction_channels != [0]:
                    raise ValueError('The frozen history-only ablation is TEP target0 only')
                # Zero means fixed FIT-standardized mean, not a QUERY statistic.
                past = torch.cat((past[:, :, :1], torch.zeros_like(past[:, :, 1:])), dim=2)
            with self._past_state_guard(past):
                return native_class._forward(self, past, None, None, None)

        def _terminal(self, prediction, current):
            channels = slice(None) if self.prediction_channels is None else self.prediction_channels
            target = current[:, -1, channels]
            time = prediction[4][:, -1, channels]
            freq_time = prediction[3][:, -1, channels]
            multi = prediction[9][:, -1, channels]
            l_time = F.smooth_l1_loss(time, target)
            l_freq = F.smooth_l1_loss(freq_time, target)
            l_ms = F.smooth_l1_loss(multi, target)
            l_cons = F.smooth_l1_loss(multi, freq_time.detach())
            loss = (self.alpha_cycle_l * (l_time + l_freq)
                    + self.alpha_ms_l * l_ms + self.alpha_cons_l * l_cons)
            return loss, dict(time=l_time, freq=l_freq, ms=l_ms, consistency=l_cons)

        def paired_losses(self, arrived):
            past, current = self._split(arrived)
            # Strict ordering is essential: current's native EMA must never
            # affect this update's past prediction, even during training.
            prediction = self.prediction_forward(past)
            terminal_loss, terms = self._terminal(prediction, current)
            # One complete native current pass owns all original EMA/step updates.
            native_loss = native_class.forward(self, current, None, None, None)[0]
            joint = .5 * native_loss + .5 * terminal_loss
            self.last_loss_parts = {k: float(v.detach()) for k, v in
                                    dict(native=native_loss, terminal=terminal_loss,
                                         joint=joint, **terms).items()}
            return native_loss, terminal_loss, joint

        def forward(self, arrived, x_mark_enc=None, x_dec=None, x_mark_dec=None):
            if self.predictive_mode == 'native':
                _, current = self._split(arrived)
                loss = native_class.forward(self, current, None, None, None)[0]
                self.last_loss_parts = dict(native=float(loss.detach()))
                return loss, None
            native_loss, terminal_loss, joint = self.paired_losses(arrived)
            return joint, None

        @torch.no_grad()
        def selection_loss(self, arrived):
            """All matched arms use the identical normal-only SELECT formula."""
            return self.paired_losses(arrived)[2]

        def _prediction_score(self, prediction, current):
            # Current truth enters only target construction/error, never encoding.
            target_scales = self.ms_utils(current)
            target_ms = self.ms_utils.concat_sampling_list(target_scales[1:] + [current])
            return self._ms_anomaly_score(
                prediction[1], target_ms, prediction[2], prediction[3],
                prediction[4], current, prediction[9], prediction[10])

        @torch.no_grad()
        def infer(self, arrived, x_mark_enc=None, x_dec=None, x_mark_dec=None):
            past, current = self._split(arrived)
            if self.predictive_mode == 'native':
                return native_class.infer(self, current, None, None, None)
            prediction = self.prediction_forward(past)
            return self._prediction_score(prediction, current), None

        @torch.no_grad()
        def native_readout(self, arrived):
            """Fixed score-off ablation on the same candidate parameters."""
            _, current = self._split(arrived)
            return native_class.infer(self, current, None, None, None)

    return SharedPredictiveLEFT
