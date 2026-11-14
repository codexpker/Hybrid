"""Regression witnesses for the stopped candidate, NOT a revised method.

The score stub models just the archived k=3 endpoint dependency; it is not
official LEFT, a historical score replay, or evidence of detection gains.
"""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from hybrid_tsad.left_predictive_cycle import predictive_cycle_class


class Sampling:
    def __call__(self, x):
        return [F.adaptive_avg_pool1d(x.transpose(1, 2), x.shape[1] // 4).transpose(1, 2)]

    def concat_sampling_list(self, values):
        return torch.cat(values, dim=1)


class NativeFixture(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len, self.enc_in = cfg.history, 1
        self.ms_utils = Sampling()
        self.alpha_cycle_l, self.alpha_ms_l, self.alpha_cons_l = .5, .5, .1

    def forward(self, current, *args):
        return current.sum() * 0 + 2., None

    def _ms_anomaly_score(self, ms_pred, ms_gt, lengths, freq, time, truth, multi, cache):
        residual = .2 * (freq-truth).abs() + .8 * (time-truth).abs()
        residual = residual + .1 * (multi-freq).abs()
        smooth = F.avg_pool1d(F.pad(residual.transpose(1, 2), (1, 1), mode='replicate'), 3, 1).transpose(1, 2)
        return .5 * smooth + .5 * F.smooth_l1_loss(ms_pred, ms_gt, reduction='none')


def fixture(history=100, mode='full'):
    model = predictive_cycle_class(NativeFixture, mode)(SimpleNamespace(history=history))
    truth = torch.zeros(1, history, 1)
    prediction = [None] * 11
    prediction[1], prediction[2] = torch.zeros_like(truth), [history // 4, history]
    for i in (3, 4, 9):
        prediction[i] = torch.zeros_like(truth)
    prediction[10] = {}
    return model, truth, prediction


class ArchivedPredictiveSemantics(unittest.TestCase):
    def test_archived_real_configs_use_single_scale_and_three_point_smoothing(self):
        root = Path(__file__).resolve().parents[1] / 'review_artifacts/code/preparations/hybrid_native_predictive_cycle_20261113_v1'
        for domain, length in [('TEP', 100), ('PSM', 192)]:
            cfg = json.loads((root/'configs'/f'{domain}.json').read_text())
            self.assertEqual(cfg['seq_len'], length)
            self.assertEqual(cfg['ms_kernels'], [4])
            self.assertEqual(cfg['smooth_k'], 3)
            self.assertEqual([cfg[k] for k in ('alpha_cycle_s', 'alpha_cons_s', 'alpha_time', 'alpha_freq')], [.5, .1, .8, .2])

    def test_F01_previous_unsupervised_position_changes_archived_score(self):
        for history in (100, 192):
            with self.subTest(history=history):
                model, truth, prediction = fixture(history)
                self.assertEqual(float(model._terminal(prediction, truth)[0]), 0.)
                self.assertEqual(float(model._prediction_score(prediction, truth)[0, -1, 0]), 0.)
                prediction[3][0, -2, 0] = prediction[4][0, -2, 0] = 9.
                self.assertEqual(float(model._terminal(prediction, truth)[0]), 0.)
                self.assertAlmostEqual(float(model._prediction_score(prediction, truth)[0, -1, 0]), 1.65, places=6)

    def test_F01_terminal_loss_gradient_is_endpoint_only(self):
        model, truth, prediction = fixture()
        for i in (3, 4, 9):
            prediction[i] = torch.ones_like(truth, requires_grad=True)
        model._terminal(prediction, truth)[0].backward()
        for i in (3, 4, 9):
            self.assertTrue(torch.equal(prediction[i].grad[:, :-1], torch.zeros_like(truth[:, :-1])))
            self.assertGreater(float(prediction[i].grad[:, -1].abs().sum()), 0.)

    def test_F01_current_target_not_in_past_input(self):
        model, _, _ = fixture()
        arrived = torch.arange(101.).reshape(1, 101, 1)
        changed = arrived.clone(); changed[:, -1] = 1e6
        before, current = model._split(arrived)
        after, other_current = model._split(changed)
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(torch.equal(current, other_current))

    def test_F02_native_train_and_joint_selection_are_different_contracts(self):
        model, _, _ = fixture(mode='native')
        arrived = torch.zeros(1, 101, 1)
        with patch.object(model, 'paired_losses', return_value=(torch.tensor(.1), torch.tensor(10.), torch.tensor(5.05))) as paired:
            self.assertEqual(float(model(arrived)[0]), 2.)
            paired.assert_not_called()
            self.assertAlmostEqual(float(model.selection_loss(arrived)), 5.05, places=5)
            paired.assert_called_once()

    def test_F02_joint_can_reverse_native_checkpoint_order(self):
        native, forecast = torch.tensor([.1, .2]), torch.tensor([10., .1])
        self.assertEqual(int(native.argmin()), 0)
        self.assertEqual(int((.5 * (native + forecast)).argmin()), 1)


if __name__ == '__main__':
    unittest.main()
