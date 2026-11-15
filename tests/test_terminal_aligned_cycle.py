"""Incremental F01/F02 new-method tests; old method witnesses remain valid."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from hybrid_tsad.left_predictive_cycle import predictive_cycle_class
from hybrid_tsad.terminal_aligned_cycle import terminal_aligned_cycle_class, terminal_evidence


class Native(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len, self.enc_in = 100, 3
        self.alpha_cycle_l, self.alpha_ms_l, self.alpha_cons_l = .5, .5, .1

    def forward(self, current, *args):
        return current.mean() + 2, None

    def infer(self, current, *args):
        return current.square(), None


class TerminalAlignedTest(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(prediction_channels=[0])
        self.old = predictive_cycle_class(Native, 'full')(self.cfg)
        self.new = terminal_aligned_cycle_class(Native, 'full')(self.cfg)
        g = torch.Generator().manual_seed(2025)
        self.current = torch.randn(3, 100, 3, generator=g)
        self.pred = [None] * 11
        for i in (3, 4, 9):
            self.pred[i] = torch.randn(3, 100, 3, generator=g, requires_grad=True)

    def test_loss_mean_and_gradient_exact_with_archived_terminal(self):
        evidence, parts = self.new.endpoint_evidence(self.pred, self.current)
        old = self.old._terminal(self.pred, self.current)[0]
        torch.testing.assert_close(evidence.mean(), old)
        self.assertEqual(evidence.shape, (3, 1))
        self.assertEqual(parts['consistency'].shape, (3, 1))
        grad0 = torch.autograd.grad(old, [self.pred[i] for i in (3, 4, 9)], retain_graph=True)
        grad1 = torch.autograd.grad(evidence.mean(), [self.pred[i] for i in (3, 4, 9)])
        for a, b in zip(grad0, grad1):
            torch.testing.assert_close(a, b)
            self.assertEqual(float(b[:, :-1].abs().sum()), 0.)

    def test_no_unsupervised_positions_or_channels_enter_score(self):
        a = self.new.endpoint_evidence(self.pred, self.current)[0]
        altered = [p.detach().clone() if isinstance(p, torch.Tensor) else p for p in self.pred]
        for i in (3, 4, 9):
            altered[i][:, :-1] = 1e6
            altered[i][:, -1, 1:] = -1e6
        torch.testing.assert_close(a, self.new.endpoint_evidence(altered, self.current)[0], rtol=0, atol=0)

    def test_samples_are_not_batch_mean(self):
        a = self.new.endpoint_evidence(self.pred, self.current)[0]
        for row in range(3):
            p = [v[row:row+1] if isinstance(v, torch.Tensor) else v for v in self.pred]
            torch.testing.assert_close(a[row:row+1], self.new.endpoint_evidence(p, self.current[row:row+1])[0])
        self.assertGreater(float(a.std()), 0.)

    def test_all_channels_and_numpy_layout_are_supported(self):
        args = [self.pred[i][:, -1] for i in (4, 3, 9)] + [self.current[:, -1]]
        a, _ = terminal_evidence(*args, alpha_cycle=.5, alpha_ms=.5, alpha_cons=.1)
        self.assertEqual(a.shape, (3, 3))
        self.old.prediction_channels = None
        torch.testing.assert_close(a.mean(), self.old._terminal(self.pred, self.current)[0])

    def test_native_selection_uses_only_native_task(self):
        model = terminal_aligned_cycle_class(Native, 'native')(self.cfg)
        with patch.object(model, 'paired_losses', side_effect=AssertionError('joint not allowed')):
            self.assertEqual(float(model.selection_loss(torch.zeros(2,101,3))), 2.)

    def test_hybrid_selection_and_training_unchanged(self):
        with patch.object(self.new, 'paired_losses', return_value=(torch.tensor(1.), torch.tensor(3.), torch.tensor(2.))):
            self.assertEqual(float(self.new.selection_loss(torch.zeros(2,101,3))), 2.)
            self.assertEqual(float(self.new(torch.zeros(2,101,3))[0]), 2.)

    def test_current_truth_not_in_prediction_and_old_score_not_called(self):
        arrived = torch.randn(3,101,3)
        with patch.object(self.new, 'prediction_forward', return_value=self.pred) as past, \
             patch.object(self.new, '_prediction_score', side_effect=AssertionError('old scoring forbidden')):
            score = self.new.infer(arrived)[0]
            torch.testing.assert_close(past.call_args.args[0], arrived[:, :-1])
            self.assertEqual(score.shape, (3,1,1))

    def test_native_readout_unchanged(self):
        model = terminal_aligned_cycle_class(Native, 'native')(self.cfg)
        x = torch.randn(2,101,3)
        torch.testing.assert_close(model.infer(x)[0], x[:,1:].square(), rtol=0, atol=0)

    def test_invalid_shapes_rejected(self):
        x = torch.zeros(2,3)
        with self.assertRaises(ValueError):
            terminal_evidence(x, x[:1], x, x, alpha_cycle=.5, alpha_ms=.5, alpha_cons=.1)


if __name__ == '__main__':
    unittest.main()
