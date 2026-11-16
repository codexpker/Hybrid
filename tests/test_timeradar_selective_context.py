"""Synthetic selection/identity/layout tests; actual FIT witness is separate."""
import copy
import unittest
import torch
from hybrid_tsad.timeradar_cross_variable import ComplexChannelAdapter, TimeRadarCrossVariable
from hybrid_tsad.timeradar_selective_context import SelectiveContextAdapter, attach
from test_timeradar_cross_variable import FakeNative


class SelectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def adapter(self, arm='selective', dimension=4, rank=4):
        torch.manual_seed(2024)
        return SelectiveContextAdapter(ComplexChannelAdapter(dimension, rank, 'self'), arm)

    def nonzero(self, adapter):
        with torch.no_grad(): adapter.up.weight.fill_(.13); adapter.up.bias.zero_()

    def test_initialization_native_residual_equal_capacity(self):
        a, b = self.adapter(), self.adapter('capacity_self')
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        for key, value in a.state_dict().items(): self.assertTrue(torch.equal(value, b.state_dict()[key]))
        old = ComplexChannelAdapter(256, 16)
        full = SelectiveContextAdapter(old)
        self.assertEqual(sum(p.numel() for p in full.parameters()), 17424)
        z = torch.randn(12, 2, 4, dtype=torch.complex64)
        self.assertTrue(torch.equal(a(z, 2, 2, 3), z))
        self.assertTrue(torch.equal(b(z, 2, 2, 3), z))

    def test_bkp_isolation_complex_layout(self):
        a = self.adapter(); self.nonzero(a)
        z = torch.randn(2, 2, 3, 2, 4, dtype=torch.complex64); changed = z.clone(); changed[0, 0, 0, 0] += 9
        before = a(z.flatten(0, 2), 2, 2, 3).reshape_as(z)-z
        after = a(changed.flatten(0, 2), 2, 2, 3).reshape_as(z)-changed
        for i in (before[1]-after[1], before[0, 1]-after[0, 1], before[0, 0, :, 1]-after[0, 0, :, 1]):
            self.assertEqual(float(i.abs().max()), 0.)
        with self.assertRaises(ValueError): a(z.flatten(0, 2), 2, 1, 3)

    def test_receiver_weights_donor_value_vs_uniform(self):
        a = self.adapter(); self.nonzero(a); a.capture_diagnostics = True
        z = torch.randn(12, 2, 4, dtype=torch.complex64)
        original = a(z, 2, 2, 3); weights = a.last_diagnostics['weights'].clone()
        self.assertTrue(torch.equal(torch.diagonal(weights, dim1=-2, dim2=-1), torch.zeros(2, 2, 2, 3)))
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, 2, 3))
        self.assertGreater(float(weights.var(-2).max()), 0.)
        a.zero_donor_value = 0; removed = a(z, 2, 2, 3)
        self.assertTrue(torch.equal(weights, a.last_diagnostics['weights']))
        self.assertFalse(torch.equal(original, removed))
        # Receiver0 cannot use donor0, so value-only deletion leaves its update equal.
        torch.testing.assert_close(original.reshape(2, 2, 3, 2, 4)[:, :, 0], removed.reshape(2, 2, 3, 2, 4)[:, :, 0], rtol=0, atol=0)
        a.zero_donor_value = None; a.force_uniform = True
        self.assertFalse(torch.equal(original, a(z, 2, 2, 3)))

    def test_qk_gradients_and_single_variable_fallback(self):
        a = self.adapter(); self.nonzero(a)
        z = torch.randn(6, 2, 4, dtype=torch.complex64)
        a(z, 2, 1, 3).abs().square().sum().backward()
        for p in (a.query.weight, a.key.weight):
            self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.abs().sum()), 0.)
        one = a(torch.randn(2, 2, 4, dtype=torch.complex64), 2, 1, 1)
        self.assertTrue(torch.isfinite(one).all())

    def test_identity_qk_selects_common_legal_donors_by_receiver(self):
        a = self.adapter(dimension=2, rank=2)
        with torch.no_grad():
            a.query.weight.copy_(torch.eye(2)); a.key.weight.copy_(torch.eye(2))
        h = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]]).reshape(1, 1, 1, 4, 2)
        weights = a.other_weights(h)[0, 0, 0]
        self.assertGreater(float(weights[0, 1]), float(weights[0, 3]))
        self.assertGreater(float(weights[2, 3]), float(weights[2, 1]))
        uniform = (1-torch.eye(4))/3
        self.assertGreater(float((weights-uniform).abs().max()), 0.)

    def test_capacity_control_no_cross_and_qk_used(self):
        a = self.adapter('capacity_self'); self.nonzero(a)
        z = torch.randn(6, 2, 4, dtype=torch.complex64); changed = z.reshape(2, 1, 3, 2, 4).clone()
        changed[:, :, 0] += 20
        first = a(z, 2, 1, 3); second = a(changed.flatten(0, 2), 2, 1, 3)
        torch.testing.assert_close(first.reshape(2, 1, 3, 2, 4)[:, :, 1:], second.reshape(2, 1, 3, 2, 4)[:, :, 1:], rtol=0, atol=0)
        a.capture_diagnostics = True; a(z, 2, 1, 3).abs().square().mean().backward()
        self.assertIsNone(a.last_diagnostics['weights'])
        for p in (a.query.weight, a.key.weight): self.assertGreater(float(p.grad.abs().sum()), 0.)
        a.zero_donor_value = 0; a.force_uniform = True
        self.assertTrue(torch.equal(first, a(z, 2, 1, 3)))

    def test_real_wrapper_zero_disabled_rng_parity(self):
        for arm in ('selective', 'capacity_self'):
            torch.manual_seed(2024); native = FakeNative().eval()
            wrapper = attach(TimeRadarCrossVariable(copy.deepcopy(native), 'self', 2), arm).eval()
            x = torch.randn(2, 6, 3); rng = torch.get_rng_state()
            expected = native(x, copies=4); after = torch.get_rng_state()
            torch.set_rng_state(rng); actual = wrapper(x, copies=4)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            for a, b in zip(expected, actual): self.assertTrue(torch.equal(a, b))
            self.nonzero(wrapper.adapter); wrapper.adapter_enabled = False
            torch.set_rng_state(rng); disabled = wrapper(x, copies=4)
            for a, b in zip(expected, disabled): self.assertTrue(torch.equal(a, b))
            wrapper.close()


if __name__ == '__main__': unittest.main()
