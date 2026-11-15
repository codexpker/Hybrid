import unittest
import torch

from hybrid_tsad.g36_energy_model import ContextEnergyResponse, distance, fixed_noise
from hybrid_tsad.g36_energy_runner import score_banks
from hybrid_tsad.normality_density_runner import validation_mode


class G36EnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, variant='context_only'):
        torch.manual_seed(2025)
        return ContextEnergyResponse(variant, hidden=8)

    def noise(self):
        return fixed_noise(8, 11), fixed_noise(8, 12), fixed_noise(31, 13)

    def test_distance_and_energy_match_explicit_formula(self):
        a, b = torch.randn(2, 4, 4), torch.randn(2, 4, 4)
        time = ((a-b)**2).sum((1, 2)).sqrt()/4
        sa = torch.log1p(torch.fft.rfft(a, dim=1, norm='ortho').abs())
        sb = torch.log1p(torch.fft.rfft(b, dim=1, norm='ortho').abs())
        expected = time+((sa-sb)**2).sum((1, 2)).sqrt()/(12**.5)
        torch.testing.assert_close(distance(a, b), expected)
        self.assertGreater(float(distance(torch.ones_like(a), torch.zeros_like(a)).min()), 0)
        m, x = self.model(), torch.randn(2, 52, 9)
        n, p, _ = self.noise()
        past, target = m.split(x)
        draws, pair = m.generate(m.context(past), n), m.generate(m.context(past), p)
        explicit = torch.stack([distance(target, draws[:, k])-.5*distance(draws[:, k], pair[:, k]) for k in range(len(n))]).mean(0)
        torch.testing.assert_close(m.energy(x, n, p), explicit)

    def test_target_never_enters_generated_reference(self):
        m, x = self.model(), torch.randn(2, 52, 9)
        y = x.clone()
        y[:, 48:] += 100
        n = self.noise()[0]
        torch.testing.assert_close(m.generate(m.context(m.split(x)[0]), n), m.generate(m.context(m.split(y)[0]), n), rtol=0, atol=0)
        y = x.clone()
        y[:, 48:, 4:] += 100
        torch.testing.assert_close(m.components(x, *self.noise())['score'], m.components(y, *self.noise())['score'], rtol=0, atol=0)

    def test_masked_feedback_zero_grad_but_context_and_generator_learn(self):
        for variant in ['context_only', 'response_feedback']:
            m = self.model(variant)
            x = torch.randn(3, 52, 9, requires_grad=True)
            n, p, _ = self.noise()
            m.energy(x, n, p).mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(float(x.grad[:, :48, 4:].abs().sum()), 0)
            self.assertGreater(float(m.generator[-1].weight.grad.abs().sum()), 0)
            self.assertEqual(float(x.grad[:, 48:, 4:].abs().sum()), 0)
            if variant == 'context_only':
                self.assertEqual(float(x.grad[:, :48, :4].abs().sum()), 0)
                y = x.detach().clone()
                y[:, :48, :4] += 100
                torch.testing.assert_close(m.context(x.detach()[:, :48]), m.context(y[:, :48]), rtol=0, atol=0)
            else:
                self.assertGreater(float(x.grad[:, :48, :4].abs().sum()), 0)

    def test_initial_tensor_identity_and_effective_receptive_field(self):
        a, b = self.model(), self.model('response_feedback')
        for key, value in a.state_dict().items():
            self.assertTrue(torch.equal(value, b.state_dict()[key]), key)
        x = torch.randn(2, 48, 9)
        y = x.clone()
        y[:, :17] += 100
        torch.testing.assert_close(b.context(x), b.context(y), rtol=0, atol=0)

    def test_score_tail_count_and_batch_order(self):
        m, x = self.model(), torch.randn(4, 52, 9)
        with torch.no_grad():
            a = m.components(x, *self.noise())
            b = m.components(x.flip(0), *self.noise())
            singles = [m.components(row[None], *self.noise()) for row in x]
        expected = (1+a['normal_tail_count'].float())/32
        torch.testing.assert_close(a['normal_tail_probability'], expected, rtol=0, atol=0)
        torch.testing.assert_close(a['score'], -expected.log(), rtol=0, atol=0)
        self.assertTrue(torch.equal(a['normal_tail_count'], b['normal_tail_count'].flip(0)))
        self.assertTrue(torch.equal(a['normal_tail_count'], torch.cat([s['normal_tail_count'] for s in singles])))
        torch.testing.assert_close(a['observed_energy'], torch.cat([s['observed_energy'] for s in singles]), atol=1e-6, rtol=1e-6)
        self.assertTrue(torch.isfinite(a['normal_energy']).all())

    def test_validation_restores_heterogeneous_modes_and_rng_on_error(self):
        m = self.model()
        m.train()
        m.generator.eval()
        modes = [module.training for module in m.modules()]
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with validation_mode(m, torch.device('cpu')):
                self.assertTrue(all(not module.training for module in m.modules()))
                torch.randn(8)
                raise RuntimeError('intentional')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(modes, [module.training for module in m.modules()])

    def test_fixed_banks_do_not_consume_training_rng(self):
        state = torch.get_rng_state().clone()
        cfg = dict(score_reference_pairs=64, score_normal_probes=255, score_noise_seeds=[84553, 84554, 84555])
        a, b, c = score_banks(cfg, torch.device('cpu'))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertFalse(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c[:64]))


if __name__ == '__main__':
    unittest.main()
