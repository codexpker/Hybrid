"""F03 CPU/synthetic regression tests; no author sources or research training."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import os
import numpy as np
import torch
from torch import nn

from hybrid_tsad.hybrid_v1 import training as legacy
from hybrid_tsad.hybrid_v1 import runtime_state as safe
from hybrid_tsad.hybrid_v1 import stage2_u_matched as matched
from hybrid_tsad.hybrid_v1.data import Run
from hybrid_tsad.hybrid_v1.fractional import FractionalBackbone


class ValidationState(unittest.TestCase):
    def setUp(self):
        self.model = nn.Sequential(nn.Linear(1, 1), nn.Dropout(.2))
        self.model.train(); self.model[1].eval()
        self.run = Run('synthetic', 'entity', 'run', 'SELECT', np.zeros((10, 1)), np.ones((10, 1), bool))

    def test_legacy_validation_restores_modes_rng_on_success_and_exception(self):
        for failure in (False, True):
            before = [m.training for m in self.model.modules()]
            rng = torch.get_rng_state().clone()
            def passes(*args):
                self.assertTrue(all(not m.training for m in self.model.modules()))
                self.assertFalse(torch.is_grad_enabled())
                torch.rand(3)
                if failure:
                    raise RuntimeError('synthetic forward failure')
                return {}, {}
            with patch.object(legacy, 'passes', side_effect=passes), patch.object(legacy, 'ssl_objective', return_value={'valid_windows': torch.tensor([True]), 'total': torch.tensor(2.)}):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, 'synthetic forward'):
                        legacy.validation(self.model, [self.run], 4, 1, 'huber', 17, limit=1)
                else:
                    self.assertEqual(legacy.validation(self.model, [self.run], 4, 1, 'huber', 17, limit=1), 2.)
            self.assertEqual(before, [m.training for m in self.model.modules()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_empty_validation_and_bad_plan_exit_restore(self):
        flags = [m.training for m in self.model.modules()]
        rng = torch.get_rng_state().clone()
        for plan in (None, [{'not': 'a valid plan'}]):
            with self.assertRaises((ValueError, KeyError)):
                legacy.validation(self.model, [], 4, 1, 'huber', 5, plan=plan)
            self.assertEqual(flags, [m.training for m in self.model.modules()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_new_helper_and_unchanged_matched_U_restore_nested_flags(self):
        for guard in (safe.evaluation_state, matched.evaluation_state):
            flags = [m.training for m in self.model.modules()]
            rng = torch.get_rng_state().clone()
            with self.assertRaises(RuntimeError):
                with guard(self.model, 5):
                    torch.rand(5)
                    with guard(self.model, 6):
                        torch.rand(5)
                    raise RuntimeError('fixture')
            self.assertEqual(flags, [m.training for m in self.model.modules()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_only_forked_cuda_generators_seeded_API_test_not_real_CUDA(self):
        with patch.object(torch.random, 'fork_rng', return_value=nullcontext()) as fork, patch.object(torch.cuda, 'device', return_value=nullcontext()) as device, patch.object(torch.cuda, 'manual_seed') as local_seed, patch.object(torch.cuda, 'manual_seed_all') as all_seed:
            with safe.isolated_rng(17, [2]):
                pass
            fork.assert_called_once_with(devices=[2], enabled=True)
            device.assert_called_once_with(2)
            local_seed.assert_called_once_with(17)
            all_seed.assert_not_called()

    def test_rng_device_discovery_includes_buffers(self):
        fixture = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=torch.device('cuda:1'))]), buffers=lambda: iter([SimpleNamespace(device=torch.device('cuda:2'))]))
        self.assertEqual(safe.cuda_devices(fixture), [1, 2])

    @unittest.skipUnless(os.environ.get('HYBRID_AUDIT_CUDA_RNG') == '1' and torch.cuda.is_available(),
                         'Opt-in minimal CUDA RNG check: set HYBRID_AUDIT_CUDA_RNG=1 on a free local device')
    def test_optional_real_CUDA_all_visible_states_unchanged(self):
        model = self.model.cuda(0)
        for failure in (False, True):
            before = [x.clone() for x in torch.cuda.get_rng_state_all()]
            flags = [m.training for m in model.modules()]
            try:
                with safe.evaluation_state(model, 17):
                    torch.rand(3, device='cuda:0')
                    if failure:
                        raise RuntimeError('fixture')
            except RuntimeError:
                self.assertTrue(failure)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, torch.cuda.get_rng_state_all())))
            self.assertEqual(flags, [m.training for m in model.modules()])

    def test_fractional_copy_masks_CPU_rng_and_active_outputs_preserved(self):
        class Author(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.real_input_embed = self.imag_input_embed = nn.Identity()
                self.encoder = lambda real, imag, order: torch.complex(real, imag)
        sources = SimpleNamespace(timeradar=lambda: SimpleNamespace(Model=Author, frft=lambda x, order, dim: x, ifrft=lambda x, order, dim: x))
        cfg = SimpleNamespace(local_window=4, copies=2, tf_patch=2, tf_repr=2, tf_hidden=2, tf_depth=1, width=2, fixed_order=.5)
        model = FractionalBackbone(cfg, sources)
        x, visible = torch.ones(1, 4, 1), torch.ones(1, 4, 1, dtype=torch.bool)
        before = torch.get_rng_state().clone()
        with patch.object(torch.cuda, 'manual_seed_all') as global_seed:
            a = model.encode(x, visible, mask_seed=7)
            b = model.encode(x, visible, mask_seed=7)
            global_seed.assert_not_called()
        self.assertTrue(torch.equal(a['transform_hidden_mask'], b['transform_hidden_mask']))
        self.assertTrue(torch.equal(a['tokens'], b['tokens']))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        # CPU masks are identical to the former single-device seeding sequence.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            first = torch.rand(1, 2) < .5
        expected = torch.cat((first, ~first), 0).reshape(1, 2, 1, 2)
        self.assertTrue(torch.equal(a['transform_hidden_mask'], expected))


if __name__ == '__main__':
    unittest.main()
