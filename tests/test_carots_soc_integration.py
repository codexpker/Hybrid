"""Small causal feature/augmentation/RNG checks, no research-model training."""
import importlib.util
from pathlib import Path
import random
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybrid_tsad.hybrid_v1.carots_soc_control import (
    relation_views, normal_relation_controls, fit_representation_calibration, representation_scores)


class ArrayChecks(unittest.TestCase):
    def test_drift_and_hard_controls_preserve_signed_lag12_response_correction(self):
        x=np.zeros((2560,3),dtype=np.float32);x[:,0]=2
        controls=normal_relation_controls(x)
        for y in controls.values():
            delta=y[:,0]-x[:,0]
            self.assertTrue(np.allclose(y[:,1],.3*np.r_[np.zeros(12),delta[:-12]],atol=1e-7))
            self.assertTrue(np.allclose(y[:,2],-.3*np.r_[np.zeros(12),delta[:-12]],atol=1e-7))
        drift=controls['NORMAL_DRIFT'][:,0]-2
        hard=controls['HARD_NORMAL'][:,0]-2
        self.assertTrue(np.array_equal(drift[:768],np.zeros(768)))
        self.assertTrue(np.array_equal(drift[1536:],np.ones(1024)))
        self.assertTrue(np.all(np.diff(drift[768:1537])>=0))
        self.assertTrue(np.array_equal(hard[:768],np.zeros(768)))
        self.assertTrue(np.array_equal(hard[1536:],np.zeros(1024)))
        self.assertEqual(hard[1152],2.)

    def test_views_keep_response_and_input_bytes_and_replace_only_driver(self):
        x = np.arange(48, dtype=np.float32).reshape(16, 3)/13
        v = x[::-1].copy()+.7
        u = np.stack((np.arange(16, dtype=np.float32)/3,
                      -np.arange(16, dtype=np.float32)/7))
        copies = [z.copy() for z in (x, v, u)]
        actual = relation_views(x, v, u)
        self.assertEqual(actual.shape, (4, 16, 3))
        self.assertTrue(np.array_equal(actual[0], x))
        self.assertTrue(np.allclose(actual[1], (x+v)/np.sqrt(2.), atol=1e-6))
        self.assertTrue(np.array_equal(actual[0, :, 1:], actual[2, :, 1:]))
        self.assertTrue(np.array_equal(actual[1, :, 1:], actual[3, :, 1:]))
        self.assertTrue(np.array_equal(actual[2, :, 0], u[0]))
        self.assertTrue(np.array_equal(actual[3, :, 0], u[1]))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip((x, v, u), copies)))

    def test_same_layout_calibration_and_score_replay(self):
        rng = np.random.default_rng(63141)
        x = rng.standard_normal((512, 16))+.4
        c = fit_representation_calibration(x)
        score = representation_scores(x, c)
        self.assertEqual(c["valid_windows"], 512)
        self.assertAlmostEqual(c["threshold"], np.sort(score)[506], places=14)
        self.assertTrue(np.array_equal(score, representation_scores(x, c)))
        self.assertTrue(((score >= -1e-12)&(score <= 1+1e-12)).all())
        with self.assertRaises(ValueError):
            representation_scores(x[:, :-1], c)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Author Torch CPU test required")
class TorchChecks(unittest.TestCase):
    def test_four_eval_with_grad_calls_restore_modes_and_propagate_soc(self):
        import torch
        from types import SimpleNamespace
        from hybrid_tsad.hybrid_v1.carots_soc_control import soc_from_views
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.config=SimpleNamespace(history=16)
                self.proj=torch.nn.Linear(16,4,bias=False)
                self.dropout=torch.nn.Dropout(.2);self.calls=[]
            def forecast(self, history, observed, mask_seed=0):
                self.calls.append(dict(training=self.training, child=self.proj.training,
                    grad_enabled=torch.is_grad_enabled(), shape=list(history.shape), seed=mask_seed))
                t=self.proj(history.transpose(1,2))
                return dict(t_features=t,f_features=torch.sin(t)+.8)
        torch.manual_seed(63141);model=Toy();model.train();model.proj.eval()
        rng=np.random.default_rng(63141)
        x=rng.standard_normal((16,3)).astype(np.float32)
        partner=rng.standard_normal((16,3)).astype(np.float32)
        drivers=rng.standard_normal((2,16)).astype(np.float32)
        views=relation_views(x,partner,drivers)
        before=torch.get_rng_state().clone();modes=[m.training for m in model.modules()]
        loss, meta, features=soc_from_views(model,views,seed=731000)
        self.assertEqual(len(model.calls),4)
        self.assertTrue(all(not r['training'] and not r['child'] and r['grad_enabled']
                            and r['shape']==[1,16,3] for r in model.calls))
        self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(features.shape,(4,24))
        self.assertFalse(meta['skipped'])
        loss.backward()
        self.assertTrue(torch.isfinite(model.proj.weight.grad).all())
        self.assertGreater(float(model.proj.weight.grad.norm()),0)

    def test_channel_layout_and_soc_extra_rng_restore(self):
        import torch
        from hybrid_tsad.hybrid_v1.carots_soc_control import forecast_features, soc_randomness
        torch.set_num_threads(1)
        t = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4).requires_grad_()
        f = t+100
        output = forecast_features({"t_features": t, "f_features": f})
        self.assertTrue(torch.equal(output, torch.cat((t.flatten(1), f.flatten(1)), 1)))
        output.sum().backward()
        self.assertTrue(torch.equal(t.grad, torch.full_like(t, 2)))
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.1))
        model.train();model[0].eval()
        states = [m.training for m in model.modules()]
        before = torch.get_rng_state().clone()
        py = random.getstate();np_before = np.random.get_state()
        with soc_randomness(model, 63141):
            torch.rand(5);random.random();np.random.rand(5)
            model.eval()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(states, [m.training for m in model.modules()])
        self.assertEqual(py, random.getstate())
        np_after = np.random.get_state()
        self.assertTrue(np.array_equal(np_before[1], np_after[1]))
        self.assertEqual(np_before[0], np_after[0])
        self.assertEqual(np_before[2:], np_after[2:])
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
