"""Bounded SOC control tests. Torch tests must run in the author CPU env."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
VENDOR = ROOT / "src/hybrid_tsad/experimental_controls/carots_soc"


def fixture():
    return np.array([[1., .1, .2], [.9, .2, .1], [1.1, -.1, .2],
                     [-.8, .5, .2], [-.7, -.4, .1], [-.9, .2, -.2]])


def reference_loop(x, threshold=.5, temperature=.2):
    z = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n = len(x) // 2
    sim = z @ z.T
    losses = []
    for i in range(n):
        pos = [j for j in range(n) if sim[i, j] >= threshold]
        negatives = sum(np.exp(sim[i, j+n]/temperature) for j in pos)
        for j in pos:
            if j == i:
                continue
            numerator = np.exp(sim[i, j]/temperature)
            losses.append(-np.log(numerator/(numerator+negatives)+1e-12))
    return float(np.mean(losses)), len(losses)


class ProvenanceAndNumpy(unittest.TestCase):
    def test_upstream_and_license_exact(self):
        origin = json.loads((VENDOR / "ORIGIN.json").read_text())
        for entry in origin["files"]:
            self.assertEqual(hashlib.sha256((VENDOR / entry["local"]).read_bytes()).hexdigest(),
                             entry["sha256"])
        self.assertIn("Non-Commercial", (VENDOR / "LICENSE").read_text())

    def test_independent_loop_matches_vectorized_author_expression(self):
        x = fixture()
        z = x / np.linalg.norm(x, axis=1, keepdims=True)
        n = len(x)//2
        sim = z @ z.T
        positive = sim[:n, :n].copy()
        low = positive < .5
        a, b = np.where(low)
        sim[a, b] = -np.inf
        sim[a, b+n] = -np.inf
        s = np.exp(sim/.2)
        s -= np.diag(np.diagonal(s))
        denominator = s + s[:, n:].sum(axis=1, keepdims=True)
        loss = -np.log(s/denominator+1e-12)
        np.fill_diagonal(loss, 0)
        actual = loss[np.where(positive >= .5)].sum() / (int((positive >= .5).sum())-n)
        expected, count = reference_loop(x)
        self.assertEqual(count, 6)
        self.assertAlmostEqual(actual, expected, places=14)
        self.assertGreater(expected, 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch unavailable; author CPU run required")
class ActualTorch(unittest.TestCase):
    def test_formula_equivalence_gradients_and_identity(self):
        import torch
        from types import SimpleNamespace
        from hybrid_tsad.experimental_controls.carots_soc import soc_objective
        from hybrid_tsad.experimental_controls.carots_soc.upstream_loss import loss_fn
        torch.set_num_threads(1)
        self.assertFalse(torch.cuda.is_initialized())
        x = torch.tensor(fixture(), dtype=torch.float64, requires_grad=True)
        saved = x.detach().clone()
        state = torch.get_rng_state().clone()
        out, meta = soc_objective(x, enabled=True, feature_identity="fixture_endpoint_v1",
                                  similarity_threshold=.5, temperature=.2)
        cfg = SimpleNamespace(CAROTS=SimpleNamespace(SIM_THRESHOLD=.5, TEMPERATURE=.2))
        expected = loss_fn(x, cfg)
        self.assertTrue(torch.equal(out, expected))
        self.assertAlmostEqual(out.item(), reference_loop(fixture())[0], places=12)
        self.assertEqual(meta["valid_nonself_positive_pairs"], 6)
        self.assertEqual(meta["projector"], "identity_no_parameters")
        self.assertTrue(torch.equal(x.detach(), saved))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        out.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.norm()), 0)
        self.assertLess(float(x.grad.norm()), 1000)
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_positive_and_empty_support_have_differentiable_zero(self):
        import torch
        from hybrid_tsad.experimental_controls.carots_soc import soc_objective
        for x, reason in [(torch.eye(4, dtype=torch.float64, requires_grad=True),
                           "no_nonself_positive_pair"),
                          (torch.empty(0, 3, dtype=torch.float64, requires_grad=True),
                           "empty_support")]:
            loss, meta = soc_objective(x, enabled=True, feature_identity="fixed_empty",
                                      similarity_threshold=.9)
            self.assertEqual(loss.item(), 0.)
            self.assertTrue(meta["skipped"])
            self.assertEqual(meta["reason"], reason)
            loss.backward()
            self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))

    def test_disabled_and_invalid_inputs(self):
        import torch
        from hybrid_tsad.experimental_controls.carots_soc import soc_objective
        x = torch.tensor(fixture(), dtype=torch.float64, requires_grad=True)
        loss, meta = soc_objective(x, feature_identity="fixed_endpoint")
        self.assertEqual(meta["reason"], "disabled")
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x)))
        for kwargs in [dict(feature_identity=""),
                       dict(feature_identity="fixed", temperature=0),
                       dict(feature_identity="fixed", similarity_threshold=1),
                       dict(feature_identity="fixed", enabled="yes")]:
            with self.assertRaises((ValueError, TypeError)):
                soc_objective(x, **kwargs)
        with self.assertRaises(ValueError):
            soc_objective(torch.full((4, 3), float("nan")), enabled=True,
                          feature_identity="fixed")
        with self.assertRaises(ValueError):
            soc_objective(x[:3], enabled=True, feature_identity="fixed")
        with self.assertRaisesRegex(ValueError, "nonzero embedding norms"):
            soc_objective(torch.zeros((4, 3), dtype=torch.float64), enabled=True,
                          feature_identity="fixed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
