"""Synthetic interface tests; fake lag is explicitly a fixture, not a model."""
import os
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from hybrid_tsad.candidate_tf_lag import (CandidateConfig, TFLagCandidate,
    candidate_objective, forecast_coordinates, load_official, residual_scores, candidate_from_config)
from hybrid_tsad.candidate_autoformer_lag import AutoformerLagEndpoint


class FixtureTF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.ones(cfg.tf_repr))
        self.decoder = nn.Linear(1, 1)

    def forward(self, x, **kwargs):
        b, h, c = x.shape
        # Native-domain identity is deliberately not made into time coordinates.
        a = x.transpose(1, 2).mean(-1).reshape(b*c, 1, 1)
        re = a * self.weight[None, None]
        re = re.expand(b*c, h//self.cfg.patch_len, self.cfg.tf_repr)
        z = torch.complex(re, re * .25).repeat(self.cfg.copies, 1, 1)
        return z, x[None].expand(self.cfg.copies, -1, -1, -1)


class FixtureLag(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embed = nn.Linear(1, width)

    def forward(self, x):
        return self.embed(x[:, -1, :, None]), {"fixture_only": True}


PROVENANCE = {"repository": "SYNTHETIC_FIXTURE_NOT_PAPER", "commit": "fixture",
              "class": "FixtureLag", "source_sha256": "fixture"}


def fixture(mode):
    cfg = CandidateConfig(mode=mode, history=20, width=8,
                          tf_repr=8, tf_depth=1, tf_hidden=8)
    with patch("hybrid_tsad.candidate_tf_lag.load_official", lambda p, c: FixtureTF(c)):
        return TFLagCandidate(cfg, "fixture", lag_encoder=FixtureLag(8), lag_provenance=PROVENANCE)


class TestCandidate(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2025)

    def test_all_modes_forward_backward_shapes_and_finite(self):
        for mode in ("tf", "lag", "concat", "gated"):
            m = fixture(mode)
            x = torch.randn(2, 20, 3)
            y = torch.randn(2, 1, 3)
            out = m(x)
            self.assertEqual(out["prediction"].shape, (2, 1, 3))
            loss = candidate_objective(out, y)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(bool(m.head.weight.grad.abs().max() > 0))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_no_automatic_lag_fallback(self):
        for mode in ("lag", "concat", "gated"):
            with self.assertRaisesRegex(ValueError, "no automatic fallback"):
                TFLagCandidate(CandidateConfig(mode=mode))

    def test_future_not_in_forward_and_score_does_not_change_model(self):
        m = fixture("gated").eval()
        x = torch.randn(1, 20, 3)
        a = m(x)
        state = {k: v.clone() for k, v in m.state_dict().items()}
        scores1 = residual_scores(a, torch.zeros(1, 1, 3))
        scores2 = residual_scores(a, torch.ones(1, 1, 3) * 100)
        self.assertFalse(torch.equal(scores1["continuous"], scores2["continuous"]))
        b = m(x)
        torch.testing.assert_close(a["prediction"], b["prediction"], rtol=0, atol=0)
        self.assertTrue(all(torch.equal(v, m.state_dict()[k]) for k, v in state.items()))
        with self.assertRaises(TypeError):
            m(x, torch.ones(1, 1, 3))

    def test_channel_count_reuses_parameters_without_channel_names(self):
        m = fixture("gated")
        keys = tuple(m.state_dict())
        for channels in (1, 3, 7):
            self.assertEqual(m(torch.randn(2, 20, channels))["prediction"].shape, (2, 1, channels))
            self.assertEqual(keys, tuple(m.state_dict()))

    def test_nonfinite_input_and_wrong_dtype_rejected(self):
        m = fixture("lag")
        for x in (torch.full((1, 20, 3), float("nan")), torch.randn(1, 20, 3).double()):
            with self.assertRaises(ValueError):
                m(x)

    def test_target_shapes_and_nonfinite_rejected(self):
        out = fixture("lag")(torch.randn(1, 20, 3))
        for y in (torch.randn(1, 2, 3), torch.full((1, 1, 3), float("inf"))):
            with self.assertRaises(ValueError):
                residual_scores(out, y)

    def test_lag_shape_is_not_silently_broadcast(self):
        class BadLag(nn.Module):
            def forward(self, x):
                return x[:, :1], {}
        m = fixture("lag")
        m.lag = BadLag()
        with self.assertRaises(RuntimeError):
            m(torch.randn(2, 20, 3))

    def test_active_parameters_and_stage_interfaces(self):
        m = fixture("tf")
        self.assertGreater(m.parameter_counts()["stored"], m.parameter_counts()["active"])
        self.assertTrue(all(not p.requires_grad for p in m.tf.decoder.parameters()))
        self.assertEqual(m.adaptation_parameters("Z"), ())
        self.assertTrue(m.adaptation_parameters("U"))
        with self.assertRaises(NotImplementedError):
            m.adaptation_parameters("F")

    def test_config_and_physical_score_coordinates(self):
        for changes in ({"history": 99}, {"copies": 1}, {"width": 0}):
            with self.assertRaises(ValueError):
                CandidateConfig(**changes)
        self.assertEqual(forecast_coordinates(200, 100, 2), {
            "input_start": 200, "input_end_inclusive": 299,
            "target_indices": [300, 301], "score_available_indices": [300, 301]})

    def test_official_wrong_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "not_official.py"
            p.write_text("raise RuntimeError('must not import')")
            with self.assertRaisesRegex(ValueError, "byte identity"):
                load_official(p, CandidateConfig())

    def test_optional_checkpoint_strict_prevalidation(self):
        m = fixture("tf")
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "synthetic.pt"
            state = {k: v.clone() for k, v in m.tf.state_dict().items()}
            torch.save(state, p)
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertTrue(m.load_tf_checkpoint(p, expected_sha256=digest)["strict"])
            before = {k: v.clone() for k, v in m.state_dict().items()}
            state["unknown_tau"] = torch.tensor(1.)
            torch.save(state, p)
            with self.assertRaisesRegex(ValueError, "missing/unexpected"):
                m.load_tf_checkpoint(p, expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertTrue(all(torch.equal(v, m.state_dict()[k]) for k, v in before.items()))

    @unittest.skipUnless(os.environ.get("CANDIDATE_TIMERADAR_FILE"), "External official file not supplied")
    def test_actual_official_module_tiny_synthetic_cpu(self):
        cfg = CandidateConfig(mode="tf", history=20, width=8,
                              tf_repr=8, tf_hidden=8, tf_depth=1)
        m = TFLagCandidate(cfg, os.environ["CANDIDATE_TIMERADAR_FILE"]).eval()
        x = torch.randn(2, 20, 3)
        out = m(x)
        loss = candidate_objective(out, torch.randn(2, 1, 3))
        loss.backward()
        self.assertEqual(out["prediction"].shape, (2, 1, 3))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(m.tf.real_input_embed.weight.grad)
        self.assertTrue(bool(m.tf.real_input_embed.weight.grad.abs().max() > 0))


@unittest.skipUnless(os.environ.get("CANDIDATE_AUTOFORMER_FILE"), "Official Autoformer file not supplied")
class TestOfficialLag(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2025)
        self.path = os.environ["CANDIDATE_AUTOFORMER_FILE"]

    def test_pair_endpoint_matches_direct_author_calls(self):
        m = AutoformerLagEndpoint(self.path, width=8, heads=2).eval()
        x = torch.randn(2, 20, 3)
        state, info = m(x)
        tokens = m.embed(x.transpose(1, 2)[..., None])
        flat = tokens.reshape(6, 20, 8)
        direct, _ = m.self_operator(flat, flat, flat, None)
        torch.testing.assert_close(info["self_endpoint"], direct[:, -1].reshape(2, 3, 8), rtol=0, atol=0)
        cross = []
        for i in range(3):
            from_sources = []
            for j in range(3):
                if i != j:
                    out, _ = m.cross_operator(tokens[:, i], tokens[:, j], tokens[:, j], None)
                    from_sources.append(out[:, -1])
            cross.append(torch.stack(from_sources, dim=1).mean(1))
        torch.testing.assert_close(info["cross_endpoint"], torch.stack(cross, dim=1), rtol=1e-5, atol=1e-6)
        self.assertEqual(state.shape, (2, 3, 8))
        self.assertEqual(info["author_selection"], "per_expanded_sample_topk")

    def test_train_shared_topk_preserved_and_gradient_finite(self):
        m = AutoformerLagEndpoint(self.path, width=8, heads=2).train()
        out, info = m(torch.randn(2, 20, 3))
        out.square().mean().backward()
        self.assertEqual(info["author_selection"], "batch_shared_topk")
        for name in ("self_operator", "cross_operator"):
            grads = [p.grad for p in getattr(m, name).parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(bool(g.abs().max() > 0) for g in grads))

    def test_circular_delay_endpoint_not_full_time_pool(self):
        m = AutoformerLagEndpoint(self.path, width=8, heads=2).eval()
        op = m.self_operator.inner_correlation
        values = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
        corr = torch.zeros_like(values)
        corr[..., 2] = 1000
        out = op.time_delay_agg_inference(values, corr)
        torch.testing.assert_close(out[..., -1], torch.tensor([[[1.]]]), rtol=0, atol=0)
        torch.testing.assert_close(out.mean(-1), values.mean(-1), rtol=0, atol=0)

    def test_switches_single_channel_and_pair_budget(self):
        for enabled in ((True, False), (False, True), (True, True)):
            m = AutoformerLagEndpoint(self.path, width=8, heads=2,
                    self_enabled=enabled[0], cross_enabled=enabled[1])
            self.assertEqual(m(torch.randn(2, 20, 1))[0].shape, (2, 1, 8))
        with self.assertRaises(ValueError):
            AutoformerLagEndpoint(self.path, self_enabled=False, cross_enabled=False)
        m = AutoformerLagEndpoint(self.path, width=8, heads=2, max_expanded_pairs=1)
        with self.assertRaisesRegex(ValueError, "no silent chunking"):
            m(torch.randn(2, 20, 3))

    @unittest.skipUnless(os.environ.get("CANDIDATE_TIMERADAR_FILE"), "Official TF file not supplied")
    def test_each_forecast_invariant_to_other_batch_history_train_and_eval(self):
        for mode in ("tf", "lag", "concat", "gated"):
            lag = AutoformerLagEndpoint(self.path, width=8, heads=2)
            cfg = CandidateConfig(mode=mode, history=20, width=8, tf_repr=8, tf_hidden=8, tf_depth=1)
            m = TFLagCandidate(cfg, os.environ["CANDIDATE_TIMERADAR_FILE"],
                              lag_encoder=lag, lag_provenance=lag.provenance)
            x = torch.randn(2, 20, 3)
            for training in (True, False):
                m.train(training)
                for protected in (0, 1):
                    changed = x.clone()
                    # A non-affine change survives per-window normalization.
                    changed[1-protected] = torch.randn_like(changed[1-protected]) * 100
                    torch.manual_seed(774)
                    first = m(x)["prediction"][protected].detach()
                    torch.manual_seed(774)
                    second = m(changed)["prediction"][protected].detach()
                    torch.testing.assert_close(first, second, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("CANDIDATE_TIMERADAR_FILE"), "Official TF file not supplied")
    def test_real_both_official_modules_all_four_modes(self):
        for mode in ("tf", "lag", "concat", "gated"):
            lag = AutoformerLagEndpoint(self.path, width=8, heads=2)
            cfg = CandidateConfig(mode=mode, history=20, width=8, tf_repr=8, tf_hidden=8, tf_depth=1)
            m = TFLagCandidate(cfg, os.environ["CANDIDATE_TIMERADAR_FILE"],
                              lag_encoder=lag, lag_provenance=lag.provenance)
            out = m(torch.randn(2, 20, 3))
            loss = candidate_objective(out, torch.randn(2, 1, 3))
            loss.backward()
            self.assertEqual(out["prediction"].shape, (2, 1, 3))
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(bool(m.head.weight.grad.abs().max() > 0))

    @unittest.skipUnless(os.environ.get("CANDIDATE_CONFIG_FILE"), "Candidate config not supplied")
    def test_config_factory_real_official_sources(self):
        m = candidate_from_config(os.environ["CANDIDATE_CONFIG_FILE"], mode="gated").eval()
        out = m(torch.randn(1, m.config.history, 3))
        self.assertEqual(out["prediction"].shape, (1, m.config.horizon, 3))
        self.assertEqual(m.lag.provenance["commit"], "51c7d416ae120b805fd5beef2f4ccf7de496a6ff")


if __name__ == "__main__":
    unittest.main()
