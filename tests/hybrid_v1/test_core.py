"""Real cached author modules, synthetic inputs only. No checkpoints or data."""
from dataclasses import replace
import os
from pathlib import Path
import unittest

import torch

from hybrid_tsad.hybrid_v1 import HybridV1, HybridV1Config, OfficialSources
from hybrid_tsad.hybrid_v1.lag import FiniteLag
from hybrid_tsad.hybrid_v1.model import visible_normalize
from hybrid_tsad.hybrid_v1.multiview import SelectedFusion


def tiny(**updates):
    cfg = HybridV1Config(history=128, local_window=20, horizon=2, width=16,
        temporal_patch=16, temporal_layers=1, heads=2, tf_hidden=8, tf_repr=16,
        tf_depth=1, tf_patch=5, lag_window=20, max_lag=8, lag_width=8,
        lag_top_k=2, history_mask_probability=0, dropout=0)
    return replace(cfg, **updates)


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.sources = OfficialSources.from_workspace(os.environ["HYBRID_V1_WORKSPACE"])

    def setUp(self):
        torch.manual_seed(937)

    def model(self, **changes):
        return HybridV1(tiny(**changes), self.sources)

    def test_01_real_sources_and_eager_parameters(self):
        m = self.model()
        self.assertEqual(len(m.source_manifest), 8)
        self.assertIn("_hybrid_v1_timesfm_", type(m.temporal.encoder).__module__)
        self.assertIn("_hybrid_v1_left_", type(m.ms_filter.author).__module__)
        before = {n: id(p) for n, p in m.named_parameters()}
        m.eval().forecast(torch.randn(1, 128, 3))
        self.assertEqual(before, {n: id(p) for n, p in m.named_parameters()})
        self.assertEqual(m.event_head.in_features, 35)

    def test_02_native_author_abs_parity(self):
        m = self.model().eval()
        x = torch.randn(1, 128, 2)
        e = m.tf_backbone.encode(x, torch.ones_like(x, dtype=torch.bool), mask_seed=59)
        torch.manual_seed(59)
        rep, out = m.tf_backbone.author(x[:, -20:], norm=0, copies=2, mask_mode="c")
        torch.testing.assert_close(e["native"].reshape_as(rep), rep, atol=0, rtol=0)
        actual = m.tf_backbone.decode(e["native"], e["order"], projection="author_abs")
        torch.testing.assert_close(actual, out.permute(1, 0, 2, 3), atol=0, rtol=0)

    def test_03_signed_integer_inverse_and_noninteger_characterization(self):
        m = self.model().eval()
        x = torch.linspace(-2, 2, 20).reshape(1, -1).to(torch.complex64)
        for order in (0., 1.):
            y = m.tf_backbone.ifrft(m.tf_backbone.frft(x, order, dim=-1), order, dim=-1)
            torch.testing.assert_close(y, x, rtol=2e-5, atol=2e-5)
        self.assertTrue(bool((y.real < 0).any()))
        order = torch.tensor(.43)
        actual = m.tf_backbone.ifrft(m.tf_backbone.frft(x, order, dim=-1), order, dim=-1)
        upstream = self.sources.timeradar()
        reference = upstream.ifrft(upstream.frft(x, order, dim=-1), order, dim=-1)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        # Keep the original counterexample. A pass means faithful operator
        # characterization, NEVER arbitrary finite-signal exact inversion.
        self.assertGreater(float((actual - x).abs().max()), .1)

    def test_04_real_inverse_differs_from_abs(self):
        m = self.model().eval()
        e = m.tf_backbone.encode(torch.randn(1, 128, 2), torch.ones(1, 128, 2, dtype=torch.bool), mask_seed=1)
        real = m.tf_backbone.decode(e["native"], e["order"], projection="real")
        amp = m.tf_backbone.decode(e["native"], e["order"], projection="author_abs")
        self.assertFalse(torch.equal(real, amp))
        self.assertTrue(bool((amp >= 0).all()))

    def test_05_left_exact_no_padding(self):
        fusion = SelectedFusion(tiny(), self.sources).eval()
        t, f, ms = torch.randn(2, 8, 16), torch.randn(2, 4, 16), torch.randn(2, 3, 16)
        expected = fusion.author(t, f, ms)
        actual = fusion(t, f, ms, torch.ones(2, 8, dtype=torch.bool),
            torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool))
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=0, rtol=0)

    def test_06_left_padding_and_empty_keys(self):
        fusion = SelectedFusion(tiny(), self.sources).eval()
        t, f, ms = torch.randn(2, 8, 16), torch.randn(2, 4, 16), torch.randn(2, 3, 16)
        tv, fv, mv = torch.ones(2, 8, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool)
        tv[0, 3:] = False; fv[0] = False; mv[1] = False
        out = fusion(t, f, ms, tv, fv, mv)
        for value, valid in zip(out, (tv, fv, mv)):
            self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertTrue(bool((value[~valid] == 0).all()))

    def test_07_temporal_real_causal_mask(self):
        m = self.model().eval()
        x = torch.randn(1, 128, 2); visible = torch.ones_like(x, dtype=torch.bool)
        first, _, info = m.temporal(x, visible)
        changed = x.clone(); changed[:, 64:] += 30
        second, _, _ = m.temporal(changed, visible)
        torch.testing.assert_close(first[:, :, :4], second[:, :, :4], atol=0, rtol=0)
        mask = info["attention_masks"][0]
        self.assertFalse(bool(mask[..., 0, 1:].any()))

    def test_08_visible_only_statistics(self):
        x = torch.randn(2, 128, 3); visible = torch.ones_like(x, dtype=torch.bool)
        visible[:, -1] = False
        a, sa = visible_normalize(x, visible)
        x[:, -1] = float("nan")
        b, sb = visible_normalize(x, visible)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for key in ("mean", "scale"):
            torch.testing.assert_close(sa[key], sb[key], atol=0, rtol=0)

    def test_09_finite_lag_formula_and_axes(self):
        lag = FiniteLag(tiny())
        x = torch.randn(2, 128, 3); visible = torch.ones_like(x, dtype=torch.bool)
        visible[:, -10, 1] = False
        score, count, valid, _ = lag.correlations(x, visible)
        q, k = lag.query(x[:, -20:, :, None].transpose(1, 2)), lag.key(x[:, -20:, :, None].transpose(1, 2))
        vm = visible[:, -20:].transpose(1, 2)
        for ell in (0, 3, 8):
            mask = vm[:, 1, ell:] & vm[:, 2, :20 - ell]
            ref = ((q[:, 1, ell:] * k[:, 2, :20 - ell]).sum(-1) * mask).sum(-1) / (mask.sum(-1) * (8 ** .5))
            torch.testing.assert_close(score[:, 1, 2, ell], ref)
        _, _, meta = lag(x, visible)
        self.assertFalse(bool(meta["self_candidate_valid"][..., 0].any()))
        self.assertFalse(bool(meta["cross_candidate_valid"][:, torch.arange(3), torch.arange(3)].any()))
        torch.testing.assert_close(meta["source_sample_offset"], 127 - torch.arange(9))

    def test_10_lag_empty_and_c1(self):
        lag = FiniteLag(tiny())
        x = torch.randn(1, 128, 1)
        own, cross, meta = lag(x, torch.zeros_like(x, dtype=torch.bool))
        self.assertTrue(bool((own == 0).all() and (cross == 0).all()))
        self.assertFalse(bool(meta["cross_candidate_valid"].any()))
        self.assertTrue(bool(torch.isfinite(own).all()))

    def test_11_var_c_and_full_shapes(self):
        m = self.model().eval()
        for c in (1, 3, 5):
            x = torch.randn(2, 128, c)
            f, r = m.forecast(x), m.reconstruct(x)
            g = min(2, c)
            self.assertEqual(f["prediction"].shape, (2, 2, c))
            self.assertEqual(r["tf_reconstruction"].shape, (2, g, 2, 20, c))
            self.assertEqual(r["ms_reconstruction_by_scale"].shape, (2, g, 2, 2, 128, c))
            self.assertEqual(r["ms_reconstruction"].shape, (2, g, 2, 128, c))
            self.assertTrue(bool((r["hidden_mask"][:, :, -1].sum(1) == 1).all()))

    def test_12_hidden_target_has_no_input_gradient(self):
        m = self.model().eval()
        x = torch.randn(1, 128, 3, requires_grad=True)
        r = m.reconstruct(x)
        # Channel0 uses group0 only. Its endpoint truth must affect no encoded
        # output of that group, including normalization/order/MS/lag.
        probe = r["tf_reconstruction"][:, 0].sum() + r["ms_reconstruction_by_scale"][:, 0].sum()
        grad, = torch.autograd.grad(probe, x)
        self.assertEqual(float(grad[0, -1, 0]), 0)
        self.assertEqual(float(grad[0, -1, 2]), 0)
        changed = x.detach().clone(); changed[:, -1, (0, 2)] += 100
        rr = m.reconstruct(changed)
        torch.testing.assert_close(r["tf_reconstruction"][:, 0], rr["tf_reconstruction"][:, 0], atol=0, rtol=0)
        torch.testing.assert_close(r["stats"]["mean"][:, 0], rr["stats"]["mean"][:, 0], atol=0, rtol=0)

    def test_13_batch_independence_train_eval(self):
        m = self.model()
        x = torch.randn(2, 128, 3); other = x.clone(); other[1] += 50
        for training in (False, True):
            m.train(training)
            torch.manual_seed(12); a = m.forecast(x, mask_seed=91)["prediction"][0]
            torch.manual_seed(12); b = m.forecast(other, mask_seed=91)["prediction"][0]
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_14_pass_isolation_no_cached_current(self):
        m = self.model().eval(); history = torch.randn(1, 128, 3)
        a = m.forecast(history)["prediction"]
        m.reconstruct(history * 100)
        b = m.forecast(history)["prediction"]
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_15_full_backward_active_groups(self):
        m = self.model().train(); x = torch.randn(2, 128, 3); target = torch.randn(2, 2, 3)
        f, r = m.forecast(x), m.reconstruct(x)
        loss = (f["prediction"] - target).square().mean()
        hidden = r["hidden_mask"]
        tf_gt = x[:, None, None, -20:]
        ms_gt = x[:, None, None, None]
        tf_mask = hidden[:, :, None, -20:] & r["tf_valid"]
        ms_mask = hidden[:, :, None, None] & r["ms_valid_by_scale"]
        loss = loss + (r["tf_reconstruction"] - tf_gt).square()[tf_mask].mean()
        loss = loss + (r["ms_reconstruction_by_scale"] - ms_gt).square()[ms_mask].mean()
        loss.backward()
        groups = ("tf_backbone.author.encoder", "tf_backbone.author.decoder", "temporal.encoder", "lag",
                  "ms_filter.author", "fusion.author", "tf_fusion_delta", "ms_decoder", "forecast_head")
        optimizer = torch.optim.AdamW(m.parameters(), lr=.001)
        evidence = {}
        for prefix in groups:
            params = [(n, p) for n, p in m.named_parameters() if n.startswith(prefix)]
            self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for _, p in params), prefix)
            self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for _, p in params), prefix)
            evidence[prefix] = [(p, p.detach().clone()) for _, p in params]
        optimizer.step()
        for prefix, pairs in evidence.items():
            self.assertTrue(any(bool((p.detach() != old).any()) for p, old in pairs), prefix)
        self.assertIsNone(m.event_head.weight.grad)  # F-only head not silently used by SSL.

    def test_16_masks_permute_with_channels(self):
        m = self.model().eval(); x = torch.randn(1, 128, 3); obs = torch.ones_like(x, dtype=torch.bool)
        hidden = m.make_hidden_masks(obs)
        perm = torch.tensor([2, 0, 1])
        a = m.reconstruct(x, obs, hidden)
        b = m.reconstruct(x[:, :, perm], obs[:, :, perm], hidden[:, :, :, perm],
                          copy_masks=a["transform_hidden_mask"][:, :, :, perm])
        torch.testing.assert_close(a["tf_reconstruction"][..., perm], b["tf_reconstruction"], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(a["ms_reconstruction_by_scale"][..., perm], b["ms_reconstruction_by_scale"], rtol=1e-4, atol=1e-5)

    def test_17_all_missing_no_fake_validity(self):
        m = self.model().eval(); x = torch.full((1, 128, 1), float("nan"))
        f, r = m.forecast(x), m.reconstruct(x)
        for key in ("prediction",):
            self.assertTrue(bool(torch.isfinite(f[key]).all()))
        self.assertFalse(bool(f["valid"].any()))
        self.assertFalse(bool(r["tf_valid"].any() or r["ms_valid"].any()))

    def test_18_controls_run_without_axis_shortcuts(self):
        x = torch.randn(1, 128, 3)
        for fusion in ("none", "feature_mean", "concat_linear", "late_score_equal"):
            m = self.model(fusion=fusion).eval()
            self.assertTrue(bool(torch.isfinite(m.forecast(x)["prediction"]).all()))
        for flag in ("tf_enabled", "ms_enabled", "self_lag_enabled", "cross_lag_enabled", "temporal_enabled"):
            m = self.model(**{flag: False}).eval()
            f, r = m.forecast(x), m.reconstruct(x)
            self.assertTrue(bool(torch.isfinite(f["prediction"]).all()))
            if flag == "tf_enabled": self.assertFalse(bool(r["tf_valid"].any()))
            if flag == "ms_enabled": self.assertFalse(bool(r["ms_valid"].any()))

    def test_19_bad_masks_and_config_reject(self):
        with self.assertRaises(ValueError): tiny(copies=3)
        with self.assertRaises(ValueError): tiny(max_lag=20)
        m = self.model(); x = torch.randn(1, 128, 3)
        with self.assertRaises(ValueError): m.forecast(x, torch.ones_like(x))
        hide = torch.zeros(1, 2, 128, 3, dtype=torch.bool); hide[:, :, -1] = True
        with self.assertRaises(ValueError): m.reconstruct(x, hide_mask=hide)

    def test_20_training_only_random_history_masks(self):
        m = self.model(history_mask_probability=.9)
        observed = torch.ones(1, 128, 3, dtype=torch.bool)
        m.eval(); evaluation = m.make_hidden_masks(observed)
        self.assertFalse(bool(evaluation[:, :, :-1].any()))
        m.train(); training = m.make_hidden_masks(observed)
        self.assertTrue(bool(training[:, :, :-1].any()))
        self.assertTrue(bool((training[:, :, -1].sum(1) == 1).all()))

    def test_21_disabled_paths_really_not_called(self):
        from unittest.mock import patch
        x = torch.randn(1, 128, 2)
        m = self.model(tf_enabled=False, ms_enabled=False).eval()
        with patch.object(m.tf_backbone, "encode", side_effect=AssertionError("TF off called")), \
             patch.object(m.tf_backbone, "decode", side_effect=AssertionError("TF decoder off called")), \
             patch.object(m.ms_filter, "forward", side_effect=AssertionError("MS off called")), \
             patch.object(m.ms_decoder, "forward", side_effect=AssertionError("MS decoder off called")):
            m.forecast(x); r = m.reconstruct(x)
        self.assertFalse(bool(r["tf_valid"].any() or r["ms_valid"].any()))

    def test_22_default_production_shapes_and_count(self):
        m = HybridV1(HybridV1Config(), self.sources).eval()
        x = torch.randn(1, 512, 3)
        with torch.no_grad():
            f, r = m.forecast(x), m.reconstruct(x)
        self.assertEqual(f["prediction"].shape, (1, 32, 3))
        self.assertEqual(r["ms_reconstruction_by_scale"].shape, (1, 2, 2, 2, 512, 3))
        self.assertEqual(r["tf_reconstruction"].shape, (1, 2, 2, 100, 3))
        self.assertTrue(bool(torch.isfinite(f["prediction"]).all()))
        print("DEFAULT_PARAMETER_COUNTS", m.parameter_counts())


if __name__ == "__main__":
    unittest.main()
