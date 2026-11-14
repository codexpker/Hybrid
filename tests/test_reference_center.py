"""Actual Torch CPU core-interface fixtures, no public arrays or checkpoints."""
import copy
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

import numpy as np
import torch

from hybrid_tsad.hybrid_v1 import model as core
from hybrid_tsad.hybrid_v1 import training, reference_center as ref
from hybrid_tsad.hybrid_v1.config import HybridV1Config
from hybrid_tsad.hybrid_v1.data import Run
from hybrid_tsad.hybrid_v1.objectives import ssl_objective
from hybrid_tsad.hybrid_v1.provenance import OfficialSources


class ReferenceCenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("This suite is CPU only")
        torch.set_num_threads(1)
        cfg = HybridV1Config(history=128, local_window=20, horizon=2, width=16,
             temporal_layers=1, heads=2, tf_hidden=8, tf_repr=16, tf_depth=1,
             lag_window=20, max_lag=4, lag_width=8, lag_top_k=2)
        sources = OfficialSources.from_workspace(Path(os.environ["HYBRID_V1_WORKSPACE"]))
        cls.factory = ref.model_class(core)
        torch.manual_seed(2025); cls.original = core.HybridV1(cfg, sources).cpu().eval()
        torch.manual_seed(2025); cls.window = cls.factory(cfg, sources, center_mode="window").cpu().eval()
        torch.manual_seed(2025); cls.reference = cls.factory(cfg, sources, center_mode="reference").cpu().eval()
        cls.original_function = core.visible_normalize
        cls.kernels = ref.training_namespace(training)

    def run_data(self, channels=3, role="FIT", start=100):
        rng = np.random.default_rng(3819)
        x = rng.normal(size=(700, channels)).astype(np.float32)
        obs = np.ones(x.shape, bool)
        obs[1:7, 0] = False; x[1:7, 0] = np.nan
        return Run("SYNTHETIC", "fixture", "run", role, x, obs, start=start)

    def batch(self, channels=3, role="FIT"):
        run = self.run_data(channels, role)
        batch = ref.reference_window(run, 640, 128, 2)
        return run, training.tensor_batch(batch)

    def exact(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a: self.exact(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.exact(x, y)
        else: self.assertEqual(a, b)

    def test_identical_initial_parameter_state_and_no_new_buffers(self):
        self.exact(self.original.state_dict(), self.window.state_dict())
        self.exact(self.original.state_dict(), self.reference.state_dict())
        self.assertEqual(self.original.parameter_counts(), self.reference.parameter_counts())

    def test_window_arm_forecast_reconstruct_and_objective_exact_V1(self):
        _, batch = self.batch()
        with torch.no_grad():
            a = training.passes(self.original, batch, 73)
            b = self.kernels.passes(self.window, batch, 73)
            self.exact(a, b)
            self.exact(ssl_objective(*a, batch, "huber"), ssl_objective(*b, batch, "huber"))
        self.assertIs(core.visible_normalize, type(self).original_function)
        self.assertIsNone(self.window._reference_bound)

    def test_reference_normalizer_preserves_original_masked_local_variance(self):
        _, batch = self.batch()
        x, visible = batch["history"].clone().requires_grad_(True), batch["observed"].clone()
        visible[:, 3:13] = False
        stats = ref.visible_reference_statistics(batch["reference_values"].clone().requires_grad_(True), batch["reference_observed"])
        before = torch.get_rng_state().clone()
        normalized, result = ref.reference_normalize(x, visible, stats, core.visible_normalize)
        _, old_stats = core.visible_normalize(x, visible)
        self.exact(result["scale"], old_stats["scale"])
        self.exact(result["local_mean"], old_stats["mean"])
        self.exact(result["mean"], stats["mean"])
        self.assertFalse(result["mean"].requires_grad)
        self.assertFalse(result["local_mean"].requires_grad)
        self.assertFalse(result["scale"].requires_grad)
        self.assertTrue(normalized.requires_grad)
        changed = x.clone(); changed[:, 3:13] = float("nan")
        other = ref.reference_normalize(changed, visible, stats, core.visible_normalize)
        self.exact((normalized, result), other)
        self.exact(before, torch.get_rng_state())

    def test_reference_missing_not_teacher_and_minimum_support(self):
        _, batch = self.batch()
        values = batch["reference_values"].clone()
        obs = batch["reference_observed"]
        a = ref.visible_reference_statistics(values, obs)
        values[~obs] = 99999
        self.exact(a, ref.visible_reference_statistics(values, obs))
        bad = obs.clone(); bad[:, :, 1] = False; bad[:, 0, 1] = True
        with self.assertRaises(ValueError): ref.visible_reference_statistics(values, bad)

    def test_actual_reference_forecast_each_reconstruction_group_and_loss(self):
        _, batch = self.batch()
        with torch.no_grad():
            f, r = self.kernels.passes(self.reference, batch, 73)
            center = ref.visible_reference_statistics(batch["reference_values"], batch["reference_observed"])["mean"]
            self.exact(f["stats"]["mean"], center)
            self.exact(f["prediction"], f["prediction_normalized"]*f["stats"]["scale"]+center)
            for group in range(r["hidden_mask"].shape[1]):
                _, old_stats = core.visible_normalize(batch["reconstruction_context"], r["visible_mask"][:, group])
                self.exact(r["stats"]["mean"][:, group], center)
                self.exact(r["stats"]["scale"][:, group], old_stats["scale"])
            self.exact(r["tf_reconstruction"], r["tf_normalized"]*r["stats"]["scale"][:, :, None]+r["stats"]["mean"][:, :, None])
            self.exact(r["ms_reconstruction_by_scale"], r["ms_normalized_by_scale"]*r["stats"]["scale"][:, :, None, None]+r["stats"]["mean"][:, :, None, None])
            loss = ssl_objective(f, r, batch, "huber")
            self.assertTrue(torch.isfinite(loss["total"]).all())
            self.assertTrue(loss["valid_windows"].all())
        self.assertIsNone(self.reference._reference_bound)

    def test_batch_run_role_anchor_and_full_footprint_guards(self):
        _, batch = self.batch()
        for key, value in [("reference_role", "CAL"), ("reference_identity", ("other", "fixture", "run", "FIT")),
                           ("reference_bounds", [100, 613]), ("index", 739), ("available_index", 739),
                           ("forecast_history_bounds", [611, 740]), ("reconstruction_bounds", [612, 741])]:
            bad = dict(batch); bad[key] = value
            with mock.patch.object(self.reference, "forecast", side_effect=AssertionError("encoder reached")):
                with self.assertRaises(ValueError): self.kernels.passes(self.reference, bad, 73)

    def test_reference_is_cleared_on_exception_and_reentrancy_rejected(self):
        _, batch = self.batch()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with self.reference.reference_context(batch):
                with self.assertRaises(RuntimeError):
                    with self.reference.reference_context(batch): pass
                raise RuntimeError("intentional")
        self.assertIsNone(self.reference._reference_bound)
        with self.assertRaises(RuntimeError):
            self.reference.forecast(batch["history"], batch["observed"])

    def test_other_thread_cannot_use_bound_reference(self):
        _, batch = self.batch(); errors = []
        def call():
            try: self.reference.forecast(batch["history"], batch["observed"])
            except RuntimeError as exc: errors.append(str(exc))
        with self.reference.reference_context(batch):
            thread = threading.Thread(target=call); thread.start(); thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)

    def test_future_GT_and_hidden_C1_endpoint_do_not_enter_actual_encoders(self):
        _, batch = self.batch(channels=1)
        with torch.no_grad():
            f1, r1 = self.kernels.passes(self.reference, batch, 31)
            changed = dict(batch, future_targets=batch["future_targets"]+10000)
            f2, r2 = self.kernels.passes(self.reference, changed, 31)
            self.exact((f1, r1), (f2, r2))
            changed = dict(batch, reconstruction_context=batch["reconstruction_context"].clone())
            changed["reconstruction_context"][:, -1] = 10000
            _, r3 = self.kernels.passes(self.reference, changed, 31)
            self.exact(r1, r3)

    def test_original_validation_and_stream_use_admitted_reference_per_batch(self):
        run, batch = self.batch(role="SELECT")
        entry = {k: batch[k] for k in ("domain", "entity", "run")}
        entry.update(role="SELECT", index=batch["index"], footprint=batch["footprint"])
        value = self.kernels.validation(self.reference, [run], 128, 2, "huber", 2025, plan=[entry])
        self.assertTrue(np.isfinite(value))
        run = self.run_data(channels=1, role="CAL")
        entry = {"domain": run.domain, "entity": run.entity, "run": run.run, "role": "CAL", "index": run.start+640,
                 "footprint": [run.start+512, run.start+641]}
        records = list(self.kernels.iter_infer(self.reference, [run], history=128, horizon=2, seed=2025, index_plan=[entry]))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["replay"]["reference"]["bounds"], [100, 612])
        self.assertEqual(records[0]["replay"]["actual_mask_seed"], 2025+640)

    def test_unique_simple_reference_control_and_current_missing_NA(self):
        run = self.run_data(channels=1, role="QUERY", start=0)
        record = ref.fixed_reference_record(run, 640, history=128)
        self.assertFalse(record["trained"])
        self.assertTrue(np.isfinite(record["r_pred"]))
        original = copy.deepcopy(record)
        run.values[641:] += 1000
        later = ref.fixed_reference_record(run, 640, history=128)
        self.assertEqual(original, later)
        run.values[640] += 10
        arrived = ref.fixed_reference_record(run, 640, history=128)
        for field in ("prediction", "scale", "reference_count", "history_visible_count", "local_mean"):
            self.assertEqual(original[field], arrived[field])
        self.assertNotEqual(original["r_pred"], arrived["r_pred"])
        run.observed[640] = False; run.values[640] = np.nan
        record = ref.fixed_reference_record(run, 640, history=128)
        self.assertTrue(np.isnan(record["r_pred"]))
        self.assertEqual(record["channel_valid"], [False])

    def test_gradient_paths_from_unchanged_actual_objective(self):
        _, batch = self.batch()
        self.reference.train(); self.reference.zero_grad(set_to_none=True)
        f, r = self.kernels.passes(self.reference, batch, 91)
        loss = ssl_objective(f, r, batch, "huber")
        loss["total"].backward()
        blocks = ("tf_backbone", "temporal", "lag", "ms_filter", "fusion", "forecast_head", "tf_fusion_delta", "ms_decoder")
        evidence = {}
        for block in blocks:
            gradients = [p.grad for n, p in self.reference.named_parameters() if n.split(".")[0] == block and p.grad is not None]
            self.assertTrue(gradients, block)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients), block)
            evidence[block] = max(float(g.abs().max()) for g in gradients)
            self.assertGreater(evidence[block], 0, block)
        self.assertTrue(all(p.grad is None for p in self.reference.event_head.parameters()))
        print("ACTIVE_GRADIENT_MAX="+str(evidence), flush=True)
        self.reference.zero_grad(set_to_none=True); self.reference.eval()

    def test_stage1_Z_legacy_UF_policy_and_explicit_new_UF_rejection(self):
        try:
            for route in ("STAGE1", "Z"):
                old = training.update_policy(self.original, route)
                new = self.kernels.update_policy(self.reference, route)
                self.assertEqual(old, new)
            for route in ("U", "F"):
                # Legacy module remains functional; the new API rejects before
                # mutating requires_grad, allocating an optimizer or deepcopy.
                self.assertTrue(training.update_policy(self.original, route))
                flags = [p.requires_grad for p in self.reference.parameters()]
                with self.assertRaises(NotImplementedError): self.kernels.update_policy(self.reference, route)
                self.assertEqual(flags, [p.requires_grad for p in self.reference.parameters()])
            before = {k: v.clone() for k, v in self.reference.state_dict().items()}
            _, log = self.kernels.fit(self.reference, [], [], history=128, horizon=2, route="Z")
            self.assertEqual(log["updates"], 0)
            self.exact(before, self.reference.state_dict())
            with self.assertRaises(NotImplementedError):
                self.kernels.fit(self.reference, [], [], history=128, horizon=2, route="U")
            with self.assertRaises(NotImplementedError): self.kernels.adapt(self.reference, "F")
        finally:
            training.update_policy(self.original, "STAGE1")
            training.update_policy(self.reference, "STAGE1")

    def test_eval_mode_rng_and_original_score_CAL_math_unchanged(self):
        from hybrid_tsad.hybrid_v1.scoring import fit_calibration, apply_calibration
        run = self.run_data(channels=1, role="CAL")
        entries = [{"domain": run.domain, "entity": run.entity, "run": run.run, "role": "CAL",
                    "index": run.start+t, "footprint": [run.start+t-128, run.start+t+1]} for t in (640, 641, 642)]
        self.window.eval()
        actual = list(self.kernels.iter_infer(self.window, [run], history=128, horizon=2, seed=2025, index_plan=entries))
        expected = list(training.iter_infer(self.original, [run], history=128, horizon=2, seed=2025, index_plan=entries))
        stripped = copy.deepcopy(actual)
        for row in stripped: row["replay"].pop("reference")
        self.exact(stripped, expected)
        self.assertEqual(fit_calibration(actual), fit_calibration(expected))
        cal = fit_calibration(actual)
        self.assertEqual([apply_calibration(x, cal) for x in actual], [apply_calibration(x, cal) for x in expected])
        run, batch = self.batch()
        self.reference.eval()
        a = self.kernels.passes(self.reference, batch, 93)
        b = self.kernels.passes(self.reference, batch, 93)
        self.exact(a, b)
        self.assertEqual(int(a[1]["hidden_mask"][:, :, :-1].sum()), 0)
        self.reference.train()
        before = torch.get_rng_state().clone()
        self.kernels.validation(self.reference, [run], 128, 2, "huber", 2025,
             plan=[{"domain": run.domain, "entity": run.entity, "run": run.run, "role": run.role,
                    "index": batch["index"], "footprint": batch["footprint"]}])
        self.assertTrue(self.reference.training)
        self.exact(before, torch.get_rng_state())
        self.reference.eval()

    @classmethod
    def tearDownClass(cls):
        if torch.cuda.is_initialized():
            raise AssertionError("CPU fixture initialized CUDA")


if __name__ == "__main__":
    unittest.main(verbosity=2)
