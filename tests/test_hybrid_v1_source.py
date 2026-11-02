"""Source-interface fixtures only; no existing public data/model/GPU needed."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1.data import Run, load_manifest, fixed_plan, resolve_plan


class SourceData(unittest.TestCase):
    def test_headerless_keeps_first_row_and_exact_hash_shape(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); x = np.arange(40).reshape(20, 2)
            np.savetxt(p/"x.txt", x, delimiter=",")
            e = dict(path="x.txt", format="csv", has_header=False, columns=[0, 1],
                sha256=hashlib.sha256((p/"x.txt").read_bytes()).hexdigest(), expected_shape=[20, 2],
                bounds=[0, 10], domain="SMD", entity="e", run="r", role="FIT")
            def load():
                (p/"m.json").write_text(json.dumps({"runs": [e]}))
                return load_manifest(p/"m.json")
            np.testing.assert_array_equal(load()[0].values, x[:10])
            e["sha256"] = "0"*64
            with self.assertRaisesRegex(ValueError, "SHA256"): load()
            e.pop("sha256"); e["expected_shape"] = [19, 2]
            with self.assertRaisesRegex(ValueError, "shape"): load()

    def test_explicit_observed_and_role_filter_before_body(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); x = np.ones((30, 1)); o = np.ones_like(x, bool); o[5] = False
            np.savez(p/"x.npz", values=x, observed=o)
            e = dict(path="x.npz", key="values", observed_key="observed", bounds=[0, 20],
                domain="Sunspot", entity="T1", run="T1", role="FIT")
            forbidden = dict(e, path="MUST_NOT_OPEN_QUERY.npz", role="QUERY", bounds=[20, 30])
            (p/"m.json").write_text(json.dumps({"runs": [e, forbidden]}))
            r = load_manifest(p/"m.json", roles={"FIT"})[0]
            self.assertFalse(r.observed[5, 0]); self.assertEqual(r.values[5, 0], 1)

    def test_fixed_plans_same_seed_observation_only_and_full_footprint(self):
        x = np.ones((100, 2), np.float32); o = np.ones_like(x, bool); o[10:20] = False
        runs = [Run(d, d, d, "FIT", x.copy(), o.copy(), 100) for d in ("SMD", "Sunspot")]
        a = fixed_plan(runs, 8, 3, 200, 2025, "FIT")
        for r in runs: r.values[:] *= 9000
        b = fixed_plan(runs, 8, 3, 200, 2025, "FIT")
        self.assertEqual(a, b); self.assertEqual(len(a), 200)
        for r, t in resolve_plan(runs, a, 8, 3):
            self.assertGreaterEqual(t, 8); self.assertLessEqual(t+3, 100)
            self.assertTrue(r.observed[t].all())
        with self.assertRaisesRegex(ValueError, "footprint"):
            resolve_plan(runs, [dict(a[0], footprint=[0, 1])], 8, 3)
        self.assertNotEqual(a, fixed_plan(runs, 8, 3, 200, 2026, "FIT"))

    def test_undefined_domain_observed_support_rejected(self):
        x = np.ones((40, 1)); r = Run("Sunspot", "T1", "T1", "CAL", x, np.zeros_like(x, bool))
        with self.assertRaisesRegex(ValueError, "Every source domain"):
            fixed_plan([r], 8, 2, 5, 17, "CAL", arrived=True)

    def test_pending_guard_before_data_or_torch_import(self):
        from hybrid_tsad.hybrid_v1.cli import source_approval
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"cfg.json"; p.write_text('{}')
            with self.assertRaises(PermissionError):
                source_approval(p, None, None, "source_train", "never_created")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires existing remote Torch")
class SourceScheduling(unittest.TestCase):
    def test_validate_every_actual_optimizer_and_final_state(self):
        import torch
        from hybrid_tsad.hybrid_v1.training import fit
        model = torch.nn.Module(); model.fusion = torch.nn.Linear(1, 1)
        x = np.ones((30, 1), np.float32); o = np.ones_like(x, bool)
        fit_runs = [Run("SMD", "e", "fit", "FIT", x, o)]
        select = [Run("SMD", "s", "select", "SELECT", x, o)]
        plan = fixed_plan(fit_runs, 4, 2, 7, 2025, "FIT")
        def objective(*args, **kwargs):
            loss = model.fusion.weight.square().mean()
            return dict(total=loss, valid_windows=torch.ones(1, dtype=torch.bool), parts={"tf": loss},
                valid_counts={"tf": torch.ones(1, dtype=torch.long)}, natural_missing_count=torch.tensor(0), artificial_hidden_observed_count=torch.tensor(1))
        with tempfile.TemporaryDirectory() as d, \
             patch("hybrid_tsad.hybrid_v1.training.passes", return_value=(None, None)), \
             patch("hybrid_tsad.hybrid_v1.training.ssl_objective", side_effect=objective), \
             patch("torch.cuda.synchronize", side_effect=AssertionError("CPU must not query CUDA")), \
             patch("hybrid_tsad.hybrid_v1.training.validation", side_effect=[3., 2., 1.]) as v:
            _, log = fit(model, fit_runs, select, history=4, horizon=2, steps=7,
                sampling_plan=plan, validate_every=3, artifact_dir=d)
            self.assertEqual(v.call_count, 3)
            self.assertEqual([i for i, r in enumerate(log["steps"]) if r["SELECT_ssl"] is not None], [2, 5, 6])
            self.assertEqual(log["updates"], 7); self.assertEqual(log["selected_attempt"], 6)
            self.assertEqual(log["optimizer"]["warmup_steps"], 0)
            self.assertTrue(all(r["train_sync_wall_seconds"] >= 0 and r["artifact_wall_seconds"] >= 0 for r in log["steps"]))
            self.assertTrue((Path(d)/"initial.pt").is_file())
            self.assertEqual(len((Path(d)/"training_trace.jsonl").read_text().splitlines()), 7)
            self.assertTrue((Path(d)/"selected_partial.pt").is_file())
            saved = torch.load(Path(d)/"final_training_state.pt", weights_only=False)
            self.assertEqual(saved["final_attempt"], 6)
            self.assertEqual(len(saved["optimizer"]["state"]), 1)
            recovery = torch.load(Path(d)/"recovery_latest.pt", weights_only=False)
            self.assertEqual(recovery["final_attempt"], 6)
            self.assertEqual(recovery["selected_attempt"], 6)
            self.assertFalse(recovery["resume_implemented"])
            self.assertFalse((Path(d)/"recovery_latest.pt.tmp").exists())

    def test_fixed_calibration_plan_does_not_sweep_all_points(self):
        # No encoder/forward: verify the iterator boundary by a sentinel failure.
        import torch
        from hybrid_tsad.hybrid_v1.training import infer
        m = torch.nn.Linear(1, 1); x = np.ones((30, 1)); o = np.ones_like(x, bool)
        r = Run("SMD", "e", "cal", "CAL", x, o, 100)
        plan = [dict(domain="SMD", entity="e", run="cal", role="CAL", index=129, footprint=[125, 130])]
        with patch("hybrid_tsad.hybrid_v1.training.passes", side_effect=RuntimeError("sentinel")) as p:
            with self.assertRaisesRegex(RuntimeError, "sentinel"):
                infer(m, [r], history=4, horizon=2, index_plan=plan)
            batch = p.call_args[0][1]
            self.assertEqual(batch["index"], 129)
            self.assertFalse(batch["future_observed"][:, 1:].any())


if __name__ == "__main__": unittest.main()
