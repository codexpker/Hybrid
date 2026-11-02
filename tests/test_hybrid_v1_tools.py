"""Synthetic tools tests; no public arrays, labels or checkpoints."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1.data import Run, targets, window, arrived_targets, arrived_window, BalancedSampler, audit_events, load_manifest
from hybrid_tsad.hybrid_v1.scoring import fit_calibration, apply_calibration


class NumpyTools(unittest.TestCase):
    def test_runs_channels_footprints_and_masks(self):
        run = Run("A", "x", "r", "FIT", np.ones((20, 3)), np.ones((20, 3), bool), 100)
        b = window(run, 8, 8, 2)
        self.assertEqual(b["history"].shape, (1, 8, 3)); self.assertEqual(b["footprint"], [100, 110])
        self.assertEqual(b["index"], b["available_index"])
        self.assertFalse(any("label" in key for key in b))
        with self.assertRaises(ValueError): window(run, 19, 8, 2)
        self.assertEqual(len(targets(run, 8, 2)), 11)

    def test_domain_then_run_balanced_not_point_balanced(self):
        runs = [Run(d, d, d, "FIT", np.ones((n, c)), np.ones((n, c), bool))
                for d, n, c in (("A", 15, 1), ("B", 1000, 3), ("C", 25, 2))]
        a = BalancedSampler(runs, 8, 2, 18); b = BalancedSampler(runs, 8, 2, 18)
        draws = [a.sample()["domain"] for _ in range(600)]
        self.assertEqual(draws, [b.sample()["domain"] for _ in range(600)])
        self.assertTrue(all(150 < draws.count(d) < 250 for d in "ABC"))

    def test_calibration_equal_domain_missing_required_na(self):
        records = [{"domain": "A", "r_tf": 1., "r_ms": 1., "r_pred": 1.}]*100
        records += [{"domain": "B", "r_tf": 9., "r_ms": 9., "r_pred": 9.}]
        cal = fit_calibration(records)
        self.assertAlmostEqual(cal["scales"]["r_tf"], 1.)
        self.assertTrue(cal["source_domain_equal_weight"])
        bad = dict(records[0], r_ms=float("nan"))
        self.assertFalse(apply_calibration(bad, cal)["valid"])
        with self.assertRaises(ValueError): fit_calibration([bad])
        with self.assertRaisesRegex(ValueError, "B"):
            fit_calibration([records[0], dict(bad, domain="B")])
        zero = fit_calibration([dict(domain="A", r_tf=0., r_ms=0., r_pred=0.)])
        self.assertEqual(zero["scales"]["r_tf"], 1e-6)
        self.assertEqual(apply_calibration(dict(domain="A", r_tf=0., r_ms=0., r_pred=0.), zero)["score"], 0.)

    def test_event_budget_matching_and_holdout(self):
        e = [{"entity": "x", "run": "r", "event_id": "E1", "start": 20, "end": 25}]
        n = [{"entity": "x", "run": "r", "matched_event_id": "E1", "start": 5, "end": 10}]
        self.assertEqual(audit_events(e, n, [], 1)["normal_points"], 5)
        with self.assertRaises(ValueError): audit_events(e, n, [], 0)
        with self.assertRaises(ValueError): audit_events(e, n, [dict(entity="x", run="r", start=23, end=40)], 1)

    def test_static_recipes_and_runtime_missing_are_distinct(self):
        for active in (("r_ms", "r_pred"), ("r_tf", "r_pred"), ("r_tf", "r_ms"), ("r_pred",)):
            record = {"domain": "A", "r_tf": float("nan"), "r_ms": float("nan"), "r_pred": float("nan")}
            record.update({k: 1. for k in active})
            calibration = fit_calibration([record], active_components=active)
            self.assertTrue(apply_calibration(record, calibration)["valid"])
            self.assertAlmostEqual(sum(calibration["weights"].values()), 1.)
            record[active[0]] = float("nan")
            self.assertFalse(apply_calibration(record, calibration)["valid"])

    def test_public_manifest_loader_preserves_missing_not_label(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); values = np.ones((30, 2)); values[4, 1] = np.nan
            np.save(p/"values.npy", values)
            entry = dict(path="values.npy", bounds=[0, 15], domain="A", entity="x", run="r", role="FIT")
            (p/"manifest.json").write_text(json.dumps({"runs": [entry]}))
            runs = load_manifest(p/"manifest.json")
            self.assertFalse(runs[0].observed[4, 1])
            entry["label_path"] = "never_open.npy"
            (p/"manifest.json").write_text(json.dumps({"runs": [entry]}))
            with self.assertRaises(ValueError): load_manifest(p/"manifest.json")

    def test_arrived_tail_no_unarrived_targets(self):
        values = np.arange(40, dtype=np.float32).reshape(20, 2)
        run = Run("A", "e", "r", "QUERY", values, np.ones_like(values, bool), 100)
        self.assertEqual(arrived_targets(run, 8).tolist(), list(range(8, 20)))
        tail = arrived_window(run, 19, 8, 4)
        self.assertEqual(tail["index"], 119)
        self.assertEqual(tail["footprint"], [111, 120])
        self.assertTrue(tail["future_observed"][:, 0].all())
        self.assertFalse(tail["future_observed"][:, 1:].any())
        self.assertTrue((tail["future_targets"][:, 1:] == 0).all())
        a = arrived_window(run, 10, 8, 4)
        run.values[11:] += 1000
        b = arrived_window(run, 10, 8, 4)
        for key in ("history", "reconstruction_context", "future_targets"):
            np.testing.assert_array_equal(a[key], b[key])

    def test_normal_ssl_full_footprint_forbidden_counterexample(self):
        events = [dict(entity="E", run="positive", event_id="one", start=200, end=202)]
        normal = [dict(entity="E", run="normal", matched_event_id="one", start=200, end=202)]
        forbidden = [dict(entity="E", run="normal", start=100, end=150)]
        with self.assertRaisesRegex(ValueError, "Normal SSL full footprint"):
            audit_events(events, normal, forbidden, 1, history=128, horizon=2)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Pending existing remote CPU Torch runtime")
class TorchTools(unittest.TestCase):
    def test_actual_additional_normal_ssl_footprint_guard(self):
        import torch
        from unittest.mock import patch
        from hybrid_tsad.hybrid_v1.training import fit
        model = torch.nn.Module(); model.fusion = torch.nn.Linear(1, 1); model.event_head = torch.nn.Linear(1, 1)
        values = np.ones((400, 1), np.float32)
        runs = [Run("A", "e", name, "SUPPORT", values.copy(), np.ones_like(values, bool)) for name in ("positive", "normal")]
        select = [Run("A", "e", "select", "NORMAL_SELECT", values.copy(), np.ones_like(values, bool))]
        support = dict(K=1, events=[dict(entity="e", run="positive", event_id="E", start=200, end=202)],
            normal=[dict(entity="e", run="normal", matched_event_id="E", start=200, end=202)],
            forbidden=[dict(entity="e", run="normal", start=100, end=150)])
        def objective(*a, **k):
            return dict(total=model.fusion.weight.square().sum(), valid_windows=torch.ones(1, dtype=torch.bool))
        # Deliberately bypass static audit to prove the actual nb path itself
        # still rejects leakage, before its extra forward is called.
        with patch("hybrid_tsad.hybrid_v1.training.audit_events", return_value={}), \
             patch("hybrid_tsad.hybrid_v1.training.passes", return_value=(None, None)), \
             patch("hybrid_tsad.hybrid_v1.training.ssl_objective", side_effect=objective), \
             patch("hybrid_tsad.hybrid_v1.training.residual_components", side_effect=lambda *a: {"event_features": model.fusion.weight.reshape(1, 1)}):
            with self.assertRaisesRegex(ValueError, "Additional normal SSL full footprint"):
                fit(model, runs, select, history=128, horizon=2, steps=2, route="F", support=support)

    def test_f_selected_checkpoint_requires_all_three_positive_and_negative(self):
        import torch
        from unittest.mock import patch
        from hybrid_tsad.hybrid_v1.training import fit
        model = torch.nn.Module(); model.fusion = torch.nn.Linear(1, 1); model.event_head = torch.nn.Linear(1, 1)
        values = np.ones((40, 1), np.float32)
        runs = [Run("A", "e", name, "SUPPORT", values.copy(), np.ones_like(values, bool)) for name in ("positive", "normal")]
        select = [Run("A", "e", "select", "NORMAL_SELECT", values.copy(), np.ones_like(values, bool))]
        events = [dict(entity="e", run="positive", event_id="E"+str(i), start=8+4*i, end=9+4*i) for i in range(3)]
        normal = [dict(entity="e", run="normal", matched_event_id="E"+str(i), start=8+4*i, end=9+4*i) for i in range(3)]
        support = dict(K=3, events=events, normal=normal, forbidden=[])
        def objective(*args, **kwargs):
            loss = model.fusion.weight.square().mean()
            return dict(total=loss, valid_windows=torch.ones(1, dtype=torch.bool), parts={"tf": loss},
                valid_counts={"tf": torch.ones(1, dtype=torch.long)}, natural_missing_count=torch.tensor(0), artificial_hidden_observed_count=torch.tensor(1))
        # Actual Adam/model updates, but explicitly synthetic helper-boundary
        # fixtures replace encoder math. Real core runs are separate artifacts.
        with patch("hybrid_tsad.hybrid_v1.training.passes", return_value=(None, None)), \
             patch("hybrid_tsad.hybrid_v1.training.ssl_objective", side_effect=objective), \
             patch("hybrid_tsad.hybrid_v1.training.residual_components", side_effect=lambda *a: {"event_features": model.fusion.weight.reshape(1, 1)}), \
             patch("hybrid_tsad.hybrid_v1.training.validation", side_effect=[.001, .002, .003, .004, .005, .006]):
            _, log = fit(model, runs, select, history=4, horizon=2, steps=6, route="F", support=support)
        self.assertEqual(log["selected_attempt"], 5)
        self.assertEqual(log["selected_seen_event_ids"], ["E0", "E1", "E2"])
        self.assertEqual(log["selected_seen_matched_normal_ids"], ["E0", "E1", "E2"])
        self.assertTrue(all(not row["selection_eligible"] for row in log["steps"][:5]))
        for row in log["steps"]:
            self.assertAlmostEqual(row["loss"], row["BCE"]+.1*row["normal_ssl_total"], places=6)
            self.assertIsNotNone(row["normal_ssl_footprint"])

    def test_disabled_tf_ms_real_optimizer_step_fixture(self):
        import torch
        from hybrid_tsad.hybrid_v1.objectives import ssl_objective
        for disabled in ("r_tf", "r_ms"):
            p = torch.nn.Parameter(torch.ones(1)); optimizer = torch.optim.SGD([p], lr=.1)
            mask = torch.ones(1, 1, 2, 3, 2, dtype=torch.bool)
            mscale = torch.ones(1, 1, 2, 2, 3, 2, dtype=torch.bool)
            rec = {"hidden_mask": torch.ones(1, 1, 3, 2, dtype=torch.bool),
                   "stats": {"mean": torch.zeros(1, 1, 1, 2), "scale": torch.ones(1, 1, 1, 2)},
                   "tf_normalized": p.expand(1, 1, 2, 3, 2), "tf_valid": mask if disabled != "r_tf" else torch.zeros_like(mask),
                   "ms_normalized_by_scale": p.expand(1, 1, 2, 2, 3, 2),
                   "ms_valid_by_scale": mscale if disabled != "r_ms" else torch.zeros_like(mscale),
                   "ms_normalized": p.expand(1, 1, 2, 3, 2),
                   "ms_valid": mask if disabled != "r_ms" else torch.zeros_like(mask)}
            fore = {"stats": {"mean": torch.zeros(1, 1, 2), "scale": torch.ones(1, 1, 2)},
                    "prediction_normalized": p.expand(1, 1, 2), "valid": torch.ones(1, 1, 2, dtype=torch.bool)}
            batch = {"reconstruction_context": torch.zeros(1, 3, 2), "reconstruction_observed": torch.ones(1, 3, 2, dtype=torch.bool),
                     "future_targets": torch.zeros(1, 1, 2), "future_observed": torch.ones(1, 1, 2, dtype=torch.bool)}
            active = [k for k in ("r_tf", "r_ms", "r_pred") if k != disabled]
            loss = ssl_objective(fore, rec, batch, active_components=active)
            self.assertTrue(loss["valid_windows"].all())
            loss["total"].backward(); optimizer.step(); self.assertNotEqual(float(p), 1.)
    def test_natural_nan_target_event_features_have_finite_backward(self):
        import torch
        from hybrid_tsad.hybrid_v1.scoring import residual_components
        p = torch.nn.Parameter(torch.ones(1))
        pred = p.expand(1, 1, 2)
        forecast = {"prediction": pred, "valid": torch.tensor([[[True, False]]]),
                    "stats": {"scale": torch.ones(1, 1, 2)}}
        rec = {"hidden_mask": torch.ones(1, 1, 3, 2, dtype=torch.bool),
               "stats": {"scale": torch.ones(1, 1, 1, 2)},
               "tf_reconstruction": p.expand(1, 1, 2, 3, 2),
               "tf_valid": torch.ones(1, 1, 2, 3, 2, dtype=torch.bool),
               "ms_reconstruction_by_scale": p.expand(1, 1, 2, 2, 3, 2),
               "ms_valid_by_scale": torch.ones(1, 1, 2, 2, 3, 2, dtype=torch.bool),
               "endpoint_t_features": p.expand(1, 2, 4), "endpoint_f_features": p.expand(1, 2, 4)}
        batch = {"current_target": torch.tensor([[0., float("nan")]]),
                 "current_observed": torch.tensor([[True, False]]), "index": 3, "available_index": 3}
        result = residual_components(forecast, rec, batch)
        torch.nn.functional.binary_cross_entropy_with_logits(result["event_features"].sum(1), torch.ones(1)).backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(torch.isnan(result["per_channel"]["r_tf"][0, 1]))

    def test_half_mse_huber_elements_and_policy(self):
        import torch
        from hybrid_tsad.hybrid_v1.objectives import elementwise
        from hybrid_tsad.hybrid_v1.training import update_policy, adapt
        x = torch.tensor([.5, 2.], requires_grad=True)
        torch.testing.assert_close(elementwise(x, "half_mse"), torch.tensor([.125, 2.]))
        torch.testing.assert_close(elementwise(x, "huber"), torch.tensor([.125, 1.5]))
        model = torch.nn.Module(); model.fusion = torch.nn.Linear(2, 2); model.event_head = torch.nn.Linear(7, 1)
        model.encoder = torch.nn.Linear(2, 2)
        self.assertEqual(set(update_policy(model, "U")), {"fusion.weight", "fusion.bias"})
        source = {k: v.clone() for k, v in model.state_dict().items()}
        z, log = adapt(model, "F", support={"K": 0}, history=8, horizon=2)
        self.assertEqual(log["updates"], 0)
        for key, v in z.state_dict().items(): torch.testing.assert_close(v, source[key], rtol=0, atol=0)


if __name__ == "__main__": unittest.main()
