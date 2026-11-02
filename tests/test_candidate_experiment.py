"""Synthetic only: no published arrays, labels, pretrained weights or SSH."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hybrid_tsad import candidate_experiment as run


EXAMPLE = Path(__file__).resolve().parents[1] / "configs/candidate-development.example.json"


class GeometryTests(unittest.TestCase):
    def test_role_footprints(self):
        for bounds in ((0, 20), (25, 45), (50, 70)):
            t = run.role_targets(bounds, 4)
            self.assertEqual(t[0] - 4, bounds[0])
            self.assertEqual(t[-1], bounds[1]-1)
            a = np.arange(140, dtype=np.float32).reshape(70, 2)
            x, y = run.windows(a, t, 4)
            np.testing.assert_array_equal(x[0], a[bounds[0]:bounds[0]+4])
            np.testing.assert_array_equal(y[:, 0], a[t])

    def test_illegal_roles(self):
        for bounds in ((-1, 30), (0, 4), (0, 3)):
            with self.assertRaises(ValueError):
                run.role_targets(bounds, 4)

    def test_target_outside_input(self):
        a = np.arange(60, dtype=np.float32).reshape(30, 2)
        before, y = run.windows(a, np.array([10]), 4)
        a[10] = 999
        after, changed_y = run.windows(a, np.array([10]), 4)
        np.testing.assert_array_equal(before, after)
        self.assertFalse(np.array_equal(y, changed_y))

    def test_sampling_independent_of_global_rng(self):
        a = run.sample_targets(np.arange(4, 30), 8, 4, 20261102)
        np.random.seed(123); np.random.normal(size=50)
        b = run.sample_targets(np.arange(4, 30), 8, 4, 20261102)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.isin(a, np.arange(4, 30)).all())

    def test_fit_only_median_and_no_backfill(self):
        a = np.array([[1., np.nan], [3., 4.], [999., 888.], [np.nan, np.nan]])
        median = run.fit_medians(a, (0, 2))
        np.testing.assert_array_equal(median, [2., 4.])
        filled = run.impute(a, median)
        np.testing.assert_array_equal(filled[0], [1., 4.])
        np.testing.assert_array_equal(filled[-1], [2., 4.])
        a[2:] = -1e5
        np.testing.assert_array_equal(run.fit_medians(a, (0, 2)), median)

    def test_allmissing_and_inf_rejected(self):
        for a in (np.array([[1., np.nan], [2., np.nan]]), np.array([[np.inf, 0.]])):
            with self.assertRaises(ValueError):
                run.fit_medians(a, (0, len(a)))
        with self.assertRaises(ValueError):
            run.impute(np.array([[np.inf]]), np.array([0.]))

    def test_calibration_and_strict_threshold(self):
        s = np.arange(101.)
        self.assertEqual(run.calibrate(s), 99.)
        np.testing.assert_array_equal(run.confirmed_alarm(np.arange(5), [99, 100, 101, 102, 103], 99),
                                      [False, False, False, True, True])

    def test_calibration_higher_with_ties(self):
        scores = np.array([0., 0., 1., 1., 9.])
        self.assertEqual(run.calibrate(scores), 9.)
        self.assertNotEqual(run.calibrate(scores), float(np.quantile(scores, .99, method="linear")))
        np.testing.assert_array_equal(run.confirmed_alarm(np.arange(3), [9., 9., 9.], 9.), [False]*3)

    def test_confirmation_not_backfilled_and_gap_resets(self):
        ix = np.array([10, 11, 12, 13, 20, 21, 22, 23, 24])
        alarm = run.confirmed_alarm(ix, [1, 1, 1, 0, 1, 1, 1, 0, 1], .5)
        np.testing.assert_array_equal(ix[alarm], [12, 22])

    def test_invalid_score_rejected(self):
        with self.assertRaises(ValueError):
            run.calibrate([np.nan])
        with self.assertRaises(ValueError):
            run.confirmed_alarm(np.array([2, 1]), [0, 1], .5)

    def test_preview_opens_no_array(self):
        with patch.object(run, "_load_array", side_effect=AssertionError("data access")):
            run.main(["--config", str(EXAMPLE)])

    def test_unreleased_actions_rejected(self):
        cfg = json.loads(EXAMPLE.read_text())
        for action in ("prepare", "train", "infer", "evaluate"):
            with self.assertRaises(ValueError):
                run.authorize(cfg, action, EXAMPLE, "tf")

    def test_test_label_spec_not_part_of_train_data(self):
        cfg = json.loads(EXAMPLE.read_text())
        self.assertEqual(set(cfg["data"]), {"train", "test", "roles", "guard"})
        self.assertIsNone(cfg["release"])

    def test_overlapping_role_and_guard_rejected(self):
        cfg = json.loads(EXAMPLE.read_text())
        for start in (79400, 79489):
            modified = copy.deepcopy(cfg)
            modified["data"]["roles"]["SELECT"][0] = start
            with self.assertRaises(ValueError):
                run.validate_config(modified)

    def test_csv_timestamp_dropped_and_hash_checked(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "synthetic.csv"
            p.write_text("timestamp,a,b\n0,1,\n1,3,4\n")
            spec = {"path": str(p), "sha256": run.sha256(p), "shape": [2, 2],
                    "format": "csv", "drop_columns": ["timestamp"], "value_columns": ["a", "b"]}
            a = run._load_array(spec, p)
            self.assertTrue(np.isnan(a[0, 1]))
            np.testing.assert_array_equal(a[:, 0], [1, 3])
            spec["sha256"] = "0"*64
            with self.assertRaises(ValueError):
                run._load_array(spec, p)

    def test_evaluation_rejects_unapproved_raw_before_labels(self):
        cfg = json.loads(EXAMPLE.read_text())
        cfg["release"] = {"approved_raw_sha256": {}}
        with patch.object(run, "_read_receipt"), patch.object(run, "_out", return_value=Path(".")), \
                patch.object(run, "_load_array", side_effect=AssertionError("label opened")):
            with self.assertRaises(ValueError):
                run.evaluate(cfg, EXAMPLE)

    def test_source_migration_rejected(self):
        cfg = {"release": {"source_sha256": {"new": "hash"}}}
        previous = {"protocol_sha256": run.protocol_hash(cfg), "source_sha256": {"old": "hash"}, "files": {}}
        with self.assertRaises(ValueError):
            run._receipt(cfg, EXAMPLE, previous)

    def test_exact_resource_parent_and_artifacts(self):
        old = {"candidate_experiment.py": "c38123ef639c3adf43c7b3ad6aa0232587f1538c265aee9477121c5a9f076a59"}
        old.update({"helper%d.py" % i: str(i)*64 for i in range(9)})
        current = dict(old, **{"candidate_experiment.py": "new-resource-runner"})
        with tempfile.TemporaryDirectory() as d:
            artifact = Path(d)/"artifact.bin"; artifact.write_bytes(b"unchanged")
            cfg = {"output_directory": d, "release": {"source_sha256": current,
                   "parent_source_compatibility": {"approved_by": "root", "approval_id": "TEST-ROOT",
                   "parent_source_sha256": old, "allowed_old_stages": ["prepare", "train"]}}}
            stage = {"stage": "prepare", "protocol_sha256": run.protocol_hash(cfg),
                     "source_sha256": old, "files": {"artifact.bin": run.sha256(artifact)}}
            run._receipt(cfg, EXAMPLE, stage)
            stage["stage"] = "train"; run._receipt(cfg, EXAMPLE, stage)
            for field, value in (("stage", "infer"), ("protocol_sha256", "bad")):
                wrong = copy.deepcopy(stage); wrong[field] = value
                with self.assertRaises(ValueError):
                    run._receipt(cfg, EXAMPLE, wrong)
            bad = copy.deepcopy(cfg); bad["release"]["source_sha256"]["helper0.py"] = "changed-model"
            with self.assertRaises(ValueError):
                run._receipt(bad, EXAMPLE, stage)
            bad = copy.deepcopy(cfg); bad["release"]["parent_source_compatibility"]["approved_by"] = "payload"
            with self.assertRaises(ValueError):
                run._receipt(bad, EXAMPLE, stage)
            bad = copy.deepcopy(cfg); bad["release"]["parent_source_compatibility"]["parent_source_sha256"]["helper0.py"] = "bad"
            with self.assertRaises(ValueError):
                run._receipt(bad, EXAMPLE, stage)
            artifact.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                run._receipt(cfg, EXAMPLE, stage)

    def test_resource_override_requires_current_action_and_root(self):
        from types import SimpleNamespace
        cpu = SimpleNamespace(type="cpu")
        cfg = {"limits": {"max_seconds": 1800, "max_cuda_bytes": 20*2**30},
               "release": {"release_id": "TEST-ROOT", "approved_actions": ["infer"],
               "action_limits": {"infer": {"approved_by": "root", "approval_id": "TEST-4800", "max_seconds": 4800}}}}
        with patch.object(run.time, "monotonic", return_value=3000):
            with self.assertRaises(RuntimeError):
                run._resources(None, cpu, 0, cfg)
            token = run._EXECUTION_ACTION.set("infer")
            try:
                self.assertEqual(run._resources(None, cpu, 0, cfg), (3000, 0))
                for field, value in (("approved_by", "payload"), ("approval_id", ""), ("max_seconds", 4801)):
                    wrong = copy.deepcopy(cfg); wrong["release"]["action_limits"]["infer"][field] = value
                    with self.assertRaises(ValueError):
                        run._resources(None, cpu, 0, wrong)
                wrong = copy.deepcopy(cfg); wrong["release"]["approved_actions"] = []
                with self.assertRaises(ValueError):
                    run._resources(None, cpu, 0, wrong)
                wrong = copy.deepcopy(cfg); wrong["release"]["action_limits"]["infer"]["max_cuda_bytes"] = 30*2**30
                with self.assertRaises(ValueError):
                    run._resources(None, cpu, 0, wrong)
            finally:
                run._EXECUTION_ACTION.reset(token)
            token = run._EXECUTION_ACTION.set("train")
            try:
                with self.assertRaises(RuntimeError):
                    run._resources(None, cpu, 0, cfg)
            finally:
                run._EXECUTION_ACTION.reset(token)

    def test_legacy_profile_preview_is_still_lazy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"profile.json"
            p.write_text(json.dumps({"scope": "MODEL-FIRST-TF-LAG-GPU-L0-20261102"}))
            with patch.object(run, "profile", side_effect=AssertionError("executed")):
                run.main(["--config", str(p)])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn optional evaluation dependency")
    def test_multi_metrics_confirmation_and_censoring(self):
        labels = np.array([0]*5 + [1]*5 + [0]*5 + [1]*5)
        index = np.arange(2, 20)
        score = np.zeros(18); score[3:6] = 2
        r = run.evaluate_arrays(index, score, labels, 1.)
        self.assertIn("trapezoid_PR_AUC", r["ranking"])
        self.assertIn("ROC_AUC", r["ranking"])
        confirmed = r["streams"]["confirmed3_current"]
        self.assertEqual(confirmed["events"][0]["delay_sample_points"], 2)
        self.assertTrue(confirmed["events"][1]["delay_censored"])
        self.assertEqual(confirmed["metrics"]["normal_exposure_points"], 8)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch integration delegated to existing Torch runtime")
class TorchSyntheticTests(unittest.TestCase):
    def test_common_initialization_and_full_synthetic_stages(self):
        import torch
        from hybrid_tsad import candidate_tf_lag

        class Tiny(torch.nn.Module):
            def __init__(self, mode):
                super().__init__()
                self.common = torch.nn.Linear(1, 4)
                self.head = torch.nn.Linear(4, 1)
                if mode == "concat":
                    self.concat = torch.nn.Linear(4, 4)

            def parameter_counts(self):
                return {"stored": sum(p.numel() for p in self.parameters()),
                        "active": sum(p.numel() for p in self.parameters())}

            def forward(self, x):
                mean = x.mean(1, keepdim=True).detach()
                scale = (x.var(1, keepdim=True, unbiased=False)+1e-5).sqrt().detach()
                state = self.common(((x[:, -1:]-mean)/scale).transpose(1, 2))
                if hasattr(self, "concat"):
                    state = self.concat(state)
                z = self.head(state).transpose(1, 2)
                return {"prediction": z*scale+mean, "prediction_normalized": z,
                        "history_mean": mean, "history_scale": scale}

        with tempfile.TemporaryDirectory() as d:
            p = Path(d); cfg = json.loads(EXAMPLE.read_text())
            cfg["output_directory"] = str(p/"out")
            cfg["device"] = "cpu"
            cfg["model"]["history"] = 3
            cfg["training"].update(updates=2, batch_size=2, validate_every=1, select_windows=4)
            cfg["data"].update(roles={"FIT": [0, 15], "SELECT": [18, 30], "CAL": [33, 45]}, guard=3)
            rng = np.random.default_rng(12)
            for role, n in (("train", 45), ("test", 151)):
                path = p/(role+".npy")
                a = rng.normal(size=(n, 2)); a[0, 0] = np.nan
                np.save(path, a)
                cfg["data"][role] = {"path": str(path), "shape": [n, 2], "sha256": run.sha256(path)}
            cfg["release"] = {"source_sha256": {}}
            conf = p/"config.json"; conf.write_text(json.dumps(cfg))
            with patch.object(candidate_tf_lag, "candidate_from_config", side_effect=lambda path, mode: Tiny(mode)):
                run.prepare(cfg, conf)
                ids = json.loads((p/"out/prepare/initialization.json").read_text())
                for mode in run.MODES:
                    self.assertEqual(ids[mode]["tensor_sha256"]["common.weight"], ids["gated"]["tensor_sha256"]["common.weight"])
                    run.train(cfg, conf, mode)
                    receipt = json.loads((p/("out/train_"+mode+"/receipt.json")).read_text())
                    self.assertGreater(receipt["changed_tensors"], 0)
                    run.infer(cfg, conf, mode)
                    run.profile_infer(cfg, conf, mode)
                    with np.load(p/("out/profile_infer_"+mode+"/fixed_prefix.npz")) as z:
                        np.testing.assert_array_equal(z["index"], np.arange(3, 131))
                    with np.load(p/("out/infer_"+mode+"/TEST.npz")) as z:
                        np.testing.assert_array_equal(z["input_end_inclusive"], z["index"]-1)
                        np.testing.assert_array_equal(z["available_index"], z["index"])
                        self.assertTrue(np.isfinite(z["score"]).all())
            artifact = p/"out/prepare/medians.npy"
            artifact.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                run._read_receipt(cfg, conf, "prepare")


if __name__ == "__main__":
    unittest.main()
