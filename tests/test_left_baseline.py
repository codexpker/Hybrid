"""CPU-only synthetic permission/loader tests; never instantiate a model."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hybrid_tsad import left_baseline as lb


class NormalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prefix = np.random.default_rng(3).normal(size=(1000, 25))

    def test_role_footprints_and_float32_dummy_labels(self):
        fit = lb.NormalPrefixDataset(self.prefix, "train", 192, 1)
        val = lb.NormalPrefixDataset(self.prefix, "val", 192, 192)
        self.assertEqual(len(fit), 609)
        self.assertEqual(fit.footprint(608), (608, 800))
        self.assertEqual(val.footprint(0), (800, 992))
        self.assertLessEqual(fit.footprint(len(fit)-1)[1], val.footprint(0)[0])
        x, y = fit[0]
        self.assertEqual(x.shape, (192, 25))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.array_equal(y, np.zeros((192, 1), np.float32)))
        with self.assertRaises(IndexError):
            fit[len(fit)]

    def test_query_and_unknown_roles_refused(self):
        for flag in ["test", "query", "score", "labels", "cal"]:
            with self.subTest(flag=flag), self.assertRaises(PermissionError):
                lb.NormalPrefixDataset(self.prefix, flag)

    def test_official_loader_exact_on_synthetic_fixture(self):
        root = Path(os.environ["LEFT_OFFICIAL_ROOT"])
        spec = importlib.util.spec_from_file_location("left_loader_test_reference", root/"data_provider/data_provider.py")
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        # Author loader sees a purely synthetic full dataframe; no disk data or real labels.
        extra = np.full((137, 25), 77.)
        values = np.concatenate([self.prefix, extra])
        full = pd.DataFrame({name: values[:, i] for i, name in enumerate(lb.FEATURES)})
        full["label"] = 0
        for flag, step in [("train", 1), ("val", 192), ("init", 192)]:
            with self.subTest(flag=flag), patch.object(official, "read_data", return_value=full):
                reference = official.TrainSegLoader("SYNTHETIC_NEVER_OPEN", 1000, 192, step, flag, 1.)
                actual = lb.NormalPrefixDataset(self.prefix, flag, 192, step)
                self.assertEqual(len(reference), len(actual))
                self.assertTrue(np.array_equal(reference.scaler.mean_, actual.scaler.mean_))
                self.assertTrue(np.array_equal(reference.scaler.scale_, actual.scaler.scale_))
                for i in sorted(set([0, len(actual)//2, len(actual)-1])):
                    for a, b in zip(reference[i], actual[i]):
                        self.assertTrue(np.array_equal(a, b))

    def test_actual_long_table_parser_layout_and_float64_before_scaler(self):
        root = Path(os.environ["LEFT_OFFICIAL_ROOT"])
        spec = importlib.util.spec_from_file_location("left_parser_test_reference", root/"data_provider/data_provider.py")
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        values = np.concatenate([self.prefix + 1e7, np.zeros((137, 25))])
        count = len(values)
        data = pd.DataFrame({"date": ["2020-01-01"]*(count*26),
                             "data": np.concatenate([values[:, i] for i in range(25)] + [np.zeros(count)]),
                             "cols": np.repeat(lb.FEATURES+["label"], count)})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"synthetic_long.csv"
            data.to_csv(path, index=False)
            parsed = official.read_data(str(path))
            prefix = parsed.iloc[:1000].loc[:, parsed.columns != "label"].to_numpy()
            self.assertTrue(prefix.flags.f_contiguous)
            actual = lb.NormalPrefixDataset(np.array(prefix, order="C"), "train", 192, 1)
            reference = official.TrainSegLoader(str(path), 1000, 192, 1, "train", 1.)
            for name in ["mean_", "var_", "scale_", "n_samples_seen_"]:
                self.assertTrue(np.array_equal(getattr(actual.scaler, name), getattr(reference.scaler, name)))
            for i in [0, 304, 608]:
                self.assertTrue(np.array_equal(actual[i][0], reference[i][0]))
            with self.assertRaises(ValueError):
                lb.NormalPrefixDataset(prefix.astype(np.float32), "train")

    def test_future_query_perturbation_does_not_affect_prefix_windows(self):
        a = lb.NormalPrefixDataset(self.prefix.copy(), "train")
        combined = np.concatenate([self.prefix, np.full((200, 25), -999.)])
        b = lb.NormalPrefixDataset(combined[:1000], "train")
        self.assertTrue(np.array_equal(a[0][0], b[0][0]))
        self.assertTrue(np.array_equal(a[len(a)-1][0], b[len(b)-1][0]))


class PrefixIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.array = self.root/"prefix.npy"
        np.save(self.array, np.zeros((132481, 25), np.float64))
        self.metadata = self.root/"meta.json"
        self.meta = {"source_sha256": lb.PSM_SHA, "feature_names": lb.FEATURES,
                     "raw_values_not_normalized": True,
                     "array_sha256": {"prefix.npy": lb.file_sha(self.array)}}
        self.write_meta()

    def write_meta(self):
        self.metadata.write_text(json.dumps(self.meta))

    def test_prepare_derivative_preserves_raw_values_and_lineage(self):
        out = self.root/"derived"
        result = lb.prepare_prefix(self.array, self.metadata, out)
        self.assertFalse(result["query_or_labels_read"])
        self.assertFalse(result["scaler_fitted"])
        self.assertEqual(lb.file_sha(self.array), lb.file_sha(out/"prefix.npy"))
        a, meta = lb.load_prefix(out/"prefix.npy", out/"meta.json")
        self.assertEqual(a.shape, (132481, 25))

    def test_source_order_and_normalized_cache_rejected(self):
        for key, value in [("source_sha256", "0"*64), ("feature_names", list(reversed(lb.FEATURES))),
                           ("raw_values_not_normalized", False)]:
            original = copy.deepcopy(self.meta)
            self.meta[key] = value
            self.write_meta()
            with self.assertRaises(ValueError):
                lb.load_prefix(self.array, self.metadata)
            self.meta = original

    def test_query_path_missing_hash_and_changed_bytes_refused(self):
        with self.assertRaises(ValueError):
            lb.load_prefix(self.root/"test.npy", self.metadata)
        self.meta["array_sha256"] = {}
        self.write_meta()
        with self.assertRaises(ValueError):
            lb.load_prefix(self.array, self.metadata)
        self.meta["array_sha256"] = {"prefix.npy": lb.file_sha(self.array)}
        self.write_meta()
        with self.array.open("ab") as f:
            f.write(b"changed")
        with self.assertRaises(ValueError):
            lb.load_prefix(self.array, self.metadata)


class ConfigPermissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        official = Path(os.environ["LEFT_OFFICIAL_ROOT"])
        hashes = {name: lb.file_sha(official/name) for name in lb.MODEL_FILES + ["configs/PSM/model_configs_Left.json"]}
        self.cfg = {"operation": "profile", "track": "R", "upstream_commit": lb.COMMIT,
                    "seed": 2025, "source_root": str(official), "source_hashes": hashes,
                    "batch_size": 128, "steps": 3, "max_seconds": 180,
                    "max_allocated_GiB": 20, "data": None, "output": str(self.root/"output"),
                    "approval_path": str(self.root/"approval.json")}
        Path(self.cfg["approval_path"]).write_text(json.dumps({"status": "PENDING"}))
        self.path = self.root/"cfg.json"

    def save(self):
        self.path.write_text(json.dumps(self.cfg))

    def test_preview_and_pending_no_execute(self):
        self.save()
        self.assertFalse(lb.run(self.path)["executed"])
        with patch.object(lb, "profile", side_effect=AssertionError("must not execute")):
            with self.assertRaises(PermissionError):
                lb.run(self.path, execute=True)
        self.assertFalse(Path(self.cfg["output"]).exists())

    def test_score_c_query_data_wrong_seed_or_steps_refused(self):
        for key, value in [("operation", "score"), ("track", "C"), ("data", {"array":"query.npy"}),
                           ("seed", 2026), ("steps", 4), ("batch_size", 2)]:
            original = copy.deepcopy(self.cfg)
            self.cfg[key] = value
            self.save()
            with self.assertRaises(ValueError):
                lb.run(self.path)
            self.cfg = original

    def test_unknown_official_bytes_and_existing_output_rejected(self):
        Path(self.cfg["output"]).mkdir()
        self.save()
        with self.assertRaises(FileExistsError):
            lb.run(self.path)
        Path(self.cfg["output"]).rmdir()
        self.cfg["source_hashes"][lb.MODEL_FILES[0]] = "0"*64
        self.save()
        with self.assertRaises(ValueError):
            lb.run(self.path)

    def test_wrong_release_identity_or_query_permission_refused(self):
        self.save()
        approval = {"status": "PRIMARY_RELEASED_LEFT_BASELINE_PROFILE", "config_sha256": "0"*64,
                    "adapter_sha256": lb.file_sha(lb.__file__), "allow_query": False,
                    "allow_labels": False, "max_seconds": 180}
        p = Path(self.cfg["approval_path"])
        p.write_text(json.dumps(approval))
        with self.assertRaises(PermissionError):
            lb.require_approval(self.path, self.cfg)
        approval.update(config_sha256=lb.file_sha(self.path), allow_query=True)
        p.write_text(json.dumps(approval))
        with self.assertRaises(PermissionError):
            lb.require_approval(self.path, self.cfg)

    def test_source_conflicting_import_rejected(self):
        import types
        fake = types.ModuleType("fake_left_test")
        fake.__file__ = str(self.root/"wrong.py")
        with patch.dict("sys.modules", {"fake_left_test": fake}), self.assertRaises(RuntimeError):
            lb.import_official(self.root, "fake_left_test")


if __name__ == "__main__":
    unittest.main()
