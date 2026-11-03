"""New public metadata/ordering tests, not repeated neural or metric tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from hybrid_tsad.hybrid_v1 import public_evaluate_cli as public


def fixture():
    cfg = dict(operation=public.OPERATION, profile=public.PROFILE,
               statistics=dict(public.named.STATISTICS_NA), comparisons=public.fixed_pairs(),
               stream=list(public.STREAM), history=512, original_start=0,
               original_length=73729, expected_records=73217,
               affiliation_identity="not_opened_aff.json", affiliation_identity_sha256=public.AFF_SHA,
               labels=dict(path="NOT_OPENED_LABELS.npz", format="npz", key="native_test_labels",
                           shape=[73729], sha256=public.LABEL_SHA), banks=[])
    ranges = [dict(zip(("domain", "entity", "run"), public.STREAM), start=512, end=73729)]
    for name in public.METHODS:
        model = name in public.MODELS
        seed = public.SEEDS[public.MODELS.index(name)] if model else None
        checkpoint, calibration = public.FULL_SOURCE[seed] if model else ("0" * 64, public.BASELINE_CAL[name])
        cfg["banks"].append(dict(id=name, kind="source_model" if model else "deterministic_baseline",
            directory="NOT_OPENED_BANK_" + name, control_id="FULL_SOURCE_seed%d" % seed if model else name,
            training_seed=seed, inference_seed=seed, calibration_origin="source", adaptation_mode="Z",
            ranges=copy.deepcopy(ranges), identity=dict(input_sha256=public.MSL_NPZ,
                checkpoint_sha256=checkpoint, calibration_sha256=calibration,
                config_sha256="1" * 64, source_sha256="2" * 64)))
    return cfg


class PublicEvaluationTests(unittest.TestCase):
    def test_all_five_fixed_metadata_without_labels(self):
        public.contract(fixture())
        self.assertEqual(len(public.fixed_pairs()), 6)
        self.assertEqual({p["candidate"] for p in public.fixed_pairs()}, set(public.MODELS))

    def test_missing_duplicate_or_best_seed_rejected(self):
        for change in ("missing", "duplicate", "selection", "pair"):
            cfg = fixture()
            if change == "missing": cfg["banks"].pop()
            if change == "duplicate": cfg["banks"][1]["directory"] = cfg["banks"][0]["directory"]
            if change == "selection": cfg["selected_seed"] = 2026
            if change == "pair": cfg["comparisons"].pop()
            with self.assertRaises(ValueError): public.contract(cfg)

    def test_original_clock_shape_tail_and_raw_identity_required(self):
        for field, value in (("history", 100), ("original_start", 58317),
                             ("original_length", 73728), ("expected_records", 73186)):
            cfg = fixture(); cfg[field] = value
            with self.assertRaises(ValueError): public.contract(cfg)
        cfg = fixture(); cfg["banks"][0]["ranges"][0]["end"] -= 31
        with self.assertRaises(ValueError): public.contract(cfg)

    def test_no_h5_or_target_cal_and_no_fake_baseline_seed(self):
        for mode in ("h5", "target", "baseline_seed"):
            cfg = fixture()
            if mode == "h5": cfg["banks"][0]["identity"]["checkpoint_sha256"] = "f" * 64
            if mode == "target": cfg["banks"][0]["calibration_origin"] = "target"
            if mode == "baseline_seed": cfg["banks"][-1]["training_seed"] = 2025
            with self.assertRaises(ValueError): public.contract(cfg)

    def test_original_label_cache_schema_and_aff_identity_required(self):
        for field, value in (("key", "old_scores"), ("sha256", "0" * 64), ("shape", [73217])):
            cfg = fixture(); cfg["labels"][field] = value
            with self.assertRaises(PermissionError): public.contract(cfg)
        cfg = fixture(); cfg["affiliation_identity_sha256"] = "0" * 64
        with self.assertRaises(PermissionError): public.contract(cfg)

    def test_failed_bank_gate_precedes_every_label_opener(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(); cfg["output"] = str(Path(temp) / "fresh")
            with mock.patch.object(public, "bank_gates", side_effect=ValueError("bank not accepted")), \
                    mock.patch.object(public, "load_labels") as labels, \
                    mock.patch.object(public.named, "scalar_inputs") as scalars:
                with self.assertRaises(ValueError): public.run_evaluation(cfg)
            labels.assert_not_called(); scalars.assert_not_called()

    def test_all_scalar_replays_must_precede_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(); cfg["output"] = str(Path(temp) / "fresh")
            with mock.patch.object(public, "bank_gates", return_value=[dict(id=n) for n in public.METHODS]), \
                    mock.patch.object(public.named, "scalar_inputs", return_value={}), \
                    mock.patch.object(public, "load_labels") as labels:
                with self.assertRaises(ValueError): public.run_evaluation(cfg)
            labels.assert_not_called()

    def test_existing_serializer_and_three_seed_aggregation_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(); cfg["output"] = str(Path(temp) / "fresh")
            calls = []
            def serialize(config, gates, arrays, labels):
                calls.append("serialize")
                out = Path(config["output"]); out.mkdir()
                for i, name in enumerate(public.METHODS):
                    folder = out / name; folder.mkdir()
                    value = dict(ranking={"AP": (i + 1) / 10.}, streams={})
                    (folder / "result.json").write_text(json.dumps(value))
                return dict(status="TEST_SYNTHETIC_SERIALIZER_STUB")
            def labels(spec):
                calls.append("labels")
                return np.zeros(73729, np.int8)
            with mock.patch.object(public, "bank_gates", side_effect=lambda c: calls.append("banks") or [dict(id=n) for n in public.METHODS]), \
                    mock.patch.object(public.named, "scalar_inputs", side_effect=lambda c: calls.append("scalars") or {n: {} for n in public.METHODS}), \
                    mock.patch.object(public, "load_labels", side_effect=labels), \
                    mock.patch.object(public.serializer, "_save_results", side_effect=serialize):
                public.run_evaluation(cfg)
            self.assertEqual(calls, ["banks", "scalars", "labels", "serialize"])
            summary = json.loads((Path(cfg["output"]) / "MODEL_SEED_SUMMARY.json").read_text())
            self.assertEqual(summary["model_seeds"], [2025, 2026, 2027])
            ap = summary["model_mean_SD"]["/ranking/AP"]
            self.assertAlmostEqual(ap["mean"], .2)
            self.assertAlmostEqual(ap["SD"], .1)
            self.assertEqual(ap["worst_seed"], 2025)
            self.assertFalse(summary["seed_selection"])
            self.assertEqual(set(summary["deterministic_controls"]), set(public.CONTROLS))


if __name__ == "__main__":
    unittest.main()
