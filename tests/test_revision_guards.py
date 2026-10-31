"""Regression cases from independent REVIEW.md; synthetic inputs only."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from engine_fixture_support import approve_fixture, calibration_file, record
from hybrid_tsad.engine.identity import SCORE_SPACE, operation_manifest
from hybrid_tsad.engine.pipeline import check_operation_inputs, load_operation, run_operation

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class ApprovalIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        contract, approval = self.root / "contract.md", self.root / "approval.json"
        contract.write_text("synthetic engineering fixture only")
        approval.write_text("{}")
        self.calibration = calibration_file(self.root / "calibration.json", scale=.7, threshold=.5)
        self.bank = self.root / "bank.npz"
        self.bank.write_bytes(b"opaque fixed score-bank; guards must fail before loading")
        metadata = self.root / "bank_metadata.json"
        metadata.write_text(json.dumps(dict(bank_sha256=record(self.bank)["sha256"],
                                           source_calibration_sha256=record(self.calibration)["sha256"],
                                           score_space=SCORE_SPACE, score_field="standardized_score", source_scale=.7)))
        labels, coverage = self.root / "labels.npy", self.root / "coverage.json"
        labels.write_bytes(b"must never be loaded by rejection tests")
        coverage.write_text("[]")
        self.cfg = dict(engine_schema_version=2, operation="evaluate", contract=record(contract), approval=record(approval),
                        inputs=dict(bank=record(self.bank), bank_metadata=record(metadata), calibration=record(self.calibration),
                                    labels=record(labels), coverage=record(coverage)),
                        parameters={"threshold": .5}, score_space=SCORE_SPACE, output=str(self.root / "output"))
        self.path = self.root / "operation.json"
        approve_fixture(self.cfg, self.path)

    def check(self):
        cfg, records, paths, output = load_operation(self.path, self.cfg["operation"])
        check_operation_inputs(records, paths, cfg, output)

    def test_unchanged_operation_is_bound_without_approval_self_reference(self):
        self.check()
        cfg, _, paths, out = load_operation(self.path, "evaluate")
        _, before = operation_manifest(cfg, paths, out)
        cfg["approval"]["sha256"] = "0" * 64
        _, after = operation_manifest(cfg, paths, out)
        self.assertEqual(before, after)

    def test_same_approval_rejects_changed_threshold_and_destination(self):
        self.cfg["parameters"]["threshold"] = .9
        self.cfg["output"] = str(self.root / "other_output")
        self.path.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError, "operation manifest"):
            run_operation(self.path, "evaluate", execute=True, allow_labels=True)
        self.assertFalse((self.root / "other_output").exists())

    def test_even_reapproved_threshold_must_match_calibration(self):
        self.cfg["parameters"]["threshold"] = .9
        approve_fixture(self.cfg, self.path)
        with self.assertRaisesRegex(ValueError, "threshold differs"):
            self.check()

    def test_infer_source_scale_cannot_change_under_existing_or_new_approval(self):
        self.cfg.update(operation="infer", parameters=dict(seed=0, mode="hybrid", mask_seed=84551,
                                                           source_scale=.7, job="fit", readout_job="query"))
        self.cfg["inputs"] = {k: record(self.bank) for k in ("official", "checkpoint", "array", "query")}
        self.cfg["inputs"]["calibration"] = record(self.calibration)
        approve_fixture(self.cfg, self.path)
        self.check()
        self.cfg["parameters"]["source_scale"] = .8
        self.path.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError, "operation manifest"):
            self.check()
        approve_fixture(self.cfg, self.path)
        with self.assertRaisesRegex(ValueError, "source_scale differs"):
            self.check()

    def test_bank_unit_calibration_and_bank_identity_mismatch_rejected(self):
        p = Path(self.cfg["inputs"]["bank_metadata"]["path"])
        baseline = json.loads(p.read_text())
        for key, wrong in (("score_space", "legacy_raw"), ("score_field", "raw_score"),
                           ("source_scale", 1.0), ("source_calibration_sha256", "0" * 64),
                           ("bank_sha256", "0" * 64)):
            with self.subTest(key=key):
                value = {**baseline, key: wrong}
                p.write_text(json.dumps(value))
                self.cfg["inputs"]["bank_metadata"] = record(p)
                approve_fixture(self.cfg, self.path)
                with self.assertRaisesRegex(ValueError, "Bank score units"):
                    self.check()

    def test_calibration_or_config_legacy_units_rejected(self):
        value = json.loads(self.calibration.read_text())
        value["score_space"] = "legacy_raw"
        self.calibration.write_text(json.dumps(value))
        self.cfg["inputs"]["calibration"] = record(self.calibration)
        approve_fixture(self.cfg, self.path)
        with self.assertRaisesRegex(ValueError, "no implicit legacy mapping"):
            self.check()
        self.cfg["score_space"] = "legacy_raw"
        self.path.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError, "score-space"):
            load_operation(self.path, "evaluate")

    @unittest.skipUnless(HAS_NUMPY, "existing NumPy required")
    def test_raw_values_mislabeled_as_standardized_fail_before_label_loading(self):
        import numpy as np
        raw = np.array([.1, .9, .1], dtype="float32")
        np.savez_compressed(self.bank, time_index=np.arange(3), raw_score=raw, standardized_score=raw)
        self.cfg["inputs"]["bank"] = record(self.bank)
        metadata_path = Path(self.cfg["inputs"]["bank_metadata"]["path"])
        metadata = json.loads(metadata_path.read_text())
        metadata["bank_sha256"] = record(self.bank)["sha256"]
        metadata_path.write_text(json.dumps(metadata))
        self.cfg["inputs"]["bank_metadata"] = record(metadata_path)
        approve_fixture(self.cfg, self.path)
        with self.assertRaisesRegex(ValueError, "raw_score/source_scale"):
            run_operation(self.path, "evaluate", execute=True, allow_labels=True)


@unittest.skipUnless(HAS_NUMPY, "existing NumPy required")
class CoverageRevisionTests(unittest.TestCase):
    @staticmethod
    def event(eid="a", start=1, end=3, scored=2, complete=True):
        return dict(event_id=eid, start=start, end_exclusive=end, total_points=end-start,
                    scored_points=scored, eligible_complete=complete)

    def test_false_complete_gap_rejected_but_true_incomplete_retained(self):
        import numpy as np
        from hybrid_tsad.engine.evaluation import validate_coverage
        index, y = np.array([0, 1, 4, 5]), np.array([0, 1, 1, 0])
        event = self.event(start=1, end=5, scored=2)
        with self.assertRaisesRegex(ValueError, "complete eligibility"):
            validate_coverage(index, y, [event])
        event["eligible_complete"] = False
        self.assertEqual(len(validate_coverage(index, y, [event])), 1)

    def test_coverage_corruption_cases_rejected_before_statistics(self):
        import numpy as np
        from hybrid_tsad.engine.evaluation import validate_coverage
        index, y = np.arange(5), np.array([0, 1, 1, 0, 0])
        valid = self.event()
        cases = [[], [valid, valid], [valid, self.event("b", 2, 4)],
                 [{**valid, "total_points": 9}], [{**valid, "scored_points": 1}],
                 [{**valid, "start": -1}], [{**valid, "start": 1.0}],
                 [{**valid, "end_exclusive": 4, "total_points": 3, "scored_points": 3}],
                 [{**valid, "eligible_complete": False}]]
        for coverage in cases:
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                validate_coverage(index, y, coverage)

    @unittest.skipUnless(HAS_SKLEARN, "existing scikit-learn required")
    def test_legal_incomplete_positive_points_remain_in_ap(self):
        import numpy as np
        from sklearn.metrics import average_precision_score
        from hybrid_tsad.engine.evaluation import evaluate_fixed_scores
        index, y = np.array([0, 1, 4, 5]), np.array([0, 1, 1, 0])
        score = np.array([0., .8, .3, .2])
        result = evaluate_fixed_scores(index, y, score, .5, [self.event(start=1, end=5, scored=2, complete=False)])
        self.assertEqual(result["metrics"]["AP"], average_precision_score(y, score))
        self.assertEqual(result["metrics"]["complete_events"], 0)
        self.assertIsNone(result["events"][0]["any_hit"])

    @unittest.skipUnless(HAS_SKLEARN, "existing scikit-learn required")
    def test_multi_block_affiliation_is_blockwise_and_global_na(self):
        import numpy as np
        from hybrid_tsad.engine.evaluation import evaluate_fixed_scores
        replies = iter([.25, .75])
        def callback(*args):
            value = next(replies)
            return {"precision": value, "recall": value}
        result = evaluate_fixed_scores(np.array([0, 1, 10, 11]), np.array([0, 1, 0, 1]),
                                       np.array([.1, .9, .1, .9]), .5,
                                       [self.event(start=1, end=2, scored=1), self.event("b", 11, 12, 1)], affiliation=callback)
        self.assertEqual([row["Aff_F1"] for row in result["affinity"]], [.25, .75])
        self.assertTrue(np.isnan(result["metrics"]["Aff_F1"]))
        self.assertEqual(result["metrics"]["Aff_F1_reason"], "MULTIPLE_REAL_BLOCKS_NO_GLOBAL_AGGREGATION")

    @unittest.skipUnless(HAS_SKLEARN, "existing scikit-learn required")
    def test_no_gt_block_skips_callback_and_single_legal_block_has_global_value(self):
        import numpy as np
        from hybrid_tsad.engine.evaluation import evaluate_fixed_scores
        calls = []
        def callback(*args):
            calls.append(args)
            return {"precision": .6, "recall": .6}
        result = evaluate_fixed_scores(np.array([0, 1, 10, 11]), np.array([0, 0, 0, 1]),
                                       np.array([.1, .2, .1, .9]), .5,
                                       [self.event(start=11, end=12, scored=1)], affiliation=callback)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["affinity"][0]["reason"], "NO_GROUND_TRUTH_EVENTS")
        self.assertTrue(np.isnan(result["metrics"]["Aff_F1"]))
        single = evaluate_fixed_scores(np.array([0, 1]), np.array([0, 1]), np.array([.1, .9]), .5,
                                       [self.event(start=1, end=2, scored=1)], affiliation=callback)
        self.assertEqual(single["metrics"]["Aff_F1"], .6)
        self.assertIsNone(single["metrics"]["Aff_F1_reason"])
