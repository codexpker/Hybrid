"""Only synthetic glue tests. No real labels/banks, sklearn/Aff or neural calls."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import stage2_u_evaluation as e
from hybrid_tsad.hybrid_v1 import scoring, evaluation


CAL = dict(scales={k: 1. for k in scoring.COMPONENTS}, threshold=1.,
           weights={"r_tf": .25, "r_ms": .25, "r_pred": .5},
           active_components=list(scoring.COMPONENTS), source_domain_equal_weight=False)


def record(index, missing=False):
    row = dict(domain="fake", entity="fixture", run="only", index=index, available_index=index,
               r_tf=None if missing else 3., r_ms=3., r_pred=3.)
    derived = scoring.apply_calibration(row, CAL)
    row.update(derived); row["reason"] = derived.get("reason")
    if not derived["valid"]:
        row["alarm"] = None
    return row


class Stage2UEvaluationTests(unittest.TestCase):
    def test_original_scorer_and_same_ZT_scales_and_tau(self):
        alternative = dict(CAL, scales={k: 3. for k in scoring.COMPONENTS}, threshold=.5)
        a, aux, residuals = e.records_to_arrays([record(4), record(5)], CAL,
            start=4, end=6, stream=("fake", "fixture", "only"), auxiliary_calibration=alternative)
        self.assertEqual(residuals.shape, (2, 3))
        self.assertEqual(aux["score"][0], scoring.apply_calibration(record(4), alternative)["score"])
        self.assertNotEqual(a["score"][0], aux["score"][0])
        # A fixed sensitivity threshold is the calibration's tau, not any own tau.
        self.assertEqual(alternative["threshold"], .5)

    def test_archived_false_valid_or_reason_is_not_repaired(self):
        bad = record(4); bad["valid"] = False
        with self.assertRaises(ValueError):
            e.records_to_arrays([bad], CAL, start=4, end=5, stream=("fake", "fixture", "only"))
        a, _, _ = e.records_to_arrays([record(4, missing=True)], CAL,
            start=4, end=5, stream=("fake", "fixture", "only"))
        self.assertFalse(a["valid"][0]); self.assertTrue(np.isnan(a["score"][0]))
        self.assertEqual(a["invalid_reasons"], ["MISSING_REQUIRED_COMPONENT"])

    def test_tail_and_arrived_clock_fail_closed(self):
        with self.assertRaises(ValueError):
            e.records_to_arrays([record(4)], CAL, start=4, end=6, stream=("fake", "fixture", "only"))
        bad = record(4); bad["available_index"] = 4.0
        with self.assertRaises(ValueError):
            e.records_to_arrays([bad], CAL, start=4, end=5, stream=("fake", "fixture", "only"))

    def test_full_axis_donor_events_not_normal_and_confirmation_not_reset(self):
        y = np.zeros(12, np.int8); y[1:3] = 1; y[5:10] = 1
        a = dict(index=np.arange(4, 12, dtype=np.int64), score=np.array([0, 2, 2, 2, 0, 0, 0, 0.]),
                 valid=np.ones(8, bool), available_index=np.arange(4, 12, dtype=np.int64))
        r = evaluation.evaluate_arrays(**a, full_labels=y, threshold=1., bank_complete=True,
                                       original_length=12, warmup_points=4)
        self.assertEqual(r["coverage"]["main_eligible_events"], 1)
        self.assertFalse(r["event_coverage"][0]["main_eligible"])
        self.assertIsNone(r["point_coverage"][1]["strict_point"])
        self.assertTrue(r["point_coverage"][7]["confirmed3_current"])
        self.assertFalse(r["point_coverage"][6]["confirmed3_current"])
        self.assertEqual(r["streams"]["strict_point"]["normal"]["normal_exposure_points"], 3)
        rows = e.long_miss(r)
        donor = next(x for x in rows if x["start"] == 1 and x["stream"] == "strict_point")
        self.assertIsNone(donor["longest_observed_consecutive_miss_points"])
        self.assertEqual(donor["unknown_positive_points"], 2)
        query = next(x for x in rows if x["start"] == 5 and x["stream"] == "confirmed3_current")
        self.assertEqual(query["longest_observed_consecutive_miss_points"], 2)

    def test_single_class_NA_is_preserved_not_dependency_failure(self):
        r = evaluation.evaluate_arrays(np.arange(4, 8), np.zeros(4), np.zeros(8, np.int8), 1.,
                                      bank_complete=True, original_length=8, warmup_points=4)
        self.assertEqual(r["ranking"]["ranking_metric_NA_reason"], "SINGLE_CLASS_SCORED_LABELS")
        e.reject_dependency_na(r)
        with self.assertRaises(PermissionError):
            e.reject_dependency_na({"reason": "SKLEARN_UNAVAILABLE_NOT_COMPUTED"})

    def test_failed_all_bank_gate_precedes_truth_opener(self):
        with patch.object(e, "prelabel", side_effect=ValueError("bank missing")), \
                patch.object(e, "load_truth") as truth:
            with self.assertRaises(ValueError):
                e.run({})
            truth.assert_not_called()

    def test_pending_config_cannot_activate(self):
        with self.assertRaises(PermissionError):
            e.contract({"operation": e.OPERATION, "status": "ROOT_PENDING"})
        self.assertNotIn("torch", sys.modules)

    def test_truth_alias_rejected_before_any_file_read(self):
        cfg = dict(truth=dict(path="fake_truth.txt"),
                   banks=[dict(calibration=dict(path="fake_truth.txt"))])
        with patch.object(Path, "open", side_effect=AssertionError("No file I/O allowed")):
            with self.assertRaises(PermissionError):
                e.reject_truth_aliases(cfg)
        e.reject_truth_aliases(dict(truth=dict(path="fake_truth.txt"), banks=[dict(directory="fake_bank")]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
