"""Synthetic CPU ranking and full-label event-coverage checks, no models."""
import unittest
import numpy as np
from hybrid_tsad.engine.evaluation import evaluate_point_ranking


class PointRankingTests(unittest.TestCase):
    def test_perfect_ranking_and_original_partial_event(self):
        labels = np.array([0, 1, 1, 0, 1, 1, 1], dtype=np.float64)
        mask = np.arange(7) < 6
        result = evaluate_point_ranking(np.arange(100, 106), np.array([0., .9, .8, .1, .7, .6]),
                                        labels, mask, query_start=100)
        m = result["metrics"]
        self.assertEqual((m["AP"], m["ROC_AUC"], m["trapezoid_PR_AUC"]), (1., 1., 1.))
        self.assertEqual(m["AP_divided_by_pi"], 1.5)
        self.assertEqual(m["positive_points"], 4)
        a, b = result["event_coverage"]
        self.assertEqual((a["start"], a["end_exclusive"], a["complete"]), (101, 103, True))
        self.assertEqual((b["total_points"], b["scored_points"], b["complete"]), (3, 2, False))
        self.assertEqual(b["coverage_ratio"], 2/3)
        self.assertTrue(b["touches_query_end"])

    def test_offset_nonbinary_nonfinite_and_compressed_coverage_refused(self):
        y = np.array([0, 1, 0, 1]); mask = np.array([True, True, True, False])
        for ix, s, labels, valid in [
                (np.arange(1, 4), np.zeros(3), y, mask),
                (np.array([0, 0, 2]), np.zeros(3), y, mask),
                (np.arange(3), np.array([0., np.nan, 1.]), y, mask),
                (np.arange(3), np.zeros(3), np.array([0, 2, 0, 1]), mask),
                (np.arange(3), np.zeros(3), y, np.ones(3, bool))]:
            with self.subTest(ix=ix), self.assertRaises(ValueError):
                evaluate_point_ranking(ix, s, labels, valid, query_start=0)

    def test_single_class_is_NA_without_fabricated_metric(self):
        result = evaluate_point_ranking(np.arange(3), np.arange(3.), np.ones(3), np.ones(3, bool), query_start=0)
        self.assertIsNone(result["metrics"]["AP"])
        self.assertIsNone(result["metrics"]["ROC_AUC"])
        self.assertEqual(result["metrics"]["ranking_metric_NA_reason"], "SINGLE_CLASS_SCORED_LABELS")
        event = result["event_coverage"][0]
        self.assertTrue(event["touches_query_start"] and event["touches_query_end"])


if __name__ == "__main__":
    unittest.main()
