import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1.event_details import event_details, write_new_details


class ObservationDetails(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros(20, np.int8); self.labels[5:15] = 1

    def test_partial_unscored_is_unknown_not_ten_point_miss(self):
        row, = event_details(self.labels, np.arange(8), np.zeros(8, bool))
        self.assertEqual((row['scored_points'], row['missing_points']), (3, 7))
        self.assertFalse(row['eligible_for_event_metric'])
        self.assertEqual(row['observation_status'], 'UNKNOWN_PARTIAL')
        self.assertIsNone(row['coverage']); self.assertIsNone(row['longest_miss_points'])
        self.assertEqual(row['observed_only_coverage'], 0.)
        self.assertEqual(row['longest_observed_miss_points'], 3)
        self.assertEqual(row['missing_intervals'], [[8, 15]])

    def test_archived_actual_evaluator_detail_vs_new_NA(self):
        path = Path(__file__).resolve().parents[1] / 'review_artifacts/code/preparations/hybrid_left_channel_blindspot_20261113_v1/fixed_metrics.py'
        spec = importlib.util.spec_from_file_location('archived_blindspot_metrics', path)
        archived = importlib.util.module_from_spec(spec); spec.loader.exec_module(archived)
        summary, details = archived.evaluate(np.arange(8), np.zeros(8), np.zeros(5), self.labels)
        row = details[0]
        self.assertFalse(row['main_eligible'])
        self.assertEqual(row['coverage'], 0.)
        self.assertEqual(row['longest_miss_points'], 10)
        self.assertEqual(summary[0]['Ntrue'], 0)  # Historical main aggregate already excludes it.
        new, = event_details(self.labels, np.arange(8), np.zeros(8, bool))
        self.assertIsNone(new['coverage']); self.assertIsNone(new['longest_miss_points'])

    def test_unknown_gap_breaks_observed_miss_run(self):
        row, = event_details(self.labels, np.array([5, 6, 10, 11, 12]), np.zeros(5, bool))
        self.assertEqual(row['longest_observed_miss_points'], 3)
        self.assertEqual(row['missing_intervals'], [[7, 10], [13, 15]])

    def test_full_observed_true_miss_is_zero_coverage(self):
        row, = event_details(self.labels, np.arange(20), np.zeros(20, bool), steady_after=2, long_min_length=5)
        self.assertTrue(row['eligible_for_event_metric'])
        self.assertEqual(row['coverage'], 0.)
        self.assertEqual(row['longest_miss_points'], 10)
        self.assertEqual(row['steady_coverage'], 0.)

    def test_no_observations_returns_NA_not_zero(self):
        row, = event_details(self.labels, np.array([], dtype=int), np.array([], dtype=bool))
        self.assertEqual(row['observation_status'], 'UNKNOWN_UNOBSERVED')
        self.assertIsNone(row['observed_only_coverage'])
        self.assertIsNone(row['longest_observed_miss_points'])

    def test_original_clock_boundary_and_steady_permissions(self):
        row, = event_details(self.labels, np.arange(105, 115), np.ones(10, bool), original_start=100, steady_after=2, long_min_length=5)
        self.assertEqual((row['start'], row['end_exclusive']), (105, 115))
        self.assertEqual(row['coverage'], 1.); self.assertEqual(row['steady_coverage'], 1.)
        labels = np.ones(10, np.int8)
        row, = event_details(labels, np.arange(10), np.ones(10, bool))
        self.assertTrue(row['boundary_censored']); self.assertFalse(row['eligible_for_event_metric'])
        self.assertIsNone(row['coverage']); self.assertEqual(row['observed_only_coverage'], 1.)

    def test_exclusive_new_CSV_writes_NA_empty_and_preserves_existing(self):
        rows = event_details(self.labels, np.arange(8), np.zeros(8, bool))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'details_v2.csv'
            receipt = write_new_details(path, rows); before = path.read_bytes()
            self.assertEqual(receipt['rows'], 1)
            with path.open(newline='', encoding='utf-8') as handle:
                row, = list(csv.DictReader(handle))
            self.assertEqual(row['coverage'], ''); self.assertEqual(row['longest_miss_points'], '')
            with self.assertRaises(FileExistsError): write_new_details(path, rows)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_axes_and_implicit_alarm_conversion_rejected(self):
        for index, bits in [(np.array([6, 5]), np.zeros(2, bool)), (np.array([5, 5]), np.zeros(2, bool)), (np.array([20]), np.zeros(1, bool)), (np.array([5]), np.array([0]))]:
            with self.assertRaises(ValueError): event_details(self.labels, index, bits)


if __name__ == '__main__':
    unittest.main()
