import unittest
import numpy as np
from hybrid_tsad.g36_energy_evaluation import paired_summary


class PairedSummaryTest(unittest.TestCase):
    def test_same_alarms_are_not_fault_specific(self):
        row = paired_summary([2]*5, [2]*5, 1., [1]*5)
        self.assertEqual(row['both_points'], 3)
        self.assertEqual(row['fault_only_points'], 0)

    def test_stratum_does_not_compress_or_backfill(self):
        row = paired_summary([0]*7, [2, 2, 2, 0, 2, 2, 2], 1.,
                             np.array([1, 0, 1, 0, 1, 0, 1], bool))
        self.assertEqual(row['fault_only_points'], 2)
        self.assertEqual(row['neither_points'], 2)
        self.assertEqual(row['longest_no_exclusive_alarm_minutes'], 5)

    def test_empty_stratum_unknown_rate(self):
        row = paired_summary([0]*3, [2]*3, 1., [0]*3)
        self.assertIsNone(row['fault_only_rate'])


if __name__ == '__main__':
    unittest.main()
