import unittest
import numpy as np
from hybrid_tsad.g36_paired_diagnostic import window_all, quadrants, longest
from hybrid_tsad.g36_evaluation import confirm


class PairedDiagnosticTest(unittest.TestCase):
    def test_quadrants_exhaustive(self):
        np.testing.assert_array_equal(quadrants([0,1,0,1],[0,0,1,1]),[0,1,2,3])

    def test_history_difference_not_hidden_by_current_match(self):
        a=np.ones(8,bool);a[3]=False
        np.testing.assert_array_equal(window_all(a,3),[0,0,1,0,0,0,1,1])

    def test_no_backfill_or_compressed_time(self):
        a=confirm([1,1,1,0,1,1,1]);mask=np.array([1,0,1,0,1,0,1],bool)
        np.testing.assert_array_equal(a,[0,0,1,0,0,0,1])
        self.assertEqual(longest(mask & ~a),1)

    def test_incomplete_window_never_identical(self):
        self.assertFalse(window_all([1,1],3).any())


if __name__=='__main__':unittest.main()
