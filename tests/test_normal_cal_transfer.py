import unittest
import numpy as np
from hybrid_tsad.normal_cal_transfer import normal_metrics,subsample


class NormalTransfer(unittest.TestCase):
    def test_confirmation_never_bridges_missing_or_role_points(self):
        ix=np.array([1,2,4,5,6,10,11]);s=np.ones(7)
        strict,c3=normal_metrics(ix,s,.5)
        self.assertEqual(strict['normal_fragments'],3)
        self.assertEqual(c3['normal_FP'],1)
        self.assertEqual(c3['normal_fragments'],1)
        self.assertEqual(c3['normal_points'],7)

    def test_run_boundaries_and_strict_threshold(self):
        ix=np.broadcast_to(np.arange(4),(2,4));s=np.array([[1,1,2,2],[2,2,1,1]])
        strict,c3=normal_metrics(ix,s,1)
        self.assertEqual(strict['normal_FP'],4)
        self.assertEqual(c3['normal_FP'],0)
        self.assertFalse(strict['empirical_budget_met'])

    def test_cal_sample_indices_do_not_depend_on_scores(self):
        ix=np.r_[np.arange(0,400),np.arange(900,1400)]
        selected=subsample(ix)
        self.assertEqual(len(selected),512)
        self.assertEqual(len(set(selected)),512)
        self.assertTrue(np.isin(selected,ix).all())
        with self.assertRaises(AssertionError):subsample(np.arange(511))


if __name__=='__main__':unittest.main()
