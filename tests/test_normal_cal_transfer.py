import unittest
import numpy as np
from hybrid_tsad.normal_cal_transfer import normal_metrics,subsample
from hybrid_tsad.normal_cal_readout import normal_intervals


class NormalTransfer(unittest.TestCase):
    def test_official_event_complement_clips_to_authorized_role(self):
        # No QUERY labels or scores enter qualification; half-open boundaries.
        got=normal_intervals(10,30,np.array([0,15,24,35]),np.array([12,18,26,40]))
        self.assertEqual(got,[(12,15),(18,24),(26,30)])
        self.assertEqual(normal_intervals(19,23,np.array([15]),np.array([30])),[])

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
