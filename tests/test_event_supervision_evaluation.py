import unittest
import numpy as np
from hybrid_tsad.event_supervision_evaluation import CAL_AXIS, CAL_SELECTED, QUERY_AXIS, extract, threshold


class FixedEventReadoutTest(unittest.TestCase):
    def test_sparse_and_full_cal_same_threshold(self):
        score = np.linspace(-3., 7., len(CAL_AXIS))
        pos = np.searchsorted(CAL_AXIS, CAL_SELECTED)
        self.assertEqual(threshold(CAL_AXIS,score),threshold(CAL_SELECTED,score[pos]))

    def test_batch_mean_and_future_timestamp_rejected(self):
        with self.assertRaises(AssertionError):
            extract(dict(index=QUERY_AXIS,score=np.array(0.)), 'QUERY')
        with self.assertRaises(AssertionError):
            extract(dict(index=QUERY_AXIS,score=np.zeros(len(QUERY_AXIS)),
                         max_input_index=QUERY_AXIS+1),'QUERY')

    def test_wrong_axis_and_nonfinite_rejected(self):
        with self.assertRaises(AssertionError):
            extract(dict(index=CAL_SELECTED+1,score=np.zeros(512)), 'CAL')
        with self.assertRaises(AssertionError):
            extract(dict(index=CAL_SELECTED,score=np.full(512,np.nan)),'CAL')


if __name__=='__main__': unittest.main()
