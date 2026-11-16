import unittest
import numpy as np
from hybrid_tsad.timeradar_cross_variable_reproducibility import RECORDS,ARMS,SEEDS,paired_statistics,curve_arrays,cal_operating_point


def rows():
    return [dict(record=r,arm=a,seed=s,AP=((2,2,5)if a=='cross'else(1,3,2))[s-2024],ROC_AUC=.5,
                 strict_long_steady_macro_coverage=None,threshold=7.)for r in RECORDS for a in ARMS for s in SEEDS]


class SeedStatistics(unittest.TestCase):
    def test_pair_before_mean_and_sample_sd(self):
        summary,paired=paired_statistics(list(reversed(rows())))
        pair=next(r for r in paired if r['record']=='PSM'and r['comparison']=='cross-minus-self'and r['metric']=='AP')
        self.assertEqual(pair['paired_differences'],[1.,-1.,3.])
        self.assertEqual(pair['mean_difference'],1.)
        self.assertEqual(pair['sample_SD_difference'],2.)
        self.assertEqual(len({r['record']for r in summary}),4)

    def test_missing_duplicate_seed_rejected(self):
        data=rows()
        with self.assertRaises(ValueError):paired_statistics(data[:-1])
        with self.assertRaises(ValueError):paired_statistics(data+[data[0]])

    def test_NA_long_event_metric_never_becomes_zero(self):
        summary,paired=paired_statistics(rows())
        metric=next(r for r in summary if r['metric']=='strict_long_steady_macro_coverage')
        self.assertIsNone(metric['mean']);self.assertIsNone(metric['sample_SD']);self.assertEqual(metric['n'],0)

    def test_curve_keeps_continuous_scores_and_ties(self):
        y=np.array([0,1,0,1]);score=np.array([.1,.2,.2,.8])
        curves=curve_arrays(y,score)
        np.testing.assert_array_equal(curves['pr_threshold'],[.1,.2,.8])
        self.assertEqual(curves['FPR'][0],0.);self.assertEqual(curves['TPR'][-1],1.)
        self.assertEqual(len(curves['roc_threshold']),4)

    def test_CAL_marker_uses_strict_greater_not_curve_threshold_tie(self):
        point=cal_operating_point(np.array([0,1,1,0]),np.array([.5,.5,.8,.2]),.5)
        self.assertEqual(point['TP'],1);self.assertEqual(point['FP'],0)
        self.assertEqual(point['recall'],.5);self.assertEqual(point['FPR'],0.)


if __name__=='__main__':unittest.main()
