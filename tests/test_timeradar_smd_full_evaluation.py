import unittest
import numpy as np
from hybrid_tsad.timeradar_smd_full_evaluation import summaries,common_indices


class FixedFullSMDStatistics(unittest.TestCase):
    def test_missing_row_kept_and_no_zero_imputation(self):
        entities=['machine-1-1','machine-1-4']
        rows=[dict(record=e,arm=a,status='OK',AP=.5,ROC_AUC=.6)for e in entities for a in ('self','cross')]
        rows[-1].update(status='MISSING_OR_FAILED',AP=None,ROC_AUC=None)
        out=summaries(rows,entities,('self','cross'))
        self.assertEqual(out['status'],'INCOMPLETE')
        row=next(r for r in out['macro']if r['group']=='all28'and r['method']=='cross'and r['metric']=='AP')
        self.assertEqual(row['expected_entities'],2);self.assertEqual(row['n_valid'],1);self.assertEqual(row['macro_mean'],.5)

    def test_strict_sign_no_new_tolerance_and_axis_tables_independent(self):
        entities=['machine-1-1','machine-1-4','machine-1-5']
        rows=[]
        for e,delta in zip(entities,(1e-12,0.,-.1)):
            rows.extend([dict(record=e,arm='self',status='OK',AP=.5,ROC_AUC=.6),
                         dict(record=e,arm='cross',status='OK',AP=.5+delta,ROC_AUC=.6)])
        out=summaries(rows,entities,('self','cross'))
        pair=next(r for r in out['paired_summary']if r['group']=='all28'and r['metric']=='AP')
        self.assertEqual((pair['wins'],pair['tie'],pair['losses']),(1,1,1))
        self.assertEqual(pair['paired_n'],3)

    def test_dropping_fixed_entity_is_rejected(self):
        with self.assertRaises(ValueError):summaries([],['machine-1-1'],('self','cross'))

    def test_common_axis_uses_indices_not_labels_and_keeps_gaps(self):
        a=np.array([510,511,512,513,514,516,517]);b=np.array([512,513,515,516])
        common=common_indices([(a,np.zeros(len(a)),a),(b,np.zeros(len(b)),b)])
        np.testing.assert_array_equal(common,[512,513,516])


if __name__=='__main__':unittest.main()
