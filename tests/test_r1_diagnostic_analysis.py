import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
from hybrid_tsad.hybrid_v1.r1_diagnostic_analysis import inverse_quantile,deduplicate,availability


class R1AnalysisTests(unittest.TestCase):
    def test_fixed_inverse_CDF_and_empty(self):
        self.assertEqual(inverse_quantile([5,1,4,2,3],.8),4)
        self.assertIsNone(inverse_quantile([],.8))

    def test_aliases_preserved_and_semantic_ambiguity_not_overwritten(self):
        base=dict(model='S',unit='u',label=0,P32_class='ALL_NORMAL',H512_class='ALL_NORMAL',sample_id='a')
        second=dict(base,sample_id='b')
        ambiguous=dict(base,label=1,sample_id='c')
        rows=deduplicate([base,second,ambiguous])
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['logical_alias_count'],2)
        self.assertEqual(rows[0]['logical_aliases'],'a|b')

    def test_offline_task_and_arrived_evidence_clocks_distinct(self):
        for key in ('source_total','student_loss_pred','delta_source_minus_student_pred','D_pred_teacher_scale_MSE'):
            self.assertEqual(availability(key)[0],31)
        for key in ('source_loss_tf','student_current_r_pred','D_T_feature_MSE'):
            self.assertEqual(availability(key)[0],0)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
