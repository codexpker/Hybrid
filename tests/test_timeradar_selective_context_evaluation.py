import unittest
from hybrid_tsad.timeradar_selective_context_evaluation import RECORDS,NEW_ARMS,fixed_keys,check_complete_seal,paired_rows,check_parameter_match


class SelectiveFixedEvaluation(unittest.TestCase):
    def test_exact_16_new_banks_required(self):
        seal=dict(status='COMPLETED',banks=[dict(relative=k)for k in fixed_keys()])
        check_complete_seal(seal)
        seal['banks'].pop()
        with self.assertRaises(ValueError):check_complete_seal(seal)

    def test_pairing_is_per_entity_and_rejects_different_axes(self):
        methods=('selective','capacity_self','cross','self','decoder_head','zero')
        rows=[dict(record=r,arm=a,scored_points=1000,AP=.4 if a=='selective'else .3,
                   ROC_AUC=.6,strict_long_steady_macro_coverage=None)for r in RECORDS for a in methods]
        pairs=paired_rows(rows,scope='native_full')
        row=next(r for r in pairs if r['record']=='PSM'and r['comparison']=='selective-minus-capacity_self'and r['metric']=='AP')
        self.assertAlmostEqual(row['delta'],.1)
        na=next(r for r in pairs if r['metric']=='strict_long_steady_macro_coverage')
        self.assertIsNone(na['delta'])
        rows[1]['scored_points']=999
        with self.assertRaises(ValueError):paired_rows(rows,scope='native_full')

    def test_exact_parameter_control(self):
        check_parameter_match(dict(trainable_parameters=1512),dict(trainable_parameters=1512),dict(trainable_parameters=1000))
        with self.assertRaises(ValueError):
            check_parameter_match(dict(trainable_parameters=1512),dict(trainable_parameters=1511),dict(trainable_parameters=1000))


if __name__=='__main__':unittest.main()
