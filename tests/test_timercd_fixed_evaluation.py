import unittest
import numpy as np
from hybrid_tsad.timercd_fixed_evaluation import terminal_gate,fixed_common_indices,describe,ARMS


class TimeRCDFixedEvaluation(unittest.TestCase):
    def test_all_bank_seal_required(self):
        records={str(i):{}for i in range(29)}
        banks=[{'relative':f'{r}/{a}/{role}.npz'}for r in records for a in ARMS for role in ('CAL','QUERY')]
        raw=dict(status='COMPLETED',banks=banks,no_labels_decoded=True)
        self.assertEqual(len(terminal_gate(raw,records)),116)
        with self.assertRaises(ValueError):terminal_gate(dict(raw,banks=banks[:-1]),records)
        with self.assertRaises(ValueError):terminal_gate(dict(raw,no_labels_decoded=False),records)

    def test_real_common_axes_not_single_method_axes(self):
        psm=fixed_common_indices(dict(id='PSM',query_points=87841))
        m13=fixed_common_indices(dict(id='machine-1-3',query_points=23703))
        m25=fixed_common_indices(dict(id='machine-2-5',query_points=28727))
        self.assertEqual((psm[0],psm[-1]+1),(512,87744))
        self.assertEqual(m13[-1]+1,23616)
        self.assertEqual(m25[-1]+1,28700)

    def test_summary_missing_is_not_zero_and_sign_is_strict(self):
        methods=ARMS+('reference',);rows=[]
        for rid,a,b,c in [('r1',.3,.2,.3),('r2',.1,.1,None)]:
            rows.extend(dict(record=rid,arm=m,AP=v)for m,v in zip(methods,(a,b,c)))
        result=describe(rows,['r1','r2'],methods,paired=True)
        s=next(r for r in result['paired_summary']if r['candidate']==ARMS[0]and r['reference']=='reference'and r['metric']=='AP')
        self.assertEqual((s['n_valid'],s['expected_entities'],s['wins'],s['ties'],s['losses']),(1,2,0,1,0))
        with self.assertRaises(ValueError):describe(rows[:-1],['r1','r2'],methods)


if __name__=='__main__':unittest.main()
