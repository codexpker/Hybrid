import copy
import unittest
import numpy as np
from hybrid_tsad.tep_normal_control_evaluation import evaluate_normal,wilson20,summarize_normal_rows,METHODS,READOUTS,SEEDS,RUNS

ENDS=np.arange(99,960,dtype=np.int64)
def score(*segments):
    v=np.zeros(861)
    for lo,hi in segments:v[lo-100:hi-99]=2.
    return v

class NormalEndpoints(unittest.TestCase):
    def test_all_high_is_carry_in_not_no_risk(self):
        r=evaluate_normal(score((100,960)),ENDS,1)
        m=r['metrics']
        self.assertEqual(m['new_segment_false_hit_40h'],0)
        self.assertEqual(m['carry_in_confirmed_state'],1)
        self.assertEqual(m['any_confirmed_state_40h'],1)
        self.assertEqual(m['confirmation_count_in_40h_window'],0)
        self.assertEqual(m['prefix_confirmed_segment_count_61points'],1)
        self.assertTrue(r['first_new_confirmation_administratively_right_censored'])
        self.assertIsNone(r['first_new_confirmation_delay_minutes'])

    def test_start160_versus161_no_reset(self):
        boundary=evaluate_normal(score((160,162)),ENDS,1)
        new=evaluate_normal(score((161,163)),ENDS,1)
        self.assertEqual(boundary['metrics']['new_segment_false_hit_40h'],0)
        self.assertEqual(boundary['metrics']['boundary_confirmed_after_entry_count'],1)
        self.assertEqual(boundary['metrics']['confirmation_count_in_40h_window'],1)
        self.assertEqual(boundary['metrics']['any_confirmed_state_40h'],1)
        self.assertEqual(new['metrics']['new_segment_false_hit_40h'],1)
        self.assertEqual(new['first_new_confirmation_delay_minutes'],6)

    def test_exact181_182_and_960_confirmations(self):
        for lo,timely,delay in [(179,1,60),(180,0,63),(958,0,2397)]:
            r=evaluate_normal(score((lo,lo+2)),ENDS,1)
            self.assertEqual(r['metrics']['new_segment_false_hit_60min'],timely)
            self.assertEqual(r['metrics']['new_segment_false_hit_40h'],1)
            self.assertEqual(r['first_new_confirmation_delay_minutes'],delay)

    def test_no_alert_two_point_segment_and_equality_reset(self):
        for a in [score(),score((161,162))]:
            r=evaluate_normal(a,ENDS,1)
            self.assertEqual(r['metrics']['any_confirmed_state_40h'],0)
            self.assertIsNone(r['first_eligible_new_confirmation_sample'])
            self.assertIsNone(r['metrics']['AP']);self.assertIsNone(r['metrics']['ROC_AUC'])
            self.assertEqual(r['censor_delay_minutes'],2397)
        a=score((161,164));a[163-100]=1.
        r=evaluate_normal(a,ENDS,1)
        self.assertEqual(r['metrics']['new_confirmed_segment_count_40h'],0)

    def test_repeated_segments_counts_rates_and_prefix(self):
        r=evaluate_normal(score((110,112),(161,165),(200,210)),ENDS,1)
        m=r['metrics']
        self.assertEqual(m['new_confirmed_segment_count_40h'],2)
        self.assertEqual(m['new_segment_false_hit_40h'],1)
        self.assertEqual(m['confirmation_count_in_40h_window'],2)
        self.assertEqual(m['new_segment_count_per_hour_40h'],.05)
        self.assertAlmostEqual(m['new_segment_count_per_hour_39_95h'],2/39.95)
        self.assertEqual(m['prefix_confirmed_segment_count_61points'],1)
        self.assertAlmostEqual(m['point_false_alarm_fraction_40h'],16/800)
        self.assertEqual(r['first_eligible_new_confirmation_sample'],163)

    def test_missing_axis_nonfinite_and_nonfixed_threshold_rejected(self):
        for a,ends,t in [(score()[:-1],ENDS,1),(score(),ENDS[::-1],1),(score(),ENDS.astype(float),1),
                         (np.full(861,np.nan),ENDS,1),(score(),ENDS,np.inf),(score(),ENDS,True)]:
            with self.assertRaises(ValueError):evaluate_normal(a,ends,t)

    def test_wilson20_zero_is_not_zero_risk(self):
        a=wilson20(0)
        self.assertAlmostEqual(a['CI95'][0],0)
        self.assertGreater(a['CI95'][1],.16)
        self.assertEqual(wilson20(20)['CI95'][1],1)
        for x in [True,0.,21,-1]:
            with self.assertRaises(ValueError):wilson20(x)

    def fixture320(self):
        rows=[]
        for m in METHODS:
            for d in READOUTS:
                for seed in (SEEDS if m=='LEFT' else (None,)):
                    for run in RUNS:
                        a=score((161,163)) if m=='LEFT' and seed==2025 else score()
                        rows.append(dict(method=m,readout=d,seed=seed,run=run,result=evaluate_normal(a,ENDS,1)))
        return rows

    def test320_rows_six_methods_two_readouts_no_seed_pseudoreplication(self):
        r=summarize_normal_rows(self.fixture320())
        self.assertEqual(r['result_rows'],320);self.assertEqual(len(r['streams']),16)
        self.assertEqual(r['independent_normal_runs'],20)
        self.assertEqual(len(r['LEFT_per_run_seed_averages']),40)
        self.assertEqual(r['LEFT_per_run_seed_averages'][0]['metrics']['new_segment_false_hit_40h'],1/3)
        self.assertIsNone(r['LEFT_per_run_seed_averages'][0]['Wilson_CI'])

    def test_missing_duplicate_unknown_identity_refused(self):
        rows=self.fixture320()
        for bad in [rows[:-1],rows+[rows[0]],copy.deepcopy(rows)]:
            if len(bad)==320:bad[0]['seed']=None
            with self.assertRaises(ValueError):summarize_normal_rows(bad)

if __name__=='__main__':unittest.main(verbosity=2)
