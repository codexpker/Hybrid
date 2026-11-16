import unittest

import numpy as np

from hybrid_tsad.timeradar_cross_variable_evaluation import cal_threshold, confirm3, evaluate_arrays


class FixedOfflineEvaluation(unittest.TestCase):
    def test_cal_quantile_higher_and_strict_greater(self):
        cal=np.arange(100,dtype=float)
        self.assertEqual(cal_threshold(cal),99.)
        labels=np.array([0,0,1,1,0])
        r=evaluate_arrays(np.arange(5),np.array([99.,100.,99.,100.,0]),labels,99.)
        self.assertEqual(r['strict']['metrics']['TP'],1)
        self.assertEqual(r['strict']['metrics']['FP'],1)
        # Label changes never enter the threshold fitting API.
        self.assertEqual(cal_threshold(cal),99.)

    def test_unknown_partial_event_is_not_full_miss(self):
        y=np.zeros(12,dtype=int);y[3:10]=1
        r=evaluate_arrays(np.arange(7),np.zeros(7),y,1.)
        e,=r['strict']['events']
        self.assertFalse(e['eligible_for_event_metric'])
        self.assertIsNone(e['coverage']);self.assertIsNone(e['longest_miss_points'])
        self.assertIsNone(e['any_hit'])
        self.assertEqual(e['observed_alarm_points'],0)
        self.assertEqual(e['missing_intervals'],[[7,10]])

    def test_discontinuous_indices_never_form_confirmation(self):
        index=np.array([0,1,4,5,6,9])
        got=confirm3(index,np.ones(6,dtype=bool))
        np.testing.assert_array_equal(got,[False,False,False,False,True,False])
        y=np.zeros(11,dtype=int);y[2:8]=1
        r=evaluate_arrays(index,np.ones(6),y,.5)
        self.assertEqual(r['strict']['metrics']['normal_alarm_fragments'],2)
        self.assertFalse(r['strict']['events'][0]['eligible_for_event_metric'])

    def test_anyhit_does_not_equal_one_to_one(self):
        y=np.zeros(20,dtype=int);y[3:6]=1;y[10:13]=1
        r=evaluate_arrays(np.arange(20),np.ones(20),y,.5)
        m=r['strict']['metrics']
        self.assertEqual(m['any_hit_events'],2)
        self.assertEqual(m['matched_events'],1)
        self.assertEqual(m['predicted_alarm_segments'],1)
        self.assertEqual(m['one_to_one_eventRecall'],.5)

    def test_offline_availability_is_not_causal_delay(self):
        y=np.zeros(10,dtype=int);y[3:7]=1
        score=np.zeros(10);score[4:7]=1
        available=np.full(10,9,dtype=int)
        r=evaluate_arrays(np.arange(10),score,y,.5,available_index=available)
        e,=r['strict']['events']
        self.assertEqual(e['offline_first_hit_offset'],1)
        self.assertIsNone(e['online_delay_points'])
        self.assertEqual(e['first_available_alarm_index'],9)
        with self.assertRaises(PermissionError):
            evaluate_arrays(np.arange(10),score,y,.5,available_index=available,online=True)

    def test_long_512_and_current_only_c3(self):
        y=np.zeros(520,dtype=int);y[4:516]=1
        score=y.astype(float)
        r=evaluate_arrays(np.arange(520),score,y,.5)
        self.assertEqual(r['strict']['metrics']['long_events'],1)
        self.assertEqual(r['strict']['metrics']['long_steady_macro_coverage'],1.)
        self.assertAlmostEqual(r['confirmed3']['metrics']['point_Recall'],510/512)
        self.assertEqual(r['confirmed3']['events'][0]['offline_first_hit_offset'],2)

    def test_ranking_only_does_not_make_query_threshold(self):
        r=evaluate_arrays(np.arange(4),np.array([0.,1.,0.,1.]),np.array([0,1,0,1]))
        self.assertEqual(r['ranking']['AP'],1.)
        self.assertIsNone(r['strict']);self.assertIsNone(r['threshold'])


if __name__=='__main__':unittest.main()
