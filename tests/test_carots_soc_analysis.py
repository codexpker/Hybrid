"""Synthetic schema/timing checks only; no new scientific query generation."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hybrid_tsad.hybrid_v1.carots_soc_control import fit_representation_calibration, representation_scores, FEATURE_ID
from hybrid_tsad.hybrid_v1.carots_soc_analysis import representation_inputs, phase_diagnostics


class SchemaTests(unittest.TestCase):
    def test_full_axis_score_replay_and_time_permission(self):
        rng=np.random.default_rng(63142)
        cal=fit_representation_calibration(rng.standard_normal((512,4))+.5)
        features=rng.standard_normal((2048,4))+.5
        scores=representation_scores(features,cal)
        rows=[dict(index=int(t),representation=dict(feature=features[j],score=float(scores[j]),
            feature_identity=FEATURE_ID,available_index=int(t),input_bounds=[int(t)-511,int(t)+1],
            forecast_target_unused=int(t)+1,threshold=cal['threshold'],alarm=bool(scores[j]>cal['threshold'])))
            for j,t in enumerate(range(512,2560))]
        actual,raw=representation_inputs(rows,cal)
        self.assertTrue(np.array_equal(raw,features));self.assertTrue(np.array_equal(actual['score'],scores))
        self.assertEqual(actual['index'][-1],2559)
        changed=copy.deepcopy(rows);changed[0]['representation']['input_bounds'][1]+=1
        with self.assertRaises(PermissionError):representation_inputs(changed,cal)
        changed=copy.deepcopy(rows);changed[0]['representation']['score']+=.01
        with self.assertRaises(ValueError):representation_inputs(changed,cal)

    def test_steady_three_confirmation_uses_full_continuous_score_axis(self):
        index=np.arange(512,2560);score=np.zeros(2048)
        score[(index>=1278)&(index<1283)]=2
        array=dict(index=index,score=score,valid=np.ones(2048,dtype=bool))
        rows=phase_diagnostics(array,1.,'CROSS_LAG12_20')
        steady={r['stream']:r for r in rows if r['phase']=='steady'}
        self.assertEqual(steady['strict_point']['alarm_points'],3)
        self.assertEqual(steady['confirmed3_current']['alarm_points'],3)
        self.assertEqual(steady['confirmed3_current']['anomaly_point_recall'],3/256)
        self.assertIsNone(steady['strict_point']['normal_control_FP'])
        normal=phase_diagnostics(array,1.,'HARD_NORMAL')
        self.assertTrue(all(r['anomaly_point_recall'] is None for r in normal))
        self.assertEqual(next(r for r in normal if r['phase']=='steady' and r['stream']=='strict_point')['normal_control_FP'],3)
        null=phase_diagnostics(array,1.,'NO_DRIVER0_COUPLING')
        self.assertTrue(all(r['normal_control_FP'] is None and r['anomaly_point_recall'] is None for r in null))


if __name__=='__main__':unittest.main(verbosity=2)
