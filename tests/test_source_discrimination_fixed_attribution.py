"""Only the added fixed-cal attribution path; no production inputs or metrics."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1.source_discrimination_psm_comparison import fixed_old_calibration_scores
from hybrid_tsad.hybrid_v1.scoring import apply_calibration


class FixedAttribution(unittest.TestCase):
    def test_original_scalar_formula_and_inputs_unchanged(self):
        cal=dict(scales=dict(r_tf=.2,r_ms=.4,r_pred=.1),weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),threshold=2.)
        raw=dict(r_tf=np.array([.1,.2,.3]),r_ms=np.array([.4,.1,.8]),r_pred=np.array([.2,.01,.4]))
        before={key:value.copy() for key,value in raw.items()};oldcal=copy.deepcopy(cal)
        expected=np.array([apply_calibration({key:value[i] for key,value in raw.items()},cal)['score'] for i in range(3)])
        actual=fixed_old_calibration_scores(raw,cal)
        np.testing.assert_array_equal(actual,expected)
        for key in raw:np.testing.assert_array_equal(raw[key],before[key])
        self.assertEqual(cal,oldcal)

    def test_no_invalid_or_partial_component_intersection(self):
        cal=dict(scales=dict(r_tf=1.,r_ms=1.,r_pred=1.),weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),threshold=2.)
        raw=dict(r_tf=np.array([1.,np.nan]),r_ms=np.ones(2),r_pred=np.ones(2))
        with self.assertRaises(ValueError):fixed_old_calibration_scores(raw,cal)
        with self.assertRaises(ValueError):fixed_old_calibration_scores(dict(r_tf=np.ones(2),r_ms=np.ones(2)),cal)


if __name__=='__main__':unittest.main(verbosity=2)
