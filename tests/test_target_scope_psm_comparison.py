"""Only affected target-scope wiring, no production arrays or truth."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1 import target_scope_psm_comparison as target
from hybrid_tsad.hybrid_v1.scoring import apply_calibration


def calibration():
    return dict(valid_windows=512,source_domain_equal_weight=False,domains=['PSM'],
                scales=dict(r_tf=.2,r_ms=.4,r_pred=.1),weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),threshold=2.)


class ScopeWiring(unittest.TestCase):
    def test_pending_never_opens_assets(self):
        with patch.object(target,'read',side_effect=AssertionError('No asset opener before admission')):
            with self.assertRaises(PermissionError):target.admit(dict(status='PENDING_NOT_EXECUTABLE'))

    def test_source_cal_cannot_masquerade_as_target_cal(self):
        source=calibration();source['source_domain_equal_weight']=True
        with self.assertRaises(ValueError):target.target_calibration(source)
        source=calibration();source['valid_windows']=1024
        with self.assertRaises(ValueError):target.target_calibration(source)

    def test_zt_uses_full_source_components_and_target_units(self):
        n=87329;cache=dict(index=np.arange(512,87841),valid=np.ones(n,bool),
            r_tf=np.full(n,.1),r_ms=np.full(n,.4),r_pred=np.full(n,.2))
        old={k:v.copy() for k,v in cache.items()}
        result=target.prepare_zt(cache,calibration())
        expected=apply_calibration(dict(r_tf=.1,r_ms=.4,r_pred=.2),calibration())['score']
        self.assertTrue(np.all(result['score']==expected))
        self.assertEqual(len(result['score']),87329)
        np.testing.assert_array_equal(result['available_index'],cache['index'])
        for k in cache:np.testing.assert_array_equal(old[k],cache[k])

    def test_no_pilot_axis_or_missing_points(self):
        fake=dict(index=np.arange(512,87840),valid=np.ones(87328,bool))
        with self.assertRaises(ValueError):target.prepare_zt(fake,calibration())


if __name__=='__main__':unittest.main(verbosity=2)
