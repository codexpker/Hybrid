import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
from hybrid_tsad.hybrid_v1.r2_learnability import lag_update_evidence


class Tests(unittest.TestCase):
    def test_self_only_never_passes_active_cross_learning_exposure(self):
        t=[{'lag_gradient_norms':{'lag.self_out.weight':1.,'lag.cross_out.weight':None}}]
        e=lag_update_evidence(t,['lag.self_out.weight'],'LAG_U')
        self.assertFalse(e['update_exposure_gate'])

    def test_active_cross_needs_all_groups_and_tracks_each_name(self):
        names=['lag.'+g+'.weight' for g in ('query','key','value','self_out','cross_out')]
        t=[{'lag_gradient_norms':dict.fromkeys(names,1.)}]
        e=lag_update_evidence(t,names,'LAG_U')
        self.assertTrue(e['update_exposure_gate'])
        self.assertEqual(e['per_parameter_nonzero_gradient_steps']['lag.query.weight'],1)
        self.assertFalse(lag_update_evidence(t,names[:-1],'LAG_U')['update_exposure_gate'])

    def test_cross_off_expected_cross_out_none_but_shared_QKV_updates(self):
        names=['lag.'+g+'.weight' for g in ('query','key','value','self_out')]
        grads=dict.fromkeys(names,1.);grads['lag.cross_out.weight']=None
        e=lag_update_evidence([{'lag_gradient_norms':grads}],names,'CROSS_OFF_LAG_U')
        self.assertTrue(e['update_exposure_gate']);self.assertEqual(e['cross_out_status'],'EXPECTED_NO_GRAD_OR_UPDATE')
        self.assertFalse(lag_update_evidence([{'lag_gradient_norms':grads}],names+['lag.cross_out.weight'],'CROSS_OFF_LAG_U')['update_exposure_gate'])
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
