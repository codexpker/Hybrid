import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
P=Path(__file__).resolve().parents[1];spec=importlib.util.spec_from_file_location('consumer',P/'scripts/evaluate_roca_public.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ConsumerTests(unittest.TestCase):
    def test_QUERY_CAL_rejected_and_loss_n_not_R3(self):
        d=dict(method='RoCA_FULL_T',domain='SMD',score_semantics='official_loss_n',calibration_role='CAL',format='scalar_npz')
        plan=[dict(role='CAL',domain='SMD') for _ in range(512)]
        m.validate_method(d,dict(threshold=1.,score='official_loss_n'),plan)
        plan[0]['role']='QUERY'
        with self.assertRaises(PermissionError):m.validate_method(d,dict(threshold=1.,score='official_loss_n'),plan)
        plan[0]['role']='CAL';d['score_semantics']='R3_own_target_CAL_scales_and_tau'
        with self.assertRaises(PermissionError):m.validate_method(d,dict(threshold=1.,score='official_loss_n'),plan)
    def test_native_axis_and_complete_tail(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'scores.npz';ix=np.arange(8958,8970,dtype=np.int64);score=np.arange(12,dtype=float)
            np.savez_compressed(p,index=ix,available_index=ix,score=score,valid=np.ones(12,bool),alarm=score>3)
            d=dict(score_array=dict(path=str(p),sha256=m.sha(p)))
            a,_=m.read_scalar_bank(d,dict(threshold=3.),8958,8970)
            self.assertEqual(a['index'][-1],8969)
            with self.assertRaises(ValueError):m.read_scalar_bank(d,dict(threshold=3.),512,524)
    def test_unknown_prefix_and_runend_censor_original_kernel(self):
        y=np.zeros(8970,np.int8);y[30:32]=1;y[8968:8970]=1;ix=np.arange(8958,8970,dtype=np.int64)
        r=m.evaluation.evaluate_arrays(ix,np.ones(12)*2,y,1.,bank_complete=True,original_length=len(y),warmup_points=8958)
        self.assertEqual(r['coverage']['original_events'],2)
        self.assertEqual(r['coverage']['main_eligible_events'],0)
        self.assertIsNone(r['point_coverage'][30]['strict_point'])
        self.assertTrue(r['event_coverage'][1]['complete']);self.assertTrue(r['event_coverage'][1]['boundary_censored'])

if __name__=='__main__':unittest.main(verbosity=2)
