import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1.data import Run
from hybrid_tsad.hybrid_v1.baselines import baseline_record
from hybrid_tsad.roca_cpu_controls import standardized_run,fixed_endpoint_score,fixed_calibration

class ControlsTests(unittest.TestCase):
    def test_fixed_footprint_history15_and_future_invariance(self):
        x=np.arange(70,dtype=np.float32)[:,None]
        r=Run('SMAP','fixture','run','QUERY',x,np.ones_like(x,bool),100)
        z=standardized_run(r,[4],[3])
        for method in ('SHORT_HISTORY_MEAN','SHORT_PERSISTENCE'):
            a=fixed_endpoint_score(z,130,16,method)
            old=baseline_record(z,30,method='history_mean512' if method=='SHORT_HISTORY_MEAN' else 'persistence',history=15)
            self.assertEqual(a['score'],old['r_pred']);self.assertEqual(a['actual_input_bounds'],[115,131])
            changed=x.copy();changed[31:]=1e6
            zz=standardized_run(Run('SMAP','fixture','run','QUERY',changed,np.ones_like(x,bool),100),[4],[3])
            self.assertEqual(a['score'],fixed_endpoint_score(zz,130,16,method)['score'])
        with self.assertRaises(PermissionError):fixed_endpoint_score(z,114,16,'SHORT_HISTORY_MEAN')
    def test_only_current_static_distance(self):
        x=np.array([[3.,4.]]*32,np.float32);r=Run('SMD','f','r','QUERY',x,np.ones_like(x,bool))
        z=standardized_run(r,[1,2],[2,1]);a=fixed_endpoint_score(z,31,32,'STATIC_FIT_DISTANCE')
        self.assertEqual(a['score'],2.5);self.assertEqual(a['actual_input_bounds'],[31,32])
    def test_CAL_only_inverse_CDF(self):
        rows=[dict(role='CAL',score=float(i),valid=True) for i in range(32)]
        self.assertEqual(fixed_calibration(rows)['threshold'],31.)
        rows[0]['role']='QUERY'
        with self.assertRaises(PermissionError):fixed_calibration(rows)

if __name__=='__main__':unittest.main(verbosity=2)
