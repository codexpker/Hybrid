import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1.r2_learnability import fixed_plan
from hybrid_tsad.hybrid_v1.data import Run,arrived_window
from hybrid_tsad.hybrid_v1.stage2_u_matched import resolve_u_plan


class Tests(unittest.TestCase):
    def test_actual_cal_resolver_H1_with_full544_eligibility(self):
        roles=[dict(domain='FIXTURE',entity='e',run=role,role=role) for role in ('FIT','SELECT','CAL')]
        p=fixed_plan(roles);r=Run('FIXTURE','e','CAL','CAL',np.zeros((2560,3),np.float32),np.ones((2560,3),bool))
        resolved=resolve_u_plan([r],p['CAL'],512,32,'CAL',arrived=True)
        self.assertEqual(len(resolved),512)
        for run,t in resolved:
            self.assertLessEqual(t+32,2560)
            batch=arrived_window(run,t,512,32)
            self.assertEqual(batch['footprint'],[t-512,t+1])
            self.assertFalse(batch['future_observed'][:,1:].any())
        self.assertNotIn('torch',sys.modules)

    def test_future_values_cannot_affect_arrived_cal(self):
        x=np.zeros((2560,3),np.float32);m=np.ones_like(x,bool)
        a=Run('F','e','r','CAL',x,m);b=Run('F','e','r','CAL',x.copy(),m.copy())
        b.values[1025:]=123
        aa,bb=arrived_window(a,1024,512,32),arrived_window(b,1024,512,32)
        for k in ('history','reconstruction_context','future_targets','future_observed'):
            self.assertTrue(np.array_equal(aa[k],bb[k]))


if __name__=='__main__':unittest.main(verbosity=2)
