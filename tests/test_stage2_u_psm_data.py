import pathlib
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1.stage2_u_psm_data import choose_donors,inject_fixed,select_query,write_role


class DataTests(unittest.TestCase):
    def test_donors_disjoint_and_background_unchanged(self):
        x=np.arange(25000*2,dtype=np.float32).reshape(-1,2);obs=np.ones_like(x,bool)
        plan=[dict(index=512,footprint=[0,544]) for _ in range(200)]
        slots,donors,_=choose_donors(x,obs,plan)
        self.assertEqual(len(set(slots)),20);self.assertEqual(len(donors),20)
        self.assertTrue(np.all(np.diff(donors)>=544));self.assertGreaterEqual(donors[0]-512,544)
        y,injected,affected,std,count=inject_fixed(x,obs,donors)
        self.assertEqual(injected.sum(),640)
        self.assertTrue(np.array_equal(x[:544],y[:544]))
        self.assertTrue(np.array_equal(y[~affected],x[~affected]))
        self.assertTrue(np.allclose(std,np.std(x.astype(np.float64),axis=0,ddof=0)))

    def test_nan_constant_and_physical_copy(self):
        x=np.column_stack([np.arange(1000,dtype=np.float32),np.ones(1000,dtype=np.float32)])
        x[3,0]=np.nan;obs=np.isfinite(x)
        y,_,_,std,_=inject_fixed(x,obs,np.array([600]))
        self.assertTrue(np.isnan(y[3,0]));self.assertEqual(std[1],0.)
        self.assertTrue(np.array_equal(y[:,1],x[:,1]))
        with tempfile.TemporaryDirectory() as tmp:
            descriptor=write_role(pathlib.Path(tmp)/'FIT.npz',x,obs,[0,1000])
            self.assertEqual(descriptor['expected_shape'],[1000,2])

    def test_query_time_rule_keeps_extra_anomaly_labels_and_gaps(self):
        labels=np.zeros(87841,np.int8)
        for s in (600,3000,6000,9000,12000):labels[s:s+10]=1
        labels[17000:18500]=1
        index,report=select_query(labels)
        self.assertEqual(len(report['selected_events']),6)
        self.assertEqual(report['selected_events'][-1]['start'],17000)
        self.assertEqual(report['selected_events'][-1]['points'],1500)
        self.assertTrue(np.any(np.diff(index)>1))
        self.assertEqual(len(report['preset_normal_ranges']),5)
        self.assertTrue(all(labels[s:e].sum()==0 for s,e in report['preset_normal_ranges']))


if __name__=='__main__':unittest.main()
