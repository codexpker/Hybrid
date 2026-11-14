import pathlib
import sys
import unittest
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1.stage2_u_psm_input_data import inject_input_and_target


class FinalInputTests(unittest.TestCase):
    def test_exact160_history_current_future_and_other_windows(self):
        x=np.arange(3000*2,dtype=np.float32).reshape(-1,2);m=np.ones_like(x,bool)
        y,mask,affected,sigma,count=inject_input_and_target(x,m,[1000,2000])
        self.assertEqual(mask.sum(),320)
        for t in (1000,2000):
            self.assertEqual(mask[t-512:t].sum(),128)
            self.assertEqual(mask[t-511:t+1].sum(),129)
            self.assertEqual(mask[t:t+32].sum(),32)
        self.assertTrue(np.array_equal(y[~affected],x[~affected]))
        self.assertTrue(np.array_equal(y[:500],x[:500]))
        expected=(x[1000].astype(np.float64)+2*sigma).astype(np.float32)
        self.assertTrue(np.array_equal(y[1000],expected))

    def test_nan_mask_constant_channel_and_illegal_footprint(self):
        x=np.column_stack([np.arange(1000,dtype=np.float32),np.ones(1000,dtype=np.float32)])
        x[590,0]=np.nan;m=np.isfinite(x)
        y,mask,_,sigma,_=inject_input_and_target(x,m,[600])
        self.assertTrue(np.isnan(y[590,0]));self.assertEqual(sigma[1],0.)
        self.assertTrue(np.array_equal(x[:,1],y[:,1]))
        self.assertEqual(mask.sum(),160)
        with self.assertRaises(ValueError):inject_input_and_target(x,m,[100])


if __name__=='__main__':unittest.main()
