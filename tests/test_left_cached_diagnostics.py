"""Hand-computed NumPy diagnostics fixtures; no actual saved experiment inputs."""
import unittest
import numpy as np
from hybrid_tsad.engine import left_cached_diagnostics as d


class CachedDiagnosticsTests(unittest.TestCase):
    def test_perfect_reverse_and_ties_hand_computed(self):
        self.assertEqual(d.endpoint_ap(np.array([1,1,0,0]),np.array([4.,3.,2.,1.])),1.)
        self.assertAlmostEqual(d.endpoint_ap(np.array([0,0,1,1]),np.array([4.,3.,2.,1.])),5/12)
        self.assertAlmostEqual(d.endpoint_ap(np.array([1,0,1,0]),np.array([.8,.8,.2,.1])),7/12)
        self.assertEqual(d.endpoint_ap(np.array([1,0,1,0]),np.ones(4)),.5)
        self.assertAlmostEqual(d.endpoint_ap(np.array([0,1,1,0]),np.array([.8,.8,.2,.1])),7/12)

    def test_na_and_invalid_scores(self):
        self.assertIsNone(d.endpoint_ap(np.array([],int),np.array([],float)))
        self.assertIsNone(d.endpoint_ap(np.zeros(3,int),np.ones(3)))
        self.assertIsNone(d.endpoint_ap(np.ones(3,int),np.ones(3)))
        for y,s in [(np.array([2,0]),np.array([1.,2.])),(np.array([1,0]),np.array([np.nan,2.]))]:
            with self.assertRaises(ValueError):d.endpoint_ap(y,s)

    def test_fixed_channels_and_accumulation_footprint(self):
        raw=np.zeros((128,192,2),np.float32);raw[:,64:,0]=2;raw[:,64:,1]=6
        raw[:,:64,:]=10000
        out=d.channel_readouts(raw)
        for readout in ('last','mean128'):
            self.assertTrue((out[readout]['original_mean']==4).all())
            self.assertTrue((out[readout]['x_only']==2).all())
            self.assertTrue((out[readout]['y_only']==6).all())
        raw[:,-1,1]=10;out=d.channel_readouts(raw)
        self.assertTrue((out['last']['y_only']==10).all())
        self.assertTrue((out['mean128']['y_only']==(127*6+10)/128).all())

    def test_only_exact_cache_input_kinds(self):
        cfg=dict(operation='LEFT_CACHED_CHANNEL_ACCUMULATION_DIAGNOSTICS',scope='EXPLICIT_POST_HOC_EXPLANATION',
            max_seconds=120,max_rss_GiB=1,gpu_budget=0,score_roots={'LIN':'/cached/LIN','WEAK':'/cached/WEAK'},
            evaluation_root='/cached/evaluation')
        cfg['inputs']=[dict(row,sha256='a'*64) for row in d.expected_inputs(cfg)]
        d.validate_config(cfg)
        cfg['inputs'][-1]['path']='/original/run00_truth.json'
        with self.assertRaises(PermissionError):d.validate_config(cfg)


if __name__=='__main__':unittest.main()
