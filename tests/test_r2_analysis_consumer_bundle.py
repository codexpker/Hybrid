import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import tempfile
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1.controlled_mechanism import save_bundle
from hybrid_tsad.hybrid_v1.controlled_wave_recovery import load_bundle
from hybrid_tsad.hybrid_v1.r2_learnability_analysis import route_rows
from hybrid_tsad.hybrid_v1.r2_learnability import ROUTE_ANCHORS

class Tests(unittest.TestCase):
    def test_real_worker_index_and_original004b_bundle_codec(self):
        rows=[]
        for t in ROUTE_ANCHORS:
            for mode in ('forecast','reconstruct_group0','reconstruct_group1'):
                allowed=np.ones((1,3,3,33),bool)
                for i in range(3):allowed[:,i,i,:]=False
                rows.append(dict(case='CROSS_N',index=t,pass_name=mode,physical_source_time='t-1-lag' if mode=='forecast' else 't-lag',meta=dict(
                    correlation=np.zeros((1,3,3,33),np.float32),cross_candidate_valid=allowed,
                    cross=dict(source=np.array([[[1],[0],[0]]]),lag=np.full((1,3,1),12),weights=np.ones((1,3,1),np.float32),selected_valid=np.ones((1,3,1),bool)))))
        with tempfile.TemporaryDirectory() as directory:
            save_bundle(directory,'ROUTE_ANCHORS',rows)
            restored=load_bundle(Path(directory)/'ROUTE_ANCHORS.json',Path(directory)/'ROUTE_ANCHORS.npz')
            selected,grid,near=route_rows(restored,'LAG_U','CROSS_N')
        self.assertEqual(len(selected),135);self.assertEqual(len(near),90)
        self.assertEqual(len(grid),13365)
        self.assertEqual(selected[0]['physical_source_index'],499)
        self.assertEqual(selected[3]['physical_source_index'],500)
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
