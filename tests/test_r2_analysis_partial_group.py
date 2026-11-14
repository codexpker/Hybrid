import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1.r2_learnability_analysis import mse_rows

class Tests(unittest.TestCase):
    def test_baseline_two_targets_cannot_masquerade_as_allC(self):
        e=np.array([[np.nan,1.,4.]])
        x=dict(index=np.array([512]),physical_squared_error=e,normalized_squared_error=e,valid=np.isfinite(e))
        rows={r['target_group']:r for r in mse_rows(x,'ARX12','CROSS_N')}
        self.assertIsNone(rows['ALL_C']['original_scale_MSE'])
        self.assertEqual(rows['RESPONSE_MEAN']['original_scale_MSE'],2.5)
        self.assertEqual(rows['RESPONSE_MEAN']['normalization_scope'],'FIT_target_std_per_response')
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__':unittest.main(verbosity=2)
