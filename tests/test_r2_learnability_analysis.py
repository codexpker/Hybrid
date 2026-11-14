import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import copy
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1 import r2_learnability_analysis as analysis


class Tests(unittest.TestCase):
    def test_h1_physical_and_archived_normalized_errors_are_distinct(self):
        r=dict(index=512,prediction=[[2.,4.,6.]]*32,per_channel=dict(r_pred=[4.,4.,4.]),replay=dict(current_target=[[0.,0.,0.]]))
        x=analysis.prediction_inputs([r]);rows=analysis.mse_rows(x,'SOURCE','CROSS_N')
        response=next(r for r in rows if r['target_group']=='RESPONSE_MEAN')
        self.assertEqual(response['original_scale_MSE'],26.)
        self.assertEqual(response['normalized_MSE'],4.)
        changed=copy.deepcopy(x);changed['index']=np.array([513])
        with self.assertRaises(ValueError):analysis.paired_mse(x,changed)

    def test_AR_absent_driver_not_filled_as_valid_zero_error(self):
        x=dict(index=np.array([512]),prediction=np.array([[np.nan,1.,2.]]),target=np.array([[0.,0.,0.]]),
            physical_squared_error=np.array([[np.nan,1.,4.]]),normalized_squared_error=np.array([[np.nan,2.,3.]]),valid=np.array([[False,True,True]]))
        rows=analysis.mse_rows(x,'ARX12','CROSS_N');driver=next(r for r in rows if r['target_group']=='DRIVER_0')
        self.assertEqual(driver['valid_channel_points'],0);self.assertIsNone(driver['original_scale_MSE'])

    def test_route_physical_time_and_inactive_cross_are_explicit(self):
        entries=[]
        for t in analysis.experiment.ROUTE_ANCHORS:
            for mode in ('forecast','reconstruct_group0','reconstruct_group1'):
                selected=dict(source=[[[1],[0],[0]]],lag=[[[12],[12],[12]]],weights=[[[1.],[1.],[1.]]],selected_valid=[[[True],[True],[True]]])
                candidate=np.ones((1,3,3,33),bool)
                for i in range(3):candidate[:,i,i,:]=False
                entries.append(dict(point=t,pass_name=mode,meta=dict(correlation=np.zeros((1,3,3,33)),cross_candidate_valid=candidate,cross=selected)))
        s,g,n=analysis.route_rows(entries,'CROSS_OFF_LAG_U','CROSS_N')
        a=next(r for r in s if r['point']==512 and r['pass_name']=='forecast' and r['target']==1)
        b=next(r for r in s if r['point']==512 and r['pass_name']=='reconstruct_group0' and r['target']==1)
        self.assertEqual(a['physical_source_index'],499);self.assertEqual(b['physical_source_index'],500)
        self.assertFalse(a['cross_output_active']);self.assertEqual(len(g),15*3*3*3*33)
        self.assertTrue(all(r['weight_at_historical_lag12']==1. for r in n))
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':unittest.main(verbosity=2)
