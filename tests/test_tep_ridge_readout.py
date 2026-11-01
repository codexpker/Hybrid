import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from hybrid_tsad.tep_ridge_readout import (indexed_design,aligned_readouts,score_run,calibrate,METHODS,FAMILIES,feature_order,COLUMNS,execute,preview)

class RidgeReadout(unittest.TestCase):
    def fixture(self):
        peaks={(t,o):12 for t in range(3) for o in range(3) if t!=o}
        selection=dict(chosen={f:100. for f in FAMILIES},features={})
        weights={}
        for f in FAMILIES:
            for t in range(3):
                spec=feature_order(f,t,peaks);selection['features'][f'{f}_target{t}']=spec
                coef=np.zeros(len(spec),np.float64);coef[0]=1
                weights[f'{f}_a100_target{t}_coef']=coef;weights[f'{f}_a100_target{t}_intercept']=np.asarray(0.,np.float64)
        scaler=dict(mean=[0.,0.,0.],scale=[1.,1.,1.],ddof=0,columns=list(COLUMNS),fit_points=15000,SELECT_used_for_fit=False,
                    fit_ids=[['TEP_REACTOR','FIT',i] for i in list(range(1,21))+list(range(51,61))])
        return selection,weights,scaler

    def test_geometry_exact_original_training_function(self):
        from hybrid_tsad.tep_normal_ridge import design
        from hybrid_tsad.normal_run_data import NormalRun
        z=np.arange(1500,dtype=np.float64).reshape(500,3)
        for target in range(3):
            for f in FAMILIES:
                peaks={(t,o):12 for t in range(3) for o in range(3) if t!=o}
                spec=feature_order(f,target,peaks);x,y,t=indexed_design(z,target,spec)
                ox,oy,ids,ot=design([NormalRun('SYNTHETIC','FIT',1,COLUMNS,z)],target,spec)
                np.testing.assert_array_equal(x,ox);np.testing.assert_array_equal(y,oy);np.testing.assert_array_equal(t,ot)
        with self.assertRaises(PermissionError):indexed_design(z,0,[{'channel':0,'name':'xmeas_9','lag':0}])

    def test_current_other_value_and_lagged_own_availability(self):
        z=np.arange(1500,dtype=np.float64).reshape(500,3);spec=feature_order('SYNC',0)
        x,_,_=indexed_design(z,0,spec)
        self.assertEqual(x[0,0],z[63,0]);self.assertEqual(x[0,63],z[0,0]);self.assertEqual(x[0,64],z[64,1])

    def test_first_mean_uses_t68_through99_not_t64_through95(self):
        v=np.arange(64,500,dtype=np.float64);a=aligned_readouts(v,first_raw_t=64)
        self.assertEqual((a['last'][0],a['mean32'][0]),(99.,83.5))
        self.assertNotEqual(a['mean32'][0],v[:32].mean());self.assertEqual(a['mean32'][-1],np.arange(468,500).mean())

    def test_residuals_three_targets_and_zmax_rule(self):
        s,w,c=self.fixture();z=np.arange(1500,dtype=np.float64).reshape(500,3)
        a=score_run(z,c,s,w,role='CAL',run_id=81)
        np.testing.assert_array_equal(a['SELF_signed_residual_z'],np.full((436,3),-3.))
        self.assertEqual(a['SELF_mean32'][0],3.)
        expected=np.abs(z).max(1);self.assertEqual(a['ZMAX_mean32'][0],expected[68:100].mean())
        with self.assertRaises(PermissionError):score_run(z,c,s,w,role='FIT',run_id=81)
        bad=copy.deepcopy(s);bad['chosen']['SELF']=10.
        with self.assertRaises(PermissionError):score_run(z,c,bad,w,role='CAL',run_id=81)

    def banks(self):
        return [dict(run_id=r,role='CAL',coefficient_sha256='a',scaler_sha256='b',methods={m:dict(method=m,last=np.full(401,5.,np.float64),mean32=np.full(401,2.,np.float64),end_index_0based=np.arange(99,500)) for m in METHODS}) for r in range(81,101)]

    def test_ten_separate8020_pools_rank_and_ties(self):
        dist,c=calibrate(self.banks(),coefficient_sha256='a',scaler_sha256='b')
        self.assertEqual((c['n'],c['rank_1based'],len(c['thresholds'])),(8020,7941,10))
        for m in METHODS:self.assertEqual(np.count_nonzero(dist[m+'_last']>c['thresholds'][m+'_last']),0)
        np.testing.assert_array_equal(dist['end_index_0based'][401:802],np.arange(99,500))
        bad=self.banks();bad[0]['methods']['SELF']['method']='SYNC'
        with self.assertRaises(PermissionError):calibrate(bad,coefficient_sha256='a',scaler_sha256='b')
        bad=self.banks();bad[0]['coefficient_sha256']='other'
        with self.assertRaises(PermissionError):calibrate(bad,coefficient_sha256='a',scaler_sha256='b')
        with self.assertRaises(ValueError):calibrate(self.banks()[:19],coefficient_sha256='a',scaler_sha256='b')
        bad=self.banks();bad[2]['methods']['SELF']['last']=bad[2]['methods']['SELF']['last'][:400]
        with self.assertRaises(ValueError):calibrate(bad,coefficient_sha256='a',scaler_sha256='b')

    def test_pending_denial_before_parameter_or_array_open(self):
        from unittest.mock import patch
        cfg=dict(operation='TEP_RIDGE_NORMAL_CALIBRATION',run_ids=list(range(81,101)),methods=list(METHODS),
            selected_alpha={f:100. for f in FAMILIES},max_seconds=120,max_AS_GiB=4,quantile_rank_1based=7941,
            comparison='strict_greater',output='/home/a312/tsad_research/results/tep_ridge_normal_CAL_20261102_v1',helper_hashes={})
        for k in ('ridge_acceptance','selection','scaler','coefficients','normal_metadata','contract'):
            cfg[k]={'path':'/home/a312/tsad_research/PARAMETER_BODY_MUST_NOT_OPEN.npz','sha256':'a'*64}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cfg.json';r=Path(d)/'release.json';p.write_text(json.dumps(cfg));r.write_text(json.dumps({'status':'PENDING'}))
            with patch('numpy.load',side_effect=AssertionError('No model or CAL body')) as load:
                with self.assertRaises(PermissionError):execute(p,r)
                load.assert_not_called()

if __name__=='__main__':unittest.main()
