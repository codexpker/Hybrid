"""Small synthetic fixtures only; never load existing normal arrays."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sklearn.linear_model import Ridge
from hybrid_tsad.normal_run_data import NormalRun
from hybrid_tsad import tep_normal_ridge as r


class RidgeFixtures(unittest.TestCase):
    def fixture(self,role='FIT'):
        rng=np.random.default_rng(591)
        return [NormalRun('CPU_SYNTHETIC',role,i,r.COLUMNS,np.asfortranarray(rng.normal(size=(90,3)))) for i in [1,2]]

    def test_orders_and_current_target_mask(self):
        peaks={(i,j):7 for i in range(3) for j in range(3) if i!=j}
        for family,size in [('SELF',64),('SYNC',66),('PEAK',66),('MULTI',194)]:
            for target in range(3):
                spec=r.feature_order(family,target,peaks);self.assertEqual(len(spec),size)
                self.assertFalse(any(f['channel']==target and f['lag']==0 for f in spec))
                self.assertTrue(all(0<=f['lag']<=64 for f in spec))
        with self.assertRaises(PermissionError):r.design(self.fixture(),0,[dict(channel=0,name=r.COLUMNS[0],lag=0)])
        with self.assertRaises(PermissionError):r.design(self.fixture(),0,[dict(channel=1,name=r.COLUMNS[1],lag=-1)])

    def test_run_boundaries_lag_direction_and_target_perturbation(self):
        runs=self.fixture();spec=r.feature_order('MULTI',0)
        X,y,ids,t=r.design(runs,0,spec)
        self.assertEqual(X.shape,(52,194));np.testing.assert_array_equal(t,np.r_[np.arange(64,90),np.arange(64,90)])
        self.assertEqual((ids[25],ids[26],t[25],t[26]),(1,2,89,64))
        for j,f in enumerate(spec):self.assertEqual(X[26,j],runs[1].values[64-f['lag'],f['channel']])
        changed=copy.deepcopy(runs);changed[0].values[64,0]+=10000
        X2,y2,_,_=r.design(changed,0,spec)
        np.testing.assert_array_equal(X[0],X2[0]);self.assertNotEqual(y[0],y2[0])
        changed=copy.deepcopy(runs);changed[0].values[65:]+=10000
        X3,_,_,_=r.design(changed,0,spec);np.testing.assert_array_equal(X[0],X3[0])

    def test_exact_alpha_and_peak_ties(self):
        self.assertEqual(r.choose_alpha({a:1. for a in r.ALPHAS}),100.)
        self.assertEqual(r.choose_alpha({.1:2.,1.:1.,10.:1.,100.:2.}),10.)
        self.assertEqual(r.choose_peak({lag:0. for lag in range(1,65)}),1)
        c={lag:0. for lag in range(1,65)};c[5]=-.7;c[8]=.7;self.assertEqual(r.choose_peak(c),5)

    def test_peak_only_fit_and_known_positive_past_direction(self):
        rng=np.random.default_rng(141);x=rng.normal(size=(180,3));x[9:,0]=x[:-9,1]
        fit=[NormalRun('CPU_SYNTHETIC','FIT',1,r.COLUMNS,x)]
        peaks,profile=r.peak_profile(fit);self.assertEqual(peaks[0,1],9);self.assertEqual(len(profile),384)
        with self.assertRaises(PermissionError):r.peak_profile(self.fixture('SELECT'))

    def test_constant_feature_has_undefined_Pearson_and_no_fake_lag1(self):
        fit=self.fixture()
        for run in fit:run.values[:,1]=3.
        with self.assertRaisesRegex(ValueError,'Undefined FIT Pearson'):
            r.peak_profile(fit)

    def test_global_variation_does_not_define_constant_legal_target_grid(self):
        fit=self.fixture()
        for run in fit:
            run.values[64:,0]=1.
            self.assertGreater(run.values[:,0].var(),0.)
        self.assertEqual(np.concatenate([run.values[64:,0] for run in fit]).var(),0.)
        with self.assertRaisesRegex(ValueError,'Undefined FIT Pearson'):
            r.peak_profile(fit)

    def test_mature_ridge_coefficients_predictions_selection(self):
        fit=self.fixture();select=self.fixture('SELECT');result=r.fit_select(fit,select)
        self.assertEqual(len(result['models']),48);self.assertEqual(len(result['selection']),16)
        for family in r.FAMILIES:
            losses={x['alpha']:x['SELECT_MSE'] for x in result['selection'] if x['family']==family}
            self.assertEqual(result['chosen'][family],r.choose_alpha(losses))
            spec=result['features'][family+'_target0'];X,y,_,_=r.design(fit,0,spec)
            expected=Ridge(alpha=1.,fit_intercept=True,solver='cholesky').fit(X,y)
            saved=result['models'][family+'_a1_target0']
            np.testing.assert_array_equal(saved['coef'],expected.coef_);self.assertEqual(saved['intercept'],expected.intercept_)
            self.assertEqual(result['predictions'][family+'_a1_SELECT'].shape,(52,3))

    def test_pending_refusal_cannot_read_arrays_or_fit(self):
        cfg={'adapter_hashes':{}}
        with patch.object(r,'preview',return_value=(cfg,{})),patch.object(r,'read_json',return_value={'status':'PENDING'}), \
             patch.object(r,'sha',return_value='fixture'),patch.object(np,'load',side_effect=AssertionError('Forbidden actual body')) as load, \
             patch.object(r,'fit_select',side_effect=AssertionError('Forbidden formal fit')) as fit:
            with self.assertRaises(PermissionError):r.execute('fake','pending')
            load.assert_not_called();fit.assert_not_called()

    def test_normal_metadata_and_scaler_provenance_rejections(self):
        from hybrid_tsad.left_normal_public import validate_normal_metadata
        source=json.loads(Path(os.environ['NORMAL_METADATA_PATH']).read_text())
        validate_normal_metadata(source)
        for key in ['role','columns','fault']:
            altered=copy.deepcopy(source)
            if key=='role':altered['roles']['CAL']=[]
            if key=='columns':altered['columns']=list(reversed(source['columns']))
            if key=='fault':altered['roles']['FIT'][0]['faultNumber']=1
            with self.assertRaises(PermissionError):validate_normal_metadata(altered)
        fit=self.fixture();scaler=dict(mean=[1.,2.,3.],scale=[2.,3.,4.],columns=list(r.COLUMNS),ddof=0,
            fit_ids=[list(v.identity) for v in fit])
        out=r.normalize_runs(fit,scaler)
        np.testing.assert_array_equal(out[0].values,(fit[0].values-[1.,2.,3.])/[2.,3.,4.])
        bad=copy.deepcopy(scaler);bad['fit_ids'][0][1]='SELECT'
        with self.assertRaises(PermissionError):r.normalize_runs(fit,bad)


if __name__=='__main__':unittest.main()
