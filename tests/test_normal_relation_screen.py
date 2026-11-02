"""Synthetic-only correctness fixtures; no SMD/raw/test array access."""
import tempfile
from pathlib import Path
import unittest
from unittest import mock
import numpy as np
from hybrid_tsad import normal_relation_screen as r

def synthetic(n,seed):
    g=np.random.RandomState(20261102+seed);x=g.normal(size=n);y=g.normal(scale=.1,size=n)
    for t in range(5,n):y[t]=.6*x[t-5]+.2*y[t-1]+y[t]
    return np.column_stack([x,y,g.normal(size=n),np.ones(n),2*x+3])

class Screen(unittest.TestCase):
    def test_ridge_mean_objective_and_unpenalized_intercept(self):
        x=np.arange(100,dtype=float)[:,None];y=10+2*x[:,0]
        m=r.ridge_fit(x,y)
        center=x-x.mean(0);expected=np.linalg.solve(center.T@center/100+.001*np.eye(1),center.T@(y-y.mean())/100)
        np.testing.assert_allclose(m['coef'],expected)
        self.assertAlmostEqual(float(r.predict(x,m).mean()),float(y.mean()))

    def test_source_only_selection_and_five_models_common_axis(self):
        a=synthetic(600,3);b=synthetic(300,4)
        first=r.normal_screen(a,b,select_origin=13861)
        other=r.normal_screen(a,b*20+90,select_origin=13861)
        self.assertEqual(first['selected_pair'],other['selected_pair'])
        self.assertEqual(first['selected_pair']['target'],1);self.assertEqual(first['selected_pair']['lag'],5)
        self.assertIn(first['selected_pair']['source'],[0,4])
        np.testing.assert_array_equal(first['scaler']['mean'],a.mean(0))
        self.assertIn(3,first['excluded_low_variance_channels'])
        self.assertTrue(any(x['source']==0 and x['target']==4 for x in first['duplicates']))
        self.assertTrue(all((x['source'],x['target']) not in [(0,4),(4,0)] for x in first['candidates']))
        self.assertEqual(set(first['predictions_z']),{'AR8','AR32','SYNC','LAG','MULTI'})
        np.testing.assert_array_equal(first['selection_t_0based'],np.arange(40,300)+13861)
        self.assertEqual(len(first['metrics']),25);self.assertEqual(sum(x['n'] for x in first['metrics'] if x['model']=='AR8' and x['block']!='ALL'),260)
        expected={a+'-'+b for a in ['AR8','AR32','SYNC','LAG','MULTI'] for b in ['AR32','SYNC'] if a!=b}
        self.assertEqual(len(first['comparisons']),40)
        self.assertEqual({x['comparison'] for x in first['comparisons']},expected)
        for comparison in expected:
            self.assertEqual({x['block'] for x in first['comparisons'] if x['comparison']==comparison},{'ALL','1','2','3','4'})
        self.assertTrue({'MULTI-AR32','LAG-SYNC','MULTI-SYNC'}<=expected)
        self.assertIsNone(first['AP']);self.assertEqual(first['scientific_decision'],'HOLD_NOT_ANOMALY_DETECTION_GO')

    def test_feature_no_current_target_no_future_no_role_cross(self):
        x=np.arange(600,dtype=float).reshape(200,3)
        for f in ['AR8','AR32','SYNC','LAG','MULTI']:
            a,y,t,spec=r.design(x,1,0,5,f)
            self.assertNotIn((1,0),spec)
            self.assertTrue(all(delay>=0 and min(t-delay)>=0 for ch,delay in spec))
            for j,(ch,delay) in enumerate(spec):np.testing.assert_array_equal(a[:,j],x[t-delay,ch])
        with self.assertRaises(PermissionError):r.design(x,1,1,5,'SYNC')

    def test_no_pair_constant_or_affine_only_holds(self):
        self.assertEqual(r.normal_screen(np.ones((100,3)),np.ones((80,3)))['status'],'HOLD')
        x=np.arange(100,dtype=float);a=np.column_stack([x,2*x+8])
        self.assertEqual(r.normal_screen(a,a[:80])['reason'],'NO_ELIGIBLE_FINITE_NON_DUPLICATE_PAIR')

    def test_select_predictions_invariant_to_later_observations(self):
        a=synthetic(300,8);b=synthetic(200,9);c=b.copy();c[150:]+=500
        first=r.normal_screen(a,b);second=r.normal_screen(a,c)
        for key in first['predictions_z']:np.testing.assert_array_equal(first['predictions_z'][key][:110],second['predictions_z'][key][:110])

    def test_numeric_loading_never_parses_guard_cal_or_pool(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'synthetic.txt';p.write_text('\n'.join(['1,2','3,4','NO_NUMERIC_PARSE,GUARD','5,6','7,8','CAL_SENTINEL,POOL_SENTINEL']))
            fit,select=r.load_roles_only(p,shape=(6,2),fit_range=(0,2),select_range=(3,5))
            np.testing.assert_array_equal(fit,[[1,2],[3,4]]);np.testing.assert_array_equal(select,[[5,6],[7,8]])

    def test_pending_blocks_raw_and_numpy_function(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cfg.json';p.write_text('{"binding_state":"PENDING"}')
            with mock.patch.object(r,'load_roles_only',side_effect=AssertionError('No raw')):
                with self.assertRaises(PermissionError):r.execute(p,None,None)

    def test_exact_ties_and_nonfinite_source_rejection(self):
        rows=[dict(abs_corr=.5,target=t,source=s,lag=l) for t,s,l in [(2,1,3),(1,2,3),(1,0,4),(1,0,2)]]
        self.assertEqual([(x['target'],x['source'],x['lag']) for x in r.rank_candidates(rows)],[(1,0,2),(1,0,4),(1,2,3),(2,1,3)])
        x=synthetic(100,2);x[1,0]=np.nan
        with self.assertRaises(ValueError):r.normal_screen(x,synthetic(80,4))

    def test_constant_select_correlation_is_numerically_undefined(self):
        out=r.normal_screen(synthetic(420,7),np.zeros((237,5)))
        self.assertEqual(len(out['SELECT_pair_corr']),5)
        for row in out['SELECT_pair_corr']:
            self.assertIsNone(row['corr']);self.assertEqual(row['direction'],'NA')
            self.assertTrue(row['degenerate'])
            self.assertTrue(row['target_degenerate']);self.assertTrue(row['source_degenerate'])
            self.assertLessEqual(row['target_std'],row['target_std_tolerance'])
            self.assertLessEqual(row['source_std'],row['source_std_tolerance'])
            self.assertGreaterEqual(row['corr_denominator'],0.)
        self.assertIsNone(out['SELECT_corr_std_across_four_blocks'])

    def test_low_noise_select_remains_nondegenerate(self):
        out=r.normal_screen(synthetic(420,7),synthetic(237,8)*1e-6)
        for row in out['SELECT_pair_corr']:
            self.assertFalse(row['degenerate']);self.assertIsNotNone(row['corr'])
            self.assertTrue(np.isfinite(row['corr']))
            self.assertGreater(row['target_std'],row['target_std_tolerance'])
            self.assertGreater(row['source_std'],row['source_std_tolerance'])

if __name__=='__main__':unittest.main(verbosity=2)
