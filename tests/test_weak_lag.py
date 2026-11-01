"""Small fixtures only. Never create the formal role pools or read real data."""
import unittest
import numpy as np
from sklearn.linear_model import Ridge
from hybrid_tsad import weak_lag as wl


class WeakLagTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(211)
        self.scaler = {'mean': np.array([.2, -.1]), 'scale': np.array([1.1, .8])}

    def model(self, name):
        p = len(wl.feature_indices(name, 0, [191], peak=12)[1])
        return {'model': name, 'peaks': [12, 20], 'directions': [
            {'coef': self.rng.normal(size=p)/p, 'intercept': .03} for _ in range(2)]}

    def test_seed_roles_stationary_sham_and_IRR_pairing(self):
        a, irr, truth = wl.generate_run('WEAK', 'DEV', 901, n=1920, h=16)
        b, _, same = wl.generate_run('WEAK', 'DEV', 901, n=1920, h=16)
        self.assertTrue(np.array_equal(a['A'], b['A']))
        self.assertTrue(np.array_equal(a['S0'], a['N']))
        self.assertTrue(np.array_equal(irr['A'][:, 1], a['A'][:, 1]))
        self.assertFalse(np.array_equal(irr['A'][:, 0], a['A'][:, 0]))
        self.assertEqual(int(truth['labels']['A'].sum()), 12*16)
        self.assertEqual(int(truth['labels']['SB'].sum()), 2*16)
        self.assertEqual(truth['seeds'], same['seeds'])
        self.assertEqual(len(set(wl.derive_seed('WEAK', r, 901, 'u') for r in wl.ROLES)), 4)
        fit, _, _ = wl.generate_run('WEAK', 'FIT', 901, n=1920, h=16)
        select, _, _ = wl.generate_run('WEAK', 'SELECT', 901, n=1920, h=16)
        self.assertFalse(np.array_equal(fit['N'], select['N']))
        self.assertEqual(set(fit), {'N'})
        with self.assertRaises(ValueError):
            wl.generate_run('WEAK', 'CONFIRM', 0, n=1920, h=16)

    def test_no_future_or_target_current_feature_and_complete_window_footprints(self):
        window = self.rng.normal(size=(192, 2))
        for name in wl.MODELS:
            for target in (0, 1):
                t = np.arange(64, 192) if name.endswith('64') else np.array([191])
                ix, channel = wl.feature_indices(name, target, t, peak=12)
                self.assertGreaterEqual(ix.min(), 0)
                self.assertTrue(np.all(ix <= t[:, None]))
                self.assertFalse(np.any((ix == t[:, None]) & (channel == target)))
                x, _ = wl.design(window, name, target, t, peak=12)
                modified = window.copy(); modified[191, target] += 1e6
                xx, _ = wl.design(modified, name, target, t, peak=12)
                self.assertTrue(np.array_equal(x, xx))
                p = self.rng.normal(size=x.shape[1])
                self.assertTrue(np.array_equal(x@p+.1, xx@p+.1))
            model = self.model(name)
            future = self.rng.normal(size=(576, 2)); future[:192] = window
            before = wl.score_window(future[:192], self.scaler, model)
            future[192:] += 1e9
            after = wl.score_window(future[:192], self.scaler, model)
            self.assertTrue(np.array_equal(before['prediction'], after['prediction']))
            if name.endswith('64'):
                self.assertEqual(before['residual'].shape, (128, 2))
                self.assertEqual(before['last'], before['residual'][-1].mean())
                self.assertEqual(before['mean128'], before['residual'].mean())

    def test_Gram_eigh_exact_objective_vs_sklearn_unpenalized_intercept(self):
        x = self.rng.normal(size=(91, 13))+7
        x[:, 12] = x[:, 0]+x[:, 1]  # rank deficient is legal with positive alpha
        y = 17+x@self.rng.normal(size=13)+self.rng.normal(size=91)*.2
        result = wl.ridge_candidates(x, y)
        for alpha, params in result.items():
            ref = Ridge(alpha=alpha, fit_intercept=True, solver='svd').fit(x, y)
            np.testing.assert_allclose(params['coef'], ref.coef_, rtol=2e-9, atol=2e-9)
            np.testing.assert_allclose(params['intercept'], ref.intercept_, rtol=2e-9, atol=2e-9)
            residual = x@params['coef']+params['intercept']-y
            np.testing.assert_allclose(x.T@residual+alpha*params['coef'], 0, atol=2e-8)
            self.assertAlmostEqual(float(residual.sum()), 0., places=9)
        self.assertEqual(wl.choose_alpha({a: 1. for a in wl.ALPHAS}), 100.)

    def test_normal_role_boundaries_select_and_shared_alpha(self):
        fit = [wl.NormalRun('UNIT', 'FIT', 901, self.rng.normal(size=(384, 2)))]
        select = [wl.NormalRun('UNIT', 'SELECT', 901, self.rng.normal(size=(384, 2)))]
        scaler = wl.fit_scaler(fit)
        model = wl.fit_model(fit, select, scaler, 'SYNC64')
        self.assertEqual(model['fit_targets'], 193)  # not 320 for short-order fit
        self.assertEqual(model['select_targets'], 2)
        self.assertEqual(len(model['select_mse']), 5)
        self.assertEqual(len(model['directions']), 2)
        for forbidden in ('CAL', 'DEV'):
            with self.assertRaises(ValueError):
                wl.fit_model(fit, [wl.NormalRun('UNIT', forbidden, 0, select[0].values)], scaler, 'AR191')
        with self.assertRaises(ValueError):
            wl.fit_scaler(select)
        with self.assertRaises(ValueError):
            wl.fit_scaler([fit[0], fit[0]])

    def test_CORE_raw_design_target_prediction_residual_SB_equality_all_models(self):
        n, s, e = 7680, 2304, 5376
        normal = self.rng.normal(size=(n, 2)); sb = normal.copy()
        sb[s:s+192, 1] += 4; sb[e-192:e, 1] -= 4
        tau = np.full(n, 12); tau[6500:] = 20
        models = [self.model(name) for name in wl.MODELS]
        count = wl.assert_sham_equivalence(normal, sb, normal.copy(), tau, s, e, self.scaler, models)
        self.assertGreater(count, 0)
        layer = wl.strata(tau, s, e)
        self.assertTrue(np.any(layer['CORE'] & (layer['endpoint'] >= s) & (layer['endpoint'] < e)))
        self.assertFalse(np.any(layer['CORE'] & layer['NORMAL_SWITCH']))
        broken = sb.copy(); k = np.flatnonzero(layer['CORE'])[0]; broken[k*192, 0] += .1
        with self.assertRaises(AssertionError):
            wl.assert_sham_equivalence(normal, broken, normal.copy(), tau, s, e, self.scaler, models)

    def test_calibration_order_and_strict_exceedance(self):
        self.assertEqual(int(np.ceil((1024+1)*.99)), 1015)
        model = self.model('AR191')
        runs = [wl.NormalRun('UNIT', 'CAL', 900, self.rng.normal(size=(384, 2)))]
        cal = wl.calibrate(runs, self.scaler, model)['last']
        self.assertEqual(cal['n'], 2)
        self.assertTrue(np.isinf(cal['threshold']))  # insufficient finite-budget points, not test chosen
        with self.assertRaises(ValueError):
            wl.calibrate([wl.NormalRun('UNIT','DEV',0,runs[0].values)],self.scaler,model)

    def test_metric_single_class_sham_and_run_bootstrap(self):
        tau=np.full(3840,12); layer=wl.strata(tau,1152,2688)
        y=np.zeros(3840,np.int8); y[1152:2688]=1
        score=y[layer['endpoint']].astype(float)
        a=wl.evaluate_readout(score,y,layer,.5,1152,2688,'A')
        core=next(row for row in a['layers'] if row['layer']=='CORE')
        self.assertEqual(core['AP'],1.)
        self.assertEqual(a['events'][0]['first_alarm'],1343)
        self.assertEqual(a['events'][0]['delay_points'],191)
        missed=wl.evaluate_readout(score,y,layer,2.,1152,2688,'A')['events'][0]
        self.assertIsNone(missed['delay_points']);self.assertTrue(missed['right_censored'])
        n=wl.evaluate_readout(score,np.zeros(3840),layer,.5,1152,2688,'N')
        self.assertTrue(n['matched_normal_interval_hit'])
        self.assertTrue(all(row['AP'] is None for row in n['layers']))
        ci,draw=wl.paired_bootstrap({'positive':[.1,.2,.3],'all_NA':[None,None,None]},repeats=40)
        self.assertEqual(draw.shape,(40,3));self.assertEqual(ci['positive']['valid_pairs'],3)
        self.assertEqual(ci['all_NA']['NA_bootstraps'],40)
        self.assertIsNone(ci['all_NA']['mean_difference'])

    def test_formal_wave_requires_real_root_release_before_output(self):
        import tempfile
        import json
        from pathlib import Path
        from hybrid_tsad.engine.weak_lag_wave import require_release
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            config={'H':192,'N':24576,'roles':wl.ROLES,'models':list(wl.MODELS),
                    'alphas':list(wl.ALPHAS),'scenarios':['LIN','WEAK','IRR'],'bootstrap_repeats':10000}
            (p/'config.json').write_text(json.dumps(config))
            (p/'pending.json').write_text(json.dumps({'status':'PENDING'}))
            with self.assertRaises(ValueError):
                require_release(p/'config.json',p/'pending.json')


if __name__ == '__main__':
    unittest.main()
