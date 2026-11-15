"""Incremental checks for common CAL and strictly-past statistical controls."""
import unittest
import numpy as np

from hybrid_tsad.normality_density_evaluation import common_cal_bank,past_mean_and_level,untrained_haar_prior


class FixedEvaluation(unittest.TestCase):
    def test_haar_prior_is_exact_untrained_full_scalar_and_observed_window_only(self):
        import torch
        from hybrid_tsad.normality_density import NormalityDensity
        x=np.random.default_rng(2025).normal(size=(2,13,3)).astype(np.float32)
        loc=np.array([[8,9],[8,9]])
        prior=untrained_haar_prior(x,loc,8)
        model=NormalityDensity(3,8,hidden=4).eval()
        with torch.no_grad():
            windows=np.stack([x[r,t-7:t+1]for r in range(2)for t in loc[r]])
            expected=model.score(torch.from_numpy(windows)).mean(-1).numpy().reshape(2,2)
        np.testing.assert_allclose(prior,expected,rtol=1e-6,atol=1e-6)
        x[:,10:]=999
        np.testing.assert_array_equal(prior,untrained_haar_prior(x,loc,8))

    def test_common_cal_retains_old_mature_axis_and_channel_alignment(self):
        ix=np.arange(445853,495000)[None]
        channels=np.repeat(ix[...,None],3,axis=2)
        raw=dict(index=ix,raw_score=ix.astype(float),channel_score=channels,run=np.zeros_like(ix))
        new=common_cal_bank(raw,'SWAT','CAL')
        np.testing.assert_array_equal(new['index'][0],np.arange(446173,495000))
        np.testing.assert_array_equal(new['channel_score'][...,0],new['index'])
        self.assertEqual(raw['index'][0,0],445853)
        self.assertIs(common_cal_bank(raw,'SWAT','QUERY'),raw)
        # All fixed calibration endpoints remain available; no query labels.
        selected=np.linspace(446173,494999,512,dtype=np.int64)
        np.testing.assert_array_equal(new['index'][0,selected-446173],selected)

    def test_history_mean_excludes_current_and_future(self):
        x=np.arange(60,dtype=float).reshape(2,10,3)
        loc=np.array([[4,6],[4,6]])
        h,l=past_mean_and_level(x,loc,4)
        expected=np.array([[np.abs(x[b,t]-x[b,t-4:t].mean(0)).mean()for t in loc[b]]for b in range(2)])
        np.testing.assert_allclose(h,expected)
        np.testing.assert_allclose(l,np.abs(x[np.arange(2)[:,None],loc]).mean(-1))
        changed=x.copy();changed[:,7:]=9999
        h2,l2=past_mean_and_level(changed,loc,4)
        np.testing.assert_array_equal(h2,h);np.testing.assert_array_equal(l2,l)
        changed=x.copy();changed[:,6]+=20
        h3,_=past_mean_and_level(changed,loc,4)
        np.testing.assert_allclose(h3[:,1]-h[:,1],20)


if __name__=='__main__':unittest.main()
