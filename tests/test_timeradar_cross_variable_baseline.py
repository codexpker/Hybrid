from types import SimpleNamespace
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1.baselines import baseline_record
from hybrid_tsad.timeradar_cross_variable_baseline import history_mean512


class FixedHistoryFormula(unittest.TestCase):
    def test_matches_existing_complete_observation_formula(self):
        values=np.random.default_rng(2024).normal(size=(650,4)).astype(np.float32)
        values[:,3]=1.25
        run=SimpleNamespace(values=values,observed=np.ones_like(values,dtype=bool),
                            start=1000,domain='SMD',entity='synthetic',run='train',role='CAL')
        score=history_mean512(values)
        for t in (512,513,600,649):
            gold=baseline_record(run,t,method='history_mean512')['r_pred']
            self.assertAlmostEqual(score[t-512],gold,places=12)

    def test_current_and_future_excluded_from_reference(self):
        values=np.random.default_rng(2).normal(size=(520,3))
        a=history_mean512(values)
        changed=values.copy();changed[515:]+=3
        b=history_mean512(changed)
        np.testing.assert_array_equal(a[:3],b[:3])
        self.assertNotEqual(a[3],b[3])


if __name__=='__main__':unittest.main()
