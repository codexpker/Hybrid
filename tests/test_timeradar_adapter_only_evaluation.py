import unittest
import numpy as np
from hybrid_tsad.timeradar_adapter_only_evaluation import auxiliary_scores,complete_seal,RECORDS,ARMS


class AdapterOnlyEvaluation(unittest.TestCase):
    def test_pointwise_error_not_scalar_batch_MSE(self):
        mean=np.array([[1,1],[4,4]],np.float32);target=np.array([[0,0],[2,4]],np.float32)
        variance=np.array([[.1,.2],[.3,.4]],np.float32)
        scores=auxiliary_scores(mean,variance,target)
        np.testing.assert_array_equal(scores['E'],[1,2])
        self.assertEqual(scores['E'].dtype,np.float32)
        np.testing.assert_allclose(scores['E+0.9V'],[1.135,2.315],rtol=1e-6)
        with self.assertRaises(ValueError):auxiliary_scores(mean,variance,np.array(1,np.float32))

    def test_sample_variance_coefficient_pointwise_identity(self):
        copies=np.random.default_rng(2024).normal(size=(10,7,3)).astype(np.float32)
        target=np.random.default_rng(2).normal(size=(7,3)).astype(np.float32)
        score=auxiliary_scores(copies.mean(0),copies.var(0,ddof=1),target)
        np.testing.assert_allclose(score['E+0.9V'],np.square(copies-target[None]).mean((0,2)),rtol=2e-6)

    def test_exact_new_bank_gate(self):
        raw=dict(status='COMPLETED',banks=[dict(relative=f'{r}/{a}/score/{role}.npz')for r in RECORDS for a in ARMS for role in ('CAL','QUERY')])
        complete_seal(raw);raw['banks'].pop()
        with self.assertRaises(ValueError):complete_seal(raw)


if __name__=='__main__':unittest.main()
