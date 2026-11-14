import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tsad.roca_conditional_normality import fit_reference, score_reference, shared_center, cosine_sum
from hybrid_tsad.roca_representation_diagnostics import branch_distance


class ConditionalReferenceTests(unittest.TestCase):
    def setUp(self):
        r = np.random.default_rng(8324)
        self.q, self.qr = r.normal(size=(64, 6)), r.normal(size=(64, 6))
        self.keys = np.array([0] * 54 + [1] * 5 + [2] * 4 + [3])

    def test_same_builder_cos_formula_and_exact_backoff(self):
        ref = fit_reference(self.q, self.qr, self.keys, reference_role='FIT')
        np.testing.assert_array_equal(ref['global_center'], np.concatenate((self.q, self.qr)).mean(0))
        q, qr = self.q[:4], self.qr[:4]
        result = score_reference(ref, q, qr, np.array([1, 2, 77, 0]))
        self.assertEqual(result['conditioned'].tolist(), [True, False, False, True])
        self.assertEqual(result['context_reference_count'].tolist(), [5, 4, 0, 54])
        expected = branch_distance(q, ref['global_center']) + branch_distance(qr, ref['global_center'])
        np.testing.assert_allclose(result['scores']['GLOBAL_FIT64'][0], expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result['scores']['COMMAND_FIT64'][0][1:3], expected[1:3])

    def test_joint_nn_no_extra_normalization(self):
        ref = fit_reference(self.q, self.qr, self.keys, reference_role='FIT')
        result = score_reference(ref, self.q[:3], self.qr[:3], np.zeros(3))
        norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        r = np.concatenate((norm(self.q), norm(self.qr)), axis=1)
        x = r[:3]
        expected = np.sort(((x[:, None] - r[None]) ** 2).sum(2), axis=1)[:, :5].mean(1)
        np.testing.assert_allclose(result['scores']['JOINT_NN5'][0], expected, rtol=0, atol=1e-14)

    def test_reference_role_no_clipping_and_invalid_context_center(self):
        with self.assertRaises(PermissionError):
            fit_reference(self.q, self.qr, self.keys, reference_role='QUERY')
        q = np.ones((64, 2)); qr = q.copy(); qr[:5] = -1
        keys = np.array([1] * 5 + [0] * 59)
        ref = fit_reference(q, qr, keys, reference_role='FIT')
        result = score_reference(ref, q[:1], q[:1], np.array([1]))
        self.assertEqual(result['conditioned'].tolist(), [False])
        self.assertEqual(result['invalid_context_center'].tolist(), [True])
        np.testing.assert_array_equal(shared_center(np.array([[.001, 0.]]), np.array([[.001, 0.]])), [.001, 0.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
