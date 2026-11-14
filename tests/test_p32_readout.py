import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tsad.hybrid_v1.p32_readout import align_predictions


class P32Tests(unittest.TestCase):
    def fixture(self):
        origin = np.arange(70)
        h = np.arange(32)
        c = np.arange(3)
        prediction = origin[:, None, None] + h[None, :, None] + c[None, None] + (h[None, :, None] - 12) * .07
        target_index = np.arange(31, 65)
        target = target_index[:, None] + c[None]
        return dict(origin_index=origin, issued=origin-1, prediction=prediction,
                    forecast_valid=np.ones_like(prediction, bool), target_index=target_index, target=target,
                    native_h1_mask=np.ones_like(target, bool), sigma_t=np.tile([1., 2., 4.], (len(target), 1)),
                    original_h1=np.full(len(target), .123456789012345))

    def test_origin_axis_latest_scale_jensen_and_h1_exact(self):
        data = self.fixture()
        result = align_predictions(**data, block_size=7)
        self.assertTrue(result['eligible'].all())
        np.testing.assert_array_equal(result['variants']['H1'], data['original_h1'])
        delta = result['variants']['EVIDENCE'] - result['variants']['MEAN']
        np.testing.assert_allclose(delta, result['between_origin_variance_normalized'], rtol=1e-12, atol=1e-14)
        expected_mean = np.mean(((np.arange(32).mean() - 12) * .07 / np.array([1., 2., 4.])) ** 2)
        np.testing.assert_allclose(result['variants']['MEAN'], expected_mean, atol=1e-14)

    def test_missing_horizon_does_not_change_native_channel_mask(self):
        data = self.fixture()
        # For t=40, origin33/h7 predicts target40. One native channel invalid
        # invalidates the target; it is not dropped or horizon-averaged away.
        data['forecast_valid'][33, 7, 1] = False
        result = align_predictions(**data)
        location = int(np.flatnonzero(data['target_index'] == 40)[0])
        self.assertFalse(result['eligible'][location])
        self.assertEqual(result['reason'][location], 'INCOMPLETE_NATIVE_CHANNEL_HORIZON')
        self.assertTrue(all(np.isnan(result['variants'][v][location]) for v in ('H1', 'MEAN', 'EVIDENCE')))
        data['native_h1_mask'][:, 1] = False
        self.assertTrue(align_predictions(**data)['eligible'].all())

    def test_missing_origin_and_future_issue_are_not_repaired(self):
        data = self.fixture()
        keep = data['origin_index'] != 12
        for name in ('origin_index', 'issued', 'prediction', 'forecast_valid'):
            data[name] = data[name][keep]
        result = align_predictions(**data)
        self.assertFalse(result['eligible'][0])
        self.assertEqual(result['reason'][0], 'MISSING_PAST_ORIGIN')
        data = self.fixture()
        data['issued'][12] = 99
        with self.assertRaises(PermissionError):
            align_predictions(**data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
