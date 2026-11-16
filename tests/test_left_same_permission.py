"""Unit checks for role axes, native score reduction and strict profile loading."""
import unittest
import numpy as np
import torch
from hybrid_tsad.left_same_permission import RoleWindows, scientific_state


class TestIndependentRoles(unittest.TestCase):
    def test_role_windows_do_not_pad_or_bridge(self):
        x = np.zeros((400, 3), np.float32)
        ix = np.arange(1234, 1634)
        windows = RoleWindows(x, ix)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows.footprint(0), [1234, 1426])
        self.assertEqual(windows.footprint(1), [1426, 1618])
        self.assertTrue(np.shares_memory(windows[1][0], x))
        with self.assertRaises(IndexError):
            windows[2]
        with self.assertRaises(ValueError):
            RoleWindows(x, np.r_[np.arange(200), np.arange(300, 500)])

    def test_native_score_reduction_has_point_axis(self):
        raw = np.arange(2*192*3, dtype=np.float32).reshape(2, 192, 3)
        flattened = raw.reshape(-1, 3)
        np.testing.assert_array_equal(flattened.mean(-1), raw.mean(-1).reshape(-1))

    def test_strict_profile_counter_allowlist(self):
        model = torch.nn.Linear(3, 2)
        saved = dict(model.state_dict())
        saved['total_ops'] = torch.tensor(1.)
        scientific_state(model, saved, ['total_ops'])
        with self.assertRaises(ValueError):
            scientific_state(model, saved, [])
        with self.assertRaises(RuntimeError):
            scientific_state(model, {'total_ops': torch.tensor(1.)}, ['total_ops'])
        with self.assertRaises(ValueError):
            scientific_state(model, {**saved, 'unknown': torch.tensor(1.)}, ['total_ops', 'unknown'])


if __name__ == '__main__':
    unittest.main()
