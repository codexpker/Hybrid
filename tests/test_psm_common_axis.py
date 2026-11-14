import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1] / 'scripts/compare_psm_common_axis.py'
spec = importlib.util.spec_from_file_location('psm_common', P)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AxisTests(unittest.TestCase):
    def test_offline_axis_stays_native_not_window_end_offset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scores = np.ones((1, 4, 25), dtype=np.float32)
            canonical = scores.mean(-1).reshape(-1)
            raw = root / 'raw.npz'
            old = root / 'old.npz'
            y = np.array([0, 1, 0, 0, 1])
            np.savez(raw, absolute_time_index=np.arange(132481, 132485), canonical_score=canonical,
                     channel_scores=scores, channel_names=np.array(['feature_' + str(i) for i in range(25)]),
                     point_input_end=np.full(4, 132484))
            np.savez(old, full_labels=y, score=canonical)
            descriptor = dict(kind='left_native_window', raw=dict(path=str(raw), sha256=module.sha(raw)),
                old_evaluation_inputs=dict(path=str(old), sha256=module.sha(old)),
                test_original_start=132481, online=False)
            actual = module.load_method(descriptor, y)
            np.testing.assert_array_equal(actual['index'], np.arange(4))
            np.testing.assert_array_equal(actual['available'], np.full(4, 3))
            descriptor['online'] = True
            with self.assertRaises(PermissionError):
                module.load_method(descriptor, y)

    def test_actual_label_mapping_not_offset_search(self):
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / 'tr.npz'
            y = np.array([0, 1, 0, 1, 1])
            np.savez(raw, index=np.arange(4), scores=np.arange(4), labels=y[:4])
            descriptor = dict(kind='timeradar_native_summary', raw=dict(path=str(raw), sha256=module.sha(raw)),
                              online=False, window=2)
            actual = module.load_method(descriptor, y)
            np.testing.assert_array_equal(actual['index'], np.arange(4))
            wrong = np.roll(y, 1)
            with self.assertRaises(ValueError):
                module.load_method(descriptor, wrong)


if __name__ == '__main__':
    unittest.main(verbosity=2)
