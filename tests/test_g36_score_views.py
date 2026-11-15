import tempfile
import unittest
from pathlib import Path

import numpy as np

from hybrid_tsad.g36_score_views import materialize, sha


class ScoreViewTest(unittest.TestCase):
    def test_exact_field_and_input_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            source = p/'raw'
            source.mkdir()
            names = ['baseline__SELECT', 'baseline__CAL', 'baseline__QUERY',
                     'fault1__QUERY', 'fault2__QUERY', 'fault3__QUERY']
            for name in names:
                index = np.arange(51, 56)
                np.savez(source/(name+'.npz'), index=index, score=np.ones(5),
                    normal_nll=np.arange(5, dtype=float)-2,
                    available_index=index, min_input_index=index-51,
                    max_input_index=index, max_condition_index=index-4)
            before = {v: sha(v) for v in source.glob('*.npz')}
            rows = materialize(source, p/'views', 'normal_nll')
            self.assertEqual(len(rows), 6)
            for path, checksum in before.items():
                self.assertEqual(sha(path), checksum)
                with np.load(p/'views'/path.name) as z:
                    np.testing.assert_array_equal(z['score'], np.arange(5)-2)
            with self.assertRaises(FileExistsError):
                materialize(source, p/'views', 'normal_nll')


if __name__ == '__main__':
    unittest.main()
