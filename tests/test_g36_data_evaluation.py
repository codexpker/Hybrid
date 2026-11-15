import numpy as np

from hybrid_tsad.g36_data import ROLES
from hybrid_tsad.g36_evaluation import confirm, max_gap, spans
from hybrid_tsad.g36_ridge import windows


def test_raw_role_footprints_disjoint():
    roles = list(ROLES.values())
    for i, (a, b) in enumerate(roles):
        for c, d in roles[i+1:]:
            assert b <= c or d <= a


def test_prediction_footprint_and_target_alignment():
    x = np.arange(70*9).reshape(70, 9)
    ends, h, y = windows(x)
    assert ends[0] == 51
    np.testing.assert_array_equal(h[0], x[:48])
    np.testing.assert_array_equal(y[0], x[48:52, :4])
    changed = x.copy()
    changed[48:52] = -999
    _, h2, y2 = windows(changed)
    np.testing.assert_array_equal(h[0], h2[0])
    assert not np.array_equal(y[0], y2[0])


def test_causal_confirmation_and_missing_run_boundaries():
    np.testing.assert_array_equal(confirm([1, 1, 1, 1, 0, 1, 1]), [0, 0, 1, 1, 0, 0, 0])
    assert confirm([1, 1]).sum() == 0
    assert confirm([1]).sum() == 0
    assert spans([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]


def test_long_miss_does_not_bridge_ineligible_points():
    assert max_gap([0, 0, 0, 0, 0], [1, 1, 0, 1, 1]) == 2
    assert max_gap([1, 1], [1, 1]) == 0
