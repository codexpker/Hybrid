import numpy as np
import pytest

from hybrid_tsad.conditional_response_evaluation import AXES, calibrated_threshold, select_bank


def test_common_psm_axis_respects512_history():
    assert AXES["PSM"]["QUERY"] == (22504 + 512, 87744)
    assert AXES["SWAT"]["QUERY"][0] == 608023 + 512
    assert 87744 - AXES["PSM"]["QUERY"][0] == 64728


def test_crop_preserves_scores_and_offline_availability(tmp_path):
    path = tmp_path / "bank.npz"
    index = np.arange(9, 20)
    np.savez(path, absolute_time_index=index, canonical_score=index * .125,
             available_index=np.full(len(index), 25))
    i, score, available = select_bank(path, (10, 17))
    np.testing.assert_array_equal(i, np.arange(10, 17))
    np.testing.assert_array_equal(score, np.arange(10, 17) * .125)
    assert np.all(available == 25)
    with pytest.raises(ValueError, match="available at t"):
        select_bank(path, (10, 17), causal=True)
    with pytest.raises(ValueError, match="fully present"):
        select_bank(path, (8, 17))


def test_threshold_uses_same512_points_and_strict_higher():
    index = np.arange(*AXES["PSM"]["CAL"])
    score = np.arange(len(index), dtype=float)
    threshold, points = calibrated_threshold(index, score, "PSM")
    chosen = np.linspace(index[0], index[-1], 512, dtype=np.int64)
    np.testing.assert_array_equal(points, chosen)
    expected = score[np.searchsorted(index, chosen)]
    assert threshold == np.quantile(expected, .99, method="higher")
    assert np.sum(expected > threshold) == 5
