import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from hybrid_tsad.controlled_complementarity_design import energy, envelope
from hybrid_tsad.controlled_complementarity_evaluate import alarms, independent_ranking
from hybrid_tsad.controlled_complementarity_run import reuse_map, smooth_control


def test_smooth_controls_change_structure_at_equal_total_energy():
    start = 640
    base = np.ones((2304, 2), dtype=np.float32)
    for kind, length in (("H1", 8), ("H1", 32), ("H2", 1024)):
        time = np.arange(length)
        target = base.copy()
        if kind == "H1":
            target[start:start + length, 1] += 0.25 * np.sin(np.pi * time / (length - 1)) ** 2
        else:
            target[start:start + length, 1] += 0.25 * envelope(length, 64) * np.sin(np.pi * time / 8)
        control, _ = smooth_control(base, target, 1, start, length, kind)
        np.testing.assert_allclose(energy(control, base), energy(target, base), rtol=1e-5)
        np.testing.assert_array_equal(control[:, 0], base[:, 0])
        np.testing.assert_array_equal(control[:start + 1], base[:start + 1])
        np.testing.assert_array_equal(control[start + length - 1:], base[start + length - 1:])
        delta = control[start:start + length, 1] - base[start:start + length, 1]
        if kind == "H1":
            assert np.any(delta < 0) and np.any(delta > 0) and abs(delta.mean()) < 1e-7
        else:
            assert np.all(delta >= 0) and np.ptp(delta[64:-64]) == 0


def test_cache_reuse_requires_full_footprint_including_arrived_target():
    base = np.arange(20, dtype=np.float32)[:, None]
    changed = base.copy()
    changed[8] = 20
    sham = base.copy()
    sham[8:14] = base[6:12]
    times = np.arange(4, 20)
    result = reuse_map(np.stack([base, changed, sham]), ["base", "target", "H3_sync2"], times, 4)
    np.testing.assert_array_equal(times[result[1] < 0], [8, 9, 10, 11, 12])
    np.testing.assert_array_equal(times[result[2, np.isin(times, [12, 13])]], [10, 11])


def test_independent_ranking_handles_ties_and_c3_uses_strict_no_backfill():
    labels = np.array([False, True, True, False, True, False])
    values = np.array([1, 2, 2, 2, 3, 1], dtype=float)
    expected = average_precision_score(labels, values), roc_auc_score(labels, values)
    np.testing.assert_allclose(independent_ranking(labels, values), expected)
    np.testing.assert_array_equal(alarms(values, 1)["confirmed3"], [False, False, False, True, True, False])
    assert not np.any(alarms(values, 2)["confirmed3"])
