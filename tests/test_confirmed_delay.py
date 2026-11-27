import numpy as np

from hybrid_tsad.confirmed_delay import (
    HISTORY,
    WINDOW,
    combine,
    fit_reference,
    score_region,
    score_window,
)


def example():
    rng = np.random.default_rng(195)
    drive = rng.normal(size=HISTORY + WINDOW + 24)
    t = 24 + np.arange(HISTORY + WINDOW)
    raw = np.column_stack((drive[t], drive[t - 4]))
    shifted = raw[HISTORY:].copy()
    shifted[:, 1] = drive[t[HISTORY:] - 16]
    return raw[:HISTORY], raw[HISTORY:], shifted


def test_history_and_cross_half_confirmation():
    past, clean, shifted = example()
    ref = fit_reference(past)
    assert [(p.driver, p.receiver, p.delay) for p in ref.links] == [(0, 1, 4)]
    clean_score = score_region(ref, past, clean, 64, 128)
    changed_score = score_region(ref, past, shifted, 64, 128)
    assert clean_score.delay == 0
    assert changed_score.best_delay == 16
    assert changed_score.confirming_links == 1 and changed_score.delay > 0.5
    assert changed_score.unconfirmed >= changed_score.delay

    # Both hypotheses fit the same nuisance across each half. Exclude the
    # intervention boundary from this algebraic check.
    modified = shifted.copy()
    modified[:, 1] = 3.5 * shifted[:, 1] + 2.0
    new = score_region(ref, past, modified, 64, 128)
    np.testing.assert_allclose(
        (new.delay, new.residual),
        (changed_score.delay, changed_score.residual),
        atol=1e-12,
    )
    modified[128:] += 100
    assert score_region(ref, past, modified, 64, 128) == new


def test_absolute_residual_can_be_high_without_a_confirmed_delay():
    past, _, shifted = example()
    ref = fit_reference(past)
    corrupted = shifted.copy()
    corrupted[:, 1] = np.random.default_rng(803).normal(size=WINDOW)
    a = score_region(ref, past, shifted, 64, 128)
    b = score_region(ref, past, corrupted, 64, 128)
    assert a.residual > 0.5 and b.residual > 0.5
    assert a.delay > 0.5 and b.delay == 0 and b.confirming_links == 0


def test_undefined_reference_preserves_native_tail():
    history = np.ones((HISTORY, 2))
    current = np.ones((WINDOW, 2))
    ref = fit_reference(history)
    assert not ref.links
    blocks = score_window(ref, history, current)
    assert all(b.available_links == 0 and b.delay == 0 for b in blocks)
    assert all(b.testable_links == 0 and b.unconfirmed == 0 for b in blocks)
    native = np.arange(WINDOW)
    normal = np.array([0, 20, 20, 200])
    expected = -np.log(
        (1 + (normal[:, None] >= native[None, :]).sum(0)) / (1 + len(normal))
    )
    np.testing.assert_array_equal(combine(native, blocks, normal, [0, 1, 2]), expected)
