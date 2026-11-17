import pytest
from hybrid_tsad.timercd_reference_repeats import finite_stats, summarize_groups


def test_sample_sd_and_unknown_are_not_population_or_zero():
    s = finite_stats([1, 2, 3])
    assert s['mean'] == 2 and s['sample_SD'] == 1
    assert finite_stats([.1, .1, .1])['sample_SD'] == 0
    s = finite_stats([None, .5, None])
    assert s['expected_seeds'] == 3 and s['valid_seeds'] == 1
    assert s['mean'] == .5 and s['sample_SD'] is None


def test_seed_group_requires_all_predeclared_seeds_and_preserves_pairs():
    rows = [dict(seed=seed, entity='r', difference=value)
            for seed, value in [(2025, -.1), (2026, .2), (2027, .2)]]
    r = summarize_groups(rows, ('entity',), ('difference',))[0]
    assert r['mean'] == pytest.approx(.1)
    assert r['sample_SD'] == pytest.approx(.17320508075688773)
    assert r['seed2025'] == -.1
    with pytest.raises(ValueError):
        summarize_groups(rows[:-1], ('entity',), ('difference',))
    with pytest.raises(ValueError):
        summarize_groups(rows + [rows[0]], ('entity',), ('difference',))
