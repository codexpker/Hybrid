"""Numerical low-FPR contracts; no models, training or real asset IO."""
import numpy as np
import pytest
from sklearn.metrics import roc_curve

from hybrid_tsad.native_v_conditioning_curves import (
    c3_scores, tie_group_roc, partial_roc_area, attainable_cap_point, low_fpr_entry)


def test_complete_ties_match_empirical_roc_without_splitting_groups():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([3, 3, 2, 2, 1, 1.])
    curve = tie_group_roc(y, s)
    x, z, cuts = roc_curve(y, s, drop_intermediate=False)
    for key, values in (("FPR", x), ("TPR", z), ("cutoffs", cuts)):
        np.testing.assert_array_equal(curve[key], values)
    assert len(curve["FPR"]) == 4
    assert attainable_cap_point(curve, .1) == 0  # no half-tie selection


def test_c3_resets_at_gaps_invalid_observation_and_never_backfills():
    index = np.array([0, 1, 2, 3, 8, 9, 10, 11])
    score = np.array([9, 8, 7, 6, 5, 4, 3, 2.])
    z, ok = c3_scores(index, score)
    np.testing.assert_array_equal(ok, [0, 0, 1, 1, 0, 0, 1, 1])
    np.testing.assert_array_equal(z[ok], [7, 6, 3, 2])
    # Even a cutoff below all finite values cannot activate history-less points.
    np.testing.assert_array_equal(ok & (z >= -1e10), ok)
    valid = np.ones(8, bool); valid[2] = False
    _, ok2 = c3_scores(index, score, valid)
    np.testing.assert_array_equal(ok2, [0, 0, 0, 0, 0, 0, 1, 1])


def test_c3_denominator_keeps_block_first_two_and_max_endpoint_below_one():
    index = np.array([0, 1, 2, 7, 8, 9])
    y = np.array([1, 0, 1, 1, 0, 0])
    z, ok = c3_scores(index, np.ones(6))
    c = tie_group_roc(y, z, ok)
    assert c["positive_points"] == c["normal_points"] == 3
    assert c["TPR"][-1] == c["FPR"][-1] == 1/3
    assert partial_roc_area(c["FPR"], c["TPR"], .5) is None
    z, ok = c3_scores(np.array([0, 9]), np.array([-2., -1.]))
    c = tie_group_roc([1, 0], z, ok)
    np.testing.assert_array_equal(c["TPR"], [0.])
    assert attainable_cap_point(c, .05) == 0


def test_integral_linear_boundary_vertical_segments_and_no_convex_hull():
    # Vertical gain at FPR0 retained; area to .01 = .4*.01 + .5*.1*.01.
    assert partial_roc_area([0, 0, .02, 1], [0, .4, .6, 1], .01) == pytest.approx(.0045)
    assert partial_roc_area([0, .02, 1], [0, .6, 1], .01) == pytest.approx(.0015)
    assert partial_roc_area([0, .01, .01, 1], [0, .2, .7, 1], .01) == pytest.approx(.001)


def test_cap_is_discrete_not_interpolated_and_plateau_prefers_minimum_fpr():
    c = dict(FPR=np.array([0, 0, .02, .03, .05]), TPR=np.array([0, .3, .6, .6, .8]))
    assert attainable_cap_point(c, .01) == 1  # attainable .3, not interpolated .45
    assert attainable_cap_point(c, .03) == 2  # sameTPR at .02 wins over .03
    with pytest.raises(ValueError): attainable_cap_point(c, -1)


def test_entry_c3_event_callback_gets_actual_bits_once_not_double_confirmation():
    index = np.arange(8); s = np.array([9., 9, 9, 0, 0, 0, 0, 0])
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    c = tie_group_roc(y, s)
    auc = 1.  # independent perfect-ranking expectation
    pair = [dict(stream="strict", threshold=5., TP=3, FP=0, Recall=1., normal_FPR=0., ROC_AUC=auc),
            dict(stream="confirmed3", threshold=5., TP=1, FP=0, Recall=1/3, normal_FPR=0., ROC_AUC=auc)]
    calls = []
    def callback(index_, bits, labels):
        calls.append(bits.copy())
        assert bits[2] and not bits[0] and not bits[1]  # one C3 point, not five-point confirmation
        return {}
    rows, arrays = low_fpr_entry(index, s, y, pair, dict(domain="fixture", method="R", seed=2026), event_summary=callback)
    assert len(rows) == 2 and len(calls) == 4  # originalCAL plus3 descriptive caps
    assert rows[1]["cap01_TPR"] == 1/3
    assert rows[1]["CAL_operating_QUERY_Recall"] == 1/3
    assert not any("threshold" in key or "cutoff" in key for key in rows[1])
    assert not arrays["confirmed3_cap01_alarm"][:2].any()


def test_exact_seed_case_filter_never_opens_old_seed_banks(tmp_path, monkeypatch):
    import json
    from hybrid_tsad import native_v_conditioning_curves as c
    monkeypatch.setattr(c, "AXES", {"fixture": (0, 4)})
    label_path = tmp_path / "labels.npy"; np.save(label_path, [0, 1, 0, 1])
    manifest = tmp_path / "external.json"
    manifest.write_text(json.dumps({"labels": {"fixture": {"path": "labels.npy", "sha256": c.sha(label_path)}}}))
    monkeypatch.setattr(c, "EXTERNAL_MANIFEST", "external.json")
    monkeypatch.setattr(c, "EXTERNAL_MANIFEST_SHA", c.sha(manifest))
    root = tmp_path / "root"; root.mkdir(); seal = root / "GLOBAL_RAW_BANK_SEAL.json"; seal.write_text("{}")
    rows, descriptors = [], []
    for seed in (2025, 2026):
        hashes = {}
        for role in ("CAL", "QUERY"):
            path = tmp_path / f"seed{seed}_{role}.npz"
            if seed == 2026:  # old references deliberately absent: any old read must fail
                index = np.arange(512 if role == "CAL" else 4)
                np.savez(path, absolute_time_index=index, available_index=index,
                         canonical_score=np.ones(len(index)))
            hashes[role] = c.sha(path) if path.exists() else "OLD_MUST_NOT_READ"
            descriptors.append(dict(domain="fixture", method="R", seed=seed, role=role,
                relative=path.name, sha256=hashes[role]))
        for stream in ("strict", "confirmed3"):
            rows.append(dict(domain="fixture", method="R", seed=seed, stream=stream, threshold=1.,
                bank_CAL_sha256=hashes["CAL"], bank_QUERY_sha256=hashes["QUERY"]))
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(dict(status="COMPLETED", global_seal_sha256=c.sha(seal), rows=rows, bank_manifest=descriptors)))
    visited = []
    def fake_entry(index, score, labels, accepted, identity, **kwargs):
        visited.append(identity["seed"])
        return [dict(identity, stream=s) for s in ("strict", "confirmed3")], {"fixture": np.array([1.])}
    monkeypatch.setattr(c, "low_fpr_entry", fake_entry)
    result, _ = c.low_fpr_cached(tmp_path, root, metrics, None, tmp_path / "out",
        expected_metrics_sha=c.sha(metrics), methods=("R",), cases=[("fixture", "R", 2026)])
    assert visited == [2026] and len(result) == 2
    assert {r["seed"] for r in result} == {2026}
