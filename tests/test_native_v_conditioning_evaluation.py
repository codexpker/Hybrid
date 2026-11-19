import numpy as np
import pytest

from hybrid_tsad.native_v_conditioning_evaluation import (
    exact_cal, quantile_score, raw_residual, step0_equivalence,
    point_formula_gate, sealed_inputs)


def test_quantiles_keep_units_ties_and_negative_scores():
    v = np.array([.2, 2., .5], np.float32)
    q = np.array([[.05, .2, .4], [1., 1., 1.], [-1., 0., 1.]], np.float32)
    actual = quantile_score(v, q, 2.)
    expected = (v.astype(float) / 2. - q[:, 2].astype(float)) / np.maximum(
        q[:, 2].astype(float) - q[:, 0].astype(float), 1e-3)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64 and actual[0] < 0 and actual[1] == 0
    with pytest.raises(ValueError):
        quantile_score(v, q[:, ::-1], 2.)


def test_raw_residual_is_pointwise_and_does_not_use_V():
    actual = raw_residual([[1, 3], [2, 4]], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(actual, [[1, 4], [1, 16]])
    np.testing.assert_array_equal(actual.mean(-1), [2.5, 8.5])


def test_D_archived_raw_predictions_are_the_only_scoring_evidence():
    pred = np.array([[1, 3], [2, 4]], dtype=np.float32)
    actual = np.array([[0, 1], [1, 0]], dtype=np.float32)
    residual = raw_residual(pred, actual)
    bank = dict(yhat_normalized=pred, actual_normalized=actual,
        residual_per_channel=residual, canonical_score=residual.mean(-1), V=np.array([1e8, 0]))
    assert point_formula_gate("D", bank)["max_score_formula_abs_difference"] == 0
    bank["V"] *= 5
    assert point_formula_gate("D", bank)["max_score_formula_abs_difference"] == 0


def test_declared_cal_points_cannot_be_replaced_or_reordered():
    index = np.arange(1000, 2024, 2, dtype=np.int64)
    score = np.arange(512, dtype=np.float64)
    assert exact_cal(index, score, index) == 506
    with pytest.raises(ValueError):
        exact_cal(index[::-1], score[::-1], index)
    with pytest.raises(ValueError):
        exact_cal(index[:-1], score[:-1], index)


def test_step0_affine_reference_preserves_native_ranking_and_alarms():
    cal = np.linspace(0, 3, 512, dtype=np.float32)
    query = np.array([0, 0, .5, .5, 1, 2, 4], dtype=np.float32)
    info, scores = step0_equivalence(cal, query, 2., [.1, .2, .8])
    assert info["order_reversals"] == info["newly_compressed_adjacent_values"] == 0
    assert info["native_tie_pairs_split"] == info["strict_alarm_disagreements"] == 0
    assert np.all(np.diff(scores) >= 0) and not info["inference"]


def test_step0_records_float_compression_without_changing_expression():
    cal = np.linspace(0, 1e-30, 512)
    query = np.array([1e-31, 1e-30, 1.1e-30])
    info, _ = step0_equivalence(cal, query, 1., [.1, .5, 1.])
    assert info["newly_compressed_adjacent_values"] > 0
    assert info["strict_alarm_disagreements"] > 0


def test_saved_quantile_evidence_must_use_one_consistent_unit():
    v = np.array([.5, 2.])
    q = np.array([[0., .1, .3], [.2, .2, .2]])
    width = q[:, 2] - q[:, 0]
    bank = dict(V=v, q_normalized=q, s_V=np.array(2.), q_physical=q * 2,
        width_normalized=width, width_physical=width * 2,
        denominator_normalized=np.maximum(width, 1e-3),
        denominator_physical=np.maximum(width, 1e-3) * 2,
        floor_active=width < 1e-3, canonical_score=quantile_score(v, q, 2.))
    assert point_formula_gate("C", bank)["max_score_formula_abs_difference"] == 0
    bank["q_physical"] = q
    with pytest.raises(AssertionError):
        point_formula_gate("C", bank)


def test_contradictory_label_seal_is_rejected_before_asset_access(tmp_path):
    import json
    (tmp_path / "GLOBAL_RAW_BANK_SEAL.json").write_text(json.dumps(
        dict(status="COMPLETED", no_labels_decoded=True, labels_read=True)))
    with pytest.raises(ValueError, match="label-free"):
        sealed_inputs(tmp_path, tmp_path)
