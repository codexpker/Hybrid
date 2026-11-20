import numpy as np
import pytest

from hybrid_tsad.native_v_conditioning_evaluation import (
    exact_cal, quantile_score, raw_residual, step0_equivalence,
    point_formula_gate, sealed_inputs, event_contrasts, METHODS, replay_pairs, replay_sealed_inputs,
    target_mean_inputs, TARGET_MEAN_ARM, adaptation_pair_rows, repeat_inputs)


def test_smd_cal_uses_separate_train_record_and_never_indexes_test_labels(monkeypatch):
    from hybrid_tsad import native_v_conditioning_evaluation as ev
    from hybrid_tsad import left_native_swat_metrics as metrics
    monkeypatch.setattr(metrics, "load_author_metric", lambda p: None)
    monkeypatch.setattr(metrics, "continuous_metrics", lambda y, s, m: (
        {"AP": .4, "ROC_AUC": .6, "VUS_PR": .3, "VUS_ROC": .5}, ([], [], [], [])))
    index = np.arange(*ev.SMD_FIXED_QUERY)
    # Different physical record: deliberately above the entire test label axis.
    cal_index = np.arange(40000, 40512, dtype=np.int64)
    def bank(t, score):
        return dict(absolute_time_index=t, available_index=t, window_start=t-99,
                    window_end_exclusive=t+1, V=score.copy(), canonical_score=score.copy())
    cal = bank(cal_index, np.linspace(.1, 1., 512))
    query = bank(index, .1 + np.abs(np.sin(np.arange(len(index))*.01)))
    labels = np.zeros(23703, np.int8)
    labels[9000:9700] = 1; labels[12000:12020] = 1
    rows, events, _ = ev.evaluate_smd_arrays("machine-3-2", "R", cal, query, cal_index, labels, None)
    assert len(rows) == 2 and all(r["domain"] == "SMD" for r in rows)
    assert all(r["entity"] == "machine-3-2" and r["CAL_selected_attack_points"] is None for r in rows)
    assert all(r["Ntrue"] == 2 and r["long_event_count"] == 1 for r in rows)
    assert rows[0]["threshold"] == np.quantile(cal["canonical_score"], .99, method="higher")
    assert len(events) == 4
    with pytest.raises(ValueError, match="predeclared"):
        ev.evaluate_smd_arrays("machine-1-1", "R", cal, query, cal_index, labels, None)


def test_smd_incomplete_seal_blocks_all_data_and_label_reads(tmp_path, monkeypatch):
    import json
    from hybrid_tsad import native_v_conditioning_evaluation as ev
    (tmp_path / "GLOBAL_RAW_BANK_SEAL.json").write_text(json.dumps(dict(
        status="COMPLETED", labels_read=True, no_labels_decoded=True)))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("incomplete seal must block arrays"))
    with pytest.raises(ValueError, match="label-free"):
        ev.smd_inputs(tmp_path, tmp_path)


def test_smd_pairs_are_per_entity_single_seed_and_keep_unqualified_as_NA():
    from hybrid_tsad import native_v_conditioning_evaluation as ev
    rows = [dict(entity=e, method=m, stream=s, AP=v, long_steady_macro_coverage=None)
            for e in ev.SMD_FIXED_ENTITIES for s in ("strict", "confirmed3")
            for m, v in (("noU", .2), ("R", .1), (ev.TARGET_MEAN_ARM, .3))]
    pairs = ev.smd_learning_pairs(rows)
    assert len(pairs) == 8
    assert all(r["n"] == 1 and r["sample_SD"] is None for r in pairs)
    assert all(r["long_steady_macro_coverage"] is None for r in pairs)
    assert {r["entity"] for r in pairs} == set(ev.SMD_FIXED_ENTITIES)
    for r in pairs:
        assert r["AP"] == pytest.approx(.2 if r["contrast"].endswith("−R") else .1)


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


def test_event_pair_export_retains_half_open_boundaries():
    events = [dict(domain="PSM", method=m, stream="strict", event_id=1,
        main_eligible=True, start=10, end_exclusive=30, total_points=20,
        coverage=.5, steady_after512_coverage=None, longest_miss_points=10,
        steady_longest_miss_points=None, delay_points=2) for m in METHODS.values()]
    result = event_contrasts(events)
    assert len(result) == 1 and result[0]["end_exclusive"] == 30
    assert result[0]["end_exclusive"] - result[0]["start"] == result[0]["total_points"]


def test_replay_comparisons_keep_noU_and_SS_distinct():
    rows = [dict(domain=d, method=m, stream=s, AP=value, FP=int(100 * value))
        for d in ("SWAT", "PSM") for s in ("strict", "confirmed3")
        for m, value in (("noU", .3), ("TT", .2), ("SS", .6), ("R", .5))]
    result = replay_pairs(rows)
    assert len(result) == 16 and all(r["n"] == 1 and r["sample_SD"] is None for r in result)
    lookup = {r["contrast"]: r for r in result if r["domain"] == "SWAT" and r["stream"] == "strict"}
    assert lookup["R−TT"]["AP"] == .5 - .2
    assert lookup["R−SS"]["AP"] == .5 - .6
    assert lookup["R−noU"]["FP"] == 20


def test_replay_label_guard_precedes_asset_loading(tmp_path):
    import json
    (tmp_path / "GLOBAL_RAW_BANK_SEAL.json").write_text(json.dumps(dict(
        status="COMPLETED", labels_read=True, no_labels_decoded=False)))
    with pytest.raises(ValueError, match="label-free"):
        replay_sealed_inputs(tmp_path, tmp_path)


def test_target_mean_label_guard_precedes_asset_loading(tmp_path):
    import json
    (tmp_path / "GLOBAL_RAW_BANK_SEAL.json").write_text(json.dumps(dict(
        status="COMPLETED", labels_read=True, no_labels_decoded=False)))
    with pytest.raises(ValueError, match="label-free"):
        target_mean_inputs(tmp_path, tmp_path)


def test_single_factor_comparisons_use_new_minus_all_four_baselines():
    rows = [dict(domain=d, method=m, stream=s, AP=value, normal_FPR=value / 10)
            for d in ("SWAT", "PSM") for s in ("strict", "confirmed3")
            for m, value in (("noU", .3), ("TT", .2), ("SS", .6), ("R", .5), (TARGET_MEAN_ARM, .4))]
    contrasts = tuple((TARGET_MEAN_ARM, m) for m in ("R", "noU", "TT", "SS"))
    result = replay_pairs(rows, contrasts)
    assert len(result) == 16 and all(r["n"] == 1 and r["sample_SD"] is None for r in result)
    assert {r["contrast"] for r in result} == {TARGET_MEAN_ARM + "−" + m for m in ("R", "noU", "TT", "SS")}
    for r in result:
        expected = dict(noU=.3, TT=.2, SS=.6, R=.5)[r["contrast"].split("−")[1]]
        assert r["AP"] == .4 - expected


def test_adaptation_repetition_pairs_same_seed_and_single_noU_reference():
    rows = []
    for d in ("SWAT", "PSM"):
        for stream in ("strict", "confirmed3"):
            rows.append(dict(domain=d, method="noU", seed=None, stream=stream, AP=.4))
            for index, seed_value in enumerate((2025, 2026, 2027)):
                rows.append(dict(domain=d, method="R", seed=seed_value, stream=stream, AP=.1 + index * .1))
                rows.append(dict(domain=d, method=TARGET_MEAN_ARM, seed=seed_value, stream=stream, AP=.2 + index * .2))
    pairs, summary = adaptation_pair_rows(rows, fields=("AP",))
    assert len(pairs) == 24 and len(summary) == 8
    same_seed = [p for p in pairs if p["domain"] == "SWAT" and p["stream"] == "strict" and p["contrast"].endswith("−R")]
    np.testing.assert_allclose([p["AP"] for p in same_seed], [.1, .2, .3])
    s = next(s for s in summary if s["domain"] == "SWAT" and s["stream"] == "strict" and s["contrast"].endswith("−R"))
    assert s["n"] == 3 and s["ddof"] == 1
    assert np.isclose(s["mean"], .2) and np.isclose(s["sample_SD"], .1)
    noU = [p for p in pairs if p["domain"] == "SWAT" and p["stream"] == "strict" and p["contrast"].endswith("−noU")]
    np.testing.assert_allclose([p["AP"] for p in noU], [-.2, 0, .2])
    assert all(p["noU_is_single_fixed_reference"] for p in noU)


def test_repeat_label_guard_runs_before_any_asset_or_label_read(tmp_path):
    import json
    (tmp_path / "GLOBAL_RAW_BANK_SEAL.json").write_text(json.dumps(dict(
        status="COMPLETED", labels_read=True, no_labels_decoded=False)))
    with pytest.raises(ValueError, match="label-free"):
        repeat_inputs(tmp_path, tmp_path)
