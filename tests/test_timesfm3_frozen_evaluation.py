import json
import numpy as np
import pytest

from hybrid_tsad.timesfm3_frozen_evaluation import formula_gate, terminal_inputs, event_overlap


def fixture_bank():
    index = np.array([700, 701], np.int64)
    quantiles = np.tile(np.arange(9, dtype=np.float32), (2, 3, 1))
    quantiles[1] += np.float32(.25)
    pred = quantiles[..., 4].copy()
    actual = np.array([[1, 2, 3], [3, 4, 5]], np.float32)
    error = (pred.astype(np.float64)-actual.astype(np.float64))**2
    return dict(absolute_time_index=index, available_index=index,
        history_start=index-512, history_end_exclusive=index,
        prediction_normalized=pred, quantiles_normalized=quantiles,
        quantile_levels=np.arange(1, 10)/10, actual_normalized=actual,
        squared_error_per_channel=error, canonical_score=error.mean(axis=1))


def test_official_median_residual_is_pointwise_and_other_quantiles_are_only_archived():
    bank = fixture_bank()
    assert formula_gate(bank) == dict(points=2, channels=3, median_quantile_index=4,
        residual_precision="storedFP32 operands promotedFP64 before square/meanC", exact=True)
    score = bank["canonical_score"].copy()
    bank["quantiles_normalized"][..., :4] -= 100
    bank["quantiles_normalized"][..., 5:] += 100
    formula_gate(bank)
    np.testing.assert_array_equal(bank["canonical_score"], score)
    bank["canonical_score"] = np.mean(score)
    with pytest.raises(AssertionError):
        formula_gate(bank)


def test_forecast_has_strictly_past512_footprint_not_arrived_target_in_input():
    bank = fixture_bank()
    bank["history_end_exclusive"] = bank["absolute_time_index"]+1
    with pytest.raises(AssertionError): formula_gate(bank)


def test_frozen_full_model_seal_is_required_before_any_label_or_bank_load(tmp_path, monkeypatch):
    (tmp_path/"GLOBAL_RAW_BANK_SEAL.json").write_text(json.dumps(dict(status="COMPLETED", labels_read=True)))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Unsealed result must not open numeric assets"))
    with pytest.raises(ValueError, match="label-free"):
        terminal_inputs(tmp_path, tmp_path)


def test_common_and_unique_event_evidence_keeps_any_hit_and_matching_distinct():
    index = np.arange(10, 20)
    labels = np.zeros(25, np.int8); labels[12:15] = 1; labels[17:19] = 1
    left_score = np.array([2, 2, 2, 2, 0, 0, 2, 2, 0, 0.])
    right_score = np.array([0, 2, 2, 2, 2, 0, 0, 0, 0, 2.])
    left, right = [], []
    for stream in ("strict", "confirmed3"):
        for eid, lo, hi in (("E0", 12, 15), ("E1", 17, 19)):
            base = dict(stream=stream, event_id=eid, main_eligible=True, start=lo, end_exclusive=hi)
            left.append(dict(base, any_hit=True, one_to_one_matched=eid == "E0"))
            right.append(dict(base, any_hit=eid == "E0", one_to_one_matched=eid == "E0"))
    rows = event_overlap(index,left_score,1.,right_score,1.,labels,left,right)
    for row in rows:
        assert row["any_hit_TimesFM_only_IDs"] == ["E1"]
        assert row["one_to_one_matched_TimesFM_only_IDs"] == []
        assert row["one_to_one_matched_neither_IDs"] == ["E1"]
        assert sum(row[k] for k in ("normal_FP_both", "normal_FP_TimesFM_only", "normal_FP_TimeRadar_only", "normal_no_alarm")) == int(np.sum(labels[index] == 0))
    right[-1]["end_exclusive"] += 1
    with pytest.raises(ValueError, match="identity"):
        event_overlap(index,left_score,1.,right_score,1.,labels,left,right)
