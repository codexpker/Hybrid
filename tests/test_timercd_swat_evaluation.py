import numpy as np
import pytest
from hybrid_tsad.timercd_swat_evaluation import (threshold, terminal_gate, MODEL_IDS,
    CAL_START, CAL_END, QUERY_START, QUERY_END, evaluate_score)


def test_original512_cal_subset_not_whole_role():
    index = np.arange(CAL_START - 512, CAL_END)
    chosen = np.linspace(CAL_START, CAL_END - 1, 512, dtype=np.int64)
    score = np.full(len(index), 1000.)
    score[chosen - index[0]] = np.arange(512.)
    tau, actual = threshold(index, score)
    assert np.array_equal(actual, chosen)
    assert tau == np.quantile(np.arange(512.), .99, method='higher')
    assert tau != np.quantile(score, .99, method='higher')


def test_all_seven_frozen_models_must_seal():
    banks = [dict(relative=f'score/{m}/{role}.npz') for m in MODEL_IDS for role in ('CAL', 'QUERY')]
    seal = dict(status='COMPLETED', no_labels_decoded=True, banks=banks)
    assert len(terminal_gate(seal)) == 14
    with pytest.raises(ValueError):
        terminal_gate(dict(seal, banks=banks[:-1]))


def test_swat_long_definition_and_unobserved_events_are_not_misses():
    labels = np.zeros(QUERY_END, dtype=np.int8)
    labels[100:200] = 1  # outside the QUERY axis, never a complete-event miss
    start = QUERY_START + 100
    for length in [513] * 9 + [512] + [20] * 12:
        labels[start:start + length] = 1
        start += length + 30
    index = np.arange(QUERY_START, QUERY_END)
    rows, details, _, _ = evaluate_score(index, np.zeros(len(index)), labels, 1., index + 1)
    assert all(r['Ntrue'] == 22 and r['long_event_count'] == 9 for r in rows)
    assert all(r['delay_hit_only_median'] is None for r in rows)
    unknown = [e for e in details if e['start'] == 100]
    assert len(unknown) == 2
    assert all(e['observed_alarm_points'] == 0 and e['coverage'] is None
               and e['longest_miss_points'] is None and e['any_hit'] is None for e in unknown)
