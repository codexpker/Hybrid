import pytest
from hybrid_tsad.timercd_reference_evaluation import terminal_gate, paired, MODELS, PERMISSION


def test_terminal_source_reference_pair_is_exactly_116_banks():
    names = [str(i) for i in range(29)]
    banks = [dict(relative=f'score/{m}/{r}/{PERMISSION}/{role}.npz') for m in MODELS
             for r in names for role in ('CAL', 'QUERY')]
    seal = dict(status='COMPLETED', no_labels_decoded=True, banks=banks)
    assert len(terminal_gate(seal, names)) == 116
    with pytest.raises(ValueError):
        terminal_gate(dict(seal, banks=banks[:-1]), names)
    with pytest.raises(ValueError):
        terminal_gate(dict(seal, no_labels_decoded=False), names)


def test_reference_pair_does_not_substitute_historical_continuation():
    rows = [dict(record='r', arm=m, AP=a) for m, a in
            [('batch', .3), ('reference', .4), (PERMISSION, .5)]]
    details, summary = paired(rows, ['r'])
    ap = [r for r in summary if r['metric'] == 'AP']
    assert len(ap) == 2
    assert ap[0]['comparison'] == 'MATCHED_SOURCE_NORMALIZATION'
    assert ap[0]['mean_difference'] == pytest.approx(.1)
    assert ap[1]['comparison'] == 'OFFICIAL_ZERO_UPDATE_REFERENCE'
    assert ap[1]['mean_difference'] == pytest.approx(-.1)
