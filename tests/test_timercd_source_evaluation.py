import pytest
from hybrid_tsad.timercd_source_evaluation import terminal_gate, paired, MODELS, PERMISSIONS


def test_exact_complete_seal_and_permissions():
    names = [str(i) for i in range(29)]
    banks = [dict(relative=f'score/{m}/{r}/{p}/{role}.npz') for m in MODELS
             for r in names for p in PERMISSIONS for role in ('CAL', 'QUERY')]
    seal = dict(status='COMPLETED', no_labels_decoded=True, banks=banks)
    assert len(terminal_gate(seal, names)) == 348
    with pytest.raises(ValueError):
        terminal_gate(dict(seal, banks=banks[:-1]), names)
    with pytest.raises(ValueError):
        terminal_gate(dict(seal, no_labels_decoded=False), names)


def test_pairing_never_subtracts_different_permissions():
    rows = []
    for p in PERMISSIONS:
        for m, score in [('native', .2), ('time', .3), ('frft', .4)]:
            rows.append(dict(record='r', arm=m + '__' + p, AP=score))
        rows.append(dict(record='r', arm=p, AP=.1))
    details, summary = paired(rows, ['r'])
    ap = [r for r in summary if r['metric'] == 'AP']
    assert len(ap) == 6
    assert all(r['paired_n'] == 1 and r['positive'] == 1 for r in ap)
    for row in details:
        p = row['candidate'].split('__')[1]
        assert row['reference'] == p or row['reference'].endswith('__' + p)
