import numpy as np
import pytest
from hybrid_tsad.left_selective_evaluation import (ARMS, terminal_gate, hard_nominal_mask,
                                                  validate_retained, contrasts)


def test_global_four_arm_seal_before_annotations():
    banks = [dict(relative=f'{a}/score/{r}.npz') for a in ARMS for r in ('CAL', 'QUERY')]
    seal = dict(status='COMPLETED', no_labels_decoded=True, banks=banks)
    assert len(terminal_gate(seal)) == 8
    with pytest.raises(ValueError):
        terminal_gate(dict(seal, banks=banks[:-1]))
    with pytest.raises(ValueError):
        terminal_gate(dict(seal, no_labels_decoded=False))


def test_nominal_difficulty_proxy_is_fixed25_and_ties_keep_original_slot():
    loss = np.zeros((2000, 128), dtype=np.float32)
    loss[:, 100] = 10
    mask = hard_nominal_mask(loss)
    assert np.all(mask.sum(1) == 25)
    assert mask[:, 100].all() and mask[:, :24].all() and not mask[:, 24:100].any()
    assert hard_nominal_mask(loss).dtype == np.bool_


def test_keep_mask_is_original103_slot_identity_not_sorted_loss_position():
    indices = np.tile(np.arange(103), (2000, 1))
    keep = np.zeros((2000, 128), bool); keep[:, :103] = True
    validate_retained(indices, keep)
    indices[0, 1] = 0
    with pytest.raises(ValueError):
        validate_retained(indices, keep)


def test_BC_and_difference_in_differences_are_conditions_not_repetitions():
    values = {'B_trim':.5, 'C_trim':.4, 'B_random':.3, 'C_random':.35,
              'U_B':.2, 'U_C':.3}
    rows = [dict(method=m, stream=s, AP=v) for m,v in values.items() for s in ('strict','confirmed3')]
    result = contrasts(rows)
    r = next(r for r in result if r['stream']=='strict' and r['comparison']=='B-C_trim_difference-minus-ordinary')
    assert r['AP_delta'] == pytest.approx(.2)
    assert all('sample_SD' not in r for r in result)
