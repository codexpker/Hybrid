import pickle
import random

import numpy as np
import torch

from hybrid_tsad.left_domain_replay import apply_numeric_replacements
from hybrid_tsad.left_exposure_bundle import legal_windows


def test_numeric_replacement_preserves_other_half_and_rng():
    batch=np.arange(256*192*3,dtype=np.float32).reshape(256,192,3)
    original=batch.copy();rows=np.full(128,-1,np.int32);rows[[0,127]]=[0,1]
    bank=np.full((2,192,3),-1,np.float32)
    nr=pickle.dumps(np.random.get_state());py=random.getstate();tr=torch.get_rng_state().clone()
    positions=apply_numeric_replacements(batch,rows,bank)
    assert np.array_equal(batch[128:],original[128:])
    untouched=np.setdiff1d(np.arange(128),positions)
    assert np.array_equal(batch[untouched],original[untouched])
    assert np.array_equal(batch[positions],bank)
    assert nr==pickle.dumps(np.random.get_state())and py==random.getstate()
    assert torch.equal(tr,torch.get_rng_state())


def test_no_intervention_is_full_batch_bitexact():
    batch=np.arange(256*192*3,dtype=np.float32).reshape(256,192,3)
    before=batch.tobytes();bank=np.zeros((1,192,3),np.float32)
    apply_numeric_replacements(batch,None,None)
    assert batch.tobytes()==before
    apply_numeric_replacements(batch,np.full(128,-1,np.int32),bank)
    assert batch.tobytes()==before


def test_donors_require_every_point_of_complete_window():
    b,c,counts=legal_windows(np.r_[np.ones(192),np.zeros(192)].astype(bool))
    assert np.array_equal(b,[0])and np.array_equal(c,[192])
    assert counts[1]==191 and counts[-2]==1
