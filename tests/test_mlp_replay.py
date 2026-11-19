import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from hybrid_tsad.mlp_replay import mapped_windows,batch_from_plan


def test_original_window_endpoints_are_preserved_in_ten_point_suffixes():
    plan=np.tile(np.arange(128,dtype=np.int64),(2000,1))
    start,end=mapped_windows(plan,[495000,585000])
    assert np.array_equal(start,plan+182)
    assert np.array_equal(end,495000+start+9)
    assert start.min()==182 and start.max()+10<=90000


def test_no_resampling_or_second_scaling_and_shared_target_slots():
    raw_t=np.arange(3000*3,dtype=np.float32).reshape(3000,3)
    raw_s=np.sin(raw_t/100).astype(np.float32)
    scaler=StandardScaler().fit(raw_s[:2000])
    values=dict(T=scaler.transform(raw_t),S=scaler.transform(raw_s))
    plans={k:np.tile(np.arange(128,dtype=np.int64)+offset,(2000,1))
           for k,offset in [('T_starts',0),('T_prime_starts',10),('S_starts',20)]}
    state=np.random.get_state()
    tt=batch_from_plan(values,plans,'TT',0);r=batch_from_plan(values,plans,'R',0)
    assert np.array_equal(tt[:128],r[:128])
    assert np.array_equal(tt[0],values['T'][182:192])
    assert np.array_equal(tt[128],values['T'][192:202])
    assert np.array_equal(r[128],values['S'][202:212])
    assert np.array_equal(np.random.get_state()[1],state[1])


def test_mapping_refuses_an_original_window_outside_its_role():
    with pytest.raises(AssertionError):mapped_windows(np.full((2000,128),809,np.int64),[100,1100])
