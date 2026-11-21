import json
from pathlib import Path
import numpy as np
from hybrid_tsad.controlled_complementarity_design import (
    envelope,delayed,energy,amplitude_control,variants,marginal_check,qualify,select_pair)


def config():return json.loads((Path(__file__).parents[1]/'configs/controlled-complementarity-design-20261121.json').read_text())


def test_delay_no_wrap_no_endpoint_seam_and_monotone():
    t=np.arange(2304,dtype=np.float32);base=np.stack([t,np.sin(t/11)],axis=1)
    out,phi=delayed(base,640,1024,32,128,[1])
    assert np.min(np.diff(phi))>0
    np.testing.assert_array_equal(out[:640],base[:640]);np.testing.assert_array_equal(out[1664:],base[1664:])
    np.testing.assert_array_equal(out[[640,1663]],base[[640,1663]])
    np.testing.assert_array_equal(out[768:1536,1],base[736:1504,1])
    np.testing.assert_array_equal(out[:,0],base[:,0])


def test_synchronous_sham_has_identical_entire_causal_and_confirmation_footprint():
    rng=np.random.default_rng(2);base=rng.normal(size=(2304,3)).astype(np.float32)
    cfg=config();a,b=cfg['H3']['primary_core']
    assert a-514>=640+128
    for lag in (8,32):
        out,_=delayed(base,640,1024,lag,128,range(3))
        np.testing.assert_array_equal(out[a-514:b],base[a-514-lag:b-lag])
        x,_=delayed(base,640,1024,lag,128,[1])
        assert not np.array_equal(x[a-514:b],base[a-514-lag:b-lag])


def test_amplitude_control_matches_energy_support_and_affected_variable():
    rng=np.random.default_rng(4);base=rng.normal(size=(2304,3)).astype(np.float32)
    out=base.copy();out[640:1664,1]+=np.asarray(.25*envelope(1024,64)*np.sin(2*np.pi*np.arange(1024)/16),np.float32)
    ctrl=amplitude_control(base,out,1,np.random.default_rng(8))
    assert abs(energy(ctrl,base)/energy(out,base)-1)<1e-5
    assert abs(energy(ctrl[1218:1600],base[1218:1600])/energy(out[1218:1600],base[1218:1600])-1)<1e-5
    np.testing.assert_array_equal(ctrl[:,[0,2]],base[:,[0,2]])
    np.testing.assert_array_equal(ctrl[:640],base[:640]);np.testing.assert_array_equal(ctrl[1664:],base[1664:])


def test_fixed_family_counts_and_distinct_controls_have_no_model():
    rng=np.random.default_rng(1);base=rng.normal(size=(2304,4)).astype(np.float32)
    cases=variants(base,1,config(),0)
    assert len(cases)==23
    assert sum(k=='H1' for _,k,_,_ in cases)==4
    assert sum(k=='H2' for _,k,_,_ in cases)==4
    assert sum(k=='H3' for _,k,_,_ in cases)==2
    assert sum(k=='amplitude_control' for _,k,_,_ in cases)==10


def test_marginal_identity_and_nonconstant_qualification():
    rng=np.random.default_rng(1);x=rng.normal(size=1024)
    m=marginal_check(x,x.copy());assert m['KS']==0 and m['max_ACF_change']==0 and m['std_ratio']==1
    assert qualify(m,config()['H3_qualification_full_and_plateau'])
    assert not qualify(marginal_check(x,x+1),config()['H3_qualification_full_and_plateau'])


def test_pair_selection_uses_fixed_source_stats_and_lexicographic_rule():
    rng=np.random.default_rng(7);x=rng.normal(size=1024)
    fit=np.stack([x,x+.01*rng.normal(size=1024),np.ones(1024)],axis=1)
    pair,columns,rows=select_pair(fit,config()['pair_qualification_SOURCE_FIT_only'])
    assert pair==(0,1)and columns==[0,1]and rows[0]['qualifies']
