import numpy as np
import pytest

from hybrid_tsad.timesfm3_cached_attribution import (
    confirmation, fragments, phase_bounds, positions, concentration,
    quantile_descriptors, describe, group_name)


def test_original_confirmation_never_backfills_or_bridges_gaps():
    index=np.array([10,11,12,13,18,19,20])
    bits=np.ones(7,bool)
    np.testing.assert_array_equal(confirmation(index,bits),[0,0,1,1,0,0,1])
    ff=fragments(index,np.array([1,1,0,1,1,0,1],bool))
    assert [index[p].tolist() for p in ff]==[[10,11],[13],[18],[20]]


def test_phase_bounds_keep_short_steady_na_and_avoid_neighbour_events():
    b=phase_bounds(1000,1100,980,1110)
    assert b['pre_normal']==(980,1000)
    assert b['onset_32']==(1000,1032)
    assert b['history_transition_32_512']==(1032,1100)
    assert b['steady_after512']==(1100,1100)
    assert b['recovery_32']==(1100,1110)
    assert b['recovery_32_512']==(1110,1110)
    b=phase_bounds(1000,1700,0,3000)
    assert b['steady_after512']==(1512,1700)


def test_positions_clip_to_cached_axis_and_use_normal_labels_only_posthoc():
    index=np.array([100,101,105,106,107]);labels=np.zeros(200,int);labels[105]=1
    np.testing.assert_array_equal(positions(index,(90,106)),[0,1,2])
    np.testing.assert_array_equal(positions(index,(90,106),labels,True),[0,1])
    with pytest.raises(ValueError):positions(index,(106,90))


def test_error_contribution_zero_total_is_missing_not_uniform():
    e=np.array([[8.,1.,1.],[1.,1.,1.],[0.,0.,0.]])
    c,top,valid=concentration(e)
    np.testing.assert_allclose(c['top1_error_share'][:2],[.8,1/3])
    np.testing.assert_allclose(c['effective_error_channels'][:2],[100/66,3])
    np.testing.assert_array_equal(c['channels_for80_error'][:2],[1,3])
    assert c['one_channel_at_least80_error_fraction'][0]==1
    assert np.isnan(c['top1_error_share'][2]) and not valid[2]
    assert top[1]==0  # Deterministic first-index tie handling, not physical attribution.


def test_nine_quantile_positions_include_equal_endpoints_and_zero_width():
    q=np.repeat(np.arange(1,10,dtype=np.float32)[None,None,:],4,axis=1)
    x=np.array([[0.,1.,9.,10.]],np.float32)
    f=quantile_descriptors(x,q)
    assert f['below_q10_fraction'][0]==.25 and f['above_q90_fraction'][0]==.25
    assert f['inside_q10_q90_fraction'][0]==.5
    assert f['actual_eq_q10_fraction'][0]==.25
    assert f['strict_quantiles_below_actual_0_fraction'][0]==.5
    assert f['strict_quantiles_below_actual_8_fraction'][0]==.25
    assert f['strict_quantiles_below_actual_9_fraction'][0]==.25
    assert sum(f[f'strict_quantiles_below_actual_{j}_fraction'][0]for j in range(10))==1
    z=quantile_descriptors(np.zeros((2,3)),np.zeros((2,3,9)))
    np.testing.assert_array_equal(z['zero_width80_fraction'],1)
    np.testing.assert_array_equal(z['inside_q10_q90_fraction'],1)


def test_empty_group_retains_missing_statistics_and_original_group_semantics():
    r=describe({'prediction_mse':np.array([1.,2.]),'inside_fraction':np.array([.5,1.])},np.array([],int))
    assert r['points']==0 and r['prediction_mse_median'] is None
    assert r['inside_fraction_mean'] is None
    assert group_name(True,False)=='TimesFM_only'
    assert group_name(False,True)=='TimeRadar_only'
    assert group_name(True,True)=='both'
    assert group_name(False,False)=='neither'
