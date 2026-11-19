import numpy as np
import pytest
import torch
from hybrid_tsad.native_v_conditioning import (
    SimpleHistoryQuantiles, conditional_score, pinball_per_sample,
    raw_residual_score, fixed_native_rng,
)


def test_quantile_units_ties_floor_and_half_precision_boundary():
    v=np.array([4.,8.,3.]);q=np.array([[1.,2.,3.],[2.,2.,2.],[0.,.0001,.0002]],np.float32)
    r=conditional_score(v,q,2.)
    np.testing.assert_allclose(r['canonical_score'],(v-r['q_physical'][:,2])/r['denominator_physical'])
    assert r['canonical_score'].dtype==np.float64
    assert r['floor_active'].tolist()==[False,True,True]
    assert r['canonical_score'][1]==2000.
    with pytest.raises(ValueError):conditional_score(v,q[:,::-1],2.)


def test_zero_head_and_real_history_gradient_path():
    torch.manual_seed(2025);model=SimpleHistoryQuantiles(2,[.1,.5,.9])
    h=torch.randn(4,512,2);pred=model(h)
    torch.testing.assert_close(pred,torch.tensor([.1,.5,.9]).expand(4,3),rtol=0,atol=0)
    pinball_per_sample(pred,torch.arange(4.)).mean().backward()
    assert all(p.grad is None or not torch.count_nonzero(p.grad)for p in model.body.parameters())
    model.zero_grad();model.head.weight.data.normal_(0,.01)
    pinball_per_sample(model(h),torch.arange(4.)).mean().backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)for p in model.body.parameters())
    with pytest.raises(ValueError):model(h[:,:511])


def test_D_targets_actual_observation_and_not_native_V():
    pred=np.array([[1.,2.],[4.,5.]],np.float32);actual=np.array([[0.,2.],[1.,1.]],np.float32)
    residual,score=raw_residual_score(pred,actual)
    np.testing.assert_array_equal(residual,np.array([[1.,0.],[9.,16.]],np.float64))
    np.testing.assert_array_equal(score,np.array([.5,12.5]))
    assert score.dtype==np.float64


def test_native_mask_rng_isolated_even_exception():
    torch.manual_seed(2025);state=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with fixed_native_rng(torch.device('cpu')):
            first=torch.rand(10);raise RuntimeError('synthetic')
    assert torch.equal(state,torch.get_rng_state())
    with fixed_native_rng(torch.device('cpu')):second=torch.rand(10)
    assert torch.equal(first,second)and torch.equal(state,torch.get_rng_state())
