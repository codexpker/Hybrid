import numpy as np
import torch
from hybrid_tsad.native_v_replay import sampling_plan,pool_starts,paired_starts,optimizer,input_windows


def test_paired_streams_and_nonoverlap_role_pools():
    source=pool_starts([0,105984]);target=pool_starts([0,17568])
    assert len(source)==1059 and len(target)==175
    assert source[-1]+100<=105984 and target[-1]+100<=17568
    plan=sampling_plan(len(source),len(target));again=sampling_plan(len(source),len(target))
    assert all(np.array_equal(plan[k],again[k])for k in plan)
    for step in [0,1999]:
        r=paired_starts(plan,source,target,'R',step);tt=paired_starts(plan,source,target,'TT',step);ss=paired_starts(plan,source,target,'SS',step)
        np.testing.assert_array_equal(r[0][1],tt[0][1]);np.testing.assert_array_equal(r[1][1],ss[1][1])
        assert r[0][0]=='U_FIT'and r[1][0]=='SOURCE_FIT'
    assert not np.array_equal(plan['T'],plan['T_prime'])and not np.array_equal(plan['S'],plan['S_prime'])


def test_declared_windows_are_real_and_never_cross_role():
    a=np.arange(202*2,dtype=np.float32).reshape(202,2)
    windows=list(input_windows(a,1000,[1000,1100],torch.device('cpu')))
    np.testing.assert_array_equal(windows[1][0].numpy(),a[100:200])
    import pytest
    with pytest.raises(PermissionError):list(input_windows(a,1000,[1103],torch.device('cpu')))


def test_optimizer_exact_original_head_grouping():
    m=torch.nn.Module();m.decoder=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.LayerNorm(3));m.other=torch.nn.Linear(2,2)
    m.other.requires_grad_(False);opt=optimizer(m)
    assert opt.defaults['betas']==(.9,.95)and opt.defaults['eps']==1e-8 and opt.defaults['lr']==1e-4
    decay_ids={id(p)for g in opt.param_groups if g['weight_decay']==.1 for p in g['params']}
    assert decay_ids=={id(m.decoder[0].weight)}
    assert not any(id(p)in {id(v)for v in m.other.parameters()}for g in opt.param_groups for p in g['params'])
