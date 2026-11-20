import numpy as np
import torch
from hybrid_tsad.native_v_replay import sampling_plan,pool_starts,paired_starts,optimizer,input_windows,accumulate,reconstruction_components,training_seed


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


def test_two_half_accumulation_equals_mean_independent_window_objective():
    class TinyNative(torch.nn.Module):
        def __init__(self):
            super().__init__();self.decoder=torch.nn.Linear(2,2);self.net=torch.nn.Linear(2,1);self.net.requires_grad_(False)
        def forward(self,x,**kwargs):
            self.net(x.mean((0,1)))
            output=self.decoder(x)
            return None,output.unsqueeze(0).expand(10,-1,-1,-1)
    torch.manual_seed(2025);m=TinyNative().eval();x=torch.randn(4,100,2)
    expected=(m.decoder(x)-x).square().mean();expected.backward()
    gradients=[p.grad.clone()for p in m.decoder.parameters()];m.zero_grad(set_to_none=True)
    left,_=accumulate(m,(v[None]for v in x[:2]),torch.device('cpu'),4)
    right,_=accumulate(m,(v[None]for v in x[2:]),torch.device('cpu'),4)
    assert abs((left+right)/2-float(expected.detach()))<1e-6
    for p,g in zip(m.decoder.parameters(),gradients):torch.testing.assert_close(p.grad,g,rtol=1e-6,atol=1e-7)


def test_copy_mean_variance_loss_and_gradient_identity():
    torch.manual_seed(2025)
    T=torch.randn(10,1,100,3,dtype=torch.float64,requires_grad=True)
    S=torch.randn(10,1,100,3,dtype=torch.float64,requires_grad=True)
    yT=torch.randn(1,100,3,dtype=torch.float64);yS=torch.randn(1,100,3,dtype=torch.float64)
    fullT,E_T,V_T=reconstruction_components(T,yT);fullS,E_S,V_S=reconstruction_components(S,yS)
    torch.testing.assert_close(fullT,E_T+.9*V_T,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(fullS,E_S+.9*V_S,rtol=1e-13,atol=1e-13)
    old=(fullT+fullS)/2;new=(E_T+fullS)/2
    torch.testing.assert_close(old-new,.45*V_T,rtol=1e-13,atol=1e-13)
    a=torch.autograd.grad(old,(T,S),retain_graph=True);b=torch.autograd.grad(new,(T,S),retain_graph=True)
    c=torch.autograd.grad(.45*V_T,T)[0]
    torch.testing.assert_close(a[0]-b[0],c,rtol=1e-12,atol=1e-14)
    torch.testing.assert_close(a[1]-b[1],torch.zeros_like(S),rtol=0,atol=0)
    assert torch.count_nonzero(b[0])>0


def test_seed2025_retains_original_stream_rule_and_new_seeds_pair():
    for source_count,target_count in [(3960,900),(1059,175)]:
        expected_T=np.random.default_rng(2025).integers(target_count,size=(2000,32))
        legacy=[np.random.default_rng(s)for s in np.random.SeedSequence(2025).spawn(3)]
        expected=dict(T=expected_T,S=legacy[0].integers(source_count,size=(2000,32)),T_prime=legacy[1].integers(target_count,size=(2000,32)),S_prime=legacy[2].integers(source_count,size=(2000,32)))
        actual=sampling_plan(source_count,target_count)
        assert all(np.array_equal(actual[k],expected[k])for k in expected)
        p26=sampling_plan(source_count,target_count,seed_value=2026);p27=sampling_plan(source_count,target_count,seed_value=2027)
        assert not np.array_equal(p26['T'],p27['T'])and not np.array_equal(p26['S'],p27['S'])


def test_actual_seed_is_parameterized_not_a_filename(monkeypatch):
    from hybrid_tsad.native_v_conditioning_runner import seed
    import pytest
    assert training_seed(dict(replay_training={}))==2025
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    state=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            draws=[]
            for value in [2026,2027]:
                chosen=training_seed(dict(replay_training=dict(seed=value)));seed(chosen)
                assert torch.initial_seed()==value;draws.append(torch.rand(4))
            assert not torch.equal(*draws)
    finally:np.random.set_state(state)
    with pytest.raises(ValueError):training_seed(dict(replay_training=dict(seed=True)))
