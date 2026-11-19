import torch
from torch import nn
from hybrid_tsad.conditional_response_energy import energy_score, ResponseGeometry, fixed_noise


def test_energy_unbiased_per_observation():
    x = torch.tensor([[[0.], [2.], [4.], [6.]], [[1.], [3.], [5.], [7.]]], requires_grad=True)
    y = torch.tensor([[0.], [2.]])
    expected = []
    for samples, target in zip(x, y):
        d = torch.linalg.vector_norm(samples - target, dim=-1).mean()
        ordered = sum(torch.linalg.vector_norm(a-b) for i,a in enumerate(samples) for j,b in enumerate(samples) if i != j)
        expected.append(d - ordered/24)
    actual = energy_score(x, y)
    torch.testing.assert_close(actual, torch.stack(expected))
    actual.mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


class Native(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor(2.))
    def forward(self, x, **kwargs):
        assert x.shape[0] == 1 and kwargs == dict(norm=0, mask_mode='c', copies=10)
        mask = torch.rand(5, 1, 100, x.shape[2]) > .5
        mask = torch.cat((mask, ~mask))
        return None, x[None] * self.weight * mask.to(x.dtype)


def test_response_rng_freeze_gradient_sample_independence():
    model = ResponseGeometry(Native(), 2.)
    h = torch.randn(2,512,3); y = torch.randn(2,3)
    v = torch.randn(2,4,3, requires_grad=True)
    before = torch.get_rng_state().clone()
    score = model.score(h,v,y)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(score[:1], model.score(h[:1],v[:1],y[:1]))
    score.mean().backward()
    assert v.grad.abs().sum() > 0 and model.native.weight.grad is None
    assert not model.native.weight.requires_grad
    out1 = model.response(torch.cat((h[0,-99:],y[0,None]))[None])
    out2 = model.response(torch.cat((h[0,-99:],y[0,None]))[None])
    assert torch.equal(out1,out2)


def test_noise_uses_only_explicit_stream():
    before = torch.get_rng_state().clone()
    a = fixed_noise(2,3,torch.Generator().manual_seed(2025),device='cpu')
    b = fixed_noise(2,3,torch.Generator().manual_seed(2025),device='cpu')
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert torch.equal(before, torch.get_rng_state())
