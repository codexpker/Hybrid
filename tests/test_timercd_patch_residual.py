import pytest
import torch
from torch import nn
from hybrid_tsad.timercd_patch_residual import PatchResidual


@pytest.mark.parametrize('mode', ['time', 'frft'])
def test_zero_output_disabled_and_gradients(mode):
    torch.manual_seed(2025)
    linear = nn.Linear(16, 24)
    layer = PatchResidual(linear, mode)
    x = torch.randn(2, 6, 16)
    assert torch.equal(linear(x), layer(x))
    opt = torch.optim.Adam(layer.parameters(), lr=.01)
    for _ in range(3):
        opt.zero_grad()
        layer(x).square().mean().backward()
        assert torch.isfinite(layer.project[-1].weight.grad).all()
        opt.step()
    assert layer.raw_order.grad.abs() > 0
    assert layer.project[0].weight.grad.abs().sum() > 0
    layer.enabled = False
    assert torch.equal(linear(x), layer(x))


def test_equal_capacity_and_batch_independence():
    torch.manual_seed(2025)
    a = PatchResidual(nn.Linear(16, 24), 'time')
    b = PatchResidual(nn.Linear(16, 24), 'frft')
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    nn.init.normal_(b.project[-1].weight)
    x = torch.randn(2, 7, 16)
    y = b(x).detach()
    x[1] *= 100
    assert torch.equal(y[0], b(x)[0])
    assert torch.allclose(b.order.detach(), torch.tensor(.5))


def test_padding_excluded_before_binary_labels():
    from hybrid_tsad.timercd_source_runner import loss_terms
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.anomaly_head = nn.Linear(1, 2)
            self.reconstruction_head = nn.Identity()
        def forward(self, time_series, mask):
            return time_series.unsqueeze(-1)
    model = Toy()
    x = torch.randn(1, 4, 2)
    batch = dict(time_series=x.clone(), masked_time_series=x.clone(),
                 labels=torch.tensor([[0, 1, -1, -1]]),
                 attention_mask=torch.tensor([[True, True, False, False]]),
                 mask=torch.tensor([[True, False, False, False]]))
    before = loss_terms(model, batch)
    batch['time_series'][:, 2:] = 1e6
    batch['masked_time_series'][:, 2:] = -1e6
    after = loss_terms(model, batch)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_fixed_sampling_and_mask_right_axis():
    from hybrid_tsad.timercd_source_runner import make_batch
    import numpy as np
    samples = [dict(time_series=np.arange(n*2).reshape(n,2), labels=np.zeros(n))
               for n in [49, 64]]
    a = make_batch(samples, [0,1], 'cpu', 2025, 16)
    torch.randn(100)
    b = make_batch(samples, [0,1], 'cpu', 2025, 16)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert not a['mask'][~a['attention_mask']].any()
    assert (a['labels'][~a['attention_mask']] == -1).all()
