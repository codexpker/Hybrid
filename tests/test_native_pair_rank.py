import numpy as np
import torch

from hybrid_tsad.native_pair_rank import NativePairRank, rank_loss, window_features


def test_features_have_strict_history_and_one_current_native_component():
    x = np.random.default_rng(2025).normal(size=(2, 100, 4)).astype(np.float32)
    variance = np.ones((2, 4), dtype=np.float32)
    features = window_features(x, variance)
    assert features.shape == (2, 4, 7) and features.dtype == np.float32
    x[:, -1] += 30
    np.testing.assert_array_equal(window_features(x, variance), features)
    other = window_features(x, 2 * variance)
    np.testing.assert_array_equal(other[:, :, 1:], features[:, :, 1:])
    assert np.all(other[:, :, 0] > features[:, :, 0])


def test_rank_symmetry_and_contamination_mixture_identity():
    p = torch.tensor([-0.2, 1.0, 2.5], dtype=torch.float64)
    n = torch.tensor([-0.8, 0.1], dtype=torch.float64)
    pp = rank_loss(p[:, None].expand(3, 3), p[None].expand(3, 3), symmetric=True).mean()
    torch.testing.assert_close(pp, torch.tensor(0.5, dtype=pp.dtype))
    pn = rank_loss(p[:, None].expand(3, 2), n[None].expand(3, 2), symmetric=True).mean()
    u = torch.cat((p, n))
    pu = rank_loss(p[:, None].expand(3, 5), u[None].expand(3, 5), symmetric=True).mean()
    torch.testing.assert_close(pu, 0.4 * pn + 0.3)
    q = torch.zeros_like(p, requires_grad=True)
    for symmetric in (True, False):
        loss = rank_loss(q, torch.zeros_like(q), symmetric=symmetric).sum()
        torch.testing.assert_close(torch.autograd.grad(loss, q)[0], torch.full_like(q, -0.25))


def test_head_preserves_permutation_and_trains_with_heterogeneous_channels():
    torch.manual_seed(2025)
    head = NativePairRank()
    x = torch.randn(4, 38, 7)
    torch.testing.assert_close(head(x), head(x[:, torch.randperm(x.shape[1])]))
    p, n = head(x), head(x[:, :25])
    rank_loss(p, n, symmetric=True).mean().backward()
    assert sum(param.numel() for param in head.parameters()) == 1137
    assert any(param.grad.abs().max() > 0 for param in head.parameters())
