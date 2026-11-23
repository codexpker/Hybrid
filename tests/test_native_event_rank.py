import numpy as np
import pytest
import torch
from torch.nn import functional as F

from hybrid_tsad.native_event_rank import event_pair_loss, source_event_roles, terminal_mean_error


def test_source_roles_keep_complete_events_and_normal_footprints():
    labels = np.zeros(900, dtype=np.int64)
    labels[120:270] = 1
    labels[530:680] = 1
    fit, select = source_event_roles(labels, 1000, fit_event_count=1)
    assert fit.bounds == (1000, 1400) and select.bounds == (1400, 1900)
    assert fit.events == ((1120, 1270),) and select.events == ((1530, 1680),)
    for role in (fit, select):
        for phases, (start, end) in zip(role.positive_phases, role.events):
            assert len(phases) == 3
            np.testing.assert_array_equal(np.concatenate(phases), np.arange(start + 99, end))
        for endpoint in role.normal_endpoints:
            assert role.bounds[0] <= endpoint - 99 <= endpoint < role.bounds[1]
            assert not labels[endpoint - 1099:endpoint - 999].any()
    labels[0] = 1
    with pytest.raises(ValueError, match="internally bounded"):
        source_event_roles(labels, 1000, fit_event_count=1)


def test_event_weighting_and_point_control_share_pairs_with_native_score_gradients():
    pos = torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]], requires_grad=True)
    normal = torch.full((2, 3), 0.2, requires_grad=True)
    phase_counts = torch.tensor([[20, 20, 20], [60, 60, 60]])
    margin = F.softplus(1 - torch.log1p(pos / 0.2) + torch.log1p(normal / 0.2))
    event_risk = event_pair_loss(pos, normal, 0.2)
    point_risk = event_pair_loss(pos, normal, 0.2, phase_sizes=phase_counts)
    torch.testing.assert_close(event_risk, margin.mean())
    torch.testing.assert_close(point_risk, (margin * phase_counts).sum() / phase_counts.sum())
    grad_pos, grad_normal = torch.autograd.grad(event_risk, (pos, normal))
    assert (grad_pos < 0).all() and (grad_normal > 0).all()
    with pytest.raises(ValueError, match="positive, fixed"):
        event_pair_loss(pos, normal, 0)


def test_terminal_error_averages_replicas_before_squaring_and_only_uses_current_point():
    window = torch.zeros(1, 100, 2)
    copies = torch.full((10, 1, 100, 2), 60.0)
    copies[:5, :, -1] = torch.tensor([1.0, 6.0])
    copies[5:, :, -1] = torch.tensor([-1.0, 2.0])
    copies.requires_grad_()
    loss = terminal_mean_error(copies, window)
    torch.testing.assert_close(loss, torch.tensor(8.0))
    loss.backward()
    assert (copies.grad[:, :, :-1] == 0).all()
    torch.testing.assert_close(copies.grad[:, 0, -1], torch.tensor([[0.0, 0.4]]).expand(10, -1))
