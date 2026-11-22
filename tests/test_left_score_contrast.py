import math

import numpy as np
import torch

from hybrid_tsad.left_score_contrast import (
    ANGLE, CORE, adam_displacement, contrast_parts, probe_windows, rotation_basis, select_pair,
)


def test_rotation_pairing_and_additive_cancellation():
    basis = torch.from_numpy(rotation_basis())
    torch.testing.assert_close(basis.T @ basis, torch.eye(2, dtype=basis.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(basis.sum(dim=0), torch.zeros(2, dtype=basis.dtype), atol=1e-14, rtol=0)
    fixture = torch.tensor([[2.0, 1.5, -0.7], [1.0, 0.8, 0.5]], dtype=basis.dtype)
    original = (basis @ fixture).unsqueeze(0) + 0.75
    assert select_pair(original.numpy()[0])[0] == (0, 1)
    arms = probe_windows(original, (0, 1))
    torch.testing.assert_close(arms.mean(dim=2), original.expand_as(arms).mean(dim=2))
    torch.testing.assert_close(arms.square().sum(dim=2), original.expand_as(arms).square().sum(dim=2))
    assert torch.equal(arms[:, :, :CORE.start], original[:, :CORE.start].expand_as(arms[:, :, :CORE.start]))
    assert torch.equal(arms[:, :, CORE.stop:], original[:, CORE.stop:].expand_as(arms[:, :, CORE.stop:]))
    assert torch.equal(arms[:, :, :, 2], original[:, :, 2].expand_as(arms[:, :, :, 2]))
    for i, j, c in ((0, 2, 0), (1, 3, 0), (0, 3, 1), (1, 2, 1)):
        assert torch.equal(arms[i, :, :, c], arms[j, :, :, c])
    energy = (arms - original).square().sum(dim=2)
    torch.testing.assert_close(energy[:, 0, :2], (4 * math.sin(ANGLE/2)**2 * fixture[:, :2].square().sum(0)).expand(4,2))
    additive = arms[:, :, :, 0].pow(4).sum(dim=-1) + arms[:, :, :, 1].sin().sum(dim=-1)
    assert contrast_parts(additive)[2].abs().max() < 1e-12
    cross = (arms[:, :, :, 0] * arms[:, :, :, 1]).sum(dim=-1)
    assert abs(contrast_parts(cross)[2].item() - (math.cos(2*ANGLE)-1) * (fixture[:, 0] @ fixture[:, 1])) < 1e-12


def test_adam_displacement_and_noop_keep_native_moments():
    def step(mode):
        parameter = torch.nn.Parameter(torch.tensor([0.5, 0.3], dtype=torch.float64))
        parameter.grad = parameter.new_tensor([1.0, 1.0])
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        kwargs = {"normal": (parameter.new_tensor([1.0 if mode == "conflict" else -1.0, 0.0]),)}
        if mode == "native":
            optimizer.step()
            stats = None
        else:
            stats = adam_displacement(optimizer, (parameter,), **kwargs)
        return parameter.detach(), optimizer.state[parameter], stats
    base, base_state, _ = step("native")
    nonconflict, normal_state, stats = step("nonconflict")
    assert torch.equal(base, nonconflict) and not stats["triggered"]
    conflict, conflict_state, stats = step("conflict")
    assert stats["triggered"] and stats["dot_applied"] == 0
    assert conflict[0] == 0.5 and conflict[1] == base[1]
    assert stats["length_ratio"] == math.sqrt(0.5)
    for key in base_state:
        assert torch.equal(base_state[key], conflict_state[key]) and torch.equal(base_state[key], normal_state[key])
    assert int(base_state["step"]) == 1


def test_length_control_scales_its_own_proposal_and_advances_once():
    initial = torch.tensor([0.5, 0.3], dtype=torch.float64)
    ordinary = torch.nn.Parameter(initial.clone())
    control = torch.nn.Parameter(initial.clone())
    base_optimizer = torch.optim.Adam([ordinary], lr=1e-4)
    control_optimizer = torch.optim.Adam([control], lr=1e-4)
    ordinary.grad = ordinary.new_tensor([1.0, -2.0])
    control.grad = ordinary.grad.clone()
    base_optimizer.step()
    ratio = 0.75
    stats = adam_displacement(control_optimizer, (control,), ratio=ratio)
    torch.testing.assert_close(control - initial, (ordinary - initial) * ratio, atol=1e-14, rtol=0)
    assert stats["replayed_length_ratio"] == ratio
    for key, value in base_optimizer.state[ordinary].items():
        assert torch.equal(value, control_optimizer.state[control][key])
