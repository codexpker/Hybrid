"""Frozen TimesFM tokens condition native LEFT before its three-view fusion."""

from pathlib import Path

import torch
from torch import nn

from . import left_multisource as base
from . import left_smd_source as smd

OUT = Path("results/left_timesfm_hidden_20261126_v1")
WIDTHS = {"PSM": 25, "SWAT": 31, "SMD": 38}
ARMS = ("A", "B", "C", "D", "E")
HISTORY, WINDOW = 512, 192


class HistoricalCondition(nn.Module):
    def __init__(self, dim, statistics=False):
        super().__init__()
        self.statistics = statistics
        self.project = nn.Linear(2 if statistics else 1280, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0.0, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, tokens, memory, valid):
        batch, channels, patches, _ = memory.shape
        assert tokens.shape[0] == batch * channels
        memory = self.memory_norm(self.project(memory)).reshape(
            batch * channels, patches, -1
        )
        mask = ~valid.reshape(batch * channels, patches)
        assert (~mask).any(-1).all()
        context = self.attention(
            self.query_norm(tokens),
            memory,
            memory,
            key_padding_mask=mask,
            need_weights=False,
        )[0]
        return tokens + self.gate.tanh() * context


class ConditionalLEFT(nn.Module):
    def __init__(self, left, arm):
        super().__init__()
        self.left = left
        self.arm = arm
        self.condition = (
            HistoricalCondition(left.d_model, arm == "B") if arm != "A" else None
        )
        self._memory = self._valid = None
        if self.condition is not None:
            # The encoder output has (batch * channels, multiscale tokens, dim).
            # Its conditioned output feeds the unchanged three-view fusion.
            self.left.encoder.register_forward_hook(self._inject)

    def _inject(self, module, args, output):
        assert self._memory is not None and self._valid is not None
        tokens, attention = output
        return self.condition(tokens, self._memory, self._valid), attention

    def set_history(self, history, valid_lengths, features=None, valid_patches=None):
        if self.arm == "A":
            return
        if self.arm == "B":
            positions = torch.arange(history.shape[1], device=history.device)
            keep = positions[None, :] >= history.shape[1] - valid_lengths[:, None]
            n = valid_lengths[:, None].to(history.dtype)
            mu = (history * keep[..., None]).sum(1) / n
            variance = ((history - mu[:, None]).square() * keep[..., None]).sum(1) / n
            self._memory = torch.stack((mu, variance), -1).unsqueeze(2)
            self._valid = torch.ones(
                self._memory.shape[:3], device=history.device, dtype=torch.bool
            )
        else:
            assert features is not None and valid_patches is not None
            assert features.shape[:2] == (history.shape[0], history.shape[2])
            assert features.shape[:3] == valid_patches.shape
            self._memory, self._valid = features.detach(), valid_patches

    def clear_history(self):
        self._memory = self._valid = None

    def forward(
        self, current, history, valid_lengths, features=None, valid_patches=None
    ):
        self.set_history(history, valid_lengths, features, valid_patches)
        try:
            return self.left(current, None, None, None)
        finally:
            self.clear_history()

    @torch.no_grad()
    def infer(self, current, history, valid_lengths, features=None, valid_patches=None):
        self.set_history(history, valid_lengths, features, valid_patches)
        try:
            return self.left.infer(current, None, None, None)
        finally:
            self.clear_history()


def initial(w, domain, arm):
    left = smd.initialize(w) if domain == "SMD" else base.initialize(w, domain)
    parent = smd.restore(w, left) if domain == "SMD" else base.restore(w, left, domain)
    # All arms keep the native dropout sequence. C/D/E condition layers share initialization.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model = ConditionalLEFT(left, arm).cuda()
    return model, parent


def load_model(w, domain, arm):
    folder = w / OUT / domain / arm
    meta = base.read(folder / "receipt.json")
    path = folder / "selected.pt"
    assert base.sha(path) == meta["sha256"]
    model, parent = initial(w, domain, arm)
    assert parent["sha256"] == meta["parent"]
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state"])
    model.left.step = checkpoint["step"]
    return model.eval(), meta
