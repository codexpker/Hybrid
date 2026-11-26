"""Source-trained history references for fixed ordered lag coordinates of LEFT windows."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from . import left_multisource as base
from . import left_smd_source as smd

OUT = Path("results/left_history_relation_20261126_v1")
LAGS = (0, 1, 2, 4, 8, 16, 32, 64)
WIDTHS = {"PSM": 25, "SWAT": 31, "SMD": 38}
HISTORY, CURRENT, PATCH = 256, 192, 32
BLOCKS = CURRENT // PATCH
RANK = 4
MIN_RMS = 1e-5


def correlations(x, ends):
    """Pearson r[block, lag, response i, driver j]: i[t] versus j[t-lag]."""
    offsets = ends[:, None] - (PATCH - 1) + torch.arange(PATCH, device=x.device)
    lags = torch.tensor(LAGS, device=x.device)
    resp = x[:, offsets]
    drivers = x[:, offsets[:, None, :] - lags[None, :, None]]
    resp = resp - resp.mean(-2, keepdim=True)
    drivers = drivers - drivers.mean(-2, keepdim=True)
    rn = torch.linalg.vector_norm(resp, dim=-2)
    dn = torch.linalg.vector_norm(drivers, dim=-2)
    eps = MIN_RMS * PATCH**0.5
    rn_ok, dn_ok = rn > eps, dn > eps
    resp = resp / rn.clamp_min(eps).unsqueeze(-2)
    drivers = drivers / dn.clamp_min(eps).unsqueeze(-2)
    corr = resp.transpose(-1, -2).unsqueeze(2) @ drivers
    valid = rn_ok[:, :, None, :, None] & dn_ok[:, :, :, None, :]
    return corr.clamp(-1, 1), valid


def mean_valid(values, valid, dims):
    count = valid.sum(dims)
    return (values * valid).sum(dims) / count.clamp_min(1), count > 0


def group_errors(prediction, truth, valid):
    """One MSE per block and relation group; a missing group has zero mass."""
    squared = (prediction - truth).square()
    own = squared.diagonal(dim1=-2, dim2=-1)[:, :, 1:]
    own_ok = valid.diagonal(dim1=-2, dim2=-1)[:, :, 1:]
    cross_ok = valid & ~torch.eye(
        valid.shape[-1], dtype=torch.bool, device=valid.device
    )
    a, ok_a = mean_valid(own, own_ok, (-2, -1))
    b, ok_b = mean_valid(squared, cross_ok, (-3, -2, -1))
    both, ok = mean_valid(torch.stack((a, b), -1), torch.stack((ok_a, ok_b), -1), -1)
    return {"self": (a, ok_a), "cross": (b, ok_b), "relation": (both, ok)}


class RelationHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        n = BLOCKS * len(LAGS) * RANK
        self.response = nn.Linear(dim, n)
        self.driver = nn.Linear(dim, n)
        self.autocorrelation = nn.Linear(dim, BLOCKS * (len(LAGS) - 1))

    def forward(self, z):
        b, c, _ = z.shape

        def project(layer):
            return (
                layer(z).reshape(b, c, BLOCKS, len(LAGS), RANK).permute(0, 2, 3, 1, 4)
            )

        pred = (
            project(self.response) @ project(self.driver).transpose(-2, -1)
        ) * RANK**-0.5
        own = (
            self.autocorrelation(z)
            .reshape(b, c, BLOCKS, len(LAGS) - 1)
            .permute(0, 2, 3, 1)
        )
        # Zero lag on the diagonal is deliberately excluded by group_errors.
        diag = pred.diagonal(dim1=-2, dim2=-1)
        pred = pred + torch.diag_embed(F.pad(own, (0, 0, 1, 0)) - diag)
        return pred.tanh()


class HistoryLEFT(nn.Module):
    def __init__(self, left, joint):
        super().__init__()
        self.left = left
        self.head = RelationHead(left.d_model) if joint else None

    def predict(self, history):
        b, h, c = history.shape
        assert h == HISTORY and self.head is not None
        patches = history.transpose(1, 2).reshape(b * c, HISTORY // PATCH, PATCH)
        # Fixed interval averages map 32 raw samples to the existing native
        # embedding width (6 or 16). No fitted history input projection.
        patches = F.adaptive_avg_pool1d(patches, self.left.patch_len)
        embed = self.left.patch_embedding
        tokens = embed.dropout(
            embed.value_embedding(patches)
        ) + self.left.pos_embedding(HISTORY // PATCH)
        features, _ = self.left.encoder(tokens, None)
        return self.head(features[:, -1].reshape(b, c, -1))


def components(model, x, with_persistence=True):
    assert x.shape[1] == HISTORY + CURRENT
    ends = torch.arange(HISTORY + PATCH - 1, HISTORY + CURRENT, PATCH, device=x.device)
    with torch.no_grad():
        truth, valid = correlations(x, ends)
    out = {}
    if model.head is not None:
        out.update(group_errors(model.predict(x[:, :HISTORY]), truth, valid))
    if with_persistence:
        reference, reference_ok = correlations(
            x[:, :HISTORY], ends.new_tensor([HISTORY - 1])
        )
        out["simple"] = group_errors(reference, truth, valid & reference_ok)["relation"]
    return out


def initial(w, d, joint):
    left = smd.initialize(w) if d == "SMD" else base.initialize(w, d)
    parent = smd.restore(w, left) if d == "SMD" else base.restore(w, left, d)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model = HistoryLEFT(left, joint).cuda()
    return model, parent


def load_model(w, domain, arm):
    dest = w / OUT / domain / ("continue" if arm == "source" else arm)
    meta = base.read(dest / "receipt.json")
    model, parent = initial(w, domain, arm == "joint")
    assert parent["sha256"] == meta["parent"]
    if arm != "source":
        path = dest / "selected.pt"
        assert base.sha(path) == meta["sha256"]
        ck = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["state"])
        model.left.step = ck["step"]
    return model.eval(), meta["parent_scales" if arm == "source" else "scales"]


@torch.no_grad()
def raw_batch(model, x):
    assert not model.training
    native = model.left.infer(x[:, HISTORY:].contiguous(), None, None, None)[0].mean(-1)
    return native, components(model, x)


@torch.no_grad()
def score_batch(model, x, scales):
    native, pieces = raw_batch(model, x)
    result = {"native": native}
    for name, (block, ok) in pieces.items():
        raw = block.repeat_interleave(PATCH, dim=1)
        result[name] = raw
        left = native / scales["native"]
        combo = (left + raw / scales[name]) * 0.5
        key = "joint" if name == "relation" else "native_" + name
        result[key] = torch.where(ok.repeat_interleave(PATCH, dim=1), combo, left)
    return result
