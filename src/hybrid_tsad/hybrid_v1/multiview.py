"""Actual LEFT MS filter/decoder and mask-aware selected-link interaction."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleView(nn.Module):
    def __init__(self, config, sources):
        super().__init__()
        self.config = config
        tools, utils = sources.left()
        self.author = tools.MS_Utils(config.ms_factors)
        self.patch_embedding = utils.PatchEmbedding(config.width, config.temporal_patch,
            config.temporal_patch, config.temporal_patch - 1, config.dropout)
        self.scale_embedding = nn.Parameter(torch.zeros(len(config.ms_factors), config.width))

    def forward(self, student, visible):
        b, h, c = student.shape
        scales = self.author(student)
        all_tokens, masks, lengths = [], [], []
        for i, scale in enumerate(scales):
            tokens, _ = self.patch_embedding(scale.transpose(1, 2))
            # Filtered values are global prefix functions; validity additionally
            # requires real visible support, not nonzero imputed values.
            coarse = F.interpolate(visible.transpose(1, 2).float(), size=scale.shape[1], mode="nearest")
            coarse = F.pad(coarse, (0, self.config.temporal_patch - 1), mode="replicate")
            valid = coarse.unfold(-1, self.config.temporal_patch, self.config.temporal_patch).any(-1)
            valid = valid.reshape(b * c, -1) & (visible.sum(1).reshape(b * c, 1) >= 2)
            tokens = tokens + self.scale_embedding[i]
            all_tokens.append(tokens.masked_fill(~valid[..., None], 0))
            masks.append(valid)
            lengths.append(tokens.shape[1])
        return torch.cat(all_tokens, 1), torch.cat(masks, 1), {
            "lengths": lengths, "factors": list(self.config.ms_factors),
            "time_lengths": [s.shape[1] for s in scales], "original_history": h,
            "axis": "concatenated_coarse_to_fine_scale_patches"}


def masked_pool(tokens, valid):
    return (tokens * valid[..., None]).sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)


class SelectedFusion(nn.Module):
    def __init__(self, config, sources):
        super().__init__()
        self.mode = config.fusion
        tools, _ = sources.left()
        self.author = tools.MultiViewFusion(config.width, config.heads, 4 * config.width,
            config.dropout, "gelu", config.fusion_layers)
        self.concat_linear = nn.Linear(3 * config.width, config.width)

    @staticmethod
    def _block(block, query, key, qvalid, kvalid):
        result = query.masked_fill(~qvalid[..., None], 0)
        active = kvalid.any(-1) & qvalid.any(-1)
        if bool(active.any()):
            # PyTorch key_padding_mask True=blocked. Real author block/FFN is
            # used; all-empty rows never enter its softmax.
            update = block(query[active], key[active], key_padding_mask=~kvalid[active])
            result = result.clone()
            result[active] = update.masked_fill(~qvalid[active, :, None], 0)
        return result

    def forward(self, time, fractional, multiscale, time_valid, fractional_valid, ms_valid):
        if self.mode == "left_interaction":
            for i in range(self.author.n_layers):
                time = self._block(self.author.t_from_f[i], time, fractional, time_valid, fractional_valid)
                fractional = self._block(self.author.f_from_t[i], fractional, time, fractional_valid, time_valid)
                multiscale = self._block(self.author.m_from_t[i], multiscale, time, ms_valid, time_valid)
        elif self.mode in ("feature_mean", "concat_linear"):
            pooled = [masked_pool(t, m) for t, m in ((time, time_valid), (fractional, fractional_valid), (multiscale, ms_valid))]
            if self.mode == "feature_mean":
                flags = torch.stack([m.any(-1) for m in (time_valid, fractional_valid, ms_valid)], -1)
                context = (torch.stack(pooled, -2) * flags[..., None]).sum(-2) / flags.sum(-1, keepdim=True).clamp_min(1)
            else:
                context = self.concat_linear(torch.cat(pooled, -1))
            time = time + context[:, None]
            fractional = fractional + context[:, None]
            multiscale = multiscale + context[:, None]
        return tuple(t.masked_fill(~m[..., None], 0) for t, m in
                     ((time, time_valid), (fractional, fractional_valid), (multiscale, ms_valid)))


def decode_scales(decoder, tokens, metadata, config):
    """Shared real MsDecoder, each scale upsampled to full H patch grid."""
    outputs = []
    n = math.ceil(config.history / config.temporal_patch)
    for scale in tokens.split(metadata["lengths"], dim=1):
        up = F.interpolate(scale.transpose(1, 2), size=n, mode="nearest").transpose(1, 2)
        decoded, _, _ = decoder(up, context=None)
        outputs.append(decoded[:, :config.history])
    return torch.stack(outputs, 1)  # BC,S,H
