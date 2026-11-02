"""Small trainable configuration of real external TimesFM3 causal blocks."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TemporalEncoder(nn.Module):
    def __init__(self, config, sources):
        super().__init__()
        self.config = config
        configs, dense, transformer = sources.timesfm()
        self.embedding = dense.ResidualBlock(configs.ResidualBlockConfig(
            hidden_dims=2 * config.width, output_dims=config.width, use_bias=True, activation="swish"))
        # Author ResidualBlock is lazy by default; finish BEFORE optimizer creation.
        self.embedding.set_input_dims(2 * config.temporal_patch)
        layer = configs.TransformerConfig(model_dims=config.width, hidden_dims=4 * config.width,
            num_heads=config.heads, attention_norm="rms", feedforward_norm="rms", qk_norm="rms",
            use_bias=True, use_rope_seq=True, use_rope_var=False, ff_activation="swish",
            deterministic=True, causal_attention=True, training=True, use_sdpa=True)
        self.encoder = transformer.StackedMixingTransformer(
            configs.StackedTransformersConfig(config.temporal_layers, layer, use_remat=False),
            use_variate_attention=config.use_variate_attention)

    def forward(self, student, visible):
        b, h, c = student.shape
        p = self.config.temporal_patch
        pad = (-h) % p
        x = F.pad(student.transpose(1, 2), (0, pad))
        m = F.pad(visible.transpose(1, 2), (0, pad))
        xp, mp = x.unfold(-1, p, p), m.unfold(-1, p, p)
        valid = mp.any(-1)
        tokens = self.embedding(torch.cat((xp, mp.to(xp.dtype)), -1))
        tokens = tokens.masked_fill(~valid[..., None], 0)
        if self.config.temporal_enabled:
            tokens, _, masks = self.encoder(tokens, patch_mask=~valid)
        else:
            masks = []
        tokens = tokens.masked_fill(~valid[..., None], 0)
        ends = (torch.arange(math.ceil(h / p), device=x.device) + 1) * p - 1
        return tokens, valid, {"patch_end_offset": ends.clamp_max(h - 1),
                               "attention_true_means_visible": True, "attention_masks": masks}
