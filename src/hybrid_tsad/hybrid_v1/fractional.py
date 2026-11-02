"""Explicit split of pinned author FRFT forward, with native complex decoding.

Splitting and signed-real readout are project adaptations. Raw author_abs
encode/decode parity is separately testable without the public-D projection.
"""
from types import SimpleNamespace
import torch
from torch import nn


class FractionalBackbone(nn.Module):
    def __init__(self, config, sources):
        super().__init__()
        self.config = config
        official = sources.timeradar()
        self.author = official.Model(SimpleNamespace(seq_len=config.local_window,
            hidden_dim=config.tf_hidden, d_model=config.tf_repr, depth=config.tf_depth,
            patch_len=config.tf_patch, mask_mode="c"))
        self.frft, self.ifrft = official.frft, official.ifrft
        self.project = nn.Linear(2 * config.tf_repr, config.width)

    def encode(self, student, visible, *, mask_seed=None, copy_masks=None):
        cfg = self.config
        x, v = student[:, -cfg.local_window:], visible[:, -cfg.local_window:]
        b, w, c = x.shape
        expected = (b, cfg.copies, c, w // cfg.tf_patch)
        if copy_masks is not None:
            if copy_masks.shape != expected or copy_masks.dtype != torch.bool or copy_masks.device != x.device:
                raise ValueError("Explicit FRFT copy masks require bool [B,K,C,N] on input device")
            if not torch.equal(copy_masks[:, :cfg.copies // 2], ~copy_masks[:, cfg.copies // 2:]):
                raise ValueError("FRFT copy masks must retain author's complementary halves")
        native, orders, masks = [], [], []
        is_cuda = x.is_cuda
        devices = [x.device.index] if is_cuda else []
        # All state required for decode is explicit, local to this invocation.
        with torch.random.fork_rng(devices=devices, enabled=mask_seed is not None):
            if mask_seed is not None:
                torch.manual_seed(int(mask_seed))
            for bi, (window, validity) in enumerate(zip(x, v)):
                input_values = window.transpose(0, 1)  # C,W
                valid_channel = validity.sum(0) >= 2
                order_input = input_values[valid_channel].mean(0) if bool(valid_channel.any()) else input_values.new_zeros(w)
                order = (self.author.net(order_input).squeeze() if cfg.fixed_order is None
                         else input_values.new_tensor(float(cfg.fixed_order)))
                transformed = self.frft(input_values.to(torch.complex64), order, dim=1)
                real = self.author.real_input_embed(transformed.real.unfold(-1, cfg.tf_patch, cfg.tf_patch))
                imag = self.author.imag_input_embed(transformed.imag.unfold(-1, cfg.tf_patch, cfg.tf_patch))
                n = real.shape[1]
                if copy_masks is None:
                    first = torch.rand((cfg.copies // 2) * c, n, device=x.device) < .5
                    mask = torch.cat((first, ~first), 0)
                else:
                    mask = copy_masks[bi].reshape(cfg.copies * c, n)
                real = real.unsqueeze(0).expand(cfg.copies, -1, -1, -1).reshape(cfg.copies * c, n, -1).masked_fill(mask[..., None], 0)
                imag = imag.unsqueeze(0).expand(cfg.copies, -1, -1, -1).reshape(cfg.copies * c, n, -1).masked_fill(mask[..., None], 0)
                representation = self.author.encoder(real, imag, order)
                native.append(representation.reshape(cfg.copies, c, n, cfg.tf_repr))
                orders.append(order)
                masks.append(mask.reshape(cfg.copies, c, n))
        native = torch.stack(native)
        tokens = self.project(torch.cat((native.real, native.imag), -1))
        valid = (v.sum(1) >= 2)[:, None, :, None].expand(b, cfg.copies, c, tokens.shape[-2])
        return {"native": native, "tokens": tokens.masked_fill(~valid[..., None], 0),
                "valid": valid, "order": torch.stack(orders), "transform_hidden_mask": torch.stack(masks),
                "axis": "native_fractional_domain_patch_not_raw_time", "local_start_offset": student.shape[1] - w,
                "inverse_operator": "author_fast_ifrft", "exact_inverse": False}

    def decode(self, native, order, *, projection=None):
        b, k, c, n, d = native.shape
        cfg = self.config
        if (k, n, d) != (cfg.copies, cfg.local_window // cfg.tf_patch, cfg.tf_repr):
            raise ValueError("Native FRFT axes changed")
        outputs = []
        for bi in range(b):
            transformed = self.author.decoder(native[bi].reshape(k * c, n, d))
            raw = self.ifrft(transformed, order[bi], dim=-1)
            mode = cfg.reconstruction_projection if projection is None else projection
            if mode not in ("real", "author_abs"):
                raise ValueError("Unknown inverse projection")
            values = raw.real if mode == "real" else raw.abs()
            outputs.append(values.reshape(k, c, cfg.local_window).transpose(1, 2))
        return torch.stack(outputs)  # B,K,W,C
