"""Complete V1 two-pass model. Targets/labels are never inputs to encoders."""
import torch
from torch import nn

from .config import HybridV1Config
from .provenance import OfficialSources
from .temporal import TemporalEncoder
from .fractional import FractionalBackbone
from .lag import FiniteLag
from .multiview import MultiScaleView, SelectedFusion, masked_pool, decode_scales


def visible_normalize(values, visible):
    """Student-only statistics: hidden GT/NaN/imputation never enter estimates."""
    safe = torch.where(visible, values, torch.zeros_like(values))
    count = visible.sum(1, keepdim=True)
    mean = (safe.sum(1, keepdim=True) / count.clamp_min(1)).detach()
    centered = torch.where(visible, safe - mean, torch.zeros_like(safe))
    variance = (centered.square().sum(1, keepdim=True) / count.clamp_min(1)).detach()
    scale = (variance + 1e-5).sqrt()
    valid = count >= 2
    student = torch.where(visible & valid, (safe - mean) / scale, torch.zeros_like(safe))
    return student, {"mean": mean, "scale": scale, "channel_valid": valid, "visible_count": count}


def last_valid(tokens, valid):
    pos = torch.arange(valid.shape[-1], device=valid.device)
    index = torch.where(valid, pos, -1).amax(-1)
    result = tokens.gather(-2, index.clamp_min(0)[..., None, None].expand(*index.shape, 1, tokens.shape[-1])).squeeze(-2)
    return result.masked_fill((index < 0)[..., None], 0), index


class HybridV1(nn.Module):
    def __init__(self, config, sources):
        super().__init__()
        self.config = config if isinstance(config, HybridV1Config) else HybridV1Config.from_dict(config)
        self.sources = sources if isinstance(sources, OfficialSources) else OfficialSources(**sources)
        cfg = self.config
        self.source_manifest = self.sources.manifest()
        self.tf_backbone = FractionalBackbone(cfg, self.sources)
        self.temporal = TemporalEncoder(cfg, self.sources)
        self.lag = FiniteLag(cfg)
        self.ms_filter = MultiScaleView(cfg, self.sources)
        self.fusion = SelectedFusion(cfg, self.sources)
        self.tf_fusion_delta = nn.Linear(cfg.width, 2 * cfg.tf_repr, bias=False)
        tools, _ = self.sources.left()
        self.ms_decoder = tools.MsDecoder(cfg.width, cfg.temporal_patch, 4 * cfg.width, cfg.dropout, "gelu")
        self.forecast_head = nn.Linear(cfg.width, cfg.horizon)
        self.event_head = nn.Linear(2 * cfg.width + 3, 1)

    def _validate(self, values, observed):
        if (values.ndim != 3 or values.shape[1] != self.config.history or min(values.shape[0], values.shape[2]) < 1
                or values.dtype != torch.float32):
            raise ValueError("Float32 independent [B,H,C] histories with configured H required")
        if observed is None:
            observed = torch.isfinite(values)
        if observed.shape != values.shape or observed.dtype != torch.bool or observed.device != values.device:
            raise ValueError("Observed mask must be bool and match input shape/device")
        if bool((observed & ~torch.isfinite(values)).any()):
            raise ValueError("Observed values must be finite; missing values are not GT")
        return observed

    def make_hidden_masks(self, observed, hide_mask=None, *, mask_seed=0):
        """Two complementary endpoint groups plus explicit/history-only masks.

        Group membership is returned and must travel with channel permutations.
        For C1 the empty second group is omitted, never duplicate-scored.
        """
        b, h, c = observed.shape
        g = min(2, c)
        groups = torch.arange(c, device=observed.device) % g
        endpoint = groups[None, None, :] == torch.arange(g, device=observed.device)[None, :, None]
        endpoint = endpoint.expand(b, -1, -1)
        hidden = torch.zeros(b, g, h, c, dtype=torch.bool, device=observed.device)
        if hide_mask is None and self.training:
            generator = torch.Generator(device=observed.device).manual_seed(int(mask_seed))
            p = self.config.temporal_patch
            n = (h + p - 1) // p
            patches = torch.rand(b, n, c, generator=generator, device=observed.device) < self.config.history_mask_probability
            additional = patches.repeat_interleave(p, dim=1)[:, :h]
            additional[:, -1] = False
            hidden |= additional[:, None]
        elif hide_mask is not None:
            if hide_mask.dtype != torch.bool or hide_mask.device != observed.device:
                raise ValueError("Hide mask must be boolean on input device")
            if hide_mask.shape == observed.shape:
                if bool(hide_mask[:, -1].any()):
                    raise ValueError("Common additional hide_mask is history-only; endpoint groups are constructed by core")
                hidden |= hide_mask[:, None]
            elif hide_mask.shape == hidden.shape:
                # Explicit masks can carry group identity through a channel
                # permutation. They must still form a complete disjoint cover.
                supplied_endpoint = hide_mask[:, :, -1]
                if bool(supplied_endpoint.any()):
                    if not bool((supplied_endpoint.sum(1) == 1).all()):
                        raise ValueError("Explicit endpoint groups must partition all channels exactly once")
                    endpoint = supplied_endpoint
                hidden |= hide_mask
            else:
                raise ValueError("Hide mask must be [B,H,C] or [B,G,H,C]")
        hidden[:, :, -1] = endpoint
        hidden &= observed[:, None]
        return hidden

    def _encode(self, values, observed, hidden, *, mask_seed, copy_masks=None):
        cfg = self.config
        visible = observed & ~hidden
        student, stats = visible_normalize(values, visible)
        b, h, c = student.shape
        time, tv, time_axes = self.temporal(student, visible)
        if cfg.self_lag_enabled or cfg.cross_lag_enabled:
            own, cross, lag_meta = self.lag(student, visible)
        else:
            own = cross = student.new_zeros(b, c, cfg.width)
            lag_meta = {"disabled": True, "circular": False}
        endpoint, endpoint_index = last_valid(time, tv)
        injection = own + cross
        marker = torch.nn.functional.one_hot(endpoint_index.clamp_min(0), time.shape[-2]).to(time.dtype)
        marker = marker * (endpoint_index >= 0)[..., None]
        time = time + marker[..., None] * injection[:, :, None]
        if cfg.tf_enabled:
            fractional = self.tf_backbone.encode(student, visible, mask_seed=mask_seed, copy_masks=copy_masks)
        else:
            n = cfg.local_window // cfg.tf_patch
            fractional = {"native": torch.zeros(b, cfg.copies, c, n, cfg.tf_repr, device=student.device, dtype=torch.complex64),
                "tokens": student.new_zeros(b, cfg.copies, c, n, cfg.width),
                "valid": torch.zeros(b, cfg.copies, c, n, device=student.device, dtype=torch.bool),
                "order": student.new_zeros(b), "transform_hidden_mask": torch.zeros(b, cfg.copies, c, n, device=student.device, dtype=torch.bool),
                "axis": "disabled_native_fractional_domain", "local_start_offset": h - cfg.local_window}
        if cfg.ms_enabled:
            ms, mv, ms_axes = self.ms_filter(student, visible)
        else:
            lengths = [((h + r - 1) // r + cfg.temporal_patch - 1) // cfg.temporal_patch for r in cfg.ms_factors]
            ms = student.new_zeros(b * c, sum(lengths), cfg.width)
            mv = torch.zeros(b * c, sum(lengths), device=student.device, dtype=torch.bool)
            ms_axes = {"lengths": lengths, "factors": list(cfg.ms_factors), "disabled": True, "original_history": h}
        k = cfg.copies
        base_f = fractional["tokens"]
        f0 = base_f.reshape(b * k * c, base_f.shape[-2], cfg.width)
        fv = fractional["valid"].reshape(b * k * c, -1)
        t = time[:, None].expand(-1, k, -1, -1, -1).reshape(b * k * c, -1, cfg.width)
        tm = tv[:, None].expand(-1, k, -1, -1).reshape(b * k * c, -1)
        ms0 = ms.reshape(b, c, ms.shape[-2], cfg.width)
        m = ms0[:, None].expand(-1, k, -1, -1, -1).reshape(b * k * c, -1, cfg.width)
        mm = mv.reshape(b, c, -1)[:, None].expand(-1, k, -1, -1).reshape(b * k * c, -1)
        t, f, m = self.fusion(t, f0, m, tm, fv, mm)
        t_end, _ = last_valid(t, tm)
        f_end = masked_pool(f, fv)
        t_end, f_end = (v.reshape(b, k, c, cfg.width) for v in (t_end, f_end))
        return {"stats": stats, "visible": visible, "student": student,
                "t_features": t_end, "f_features": f_end, "fractional": fractional,
                "fused_f": f.reshape(b, k, c, f.shape[-2], cfg.width), "f_base": base_f,
                "fused_ms": m, "ms_axes": ms_axes, "ms_valid": mm,
                "axes": {"time": time_axes, "fractional": fractional["axis"], "multiscale": ms_axes,
                         "lag": lag_meta, "time_endpoint_patch": endpoint_index}}

    def forecast(self, history, observed_mask=None, mask_seed=0, *, copy_masks=None):
        observed = self._validate(history, observed_mask)
        encoded = self._encode(history, observed, torch.zeros_like(observed), mask_seed=mask_seed, copy_masks=copy_masks)
        per_copy = self.forecast_head(encoded["t_features"]).permute(0, 1, 3, 2)
        normalized = per_copy.mean(1)  # Head applied to every copy BEFORE averaging.
        stats = encoded["stats"]
        valid = stats["channel_valid"].expand(-1, self.config.horizon, -1)
        return {"prediction": normalized * stats["scale"] + stats["mean"],
                "prediction_normalized": normalized, "prediction_by_copy_normalized": per_copy,
                "valid": valid, "stats": stats, "t_features": encoded["t_features"].mean(1),
                "f_features": encoded["f_features"].mean(1), "axes": encoded["axes"],
                "observed_mask": observed, "copy_masks": encoded["fractional"]["transform_hidden_mask"],
                "tf_order": encoded["fractional"]["order"],
                "pass": "past_only_forecast"}

    def reconstruct(self, context, observed_mask=None, hide_mask=None, mask_seed=0, *, copy_masks=None):
        observed = self._validate(context, observed_mask)
        hidden = self.make_hidden_masks(observed, hide_mask, mask_seed=mask_seed)
        cfg = self.config
        b, h, c = context.shape
        if copy_masks is not None and copy_masks.shape != (b, hidden.shape[1], cfg.copies, c, cfg.local_window // cfg.tf_patch):
            raise ValueError("Reconstruction copy_masks must be [B,G,K,C,N]")
        rows = []
        for group in range(hidden.shape[1]):
            e = self._encode(context, observed, hidden[:, group], mask_seed=int(mask_seed) + group,
                             copy_masks=None if copy_masks is None else copy_masks[:, group])
            native = e["fractional"]["native"]
            delta = self.tf_fusion_delta(e["fused_f"] - e["f_base"])
            real, imag = delta.chunk(2, -1)
            fused_native = native + torch.complex(real, imag)
            if cfg.tf_enabled:
                tf_normalized = self.tf_backbone.decode(fused_native, e["fractional"]["order"])
            else:
                tf_normalized = context.new_zeros(b, cfg.copies, cfg.local_window, c)
            if cfg.ms_enabled:
                ms_by_scale = decode_scales(self.ms_decoder, e["fused_ms"], e["ms_axes"], cfg)
                ms_by_scale = ms_by_scale.reshape(b, cfg.copies, c, len(cfg.ms_factors), h).permute(0, 1, 3, 4, 2)
            else:
                ms_by_scale = context.new_zeros(b, cfg.copies, len(cfg.ms_factors), h, c)
            mean, scale = e["stats"]["mean"], e["stats"]["scale"]
            tf = tf_normalized * scale[:, None] + mean[:, None]
            ms = ms_by_scale * scale[:, None, None] + mean[:, None, None]
            tf_ok = e["fractional"]["valid"].any(-1)[:, :, None, :].expand(-1, -1, cfg.local_window, -1)
            # MS normalization requires visible support and an enabled scale path.
            ms_ok = e["stats"]["channel_valid"][:, None, None].expand(-1, cfg.copies, len(cfg.ms_factors), h, -1)
            if not cfg.ms_enabled:
                ms_ok = torch.zeros_like(ms_ok)
            rows.append({"tf_reconstruction": tf, "tf_normalized": tf_normalized,
                         "ms_reconstruction_by_scale": ms, "ms_normalized_by_scale": ms_by_scale,
                         "tf_valid": tf_ok, "ms_valid_by_scale": ms_ok,
                         "t_features": e["t_features"], "f_features": e["f_features"],
                         "stats": e["stats"], "axes": e["axes"], "visible_mask": e["visible"],
                         "order": e["fractional"]["order"], "transform_hidden_mask": e["fractional"]["transform_hidden_mask"]})
        keys = ("tf_reconstruction", "tf_normalized", "ms_reconstruction_by_scale", "ms_normalized_by_scale",
                "tf_valid", "ms_valid_by_scale", "t_features", "f_features", "visible_mask", "order", "transform_hidden_mask")
        result = {key: torch.stack([r[key] for r in rows], 1) for key in keys}
        result["ms_reconstruction"] = result["ms_reconstruction_by_scale"].mean(3)
        result["ms_normalized"] = result["ms_normalized_by_scale"].mean(3)
        result["ms_valid"] = result["ms_valid_by_scale"].all(3)
        result["stats"] = {key: torch.stack([r["stats"][key] for r in rows], 1) for key in rows[0]["stats"]}
        own = hidden[:, :, -1].to(context.dtype)
        for key in ("t_features", "f_features"):
            features = result[key].mean(2)  # B,G,C,D
            result["endpoint_" + key] = (features * own[..., None]).sum(1)
        result.update(hidden_mask=hidden, observed_mask=observed, axes=[r["axes"] for r in rows],
                      pass_name="current_arrived_endpoint_hidden_reconstruction", group_count=len(rows),
                      normalization_scope="visible_student_only", reconstruction_projection=cfg.reconstruction_projection)
        return result

    def forward(self, history, observed_mask=None, mask_seed=0):
        return self.forecast(history, observed_mask, mask_seed)

    def parameter_counts(self):
        return {"stored": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)}
