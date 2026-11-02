"""New endpoint forecasting candidate; external official TimeRadar stays read-only.

The lag module must be supplied by an explicit, provenance-bound adapter.
FRFT patches are pooled in their native domain, never assigned
raw-time coordinates. No datasets, checkpoints, labels or thresholds are loaded.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

OFFICIAL_SHA256 = "881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317"
MODES = ("tf", "lag", "concat", "gated")


@dataclass(frozen=True)
class CandidateConfig:
    history: int = 100
    horizon: int = 1
    width: int = 64
    mode: str = "gated"
    tf_hidden: int = 64
    tf_repr: int = 256
    tf_depth: int = 10
    patch_len: int = 5
    copies: int = 2

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("Unknown ablation mode")
        integers = (self.history, self.horizon, self.width, self.tf_hidden,
                    self.tf_repr, self.tf_depth, self.patch_len, self.copies)
        if any(type(v) is not int or v < 1 for v in integers):
            raise ValueError("Positive integer dimensions required")
        if self.history % self.patch_len or self.copies % 2:
            raise ValueError("Pinned upstream needs divisible history and even copies")


def load_official(path, config):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != OFFICIAL_SHA256:
        raise ValueError("Official TimeRadar byte identity mismatch")
    spec = importlib.util.spec_from_file_location("candidate_external_timeradar", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model(SimpleNamespace(seq_len=config.history,
        hidden_dim=config.tf_hidden, d_model=config.tf_repr, depth=config.tf_depth,
        patch_len=config.patch_len, mask_mode="c"))


class TFLagCandidate(nn.Module):
    """Four explicit ablations with one common future target and residual score."""
    def __init__(self, config, official_file=None, *, lag_encoder=None, lag_provenance=None):
        super().__init__()
        self.config = config
        if config.mode != "tf":
            if (not isinstance(lag_encoder, nn.Module) or not isinstance(lag_provenance, dict)
                    or not all(lag_provenance.get(k) for k in ("repository", "commit", "class", "source_sha256"))):
                raise ValueError("Explicit official lag adapter and provenance required; no automatic fallback")
        self.lag_provenance = lag_provenance
        self.tf = None
        self.lag = None
        if config.mode != "lag":
            if official_file is None:
                raise ValueError("External official file required for TF modes")
            self.tf = load_official(official_file, config)
            # Official forward computes this decoder, but the candidate objective
            # uses its returned representation only. Report it as inactive capacity.
            self.tf.decoder.requires_grad_(False)
            self.tf_project = nn.Linear(2 * config.tf_repr, config.width)
        if config.mode != "tf":
            self.lag = lag_encoder
        if config.mode == "concat":
            self.concat = nn.Linear(2 * config.width, config.width)
        if config.mode == "gated":
            self.gate = nn.Linear(2 * config.width, config.width)
        self.head = nn.Linear(config.width, config.horizon)

    def forward(self, history):
        cfg = self.config
        if (history.ndim != 3 or history.shape[1] != cfg.history or history.shape[0] < 1
                or history.shape[2] < 1 or history.dtype != torch.float32
                or not bool(torch.isfinite(history).all())):
            raise ValueError("Finite float32 [B,history,C] input required; missing data is not silently filled")
        mean = history.mean(1, keepdim=True).detach()
        scale = (history.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        z = (history - mean) / scale
        b, _, c = z.shape
        tf_state = lag_state = gate = None
        diagnostics = {}
        if self.tf is not None:
            n = cfg.history // cfg.patch_len
            summaries = []
            # The author's order is shared over its B*C input. Separate original
            # windows so another example cannot contribute later observations
            # to this example's order, even during batched source training.
            for window in z.split(1, dim=0):
                representation, _ = self.tf(window, norm=0, mask_mode="c", copies=cfg.copies)
                expected = (cfg.copies * c, n, cfg.tf_repr)
                if representation.shape != expected or not representation.is_complex():
                    raise RuntimeError("Official representation shape or domain changed")
                summaries.append(representation.reshape(cfg.copies, 1, c, n, cfg.tf_repr).mean((0, 3)))
            # Pool native FRFT patches and mask copies. This is a whole-history
            # per-channel endpoint summary, not interpolated raw-time tokens.
            pooled = torch.cat(summaries, dim=0)
            tf_state = self.tf_project(torch.cat((pooled.real, pooled.imag), dim=-1))
        if self.lag is not None:
            lag_state, diagnostics = self.lag(z)
            if lag_state.shape != (b, c, cfg.width) or not bool(torch.isfinite(lag_state).all()):
                raise RuntimeError("Lag adapter must return finite original-endpoint [B,C,width]")
        if cfg.mode == "tf":
            state = tf_state
        elif cfg.mode == "lag":
            state = lag_state
        elif cfg.mode == "concat":
            state = self.concat(torch.cat((tf_state, lag_state), dim=-1))
        else:
            gate = torch.sigmoid(self.gate(torch.cat((tf_state, lag_state), dim=-1)))
            state = gate * tf_state + (1 - gate) * lag_state
        normalized = self.head(state).transpose(1, 2)
        return {"prediction_normalized": normalized, "prediction": normalized * scale + mean,
                "history_mean": mean, "history_scale": scale, "endpoint_state": state,
                "tf_endpoint": tf_state, "lag_endpoint": lag_state, "gate": gate,
                "lag_diagnostics": diagnostics}

    def parameter_counts(self):
        return {"stored": sum(p.numel() for p in self.parameters()),
                "active": sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def load_tf_checkpoint(self, path, *, expected_sha256):
        """Optional exact flat official state_dict, no guessed wrappers/extra keys.

        Additional tau/buffers need an explicitly reviewed compatibility rule;
        unknown keys are rejected before any model state can be changed.
        """
        if self.tf is None:
            raise ValueError("Lag-only model has no TF checkpoint target")
        p = Path(path)
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError("TF checkpoint SHA256 mismatch")
        state = torch.load(p, map_location="cpu", weights_only=True)
        own = self.tf.state_dict()
        if not isinstance(state, dict) or set(state) != set(own):
            raise ValueError("Exact flat TF state_dict required; missing/unexpected keys rejected")
        if any(not isinstance(state[k], torch.Tensor) or state[k].shape != v.shape
               or state[k].dtype != v.dtype or not bool(torch.isfinite(state[k]).all())
               for k, v in own.items()):
            raise ValueError("TF checkpoint shape/dtype/nonfinite mismatch")
        self.tf.load_state_dict(state, strict=True)
        return {"sha256": actual, "strict": True, "missing": [], "unexpected": [],
                "new_lag_fusion_head_not_loaded": True}

    def adaptation_parameters(self, path):
        if path == "Z":
            return ()
        if path == "U":
            return tuple(p for p in self.parameters() if p.requires_grad)
        if path == "F":
            raise NotImplementedError("Event-supervised objective/roles not implemented")
        raise ValueError("Unknown adaptation path")


def residual_scores(output, target):
    """No labels, ranking transform, threshold, source or target fitting here.

    Output score [B,P] becomes available only when the corresponding future
    target arrives; forecasts themselves were computed from history alone.
    """
    if target.shape != output["prediction"].shape or not bool(torch.isfinite(target).all()):
        raise ValueError("Finite future target with exact forecast shape required")
    error = ((target - output["history_mean"]) / output["history_scale"]
             - output["prediction_normalized"])
    per_channel = error.square()
    return {"per_channel": per_channel, "continuous": per_channel.mean(-1)}


def candidate_objective(output, target):
    return residual_scores(output, target)["per_channel"].mean()


def forecast_coordinates(history_start, history, horizon):
    """Absolute sample indices, no invented seconds or retrospective backfill."""
    if any(type(v) is not int or v < 0 for v in (history_start, history, horizon)) or min(history, horizon) < 1:
        raise ValueError("Nonnegative start and positive history/horizon required")
    return {"input_start": history_start, "input_end_inclusive": history_start + history - 1,
            "target_indices": list(range(history_start + history, history_start + history + horizon)),
            "score_available_indices": list(range(history_start + history, history_start + history + horizon))}


def candidate_from_config(path, *, mode=None):
    """Construct from explicit local configuration, never read data or weights."""
    import json
    import os
    from .candidate_autoformer_lag import AutoformerLagEndpoint
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    settings = dict(data["model"])
    if mode is not None:
        settings["mode"] = mode
    config = CandidateConfig(**settings)
    def external(value):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if "$" in expanded:
            raise ValueError("Unresolved external source path")
        p = Path(expanded)
        return p if p.is_absolute() else path.parent / p
    if data["official_sha256"] != OFFICIAL_SHA256:
        raise ValueError("Config official TimeRadar identity mismatch")
    lag = None
    if config.mode != "tf":
        lag = AutoformerLagEndpoint(external(data["lag_official_file"]), width=config.width,
            expected_sha256=data["lag_official_sha256"], commit=data["lag_official_commit"], **data["lag"])
    model = TFLagCandidate(config, external(data["official_file"]) if config.mode != "lag" else None,
        lag_encoder=lag, lag_provenance=lag.provenance if lag is not None else None)
    if data.get("tf_checkpoint") is not None:
        model.load_tf_checkpoint(external(data["tf_checkpoint"]["path"]),
                                 expected_sha256=data["tf_checkpoint"]["sha256"])
    return model
