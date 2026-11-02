"""Validated V1 configuration; no Torch/model/data import on metadata paths."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class HybridV1Config:
    history: int = 512
    local_window: int = 100
    horizon: int = 32
    width: int = 128
    temporal_patch: int = 16
    temporal_layers: int = 4
    heads: int = 4
    tf_hidden: int = 64
    tf_repr: int = 256
    tf_depth: int = 10
    tf_patch: int = 5
    copies: int = 2
    lag_window: int = 100
    max_lag: int = 32
    lag_width: int = 32
    lag_top_k: int = 4
    lag_min_overlap: int = 2
    ms_factors: tuple = (4, 2)
    fusion: str = "left_interaction"
    fusion_layers: int = 1
    dropout: float = 0.0
    reconstruction_projection: str = "real"
    fixed_order: object = None
    history_mask_probability: float = 0.15
    temporal_enabled: bool = True
    tf_enabled: bool = True
    self_lag_enabled: bool = True
    cross_lag_enabled: bool = True
    cross_synchronous_only: bool = False
    ms_enabled: bool = True
    use_variate_attention: bool = False

    def __post_init__(self):
        integers = ("history", "local_window", "horizon", "width", "temporal_patch",
                    "temporal_layers", "heads", "tf_hidden", "tf_repr", "tf_depth",
                    "tf_patch", "copies", "lag_window", "max_lag", "lag_width",
                    "lag_top_k", "lag_min_overlap", "fusion_layers")
        if any(type(getattr(self, n)) is not int or getattr(self, n) < 1 for n in integers):
            raise ValueError("Positive integer model dimensions required")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("TimesFM RoPE requires even per-head width")
        if self.local_window > self.history or self.lag_window > self.history:
            raise ValueError("Local/lag windows must fit history")
        if self.local_window % self.tf_patch or self.copies % 2:
            raise ValueError("Native TF requires divisible W and positive even copies")
        if self.max_lag >= self.lag_window or self.lag_min_overlap > self.lag_window:
            raise ValueError("Lag/overlap outside original sample window")
        factors = tuple(self.ms_factors)
        if not factors or any(type(r) is not int or r < 1 for r in factors) or list(factors) != sorted(factors, reverse=True):
            raise ValueError("MS factors must be positive coarse-to-fine")
        object.__setattr__(self, "ms_factors", factors)
        if self.fusion not in ("none", "feature_mean", "concat_linear", "left_interaction", "late_score_equal"):
            raise ValueError("Unknown fusion control")
        if self.reconstruction_projection not in ("real", "author_abs"):
            raise ValueError("Explicit signed-real or author-abs reconstruction required")
        if self.fixed_order is not None and self.fixed_order not in (0, 1):
            raise ValueError("Only predeclared order0/order1 controls are supported")
        if not 0 <= self.dropout < 1 or not 0 <= self.history_mask_probability < 1:
            raise ValueError("Invalid dropout/historical-mask probability")
        for n in ("temporal_enabled", "tf_enabled", "self_lag_enabled", "cross_lag_enabled",
                  "cross_synchronous_only", "ms_enabled", "use_variate_attention"):
            if type(getattr(self, n)) is not bool:
                raise ValueError("Model switches require explicit booleans")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)
