"""External author Autoformer delay operator with explicit project adaptations.

The operator itself is imported unchanged. Channel pair expansion, scalar
embedding and endpoint/source aggregation are new project glue, not the author's
multivariate anomaly detector. No physical delay/causal interpretation is made.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

import torch
from torch import nn

AUTOFORMER_COMMIT = "51c7d416ae120b805fd5beef2f4ccf7de496a6ff"
AUTOFORMER_SHA256 = "dddbc3cc533f9640bbee0e624a49532efe4ee99735101f51a4b03e105772af01"

class AutoformerLagEndpoint(nn.Module):
    def __init__(self, official_file, *, expected_sha256=AUTOFORMER_SHA256,
                 commit=AUTOFORMER_COMMIT, width=64,
                 heads=4, factor=1., self_enabled=True, cross_enabled=True,
                 max_expanded_pairs=4096):
        super().__init__()
        if (width < 1 or heads < 1 or width % heads or factor <= 0
                or not math.isfinite(factor) or not (self_enabled or cross_enabled)):
            raise ValueError("Valid dimensions/factor and at least one lag path required")
        if commit != AUTOFORMER_COMMIT or expected_sha256 != AUTOFORMER_SHA256:
            raise ValueError("Pinned official revision and full SHA256 required")
        path = Path(official_file)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("Official Autoformer module identity mismatch")
        spec = importlib.util.spec_from_file_location("candidate_external_autoformer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.provenance = {"repository": "https://github.com/thuml/Autoformer",
            "commit": commit, "class": "layers.AutoCorrelation.AutoCorrelationLayer",
            "source_sha256": expected_sha256}
        self.width, self.factor = width, factor
        self.self_enabled, self.cross_enabled = self_enabled, cross_enabled
        self.max_expanded_pairs = max_expanded_pairs
        self.embed = nn.Linear(1, width)
        def author_layer():
            return module.AutoCorrelationLayer(module.AutoCorrelation(
                mask_flag=False, factor=factor, output_attention=True), width, heads)
        self.self_operator = author_layer() if self_enabled else None
        self.cross_operator = author_layer() if cross_enabled else None
        self.merge = nn.Linear(2 * width, width)

    def forward(self, history):
        if history.ndim != 3 or not bool(torch.isfinite(history).all()):
            raise ValueError("Finite [B,H,C] history required")
        if history.shape[0] < 1 or history.shape[2] < 1:
            raise ValueError("Nonempty independent windows and channels required")
        # Preserve the original batch-shared top-k within each original window's
        # channels/pairs, never across independent training windows.
        grouped = [self._window_forward(window) for window in history.split(1, dim=0)]
        state = torch.cat([item[0] for item in grouped], dim=0)
        metadata = dict(grouped[0][1])
        for key in ("self_endpoint", "cross_endpoint"):
            metadata[key] = torch.cat([item[1][key] for item in grouped], dim=0)
        metadata["selection_scope"] = "within_one_original_window_only"
        metadata["original_window_count"] = len(grouped)
        return state, metadata

    def _window_forward(self, history):
        b, h, c = history.shape
        if b != 1:
            raise ValueError("Author lag call must be grouped by one original window")
        k = int(self.factor * math.log(h)) if h else 0
        if not 1 <= k <= h:
            raise ValueError("Author top-k requires 1 <= int(factor*log(H)) <= H")
        if b*c*(c-1) > self.max_expanded_pairs and self.cross_enabled:
            raise ValueError("Pair budget exceeded; no silent chunking of batch-shared top-k")
        tokens = self.embed(history.transpose(1, 2)[..., None])  # B,C,H,d
        self_state = history.new_zeros(b, c, self.width)
        cross_state = torch.zeros_like(self_state)
        if self.self_operator is not None:
            x = tokens.reshape(b*c, h, self.width)
            out, _ = self.self_operator(x, x, x, None)
            self_state = out[:, -1].reshape(b, c, self.width)
        pairs = [(i, j) for i in range(c) for j in range(c) if i != j]
        if self.cross_operator is not None and pairs:
            target = torch.tensor([p[0] for p in pairs], device=history.device)
            source = torch.tensor([p[1] for p in pairs], device=history.device)
            q = tokens.index_select(1, target).reshape(-1, h, self.width)
            kv = tokens.index_select(1, source).reshape(-1, h, self.width)
            out, _ = self.cross_operator(q, kv, kv, None)
            # Pool sources only AFTER selecting the endpoint. Temporal means
            # would erase the effect of circular time-delay aggregation.
            cross_state = out[:, -1].reshape(b, c, c-1, self.width).mean(2)
        state = self.merge(torch.cat((self_state, cross_state), dim=-1))
        return state, {"self_endpoint": self_state, "cross_endpoint": cross_state,
            "pair_order_target_source": pairs, "token_delay_unit": "one_original_sample",
            "author_selection": "batch_shared_topk" if self.training else "per_expanded_sample_topk",
            "circular_prefix_aggregation": True, "causal_per_token_mask": False,
            "prediction_target_outside_prefix": True}
