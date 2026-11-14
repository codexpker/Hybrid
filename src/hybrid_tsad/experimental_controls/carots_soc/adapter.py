"""Opt-in existing-embedding SOC control, not a new Hybrid architecture.

Derived from CAROTS; this directory retains its Modified MIT Non-Commercial
license. There is no augmentor, projector, backbone update, scoring or trainer
here. Rows must be [G_P, G_N], with equal lengths and corresponding negatives.
"""
import math
from types import SimpleNamespace


def soc_objective(embeddings, *, feature_identity, enabled=False,
                  similarity_threshold=.5, temperature=.1):
    """Return (differentiable loss, JSON-safe diagnostics).

    Caller supplies fixed feature provenance and all normal/augmented rows.
    Identity projection means precisely these supplied embeddings are used.
    No threshold fitting, data access, RNG or model-mode mutation occurs.
    """
    import torch
    import torch.nn.functional as F
    if not isinstance(feature_identity, str) or not feature_identity.strip():
        raise ValueError("A frozen, nonempty feature_identity is required")
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    if embeddings.ndim != 2 or embeddings.shape[1] < 1 or embeddings.shape[0] % 2:
        raise ValueError("Expected [2*n, features] in ordered [G_P, G_N] groups")
    if not embeddings.is_floating_point() or not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Embeddings must be finite floating point")
    if (not math.isfinite(float(temperature)) or float(temperature) <= 0 or
            not math.isfinite(float(similarity_threshold)) or
            not -1 <= float(similarity_threshold) < 1):
        raise ValueError("Require finite temperature>0 and -1<=threshold<1")
    n = embeddings.shape[0] // 2
    meta = dict(control="CAROTS_SOC_PUBLISHED_OBJECTIVE", enabled=enabled,
                feature_identity=feature_identity, projector="identity_no_parameters",
                positive_rows=n, negative_rows=n,
                similarity_threshold=float(similarity_threshold),
                temperature=float(temperature),
                upstream_commit="63a41f45a2ae680d5cdbb2d5739e7a11264867cb",
                license_scope="Modified MIT Non-Commercial with Permission",
                valid_nonself_positive_pairs=0, skipped=False, reason=None)
    # Multiplication before sum prevents overflow for large but finite inputs.
    zero = (embeddings * 0).sum()
    if not enabled:
        meta.update(skipped=True, reason="disabled")
        return zero, meta
    if n == 0:
        meta.update(skipped=True, reason="empty_support")
        return zero, meta
    with torch.no_grad():
        norms = embeddings.detach().norm(p=2, dim=1)
        if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
            raise ValueError("Enabled SOC requires finite nonzero embedding norms")
        normed = F.normalize(embeddings.detach(), p=2, dim=1)
        gram = normed @ normed.T
        positive = gram[:n, :n] >= float(similarity_threshold)
        diagonal = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        pairs = int((positive & ~diagonal).sum().item())
        meta["valid_nonself_positive_pairs"] = pairs
        if pairs == 0:
            meta.update(skipped=True, reason="no_nonself_positive_pair")
            return zero, meta
        # The upstream denominator subtracts n; it assumes every self-pair passes.
        # Reject unsupported zero-norm/roundoff configurations rather than silently
        # replacing its formula or changing the eligible sample set.
        if not bool(positive.diagonal().all()):
            raise ValueError("Upstream SOC requires eligible self-pairs for every G_P row")
    from .upstream_loss import loss_fn
    cfg = SimpleNamespace(CAROTS=SimpleNamespace(
        SIM_THRESHOLD=float(similarity_threshold), TEMPERATURE=float(temperature)))
    loss = loss_fn(embeddings, cfg)
    if loss.ndim != 0 or not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite CAROTS SOC loss; retain inputs and stop")
    return loss, meta
