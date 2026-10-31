"""Thin prediction API over the byte-identical core; no new model or scorer."""
from pathlib import Path

from .config import OFFICIAL_SHA, sha256, validate_config


def create_predictor(official_file, *, mode="hybrid", history=400, copies=2,
                     temporal_depth=4, checkpoint=None, device="cpu"):
    """Construct the existing model; optionally load a plain legacy state_dict strictly.

    No upstream code is bundled. The caller must supply the pinned official file
    and its import dependencies. This is random initialization unless checkpoint
    is supplied. RNG ownership stays with the caller.
    """
    official_file = Path(official_file).resolve()
    if sha256(official_file) != OFFICIAL_SHA:
        raise ValueError("Official file SHA256 mismatch")
    import torch
    from .model import HybridPredictor

    model = HybridPredictor(str(official_file), history=history, copies=copies,
                            mode=mode, temporal_depth=temporal_depth).to(device)
    if checkpoint is not None:
        # Plain state_dict only; never unpickle an arbitrary checkpoint object.
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
    return model.eval()


def predictor_from_config(config_file, *, checkpoint=None, device="cpu"):
    """Use model fields and external official path from a validated inventory."""
    cfg = validate_config(config_file)
    return create_predictor(cfg["resolved_paths"]["official_file"],
                            checkpoint=checkpoint, device=device, **cfg["config"]["model"])


def predict(model, history):
    """Return the core output dict for [batch, history, channels] past values.

    List/NumPy inputs become the model's dtype/device; target values are never
    accepted. eval() is required so this wrapper cannot change a training model.
    TF masking is stochastic: seed/reset RNG explicitly for reproducible replay.
    """
    import torch

    if model.training:
        raise ValueError("Prediction API requires model.eval()")
    parameter = next(model.parameters())
    values = torch.as_tensor(history, dtype=parameter.dtype, device=parameter.device)
    if values.ndim != 3 or values.shape[1] != model.history:
        raise ValueError("Expected [batch, model.history, channels]")
    if values.shape[0] < 1 or values.shape[2] < 1:
        raise ValueError("Batch and channel dimensions must be nonempty")
    with torch.inference_mode():
        return model(values)
