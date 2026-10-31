"""Thin prediction API over the byte-identical core; no new model or scorer."""
from pathlib import Path

from .config import OFFICIAL_SHA, read_json, sha256, validate_config


def verify_checkpoint_contract(checkpoint, metadata, *, history, mode, copies=2, temporal_depth=4):
    if metadata is None:
        if history != 400:
            raise ValueError("A history100 checkpoint requires explicit hash-bound checkpoint metadata")
        return
    value = read_json(metadata)
    expected = dict(checkpoint_sha256=sha256(checkpoint), history=history, horizon=5,
                    mode=mode, copies=copies, temporal_depth=temporal_depth)
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint identity/history/architecture differs from the inference contract")


def create_predictor(official_file, *, mode="hybrid", history=400, copies=2,
                     temporal_depth=4, checkpoint=None, checkpoint_metadata=None, device="cpu"):
    """Construct the existing model; optionally load a plain legacy state_dict strictly.

    No upstream code is bundled. The caller must supply the pinned official file
    and its import dependencies. This is random initialization unless checkpoint
    is supplied. RNG ownership stays with the caller.
    """
    official_file = Path(official_file).resolve()
    if sha256(official_file) != OFFICIAL_SHA:
        raise ValueError("Official file SHA256 mismatch")
    if checkpoint is not None:
        verify_checkpoint_contract(checkpoint, checkpoint_metadata, history=history, mode=mode,
                                   copies=copies, temporal_depth=temporal_depth)
    import torch
    from .model import HybridPredictor

    model = HybridPredictor(str(official_file), history=history, copies=copies,
                            mode=mode, temporal_depth=temporal_depth).to(device)
    if checkpoint is not None:
        # Plain state_dict only; never unpickle an arbitrary checkpoint object.
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=True)
    return model.eval()


def predictor_from_config(config_file, *, checkpoint=None, checkpoint_metadata=None, device="cpu"):
    """Use model fields and external official path from a validated inventory."""
    cfg = validate_config(config_file)
    return create_predictor(cfg["resolved_paths"]["official_file"],
                            checkpoint=checkpoint, checkpoint_metadata=checkpoint_metadata,
                            device=device, **cfg["config"]["model"])


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
