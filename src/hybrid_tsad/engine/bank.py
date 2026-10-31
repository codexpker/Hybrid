"""Shared forecast-bank collection; original per-block forward/RNG order retained."""
import json
from pathlib import Path
import time

import numpy as np

from ..config import sha256
from ..protocol import score_block


def collect_forecasts(model, data, origins, source_scale, *, mask_seed,
                      expected_index=None, replay_first2=True):
    """Collect contracted past100/400→next5 forecasts with no labels or optimizer.

    CPU is an engineering validation path. CUDA uses the original device RNG
    before each block; model initialization/checkpoint loading precedes this call.
    """
    import torch

    if model.training or model.history not in (100, 400):
        raise ValueError("Frozen forecast collection requires eval model and history100/400")
    history = model.history
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
    origins = list(origins)
    if not origins or any(b < history or b + 5 > len(data) for b in origins):
        raise ValueError("Origins must have complete contracted past/next5 footprints")
    if any(b + 5 > c for b, c in zip(origins, origins[1:])):
        raise ValueError("Forecast blocks must be ordered and disjoint")
    get_rng = (lambda: torch.cuda.get_rng_state(device)) if is_cuda else torch.get_rng_state
    set_rng = (lambda state: torch.cuda.set_rng_state(state, device)) if is_cuda else torch.set_rng_state
    synchronize = (lambda: torch.cuda.synchronize(device)) if is_cuda else (lambda: None)
    torch.manual_seed(mask_seed)
    values = {k: [] for k in ["prediction", "target", "history_scale_by_block",
                              "normalized_channel_residual", "raw_score", "standardized_score"]}
    states, indices, times = [], [], []
    with torch.no_grad():
        for i, b in enumerate(origins):
            x = torch.tensor(data[None, b - history:b].copy(), device=device)
            before = get_rng()
            states.append(before.cpu().numpy())
            synchronize()
            ts = time.perf_counter()
            output = model(x)
            synchronize()
            times.append(time.perf_counter() - ts)
            pred = output["prediction"].cpu().numpy()[0]
            scale = output["scale"].cpu().numpy()[0]
            target = data[b:b + 5].copy()
            residual = np.abs(pred - target) / scale
            score = score_block(pred, target, scale, source_scale)
            assert np.isfinite(residual).all() and np.isfinite(score).all()
            if replay_first2 and i < 2:
                after = get_rng()
                set_rng(before)
                replay = model(x)
                assert torch.equal(output["prediction"], replay["prediction"])
                set_rng(after)
            for key, value in [("prediction", pred), ("target", target),
                               ("history_scale_by_block", scale),
                               ("normalized_channel_residual", residual),
                               ("raw_score", residual.mean(-1)), ("standardized_score", score)]:
                values[key].append(value)
            indices.extend(range(b, b + 5))
            if is_cuda and torch.cuda.max_memory_allocated(device) >= 30 * 1024**3:
                raise ValueError("Frozen inference memory limit exceeded")
    index = np.array(indices)
    if expected_index is not None and not np.array_equal(index, expected_index):
        raise ValueError("Forecast time index does not match frozen expected index")
    arrays = {k: (np.stack(v) if k == "history_scale_by_block" else np.concatenate(v))
              for k, v in values.items()}
    arrays.update(time_index=index, forecast_origin=np.asarray(origins), valid=np.ones(len(index), bool))
    return arrays, np.stack(states), times


def write_bank(output, arrays, rng_states, metadata):
    """New directory only. No append/resume or mutation of existing banks."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "predictions_and_scores.npz", **arrays)
    np.savez_compressed(output / "rng_states.npz", before_each_block=rng_states)
    metadata = {**metadata, "bank_sha256": sha256(output / "predictions_and_scores.npz")}
    (output / "inference.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return {p.name: sha256(p) for p in output.iterdir() if p.is_file()}


def inference_metadata(readout_job, spec, *, points, blocks, mask_seed, checkpoint_sha256,
                       query_array_sha256, seconds=None, peak_allocated_bytes=None,
                       forward_p50_seconds=None, forward_p95_seconds=None):
    """v2 readout_job disambiguates the training spec's job key."""
    if "readout_job" in spec:
        raise ValueError("Training spec cannot shadow readout_job")
    return dict(status="INFERENCE_COMPLETE_LABELS_NOT_OPENED", readout_job=readout_job, **spec,
                points=points, blocks=blocks, mask_seed=mask_seed, batch=1,
                seconds=seconds, peak_allocated_bytes=peak_allocated_bytes,
                forward_p50_seconds=forward_p50_seconds, forward_p95_seconds=forward_p95_seconds,
                checkpoint_sha256=checkpoint_sha256, query_array_sha256=query_array_sha256,
                labels_opened=False, replay_first2_bit_exact=True, training=False)


def recovered_metadata(readout_job, spec, *, points, blocks, mask_seed,
                       checkpoint_sha256, query_array_sha256, bank_hashes):
    """Represent already verified banks; never reruns a forward or invents timing."""
    result = inference_metadata(readout_job, spec, points=points, blocks=blocks,
                                mask_seed=mask_seed, checkpoint_sha256=checkpoint_sha256,
                                query_array_sha256=query_array_sha256)
    result.update(status="RECOVERED_FROM_METADATA_FAILURE", new_forward_passes=0,
                  bank_hashes=dict(bank_hashes),
                  provenance="caller must independently verify bank/RNG/time index before recovery")
    return result
