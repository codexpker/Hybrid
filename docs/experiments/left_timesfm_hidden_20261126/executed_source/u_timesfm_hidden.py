"""Frozen official TimesFM 3.0 historical patch representations.

No forecasts are used by this interface. The original full forward remains intact.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .timesfm3_frozen import (
    SOURCE_COMMIT,
    WEIGHT_REVISION,
    WEIGHT_SHA,
    asset_root,
    load_forecaster,
)

PATCH = 32
MAX_HISTORY = 512
HIDDEN = 1280


def tensor_sha(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def model_sha(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_encoder(workspace, random=False, device="cuda"):
    """Use the exact released architecture in both arms, frozen and fp32."""
    if not random:
        model = load_forecaster(workspace, device).model
    else:
        root = asset_root(workspace)
        src = root / f"timesfm-{SOURCE_COMMIT}" / "src"
        sys.path.insert(0, str(src))
        from timesfm3.torch.model import TimesFM3Torch

        config = json.loads((root / "config.json").read_text())
        devices = (
            [torch.cuda.current_device()] if str(device).startswith("cuda") else []
        )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(2025)
            model = TimesFM3Torch(**config).to(device)
    model.requires_grad_(False).eval()
    assert model.use_variate_attention and model.input_patch_len == PATCH
    assert len(model.transformer_stack.layers) == 20
    assert model.transformer_config.transformer.causal_attention
    return model


def inputs(history, valid_lengths=None, padding_value=0.0):
    """Pack [B,T,C] chronological past observations into official [B,C,P,32].

    A shorter valid series is right-aligned. Its leading positions never enter
    running statistics or valid attention keys. Whole patch lengths are required.
    """
    assert history.ndim == 3 and history.shape[1] <= MAX_HISTORY
    b, t, c = history.shape
    if valid_lengths is None:
        valid_lengths = torch.full((b,), t, dtype=torch.long, device=history.device)
    valid_lengths = torch.as_tensor(valid_lengths, device=history.device)
    assert valid_lengths.shape == (b,)
    assert torch.all(
        (valid_lengths > 0) & (valid_lengths <= t) & (valid_lengths % PATCH == 0)
    )
    assert t % PATCH == 0 and torch.isfinite(history).all()
    masks = torch.arange(t, device=history.device)[None, :] < t - valid_lengths[:, None]
    values = torch.where(masks[:, :, None], padding_value, history)
    values = values.transpose(1, 2).reshape(b, c, t // PATCH, PATCH).contiguous()
    masks = masks[:, None, :].expand(b, c, t).reshape_as(values)
    return {
        "values": values,
        "masks": masks,
        "patch_is_target": torch.ones(
            (b, c, t // PATCH), dtype=torch.bool, device=history.device
        ),
    }


@torch.no_grad()
def encode(model, history, valid_lengths=None):
    """Return fp32 tokens [B,C,P,1280] and True-for-valid patch mask [B,C,P]."""
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    packed = inputs(history, valid_lengths)
    output = model.forward(packed, return_aux_outputs=True)
    hidden = output["__call__:transformer_output"]
    valid = ~packed["masks"].all(-1)
    assert hidden.shape == (*valid.shape, HIDDEN) and torch.isfinite(hidden).all()
    return hidden, valid


def check(workspace, destination):
    """Normal-source engineering evidence, with no target test input."""
    from .intervention_reference import source_records

    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        30 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    roots = source_records(workspace)["fit"]
    device = torch.device("cuda")
    model = load_encoder(workspace, device=device)
    before = model_sha(model)
    config = model.to_dict()
    te = torch.tensor(np.stack([r[:256, :25] for r in roots[0][:2]]), device=device)
    padded = torch.nn.functional.pad(te, (0, 0, 256, 0))
    counts = torch.full((2,), 256, device=device)
    with torch.no_grad():
        a, am = encode(model, te)
        b, bm = encode(model, padded, counts)
        altered = inputs(padded, counts, padding_value=12345.0)
        badpad = model.forward(altered, return_aux_outputs=True)[
            "__call__:transformer_output"
        ]
        error_padding = (b[:, :, -8:] - badpad[:, :, -8:]).abs().max().item()
        error_length = (a - b[:, :, -8:]).abs().max().item()
        # Short and padded input formats produce different representations.
        # Do not attribute that difference to one internal mechanism here;
        # deployment always preserves the same left-padded 512-point format.
        padded_output = model.forward(inputs(padded, counts), return_aux_outputs=True)
        short_output = model.forward(inputs(te), return_aux_outputs=True)
        for padded_stat, short_stat in zip(
            padded_output["revin_stats"], short_output["revin_stats"]
        ):
            torch.testing.assert_close(
                padded_stat[:, :, -8:], short_stat, rtol=0, atol=0
            )
        for mask in padded_output["__call__:seq_attn_mask"]:
            assert not mask[..., :8].any()
        torch.testing.assert_close(b[:, :, -8:], badpad[:, :, -8:], rtol=0, atol=0)
        alone, _ = encode(model, te[:1])
        torch.testing.assert_close(a[:1], alone, rtol=1e-4, atol=1e-4)
        changed = te.clone()
        changed[:, 128:, 0] += torch.sin(torch.arange(128, device=device) * 0.17) * 2
        ch, _ = encode(model, changed)
        temporal_error = (ch[:, :, :4] - a[:, :, :4]).abs().max().item()
        cross_change = (ch[:, 1:, 4:] - a[:, 1:, 4:]).abs().max().item()
        assert temporal_error < 1e-4 and cross_change > 0
        timing = []
        for c in (25, 31, 38):
            for batch in (4, 16, 32):
                x = torch.tensor(
                    np.stack([roots[1][i % 4][:512, :c] for i in range(batch)]),
                    device=device,
                )
                encode(model, x)
                torch.cuda.synchronize()
                tic = time.monotonic()
                for _ in range(3):
                    z, m = encode(model, x)
                torch.cuda.synchronize()
                timing.append(
                    dict(
                        channels=c,
                        batch=batch,
                        history=512,
                        shape=list(z.shape),
                        seconds_per_window=(time.monotonic() - tic) / (3 * batch),
                        peak_bytes=torch.cuda.max_memory_allocated(),
                    )
                )
    assert before == model_sha(model)
    assert all(p.grad is None for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    random = load_encoder(workspace, random=True, device=device)
    rz, _ = encode(random, te)
    random_hash = model_sha(random)
    result = dict(
        source_commit=SOURCE_COMMIT,
        weight_revision=WEIGHT_REVISION,
        weight_sha256=WEIGHT_SHA,
        full_model_state_sha256=before,
        random_model_state_sha256=random_hash,
        random_seed=2025,
        official_config=config,
        pretrained_layers=20,
        hidden_dim=HIDDEN,
        patch_len=PATCH,
        hidden_axes="batch,variable,chronological_patch,hidden",
        no_current_input=True,
        padding_mask_true_means_missing=True,
        valid_mask_shape=list(bm.shape),
        padded_valid_max_abs_error=error_length,
        short_vs_padded_agreement_required=False,
        short_vs_padded_reason="Native input-format sensitivity observed; its internal cause is not isolated. Deployment fixes left-padded512 indexing.",
        padded_running_stats_equal=True,
        padded_attention_keys_excluded=True,
        padding_value_max_abs_error=error_padding,
        temporal_future_mutation_max_abs_error=temporal_error,
        cross_variable_max_change=cross_change,
        random_shape=list(rz.shape),
        frozen=True,
        state_unchanged=True,
        precision="float32",
        timings=timing,
        source_only=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    check(args.workspace, args.destination)
