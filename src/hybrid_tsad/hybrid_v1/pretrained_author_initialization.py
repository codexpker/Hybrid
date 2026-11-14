"""One controlled initialization: archived base initial -> full official author.

No new modules/parameters, RNG draws, modes, optimizer states, normalization,
losses or scores are introduced. This file can be loaded by absolute path next
to the immutable original13 runtime; it does not import or replace that runtime.
"""
import hashlib
from pathlib import Path

import torch

OFFICIAL_SHA256 = '528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c'
PREFIX = 'tf_backbone.author.'
EMA_KEYS = frozenset(('tau_n', 'tau_a', 'tau_inited'))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def scientific_author_state(state, own):
    """Pinned official DDP namespace and exactly three non-trainable EMA buffers."""
    if not isinstance(state, dict) or not state or not all(key.startswith('module.') for key in state):
        raise ValueError('Official528 requires exactly one uniform DDP module namespace')
    flat = {key[len('module.'):]: value for key, value in state.items()}
    if set(flat) - set(own) != EMA_KEYS or set(own) - set(flat):
        raise ValueError('Only complete author107 plus exact EMA triple accepted')
    metadata = {}
    for key in sorted(EMA_KEYS):
        value = flat[key]
        if not isinstance(value, torch.Tensor) or value.shape != torch.Size([]):
            raise ValueError('Official EMA buffer must be scalar tensor')
        if key == 'tau_inited':
            if value.dtype != torch.bool:
                raise ValueError('Official tau_inited must be scalar bool')
        elif not value.is_floating_point() or not bool(torch.isfinite(value)):
            raise ValueError('Official tau_n/tau_a must be finite floating scalars')
        metadata[key] = dict(value=value.item(), shape=[], dtype=str(value.dtype), tensor_sha256=tensor_sha(value))
    scientific = {key: value for key, value in flat.items() if key not in EMA_KEYS}
    for key, expected in own.items():
        value = scientific[key]
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Exact scientific key/shape/dtype/finite mismatch: ' + key)
    return scientific, metadata


def initialize_pretrained_author(model, *, base_initial_path, base_initial_sha256, official_path):
    """Restore exact matched random initial then replace only the full author.

Call after the unchanged old constructor and before its unchanged fit(). All
input artifacts stay read-only. No torch/cuda seed reset occurs in this method.
"""
    if file_sha(base_initial_path) != base_initial_sha256 or file_sha(official_path) != OFFICIAL_SHA256:
        raise PermissionError('Only exact matched initial and qualified official528 allowed')
    cpu_before = torch.get_rng_state().clone()
    cuda_devices = sorted({value.device.index for value in list(model.parameters()) + list(model.buffers()) if value.is_cuda})
    cuda_before = {device: torch.cuda.get_rng_state(device).clone() for device in cuda_devices}
    modes_before = {name: module.training for name, module in model.named_modules()}
    requires_before = {name: value.requires_grad for name, value in model.named_parameters()}
    base = torch.load(base_initial_path, map_location='cpu', weights_only=True)
    official = torch.load(official_path, map_location='cpu', weights_only=True)
    own = model.state_dict()
    if not isinstance(base, dict) or set(base) != set(own):
        raise ValueError('Whole archived base initial must strictly match old architecture')
    for key, value in base.items():
        if (not isinstance(value, torch.Tensor) or value.shape != own[key].shape or value.dtype != own[key].dtype
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Base initial key/shape/dtype/finite mismatch: ' + key)
    scientific, metadata = scientific_author_state(official, model.tf_backbone.author.state_dict())
    if {key[len(PREFIX):] for key in base if key.startswith(PREFIX)} != set(scientific):
        raise ValueError('Whole author subtree identity differs from original base')
    model.load_state_dict(base, strict=True)
    model.tf_backbone.author.load_state_dict(scientific, strict=True)
    after = model.state_dict()
    if set(after) != set(base):
        raise ValueError('Initializer added or removed state keys')
    nonauthor = [key for key in after if not key.startswith(PREFIX)]
    if any(not torch.equal(after[key].detach().cpu(), base[key]) for key in nonauthor):
        raise ValueError('Initializer changed non-author old initial tensors')
    if any(not torch.equal(after[PREFIX+key].detach().cpu(), value) for key, value in scientific.items()):
        raise ValueError('Official author strict loading did not preserve values exactly')
    if (not torch.equal(cpu_before, torch.get_rng_state())
            or any(not torch.equal(value, torch.cuda.get_rng_state(device)) for device, value in cuda_before.items())
            or modes_before != {name: module.training for name, module in model.named_modules()}
            or requires_before != {name: value.requires_grad for name, value in model.named_parameters()}):
        raise ValueError('Initialization changed RNG, module modes, or gradient policy')
    if file_sha(base_initial_path) != base_initial_sha256 or file_sha(official_path) != OFFICIAL_SHA256:
        raise ValueError('Initializer mutated an original weight artifact')
    return dict(policy='OFFICIAL528_FULL_AUTHOR_FROM_MATCHED_INITIAL_V1',
        base_initial_sha256=base_initial_sha256, official_sha256=OFFICIAL_SHA256,
        official_scientific_tensors=len(scientific), official_scientific_numel=sum(value.numel() for value in scientific.values()),
        nonauthor_tensors_preserved=len(nonauthor), strict_full_author=True, no_partial_or_shape_adaptation=True,
        removed_wrapper_metadata=metadata, CPU_RNG_unchanged=True, CUDA_RNG_devices_unchanged=cuda_devices,
        module_modes_unchanged=True, requires_grad_unchanged=True,
        author_after_tensor_sha256={key: tensor_sha(after[PREFIX+key]) for key in scientific},
        nonauthor_after_tensor_sha256={key: tensor_sha(after[key]) for key in nonauthor})
