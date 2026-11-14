"""CPU-only identity/initialization/semantic map, no training or candidate save."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(left, right):
    common = sorted(set(left) & set(right))
    different = [key for key in common if left[key].dtype != right[key].dtype
                 or left[key].shape != right[key].shape or not torch.equal(left[key], right[key])]
    return dict(left_keys=len(left), right_keys=len(right), missing=sorted(set(left)-set(right)),
                extra=sorted(set(right)-set(left)), differing_tensors=different,
                exact=not different and set(left) == set(right))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--official', required=True)
    parser.add_argument('--official-sha256', required=True)
    parser.add_argument('--snapshot-src', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    begin = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    workspace = Path(args.workspace)
    official_path = Path(args.official)
    if sha(official_path) != args.official_sha256:
        raise PermissionError('Exact recovered original official public checkpoint required')
    train = workspace / 'results/hybrid_v1_stage1_full_N8000_seed2025_20261102_v1/source_train'
    config = json.loads((train / 'config.json').read_text())
    initial_path, trained_path = train / 'initial.pt', train / 'source.pt'
    receipt = json.loads((train / 'receipt.json').read_text())
    snapshot = Path(args.snapshot_src)
    core_identity = {}
    for name, digest in receipt['source_sha256'].items():
        path = snapshot / 'hybrid_tsad/hybrid_v1' / name
        if sha(path) != digest:
            raise PermissionError('Original13 source snapshot changed: ' + name)
        core_identity[name] = dict(path=str(path), sha256=digest)
    sys.path.insert(0, str(snapshot))
    from hybrid_tsad.hybrid_v1 import HybridV1, HybridV1Config, OfficialSources
    for path in (initial_path, trained_path):
        if sha(path) != receipt['files'][path.name]:
            raise PermissionError('Original V1 weight changed')
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    trained = torch.load(trained_path, map_location='cpu', weights_only=True)
    official = torch.load(official_path, map_location='cpu', weights_only=True)
    if not isinstance(official, dict) or not all(isinstance(value, torch.Tensor) for value in official.values()):
        raise ValueError('A real flat official tensor state_dict is required')
    if not all(key.startswith('module.') for key in official):
        raise ValueError('Pinned528 must have one uniform official DDP module namespace')
    official = {key[len('module.'):]: value for key, value in official.items()}
    torch.manual_seed(2025)
    sources = OfficialSources.from_workspace(workspace)
    settings = HybridV1Config.from_dict(config['model'])
    load_calls = []
    actual_load = torch.load
    def reject_constructor_load(*values, **keywords):
        load_calls.append(str(values[0]) if values else 'unknown')
        raise RuntimeError('Constructor tried to load pretrained weight')
    torch.load = reject_constructor_load
    try:
        model = HybridV1(settings, sources).cpu()
    finally:
        torch.load = actual_load
    fresh = model.state_dict()
    init_match = compare(initial, fresh)
    author = model.tf_backbone.author.state_dict()
    extra = set(official) - set(author)
    metadata = {}
    if extra:
        if extra != {'tau_n', 'tau_a', 'tau_inited'}:
            raise ValueError('Unrecognized official extra state, do not strip by guess')
        for key in sorted(extra):
            value = official[key]
            if value.shape != torch.Size([]) or (key == 'tau_inited' and value.dtype != torch.bool):
                raise ValueError('Unexpected author EMA metadata shape/dtype')
            metadata[key] = dict(value=value.item(), dtype=str(value.dtype), sha256=tensor_sha(value),
                usage='author experiment-wrapper EMA provenance only; not V1 parameters or calibration')
    learned = {key: value for key, value in official.items() if key not in metadata}
    mapping = []
    mismatched = []
    for key, value in author.items():
        source = learned.get(key)
        matches = source is not None and source.shape == value.shape and source.dtype == value.dtype and bool(torch.isfinite(source).all())
        mapping.append(dict(source_key=key, target_key='tf_backbone.author.' + key,
            shape=list(value.shape), dtype=str(value.dtype), numel=value.numel(), compatible=matches,
            official_tensor_sha256=tensor_sha(source) if source is not None else None,
            old_initial_equal=bool(torch.equal(initial['tf_backbone.author.'+key], source)) if matches else False,
            old_trained_equal=bool(torch.equal(trained['tf_backbone.author.'+key], source)) if matches else False))
        if not matches:
            mismatched.append(key)
    complete = not mismatched and set(learned) == set(author)
    # Actual old full V1 graph: use original trained weights, pure synthetic
    # CPU inputs, no optimizer or backward. This is not an accuracy experiment.
    model.load_state_dict(trained, strict=True)
    model.eval()
    calls = []
    handles = []
    for name, module in model.named_modules():
        if name in ('tf_backbone.author.net', 'tf_backbone.author.real_input_embed', 'tf_backbone.author.imag_input_embed',
                    'tf_backbone.author.encoder', 'tf_backbone.author.decoder', 'tf_backbone.project'):
            def hook(_module, inputs, result, name=name):
                calls.append(dict(module=name, input_shapes=[list(value.shape) for value in inputs if isinstance(value, torch.Tensor)],
                    output_shape=list(result.shape) if isinstance(result, torch.Tensor) else None,
                    output_dtype=str(result.dtype) if isinstance(result, torch.Tensor) else None))
            handles.append(module.register_forward_hook(hook))
    before = {key: tensor_sha(value) for key, value in model.state_dict().items()}
    x = torch.linspace(-2, 2, 512)
    values = torch.stack((torch.sin(x*3), torch.cos(x*2)+x*.1), dim=-1)[None].float()
    observed = torch.ones_like(values, dtype=torch.bool)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        forecast = model.forecast(values, observed, mask_seed=2025)
        reconstruction = model.reconstruct(values, observed, mask_seed=2025)
    for handle in handles:
        handle.remove()
    after = {key: tensor_sha(value) for key, value in model.state_dict().items()}
    if before != after:
        raise ValueError('Inspection changed original trained model tensors')
    all_author = [key for key in initial if key.startswith('tf_backbone.author.')]
    changed_author = [key for key in all_author if not torch.equal(initial[key], trained[key])]
    result = dict(status='COMPLETED_CPU_INITIALIZATION_AND_SEMANTIC_MAP', seconds=time.monotonic()-begin,
        environment=dict(torch=str(torch.__version__), python=sys.version, device='cpu'),
        original_source_config=dict(path=str(train/'config.json'), sha256=sha(train/'config.json'), checkpoint=config.get('checkpoint'), seed=config['seed'], steps=config['steps']),
        original_weights=dict(initial=dict(path=str(initial_path),sha256=sha(initial_path)), source=dict(path=str(trained_path),sha256=sha(trained_path))),
        constructor_torch_load_calls=load_calls, seed2025_fresh_vs_archived_initial=init_match,
        original13_core=core_identity, graph_source_manifest=model.source_manifest, official=dict(path=str(official_path),sha256=sha(official_path),
            flat_tensors=len(official), scientific_keys=len(learned), wrapper_metadata=metadata),
        mapping_complete=complete, missing_official_keys=sorted(set(author)-set(learned)), extra_scientific_keys=sorted(set(learned)-set(author)),
        mapping=mapping, author_tensor_count=len(all_author), original_stage1_changed_author_tensors=changed_author,
        author_parameters=sum(value.numel() for value in author.values()), total_stored_parameters=sum(value.numel() for value in trained.values()),
        actual_graph_calls=calls, technical_forecast_shape=list(forecast['prediction'].shape),
        technical_tf_reconstruction_shape=list(reconstruction['tf_reconstruction'].shape),
        original_trained_state_unchanged=True, no_new_training=True, no_optimizer=True, no_research_QUERY=True, GPUh=0,
        old_checkpoint_bridge='candidate_tf_lag.load_tf_checkpoint is a separate prototype path; not called by HybridV1 constructor/cli.construct',
        semantic_compatibility=dict(same_author_source_sha256='881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317',
            direct_modules=['net(100→256→1 order)', 'real_input_embed(5→64)', 'imag_input_embed(5→64)', 'encoder(complex native patch→256)', 'decoder(256→complex patch5)'],
            new_unmapped_modules=['tf_backbone.project', 'temporal', 'lag', 'ms_filter', 'fusion', 'tf_fusion_delta', 'ms_decoder', 'forecast_head', 'event_head'],
            not_native_equivalent=['visible-only H512 normalization vs native full100 normalization', 'per-window/visible-channel order vs native batch-channel order', 'copies2 vs native10 at evaluation', 'signed real inverse projection vs native abs', 'LEFT interaction delta before author decoder', 'endpoint R3 score vs native whole-window L1 and uncertainty', 'no TimeRadar CDL/EMA training objective imported']),
        proposed_single_change='only strict whole author scientific state load into tf_backbone.author after old seeded constructor, leave all other initial tensors/modes/RNG unchanged; pending root decision')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({key:result[key] for key in ('status','seconds','seed2025_fresh_vs_archived_initial','mapping_complete','author_tensor_count','author_parameters','constructor_torch_load_calls')}))


if __name__=='__main__':
    main()
