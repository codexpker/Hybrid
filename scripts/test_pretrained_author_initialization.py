"""Real original13 three-seed CPU initializer checks, no optimizer/training."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


def load_file(path):
    spec = importlib.util.spec_from_file_location('_author_initializer_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--snapshot-src', required=True)
    parser.add_argument('--initializer', required=True)
    parser.add_argument('--initializer-sha256', required=True)
    parser.add_argument('--official', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    began = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    init = load_file(args.initializer)
    if init.file_sha(args.initializer) != args.initializer_sha256:
        raise PermissionError('Frozen standalone initializer source required')
    workspace, snapshot = Path(args.workspace), Path(args.snapshot_src)
    first = workspace / 'results/hybrid_v1_stage1_full_N8000_seed2025_20261102_v1/source_train'
    receipt = json.loads((first/'receipt.json').read_text())
    for name, digest in receipt['source_sha256'].items():
        if init.file_sha(snapshot/'hybrid_tsad/hybrid_v1'/name) != digest:
            raise PermissionError('Immutable original13 source required')
    sys.path.insert(0, str(snapshot))
    from hybrid_tsad.hybrid_v1 import HybridV1, HybridV1Config, OfficialSources
    own = {'weight': torch.ones(2, 3)}
    valid = {'module.weight': torch.ones(2, 3), 'module.tau_n': torch.tensor(0.),
             'module.tau_a': torch.tensor(1.), 'module.tau_inited': torch.tensor(True)}
    scientific, _ = init.scientific_author_state(valid, own)
    assert set(scientific) == set(own)
    invalids = []
    wrong = dict(valid); wrong.pop('module.weight'); invalids.append(wrong)
    wrong = dict(valid); wrong['module.unexpected'] = torch.tensor(0.); invalids.append(wrong)
    wrong = dict(valid); wrong['module.weight'] = torch.ones(3, 2); invalids.append(wrong)
    wrong = dict(valid); wrong['module.weight'] = torch.ones(2, 3, dtype=torch.float64); invalids.append(wrong)
    wrong = dict(valid); wrong['module.tau_inited'] = torch.tensor(1.); invalids.append(wrong)
    wrong = dict(valid); wrong['weight'] = wrong.pop('module.weight'); invalids.append(wrong)
    for value in invalids:
        try:
            init.scientific_author_state(value, own)
        except ValueError:
            continue
        raise AssertionError('Invalid partial/shape/dtype/EMA/namespace input accepted')
    reports = []
    for seed in (2025, 2026, 2027):
        root = workspace / ('results/hybrid_v1_stage1_full_N8000_seed'+str(seed)+'_20261102_v1/source_train')
        config = json.loads((root/'config.json').read_text())
        rec = json.loads((root/'receipt.json').read_text())
        if config['seed'] != seed:
            raise ValueError('Matched seed initial identity differs')
        torch.manual_seed(seed)
        model = HybridV1(HybridV1Config.from_dict(config['model']), OfficialSources.from_workspace(workspace)).cpu()
        # Deliberately heterogeneous states exercise preservation, not resetting.
        model.train(); model.tf_backbone.author.encoder.eval()
        next(model.event_head.parameters()).requires_grad_(False)
        torch.rand(13)
        report = init.initialize_pretrained_author(model, base_initial_path=root/'initial.pt',
            base_initial_sha256=rec['files']['initial.pt'], official_path=args.official)
        base = torch.load(root/'initial.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(value, base[key]) for key,value in model.state_dict().items() if not key.startswith(init.PREFIX))
        assert report['official_scientific_tensors'] == 107 and report['nonauthor_tensors_preserved'] == 161
        report['seed'] = seed
        if seed == 2025:
            # Operator-level parity only: same normalized100 input/order/masks,
            # no H512 normalization, no fusion, author's abs projection.
            model.eval()
            x = torch.linspace(-1, 1, 100)
            window = torch.stack((torch.sin(3*x), torch.cos(2*x)), dim=-1)[None]
            window = (window-window.mean(1,keepdim=True))/torch.sqrt(window.var(1,keepdim=True,unbiased=False)+1e-5)
            history = torch.zeros(1,512,2); history[:,-100:] = window
            observed = torch.ones_like(history,dtype=torch.bool)
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(84551)
                native_repr,native_abs=model.tf_backbone.author(window,norm=0,copies=2)
                encoded=model.tf_backbone.encode(history,observed,mask_seed=84551)
                split_abs=model.tf_backbone.decode(encoded['native'],encoded['order'],projection='author_abs')
            repr_error=float((native_repr-encoded['native'][0].reshape_as(native_repr)).abs().max())
            abs_error=float((native_abs[:,0]-split_abs[0]).abs().max())
            if repr_error>1e-6 or abs_error>1e-6 or model.config.reconstruction_projection!='real':
                raise AssertionError('Exact author subpath semantic parity failed')
            report['operator_parity']=dict(representation_max_absolute=repr_error,author_abs_max_absolute=abs_error,
                scope='Standalone native author encode/decode only, not full Hybrid/native detector equivalence')
        reports.append(report)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='PASS_REAL_THREE_SEED_INITIALIZATION_CPU',seconds=time.monotonic()-began,
        seeds=[2025,2026,2027],schema_negative_tests=len(invalids),reports=reports,
        source_sha256=init.file_sha(args.initializer),torch=str(torch.__version__),new_training=False,GPUh=0)
    (output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status=result['status'],seconds=result['seconds'],seeds=result['seeds'],schema_negative_tests=len(invalids))))


if __name__=='__main__':
    main()
