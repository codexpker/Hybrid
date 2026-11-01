"""Fixed native CATCH readout; import/preview never imports Torch or reads arrays.

No label IO or evaluation entry point. Execution requires a separate exact root
release. Author source remains external. Hooks only observe existing outputs.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import re

from . import catch_baseline as base
from .catch_normal_runs import PARAMS
from .left_normal_scores import (N, H, B, C, READOUTS, MANIFEST_SHA,
    validate_bank, load_feature, coordinates, calibrate_frozen)

OPERATION = 'CATCH_WL_C2_FIXED_SCORE'
HELPERS = ('__init__.py', 'catch_normal_scores.py', 'catch_normal_runs.py',
           'catch_baseline.py', 'left_normal_scores.py', 'left_baseline.py')
DERIVED = dict(freq='s', enc_in=2, dec_in=2, c_out=2, c_in=2,
               label_len=48, task_name='anomaly_detection')


def bank_seed(scene, role, run_id):
    if scene not in ('LIN', 'WEAK') or role not in ('CAL', 'DEV') or type(run_id) is not int or run_id not in range(8):
        raise PermissionError('Fixed scene/role/run required')
    return 2025 + {'LIN': 0, 'WEAK': 100000}[scene] + {'CAL': 1000, 'DEV': 2000}[role] + run_id


def seed_bank(torch, scene, role, run_id):
    import numpy as np
    seed = bank_seed(scene, role, run_id)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def state_sha(model):
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_sha(torch):
    import numpy as np
    import pickle
    return dict(python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        torch=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        cuda=hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest())


@contextmanager
def capture_native(torch, detector):
    """Identity-filtered global observer sees criteria created inside detect_score.

    Returns detached *copies* of already computed tensors, never a replacement.
    Exactly one model and one of each criterion call must occur per full bank.
    """
    captured = {'input': [], 'reconstruction': [], 'time_error': [], 'frequency_error': []}
    def observe(module, args, output):
        if module is detector.model:
            captured['input'].append(args[0].detach().clone())
            captured['reconstruction'].append(output[0].detach().clone())
        elif module is getattr(detector, 'temp_anomaly_criterion', None):
            captured['time_error'].append(output.detach().clone())
        elif module is getattr(detector, 'freq_anomaly_criterion', None):
            captured['frequency_error'].append(output.detach().clone())
        # Returning None preserves the original forward result.
    handle = torch.nn.modules.module.register_module_forward_hook(observe)
    try:
        yield captured
    finally:
        handle.remove()


def reduce_captured(torch, captured, native, alias):
    """Torch author channelmean/add order is canonical, including rounding."""
    import numpy as np
    if set(captured) != {'input', 'reconstruction', 'time_error', 'frequency_error'} or any(len(v) != 1 for v in captured.values()):
        raise ValueError('Exactly one native B128 forward and two criterion calls required')
    x = {k: v[0] for k, v in captured.items()}
    if any(tuple(v.shape) != (B, H, C) or v.dtype != torch.float32 or not torch.isfinite(v).all().item() for v in x.values()):
        raise ValueError('Finite float32 native [128,192,2] tensors required')
    canonical = torch.mean(x['time_error'], dim=-1) + .5 * torch.mean(x['frequency_error'], dim=-1)
    score = canonical.detach().cpu().numpy()
    native, alias = np.asarray(native), np.asarray(alias)
    if native.shape != (N,) or native.dtype != np.float32 or not np.array_equal(native, alias) or not np.array_equal(native, score.reshape(-1)):
        raise ValueError('Observed errors do not reproduce native detect_score exactly')
    y = x['time_error'][:, :, 1] + .5 * x['frequency_error'][:, :, 1]
    arrays = {k: v.detach().cpu().numpy() for k, v in x.items()}
    arrays.update(native_score=native, canonical_score=score,
        last=score[:, -1], mean128=score[:, 64:].mean(axis=1),
        y_last=y[:, -1].detach().cpu().numpy(),
        y_mean128=y[:, 64:].detach().cpu().numpy().mean(axis=1))
    return arrays


def preview(config_path):
    """Metadata-only validation; deliberately no checkpoint/source/feature IO."""
    cfg = json.loads(Path(config_path).read_text())
    if (cfg.get('operation') != OPERATION or cfg.get('upstream_commit') != base.COMMIT
        or cfg.get('seed') != 2025 or cfg.get('params') != PARAMS
        or cfg.get('derived_config') != DERIVED or cfg.get('source_manifest_sha256') != MANIFEST_SHA
        or cfg.get('max_seconds') != 180 or cfg.get('max_allocated_GiB') != 30):
        raise PermissionError('Fixed native C2 configuration required')
    for role in ('CAL', 'DEV'):
        validate_bank(cfg['banks'][role], cfg['scenario'], role)
    return cfg


def require_release(config_path, release_path):
    cfg = preview(config_path)
    release = json.loads(Path(release_path).read_text())
    if (cfg.get('status') != 'FROZEN' or release.get('status') != 'APPROVED'
        or release.get('operation') != OPERATION or release.get('approved_by') != 'root'
        or release.get('allow_labels') is not False or release.get('config_sha256') != base.sha(config_path)):
        raise PermissionError('Exact root no-label release required before runtime imports')
    if set(cfg['adapter_hashes']) != set(HELPERS) or release.get('adapter_hashes') != cfg['adapter_hashes']:
        raise PermissionError('Complete frozen shared-helper identities required')
    for name, digest in cfg['adapter_hashes'].items():
        if base.sha(Path(__file__).parent/name) != digest:
            raise ValueError('Shared helper identity differs: '+name)
    root = Path(cfg['source_root'])
    actual = {str(p.relative_to(root)).replace('\\', '/') for p in (root/'ts_benchmark/baselines/catch').rglob('*.py')}
    if set(cfg['source_hashes']) != actual | set(base.SUPPORT):
        raise ValueError('Complete external source inventory required')
    for rel, digest in cfg['source_hashes'].items():
        if base.sha(root/rel) != digest:
            raise ValueError('External source differs: '+rel)
    for key in ('checkpoint', 'scaler', 'training_report', 'train_acceptance', 'normal_inputs'):
        if base.sha(cfg[key]['path']) != cfg[key]['sha256']:
            raise ValueError('Training artifact differs: '+key)
    report = json.loads(Path(cfg['training_report']['path']).read_text())
    normal_inputs = json.loads(Path(cfg['normal_inputs']['path']).read_text())
    acceptance = json.loads(Path(cfg['train_acceptance']['path']).read_text())
    validate_training_scene(cfg, report, normal_inputs, acceptance)
    if (report['status'] != 'COMPLETED_PENDING_REVIEW' or report['params'] != PARAMS
        or report['seed'] != 2025 or report['source_hashes'] != cfg['source_hashes']
        or report['best_checkpoint_sha256'] != cfg['checkpoint']['sha256']
        or report['scaler_sha256'] != cfg['scaler']['sha256']):
        raise ValueError('Training lineage/config mismatch')
    if Path(cfg['output']).exists():
        raise FileExistsError(cfg['output'])
    return cfg, report


def validate_training_scene(cfg, report, normal_inputs, acceptance):
    """Pure metadata closure: a legal WEAK checkpoint cannot pose as LIN."""
    scene = cfg['scenario']
    if (normal_inputs.get('scenario') != scene
        or normal_inputs.get('source_manifest_sha256') != MANIFEST_SHA
        or report.get('formal_data_hashes') != normal_inputs
        or set(normal_inputs.get('roles', {})) != {'FIT', 'SELECT'}):
        raise PermissionError('Original normal training metadata crosses scene/source')
    for role, count in (('FIT', 4), ('SELECT', 2)):
        rows = normal_inputs['roles'][role]
        if len(rows) != count: raise PermissionError('Exact original normal run count required')
        for i, row in enumerate(rows):
            path = Path(row['path'])
            if ((row['scenario'], row['role'], row['run_id']) != (scene, role, i)
                or tuple(path.parts[-3:-1]) != (scene, role) or path.name != f'run{i:02d}_N.npy'
                or row['shape'] != [24576, 2] or row['dtype'] != 'float64'
                or row.get('allowed_in_training_loader') is not True
                or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
                raise PermissionError('Original FIT/SELECT normal reference differs')
    # The acceptance schema is deliberately explicit and bound by the release.
    if (acceptance.get('status') != 'PRIMARY_ACCEPTED_CATCH_C2_NORMAL_TRAIN'
        or acceptance.get('accepted_by') != 'root' or acceptance.get('scenario') != scene
        or acceptance.get('checkpoint_sha256') != cfg['checkpoint']['sha256']
        or acceptance.get('scaler_sha256') != cfg['scaler']['sha256']
        or acceptance.get('training_report_sha256') != cfg['training_report']['sha256']
        or acceptance.get('normal_inputs_sha256') != cfg['normal_inputs']['sha256']
        or acceptance.get('source_manifest_sha256') != MANIFEST_SHA):
        raise PermissionError('Root acceptance does not bind same scene/best/scaler')


def restore_detector(api, torch, cfg, report):
    """Same native config tuning on placeholder RangeIndex, no normal data IO."""
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    detector = api.CATCH(**PARAMS)
    detector.detect_hyper_param_tune(pd.DataFrame(np.zeros((3, 2)), columns=['u', 'y']))
    detector.config.task_name = 'anomaly_detection'; detector.config.c_in = C
    if any(getattr(detector.config, k) != v for k, v in DERIVED.items()):
        raise ValueError('Native RangeIndex training derived config mismatch')
    detector.model = api.CATCHModel(detector.config).to(detector.device)
    state = torch.load(cfg['checkpoint']['path'], map_location='cpu', weights_only=True)
    if set(state) != set(detector.model.state_dict()):
        raise ValueError('Native CATCH checkpoint keys must match exactly')
    detector.model.load_state_dict(state, strict=True)
    detector.early_stopping = SimpleNamespace(check_point=state)
    if state_sha(detector.model) != report['best_state_sha256']:
        raise ValueError('Loaded best scientific-state hash differs')
    with np.load(cfg['scaler']['path'], allow_pickle=False) as stored:
        if set(stored.files) != {'mean', 'scale', 'var', 'n_samples_seen'}:
            raise ValueError('Original native scaler schema differs')
        arrays = {k: stored[k].copy() for k in stored.files}
    if (any(arrays[k].shape != (C,) or not np.isfinite(arrays[k]).all() for k in ('mean', 'scale', 'var'))
        or (arrays['scale'] <= 0).any() or int(arrays['n_samples_seen']) != 98304):
        raise ValueError('Finite FIT4 native scaler required')
    detector.scaler.mean_ = arrays['mean']; detector.scaler.scale_ = arrays['scale']
    detector.scaler.var_ = arrays['var']; detector.scaler.n_samples_seen_ = arrays['n_samples_seen']
    detector.scaler.n_features_in_ = C
    detector.model.eval()
    return detector


def score_scene(config_path, release_path):
    """Run once only with external timeout180 and exact separately approved GPU."""
    cfg, training = require_release(config_path, release_path)
    import os
    import shutil
    import sys
    import time
    import numpy as np
    import pandas as pd
    import torch
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='Running', operation=OPERATION, label_files_read=False,
                  banks=[], config_sha256=base.sha(config_path), adapter_hashes=cfg['adapter_hashes'])
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(cfg['physical_gpu_index']) or not torch.cuda.is_available():
            raise RuntimeError('Exact approved CUDA mapping required; no fallback')
        torch.cuda.set_device(0); torch.set_num_threads(1)
        props = torch.cuda.get_device_properties(0)
        if str(props.uuid).lower().removeprefix('gpu-') != cfg['expected_gpu_uuid'].lower().removeprefix('gpu-'):
            raise RuntimeError('GPU UUID differs')
        torch.cuda.set_per_process_memory_fraction(30*2**30/props.total_memory, 0)
        torch.cuda.reset_peak_memory_stats()
        snapshot = out/'source'; snapshot.mkdir()
        for rel, digest in cfg['source_hashes'].items():
            target = snapshot/rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(cfg['source_root'])/rel, target)
            if base.sha(target) != digest: raise ValueError('Snapshot identity differs')
        for name in HELPERS: shutil.copyfile(Path(__file__).parent/name, snapshot/name)
        shutil.copyfile(config_path, out/'config.json'); shutil.copyfile(release_path, out/'release.json')
        api = base.external_api(Path(cfg['source_root']))
        detector = restore_detector(api, torch, cfg, training)
        loaded = {}
        for name, module in list(sys.modules.items()):
            if not name.startswith('ts_benchmark') or not getattr(module, '__file__', None): continue
            path = Path(module.__file__).resolve()
            rel = str(path.relative_to(Path(cfg['source_root']).resolve())).replace('\\', '/')
            if cfg['source_hashes'].get(rel) != base.sha(path):
                raise ValueError('Actually imported source not frozen: '+rel)
            loaded[name] = dict(path=str(path), sha256=base.sha(path))
        before = state_sha(detector.model); cal = []
        report.update(loaded_modules=loaded, model_config=vars(detector.config), state_before=before,
            checkpoint=cfg['checkpoint'], scaler=cfg['scaler'], torch_version=torch.__version__,
            numpy_version=np.__version__, GPU_UUID=str(props.uuid), eval_gumbel_preserved=True)
        for role in ('CAL', 'DEV'):
            for ref in cfg['banks'][role]:
                if time.monotonic()-started > 180: raise TimeoutError('No retry after budget')
                values = load_feature(ref, cfg['scenario'], role)
                seed = seed_bank(torch, cfg['scenario'], role, ref['run_id'])
                rng_before = rng_sha(torch)
                with capture_native(torch, detector) as capture:
                    native, alias = detector.detect_score(pd.DataFrame(values, columns=['u', 'y']))
                arrays = reduce_captured(torch, capture, native, alias)
                torch.cuda.synchronize()
                if len(detector.thre_loader) != 1 or len(detector.thre_loader.dataset) != B:
                    raise ValueError('Native single B128 loader differs')
                if state_sha(detector.model) != before or any(m.training for m in detector.model.modules()):
                    raise ValueError('Native evaluation state changed')
                if torch.cuda.max_memory_allocated() > 30*2**30: raise RuntimeError('Allocator cap exceeded')
                filename = f"{role}_run{ref['run_id']:02d}_{ref['variant']}.npz"
                with (out/filename).open('xb') as handle:
                    np.savez_compressed(handle, **arrays, **coordinates())
                report['banks'].append(dict(reference=ref, file=filename, sha256=base.sha(out/filename),
                    seed=seed, rng_before=rng_before, rng_after=rng_sha(torch),
                    raw_shape=[B, H, C], dtype='float32', native_score_calls=1, model_forward_calls=1,
                    canonical='native Torch mean(time_error,-1)+0.5*mean(frequency_error,-1)'))
                if role == 'CAL': cal.append(dict(reference=ref, **{k: arrays[k] for k in READOUTS}))
        for key in ('checkpoint', 'scaler', 'training_report', 'train_acceptance', 'normal_inputs'):
            if base.sha(cfg[key]['path']) != cfg[key]['sha256']: raise ValueError('Training artifact changed')
        if any(base.sha(Path(cfg['source_root'])/rel) != digest for rel, digest in cfg['source_hashes'].items()):
            raise ValueError('External source changed during scoring')
        report.update(status='Completed', calibration=calibrate_frozen(cal, cfg['scenario']),
            state_after=state_sha(detector.model), unscored_points=0,
            availability='retrospective H192; each point available at own window end; B128 execution waits full run')
    except BaseException as exc:
        report.update(status='Failed', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        report.update(wall_seconds=time.monotonic()-started, ended_unix=time.time())
        report['process_gpu_hours'] = report['wall_seconds']/3600
        if torch.cuda.is_initialized():
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--release')
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args()
    if args.execute:
        if not args.release: parser.error('Separate exact root --release required')
        score_scene(args.config, args.release)
    else:
        cfg = preview(args.config)
        print(json.dumps(dict(status='METADATA_PREVIEW_ONLY', scenario=cfg['scenario'], banks=40,
            arrays_read=False, models_instantiated=False, labels_read=False)))


if __name__ == '__main__': main()
