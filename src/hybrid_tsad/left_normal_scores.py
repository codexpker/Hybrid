"""Candidate fixed C2 readout. No model/data/torch import occurs at import time.

Execution is deliberately a separate release from training. Feature references
are allowlisted; truth never enters this module's scorer. Official LEFT is
external, unchanged. Evaluation adapters require a completed score seal first.
"""
import hashlib
import json
from pathlib import Path
import re

MANIFEST_SHA = '07287e30bb63dd7dae5f785955a370c4ee7420b6d0cbeb6363324d906f631bb7'
N, H, B, C = 24576, 192, 128, 2
READOUTS = ('last', 'mean128')


def sha(path):
    from .left_baseline import file_sha
    return file_sha(path)


def validate_reference(ref, scenario, role):
    """Deny labels/unknown roles before opening any feature bytes."""
    if scenario not in ('LIN', 'WEAK') or role not in ('CAL', 'DEV'):
        raise PermissionError('Only LIN/WEAK CAL and DEV feature banks admitted')
    rid, variant = ref['run_id'], ref['variant']
    if type(rid) is not int or rid not in range(8):
        raise PermissionError('Exactly run00..run07 admitted')
    variants = ('N',) if role == 'CAL' else ('A', 'N', 'S0', 'SB')
    path = Path(ref['path'])
    if (variant not in variants or ref['scenario'] != scenario or ref['role'] != role or
            tuple(path.parts[-3:-1]) != (scenario, role) or
            path.name != f'run{rid:02d}_{variant}.npy'):
        raise PermissionError('Not an exact permitted feature reference; labels denied')
    if not re.fullmatch('[0-9a-f]{64}', ref['sha256']):
        raise ValueError('Feature SHA256 required')
    return ref


def validate_bank(refs, scenario, role):
    for ref in refs:
        validate_reference(ref, scenario, role)
    got = [(r['run_id'], r['variant']) for r in refs]
    expected = [(i, v) for i in range(8) for v in
                (('N',) if role == 'CAL' else ('A', 'N', 'S0', 'SB'))]
    if got != expected:
        raise ValueError('Complete ordered bank required; no omitted/duplicate runs')


def load_feature(ref, scenario, role):
    import numpy as np
    validate_reference(ref, scenario, role)
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('Frozen feature SHA differs; no regeneration')
    values = np.load(ref['path'], allow_pickle=False)
    if values.shape != (N, C) or values.dtype != np.float64 or not np.isfinite(values).all():
        raise ValueError('Expected complete finite float64 [24576,2] run')
    return values


def fixed_windows(values, scaler, scenario):
    import numpy as np
    if (values.shape != (N, C) or values.dtype != np.float64 or
            not np.isfinite(values).all()):
        raise ValueError('Exact complete C2 run required')
    ids = [list(row) for row in scaler['fit_ids']]
    if (ids != [[scenario, 'FIT', i] for i in range(4)] or
            scaler['scenario'] != scenario or scaler['ddof'] != 0 or
            scaler['fit_points'] != 98304 or scaler['SELECT_used_for_fit'] is not False):
        raise PermissionError('Only frozen same-scene FIT4 scaler admitted')
    mean, scale = np.asarray(scaler['mean']), np.asarray(scaler['scale'])
    if (mean.shape != (C,) or scale.shape != (C,) or not np.isfinite(mean).all() or
            not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError('Finite FIT scaler required')
    # Identical float64 standardize then float32 cast to training dataset.
    return ((values-mean)/scale).reshape(B, H, C).astype(np.float32)


def coordinates():
    import numpy as np
    starts = np.arange(B, dtype=np.int64)*H
    ends = starts+H
    return {'absolute_run_index': np.arange(N, dtype=np.int64).reshape(B, H),
            'window_start': starts, 'window_end_exclusive': ends,
            'endpoint': ends-1, 'input_end': ends-1,
            'batch_input_end': np.full(B, N-1, dtype=np.int64),
            'channel_index': np.arange(C, dtype=np.int64)}


def reduce_raw(raw):
    """Canonical author NumPy channel mean, plus frozen 128-point readout."""
    import numpy as np
    if raw.shape != (B, H, C) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Exact finite official raw float32 [128,192,2] required')
    energy = np.concatenate([raw], axis=0)
    canonical = np.mean(np.array(energy.reshape(-1, C)), axis=-1).reshape(B, H)
    return {'channel_scores': energy, 'canonical_score': canonical,
            'last': canonical[:, -1],
            'mean128': energy[:, 64:, :].mean(axis=(1, 2)),
            'y_last': energy[:, -1, 1], 'y_mean128': energy[:, 64:, 1].mean(axis=1)}


def calibrate_frozen(records, scenario):
    """Only all eight CAL N score banks; 1015th ascending, strict >."""
    import numpy as np
    validate_bank([r['reference'] for r in records], scenario, 'CAL')
    result = {}
    for key in READOUTS:
        parts = [np.asarray(r[key]) for r in records]
        if any(p.shape != (B,) or not np.isfinite(p).all() for p in parts):
            raise ValueError('Eight complete finite CAL readouts required')
        pool = np.concatenate(parts)
        result[key] = {'threshold': float(np.sort(pool)[1014]), 'n': 1024,
                       'order_1based': 1015, 'rule': 'score > threshold',
                       'scores_sha256': hashlib.sha256(pool.tobytes()).hexdigest()}
    return result


def load_c2_checkpoint(model, state, expected_thop_keys):
    """Generalize original R exact-key load without assuming its 86-key count."""
    expected = set(expected_thop_keys)
    if (not expected or len(expected) != len(expected_thop_keys) or
            not all(k.rsplit('.', 1)[-1] in ('total_ops', 'total_params') for k in expected)):
        raise ValueError('Training-observed complete thop key list required')
    if set(state)-set(model.state_dict()) != expected:
        raise ValueError('Checkpoint extra keys differ from frozen training list')
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or set(result.unexpected_keys) != expected:
        raise ValueError('Missing scientific state or extra non-thop keys')
    return {'missing_keys': list(result.missing_keys), 'unexpected_keys': list(result.unexpected_keys)}


def require_score_release(config_path, release_path):
    """Release itself must be issued by root, never generated by this module."""
    cfg = json.loads(Path(config_path).read_text())
    release = json.loads(Path(release_path).read_text())
    if (release.get('status') != 'APPROVED' or release.get('approved_by') != 'root' or
            release.get('operation') != 'LEFT_WL_C2_FIXED_SCORE' or
            release.get('config_sha256') != sha(config_path) or
            release.get('module_sha256') != sha(__file__) or
            release.get('allow_labels') is not False):
        raise PermissionError('Exact root no-label score release required')
    if (cfg.get('status') != 'FROZEN' or cfg.get('operation') != 'LEFT_WL_C2_FIXED_SCORE' or
            cfg['seed'] != 2025 or cfg['source_manifest_sha256'] != MANIFEST_SHA or
            cfg['max_seconds'] != 180 or cfg['max_allocated_GiB'] != 30):
        raise ValueError('Frozen source/seed/budget required')
    scene = cfg['scenario']
    for role in ('CAL', 'DEV'):
        validate_bank(cfg['banks'][role], scene, role)
    for key in ('checkpoint', 'scaler'):
        if not re.fullmatch('[0-9a-f]{64}', cfg[key]['sha256']):
            raise PermissionError('Final checkpoint/scaler identities still PENDING')
        if sha(cfg[key]['path']) != cfg[key]['sha256']:
            raise ValueError('Final artifact hash mismatch')
    for rel, digest in cfg['source_hashes'].items():
        if sha(Path(cfg['source_root'])/rel) != digest:
            raise ValueError('Official source bytes differ')
    from . import left_baseline, left_normal_runs
    for module in (left_baseline, left_normal_runs):
        if release.get('adapter_hashes', {}).get(Path(module.__file__).name) != sha(module.__file__):
            raise PermissionError('Reused adapter bytes not released')
    return cfg


def score_scene(config_path, release_path):
    """Future separately released job; no automatic retry or label access.

    Caller must wrap in external 180s timeout (+ documented kill grace). Module
    caps Torch allocator to 30GiB; that is not a whole-card memory guarantee.
    """
    import time
    import os
    import shutil
    import sys
    from types import SimpleNamespace
    import numpy as np
    from . import left_baseline as baseline
    from .left_normal_runs import c2_model_config
    cfg = require_score_release(config_path, release_path)
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'Running', 'label_files_read': False, 'config_sha256': sha(config_path),
              'module_sha256': sha(__file__), 'seed': 2025, 'started_unix': time.time(),
              'banks': [], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    started = time.monotonic(); torch = None
    try:
        shutil.copyfile(config_path, out/'config.json')
        shutil.copyfile(release_path, out/'release.json')
        snapshot = out/'source'; snapshot.mkdir()
        for rel in cfg['source_hashes']:
            target = snapshot/rel; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(cfg['source_root'])/rel, target)
            if sha(target) != cfg['source_hashes'][rel]:
                raise ValueError('Source snapshot SHA differs')
        for module in (baseline, __import__('hybrid_tsad.left_normal_runs', fromlist=['left_normal_runs'])):
            shutil.copyfile(module.__file__, snapshot/Path(module.__file__).name)
        shutil.copyfile(__file__, snapshot/Path(__file__).name)
        if report['cuda_visible_devices'] != str(cfg['physical_gpu_index']):
            raise RuntimeError('Approved CUDA mapping differs; no fallback')
        root = Path(cfg['source_root'])
        torch, Left, loaded = baseline.model_runtime(cfg, root)
        props = torch.cuda.get_device_properties(0)
        if str(props.uuid).lower().removeprefix('gpu-') != cfg['expected_gpu_uuid'].lower().removeprefix('gpu-'):
            raise RuntimeError('Approved GPU UUID differs')
        model_cfg, diff = c2_model_config(baseline.read_json(root/'configs/PSM/model_configs_Left.json'))
        model = Left(SimpleNamespace(**model_cfg)).cuda()
        for rel, digest in cfg['source_hashes'].items():
            if not rel.endswith('.py'):
                continue
            module = sys.modules.get(rel[:-3].replace('/', '.'))
            if module is not None:
                path = Path(module.__file__).resolve()
                if path != (root/rel).resolve() or sha(path) != digest:
                    raise RuntimeError('Actually imported dependency identity differs')
                loaded[rel] = {'path': str(path), 'sha256': digest}
        report['checkpoint_load'] = load_c2_checkpoint(model, torch.load(cfg['checkpoint']['path'],
            map_location='cpu', weights_only=True), cfg['expected_thop_keys'])
        model.eval()
        before = baseline.score_state(model)
        if before['step'] != 0 or any(before['modes'].values()) or model._curriculum_lambda() != .3:
            raise RuntimeError('Expected eval step0 and official lambda_max=.3')
        scaler = baseline.read_json(cfg['scaler']['path'])
        report.update(loaded_modules=loaded, model_config=model_cfg, config_difference=diff,
                      checkpoint=cfg['checkpoint'], scaler=cfg['scaler'], state_before=before,
                      device={'name': props.name, 'uuid': str(props.uuid)},
                      torch_version=torch.__version__, numpy_version=np.__version__,
                      rng_before=sha_rng(torch))
        cal = []
        with torch.no_grad():
            for role in ('CAL', 'DEV'):
                for ref in cfg['banks'][role]:
                    bank_started = time.monotonic()
                    if time.monotonic()-started > 180:
                        raise TimeoutError('Fixed job budget exhausted; no retry')
                    values = load_feature(ref, cfg['scenario'], role)
                    windows = fixed_windows(values, scaler, cfg['scenario'])
                    raw, auxiliary = model.infer(torch.from_numpy(windows).to('cuda:0'), None, None, None)
                    if auxiliary is not None:
                        raise ValueError('Official infer auxiliary contract changed')
                    scores = reduce_raw(raw.detach().cpu().numpy())
                    baseline.require_frozen_score_state(before, baseline.score_state(model))
                    rel = f"{role}_run{ref['run_id']:02d}_{ref['variant']}.npz"
                    with (out/rel).open('xb') as f:
                        np.savez_compressed(f, **scores, **coordinates())
                    with np.load(out/rel, allow_pickle=False) as stored:
                        recomputed = reduce_raw(stored['channel_scores'])
                        if any(not np.array_equal(stored[k], v) for k, v in recomputed.items()):
                            raise RuntimeError('Serialized raw/readout reduction differs')
                    report['banks'].append({'reference': ref, 'file': rel, 'sha256': sha(out/rel),
                                            'wall_seconds': time.monotonic()-bank_started,
                                            'raw_shape': [B, H, C], 'dtype': 'float32'})
                    if role == 'CAL':
                        cal.append(dict(reference=ref, **{k: scores[k] for k in READOUTS}))
        report.update(status='Completed', calibration=calibrate_frozen(cal, cfg['scenario']),
                      state_after=baseline.score_state(model), rng_after=sha_rng(torch),
                      availability='retrospective H192 localization; no early warning', unscored_points=0)
        for key in ('checkpoint', 'scaler'):
            if sha(cfg[key]['path']) != cfg[key]['sha256']:
                raise RuntimeError('Frozen artifact changed during readout')
    except BaseException as exc:
        report.update(status='Failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        elapsed = time.monotonic()-started
        report.update(wall_seconds=elapsed, process_gpu_hours=elapsed/3600, ended_unix=time.time())
        if torch is not None and torch.cuda.is_initialized():
            report.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        (out/'report.json').write_text(json.dumps(report, indent=2))
    return report


def sha_rng(torch):
    return {'torch': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'cuda': hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}


def require_evaluation_seal(seal_path):
    """All 80 raw banks must already exist and be root-frozen before truth IO."""
    seal = json.loads(Path(seal_path).read_text())
    if (seal.get('status') != 'APPROVED' or seal.get('approved_by') != 'root' or
            seal.get('operation') != 'LEFT_WL_C2_LABEL_EVALUATION' or
            seal.get('module_sha256') != sha(__file__) or
            seal.get('source_manifest_sha256') != MANIFEST_SHA or
            seal.get('allow_truth') is not True or len(seal.get('banks', [])) != 80):
        raise PermissionError('Full raw-score seal and separate label permission required')
    identities = []
    for row in seal['banks']:
        validate_reference(row['reference'], row['reference']['scenario'], row['reference']['role'])
        r = row['reference']; identities.append((r['scenario'], r['role'], r['run_id'], r['variant']))
        if sha(row['path']) != row['sha256']:
            raise ValueError('Raw bank changed after freeze')
    expected = {(s, role, i, v) for s in ('LIN', 'WEAK') for role in ('CAL', 'DEV')
                for i in range(8) for v in (('N',) if role == 'CAL' else ('A', 'N', 'S0', 'SB'))}
    if len(set(identities)) != 80 or set(identities) != expected:
        raise PermissionError('Seal omits/duplicates required banks')
    return seal


def evaluate_fixed(scores, truth, variant, thresholds, *, seal_path):
    """Pure post-seal evaluation adapter; caller handles authorized truth IO.

    No label reader is provided here. Use require_evaluation_seal before calling
    a separately authorized reader. Both readouts always returned, never selected.
    """
    import numpy as np
    from . import weak_lag as wl
    require_evaluation_seal(seal_path)
    if variant not in ('A', 'N', 'S0', 'SB'):
        raise PermissionError('Unknown DEV variant')
    layer = wl.strata(np.asarray(truth['normal_lag']), truth['event_start'], truth['event_end'])
    return {r: wl.evaluate_readout(scores[r], np.asarray(truth['labels'][variant]), layer,
            thresholds[r]['threshold'], truth['event_start'], truth['event_end'], variant) for r in READOUTS}


def control_equivalence(normal, s0, sb, core):
    """Pure array assertion for the post-seal evaluator; no feature/truth IO."""
    import numpy as np
    if np.asarray(core).shape != (B,) or np.asarray(core).dtype != np.bool_:
        raise ValueError('Exact frozen 128-window CORE boolean mask required')
    keys = ('channel_scores', 'canonical_score', 'last', 'mean128', 'y_last', 'y_mean128')
    checks = {key: {'S0_full_equal': bool(np.array_equal(normal[key], s0[key])),
                    'SB_CORE_equal': bool(np.array_equal(normal[key][core], sb[key][core]))}
              for key in keys}
    if not all(all(row.values()) for row in checks.values()):
        raise ValueError('Frozen sham control equivalence failed; no metrics replacement')
    return {'core_windows': int(np.count_nonzero(core)), 'checks': checks}


def paired_core_ap_comparisons(ap):
    """ap[(scene,model,readout,run)] includes all eight DEV identities, NA kept."""
    from .weak_lag import paired_bootstrap
    differences = {}
    for scene in ('LIN', 'WEAK'):
        specs = [('last', m, 'last') for m in ('MULTI191', 'PEAK191', 'SYNC191')]
        specs += [('mean128', m, 'mean128') for m in ('MULTI64', 'SYNC64')]
        specs += [('mean128', 'LEFT', 'last')]
        for readout, other, other_readout in specs:
            pairs = [(ap[(scene, 'LEFT', readout, i)], ap[(scene, other, other_readout, i)]) for i in range(8)]
            differences[f'{scene}_LEFT.{readout}_minus_{other}.{other_readout}'] = [
                a-b if a is not None and b is not None else None for a, b in pairs]
    return paired_bootstrap(differences, repeats=10000)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--release')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        if not args.release:
            parser.error('--execute requires separate root --release')
        score_scene(args.config, args.release)
    else:
        cfg = json.loads(Path(args.config).read_text())
        for role in ('CAL', 'DEV'):
            validate_bank(cfg['banks'][role], cfg['scenario'], role)
        print(json.dumps({'status': 'PREVIEW_ONLY', 'config_status': cfg['status'],
                          'scenario': cfg['scenario'], 'banks': 40, 'windows': 5120,
                          'readouts': READOUTS, 'arrays_read': False,
                          'models_instantiated': False, 'labels_read': False}))


if __name__ == '__main__':
    main()
