"""WL1 fixed CPU first wave, reusable orchestration, never implicitly executed.

Only the CLI's explicit root release can generate formal pools. Unit tests use
the small kernels, not this matrix. No LEFT, CONFIRM, alpha/threshold search on DEV.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import resource
import sys
import time
from datetime import datetime, timezone

import numpy as np
from hybrid_tsad import weak_lag as wl


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def dump(path, obj):
    with Path(path).open('x') as f:
        json.dump(jsonable(obj), f, indent=2, allow_nan=False)


def _write_csv(path, rows):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def require_release(config, release_path):
    """Small explicit release check, not a new scheduler or approval service."""
    wl.preview(config)
    release = json.loads(Path(release_path).read_text())
    needed = {'status': 'PRIMARY_RELEASED_WL_CPU_FIRST_WAVE', 'allow_generate': True,
              'allow_fit': True, 'allow_DEV_evaluation': True, 'allow_CONFIRM': False,
              'allow_LEFT': False, 'allow_GPU': False, 'config_sha256': sha(config)}
    if any(release.get(k) != value for k, value in needed.items()):
        raise ValueError('Explicit root first-wave release and exact config required')
    expected = release.get('source_sha256', {})
    for path in (Path(wl.__file__), Path(__file__)):
        if expected.get(path.name) != sha(path):
            raise ValueError('Root release must bind actual two shared source files')
    return release


def run_wave(config_path, release_path):
    release = require_release(config_path, release_path)
    out = Path(release['output']).resolve()
    if out.exists():
        raise ValueError('Immutable output already exists; never retry over results')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('identity', 'data', 'models', 'scores', 'evaluation', 'logs'):
        (out/name).mkdir()
    mono, utc = time.monotonic(), datetime.now(timezone.utc).isoformat()

    def check_budget():
        if time.monotonic()-mono > 1800 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > 4*1024**3:
            raise RuntimeError('CPU1800s/RSS4GiB limit exceeded; preserve partial, no retry')

    dump(out/'identity/config.json', json.loads(Path(config_path).read_text()))
    dump(out/'identity/release.json', release)
    dump(out/'identity/source.json', [{'path': str(p), 'sha256': sha(p)} for p in [Path(wl.__file__), Path(__file__)]])
    pools = {s: {r: [] for r in wl.ROLES} for s in ('LIN', 'WEAK', 'IRR')}
    truths, dev_values, seed_table = {}, {}, []
    for scene in ('LIN', 'WEAK'):
        for role, count in wl.ROLES.items():
            for run_id in range(count):
                arrays, irr, truth = wl.generate_run(scene, role, run_id)
                truths[(scene, role, run_id)] = truth
                for stream, seed in truth['seeds'].items():
                    seed_table.append({'scene': scene, 'role': role, 'run_id': run_id, 'stream': stream, 'seed': seed})
                for obs_scene, values in [(scene, arrays)]+([('IRR', irr)] if scene == 'WEAK' else []):
                    directory = out/'data'/obs_scene/role; directory.mkdir(parents=True, exist_ok=True)
                    for variant, data in values.items():
                        np.save(directory/f'run{run_id:02d}_{variant}.npy', data)
                    # Truth is separate and never passed into fit_model/calibrate.
                    dump(directory/f'run{run_id:02d}_truth.json', truth)
                    if role == 'DEV':
                        dev_values[(obs_scene, run_id)] = values
                    else:
                        pools[obs_scene][role].append(wl.NormalRun(obs_scene, role, run_id, values['N']))
                check_budget()
    _write_csv(out/'identity/seeds.csv', seed_table)
    all_metrics, all_events, matched, scores_for_stats, assertions, weaknesses = [], [], [], {}, [], {}
    for scene in ('LIN', 'WEAK', 'IRR'):
        fit, select, cal = [pools[scene][r] for r in ('FIT', 'SELECT', 'CAL')]
        scaler = wl.fit_scaler(fit)
        dump(out/'models'/f'{scene}_scaler.json', scaler)
        base_scene = 'WEAK' if scene == 'IRR' else scene
        dcf = np.concatenate([truths[(base_scene, 'FIT', i)]['fit_counterfactual_delta'] for i in range(4)])
        correlations = [wl.select_peak(fit, scaler, target)[1] for target in (0, 1)]
        a = .9 if scene == 'LIN' else .3
        weaknesses[scene] = {'a_squared': a*a, 'delta_over_H': 24/192,
                             'D_cf': float(np.sqrt(np.mean(dcf*dcf))/scaler['scale'][1]),
                             'coupling_SNR': a*a/(1-a*a), 'FIT_Pearson_profiles': correlations,
                             'IRR_D_cf_note': 'latent WEAK counterfactual; unavailable to IRR learner' if scene=='IRR' else None}
        models = [wl.fit_model(fit, select, scaler, name) for name in wl.MODELS]
        for model in models:
            dump(out/'models'/f'{scene}_{model["model"]}.json', model)
        cals = {m['model']: wl.calibrate(cal, scaler, m) for m in models}
        for name, calibration in cals.items():
            dump(out/'models'/f'{scene}_{name}_cal.json', calibration)
        # Freeze layers/counts independently of model scores.
        layers = {}
        for run_id in range(8):
            truth = truths[(base_scene, 'DEV', run_id)]
            layers[run_id] = wl.strata(truth['normal_lag'], truth['event_start'], truth['event_end'])
            endpoint = layers[run_id]['endpoint']
            counts = {}
            for variant, labels in truth['labels'].items():
                counts[variant] = {name: {'points':int(mask.sum()),
                    'positive_points':int(labels[endpoint[mask]].sum()),
                    'positive_fraction':float(labels[endpoint[mask]].mean()) if mask.any() else None}
                    for name,mask in layers[run_id].items() if name not in ('endpoint','input_start')}
            dump(out/'evaluation'/f'{scene}_run{run_id:02d}_layers.json',
                 {'masks':layers[run_id],'counts_before_DEV_scores':counts})
            values = dev_values[(scene, run_id)]
            count = wl.assert_sham_equivalence(values['N'], values['SB'], values['S0'], truth['normal_lag'],
                                               truth['event_start'], truth['event_end'], scaler, models)
            assertions.append({'scenario': scene, 'run_id': run_id, 'CORE_complete_window_equal': count,
                               'S0_full_array_equal': True, 'all_model_design_prediction_residual_equal': True})
        for model in models:
            name = model['model']; calibration = cals[name]
            for run_id in range(8):
                truth = truths[(base_scene, 'DEV', run_id)]; layer = layers[run_id]
                for variant, values in dev_values[(scene, run_id)].items():
                    scored = wl.score_run(values, scaler, model)
                    directory = out/'scores'/scene/name; directory.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(directory/f'run{run_id:02d}_{variant}.npz',
                        endpoint=layer['endpoint'], input_start=layer['input_start'],
                        prediction=np.stack([v['prediction'] for v in scored]),
                        standardized_target=np.stack([v['target'] for v in scored]),
                        residual=np.stack([v['residual'] for v in scored]),
                        raw_prediction=np.stack([v['prediction'] for v in scored])*scaler['scale']+scaler['mean'])
                    for readout, ca in calibration.items():
                        score = np.array([row[readout] for row in scored])
                        detail = wl.evaluate_readout(score, truth['labels'][variant], layer, ca['threshold'],
                                                     truth['event_start'], truth['event_end'], variant)
                        identity = {'scenario': scene, 'model': name, 'readout': readout, 'run_id': run_id, 'variant': variant}
                        for row in detail['layers']:
                            all_metrics.append(dict(identity, **row, threshold=ca['threshold']))
                            if variant == 'A' and row['layer'] == 'CORE':
                                scores_for_stats[(scene, name, readout, run_id)] = row['AP']
                        all_events.extend(dict(identity, **row) for row in detail['events'])
                        if detail['matched_normal_interval_hit'] is not None:
                            matched.append(dict(identity, matched_normal_interval_hit=detail['matched_normal_interval_hit']))
                        np.savez_compressed(directory/f'run{run_id:02d}_{variant}_{readout}.npz',
                                            score=score, y_score=np.array([v['y_'+readout] for v in scored]),
                                            alarm=detail['alarm'], labels=detail['label_at_endpoint'],
                                            endpoint=layer['endpoint'], threshold=ca['threshold'])
                check_budget()
    comparisons = {}
    specs = [('LIN_MULTI191_minus_AR191', 'LIN', 'MULTI191', 'last', 'AR191', 'last'),
             ('WEAK_MULTI191_minus_SYNC191', 'WEAK', 'MULTI191', 'last', 'SYNC191', 'last'),
             ('IRR_MULTI191_minus_SYNC191', 'IRR', 'MULTI191', 'last', 'SYNC191', 'last')]
    for scene in ('LIN', 'WEAK', 'IRR'):
        for readout in ('last', 'mean128'):
            specs.append((f'{scene}_MULTI64_minus_SYNC64_{readout}', scene,'MULTI64',readout,'SYNC64',readout))
        for name in ('SYNC64', 'MULTI64'):
            specs.append((f'{scene}_{name}_mean128_minus_last',scene,name,'mean128',name,'last'))
    for label, scene, m, r, other, r_other in specs:
        pairs = [(scores_for_stats[(scene,m,r,i)],scores_for_stats[(scene,other,r_other,i)]) for i in range(8)]
        comparisons[label] = [a-b if a is not None and b is not None else None for a,b in pairs]
    comparisons['WEAK_minus_IRR_increment'] = [a-b if a is not None and b is not None else None for a,b in zip(
        comparisons['WEAK_MULTI191_minus_SYNC191'], comparisons['IRR_MULTI191_minus_SYNC191'])]
    bootstrap, draws = wl.paired_bootstrap(comparisons)
    _write_csv(out/'evaluation/all_layers.csv', all_metrics)
    _write_csv(out/'evaluation/all_events.csv', all_events)
    _write_csv(out/'evaluation/matched_normal_intervals.csv', matched)
    dump(out/'evaluation/paired_bootstrap.json', bootstrap)
    np.save(out/'evaluation/bootstrap_run_draws.npy', draws)
    dump(out/'evaluation/assertions.json', assertions)
    dump(out/'evaluation/FIT_weakness_audit.json', weaknesses)
    check_budget()
    import sklearn
    dump(out/'logs/summary.json', {'status':'COMPLETED_PENDING_INDEPENDENT_REVIEW','started_utc':utc,
         'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-mono,
         'RSS_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
         'numpy':np.__version__,'sklearn':sklearn.__version__,'python':platform.python_version(),
         'torch_imported':'torch' in sys.modules,'GPU_hours':0,'CONFIRM_generated':False,
         'model_count':18,'readouts_per_scenario':8,'experimental_inference':'exploratory DEV, not new method or independent real data'})
    dump(out/'MANIFEST.json', [{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)}
                              for p in sorted(out.rglob('*')) if p.is_file()])
    return {'output':str(out),'status':'COMPLETED_PENDING_INDEPENDENT_REVIEW'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--release')
    args = parser.parse_args()
    if args.execute and not args.release:
        parser.error('--execute requires explicit root --release')
    result = run_wave(args.config,args.release) if args.execute else wl.preview(args.config)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
