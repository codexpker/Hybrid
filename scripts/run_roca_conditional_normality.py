"""One predeclared conditional-reference experiment on frozen FULL q+qr.

All four complete score banks are sealed before any original labels are read.
No training, forward, context/threshold search, or new layer selection.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tsad.roca_conditional_normality import METHODS, fit_reference, score_reference
from hybrid_tsad.hybrid_v1.scoring import weighted_quantile
from hybrid_tsad.hybrid_v1 import evaluation
from hybrid_tsad.hybrid_v1.public_native_evaluation import aggregate_dataset, event_diagnostics
from hybrid_tsad.hybrid_v1.controlled_wave_evaluation import require_metric_dependencies, reject_dependency_na


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def loc(config, path):
    if path.startswith('/home/a312/tsad_research/'):
        return Path(config['workspace']) / path.split('/home/a312/tsad_research/', 1)[1]
    return Path(path)


def bound(config, descriptor):
    path = loc(config, descriptor['path'])
    if sha(path) != descriptor['sha256']:
        raise PermissionError('Frozen input changed: ' + str(path))
    return path


def read_features(config, descriptor):
    path = bound(config, descriptor)
    with np.load(path, allow_pickle=False) as data:
        return data['index'].copy(), data['features'].copy()


def native_switch_proxy(native, index):
    # Change at score endpoint s means native command[s-1] != command[s-2].
    # Symmetric +/-16 is POSTHOC only, never a score input.
    changes = np.flatnonzero(np.any(native[1:, 1:25] != native[:-1, 1:25], axis=1)) + 2
    result = np.zeros(len(index), dtype=bool)
    for change in changes:
        result |= np.abs(index - change) <= 16
    return result


def main_run(config):
    started = time.monotonic()
    require_metric_dependencies()
    for path, digest in config['source_sha256'].items():
        if sha(loc(config, path)) != digest:
            raise PermissionError('Frozen consumer source changed')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    products = json.loads(bound(config, config['products']).read_text())
    lookup = {(row['method'], row['entity'], row['role'], row['layer']): row for row in products}
    old = json.loads(bound(config, config['old_query_banks']).read_text())
    banks = {row['entity']: row for row in old if row['method'] == 'RoCA_FULL_T'}
    contexts = json.loads(bound(config, config['context_support']).read_text())
    context_files = {Path(row['path']).parts[-2:] : row for row in contexts['derived_context_files']}
    entities = sorted(banks)
    if len(entities) != 53 or config['methods'] != list(METHODS):
        raise PermissionError('Keep frozen all53/four-score comparison')
    rng = np.random.default_rng(2025)
    arrays, calibrations, staged, switches, metadata, references = {}, {}, {}, {}, [], []
    for entity in entities:
        role = {}
        for name in ['FIT', 'CAL']:
            index, q = read_features(config, lookup['RoCA_FULL_T', entity, name, 'q'])
            other_index, qr = read_features(config, lookup['RoCA_FULL_T', entity, name, 'qr'])
            if not np.array_equal(index, other_index):
                raise ValueError('Two branches must share original native endpoints')
            role[name] = dict(index=index, q=q, qr=qr)
        bank = banks[entity]
        path = bound(config, bank['scores'])
        with np.load(path, allow_pickle=False) as data:
            role['QUERY'] = dict(index=data['index'].copy(), q=data['projection'].copy(), qr=data['reconstructed_projection'].copy())
        for name, data in role.items():
            descriptor = context_files[entity, name + '.npz']
            with np.load(bound(config, descriptor), allow_pickle=False) as command:
                if not np.array_equal(data['index'], command['index']):
                    raise ValueError('Original native context/feature axes differ')
                data['command_key'] = command['command_key'].copy()
        fit = role['FIT']
        reference = fit_reference(fit['q'], fit['qr'], fit['command_key'], reference_role='FIT')
        permutation = rng.permutation(64)
        sham_reference = fit_reference(fit['q'], fit['qr'], fit['command_key'][permutation], reference_role='FIT')
        reference_path = output / 'references' / (entity + '.npz')
        reference_path.parent.mkdir(exist_ok=True)
        keys = sorted(reference['contexts'])
        np.savez_compressed(reference_path, FIT_index=fit['index'], command_key=fit['command_key'],
            sham_permutation=permutation, global_center=reference['global_center'],
            context_key=np.array(keys), context_count=np.array([reference['contexts'][k]['count'] for k in keys]),
            context_centers=np.array([reference['contexts'][k]['center'] for k in keys]),
            sham_context_centers=np.array([sham_reference['contexts'][k]['center'] for k in keys]))
        references.append(dict(path=str(reference_path), sha256=sha(reference_path)))
        for name in ['CAL', 'QUERY']:
            data = role[name]
            scored = score_reference(reference, data['q'], data['qr'], data['command_key'], sham_reference=sham_reference)
            # Match counts and fallback eligibility; do not hide an invalid center.
            for key in keys:
                if reference['contexts'][key]['usable'] != sham_reference['contexts'][key]['usable']:
                    raise ValueError('SHAM and true reference fallback coverage differs')
            staged[entity, name] = data['index'], scored
            for method in METHODS:
                score, valid = scored['scores'][method]
                folder = output / 'raw' / entity / method
                folder.mkdir(parents=True, exist_ok=True)
                destination = folder / (name + '.npz')
                np.savez_compressed(destination, index=data['index'], available_index=data['index'], score=score, valid=valid,
                    command_key=data['command_key'], context_reference_count=scored['context_reference_count'],
                    conditioned=scored['conditioned'], invalid_context_center=scored['invalid_context_center'])
                metadata.append(dict(entity=entity, method=method, role=name, points=len(score), valid_points=int(valid.sum()),
                    conditioned_points=int(scored['conditioned'].sum()), path=str(destination), sha256=sha(destination)))
        native_path = bound(config, config['native_test'][entity])
        native = np.load(native_path, allow_pickle=False)
        qi = role['QUERY']['index']
        if not np.array_equal(qi, np.arange(512, len(native))):
            raise ValueError('Keep every original query endpoint')
        switches[entity] = native_switch_proxy(native, qi)
    for method in METHODS:
        complete = all(staged[entity, 'CAL'][1]['scores'][method][1].all() for entity in entities)
        pooled = np.concatenate([staged[entity, 'CAL'][1]['scores'][method][0] for entity in entities])
        if len(pooled) != 1696:
            raise ValueError('Exactly original32 CAL endpoints per entity')
        tau = weighted_quantile(pooled, np.ones(1696) / 1696, .99) if complete else None
        calibrations[method] = dict(threshold=tau, support_complete=complete, original_points=1696,
            calibration='equal32_per_entity_pooled_inverseCDF_q99_strict_gt')
        for entity in entities:
            index, score = staged[entity, 'QUERY']
            value, valid = score['scores'][method]
            arrays[method, 'SMAP', entity, banks[entity]['run']] = dict(index=index, available_index=index.copy(), score=value,
                valid=valid, invalid_reasons=[None if v else 'NONFINITE_ZERO_BRANCH_OR_REFERENCE' for v in valid])
    dump(output / 'ALL_SCORES_BEFORE_LABELS.json', dict(status='SEALED_ALL_FOUR_SCORES', banks=metadata, references=references,
        calibrations=calibrations, query_label_reads=0, new_forward=False, training=False))
    # First and only label access occurs after every new raw score and calibration.
    labels, catalogs, results, phases, subgroup_rows, paired_rows = {}, {}, {}, [], [], []
    static = {row['entity']: row for row in json.loads(bound(config, config['static_banks']).read_text())
              if row['domain'] == 'SMAP' and row['method'] == 'STATIC_FIT_DISTANCE'}
    accepted = Path(config['accepted_evaluation'])
    for entity in entities:
        old_root = accepted / 'entities' / 'SMAP' / entity / 'RoCA_FULL_T'
        provenance = json.loads((old_root / 'provenance.json').read_text())
        if sha(old_root / 'calculation_inputs.npz') != provenance['calculation_inputs_sha256']:
            raise PermissionError('Original posthoc truth changed')
        with np.load(old_root / 'calculation_inputs.npz', allow_pickle=False) as data:
            y = data['full_labels'].copy()
        key = ('SMAP', entity, banks[entity]['run'])
        labels[key] = y
        with (old_root / 'event_coverage.csv').open() as handle:
            catalog = list(csv.DictReader(handle))
        catalogs[entity] = catalog
        qi = staged[entity, 'QUERY'][0]
        sd = static[entity]
        with np.load(bound(config, dict(path=sd['path'], sha256=sd['sha256'])), allow_pickle=False) as data:
            if not np.array_equal(data['index'], qi):
                raise ValueError('Original fixed hard-normal mask axis differs')
            hard = (y[qi] == 0) & (data['score'] > sd['calibration']['threshold'])
        strict_by, confirmed_by = {}, {}
        for method in METHODS:
            k = (method,) + key
            tau = calibrations[method]['threshold']
            a = arrays[k]
            if tau is None:
                raise ValueError('CAL_SUPPORT_INCOMPLETE: no operating claim; no refitting permitted')
            r = evaluation.evaluate_arrays(**a, full_labels=y, threshold=tau, bank_complete=True,
                original_length=len(y), warmup_points=512, affiliation_identity=config['affiliation_identity']['path'])
            reject_dependency_na(r)
            source_ids = {(int(row['start']), int(row['end_exclusive'])): row['source_event_id'] for row in catalog}
            for event in r['event_coverage']:
                event['source_event_id'] = source_ids[event['start'], event['end_exclusive']]
            folder = output / 'entities' / 'SMAP' / entity / method
            evaluation.save_evaluation(folder, r, **a, full_labels=y, provenance=dict(config=config, reference='same_entity_nominal_FIT64', method=method))
            phases.extend(dict(method=method, domain='SMAP', entity=entity, **row) for row in event_diagnostics(r))
            points = {int(row['index']): row for row in r['point_coverage']}
            strict = np.array([points[int(t)]['strict_point'] is True for t in qi])
            confirmed = np.array([points[int(t)]['confirmed3_current'] is True for t in qi])
            strict_by[method], confirmed_by[method] = strict, confirmed
            for group, mask in [('OFFICIAL_NORMAL', y[qi] == 0), ('HARD_NORMAL_STATIC_CAL', hard),
                                ('NORMAL_COMMAND_SWITCH_PROXY_PM16', (y[qi] == 0) & switches[entity]),
                                ('OTHER_NORMAL', (y[qi] == 0) & ~hard & ~switches[entity])]:
                valid_mask = mask & a['valid']
                n = int(valid_mask.sum())
                subgroup_rows.append(dict(method=method, entity=entity, group=group, original_points=int(mask.sum()),
                    valid_points=n, strict_FP=int((strict & valid_mask).sum()), confirmed_FP=int((confirmed & valid_mask).sum()),
                    strict_FPR=float((strict & valid_mask).sum()/n) if n else None,
                    confirmed_FPR=float((confirmed & valid_mask).sum()/n) if n else None))
            del r['point_coverage']
            results[k] = r
        positive = y[qi] == 1
        global_s = arrays['GLOBAL_FIT64', *key]['score']
        cond_s = arrays['COMMAND_FIT64', *key]['score']
        for alarm_name, alarm in [('strict', strict_by), ('confirmed3', confirmed_by)]:
            global_tp = positive & alarm['GLOBAL_FIT64']
            paired_rows.append(dict(entity=entity, alarm=alarm_name, global_anomaly_hits=int(global_tp.sum()),
                conditional_lost_global_hits=int((global_tp & ~alarm['COMMAND_FIT64']).sum()),
                conditional_new_hits=int((positive & ~alarm['GLOBAL_FIT64'] & alarm['COMMAND_FIT64']).sum()),
                conditional_lower_score_among_global_hits=int((global_tp & (cond_s < global_s)).sum())))
    dataset = [aggregate_dataset(results, arrays, labels, 'SMAP', method) for method in METHODS]
    paired_entity = []
    for entity in entities:
        row = dict(entity=entity)
        for method in METHODS:
            metric = results[method, 'SMAP', entity, banks[entity]['run']]['ranking']
            row[method + '_AP'] = metric['AP']
            row[method + '_ROC_AUC'] = metric['ROC_AUC']
        paired_entity.append(row)
    table(output / 'event_phases.csv', phases)
    table(output / 'normal_subgroups.csv', subgroup_rows)
    table(output / 'global_hits_suppressed.csv', paired_rows)
    table(output / 'paired_entity_ranking.csv', paired_entity)
    dump(output / 'SUMMARY.json', dict(status='COMPLETED_ONE_CONDITIONAL_REFERENCE_MECHANISM', dataset=dataset,
        methods=METHODS, calibrations=calibrations, no_model_training=True, no_new_forward=True,
        scientific_boundary='Only COMMAND minus GLOBAL isolates command conditioning; anonymous commands, nominal FIT, historical development QUERY',
        old_baselines_reused_not_reevaluated=config['old_baseline_summaries']))
    dump(output / 'CONFIG.json', config)
    dump(output / 'RECEIPT.json', dict(status='COMPLETED', seconds=time.monotonic() - started,
        GPUh=0, entities=53, methods=4, new_query_score_banks=212, all_scores_before_labels=True))
    print(json.dumps(dict(status='COMPLETED', seconds=time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--trusted-config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.trusted_config_sha256:
        raise PermissionError('Frozen one-mechanism configuration required')
    main_run(json.loads(Path(args.config).read_text()))


if __name__ == '__main__':
    main()
