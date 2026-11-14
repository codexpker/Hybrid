"""One source-calibrated P32 three-view comparison, no model or test fitting.

Old TF/MS and exact H1 are reused. Only missing same-role source forecasts may
be supplied by a frozen-forward producer. All three CALs and full mature QUERY
scores are sealed before the first label read in this consumer.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tsad.hybrid_v1.p32_readout import align_predictions, exact_float32, VARIANTS
from hybrid_tsad.hybrid_v1.controlled_mechanism import _unpack
from hybrid_tsad.hybrid_v1.scoring import fit_calibration, apply_calibration
from hybrid_tsad.hybrid_v1 import evaluation
from hybrid_tsad.hybrid_v1.public_native_evaluation import event_diagnostics
from hybrid_tsad.hybrid_v1.controlled_wave_evaluation import require_metric_dependencies, reject_dependency_na
from compare_psm_common_axis import load_method


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(asset):
    path = Path(asset['path'])
    if sha(path) != asset['sha256']:
        raise PermissionError('Frozen P32 artifact changed: ' + str(path))
    return path


def dump(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def old_origin(record, group):
    replay = record['replay']
    index = record['index']
    if (replay['input_identity'] != group['input_identity']
            or replay['forecast_history_bounds'] != [index - 512, index]
            or record['forecast_issued_index'] != index - 1
            or replay['actual_mask_seed'] != 2025 + index - group['role_start']
            or replay['unarrived_future_observed'] is not False):
        raise PermissionError('Original source forecast role/clock/seed identity differs')
    return dict(origin=index, issued=index - 1, prediction=exact_float32(record['prediction']),
                valid=_unpack(replay['forecast_valid'])[0],
                scale=exact_float32(replay['forecast_stats']['scale']).reshape(group['channels']),
                mean=exact_float32(replay['forecast_stats']['mean']).reshape(group['channels']))


def source_readouts(config, output, supplemental_parts):
    plan = json.loads(bound(config['source_plan']).read_text())
    old = json.loads(bound(config['old_source_records']).read_text())
    indexed = {(row['domain'], row['entity'], row['run'], row['index']): row for row in old}
    source_rows = {variant: [] for variant in VARIANTS}
    reports, files = [], []
    for group in plan['groups']:
        stream = tuple(group[key] for key in ('domain', 'entity', 'run'))
        required = np.asarray(group['required_origins'], np.int64)
        channels = group['channels']
        positions = {int(origin): position for position, origin in enumerate(required)}
        prediction = np.empty((len(required), 32, channels), np.float32)
        forecast_valid = np.empty((len(required), 32, channels), bool)
        issued = np.empty(len(required), np.int64)
        scales = np.empty((len(required), channels), np.float32)
        filled = np.zeros(len(required), bool)
        for origin in group['reused_origins']:
            data = old_origin(indexed[stream + (origin,)], group)
            where = positions[origin]
            prediction[where], forecast_valid[where] = data['prediction'], data['valid']
            scales[where], issued[where], filled[where] = data['scale'], data['issued'], True
        seen_new = set()
        for part in supplemental_parts:
            if tuple(part[key] for key in ('domain', 'entity', 'run')) != stream:
                continue
            if (part['role'] != 'CAL' or part['checkpoint_sha256'] != plan['parent_checkpoint']['sha256']
                    or part['input_identity'] != group['input_identity']):
                raise PermissionError('New forecast cannot cross source role/model/input')
            with np.load(bound(part['asset']), allow_pickle=False) as data:
                origin = data['origin']
                if (not np.array_equal(data['issued_index'], origin - 1)
                        or not np.array_equal(data['history_bounds'], np.stack((origin - 512, origin), axis=1))
                        or not np.array_equal(data['actual_mask_seed'], 2025 + origin - group['role_start'])
                        or np.any(origin - 512 < group['role_start']) or np.any(origin > group['role_end'])):
                    raise PermissionError('New forecast causal footprint/seed is illegal')
                for j, value in enumerate(origin):
                    value = int(value)
                    if value not in positions or value not in group['missing_origins'] or value in seen_new:
                        raise ValueError('Extra/duplicate or already archived source origin')
                    where = positions[value]
                    prediction[where] = exact_float32(data['prediction'][j])
                    forecast_valid[where] = data['forecast_valid'][j]
                    scales[where] = exact_float32(data['forecast_scale'][j]).reshape(channels)
                    issued[where], filled[where] = int(data['issued_index'][j]), True
                    seen_new.add(value)
        if seen_new != set(group['missing_origins']) or not filled.all():
            raise ValueError('Missing source origin; no role crossing or partial-horizon repair')
        targets = np.asarray(group['common_targets'], np.int64)
        original = [indexed[stream + (int(target),)] for target in targets]
        target = np.stack([exact_float32(row['replay']['current_target']).reshape(channels) for row in original])
        observed = np.stack([_unpack(row['replay']['current_observed']).reshape(channels) for row in original])
        h1_valid = np.stack([_unpack(row['replay']['forecast_valid'])[0, 0] for row in original])
        native_mask = observed & h1_valid
        sigma = np.stack([exact_float32(row['replay']['forecast_stats']['scale']).reshape(channels) for row in original])
        latest = np.searchsorted(required, targets)
        if not np.array_equal(scales[latest], sigma):
            raise ValueError('Common sigma_t must be exact original latest h1 past-scale')
        h1 = np.asarray([row['r_pred'] for row in original])
        safe_target = np.where(native_mask, target, 0.).astype(np.float64)
        latest_prediction = prediction[latest, 0].astype(np.float64)
        replay_h1 = (np.where(native_mask, np.square((latest_prediction-safe_target)/np.where(native_mask, sigma, 1.)), 0.)).sum(1) / native_mask.sum(1).clip(min=1)
        if not np.allclose(replay_h1, h1, rtol=5e-6, atol=1e-6):
            raise ValueError('Original source h1 native residual fails float32 arithmetic check')
        result = align_predictions(required, issued, prediction, forecast_valid, target_index=targets,
            target=target, native_h1_mask=native_mask, sigma_t=sigma, original_h1=h1)
        destination = output / ('SOURCE_CAL_' + group['domain'] + '.npz')
        np.savez_compressed(destination, index=targets, eligible=result['eligible'], reason=result['reason'],
            H1=result['variants']['H1'], MEAN=result['variants']['MEAN'], EVIDENCE=result['variants']['EVIDENCE'],
            r_tf=np.array([row['r_tf'] for row in original]), r_ms=np.array([row['r_ms'] for row in original]),
            between_origin_variance_normalized=result['between_origin_variance_normalized'])
        files.append(dict(path=str(destination), sha256=sha(destination)))
        reports.append(dict(domain=group['domain'], original_targets=len(group['original_targets']),
            common_targets=len(targets), eligible=int(result['eligible'].sum()),
            required_origins=len(required), reused_origins=len(group['reused_origins']), new_origins=len(seen_new)))
        if not result['eligible'].all():
            dump(output / 'SOURCE_ELIGIBILITY_INCOMPLETE.json', dict(groups=reports, files=files))
            raise ValueError('Frozen source CAL1018 support incomplete; no resampling or partial repair')
        difference = result['variants']['EVIDENCE'] - result['variants']['MEAN'] - result['between_origin_variance_normalized']
        if not np.allclose(difference, 0., atol=1e-10, rtol=0):
            raise ValueError('Source Jensen identity failed')
        for variant in VARIANTS:
            for row, residual in zip(original, result['variants'][variant]):
                source_rows[variant].append(dict(domain=row['domain'], entity=row['entity'], run=row['run'],
                    index=row['index'], r_tf=row['r_tf'], r_ms=row['r_ms'], r_pred=float(residual)))
    if any(len(rows) != 1018 for rows in source_rows.values()):
        raise ValueError('Exactly 509+509 same original source targets required')
    return source_rows, reports, files


def run(config):
    began = time.monotonic()
    require_metric_dependencies()
    for path, digest in config['source_sha256'].items():
        if sha(path) != digest:
            raise PermissionError('Frozen P32/evaluation consumer changed')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    # Verify producer identity document bytehash, without inventing its schema.
    producer_manifest = bound(config['supplemental_manifest'])
    bundle = json.loads(producer_manifest.read_text())
    if (bundle['status'] != 'COMPLETE_BOTH_SOURCE_CAL_GAPS' or bundle['total_new_origins'] != 17070
            or bundle['source_target_count'] != 1018
            or bundle['fixed_missing_plan']['sha256'] != config['source_plan']['sha256']):
        raise PermissionError('Incomplete frozen two-domain source origin bundle')
    supplemental_parts = []
    plan = json.loads(bound(config['source_plan']).read_text())
    groups = {(group['domain'], group['entity'], group['run']): group for group in plan['groups']}
    seen_streams = set()
    for product in bundle['products']:
        descriptor = product['manifest']
        manifest = json.loads(bound(descriptor).read_text())
        stream = tuple(manifest[key] for key in ('domain', 'entity', 'run'))
        if stream not in groups or stream in seen_streams:
            raise ValueError('Supplemental source streams must match each original group once')
        group = groups[stream]
        if (manifest['schema'] != 'FROZEN_PAST_ONLY_P32_ORIGINS_V1' or not manifest['complete']
                or manifest['checkpoint_sha256'] != plan['parent_checkpoint']['sha256']
                or manifest['plan_sha256'] != config['source_plan']['sha256']
                or manifest['science_sha256'] != config['forecast_science_sha256']
                or manifest['role'] != 'CAL' or manifest['input_identity'] != group['input_identity']
                or manifest['history'] != 512 or manifest['horizon'] != 32
                or manifest['records'] != group['missing_origin_count']
                or manifest['no_current_or_future_GT_passed_to_model'] is not True):
            raise PermissionError('Frozen forecast producer identity/permission differs')
        seen_streams.add(stream)
        for chunk in manifest['chunks']:
            path = chunk['path']
            if path.startswith('/home/a312/tsad_research/'):
                path = str(Path(config['workspace']) / path.split('/home/a312/tsad_research/', 1)[1])
            supplemental_parts.append(dict(domain=stream[0], entity=stream[1], run=stream[2], role='CAL',
                checkpoint_sha256=manifest['checkpoint_sha256'], input_identity=manifest['input_identity'],
                asset=dict(path=path, sha256=chunk['sha256'])))
    if seen_streams != set(groups):
        raise ValueError('Do not evaluate before both source domains are complete')
    source, source_reports, source_files = source_readouts(config, output, supplemental_parts)
    prepared = json.loads(bound(config['prepared_psm']).read_text())
    with np.load(bound(prepared['aligned']), allow_pickle=False) as data:
        aligned = {key: data[key].copy() for key in data.files}
    calibrations, arrays, raw_assets = {}, {}, []
    for variant in VARIANTS:
        cal = fit_calibration(source[variant], source=True)
        if cal['valid_windows'] != 1018 or cal['weights'] != {'r_tf': .25, 'r_ms': .25, 'r_pred': .5}:
            raise ValueError('Same CAL1018 and original weights required')
        calibrations[variant] = cal
        dump(output / (variant + '_calibration.json'), cal)
        dump(output / (variant + '_CAL1018_records.json'), source[variant])
        values = np.full(len(aligned['index']), np.nan)
        for position in np.flatnonzero(aligned['eligible']):
            row = {key: float(aligned[key][position]) for key in ('r_tf', 'r_ms')}
            row['r_pred'] = float(aligned[variant][position])
            scored = apply_calibration(row, cal)
            if not scored['valid']:
                raise ValueError('Eligible target has invalid fixed R3 components')
            values[position] = scored['score']
        arrays[variant] = dict(index=aligned['index'], score=values, valid=aligned['eligible'], available_index=aligned['index'])
        destination = output / (variant + '_mature_QUERY_scores.npz')
        np.savez_compressed(destination, **arrays[variant], raw_r_pred=aligned[variant])
        raw_assets.append(dict(variant=variant, path=str(destination), sha256=sha(destination)))
    for variant in VARIANTS:
        for key in ('r_tf', 'r_ms'):
            if calibrations[variant]['scales'][key] != calibrations['H1']['scales'][key]:
                raise ValueError('TF/MS input or source CAL permission changed between versions')
    dump(output / 'ALL_THREE_SCORES_BEFORE_LABELS.json', dict(status='SEALED_ALL_THREE_SOURCE_CAL_AND_QUERY',
        query_label_reads=0, CAL_targets=1018, source_groups=source_reports, source_files=source_files,
        raw_scores=raw_assets, calibrations=calibrations,
        supplemental_manifest=dict(path=str(producer_manifest), sha256=sha(producer_manifest))))
    # All source-only decisions are now fixed. Read original full truth exactly once.
    with np.load(bound(config['truth']), allow_pickle=False) as data:
        full_labels = data['full_labels'].copy()
    common = np.arange(543, 87744)
    location = np.searchsorted(aligned['index'], common)
    if not np.array_equal(aligned['index'][location], common) or not aligned['eligible'][location].all():
        raise ValueError('Frozen external common axis must remain, not a score-selected mask')
    y = full_labels[common]
    rows, phases, raw_pred_rows = [], [], []
    methods = []
    for variant in VARIANTS:
        methods.append(dict(id='Hybrid_' + variant + '_CAL1018', array=arrays[variant], online=True,
            threshold=calibrations[variant]['threshold'], protocol='same_source_checkpoint_no_training_same_CAL1018', variant=variant))
    for descriptor in config['old_comparators']:
        original = load_method(descriptor, full_labels)
        methods.append(dict(id=descriptor['id'], array=original, online=descriptor['online'],
            threshold=descriptor.get('threshold'), protocol=descriptor['protocol'],
            role_permissions=descriptor['role_permissions']))
    for method in methods:
        a = method['array']
        where = np.searchsorted(a['index'], common)
        if not np.array_equal(a['index'][where], common) or not a['valid'][where].all():
            raise ValueError('Every main-table method must use the same valid native points')
        score = a['score'][where]
        row = dict(method=method['id'], AP=float(average_precision_score(y, score)),
            ROC_AUC=float(roc_auc_score(y, score)), points=len(common), positive_points=int(y.sum()),
            protocol=method['protocol'], online=method['online'], threshold=method['threshold'],
            role_permissions=method.get('role_permissions', 'source_only_CAL1018_no_target_update'))
        if 'variant' in method:
            residual = aligned[method['variant']][location]
            row['raw_prediction_residual_AP'] = float(average_precision_score(y, residual))
            row['raw_prediction_residual_ROC_AUC'] = float(roc_auc_score(y, residual))
            raw_pred_rows.append({key: row[key] for key in ('method', 'raw_prediction_residual_AP', 'raw_prediction_residual_ROC_AUC')})
        if method['online']:
            availability = a.get('available_index', a.get('available'))[where]
            result = evaluation.evaluate_arrays(index=common, score=score, valid=np.ones(len(common), bool),
                available_index=availability, invalid_reasons=[None]*len(common), full_labels=full_labels,
                threshold=method['threshold'], bank_complete=True, original_length=len(full_labels),
                warmup_points=543, affiliation_identity=config['affiliation_identity']['path'])
            reject_dependency_na(result)
            folder = output / 'methods' / method['id']
            evaluation.save_evaluation(folder, result, index=common, score=score, valid=np.ones(len(common), bool),
                available_index=availability, invalid_reasons=[None]*len(common), full_labels=full_labels,
                provenance=dict(config=config, method=method['id'], source_CAL='1018_matched' if 'variant' in method else 'original_frozen_comparator'))
            method_phases = event_diagnostics(result)
            phases.extend(dict(method=method['id'], **phase) for phase in method_phases)
            row['strict'] = result['streams']['strict_point']
            row['confirmed3'] = result['streams']['confirmed3_current']
            row['coverage'] = result['event_coverage']
            row['operating_columns'] = operating_columns(result, method_phases)
        else:
            row['strict'] = row['confirmed3'] = None
            row['online_NA_reason'] = 'NATIVE_OFFLINE_WINDOW_AND_BATCH_METHOD_NO_ONLINE_CLAIM'
        rows.append(row)
    table(output / 'event_phases.csv', phases)
    table(output / 'raw_prediction_residual_ranking.csv', raw_pred_rows)
    operation_fields = list(dict.fromkeys(key for row in rows for key in row.get('operating_columns', {})))
    simple = []
    for row in rows:
        summary = {key: row.get(key) for key in ('method', 'AP', 'ROC_AUC', 'points', 'online', 'threshold', 'raw_prediction_residual_AP', 'role_permissions')}
        summary.update({key: row.get('operating_columns', {}).get(key) for key in operation_fields})
        simple.append(summary)
    table(output / 'main_comparison.csv', simple)
    table(output / 'internal_three_readouts.csv', simple[:3])
    dump(output / 'RESULT.json', dict(status='COMPLETED_FIXED_P32_THREE_READOUTS', seconds=time.monotonic()-began,
        common_axis=[543, 87744], rows=rows, calibrations=calibrations, source_reports=source_reports,
        interpretation='All32 origins; fixed original weights; only source CAL. H1 matched1018 differs from archived original1024 CAL. No readout chosen from test performance.',
        source_config=config, new_training=False, new_PSM_forward=False, GPUh_consumer=0))
    print(json.dumps(dict(status='COMPLETED', seconds=time.monotonic()-began, ranking=simple)))


def operating_columns(result, phases):
    """Output flattening only; definitions remain the accepted event kernel."""
    row = dict(original_points=result['coverage']['original_length'] if 'original_length' in result['coverage'] else len(result['point_coverage']),
        scored_points=result['ranking']['evaluated_points'],
        unknown_original_points=len(result['point_coverage'])-result['ranking']['evaluated_points'],
        original_events=len(result['event_coverage']),
        complete_scoring_events=sum(event['complete'] for event in result['event_coverage']),
        main_eligible_events=sum(event['main_eligible'] for event in result['event_coverage']))
    for name, stream in [('strict', 'strict_point'), ('confirmed3', 'confirmed3_current')]:
        current = result['streams'][stream]
        point, events, normal = current['point'], current['main_events'], current['normal']
        main = events['metrics']
        for key in ('TP', 'FP', 'Precision', 'Recall', 'F1'):
            row[name + '_' + key] = point[key]
        row[name + '_main_events_hit'] = main['M']
        row[name + '_main_events_total'] = main['Ntrue']
        row[name + '_event_precision'] = main['one_to_one_eventPrecision']
        row[name + '_event_recall'] = main['one_to_one_eventRecall']
        row[name + '_normal_exposure'] = normal['normal_exposure_points']
        row[name + '_normal_FPR'] = normal['normal_FPR']
        row[name + '_normal_FP_fragments'] = normal['normal_alarm_fragments']
        row[name + '_normal_FP_fragments_per1000'] = normal['normal_alarm_fragments_per_1000_points']
        delays = [event['delay_points'] for event in events['events'] if event['eligible'] and event['delay_points'] is not None]
        row[name + '_hit_only_delay_median_points'] = float(np.median(delays)) if delays else None
        row[name + '_right_censored_main_events'] = main['delay_right_censored_events']
        all_events = [phase for phase in phases if phase['stream'] == stream and phase['phase'] == 'all']
        eligible = [phase for phase in all_events if phase['eligible']]
        row[name + '_max_main_event_observed_miss_points'] = max((phase['event_longest_miss'] for phase in eligible), default=None)
        steady = [phase for phase in phases if phase['stream'] == stream and phase['phase'] == 'steady_after512' and phase['duration'] > 512]
        row[name + '_steady_long_events'] = len(steady)
        row[name + '_steady_original_points'] = sum(phase['total_points'] for phase in steady)
        row[name + '_steady_scored_points'] = sum(phase['scored_points'] for phase in steady)
        row[name + '_steady_unknown_points'] = sum(phase['unknown_points'] for phase in steady)
        row[name + '_steady_detected_points'] = sum(phase['detected_points'] for phase in steady)
        row[name + '_steady_all_zero_detected_events'] = sum(phase['detected_points'] == 0 for phase in steady)
        row[name + '_steady_main_events'] = sum(phase['eligible'] for phase in steady)
        row[name + '_steady_main_zero_detected_events'] = sum(phase['eligible'] and phase['detected_points'] == 0 for phase in steady)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--trusted-config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.trusted_config_sha256:
        raise PermissionError('Frozen P32 evaluation configuration required')
    run(json.loads(Path(args.config).read_text()))


if __name__ == '__main__':
    main()
