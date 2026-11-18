"""Seven frozen TimeRCD B models on the existing SWaT role/event protocol.

This deliberately does not use the PSM/SMD long-event definition. SWaT keeps
its fixed512 CAL endpoints, QUERY608535:944919, long duration>512 and steady
start+512. Labels are joined only after all14 banks have sealed.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import timeradar_cross_variable_evaluation as io
from . import timercd_fixed_evaluation as fixed
from .alarm_complementarity import confirmed, longest_miss, point_counts
from .hybrid_v1 import evaluation as legacy
from .timeradar_smd_full_evaluation import read, local, checked
from .timercd_reference_repeats import finite_stats, SEEDS

PREP = 'preparations/left_multievent_mixed_u_20261111_v1'
OLD = 'results/left_multievent_mixed_u_20261111_v1'
LABEL_SHA = '3b36bcfb5c519e40b2d4c51d56aff36a015718b826436e2d91657d0864be14bc'
SPLIT_SHA = '07618aa08c944aedbc8477dbb2f6d81029c37a6bb3278dd6f2dd40cf07dece2f'
BASELINE_SHA = '9987795c25df4ef2fa7e2566219a9d4aa7d33b6ab62de49d790b37b192e7f1e3'
BASELINE_SEAL_SHA = '58630216d7acc535c72b3b25cc4874d4aa0d38d334ecb2b4d16248870a1d1e6c'
QUERY_START, QUERY_END = 608535, 944919
CAL_START, CAL_END = 446173, 495000
MODEL_IDS = ('official_zero', 'batch2025', 'reference2025', 'batch2026', 'reference2026', 'batch2027', 'reference2027')
STREAMS = ('strict', 'confirmed3')
METRICS = ('AP', 'ROC_AUC', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'F1',
    'M', 'Npred', 'Ntrue', 'auxiliary_any_hit_events', 'auxiliary_any_hit_recall',
    'one_to_one_eventPrecision', 'one_to_one_eventRecall', 'one_to_one_eventF1',
    'normal_FPR', 'normal_false_alarm_points', 'normal_alarm_fragments',
    'normal_alarm_fragments_per_1000_points', 'event_macro_coverage', 'event_point_coverage',
    'worst_event_coverage', 'long_steady_macro_coverage', 'long_steady_point_coverage',
    'max_event_miss_points', 'offline_first_hit_offset_median')


def terminal_gate(seal):
    expected = {f'score/{m}/{role}.npz' for m in MODEL_IDS for role in ('CAL', 'QUERY')}
    actual = [b['relative'] for b in seal['banks']]
    if (seal['status'] != 'COMPLETED' or not seal['no_labels_decoded']
            or len(actual) != len(set(actual)) or set(actual) != expected):
        raise ValueError('All14 fixed model/role banks required before labels open')
    return {b['relative']: b for b in seal['banks']}


def threshold(index, score):
    selected = np.linspace(CAL_START, CAL_END - 1, 512, dtype=np.int64)
    pos = np.searchsorted(index, selected)
    if np.any(pos >= len(index)) or not np.array_equal(index[pos], selected):
        raise ValueError('The exact original512 SWaT CAL endpoints must be present')
    return float(np.quantile(score[pos], .99, method='higher')), selected


def evaluate_score(index, score, labels, tau, available, *, query_stop=QUERY_END, ranking=None,
                   query_start=QUERY_START, original_length=QUERY_END, expected_events=(22, 9)):
    index, score, labels = np.asarray(index), np.asarray(score), np.asarray(labels)
    if (not 0 <= query_start < query_stop <= original_length
            or not np.array_equal(index, np.arange(query_start, query_stop)) or score.shape != index.shape
            or not np.isfinite(score).all() or labels.shape != (original_length,)
            or not np.isin(labels, [0, 1]).all()):
        raise ValueError('Keep the complete original label axis and exact fixed scored QUERY axis')
    if available.shape != index.shape or np.any(available < index):
        raise ValueError('Preserve actual offline input availability')
    scored = np.zeros(len(labels), bool); scored[index] = True
    available_full = np.arange(len(labels)); available_full[index] = available
    events = legacy._coverage(labels, scored, 0)
    eligible = [e for e in events if e['main_eligible']]
    if expected_events is not None and (len(eligible), sum(e['total_points'] > 512 for e in eligible)) != expected_events:
        raise ValueError('Original complete event and long-event counts changed')
    if ranking is None:
        ranking = dict(AP=float(average_precision_score(labels[index], score)),
                       ROC_AUC=float(roc_auc_score(labels[index], score)))
    ap, roc = ranking['AP'], ranking['ROC_AUC']
    strict = score > tau; rows, details, alarms = [], [], []
    for stream, bits in (('strict', strict), ('confirmed3', confirmed(index, strict))):
        full = np.zeros(len(labels), bool); full[index] = bits
        view, _ = legacy._event_view(labels, scored, full, available_full, events, 0, 'main_eligible')
        normal, _ = legacy._normal(labels, scored, full, 0)
        by_id = {e['event_id']: e for e in view['events']}
        coverage, misses, steady, first_offsets, available_offsets = [], [], [], [], []
        steady_alarm_points = steady_points = eligible_alarm_points = eligible_points = 0
        for event in events:
            e = dict(event); info = dict(by_id[e['event_id']]); lo, hi = e['local_start'], e['local_end']
            full_event = full[lo:hi]; ok = e['main_eligible']
            first = info.get('first_observed_alarm_index')
            offset = first - e['start'] if ok and first is not None else None
            availability_offset = info.pop('delay_points', None)
            info['offline_first_hit_offset'] = offset
            info['offline_availability_offset'] = availability_offset
            info['delay_points'] = None
            info['timing_status'] = 'OFFLINE_CONTEXT_OFFSETS_NOT_CAUSAL_DELAY'
            cover = float(full_event.mean()) if ok else None
            miss = longest_miss(full_event) if ok else None
            st = None; st_miss = None
            if ok:
                coverage.append(cover); misses.append(miss)
                eligible_alarm_points += int(full_event.sum()); eligible_points += hi - lo
                if offset is not None:
                    first_offsets.append(offset); available_offsets.append(availability_offset)
                if hi - lo > 512:
                    segment = full_event[512:]
                    st = float(segment.mean()); st_miss = longest_miss(segment)
                    steady.append(st); steady_alarm_points += int(segment.sum()); steady_points += len(segment)
            details.append(dict(stream=stream, **e, **{k:v for k,v in info.items() if k != 'event_id'},
                coverage=cover, longest_miss_points=miss, steady_after512_coverage=st,
                steady_longest_miss_points=st_miss,
                observed_alarm_points=int(full_event.sum()),
                observation_status='COMPLETE' if ok else 'UNOBSERVED_OR_INELIGIBLE',
                observed_count_is_not_complete_event_miss=not ok))
        row = dict(stream=stream, threshold=tau, AP=ap, ROC_AUC=roc,
            **point_counts(labels, scored, full), **view['metrics'],
            **{k:v for k,v in normal.items() if not isinstance(v, list)},
            event_macro_coverage=float(np.mean(coverage)),
            event_point_coverage=eligible_alarm_points / eligible_points,
            worst_event_coverage=min(coverage), long_event_count=len(steady),
            long_steady_macro_coverage=float(np.mean(steady)) if steady else None,
            long_steady_point_coverage=steady_alarm_points / steady_points if steady_points else None,
            long_steady_total_points=steady_points, max_event_miss_points=max(misses),
            delay_hit_only_median=None,
            offline_first_hit_offset_median=float(np.median(first_offsets)) if first_offsets else None,
            offline_availability_offset_median=float(np.median(available_offsets)) if available_offsets else None,
            mode='OFFLINE_CONTEXT', scored_points=len(index),
            long_definition='duration>512, steady from original start+512')
        row['delay_definition'] = 'Offline observed/available offsets, never causal detection delays'
        rows.append(row); alarms.extend(dict(stream=stream, **a) for a in view['alarms'])
    return rows, details, alarms, events


def parse_csv(path):
    rows = []
    with Path(path).open(encoding='utf-8', newline='') as handle:
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if value == '':
                    row[key] = None
                else:
                    try: row[key] = json.loads(value)
                    except (ValueError, TypeError): row[key] = value
            rows.append(row)
    return rows


def reused_baselines(workspace):
    w = Path(workspace)
    p = checked(w / OLD / 'fixed_evaluation/method_comparison.csv', BASELINE_SHA)
    seal_path = checked(w / OLD / 'fixed_scores/ALL_SCORES_THRESHOLDS_MASKS_SEALED.json', BASELINE_SEAL_SHA)
    seal = read(seal_path); inputs = {str(p): BASELINE_SHA, str(seal_path): BASELINE_SEAL_SHA}
    rows = []
    for old_name, name in (('noU', 'LEFT_native'), ('HistoryMean512', 'HistoryMean512')):
        descriptor = next(e for e in seal['descriptors'] if e['method'] == old_name)
        bank_path = w / OLD / 'fixed_scores' / (old_name + '.npz')
        checked(bank_path, descriptor['sha256']); inputs[str(bank_path)] = descriptor['sha256']
        with np.load(bank_path, allow_pickle=False) as z:
            assert np.array_equal(z['index'], np.arange(QUERY_START, QUERY_END))
            assert np.array_equal(z['CAL_index'], np.arange(CAL_START, CAL_END))
            tau, selected = threshold(z['CAL_index'], z['CAL_score'])
            assert np.array_equal(z['selected_CAL'], selected)
            assert tau == descriptor['thresholds']['own_CAL_q99']
        matches = [dict(r) for r in parse_csv(p) if r['method'] == old_name and r['track'] == 'own_CAL_q99']
        if {r['stream'] for r in matches} != set(STREAMS):
            raise ValueError('Existing baseline must have both exact SWaT stream definitions')
        for row in matches:
            if row['threshold'] != tau or row['Ntrue'] != 22 or row['long_event_count'] != 9:
                raise ValueError('Existing baseline axis/CAL/event eligibility differs')
            row.update(method=name, model_id=name, seed=None, status='REUSED_ACCEPTED_RESULT',
                metric_origin='HASH_REUSED_SWAT_20261111_CSV', scored_points=QUERY_END - QUERY_START,
                mode='CAUSAL_ENDPOINT', event_point_coverage=row['Recall'],
                training_permission='Target normal FIT/SELECT trained LEFT' if name == 'LEFT_native' else 'No neural training; strictly past512 predictor',
                calibration_permission='Fixed512 CAL threshold only' if name == 'LEFT_native'
                    else 'Fixed512 CAL per-channel median residual scale plus threshold; extra CAL fitting',
                offline_first_hit_offset_median=None, offline_availability_offset_median=None)
            rows.append(row)
    for stream in STREAMS:
        rows.append(dict(method='TimeRadar', model_id='TimeRadar', seed=None, stream=stream,
            status='NA_NO_COMPATIBLE_ROLE_BANK', AP=None, ROC_AUC=None,
            reason='Historical reproduction has not supplied a score bank with this exact SWaT QUERY/CAL role identity; no rerun to fill table'))
    return rows, inputs


def summarize(rows):
    lookup = {(r['method'], r.get('seed'), r['stream']): r for r in rows if r['method'] in ('batch', 'reference', 'official_zero')}
    summaries, differences, paired = [], [], []
    for stream in STREAMS:
        for method in ('batch', 'reference'):
            for metric in METRICS:
                values = [lookup[method, seed, stream].get(metric) for seed in SEEDS]
                summaries.append(dict(method=method, stream=stream, metric=metric,
                    **{f'seed{s}':v for s,v in zip(SEEDS,values)}, **finite_stats(values)))
        for comparator in ('batch', 'official_zero'):
            for metric in METRICS:
                values = []
                for seed in SEEDS:
                    a = lookup['reference', seed, stream].get(metric)
                    b = lookup[comparator, seed if comparator == 'batch' else None, stream].get(metric)
                    d = float(a-b) if a is not None and b is not None and np.isfinite(a) and np.isfinite(b) else None
                    values.append(d)
                    differences.append(dict(candidate='reference', reference=comparator, seed=seed,
                                             stream=stream, metric=metric, difference=d))
                paired.append(dict(candidate='reference', reference=comparator, stream=stream, metric=metric,
                    **{f'seed{s}':v for s,v in zip(SEEDS,values)}, **finite_stats(values),
                    positive=sum(v > 0 for v in values if v is not None),
                    tie=sum(v == 0 for v in values if v is not None),
                    negative=sum(v < 0 for v in values if v is not None)))
    return summaries, differences, paired


def run(workspace, output, config):
    start = time.monotonic(); w, output = Path(workspace).resolve(), Path(output).resolve()
    cp = local(w, config); cfg = read(cp); root = local(w, cfg['output'])
    bp = checked(local(w, cfg['data_binding']), cfg['data_binding_sha256']); binding = read(bp)
    if (binding['channels_count'] != 31 or binding['query_axis'] != [QUERY_START, QUERY_END]
            or cfg['query_axis'] != [QUERY_START, QUERY_END] or cfg['arms'] != ['B_fit_only']
            or [m['id'] for m in cfg['models']] != list(MODEL_IDS)):
        raise PermissionError('Keep the frozen SWaT31 roles, B-only track and seven checkpoints')
    sp = root / 'RAW_BANK_SEAL.json'; seal = read(sp); descriptors = terminal_gate(seal)
    if seal['config_sha256'] != io.sha(cp) or seal['data_binding_sha256'] != cfg['data_binding_sha256']:
        raise PermissionError('Terminal configuration or data binding changed')
    for relative, description in descriptors.items():
        checked(root / relative, description['sha256'])
    sourcefacts = checked(w / PREP / 'SWAT_SPLIT_FACTS.json', SPLIT_SHA)
    facts = read(sourcefacts)
    if facts['QUERY_event_count'] != 22 or facts['QUERY_long_gt512'] != 9:
        raise PermissionError('Preserve the original event qualification; do not select by scores')
    selection = checked(local(w, binding['cal_selection']['path']), binding['cal_selection']['sha256'])
    with np.load(selection, allow_pickle=False) as z:
        assert np.array_equal(z['index'], np.linspace(CAL_START, CAL_END - 1, 512, dtype=np.int64))
    record = dict(roles=[dict(role=e['role'], bounds_half_open=[e['original_start'], e['end']])
                         for e in binding['roles']])
    inputs = {str(p): io.sha(p) for p in (cp, bp, sp, sourcefacts, selection)}
    for model in cfg['models']:
        checked(local(w, model['checkpoint']), model['checkpoint_sha256'])
        rp = root / 'score' / model['id'] / 'SCORE_RECEIPT.json'; receipt = read(rp)
        if (receipt['model_id'] != model['id'] or receipt['permission'] != 'B_fit_only'
                or receipt['checkpoint_sha256'] != model['checkpoint_sha256']
                or receipt['selected_step'] != model['selected_step']
                or receipt['data_binding_sha256'] != cfg['data_binding_sha256']
                or receipt['optimizer_updates'] != 0 or receipt['normalization_count'] != 1
                or not receipt['frozen_parameters_and_buffers_unchanged'] or not receipt['no_labels_decoded']):
            raise PermissionError('Frozen checkpoint / B statistics / zero-update identity mismatch')
        inputs[str(rp)] = io.sha(rp)
    # All new scores and identities are fixed before any new label-array join.
    baseline, inherited_inputs = reused_baselines(w); inputs.update(inherited_inputs)
    lp = checked(w / PREP / 'SWAT_LABEL_METADATA.npz', LABEL_SHA); inputs[str(lp)] = LABEL_SHA
    with np.load(lp, allow_pickle=False) as z:
        labels = z['label'].astype(np.int8)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'EVALUATOR_EXECUTED_SNAPSHOT.py').open('xb') as handle:
        handle.write(Path(__file__).read_bytes())
    rows, all_events, all_alarms, costs = [], [], [], []
    coverage = None
    for model in cfg['models']:
        model_id = model['id']; values = {}
        for role in ('CAL', 'QUERY'):
            relative = f'score/{model_id}/{role}.npz'; path = root / relative
            values[role] = fixed.load_new_bank(path, record, role, 'B_fit_only', descriptors[relative],
                                                score_expression='logit')
            idx = values[role][0]; role_spec = next(e for e in binding['roles'] if e['role'] == role)
            with np.load(path, allow_pickle=False) as z:
                if not np.array_equal(z['eligible_scored'], idx >= role_spec['score_start']):
                    raise ValueError('Do not alter native input windows when applying old scored-axis crop')
                if role == 'CAL':
                    original = np.linspace(CAL_START, CAL_END - 1, 512, dtype=np.int64)
                    if not np.array_equal(z['original_cal_selected'], np.isin(idx, original)):
                        raise ValueError('Original512 CAL position mask changed')
        tau, selected = threshold(values['CAL'][0], values['CAL'][1])
        idx, scores, available = values['QUERY']; axis = np.arange(QUERY_START, QUERY_END)
        pos = np.searchsorted(idx, axis)
        if np.any(pos >= len(idx)) or not np.array_equal(idx[pos], axis):
            raise ValueError('New QUERY scores do not cover the fixed complete evaluation axis')
        result, event_details, alarm_details, coverage = evaluate_score(axis, scores[pos], labels, tau, available[pos])
        method = model['training_arm'] if model_id != 'official_zero' else 'official_zero'
        meta = dict(model_id=model_id, method=method, seed=model['seed'], status='OK',
                    checkpoint_sha256=model['checkpoint_sha256'], selected_step=model['selected_step'],
                    training_permission='Official synthetic pretrained, zero continuation' if model_id == 'official_zero'
                        else 'Synthetic supervised source continuation only; no target training',
                    calibration_permission='Original512 legitimate CAL points, threshold only; target FIT population statistics once',
                    metric_origin='NEW_SEALED_SWAT_BANK', threshold_rule='original512CAL_q99_higher_strict_greater')
        rows.extend(dict(**r, **meta) for r in result)
        all_events.extend(dict(**r, **meta) for r in event_details)
        all_alarms.extend(dict(**r, **meta) for r in alarm_details)
        folder = output / model_id; folder.mkdir()
        np.savez_compressed(folder / 'calculation_inputs.npz', index=axis, score=scores[pos],
            available_index=available[pos], full_labels=labels, selected_CAL_index=selected,
            selected_CAL_score=values['CAL'][1][np.searchsorted(values['CAL'][0], selected)])
        io._dump(folder / 'metrics.json', dict(rows=[dict(**r, **meta) for r in result],
            inputs={str(root / f'score/{model_id}/{role}.npz'): descriptors[f'score/{model_id}/{role}.npz']['sha256']
                    for role in ('CAL', 'QUERY')}, evaluator_sha256=io.sha(__file__)))
        osfile = root / 'score' / model_id / 'OS.json'; process = read(osfile)
        job = next(j for j in seal['jobs'] if j['model_id'] == model_id)
        costs.append(dict(model_id=model_id, method=method, seed=model['seed'],
            parameters=37056515, GPU_hours=job['GPU_hours'], peak_allocated_bytes=process['peak_allocated_bytes'],
            optimizer_updates=0, selected_checkpoint_reused=True))
    rows.extend(baseline)
    summary, differences, paired_summary = summarize(rows)
    rankings = [dict(model_id=r['model_id'], method=r['method'], seed=r.get('seed'),
                     AP=r.get('AP'), ROC_AUC=r.get('ROC_AUC'), status=r['status'],
                     scored_points=r.get('scored_points')) for r in rows if r['stream'] == 'strict']
    exports = {'comparison': rows, 'ranking': rankings, 'three_seed_summary': summary,
               'paired_per_seed': differences, 'paired_three_seed_summary': paired_summary,
               'per_event_complete_query': [e for e in all_events if e['main_eligible']],
               'event_coverage_all_original': coverage, 'parameters_and_compute': costs}
    for name, entries in exports.items():
        io._write_csv(output / (name + '.csv'), entries)
    io._write_csv(output / 'raw_event_details_all_original.csv', all_events)
    io._write_csv(output / 'alarm_details.csv', all_alarms)
    io._dump(output / 'RESULT.json', dict(status='COMPLETED', tables=exports, inputs=inputs,
        CPU_seconds=time.monotonic() - start, GPU_hours=0, inference_GPU_hours=seal['GPU_hours'],
        new_forward=False, optimizer_updates=0, new_model_evaluations=7, source_sha256=io.sha(__file__),
        label_sha256=LABEL_SHA, reused_baseline_rows=4, unavailable_TimeRadar_rows=2,
        event_kernel_sha256=io.sha(legacy.__file__), alarm_kernel_sha256=io.sha(confirmed.__code__.co_filename),
        source_model_uncertainty='Three source training seeds on the same one exposed SWaT recording and22 events; not independent domains',
        frozen_rules='B-only; original512CAL/higher/q99/>; original QUERY crop; duration>512/steady_after512',
        permissions='LEFT has target FIT/SELECT learning, H512 has CAL residual scale fitting; cross-permission comparison is descriptive only'))
    return exports


def publish(output, public):
    """Export only derived SWaT comparison/complete-event summaries and hashes."""
    output, public = Path(output), Path(public); result = read(output / 'RESULT.json')
    if result['status'] != 'COMPLETED':
        raise ValueError('Complete seven-model evaluation required')
    names = ('comparison', 'ranking', 'three_seed_summary', 'paired_per_seed',
             'paired_three_seed_summary', 'per_event_complete_query', 'parameters_and_compute')
    public.mkdir(parents=True, exist_ok=True)
    if any((public / (n + '.csv')).exists() for n in names) or (public / 'RESULTS.md').exists():
        raise FileExistsError('Do not overwrite published scientific results')
    identities = []
    for name in names:
        source = output / (name + '.csv'); text = source.read_text(encoding='utf-8')
        if any(s in text for s in ('/home/', '192.168.', 'D:\\', 'C:\\', 'ghp_', 'token=')):
            raise PermissionError('Potentially private export: ' + name)
        target = public / source.name; target.write_text(text, encoding='utf-8', newline='\n')
        identities.append(dict(name=source.name, raw_sha256=io.sha(source), public_sha256=io.sha(target)))
    rows = result['tables']['comparison']
    lookup = {(r['model_id'], r['stream']): r for r in rows}
    summary = {(r['method'], r['stream'], r['metric']): r for r in result['tables']['three_seed_summary']}
    model_ids = list(MODEL_IDS) + ['LEFT_native', 'HistoryMean512', 'TimeRadar']
    def f(v, percent=False):
        return 'NA' if v is None else f'{v * (100 if percent else 1):.6f}'
    def ms(r, percent=False):
        return f(r['mean'], percent) + ' ± ' + f(r['sample_SD'], percent)
    lines = ['# SWaT31: seven frozen TimeRCD models, original B-only protocol', '',
        'No model is trained or selected in this evaluation. The official zero-update model and the '
        'already source-selected batch/reference checkpoints for seeds2025/2026/2027 are scored once '
        'on the existing exposed SWaT roles. These are model-comparison results, not a new Hybrid '
        'architecture, lag mechanism or contamination-robustness claim.', '',
        '## Fixed common QUERY axis', '',
        '| Model | AP | ROC | Strict normal FPR (%) | C3 normal FPR (%) | C3 any-hit | C3 one-to-one matches | '
        'C3 event F1 (%) | C3 anomaly point coverage (%) | Long9 steady macro (%) | Long9 steady pooled (%) | C3 longest miss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in model_ids:
        strict, c3 = lookup[model, 'strict'], lookup[model, 'confirmed3']
        values = [f(strict.get('AP')), f(strict.get('ROC_AUC')), f(strict.get('normal_FPR'), True),
            f(c3.get('normal_FPR'), True), str(c3.get('auxiliary_any_hit_events', 'NA')),
            str(c3.get('M', 'NA')), f(c3.get('one_to_one_eventF1'), True),
            f(c3.get('event_point_coverage'), True), f(c3.get('long_steady_macro_coverage'), True),
            f(c3.get('long_steady_point_coverage'), True), str(c3.get('max_event_miss_points', 'NA'))]
        lines.append('| ' + model + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'TimeRadar is NA because no cached bank was qualified with this exact QUERY/CAL '
        'role identity; the old reproduction is retained and is not rerun. LEFT is a target-normal '
        'FIT/SELECT-trained external comparison, whereas the seven TimeRCD models have no target '
        'training. History Mean512 additionally fits per-channel median residual scales on the '
        'same512 CAL points before setting its threshold. Its operating results are not represented '
        'as equal calibration permission. No cross-permission superiority claim follows from this table.', '',
        '## Three source-seed summaries (mean ± sample SD)', '',
        '| Arm | AP | ROC | Strict normal FPR (%) | C3 normal FPR (%) | C3 event F1 (%) | '
        'C3 anomaly point coverage (%) | Long9 steady macro (%) | Long9 steady pooled (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in ('batch', 'reference'):
        keys = [('strict', 'AP', False), ('strict', 'ROC_AUC', False), ('strict', 'normal_FPR', True),
                ('confirmed3', 'normal_FPR', True), ('confirmed3', 'one_to_one_eventF1', True),
                ('confirmed3', 'event_point_coverage', True), ('confirmed3', 'long_steady_macro_coverage', True),
                ('confirmed3', 'long_steady_point_coverage', True)]
        lines.append('| ' + method + ' | ' + ' | '.join(ms(summary[method, stream, metric], pct)
                                                       for stream, metric, pct in keys) + ' |')
    lines += ['', 'The official frozen model appears once per stream and receives no artificial '
        'three-seed SD. Training-seed summaries use ddof1 on the same fixed recording and events, '
        'not three independent SWaT domains.', '',
        '## Fixed source-seed paired differences', '',
        '| Reference arm minus comparator | Metric / stream | Seed2025 | Seed2026 | Seed2027 | Mean ± sample SD |',
        '|---|---|---:|---:|---:|---:|']
    for comparator in ('batch', 'official_zero'):
        for metric, stream in [('AP', 'strict'), ('ROC_AUC', 'strict'),
                               ('normal_FPR', 'confirmed3'), ('event_point_coverage', 'confirmed3'),
                               ('long_steady_macro_coverage', 'confirmed3')]:
            row = next(r for r in result['tables']['paired_three_seed_summary']
                       if r['reference'] == comparator and r['metric'] == metric and r['stream'] == stream)
            values = [f(row[f'seed{s}']) for s in SEEDS] + [ms(row)]
            lines.append('| reference − ' + comparator + ' | ' + metric + ' / ' + stream + ' | '
                         + ' | '.join(values) + ' |')
    lines += ['', 'Pair differences above use raw fractions, not percentage-point scaling. Positive '
        'false-alarm differences are costs. Reference-minus-batch is the matched source-training '
        'comparison; reference-minus-official tests change from frozen pretrained weights, with '
        'different source-training exposure.', '',
        '## Preserved SWaT definitions and permissions', '',
        '- Original31 channel identities/order and raw FIT/CAL/QUERY roles are retained. Each model '
        'uses exactly the original FIT population mean/std once, with the official zero-std safeguard. '
        'No QUERY statistic or target training is used.',
        '- Full input roles remain CAL445661:495000 and QUERY608023:944919. Native5000-point windows '
        'are anchored at those original input starts. Only after forward inference are scores cropped '
        'to the old QUERY608535:944919 axis (336384 points). This is not a new window origin.',
        '- All seven use raw abnormal-minus-normal head logits and the same pre-existing512 CAL '
        'indices linspace(446173,494999,512,int64), q99/higher and strict greater-than. Full CAL is '
        'read out but is not substituted as a new calibration pool. CAL and QUERY never share context.',
        '- Main event evaluation retains22 fully covered, non-boundary-censored QUERY events. '
        'Long events are duration>512 (nine events), with steady coverage from original event start+512. '
        'This is the existing SWaT definition, not the PSM/SMD steady-after192 definition.',
        '- Strict and current-point C3 confirmation, any-hit and one-to-one event matching remain '
        'distinct. The full CSVs retain point counts, false-alarm fragments, event metrics and long '
        'missed spans. Only the fixed22 complete QUERY events are exported in per_event_complete_query; '
        'outside-QUERY events remain unobserved/NA in raw coverage records, never false misses.',
        '- TimeRCD uses noncausal full-window context: first-hit and availability offsets are offline '
        'descriptions, not online detection-delay claims. LEFT/H512 native endpoint permissions are '
        'disclosed separately. Processed sample positions do not establish a physical sampling clock.', '',
        '## Cost and provenance', '',
        f"New frozen inference cost is {result['inference_GPU_hours']:.9f} GPUh, with zero optimizer "
        'updates. parameters_and_compute.csv records each model cost and peak allocated memory. '
        'No historical training cost is charged again. Safe provenance identifies raw seals and '
        'accepted baseline tables without exporting raw observations, labels, weights or private '
        'host paths. This fixed evaluation does not automatically authorize a new model, normalization, '
        'window, score or threshold variant.', '']
    (public / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    io._dump(public / 'PROVENANCE_PUBLIC.json', dict(
        official_code_commit='b031773e323bc0e6c9dcd8c9119b800ec4d70f8c',
        raw_result_sha256=io.sha(output / 'RESULT.json'),
        executed_evaluator_source_sha256=result['source_sha256'], publication_source_sha256=io.sha(__file__),
        event_kernel_sha256=result['event_kernel_sha256'], alarm_kernel_sha256=result['alarm_kernel_sha256'],
        accepted_baseline_comparison_sha256=BASELINE_SHA, accepted_baseline_seal_sha256=BASELINE_SEAL_SHA,
        label_metadata_sha256=LABEL_SHA, split_facts_sha256=SPLIT_SHA,
        exports=identities, withheld='Raw observations, label arrays, weights, host paths and raw receipts.'))
    return public


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); run(a.workspace, a.output, a.config)
