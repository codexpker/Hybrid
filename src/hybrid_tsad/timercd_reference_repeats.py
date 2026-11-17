"""Fixed 2025/2026/2027 repetition, no pooling of time points or devices.

Seed2025 metrics are hash-reused without reevaluation. New seed banks are all
sealed before their labels open. Every seed uses the existing B-only evaluator.
Summaries report source-training randomness, not independent-data confirmation.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from . import timercd_reference_evaluation as evaluation
from . import timeradar_cross_variable_evaluation as ev
from .timeradar_smd_full_evaluation import read, local, checked, METRICS

SEEDS = (2025, 2026, 2027)
OLD_RESULT = 'results/timercd_reference_source_evaluation_20261117_v1/RESULT.json'
OLD_RESULT_SHA = 'ae3930022aa2c8a4130e0e2500adf66a36fe300182cfdc8feb2cb8e9d2d6046c'


def finite_stats(values):
    """Sample SD is across the three predeclared seeds, not query points."""
    valid = [float(x) for x in values if x is not None and np.isfinite(x)]
    sd = (0.0 if len(set(valid)) == 1 else float(np.std(valid, ddof=1))) if len(valid) > 1 else None
    return dict(expected_seeds=len(values), valid_seeds=len(valid),
                mean=float(np.mean(valid)) if valid else None,
                sample_SD=sd)


def summarize_groups(rows, keys, value_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    answer = []
    for identity, group in groups.items():
        by_seed = {r['seed']: r for r in group}
        if len(by_seed) != len(group) or set(by_seed) != set(SEEDS):
            raise ValueError('Exactly one observation from each fixed seed per summary group')
        for value_key in value_keys:
            values = [by_seed[s].get(value_key) for s in SEEDS]
            row = dict(zip(keys, identity)); row['quantity'] = value_key
            row.update({f'seed{s}': v for s, v in zip(SEEDS, values)})
            row.update(finite_stats(values)); answer.append(row)
    return answer


def aggregate(results):
    """Pure metric-table aggregation. Does not read scores, labels or CAL."""
    if set(results) != set(SEEDS):
        raise ValueError('Keep all three fixed training seeds')
    external_common = [dict(r) for r in results[2025]['common_rows'] if r['arm'] not in evaluation.MODELS]
    external_native = [dict(r) for r in results[2025]['native_rows'] if r['arm'] not in evaluation.MODELS]
    records = list(dict.fromkeys(r['record'] for r in results[2025]['common_rows']))
    if len(records) != 29 or 'PSM' not in records:
        raise ValueError('Fixed PSM plus all28 SMD entities required')
    common, native, macro, pair_rows, pair_summaries, costs = [], [], [], [], [], []
    for seed in SEEDS:
        result = results[seed]
        if result['status'] != 'COMPLETED':
            raise ValueError('No incomplete seed may be silently omitted')
        if [r for r in result['common_rows'] if r['arm'] not in evaluation.MODELS] != external_common:
            raise PermissionError('External baseline rows differ between repetitions')
        for scope, destination in (('common_rows', common), ('native_rows', native)):
            rows = [dict(r, seed=seed) for r in result[scope] if r['arm'] in evaluation.MODELS]
            if len(rows) != 58 or len({(r['record'], r['arm']) for r in rows}) != 58:
                raise ValueError('All29 entities and both arms must appear exactly once per seed')
            destination.extend(rows)
        macro.extend(dict(r, seed=seed) for r in result['common_SMD']['macro']
                     if r['method'] in evaluation.MODELS)
        for row in result['common_SMD_paired_summary']:
            pair_summaries.append(dict(row, seed=seed))
        pairs, _ = evaluation.paired(result['common_rows'], records)
        pair_rows.extend(dict(r, seed=seed) for r in pairs)
        costs.extend(dict(row, seed=seed,
                          execution_origin='REUSED_ACCEPTED_SEED2025' if seed == 2025 else 'NEW_FIXED_REPETITION')
                     for row in result['parameters_and_compute'])
    entity_metric_rows = []
    for row in common:
        for metric in METRICS:
            entity_metric_rows.append(dict(seed=row['seed'], record=row['record'], arm=row['arm'],
                                           metric=metric, value=row.get(metric)))
    entity = summarize_groups(entity_metric_rows, ('record', 'arm', 'metric'), ('value',))
    macro_summary = summarize_groups(macro, ('method', 'metric'), ('macro_mean', 'median'))
    entity_pairs = summarize_groups(pair_rows, ('record', 'candidate', 'reference', 'metric'), ('difference',))
    paired_summary = summarize_groups(pair_summaries, ('candidate', 'reference', 'metric'),
                                      ('mean_difference', 'median_difference'))
    return dict(per_seed_common=common, per_seed_native=native, external_common=external_common,
        external_native=external_native, per_entity_three_seed_summary=entity,
        per_seed_SMD_macro_median=macro, SMD_macro_median_three_seed_summary=macro_summary,
        per_seed_paired_deltas=pair_rows, per_entity_paired_three_seed_summary=entity_pairs,
        per_seed_SMD_paired_summary=pair_summaries, SMD_paired_three_seed_summary=paired_summary,
        parameters_and_compute=costs)


def run(workspace, output, config2026, config2027):
    begin = time.monotonic(); w, output = Path(workspace).resolve(), Path(output).resolve()
    old = checked(w / OLD_RESULT, OLD_RESULT_SHA)
    configs = {2026: local(w, config2026), 2027: local(w, config2027)}
    inputs = {str(old): OLD_RESULT_SHA}
    # Both repetitions must reach the complete terminal state before this
    # orchestrator runs even the first new seed's fixed QUERY evaluation.
    for seed, path in configs.items():
        cfg = read(path)
        if cfg['train_seed'] != seed:
            raise ValueError('Configuration must identify the prespecified seed')
        sp = local(w, cfg['output']) / 'RAW_BANK_SEAL.json'; seal = read(sp)
        bp = local(w, cfg['data_binding']); records = {r['id']: r for r in read(bp)['records']}
        evaluation.terminal_gate(seal, records)
        if seal['config_sha256'] != ev.sha(path):
            raise PermissionError('Frozen repetition config identity mismatch')
        inputs[str(path)] = ev.sha(path); inputs[str(sp)] = ev.sha(sp)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'AGGREGATOR_EXECUTED_SNAPSHOT.py').open('xb') as handle:
        handle.write(Path(__file__).read_bytes())
    results = {2025: read(old)}
    for seed, path in configs.items():
        destination = output / f'seed{seed}'
        evaluation.run(w, destination, path)
        result_path = destination / 'RESULT.json'
        results[seed] = read(result_path); inputs[str(result_path)] = ev.sha(result_path)
    combined = aggregate(results)
    for name, rows in combined.items():
        ev._write_csv(output / (name + '.csv'), rows)
    ev._dump(output / 'RESULT.json', dict(status='COMPLETED', seeds=list(SEEDS),
        reused_seed2025_result_sha256=OLD_RESULT_SHA, tables=combined, inputs=inputs,
        CPU_seconds=time.monotonic() - begin, GPU_hours=0, new_forward=False,
        source_sha256=ev.sha(__file__), evaluator_sha256=ev.sha(evaluation.__file__),
        evidence_scope='Three source-training randomness repetitions on the same exposed PSM/SMD '
                       'points and events. No pooling of query points; not84 independent entities '
                       'or independent data domains. Sample SD uses ddof1 and missing metrics stay NA.'))
    return combined


def publish(output, public):
    """Safe repetition tables; fixed external scores appear only once."""
    output, public = Path(output), Path(public)
    result = read(output / 'RESULT.json')
    if result['status'] != 'COMPLETED' or result['seeds'] != list(SEEDS):
        raise ValueError('Complete three-seed result required')
    public.mkdir(parents=True, exist_ok=True)
    names = tuple(result['tables'])
    if any((public / (n + '.csv')).exists() for n in names) or (public / 'RESULTS.md').exists():
        raise FileExistsError('Do not overwrite a published repetition result')
    exports = []
    for name in names:
        source = output / (name + '.csv'); text = source.read_text(encoding='utf-8')
        if any(k in text for k in ('/home/', '192.168.', 'D:\\', 'C:\\', 'ghp_', 'token=')):
            raise PermissionError('Potentially private export: ' + name)
        target = public / source.name; target.write_text(text, encoding='utf-8', newline='\n')
        exports.append(dict(name=source.name, raw_sha256=ev.sha(source), public_sha256=ev.sha(target)))
    tables = result['tables']
    entity = {(r['record'], r['arm'], r['metric']): r
              for r in tables['per_entity_three_seed_summary']}
    macro = {(r['method'], r['metric'], r['quantity']): r
             for r in tables['SMD_macro_median_three_seed_summary']}
    external = {(r['record'], r['arm']): r for r in tables['external_common']}
    def number(v, percent=False):
        return 'NA' if v is None else f'{v * (100 if percent else 1):.6f}'
    def mean_sd(row, percent=False):
        return number(row['mean'], percent) + ' ± ' + number(row['sample_SD'], percent)
    metrics = ('AP', 'ROC_AUC', 'strict_normal_FPR', 'confirmed3_normal_FPR',
               'confirmed3_one_to_one_eventRecall', 'confirmed3_one_to_one_eventF1',
               'confirmed3_complete_event_point_coverage', 'confirmed3_long_steady_macro_coverage')
    lines = ['# Historical-reference source training: fixed three-seed repetition', '',
        'Source seeds2025/2026/2027 repeat exactly the accepted batch/reference training recipe. '
        'Seed2025 metrics are hash-reused without retraining, rescoring or reevaluation. Only the two '
        'prespecified new training seeds are evaluated. Source data and roles stay fixed; within each '
        'seed, batch/reference share draws, masks and official initialization. Source SELECT masks '
        'remain fixed. Target inference remains original B_fit_only with the fixed logit score, '
        'context and lawful CAL quantile. No QUERY feedback chooses a variant.', '',
        '## PSM common-axis repetitions', '',
        '| Method | AP mean ± sample SD | ROC mean ± sample SD | Strict normal FPR (%) | C3 normal FPR (%) | '
        'C3 one-to-one event recall (%) | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in evaluation.MODELS:
        values = [mean_sd(entity['PSM', model, metric], metric not in ('AP', 'ROC_AUC')) for metric in metrics]
        lines.append('| ' + model + ' | ' + ' | '.join(values) + ' |')
    for model in tuple(m for m in evaluation.COMMON_METHODS if m not in evaluation.MODELS):
        row = external['PSM', model]
        values = [number(row.get(metric), metric not in ('AP', 'ROC_AUC')) for metric in metrics]
        lines.append('| ' + model + ' (reused fixed result) | ' + ' | '.join(values) + ' |')
    lines += ['', '## SMD28: per-seed entity macro, then mean ± sample SD across seeds', '',
        '| Method | AP macro | AP median | ROC macro | Strict normal FPR (%) | C3 normal FPR (%) | '
        'C3 one-to-one event recall (%) | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in evaluation.MODELS:
        values = [mean_sd(macro[model, 'AP', 'macro_mean']), mean_sd(macro[model, 'AP', 'median'])]
        values += [mean_sd(macro[model, metric, 'macro_mean'], metric not in ('AP', 'ROC_AUC')) for metric in metrics[1:]]
        lines.append('| ' + model + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'The table first averages eligible metrics over the same fixed SMD28 entities within '
        'each seed, then summarizes the three resulting seed values. It does not concatenate84 '
        'entity-seed sequences or treat them as independent domains. Entity medians are summarized '
        'separately from macros; noneligible long-event metrics stay NA.', '',
        '## Per-seed PSM detection counts', '',
        '| Seed | Method | AP | ROC | C3 any-hit | C3 one-to-one matches | C3 longest missed span |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in tables['per_seed_common']:
        if row['record'] == 'PSM':
            lines.append(f"| {row['seed']} | {row['arm']} | {number(row['AP'])} | {number(row['ROC_AUC'])} | "
                f"{row.get('confirmed3_any_hit_events')} | {row.get('confirmed3_matched_events')} | "
                f"{row.get('confirmed3_longest_complete_event_miss')} |")
    lines += ['', '## Fixed paired AP differences', '',
        '| Scope | Reference minus comparator | Seed2025 | Seed2026 | Seed2027 | Mean ± sample SD |',
        '|---|---|---:|---:|---:|---:|']
    for comparator in ('batch', evaluation.PERMISSION):
        row = next(r for r in tables['per_entity_paired_three_seed_summary'] if r['record'] == 'PSM'
                   and r['reference'] == comparator and r['metric'] == 'AP')
        lines.append(f"| PSM | reference − {comparator} | " + ' | '.join(number(row[f'seed{s}']) for s in SEEDS)
                     + ' | ' + mean_sd(row) + ' |')
        row = next(r for r in tables['SMD_paired_three_seed_summary'] if r['reference'] == comparator
                   and r['metric'] == 'AP' and r['quantity'] == 'mean_difference')
        lines.append(f"| SMD28 entity macro | reference − {comparator} | "
                     + ' | '.join(number(row[f'seed{s}']) for s in SEEDS) + ' | ' + mean_sd(row) + ' |')
    lines += ['', 'Per-machine, per-seed paired deltas and signs remain in the complete CSVs, including '
        'all unfavorable cases. Reference-minus-batch is the matched training comparison; reference '
        'minus official B is a zero-update reference, not a same-training-budget control.', '',
        '## Scope, calibration and evidence limits', '',
        '- Every seed retains its own selected checkpoint and CAL q99/higher threshold with strict greater-than. '
        'All new and official TimeRCD scores use archived abnormal-minus-normal head logits. No probability '
        'inversion, per-machine score selection, threshold refitting on QUERY or cross-axis subtraction occurs.',
        '- Common-axis new-arm tables and method-own full-axis tables are separate. External baseline '
        'scores/metrics are reused once and do not receive an artificial three-seed variance. Compatible '
        'LEFT covers PSM and only3/28 SMD machines; missing LEFT results remain NA.',
        '- Strict and C3 metrics, any-hit and one-to-one events, false-alarm points/fragments and sustained '
        'coverage remain distinct. First-hit positions are offline offsets, not verified online delays.',
        '- Sample SD uses ddof1 across three training seeds. It describes training randomness on the '
        'same exposed PSM/SMD data and events, not unseen-domain confirmation. No pooled-point significance '
        'or84-independent-domain inference is made.',
        '- Original2025 positive and negative findings are preserved. No model, normalization recipe, '
        'mask, selection rule or inference score is changed in this repetition task.', '',
        '## Cost and provenance', '',
        'parameters_and_compute.csv distinguishes reused2025 accounting from newly executed2026/2027 '
        'training/scoring. Source datasets and raw banks, weights, private host paths and raw receipts '
        'stay outside Git. Safe hashes identify accepted2025 metrics and all public exports. No follow-up '
        'candidate or automatic training is authorized by this publication.', '']
    (public / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    ev._dump(public / 'PROVENANCE_PUBLIC.json', dict(
        seeds=list(SEEDS), accepted_seed2025_result_sha256=OLD_RESULT_SHA,
        raw_aggregation_result_sha256=ev.sha(output / 'RESULT.json'),
        executed_aggregator_source_sha256=result['source_sha256'], publication_source_sha256=ev.sha(__file__),
        fixed_evaluator_source_sha256=result['evaluator_sha256'], exports=exports,
        inference='No new forward pass performed by aggregation; new banks produced by the fixed executor',
        withheld='Raw arrays, checkpoints, private host paths and raw resource/score receipts.'))
    return public


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config2026', type=Path, required=True)
    p.add_argument('--config2027', type=Path, required=True)
    a = p.parse_args(); run(a.workspace, a.output, a.config2026, a.config2027)
