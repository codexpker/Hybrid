"""Fixed source reference-normalization experiment, original B-only detection.

All model/metric functions remain the accepted TimeRCD functions. This consumer
opens labels only after the complete two-arm CAL/QUERY bank set seals. Historical
source continuation is never relabeled as a matched training control.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from . import timeradar_cross_variable_evaluation as ev
from . import timercd_fixed_evaluation as fixed
from .timeradar_smd_full_evaluation import read, local, checked, METRICS
from .timercd_source_evaluation import BASELINE_RESULT, BASELINE_SHA

MODELS = ('batch', 'reference')
PERMISSION = 'B_fit_only'
COMMON_METHODS = tuple(m for m in fixed.COMMON_ARMS if m != fixed.ARMS[0]) + MODELS
NATIVE_METHODS = tuple(m for m in fixed.NATIVE_ARMS if m != fixed.ARMS[0]) + MODELS


def terminal_gate(seal, records):
    expected = {f'score/{m}/{r}/{PERMISSION}/{role}.npz' for m in MODELS for r in records
                for role in ('CAL', 'QUERY')}
    actual = [b['relative'] for b in seal['banks']]
    if (seal['status'] != 'COMPLETED' or not seal['no_labels_decoded']
            or len(actual) != len(set(actual)) or set(actual) != expected):
        raise ValueError('All fixed 116 B-only banks must seal before evaluation-label access')
    return {b['relative']: b for b in seal['banks']}


def paired(rows, entities):
    lookup = {(r['record'], r['arm']): r for r in rows}
    details, summaries = [], []
    for reference in ('batch', PERMISSION):
        for metric in METRICS:
            values = []
            for rid in entities:
                a = lookup[rid, 'reference'].get(metric)
                b = lookup[rid, reference].get(metric)
                delta = (float(a - b) if a is not None and b is not None
                         and np.isfinite(a) and np.isfinite(b) else None)
                details.append(dict(record=rid, candidate='reference', reference=reference,
                                    metric=metric, difference=delta))
                if delta is not None:
                    values.append(delta)
            summaries.append(dict(candidate='reference', reference=reference, metric=metric,
                expected_entities=len(entities), paired_n=len(values),
                mean_difference=float(np.mean(values)) if values else None,
                median_difference=float(np.median(values)) if values else None,
                positive=sum(v > 0 for v in values), tie=sum(v == 0 for v in values),
                negative=sum(v < 0 for v in values),
                comparison='MATCHED_SOURCE_NORMALIZATION' if reference == 'batch' else 'OFFICIAL_ZERO_UPDATE_REFERENCE',
                note='Strict numerical signs; positive false-alarm delta is a cost. SMD is one exposed domain.'))
    return details, summaries


def run(workspace, output, config):
    start = time.monotonic()
    w, output = Path(workspace).resolve(), Path(output).resolve()
    cp = local(w, config); cfg = read(cp)
    root = local(w, cfg['output']); sp = root / 'RAW_BANK_SEAL.json'; seal = read(sp)
    bp = local(w, cfg['data_binding']); checked(bp, fixed.DATA_SHA)
    if cfg['data_binding_sha256'] != fixed.DATA_SHA or seal['config_sha256'] != ev.sha(cp):
        raise PermissionError('Frozen configuration / original raw target role identity changed')
    records = {r['id']: r for r in read(bp)['records']}
    if len(records) != 29 or tuple(cfg['arms']) != MODELS:
        raise ValueError('The two prespecified source normalization arms and all29 objects are required')
    descriptors = terminal_gate(seal, records)
    for rel, desc in descriptors.items():
        checked(root / rel, desc['sha256'])
    training = {}
    for model in MODELS:
        selected = root / 'train' / model / 'selected.pth'
        rp = root / 'train' / model / 'TRAIN_RECEIPT.json'; receipt = read(rp)
        if receipt['source_plan_sha256'] != cfg['source_plan_sha256']:
            raise PermissionError('Source exposure plan identity mismatch')
        training[model] = dict(checkpoint_sha256=ev.sha(selected), receipt=receipt, receipt_path=rp)
        for rid in records:
            p = root / 'score' / model / rid / PERMISSION / 'SCORE_RECEIPT.json'
            s = read(p)
            if (s['arm'] != model or s['permission'] != PERMISSION
                    or s['checkpoint_sha256'] != training[model]['checkpoint_sha256']
                    or s['selected_step'] != receipt['selected_step']):
                raise PermissionError('Model / B statistics / selected checkpoint identity mismatch')
    old_path = checked(w / BASELINE_RESULT, BASELINE_SHA); old = read(old_path)
    native = [dict(r, metric_origin='ACCEPTED_IMMUTABLE_COMPARISON') for r in old['native_rows']
              if r['arm'] != fixed.ARMS[0]]
    common = [dict(r, metric_origin='ACCEPTED_IMMUTABLE_COMPARISON') for r in old['common_rows']
              if r['arm'] != fixed.ARMS[0]]
    ep = checked(w / 'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json', fixed.LABEL_OLD_SHA)
    fp = checked(w / 'preparations/timeradar_smd_full_20261116_v1/evaluation_manifest_full28.json', fixed.LABEL_FULL_SHA)
    labelmeta = {r['id']: r for r in read(fp)['records']}
    labelmeta['PSM'] = next(r for r in read(ep)['records'] if r['id'] == 'PSM')
    inputs = {str(p): ev.sha(p) for p in (cp, sp, bp, old_path, ep, fp)}
    for value in training.values():
        inputs[str(value['receipt_path'])] = ev.sha(value['receipt_path'])
    output.mkdir(parents=True, exist_ok=False)
    # Preserve the actual executed bytes before any publication-only edits.
    with (output / 'EVALUATOR_EXECUTED_SNAPSHOT.py').open('xb') as handle:
        handle.write(Path(__file__).read_bytes())
    event_refs = []
    for rid, record in records.items():
        lp = local(w, labelmeta[rid]['path']); checked(lp, labelmeta[rid]['sha256'])
        inputs[str(lp)] = ev.sha(lp)
        with np.load(lp, allow_pickle=False) as z:
            labels, ix = z['label'].copy(), z['index'].copy()
        if not np.array_equal(ix, np.arange(len(labels))):
            raise ValueError('Original complete event axis required')
        for model in MODELS:
            keys = {role: f'score/{model}/{rid}/{PERMISSION}/{role}.npz' for role in ('CAL', 'QUERY')}
            banks = {role: fixed.load_new_bank(root / key, record, role, PERMISSION,
                     descriptors[key], score_expression='logit') for role, key in keys.items()}
            threshold = ev.cal_threshold(banks['CAL'][1])
            index, score, available = banks['QUERY']
            for scope, axis, rows in (('native_full', index, native),
                                      ('shared_common', fixed.fixed_common_indices(record), common)):
                pos = np.searchsorted(index, axis)
                if np.any(pos >= len(index)) or not np.array_equal(index[pos], axis):
                    raise ValueError('Use only the actual fixed common scoring axis')
                provenance = {str(root / key): descriptors[key]['sha256'] for key in keys.values()}
                provenance.update({str(p): ev.sha(p) for p in (cp, sp, bp, lp)})
                result = ev.evaluate_arrays(axis, score[pos], labels, threshold, available_index=available[pos])
                folder = output / scope / rid / model
                ev._save_method(folder, result, axis, score[pos], labels, available[pos], provenance)
                row = ev._flat(rid, model, result)
                row.update(status='OK', metric_origin='REFERENCE_SOURCE_FROZEN_BANK',
                    source_training_normalization=model, target_statistics_track=PERMISSION,
                    axis_start=int(axis[0]), axis_end_exclusive=int(axis[-1]) + 1,
                    score_expression='logit', score_definition='Original head abnormal minus normal; own CAL q99 higher strict >',
                    training_permission='Synthetic source CE + masked reconstruction; no target optimization',
                    statistics_permission='Original raw target FIT-only mean/std exactly once; offline full context')
                rows.append(row)
                for stream in ('strict', 'confirmed3'):
                    p = folder / (stream + '_event_details.json')
                    event_refs.append(dict(record=rid, arm=model, scope=scope, stream=stream,
                                           path=str(p.relative_to(w)), sha256=ev.sha(p)))
    for rows, methods in ((native, NATIVE_METHODS), (common, COMMON_METHODS)):
        rows.sort(key=lambda r: (r['record'] != 'PSM', r['record'], methods.index(r['arm'])))
    smd = [r for r in records if r != 'PSM']
    native_stats = fixed.describe([r for r in native if r['record'] != 'PSM'], smd, NATIVE_METHODS)
    common_stats = fixed.describe([r for r in common if r['record'] != 'PSM'], smd, COMMON_METHODS)
    all_pairs, _ = paired(common, list(records)); _, smd_pairs = paired(common, smd)
    controllers = {}
    for phase in ('profile', 'train'):
        p = root / (phase + '_controller') / 'COMPLETED.json'
        if p.exists():
            controllers[phase] = read(p); inputs[str(p)] = ev.sha(p)
    controllers['score'] = seal
    costs = []
    for model in MODELS:
        r = training[model]['receipt']
        cost = dict(model=model, parameters=r['parameters'], trainable_parameters=r['trainable'],
                    selected_step=r['selected_step'], updates=r['updates'],
                    source_instances=r['source_instances'], sample_exposures=r['sample_exposures'],
                    timepoints_exposed=r['timepoints_exposed'], checkpoint_sha256=training[model]['checkpoint_sha256'])
        for phase in ('profile', 'train', 'score'):
            jobs = [j for j in controllers.get(phase, {}).get('jobs', []) if j['arm'] == model]
            cost[phase + '_jobs'] = len(jobs)
            cost[phase + '_GPU_hours'] = sum(j['GPU_hours'] for j in jobs) if jobs else None
        costs.append(cost)
    exports = {'common_29x9.csv': common, 'native_29x7.csv': native,
               'common_SMD_macro_median.csv': common_stats['macro'],
               'native_SMD_macro_median.csv': native_stats['macro'],
               'common_all29_paired_deltas.csv': all_pairs,
               'common_SMD_paired_summary.csv': smd_pairs, 'parameters_and_compute.csv': costs}
    for name, rows in exports.items():
        ev._write_csv(output / name, rows)
    ev._dump(output / 'event_detail_references.json', dict(new_references=event_refs))
    ev._dump(output / 'RESULT.json', dict(status='COMPLETED', common_rows=common, native_rows=native,
        common_SMD=common_stats, native_SMD=native_stats, common_SMD_paired_summary=smd_pairs,
        parameters_and_compute=costs, inputs=inputs, new_evaluations=116,
        CPU_seconds=time.monotonic() - start, GPU_hours=0, new_forward=False,
        source_sha256=ev.sha(__file__), evaluator_sha256=ev.sha(ev.__file__),
        limits='One source seed, offline development PSM and exposed SMD. B-only target FIT statistics. '
               'Prior source continuation is historical unmatched evidence, not a current matching arm. '
               'LEFT 3/28 compatible entities; missing 25 retained NA. No robustness/lag claim.'))
    return native, common


def publish(output, public):
    """Publish tables and safe identities, never raw receipts or machine paths."""
    output, public = Path(output), Path(public)
    result = read(output / 'RESULT.json')
    if result['status'] != 'COMPLETED':
        raise ValueError('Complete evaluation required')
    names = ('common_29x9.csv', 'native_29x7.csv', 'common_SMD_macro_median.csv',
             'native_SMD_macro_median.csv', 'common_all29_paired_deltas.csv',
             'common_SMD_paired_summary.csv', 'parameters_and_compute.csv')
    public.mkdir(parents=True, exist_ok=True)
    if any((public / n).exists() for n in names + ('RESULTS.md', 'PROVENANCE_PUBLIC.json')):
        raise FileExistsError('Do not overwrite published results')
    exports = []
    for name in names:
        p = output / name; text = p.read_text(encoding='utf-8')
        if any(s in text for s in ('/home/', '192.168.', 'D:\\', 'C:\\', 'ghp_', 'token=')):
            raise PermissionError('Potentially private output content: ' + name)
        target = public / name; target.write_text(text, encoding='utf-8', newline='\n')
        exports.append(dict(name=name, raw_sha256=ev.sha(p), public_sha256=ev.sha(target)))
    lookup = {(r['record'], r['arm']): r for r in result['common_rows']}
    summary = {(r['method'], r['metric']): r for r in result['common_SMD']['macro']}
    def f(v, percent=False):
        return 'NA' if v is None else f'{v * (100 if percent else 1):.6f}'
    lines = ['# TimeRCD source reference-consistency training: fixed B-only comparison', '',
        'Two matched source training arms use the same official architecture/initial weights, '
        'generated X/Y, source-instance roles, exposure and update budget. The source normalization '
        'is the intended factor: batch input statistics versus lawful historical reference statistics. '
        'Both are selected on the same reference-normalized source SELECT evaluation. No target QUERY '
        'or target labels enter training, source generation, model selection or threshold fitting. '
        'This is synthetic supervised anomaly learning with auxiliary masked reconstruction, not '
        'pure unsupervised learning and not a proven mixed-target robustness contribution.', '',
        '## Common-axis SMD28', '',
        '| Method | Valid n/28 | AP macro | AP median | ROC macro | Strict normal FPR (%) | '
        'C3 normal FPR (%) | C3 event recall (%) | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    metrics = ('strict_normal_FPR', 'confirmed3_normal_FPR', 'confirmed3_one_to_one_eventRecall',
               'confirmed3_one_to_one_eventF1', 'confirmed3_complete_event_point_coverage',
               'confirmed3_long_steady_macro_coverage')
    for m in COMMON_METHODS:
        s = lambda k: summary[m, k]['macro_mean']
        values = [f(s('AP')), f(summary[m, 'AP']['median']), f(s('ROC_AUC'))]
        values += [f(s(k), True) for k in metrics]
        lines.append(f"| {m} | {summary[m, 'AP']['n_valid']} | " + ' | '.join(values) + ' |')
    lines += ['', 'LEFT covers only 3/28 compatible SMD entities; its subset macro is not directly '
        'comparable with all-28 results. Long-event statistics retain their own eligible denominator '
        'in the complete macro table. SMD28 is one historically exposed data domain.', '',
        '## PSM common axis', '',
        '| Method | AP | ROC | Strict normal FPR (%) | C3 normal FPR (%) | C3 any-hit events | '
        'C3 one-to-one matched events | C3 event F1 (%) | C3 anomaly coverage (%) | C3 long steady (%) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in COMMON_METHODS:
        r = lookup['PSM', m]
        values = [f(r.get('AP')), f(r.get('ROC_AUC')), f(r.get('strict_normal_FPR'), True),
            f(r.get('confirmed3_normal_FPR'), True), str(r.get('confirmed3_any_hit_events', 'NA')),
            str(r.get('confirmed3_matched_events', 'NA')), f(r.get('confirmed3_one_to_one_eventF1'), True),
            f(r.get('confirmed3_complete_event_point_coverage'), True),
            f(r.get('confirmed3_long_steady_macro_coverage'), True)]
        lines.append('| ' + m + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Fixed paired differences, SMD28', '',
        '| Candidate minus reference | AP mean difference | AP median difference | Positive / tie / negative | Scope |',
        '|---|---:|---:|---|---|']
    for row in result['common_SMD_paired_summary']:
        if row['metric'] == 'AP':
            lines.append(f"| reference − {row['reference']} | {f(row['mean_difference'])} | "
                f"{f(row['median_difference'])} | {row['positive']} / {row['tie']} / {row['negative']} | {row['comparison']} |")
    lines += ['', '## Scoring, permissions and preserved limits', '',
        '- The sole target track is original B_fit_only: each raw CAL/QUERY role uses its original '
        'target FIT mean/std exactly once. CAL and QUERY are separate contexts. Official context '
        'window5000, batch1, anomaly head and abnormal-minus-normal logit expression remain fixed.',
        '- Every method uses its own lawful original CAL q99/higher threshold and strict greater-than; '
        'neither old numeric thresholds nor QUERY labels are substituted. A and B statistics permissions '
        'are not mixed. All reported new methods are offline retrospective, not online detectors.',
        '- common_29x9.csv is the matched scoring-axis table. native_29x7.csv preserves method-own full '
        'QUERY axes separately; no cross-axis subtraction. TimeRadar and compatible LEFT/history rows '
        'are reused and retain their differing model/training/context permissions.',
        '- Prior source continuation is historical unmatched evidence. It is not substituted for the '
        'new batch arm or used to choose source scale, checkpoints or hyperparameters.',
        '- Any-hit is distinct from one-to-one event matching. Full strict/C3 points, normal alarm '
        'fragments, event metrics, long-steady coverage and longest missed spans remain in the CSVs. '
        'First-hit offsets are offline positions, not causal delay claims.',
        '- One training seed and exposed development datasets limit repeatability and independent '
        'confirmation claims. This comparison does not prove lag identification, fractional-feature '
        'benefit or mixed-unlabeled target adaptation robustness.', '',
        '## Resource accounting', '',
        'parameters_and_compute.csv retains selected steps, updates, source sample exposure and actual '
        'controller profile/training/scoring GPU wall time per arm. Raw arrays, checkpoint files, '
        'private server paths and receipts are withheld from the public repository. Their numerical '
        'identities and safe export hashes are retained in PROVENANCE_PUBLIC.json.', '']
    (public / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    ev._dump(public / 'PROVENANCE_PUBLIC.json', dict(
        official_code_commit='b031773e323bc0e6c9dcd8c9119b800ec4d70f8c',
        official_checkpoint_sha256='a497ac910f24f8fb347d690a21b60d74bc04785ebc777880553c158dea4434d0',
        source_generator_commit='73eb733f3026e32bf9edfde1d4c3b787358908e7',
        accepted_baseline_result_sha256=BASELINE_SHA, raw_result_sha256=ev.sha(output / 'RESULT.json'),
        executed_evaluator_source_sha256=result['source_sha256'], publication_source_sha256=ev.sha(__file__),
        metrics_source_sha256=ev.sha(ev.__file__), exports=exports,
        withheld='Raw weights, arrays, raw receipts and private machine paths are not public exports.'))
    return public


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); run(a.workspace, a.output, a.config)
