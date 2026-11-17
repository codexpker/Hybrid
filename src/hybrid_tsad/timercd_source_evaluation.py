"""Source-only TimeRCD continuation: fixed three-arm, two-permission evaluation.

Existing official-logit and external comparison rows are hash-reused. Source
training arms and normalization permissions remain separate factors. Evaluation
uses the established metrics/event kernel, original role binding and common axis.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from . import timeradar_cross_variable_evaluation as ev
from .timeradar_smd_full_evaluation import read, local, checked, METRICS
from . import timercd_fixed_evaluation as fixed

MODELS = ('native', 'time', 'frft')
PERMISSIONS = fixed.ARMS
BASELINE_RESULT = 'results/timercd_logit_evaluation_20261117_v1/RESULT.json'
BASELINE_SHA = 'f2121a878eab1ddb753b7db24b3f4fcedbf6742f68bceebe7e9c62ff30291b76'


def method(model, permission):
    return model + '__' + permission


def terminal_gate(seal, records):
    expected = {f'score/{m}/{r}/{p}/{role}.npz' for m in MODELS for r in records
                for p in PERMISSIONS for role in ('CAL', 'QUERY')}
    actual = [b['relative'] for b in seal['banks']]
    if (seal['status'] != 'COMPLETED' or not seal['no_labels_decoded']
            or len(actual) != len(set(actual)) or set(actual) != expected):
        raise ValueError('All fixed 348 CAL/QUERY banks must seal before evaluation labels open')
    return {b['relative']: b for b in seal['banks']}


def paired(rows, entities):
    lookup = {(r['record'], r['arm']): r for r in rows}
    details, summaries = [], []
    for permission in PERMISSIONS:
        candidate = method('frft', permission)
        for reference in (method('native', permission), method('time', permission), permission):
            for metric in METRICS:
                deltas = []
                for rid in entities:
                    a = lookup[rid, candidate].get(metric)
                    b = lookup[rid, reference].get(metric)
                    value = (float(a - b) if a is not None and b is not None
                             and np.isfinite(a) and np.isfinite(b) else None)
                    details.append(dict(record=rid, candidate=candidate, reference=reference,
                                        metric=metric, difference=value))
                    if value is not None:
                        deltas.append(value)
                summaries.append(dict(candidate=candidate, reference=reference, metric=metric,
                    expected_entities=len(entities), paired_n=len(deltas),
                    mean_difference=float(np.mean(deltas)) if deltas else None,
                    median_difference=float(np.median(deltas)) if deltas else None,
                    positive=sum(v > 0 for v in deltas), tie=sum(v == 0 for v in deltas),
                    negative=sum(v < 0 for v in deltas),
                    note='Strict numerical signs; positive false-alarm delta is a cost. One exposed SMD domain.'))
    return details, summaries


def run(workspace, output, config):
    start = time.monotonic()
    w, output = Path(workspace).resolve(), Path(output).resolve()
    cp = local(w, config); cfg = read(cp)
    root = local(w, cfg['output']); sp = root / 'RAW_BANK_SEAL.json'; seal = read(sp)
    bp = local(w, cfg['data_binding']); checked(bp, fixed.DATA_SHA)
    if cfg['data_binding_sha256'] != fixed.DATA_SHA or seal['config_sha256'] != ev.sha(cp):
        raise PermissionError('Frozen configuration / original-unit data identity changed')
    records = {r['id']: r for r in read(bp)['records']}
    if len(records) != 29 or set(cfg['arms']) != set(MODELS):
        raise ValueError('Fixed three source models, PSM and SMD28 required')
    descriptors = terminal_gate(seal, records)
    for rel, desc in descriptors.items():
        checked(root / rel, desc['sha256'])
    # Bind every readout to the selected source-only checkpoint, not just its
    # schema. The selection receipt itself is retained as calculation provenance.
    training = {}
    for model in MODELS:
        selected = root / 'train' / model / 'selected.pth'
        receipt_path = root / 'train' / model / 'TRAIN_RECEIPT.json'
        receipt = read(receipt_path)
        if receipt['source_plan_sha256'] != cfg['source_plan_sha256']:
            raise PermissionError('Training source exposure identity differs between arms')
        training[model] = dict(checkpoint_sha256=ev.sha(selected), receipt=receipt,
                               receipt_path=receipt_path)
        for rid in records:
            for permission in PERMISSIONS:
                s = read(root / 'score' / model / rid / permission / 'SCORE_RECEIPT.json')
                if (s['arm'] != model or s['permission'] != permission
                        or s['checkpoint_sha256'] != training[model]['checkpoint_sha256']
                        or s['selected_step'] != receipt['selected_step']):
                    raise PermissionError('Readout checkpoint/model/permission identity mismatch')
    # Prior comparisons and all new scores exist before any evaluation-label load.
    baseline_path = checked(w / BASELINE_RESULT, BASELINE_SHA)
    accepted = read(baseline_path)
    native = [dict(r) for r in accepted['native_rows']]
    common = [dict(r) for r in accepted['common_rows']]
    for row in native + common:
        row['metric_origin'] = 'ACCEPTED_IMMUTABLE_COMPARISON_CSV_OR_RESULT'
    ep = checked(w / 'data/timeradar_cross_variable_20261116_v1/evaluation/evaluation_manifest.json', fixed.LABEL_OLD_SHA)
    fp = checked(w / 'preparations/timeradar_smd_full_20261116_v1/evaluation_manifest_full28.json', fixed.LABEL_FULL_SHA)
    labelmeta = {r['id']: r for r in read(fp)['records']}
    labelmeta['PSM'] = next(r for r in read(ep)['records'] if r['id'] == 'PSM')
    inputs = {str(p): ev.sha(p) for p in (cp, sp, bp, baseline_path, ep, fp)}
    for value in training.values():
        inputs[str(value['receipt_path'])] = ev.sha(value['receipt_path'])
    output.mkdir(parents=True, exist_ok=False)
    event_refs = []
    for rid, record in records.items():
        lp = local(w, labelmeta[rid]['path']); checked(lp, labelmeta[rid]['sha256'])
        inputs[str(lp)] = ev.sha(lp)
        with np.load(lp, allow_pickle=False) as z:
            labels, labelindex = z['label'].copy(), z['index'].copy()
        if not np.array_equal(labelindex, np.arange(len(labels))):
            raise ValueError('Preserve original complete event boundaries')
        for model in MODELS:
            for permission in PERMISSIONS:
                keys = {r: f'score/{model}/{rid}/{permission}/{r}.npz' for r in ('CAL', 'QUERY')}
                banks = {r: fixed.load_new_bank(root / key, record, r, permission,
                         descriptors[key], score_expression='logit') for r, key in keys.items()}
                threshold = ev.cal_threshold(banks['CAL'][1])
                index, score, available = banks['QUERY']
                name = method(model, permission)
                for scope, axis, rows in (('native_full', index, native),
                                          ('shared_common', fixed.fixed_common_indices(record), common)):
                    pos = np.searchsorted(index, axis)
                    if np.any(pos >= len(index)) or not np.array_equal(index[pos], axis):
                        raise ValueError('Common axis must contain genuinely scored real points')
                    provenance = {str(root / k): descriptors[k]['sha256'] for k in keys.values()}
                    provenance.update({str(p): ev.sha(p) for p in (cp, sp, bp, lp)})
                    result = ev.evaluate_arrays(axis, score[pos], labels, threshold,
                                                available_index=available[pos])
                    folder = output / scope / rid / name
                    ev._save_method(folder, result, axis, score[pos], labels, available[pos], provenance)
                    row = ev._flat(rid, name, result)
                    row.update(status='OK', metric_origin='SOURCE_CONTINUATION_FROZEN_SCORE_BANK',
                        source_model=model, statistics_track=permission, score_expression='logit',
                        axis_start=int(axis[0]), axis_end_exclusive=int(axis[-1]) + 1,
                        training_permission='Official synthetic source supervised CE + masked reconstruction; no target updates',
                        statistics_permission=('whole separate role statistics; offline future-role statistics'
                            if permission == PERMISSIONS[0] else 'Original raw FIT-only mean/std exactly once; offline context window'),
                        score_definition='mean-channel abnormal minus normal head logit; original CAL q99 higher, strict >')
                    rows.append(row)
                    for stream in ('strict', 'confirmed3'):
                        path = folder / (stream + '_event_details.json')
                        event_refs.append(dict(record=rid, arm=name, scope=scope, stream=stream,
                            path=str(path.relative_to(w)), sha256=ev.sha(path)))
    native_methods = fixed.NATIVE_ARMS + tuple(method(m, p) for m in MODELS for p in PERMISSIONS)
    common_methods = fixed.COMMON_ARMS + tuple(method(m, p) for m in MODELS for p in PERMISSIONS)
    smd = [r for r in records if r != 'PSM']
    for rows, methods in ((native, native_methods), (common, common_methods)):
        rows.sort(key=lambda r: (r['record'] != 'PSM', r['record'], methods.index(r['arm'])))
    native_stats = fixed.describe([r for r in native if r['record'] != 'PSM'], smd, native_methods)
    common_stats = fixed.describe([r for r in common if r['record'] != 'PSM'], smd, common_methods)
    smd_pairs, smd_summary = paired(common, smd)
    all_pairs, _ = paired(common, list(records))
    ev._write_csv(output / 'common_29x14.csv', common)
    ev._write_csv(output / 'native_29x12.csv', native)
    ev._write_csv(output / 'common_SMD_macro_median.csv', common_stats['macro'])
    ev._write_csv(output / 'native_SMD_macro_median.csv', native_stats['macro'])
    ev._write_csv(output / 'common_all29_paired_deltas.csv', all_pairs)
    ev._write_csv(output / 'common_SMD_paired_summary.csv', smd_summary)
    ev._dump(output / 'event_detail_references.json', dict(new_references=event_refs))
    receipts = []
    for model in MODELS:
        for filename in ('TRAIN_RECEIPT.json', 'SELECT.json', 'OS.json'):
            p = root / 'train' / model / filename
            if p.exists():
                receipts.append(dict(model=model, filename=filename, sha256=ev.sha(p), content=read(p)))
    ev._dump(output / 'RESULT.json', dict(status='COMPLETED', native_rows=native, common_rows=common,
        common_SMD=common_stats, native_SMD=native_stats, source_training_receipts=receipts,
        common_SMD_paired_summary=smd_summary, new_evaluations=348,
        CPU_seconds=time.monotonic() - start, GPU_hours=0, inputs=inputs,
        source_sha256=ev.sha(__file__), evaluator_sha256=ev.sha(ev.__file__),
        new_forward=False, score_expression='original archived abnormal-minus-normal logits for all new arms',
        limits='Development offline comparison. All 29 objects retained. A and B permissions differ. '
               'SMD is one exposed domain; LEFT only 3/28, missing entities remain NA. No source/target contamination claim.'))
    return native, common


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    run(a.workspace, a.output, a.config)
