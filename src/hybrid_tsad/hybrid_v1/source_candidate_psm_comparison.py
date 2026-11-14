"""Fixed single structural source candidate: canonical common/full PSM readout.

CPU only. No model import, fitting, auxiliary attribution or score replacement.
Production identities are bound after training/raw completion, before truth I/O.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from .pretrained_psm_comparison import bound, read, sha, write, archive_scalars
from .source_discrimination_psm_comparison import (
    INITIAL, PLAN, BASELINE, OLD_SOURCE_CAL, RANGES, exact_fields, import_bound,
    crop, corrected_operating_columns, reuse_operating, table)

ARCHITECTURE = 'HYBRID_V1_T_ANCHOR_COMPLETION_V1'
BASELINES = [BASELINE, 'HistoryMean512_ZS', 'TimeRadar_official_native', 'LEFT_R_targettrained']


def admission(config):
    """Authenticate completed producer chain without importing model or truth."""
    exact_fields(config, dict(status='FROZEN_READY', seed=2025, common_axis=[512, 87744],
        full_online_axis=[512, 87841], original_length=87841))
    if not config['source_sha256']:
        raise PermissionError('Bind the actual evaluation dependency package')
    for path, expected in config['source_sha256'].items():
        if sha(path) != expected:
            raise PermissionError('Frozen evaluation dependency changed: ' + path)
    if 'fixed_auxiliary_diagnostics' in config:
        raise PermissionError('No attribution or old-unit diagnostic admitted')
    science = read(config['science_freeze'])
    exact_fields(science, dict(candidate='Hybrid_T_anchor_crossview_completion_v1', new_network_parameters=0))
    d = config['candidate']
    for key in ('initial', 'source_plan', 'original_source_config', 'source_CAL_records', 'implementation'):
        bound(d[key])
    exact_fields(d['initial'], dict(sha256=INITIAL))
    exact_fields(d['source_plan'], dict(sha256=PLAN))
    implementation = d['implementation']['sha256']
    shared = dict(architecture=ARCHITECTURE, implementation_sha256=implementation)
    exact_fields(config['expected_shared_receipt_fields'], shared)
    checkpoint = d['checkpoint']['sha256']; bound(d['checkpoint'])
    metadata = read(d['checkpoint_metadata'])
    exact_fields(metadata, dict(shared, checkpoint_sha256=checkpoint,
        same_original_parameter_keys=True, no_extra_trainable_parameters=True,
        fixed_query='standard_sinusoidal_position_0_to_Nminus1_per_branch'))
    train, cal, query = [read(d[k]) for k in ('training_receipt', 'calibration_receipt', 'query_receipt')]
    for receipt, phase, status in ((train, 'train', 'VALID_SOURCE_SELECTED_ENDPOINT'),
            (cal, 'cal', 'VALID_SOURCE_CAL'), (query, 'query', 'COMPLETE_PENDING_FIXED_EVALUATION')):
        exact_fields(receipt, dict(status='COMPLETED', phase=phase, seed=2025,
            scientific_status=status, labels_read=False, checkpoint_sha256=checkpoint))
        exact_fields(receipt, config['expected_shared_receipt_fields'])
    exact_fields(train, dict(attempts=8000, updates=8000, initial_sha256=INITIAL, initial_tensors_exact=True,
        config_sha256=d['training_config']['sha256']))
    producer = read(d['training_config'])
    exact_fields(producer, dict(operation='HYBRID_T_ANCHOR_COMPLETION', phase='train', seed=2025))
    exact_fields(producer['initial'], dict(sha256=INITIAL))
    if producer['original_config'] != d['original_source_config']:
        raise PermissionError('The original normal source configuration must be unchanged')
    original = read(d['original_source_config'])
    bound(dict(path=original['plans'], sha256=PLAN))
    # Producer-specific metadata is required verbatim, without a loss-specific gate.
    exact_fields(producer, config['expected_training_config_fields'])
    for path, expected in producer['immutable_sha256'].items():
        if sha(path) != expected:
            raise PermissionError('Producer immutable source changed: ' + path)
    report = read(d['training_report'])
    exact_fields(report, dict(attempts=8000, updates=8000, selected_attempt=train['selected_attempt']))
    if len(report['steps']) != 8000 or not 0 <= train['selected_attempt'] <= 8000:
        raise ValueError('Complete source8000/selected checkpoint trace required')
    if [v['id'] for v in config['baselines']] != BASELINES:
        raise PermissionError('All four fixed historical comparators are mandatory')
    if [v['online'] for v in config['baselines']] != [True, True, False, False]:
        raise PermissionError('Historical online/offline permissions changed')
    old = config['original_source_calibration']
    exact_fields(old, dict(sha256=OLD_SOURCE_CAL))
    original_cal = read(old)
    exact_fields(original_cal, dict(valid_windows=1024, source_domain_equal_weight=True,
        domains=['SMD', 'Sunspot'], weights=dict(r_tf=.25, r_ms=.25, r_pred=.5),
        threshold=config['baselines'][0]['threshold']))
    if not config['resource_receipts'] or not config['required_evidence']:
        raise PermissionError('Actual phase resources and CPU/CUDA/loader evidence required')
    for asset in config['resource_receipts']:
        bound(asset)
    for evidence in config['required_evidence']:
        exact_fields(read(evidence['asset']), evidence['expected_fields'])
    manifest, calibration = read(d['bank_manifest']), read(d['source_CAL1024'])
    exact_fields(cal, dict(records=1024, calibration_sha256=d['source_CAL1024']['sha256']))
    exact_fields(query, dict(points=87329, calibration_sha256=d['source_CAL1024']['sha256'],
        bank_manifest_sha256=d['bank_manifest']['sha256']))
    exact_fields(calibration, dict(valid_windows=1024, domains=['SMD', 'Sunspot'],
        source_domain_equal_weight=True, weights=dict(r_tf=.25, r_ms=.25, r_pred=.5),
        active_components=['r_tf', 'r_ms', 'r_pred']))
    exact_fields(manifest, dict(complete=True, committed_records=87329))
    exact_fields(manifest['contract'], dict(ranges=RANGES, threshold=calibration['threshold']))
    keys = {'input_sha256', 'config_sha256', 'checkpoint_sha256', 'calibration_sha256', 'source_sha256'}
    if set(d['expected_five_identity']) != keys:
        raise PermissionError('All five actual producer identities required')
    exact_fields(manifest['contract']['identity'], d['expected_five_identity'])
    exact_fields(manifest['contract']['identity'], dict(checkpoint_sha256=checkpoint,
        calibration_sha256=d['source_CAL1024']['sha256']))
    seal = read(config['whole_raw_seal'])
    exact_fields(seal, dict(shared, labels_read=False, bank_manifest_sha256=d['bank_manifest']['sha256'],
        checkpoint_sha256=checkpoint, calibration_sha256=d['source_CAL1024']['sha256']))
    return dict(status='SINGLE_STRUCTURAL_CANDIDATE_ADMITTED', **shared,
        checkpoint_sha256=checkpoint, manifest_sha256=d['bank_manifest']['sha256'])


def _evaluate(data, axis, labels, descriptor, folder, config, formatter, *, reuse=False):
    """Formatting/selection glue; all operating metrics use accepted kernels."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    from . import evaluation
    from .public_native_evaluation import event_diagnostics
    from .controlled_wave_evaluation import reject_dependency_na
    where = crop(data, axis)
    score = data['score'][where]
    available = data.get('available_index', data.get('available'))[where]
    y = labels[axis]
    row = dict(method=descriptor['id'], points=len(axis), positive_points=int(y.sum()),
        AP=float(average_precision_score(y, score)), ROC_AUC=float(roc_auc_score(y, score)),
        online=descriptor['online'], threshold=descriptor.get('threshold'),
        protocol=descriptor['protocol'], role_permissions=descriptor['role_permissions'])
    folder.mkdir()
    args = dict(index=axis, score=score, valid=np.ones(len(axis), bool), available_index=available,
        invalid_reasons=[None]*len(axis), full_labels=labels)
    phases = []
    if descriptor['online']:
        if (available > axis).any():
            raise PermissionError('Offline output cannot be made causal by cropping')
        if reuse:
            result = reuse_operating(descriptor, axis, score, available, labels)
            write(folder/'REUSED_ACCEPTED_OPERATING_RESULT.json', descriptor['accepted_common_evaluation'])
        else:
            result = evaluation.evaluate_arrays(**args, threshold=descriptor['threshold'], bank_complete=True,
                original_length=87841, warmup_points=512, affiliation_identity=bound(config['affiliation_identity']))
            reject_dependency_na(result)
            evaluation.save_evaluation(folder/'fixed_source_tau_online', result, **args,
                provenance=dict(config=config, method=descriptor['id']))
        phases = event_diagnostics(result)
        row.update(corrected_operating_columns(formatter, result, phases))
    else:
        row['online_NA_reason'] = 'native retrospective/target-trained external reference; permissions remain offline'
    with (folder/'canonical_inputs.npz').open('xb') as handle:
        np.savez_compressed(handle, **{k:v for k,v in args.items() if k != 'invalid_reasons'})
    return row, [dict(method=descriptor['id'], **v) for v in phases]


def run(config):
    began = time.monotonic(); gate = admission(config)
    from .controlled_wave_evaluation import require_metric_dependencies
    require_metric_dependencies()
    loader = import_bound(config['baseline_loader'], 'accepted_psm_comparison')
    formatter = import_bound(config['operating_formatter'], 'accepted_operating_formatter').operating_columns
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=False)
    d = config['candidate']
    archive = archive_scalars(dict(manifest=d['bank_manifest'], calibration=d['source_CAL1024']),
        output/'CANDIDATE_CANONICAL_SCALARS.npz')
    write(output/'RAW_SEALED_BEFORE_LABELS.json', dict(gate=gate, archive=archive, labels_read=False))
    # First numerical truth access follows all production/raw/replay closure.
    with np.load(bound(config['truth']), allow_pickle=False) as raw:
        labels = raw['full_labels'].copy()
    if labels.shape != (87841,):
        raise ValueError('Keep full original event axis')
    with np.load(archive['path'], allow_pickle=False) as raw:
        new_data = {k:raw[k].copy() for k in ('index','score','valid','available_index')}
    candidate = dict(id=d['id'], online=True, threshold=archive['threshold'],
        protocol='source8000_T_anchor_completion_original_h1_R3_sourceCAL1024',
        role_permissions='sourceFIT_only_no_targetFIT_or_targetCAL')
    descriptors = [config['baselines'][0], candidate] + config['baselines'][1:]
    if len({v['id'] for v in descriptors}) != 5:
        raise ValueError('Exactly five distinct fixed methods')
    data = {v['id']:new_data if v is candidate else loader.load_method(v, labels) for v in descriptors}
    allrows = {}
    for label, axis, selected in (('common', np.arange(512,87744), descriptors),
            ('full_online', np.arange(512,87841), [v for v in descriptors if v['online']])):
        root = output/label; root.mkdir(); rows = []; phases = []
        for descriptor in selected:
            row, p = _evaluate(data[descriptor['id']], axis, labels, descriptor, root/descriptor['id'],
                config, formatter, reuse=(label=='common' and descriptor is not candidate))
            rows.append(row); phases.extend(p)
        table(root/'all_methods.csv', rows); table(root/'event_phases.csv', phases)
        old, new = rows[:2]
        delta = {k:float(new[k]-v) for k,v in old.items() if k not in ('points','positive_points','threshold','online')
            and type(v) in (float,int) and type(new.get(k)) in (float,int)}
        table(root/'candidate_minus_source.csv', [delta]); allrows[label] = dict(rows=rows, candidate_minus_source=delta)
    table(output/'all_methods_common_axis.csv', allrows['common']['rows'])
    table(output/'all_online_full_axis.csv', allrows['full_online']['rows'])
    write(output/'RESULT.json', dict(status='COMPLETED_SINGLE_SOURCE_CANDIDATE_PSM',
        seconds=time.monotonic()-began, results=allrows, source_config=config, labels_after_raw_seal=True,
        GPUh=0, new_forward=False, new_training=False, scope='PSM development only', limits=config['limits']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--trusted-config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.trusted_config_sha256:
        raise PermissionError('Final configuration hash mismatch')
    run(json.loads(Path(args.config).read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
