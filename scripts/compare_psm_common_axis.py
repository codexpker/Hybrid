"""Recompute fixed historical PSM scores on one declared common native axis.

No model forward, score selection, threshold fitting, or offline-to-online
conversion. Canonical accepted continuous inputs retain their original hashes.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tsad.hybrid_v1 import evaluation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_file(descriptor):
    path = Path(descriptor['path'])
    if sha(path) != descriptor['sha256']:
        raise PermissionError('Frozen score/label provenance changed: ' + str(path))
    return path


def dump(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def load_method(descriptor, full_labels):
    path = require_file(descriptor['raw'])
    with np.load(path, allow_pickle=False) as data:
        kind = descriptor['kind']
        if kind == 'accepted_continuous_inputs':
            provenance = json.loads(require_file(descriptor['provenance']).read_text())
            if sha(path) != provenance['calculation_inputs_sha256']:
                raise ValueError('Accepted continuous input not linked to original bank')
            index, score, valid = data['index'].copy(), data['score'].copy(), data['valid'].copy()
            available = data['available_index'].copy()
            if not np.array_equal(data['full_labels'], full_labels):
                raise ValueError('Canonical full labels must match exactly')
        elif kind == 'timeradar_native_summary':
            index, score = data['index'].copy(), data['scores'].copy()
            valid = np.isfinite(score)
            if not np.array_equal(data['labels'], full_labels[index]):
                raise ValueError('Official TimeRadar labels differ at actual native positions')
            width = descriptor['window']
            available = (index // width + 1) * width - 1
        elif kind == 'left_native_window':
            absolute = data['absolute_time_index'].copy()
            index = absolute - descriptor['test_original_start']
            score = data['canonical_score'].copy()
            expected = data['channel_scores'].mean(axis=-1).reshape(-1)
            if not np.array_equal(score, expected):
                raise ValueError('Preserve original CPU float32 mean, no new channel reduction')
            if data['channel_names'].tolist() != ['feature_' + str(i) for i in range(25)]:
                raise ValueError('Native 25-channel order changed')
            valid = np.isfinite(score)
            available = data['point_input_end'].copy() - descriptor['test_original_start']
            with np.load(require_file(descriptor['old_evaluation_inputs']), allow_pickle=False) as old:
                if not np.array_equal(old['full_labels'], full_labels) or not np.array_equal(old['score'], score):
                    raise ValueError('LEFT full native truth/canonical score identity differs')
        else:
            raise ValueError('Unknown score source')
    if len(index) != len(np.unique(index)) or not np.all(np.diff(index) > 0):
        raise ValueError('Original unique chronological native index required')
    if descriptor['online'] and np.any(available[valid] > index[valid]):
        raise PermissionError('Future-window scores cannot become online by axis cropping')
    return dict(index=index, score=score, valid=valid, available=available)


def run(config):
    started = time.monotonic()
    if sha(__file__) != config['source_sha256']:
        raise PermissionError('Frozen shared AP consumer changed')
    for path, expected in config['metric_source_sha256'].items():
        if sha(path) != expected:
            raise PermissionError('Original metric source changed')
    with np.load(require_file(config['truth']), allow_pickle=False) as truth:
        full_labels = truth['full_labels'].copy()
    if len(full_labels) != 87841:
        raise ValueError('Keep full original 87841-point event axis')
    loaded = {descriptor['id']: load_method(descriptor, full_labels) for descriptor in config['methods']}
    common = None
    for data in loaded.values():
        valid_index = data['index'][data['valid'] & np.isfinite(data['score'])]
        common = valid_index if common is None else np.intersect1d(common, valid_index)
    expected = np.arange(config['common_start'], config['common_end'])
    if not np.array_equal(common, expected):
        raise ValueError('Declared complete common axis differs; no performance-selected mask')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    y = full_labels[common]
    rows, assets = [], []
    for descriptor in config['methods']:
        name, data = descriptor['id'], loaded[descriptor['id']]
        where = np.searchsorted(data['index'], common)
        score = data['score'][where]
        # Exactly the same sklearn functions and identical y/points for all rows.
        precision, recall, _ = precision_recall_curve(y, score)
        row = dict(method=name, points=len(common), positive_points=int(y.sum()),
            AP=float(average_precision_score(y, score)), ROC_AUC=float(roc_auc_score(y, score)),
            PR_AUC_trapezoid=float(auc(recall, precision)),
            native_first=int(common[0]), native_end_exclusive=int(common[-1]) + 1,
            protocol=descriptor['protocol'], online_eligible=descriptor['online'],
            original_threshold=descriptor.get('threshold'),
            role_permissions=descriptor['role_permissions'])
        folder = output / name
        folder.mkdir()
        destination = folder / 'common_axis_inputs.npz'
        np.savez_compressed(destination, index=common, score=score, labels=y,
                            available_index=data['available'][where], full_labels=full_labels)
        assets.append(dict(method=name, path=str(destination), sha256=sha(destination)))
        if descriptor['online']:
            threshold = descriptor['threshold']
            result = evaluation.evaluate_arrays(index=common, score=score, valid=np.ones(len(common), bool),
                available_index=data['available'][where], invalid_reasons=[None] * len(common), full_labels=full_labels,
                threshold=threshold, bank_complete=True, original_length=len(full_labels),
                warmup_points=config['common_start'], affiliation_identity=config['affiliation_identity']['path'])
            evaluation.save_evaluation(folder / 'fixed_source_tau_online', result, index=common, score=score,
                valid=np.ones(len(common), bool), available_index=data['available'][where],
                invalid_reasons=[None] * len(common), full_labels=full_labels,
                provenance=dict(config=config, threshold_permission='unchanged sourceCAL', native_axis='unscored prefix/tail UNKNOWN'))
            row['strict'] = result['streams']['strict_point']
            row['confirmed3'] = result['streams']['confirmed3_current']
            row['coverage'] = result['event_coverage']
        else:
            row['strict'] = row['confirmed3'] = None
            row['online_metric_reason'] = 'OFFLINE_WINDOW_NATIVE_SCORE_NOT_CURRENT_TIME_AVAILABLE'
        rows.append(row)
    dump(output / 'RESULT.json', dict(status='COMPLETED_FIXED_COMMON_AXIS', seconds=time.monotonic() - started,
        common_axis=[config['common_start'], config['common_end']], rows=rows,
        continuous_inputs=assets, GPUh=0, new_forward=False, new_training=False, source_config=config))
    fields = ['method', 'points', 'positive_points', 'AP', 'ROC_AUC', 'PR_AUC_trapezoid', 'protocol', 'online_eligible']
    with (output / 'metrics.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(json.dumps(dict(status='COMPLETED', seconds=time.monotonic() - started,
                         ranking=[{key: row[key] for key in fields} for row in rows])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--trusted-config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.trusted_config_sha256:
        raise PermissionError('Frozen common-axis comparison required')
    run(json.loads(Path(args.config).read_text()))


if __name__ == '__main__':
    main()
