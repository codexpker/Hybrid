"""Read one frozen full V1 bank once; diagnose every stored scalar component.

No new score recipe is deployed. Seven fixed ablations reuse accepted continuous
arrays. All ranking is recomputed on the same predeclared native common axis.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tsad.hybrid_v1.chunked_bank import read_entries


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(asset):
    path = Path(asset['path'])
    if sha(path) != asset['sha256']:
        raise PermissionError('Frozen raw or accepted input changed')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--trusted-config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.trusted_config_sha256:
        raise PermissionError('Fixed all-component diagnostic config required')
    config = json.loads(Path(args.config).read_text())
    if sha(__file__) != config['source_sha256']:
        raise PermissionError('Diagnostic implementation changed')
    start = time.monotonic()
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(bound(config['full_bank_manifest']).read_text())
    identity = manifest['contract']['identity']
    if identity['checkpoint_sha256'] != config['expected_checkpoint_sha256']:
        raise PermissionError('Use full V1 c00f, never prototype or best seed')
    cal = json.loads(bound(config['source_calibration']).read_text())
    if identity['calibration_sha256'] != config['source_calibration']['sha256']:
        raise ValueError('Do not replace original source calibration')
    with np.load(bound(config['canonical']), allow_pickle=False) as data:
        index, canonical, valid, full_y = [data[key].copy() for key in ['index', 'score', 'valid', 'full_labels']]
    count = len(index)
    keys = ('r_tf', 'r_ms', 'r_pred', 'r_var')
    components = {key: np.empty(count, dtype=np.float64) for key in keys}
    canonical_from_raw = np.empty(count, dtype=np.float64)
    consumed = 0
    for position, entry in enumerate(read_entries(config['bank_directory'])):
        raw = entry['raw']
        if raw['index'] != int(index[position]) or raw['available_index'] != raw['index']:
            raise ValueError('Original causal native axes differ')
        canonical_from_raw[position] = raw['score']
        for key in keys:
            components[key][position] = raw[key]
        consumed += 1
    if consumed != count or not np.array_equal(canonical_from_raw, canonical):
        raise ValueError('Full raw canonical score differs from accepted compact inputs')
    contribution = {key: cal['weights'][key] * np.log1p(components[key] / cal['scales'][key])
                    for key in cal['active_components']}
    replay = sum(contribution.values())
    closure = float(np.max(np.abs(replay - canonical)))
    if closure > 1e-12:
        raise ValueError('Original fixed R3 recipe cannot reproduce canonical scores')
    common = np.arange(config['common_start'], config['common_end'])
    pos = np.searchsorted(index, common)
    if not np.array_equal(index[pos], common) or not valid[pos].all():
        raise ValueError('Keep identical valid common points for every diagnostic')
    y = full_y[common]
    diagnostics = []
    for key in keys:
        score = components[key][pos]
        groups = {}
        for name, mask in [('normal', y == 0), ('anomaly', y == 1)]:
            values = score[mask]
            groups[name] = dict(points=len(values), mean=float(values.mean()), median=float(np.median(values)),
                                q90=float(np.quantile(values, .9)))
        diagnostics.append(dict(component=key, AP=float(average_precision_score(y, score)),
            ROC_AUC=float(roc_auc_score(y, score)), groups=groups, used_in_R3=key in cal['active_components'],
            fixed_weight=cal['weights'].get(key, 0),
            interpretation='Scalar normalized residual/variance diagnostic only, not new component detector or selected score'))
    ablations = []
    for descriptor in config['ablations']:
        path = bound(descriptor['input'])
        provenance = json.loads(bound(descriptor['provenance']).read_text())
        if sha(path) != provenance['calculation_inputs_sha256']:
            raise PermissionError('Accepted ablation continuous input link broken')
        with np.load(path, allow_pickle=False) as data:
            locations = np.searchsorted(data['index'], common)
            if not np.array_equal(data['index'][locations], common) or not data['valid'][locations].all():
                raise ValueError('Ablation common native points differ')
            if not np.array_equal(data['full_labels'], full_y):
                raise ValueError('Ablation uses different original labels')
            score = data['score'][locations]
        ablations.append(dict(method=descriptor['method'], AP=float(average_precision_score(y, score)),
                              ROC_AUC=float(roc_auc_score(y, score)), points=len(common)))
    np.savez_compressed(output / 'all_original_scalar_components.npz', index=index, valid=valid, canonical=canonical,
        **components, **{'weighted_log_' + key: value for key, value in contribution.items()})
    result = dict(status='COMPLETED_ALL_COMPONENTS_AND_SEVEN_ABLATIONS', seconds=time.monotonic() - start,
        common_axis=[config['common_start'], config['common_end']], raw_records_verified=consumed,
        raw_canonical_bitexact=True, R3_max_absolute_closure=closure,
        components=diagnostics, ablations=ablations, new_forward=False, new_training=False, GPUh=0,
        no_new_thresholds=True, no_selected_components=True, no_score_weight_change=True, source_config=config)
    with (output / 'RESULT.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'], components=diagnostics, ablations=ablations)))


if __name__ == '__main__':
    main()
