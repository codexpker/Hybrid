"""Fixed P32 readout phases; all source calibration precedes query metrics.

The prepare-psm phase only reorders already frozen forecasts and never opens
labels. Source assembly/evaluation phases consume sole-operator forecast assets.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hybrid_tsad.hybrid_v1.p32_readout import align_predictions, load_cache, VARIANTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(descriptor):
    path = Path(descriptor['path'])
    if sha(path) != descriptor['sha256']:
        raise PermissionError('Frozen P32 input changed: ' + str(path))
    return path


def dump(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)


def prepare_psm(config):
    start = time.monotonic()
    info, cache = load_cache(bound(config['cache']))
    if info['stream'] != {'domain': 'PSM', 'entity': 'anonymous_PSM', 'run': 'test_original'}:
        raise ValueError('Do not cross domain, entity, or role')
    target_index = np.arange(543, 87841)
    location = np.searchsorted(cache['index'], target_index)
    if not np.array_equal(cache['index'][location], target_index):
        raise ValueError('Complete predeclared mature PSM axis required')
    result = align_predictions(cache['index'], cache['issued'], cache['prediction'], cache['forecast_valid'],
        target_index=target_index, target=cache['target'][location], native_h1_mask=cache['native_h1_mask'][location],
        sigma_t=cache['scale'][location], original_h1=cache['r_pred'][location])
    use = result['eligible']
    error = result['variants']['EVIDENCE'][use] - result['variants']['MEAN'][use] - result['between_origin_variance_normalized'][use]
    maximum_error = float(np.max(np.abs(error))) if use.any() else None
    if use.any() and not np.allclose(error, 0., atol=1e-10, rtol=0):
        raise ValueError('Jensen decomposition failed under fixed common scale/mask')
    if not np.array_equal(result['variants']['H1'][use], cache['r_pred'][location][use]):
        raise ValueError('H1 must retain exact archived native arithmetic')
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    destination = output / 'PSM_ALIGNED.npz'
    np.savez_compressed(destination, index=target_index, eligible=use, reason=result['reason'],
        r_tf=cache['r_tf'][location], r_ms=cache['r_ms'][location],
        H1=result['variants']['H1'], MEAN=result['variants']['MEAN'], EVIDENCE=result['variants']['EVIDENCE'],
        between_origin_variance_normalized=result['between_origin_variance_normalized'],
        native_h1_channels=cache['native_h1_mask'][location].sum(1))
    summary = dict(status='PREPARED_ALL_THREE_READOUTS_NO_LABELS', seconds=time.monotonic() - start,
        points=len(target_index), eligible=int(use.sum()), mature_axis=[543, 87841], common_axis=[543, 87744],
        reasons={str(reason): int((result['reason'] == reason).sum()) for reason in np.unique(result['reason'])},
        Jensen_max_absolute_error=maximum_error, H1_native_exact=True, no_labels_read=True,
        no_new_forward=True, no_new_training=True, GPUh=0, source_config=config,
        aligned=dict(path=str(destination), sha256=sha(destination)))
    dump(output / 'PREPARED.json', summary)
    print(json.dumps({key: summary[key] for key in ('status', 'seconds', 'points', 'eligible', 'Jensen_max_absolute_error')}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare-psm'], required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--trusted-config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.trusted_config_sha256:
        raise PermissionError('Frozen P32 phase configuration required')
    config = json.loads(Path(args.config).read_text())
    for path, digest in config['source_sha256'].items():
        if sha(path) != digest:
            raise PermissionError('Frozen P32 consumer changed')
    prepare_psm(config)


if __name__ == '__main__':
    main()
