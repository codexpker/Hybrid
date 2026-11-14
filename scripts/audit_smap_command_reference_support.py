"""Label-free eligibility audit of exact published command-vector FIT support.

This does not produce anomaly scores or fit a detector. Context is all 24
command encoding columns at t-1, never argmax, target, or a future row.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_keys(native, index):
    index = np.asarray(index, dtype=np.int64)
    if native.ndim != 2 or native.shape[1] != 25:
        raise ValueError('Expected original 25-column Telemanom input')
    if np.any(index < 1) or np.any(index >= len(native)):
        raise ValueError('Context at t-1 must be in the same original run')
    command = native[index - 1, 1:25]
    if not np.isfinite(command).all() or not np.isin(command, [0., 1.]).all():
        raise ValueError('Published binary command encoding assumption failed')
    command = command.astype(np.uint8)
    packed = np.packbits(command, axis=1, bitorder='little')
    keys = np.asarray([int(row[0]) | int(row[1]) << 8 | int(row[2]) << 16
                       for row in packed], dtype=np.uint32)
    return command, keys


def counts_for_reference(reference_keys, keys):
    unique, counts = np.unique(reference_keys, return_counts=True)
    lookup = {int(key): int(count) for key, count in zip(unique, counts)}
    return np.asarray([lookup.get(int(key), 0) for key in keys], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    workspace = Path(args.workspace)
    native_root = workspace / 'data/smap_native_version1_53_20261105'
    product_root = workspace / 'results/roca_information_smap_20261106_v1'
    products = json.loads((product_root / 'PRODUCTS.json').read_text())
    entities = sorted({row['entity'] for row in products if row['method'] == 'RoCA_FULL_T'})
    if len(entities) != 53:
        raise ValueError('Keep all original 53 SMAP entities')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows, bindings = [], []
    for entity in entities:
        natives = {}
        for original_role in ['train', 'test']:
            path = native_root / original_role / (entity + '.npy')
            natives[original_role] = np.load(path, allow_pickle=False)
            bindings.append(dict(path=str(path), sha256=sha(path)))
        role_data = {}
        for role in ['FIT', 'CAL', 'QUERY']:
            axes_path = product_root / 'RoCA_FULL_T' / entity / role / 'AXES.json'
            axes = json.loads(axes_path.read_text())
            bindings.append(dict(path=str(axes_path), sha256=sha(axes_path)))
            index = np.asarray(axes['indices'], dtype=np.int64)
            if len(index) != len(np.unique(index)) or not np.all(np.diff(index) > 0):
                raise ValueError('Unique chronological original endpoints required')
            if role == 'FIT' and len(index) != 64 or role == 'CAL' and len(index) != 32:
                raise ValueError('Do not replace frozen reference or calibration indices')
            native = natives['test' if role == 'QUERY' else 'train']
            if role == 'QUERY' and not np.array_equal(index, np.arange(512, len(native))):
                raise ValueError('All original query endpoints must remain')
            command, keys = command_keys(native, index)
            role_data[role] = dict(index=index, command=command, command_key=keys)
        reference = role_data['FIT']['command_key']
        for role, data in role_data.items():
            support = counts_for_reference(reference, data['command_key'])
            counts = data['command'].sum(axis=1)
            missing_keys = sorted({int(key) for key in data['command_key'][support == 0]})
            row = dict(entity=entity, role=role, points=len(support),
                unique_command_vectors=len(np.unique(data['command_key'])),
                reference_vectors=len(np.unique(reference)), zero_active=int((counts == 0).sum()),
                one_active=int((counts == 1).sum()), multi_active=int((counts > 1).sum()),
                exact_reference_support_zero=int((support == 0).sum()),
                exact_reference_support_1_to_4=int(((support > 0) & (support < 5)).sum()),
                exact_reference_support_ge5=int((support >= 5).sum()),
                minimum_reference_support=int(support.min()), all_points_supported_ge5=bool(np.all(support >= 5)),
                missing_reference_command_keys=missing_keys)
            rows.append(row)
            destination = output / 'context' / entity / (role + '.npz')
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, **data, exact_FIT_reference_count=support)
    aggregates = []
    fields = ['points', 'zero_active', 'one_active', 'multi_active',
              'exact_reference_support_zero', 'exact_reference_support_1_to_4',
              'exact_reference_support_ge5']
    for role in ['FIT', 'CAL', 'QUERY']:
        selected = [row for row in rows if row['role'] == role]
        aggregates.append(dict(role=role, entities=len(selected),
            entities_all_points_supported_ge5=sum(row['all_points_supported_ge5'] for row in selected),
            **{key: sum(row[key] for row in selected) for key in fields}))
    result = dict(status='COMPLETED_LABEL_FREE_REFERENCE_ELIGIBILITY',
        seconds=time.monotonic() - started, context='full native command columns 1:25 at original t-1',
        interpretation='Support audit only; matching counts do not establish physical regimes or detection value',
        no_query_labels=True, no_query_scores=True, no_new_forward=True, GPUh=0,
        aggregates=aggregates, entities=rows, input_bindings=bindings,
        derived_context_files=[dict(path=str(path), sha256=sha(path))
                               for path in sorted((output / 'context').rglob('*.npz'))],
        source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'], aggregates=aggregates)))


if __name__ == '__main__':
    main()
