"""Canonical scalar archives and paired reporting; no model or numerical truth I/O.

Only newly produced banks are decoded. Historical accepted continuous inputs
are consumed separately, without decoding their prediction/reconstruction banks.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (2025, 2026, 2027)
OFFICIAL = '528aea6e803f9e6b1ea5d07e64e44162309bdb016cc7721a8ac3753d942e6c7c'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bound(asset):
    path = Path(asset['path'])
    if sha(path) != asset['sha256']:
        raise PermissionError('Frozen asset changed: ' + str(path))
    return path


def read(asset):
    return json.loads(bound(asset).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def validate_new_banks(descriptors):
    """Authenticate ALL three completion receipts before any scalar/label access."""
    if sorted(d['seed'] for d in descriptors) != list(SEEDS):
        raise PermissionError('All three prespecified seeds are required')
    gates = []
    for d in descriptors:
        train, cal, query = [read(d[k]) for k in ('train_receipt', 'cal_receipt', 'query_receipt')]
        manifest, calibration = read(d['manifest']), read(d['calibration'])
        for receipt, phase in ((train, 'train'), (cal, 'cal'), (query, 'query')):
            if receipt['phase'] != phase or receipt['status'] != 'COMPLETED' or receipt['seed'] != d['seed']:
                raise PermissionError('Incorrect phase/seed completion')
            if receipt['labels_read'] is not False or receipt['official_sha256'] != OFFICIAL:
                raise PermissionError('No-query-label official-initialization receipt required')
        if train['attempts'] != 8000 or train['updates'] != 8000:
            raise ValueError('The full matched 8000 updates must finish')
        if train['scientific_status'] != 'VALID_SOURCE_SELECTED_ENDPOINT' or cal['scientific_status'] != 'VALID_SOURCE_CAL':
            raise ValueError('Invalid source-only selection/calibration')
        checkpoint = train['checkpoint_sha256']
        if len({r['initializer_sha256'] for r in (train,cal,query)}) != 1:
            raise ValueError('Initialization implementation differs across phases')
        if any(r['checkpoint_sha256'] != checkpoint for r in (cal, query)):
            raise ValueError('Mixed checkpoint chain')
        if sha(bound(d['checkpoint'])) != checkpoint:
            raise ValueError('Trained payload differs from phase receipts')
        if cal['records'] != 1024 or calibration['valid_windows'] != 1024:
            raise ValueError('Original CAL1024 required, not P32 CAL1018')
        if calibration['domains'] != ['SMD', 'Sunspot'] or calibration['source_domain_equal_weight'] is not True:
            raise PermissionError('Only original domain-balanced source calibration')
        if calibration['weights'] != {'r_tf': .25, 'r_ms': .25, 'r_pred': .5}:
            raise ValueError('Original h1 R3 recipe required')
        calsha = d['calibration']['sha256']
        if cal['calibration_sha256'] != calsha or query['calibration_sha256'] != calsha:
            raise ValueError('Mixed calibration chain')
        contract = manifest['contract']
        if not manifest['complete'] or manifest['committed_records'] != 87329 or query['points'] != 87329:
            raise PermissionError('All complete original 87329 records required')
        if query['bank_manifest_sha256'] != d['manifest']['sha256']:
            raise ValueError('Query completion manifest changed')
        if contract['ranges'] != [{'domain': 'PSM', 'entity': 'anonymous_PSM', 'run': 'test_original', 'start': 512, 'end': 87841}]:
            raise ValueError('Exact original PSM stream footprint required')
        identity = contract['identity']
        identity_keys = {'input_sha256','config_sha256','checkpoint_sha256','calibration_sha256','source_sha256'}
        if set(d['expected_identity']) != identity_keys:
            raise PermissionError('Bind all five actual producer identities')
        for key, expected in d['expected_identity'].items():
            if identity[key] != expected:
                raise ValueError('Bound bank identity changed: ' + key)
        if identity['checkpoint_sha256'] != checkpoint or identity['calibration_sha256'] != calsha:
            raise ValueError('Bank checkpoint/calibration chain mismatch')
        if contract['threshold'] != calibration['threshold']:
            raise ValueError('No target/query threshold fitting')
        gates.append(dict(seed=d['seed'], manifest_sha256=d['manifest']['sha256'], checkpoint_sha256=checkpoint,
                          calibration_sha256=calsha, threshold=calibration['threshold'], records=87329))
    return gates


def _field(node, key):
    if node[0] != 'dict':
        raise ValueError('Typed dictionary required')
    return dict(node[1])[key]


def _scalar(node, arrays):
    if node[0] == 'str':
        return node[1]
    if node[0] == 'null':
        return None
    kind, name, dtype, start, count, shape = node
    if kind not in ('int', 'float', 'bool', 'numpy_scalar') or count != 1 or shape != []:
        raise ValueError('Canonical scalar field required')
    values = arrays[name]
    if values.dtype.str != dtype:
        raise ValueError('Typed scalar payload mismatch')
    return values[start].item()


def validate_score_replay(canonical, replay):
    """Verification only across NumPy/libm versions; NEVER replace canonical.

    Tolerances are fixed before new QUERY metrics. Relative error is symmetric
    and uses a 1e-300 denominator floor solely for this diagnostic statistic.
    """
    if not np.isfinite(canonical) or not np.isfinite(replay) or not np.isclose(
            canonical, replay, rtol=1e-12, atol=1e-12):
        raise ValueError('Canonical R3 replay exceeds fixed numerical verification tolerance')
    difference = abs(float(canonical)-float(replay))
    relative = difference/max(abs(float(canonical)), abs(float(replay)), 1e-300)
    return difference, relative


def archive_scalars(descriptor, output):
    """One bounded scalar-only decode of NEW immutable typed banks. No truth."""
    from .scoring import apply_calibration
    manifest = read(descriptor['manifest'])
    calibration = read(descriptor['calibration'])
    directory = bound(descriptor['manifest']).parent
    fields = ('domain', 'entity', 'run', 'index', 'available_index', 'score', 'valid', 'r_tf', 'r_ms', 'r_pred')
    rows = {k: [] for k in fields if k not in ('domain', 'entity', 'run')}
    position = 0
    verification = dict(rtol=1e-12, atol=1e-12, valid_compared=0,
        max_absolute_difference=0., max_relative_difference=0.,
        non_bitexact_replays=0, strict_threshold_sensitive_points=0,
        relative_denominator='max(abs(canonical),abs(replay),1e-300)',
        archived_and_ranked_score='original bank canonical; replay is verification only')
    for chunk in manifest['chunks']:
        path = directory / chunk['file']
        if path.parent != directory or sha(path) != chunk['sha256'] or chunk['start_position'] != position:
            raise ValueError('Immutable chunk path/hash/prefix changed')
        with np.load(path, allow_pickle=False) as arrays:
            trees = json.loads(arrays['tree_utf8'].tobytes().decode('utf-8'))
            # NpzFile does not cache decompression. Decode each numeric pool once
            # per bounded chunk, without constructing forecast/replay objects.
            pools = {name: arrays[name] for name in arrays.files if name != 'tree_utf8'}
            if len(trees) != chunk['records']:
                raise ValueError('Chunk record count differs')
            for tree in trees:
                raw = _field(tree, 'raw')
                record = {k: _scalar(_field(raw, k), pools) for k in fields}
                if tuple(record[k] for k in ('domain','entity','run','index')) != ('PSM','anonymous_PSM','test_original',512+position):
                    raise ValueError('Changed canonical original axis')
                canonical = apply_calibration(record, calibration)
                if bool(record['valid']) != canonical['valid']:
                    raise ValueError('Canonical validity differs from original R3')
                if canonical['valid']:
                    absolute, relative = validate_score_replay(record['score'], canonical['score'])
                    verification['valid_compared'] += 1
                    verification['max_absolute_difference'] = max(verification['max_absolute_difference'], absolute)
                    verification['max_relative_difference'] = max(verification['max_relative_difference'], relative)
                    verification['non_bitexact_replays'] += int(record['score'] != canonical['score'])
                    verification['strict_threshold_sensitive_points'] += int(
                        (record['score'] > calibration['threshold']) != (canonical['score'] > calibration['threshold']))
                if record['available_index'] > record['index']:
                    raise PermissionError('Hybrid current-time score must be causal')
                for key in rows:
                    rows[key].append(record[key])
                position += 1
    if position != 87329:
        raise ValueError('Incomplete new scalar archive')
    output = Path(output)
    with output.open('xb') as handle:
        np.savez_compressed(handle, **{k: np.asarray(v) for k, v in rows.items()})
    return dict(path=str(output), sha256=sha(output), records=position,
                bank_manifest=descriptor['manifest'], threshold=calibration['threshold'], labels_read=False,
                score_replay_verification=verification)


def paired_tables(rows):
    """All seeds, sample SD, paired differences. No seed selection or ensemble."""
    lookup = {(r.get('initialization'), r.get('seed')): r for r in rows if r.get('initialization')}
    numeric = sorted(set.intersection(*(set(r) for r in lookup.values())) - {'seed','points','positive_points','threshold'})
    numeric = [k for k in numeric if all(type(r[k]) in (float, int) or r[k] is None for r in lookup.values())]
    paired, summary = [], []
    for metric in numeric:
        differences = []
        for seed in SEEDS:
            old, new = lookup['RANDOM', seed].get(metric), lookup['OFFICIAL528', seed].get(metric)
            delta = None if old is None or new is None else new-old
            paired.append(dict(seed=seed, metric=metric, random=old, official528=new, delta_new_minus_old=delta))
            differences.append(delta)
        for kind in ('RANDOM', 'OFFICIAL528', 'PAIRED_DELTA'):
            values = differences if kind == 'PAIRED_DELTA' else [lookup[kind, s].get(metric) for s in SEEDS]
            complete = all(v is not None for v in values)
            higher = metric in ('AP','ROC_AUC') or metric.endswith(('_TP','_Precision','_Recall','_F1','_event_precision','_event_recall','_main_events_hit','_steady_detected_points'))
            lower = metric.endswith(('_FP','_normal_FPR','_normal_FP_fragments','_normal_FP_fragments_per1000','_hit_only_delay_median_points','_right_censored_main_events','_max_main_event_observed_miss_points','_steady_all_zero_detected_events','_steady_main_zero_detected_events'))
            worst = (min(values) if higher else max(values)) if complete and (higher or lower) else None
            # Report both extrema with seed identities rather than silently
            # imposing "larger is better" on FPR, delays or TP counts.
            summary.append(dict(initialization=kind, metric=metric, n_prespecified=3,
                n_valid=sum(v is not None for v in values), mean=float(np.mean(values)) if complete else None,
                sample_SD=float(np.std(values, ddof=1)) if complete else None,
                min=float(min(values)) if complete else None, max=float(max(values)) if complete else None,
                min_seed=[SEEDS[i] for i,v in enumerate(values) if complete and v==min(values)] if complete else [],
                max_seed=[SEEDS[i] for i,v in enumerate(values) if complete and v==max(values)] if complete else [],
                benefit_direction='higher' if higher else ('lower' if lower else 'descriptive_only'),
                worst_seed=[SEEDS[i] for i,v in enumerate(values) if worst is not None and v==worst],
                worst_value=worst,
                inference='three optimization seeds on one historical development domain; no independent-domain confidence claim'))
    return paired, summary
