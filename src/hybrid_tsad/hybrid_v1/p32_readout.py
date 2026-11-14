"""Frozen P32 past-origin readouts, never forecast/model/weight modification.

Origin record o is issued at o-1 and prediction[j] targets o+j. Every target
t uses o=t-j for all j=0..31. Predictions are already in raw sensor units.
The three views share the original h1 channel mask and target-t past scale.
"""
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .chunked_bank import read_entries
from .controlled_mechanism import _unpack

HORIZON = 32
HISTORY = 512
VARIANTS = ('H1', 'MEAN', 'EVIDENCE')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_float32(value):
    original = np.asarray(value, dtype=np.float64)
    result = original.astype(np.float32)
    if not np.array_equal(original, result.astype(np.float64), equal_nan=True):
        raise ValueError('Cache float32 would lose archived values; do not silently round')
    return result


def cache_bank(bank_directory, output_directory, *, expected_checkpoint, role):
    """One SHA-verified raw pass; no new anomaly score or label access."""
    begin = time.monotonic()
    bank, output = Path(bank_directory), Path(output_directory)
    manifest = json.loads((bank / 'manifest.json').read_text())
    if not manifest['complete'] or manifest['contract']['identity']['checkpoint_sha256'] != expected_checkpoint:
        raise PermissionError('Only completed exact frozen model bank is eligible')
    output.mkdir(parents=True, exist_ok=False)
    entries = iter(read_entries(bank))
    first = next(entries)
    first_raw = first['raw']
    channels = np.asarray(first_raw['prediction']).shape[-1]
    records = manifest['committed_records']
    specs = dict(index=((records,), np.int64), issued=((records,), np.int64),
        prediction=((records, HORIZON, channels), np.float32), forecast_valid=((records, HORIZON, channels), bool),
        scale=((records, channels), np.float32), mean=((records, channels), np.float32),
        target=((records, channels), np.float32), observed=((records, channels), bool),
        native_h1_mask=((records, channels), bool), r_tf=((records,), np.float64),
        r_ms=((records,), np.float64), r_pred=((records,), np.float64), r_var=((records,), np.float64),
        canonical=((records,), np.float64))
    arrays = {name: np.lib.format.open_memmap(output / (name + '.npy'), mode='w+', dtype=dtype, shape=shape)
              for name, (shape, dtype) in specs.items()}
    stream = {key: first_raw[key] for key in ('domain', 'entity', 'run')}
    input_identity = first_raw['replay']['input_identity']
    maximum_abs_h1 = maximum_relative_h1 = 0.
    for position, entry in enumerate(itertools.chain([first], entries)):
        raw, replay = entry['raw'], entry['raw']['replay']
        if {key: raw[key] for key in stream} != stream or replay['input_identity'] != input_identity:
            raise ValueError('Never join roles/runs/input slices in one origin cache')
        origin = raw['index']
        if raw['forecast_issued_index'] != origin - 1 or raw['available_index'] != origin:
            raise ValueError('Archived origin/issued/current axes do not agree')
        if replay['forecast_history_bounds'] != [origin - HISTORY, origin] or replay['unarrived_future_observed'] is not False:
            raise PermissionError('Future observation or incorrect causal history footprint')
        prediction = exact_float32(raw['prediction'])
        valid = _unpack(replay['forecast_valid'])[0]
        target = exact_float32(replay['current_target']).reshape(channels)
        observed = _unpack(replay['current_observed']).reshape(channels)
        sigma = exact_float32(replay['forecast_stats']['scale']).reshape(channels)
        mean = exact_float32(replay['forecast_stats']['mean']).reshape(channels)
        if prediction.shape != (HORIZON, channels) or valid.shape != prediction.shape:
            raise ValueError('Keep complete original P32/C forecast axes')
        mask = observed & valid[0]
        if not np.array_equal(mask, np.isfinite(np.asarray(raw['per_channel']['r_pred']))):
            raise ValueError('Original h1 channel eligibility differs from archive')
        if mask.any():
            if not (np.isfinite(target[mask]).all() and np.isfinite(prediction[0, mask]).all()
                    and np.isfinite(sigma[mask]).all() and (sigma[mask] > 0).all()):
                raise ValueError('Invalid operands in originally eligible h1 channel')
            replay_h1 = np.square((prediction[0, mask].astype(float) - target[mask]) / sigma[mask]).mean()
            old_h1 = raw['r_pred']
            maximum_abs_h1 = max(maximum_abs_h1, abs(replay_h1 - old_h1))
            maximum_relative_h1 = max(maximum_relative_h1, abs(replay_h1 - old_h1) / max(abs(old_h1), 1e-12))
            # Only validation tolerance for float32 archived arithmetic; H1
            # output itself is exact saved r_pred, never this float64 replay.
            if not np.isclose(replay_h1, old_h1, rtol=5e-6, atol=1e-6):
                raise ValueError('Original h1 formula cannot reproduce archived r_pred')
        arrays['index'][position], arrays['issued'][position] = origin, origin - 1
        for name, value in dict(prediction=prediction, forecast_valid=valid, scale=sigma, mean=mean,
                                target=target, observed=observed, native_h1_mask=mask).items():
            arrays[name][position] = value
        for name in ('r_tf', 'r_ms', 'r_pred', 'r_var'):
            arrays[name][position] = raw[name]
        arrays['canonical'][position] = raw['score']
    if position + 1 != records or not np.all(np.diff(arrays['index']) > 0):
        raise ValueError('Complete unique chronological raw origin cache required')
    for array in arrays.values():
        array.flush()
    del arrays
    receipt = dict(status='COMPLETE_LOSSLESS_FORECAST_CACHE', seconds=time.monotonic() - begin,
        records=records, channels=channels, horizon=HORIZON, role=role, stream=stream,
        input_identity=input_identity, bank_manifest_sha256=sha(bank / 'manifest.json'),
        checkpoint=expected_checkpoint, raw_prediction_is_already_denormalized=True,
        h1_native_formula_float64_check=dict(max_absolute=maximum_abs_h1, max_relative=maximum_relative_h1,
            validation_rtol=5e-6, validation_atol=1e-6, H1_output='exact_original_r_pred_not_replay'),
        arrays={path.stem: dict(path=str(path), sha256=sha(path)) for path in sorted(output.glob('*.npy'))},
        no_labels_read=True, new_forward=False)
    with (output / 'CACHE.json').open('x') as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    return receipt


def load_cache(cache_json):
    info = json.loads(Path(cache_json).read_text())
    arrays = {}
    for name, descriptor in info['arrays'].items():
        if sha(descriptor['path']) != descriptor['sha256']:
            raise PermissionError('Frozen forecast cache changed')
        arrays[name] = np.load(descriptor['path'], mmap_mode='r', allow_pickle=False)
    return info, arrays


def align_predictions(origin_index, issued, prediction, forecast_valid, *, target_index, target,
                      native_h1_mask, sigma_t, original_h1, block_size=512):
    """Pure arrays; no labels, fitting, future GT or partial-horizon repair."""
    origin_index, target_index = np.asarray(origin_index), np.asarray(target_index)
    if np.any(np.diff(origin_index) <= 0) or np.any(np.diff(target_index) <= 0):
        raise ValueError('Use original unique sorted origin and target axes')
    count, channels = np.asarray(target).shape
    if (count != len(target_index) or np.asarray(native_h1_mask).shape != (count, channels)
            or np.asarray(sigma_t).shape != (count, channels)):
        raise ValueError('Keep target, original h1 mask and latest past-scale aligned')
    output = {variant: np.full(count, np.nan) for variant in VARIANTS}
    variance = np.full(count, np.nan)
    eligible = np.zeros(count, dtype=bool)
    reason = np.full(count, 'MISSING_PAST_ORIGIN', dtype='<U40')
    for start in range(0, count, block_size):
        end = min(start + block_size, count)
        times = target_index[start:end]
        required = times[:, None] - np.arange(HORIZON)[None]
        positions = np.searchsorted(origin_index, required)
        in_bounds = positions < len(origin_index)
        safe = np.clip(positions, 0, len(origin_index) - 1)
        present = in_bounds & (origin_index[safe] == required)
        causal = np.asarray(issued)[safe] == required - 1
        causal &= np.asarray(issued)[safe] < times[:, None]
        if np.any(present & ~causal):
            raise PermissionError('Forecast issue time is not before target availability')
        values = np.asarray(prediction[safe, np.arange(HORIZON)[None]], np.float64)
        valid = np.asarray(forecast_valid[safe, np.arange(HORIZON)[None]], bool)
        mask = np.asarray(native_h1_mask[start:end], bool)
        sigma = np.asarray(sigma_t[start:end], np.float64)
        y = np.asarray(target[start:end], np.float64)
        good_operands = np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
        native_valid = mask.any(1) & np.all(~mask | good_operands, axis=1) & np.isfinite(original_h1[start:end])
        complete_channel = np.all(~mask[:, None] | (valid & np.isfinite(values)), axis=(1, 2))
        good = present.all(1) & causal.all(1) & native_valid & complete_channel
        reason[start:end][present.all(1) & ~native_valid] = 'INVALID_ORIGINAL_H1'
        reason[start:end][present.all(1) & native_valid & ~complete_channel] = 'INCOMPLETE_NATIVE_CHANNEL_HORIZON'
        reason[start:end][good] = 'ELIGIBLE'
        # Mask before arithmetic, never silently change native h1 channels.
        safe_values = np.where(mask[:, None] & np.isfinite(values), values, 0.)
        safe_y, safe_sigma = np.where(mask, y, 0.), np.where(mask & good_operands, sigma, 1.)
        mean_prediction = safe_values.mean(axis=1)
        divisor = mask.sum(1).clip(min=1)
        mean_error = (np.square((mean_prediction - safe_y) / safe_sigma) * mask).sum(1) / divisor
        errors = np.square((safe_values - safe_y[:, None]) / safe_sigma[:, None])
        evidence_error = (errors * mask[:, None]).mean(1).sum(1) / divisor
        disagreement = (np.square((safe_values - mean_prediction[:, None]) / safe_sigma[:, None]) * mask[:, None]).mean(1).sum(1) / divisor
        slots = np.flatnonzero(good) + start
        output['H1'][slots] = original_h1[start:end][good]
        output['MEAN'][slots] = mean_error[good]
        output['EVIDENCE'][slots] = evidence_error[good]
        variance[slots] = disagreement[good]
        eligible[slots] = True
    return dict(index=target_index.copy(), variants=output, eligible=eligible, reason=reason,
                between_origin_variance_normalized=variance)
