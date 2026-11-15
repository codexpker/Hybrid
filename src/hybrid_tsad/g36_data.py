"""Fixed G36-5wk simulation mechanism data, selected without model outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ['BaselineSystem', 'CooCoiValStuck_0', 'OADamStuck_100', 'SupDucLea_20']
Y_COLUMNS = [46, 50, 59, 66]
U_COLUMNS = [48, 49, 71]
ROLES = {'FIT': [0, 1152], 'SELECT': [1440, 1728], 'CAL': [1728, 2016],
         'QUERY': [2016, 10021]}
HISTORY = 48
BLOCK = 4


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    frames = {name: pd.read_csv(source / (name + '.csv')) for name in FILES}
    base = frames[FILES[0]]
    assert base.shape == (10081, 114)
    time = base.iloc[:, 0].to_numpy()
    assert np.all(np.diff(time) == 300) and time[0] == 15724800
    assert time[2100] == 16354800 and time[10020] == 18730800
    columns = list(base.columns)
    y_names = [columns[c] for c in Y_COLUMNS]
    u_names = [columns[c] for c in U_COLUMNS] + ['clock_sin', 'clock_cos']
    processed = {}
    raw_audit = {}
    for name, frame in frames.items():
        assert list(frame.columns) == columns and frame.shape == base.shape
        values = frame.to_numpy(dtype=np.float64)
        assert np.isfinite(values).all() and np.array_equal(values[:, 0], time)
        clock = 2 * np.pi * (time % 86400) / 86400
        x = np.column_stack([values[:, Y_COLUMNS], values[:, U_COLUMNS],
                             np.sin(clock), np.cos(clock)])
        processed[name] = x
        raw_audit[name] = {
            'sha256': hashlib.sha256((source / (name + '.csv')).read_bytes()).hexdigest(),
            'weather_exact_to_baseline': bool(np.array_equal(values[:, [48, 49]], base.iloc[:, [48, 49]].to_numpy())),
            'mode_different_points': int(np.sum(values[:, 71] != base.iloc[:, 71].to_numpy())),
            'prefault_response_max_abs_diff': np.max(np.abs(x[:2100, :4] - processed[FILES[0]][:2100, :4]), axis=0).tolist(),
        }
    fit = processed[FILES[0]][slice(*ROLES['FIT'])]
    mean = fit.mean(0)
    scale = np.maximum(fit.std(0), 1e-6)
    records = []
    for name, x in processed.items():
        for role, (start, stop) in ROLES.items():
            if name != FILES[0] and role != 'QUERY':
                continue
            idx = np.arange(start, stop, dtype=np.int64)
            # Label is evaluation metadata only. Roles are separate physical files.
            label = ((idx >= 2100) & (idx <= 10020) & (name != FILES[0])).astype(np.int8)
            mode = x[idx, 6]
            file = output / f'{name}__{role}.npz'
            np.savez_compressed(file, x=((x[idx] - mean) / scale).astype(np.float32),
                                index=idx, time=time[idx], label=label, mode=mode)
            records.append({'run': name, 'role': role, 'path': file.name,
                            'bounds_half_open': [start, stop],
                            'first_scored_index': start + HISTORY + BLOCK - 1,
                            'n_scored': stop - start - HISTORY - BLOCK + 1,
                            'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
    np.savez(output / 'scaler.npz', mean=mean, scale=scale)
    manifest = dict(dataset='G36-5wk', source_version='figshare29297999v3/file57268553',
                    simulation=True, seed=2025, history=HISTORY, block=BLOCK,
                    response_columns=y_names, context_columns=u_names,
                    raw_audit=raw_audit, records=records, roles=ROLES,
                    fault_interval_inclusive=[2100, 10020], normal_budget=.01,
                    threshold_rule='normal CAL quantile .99 method=higher',
                    switching_radius_points=6, steady_exclude_start_points=12,
                    updates=3000, batch_size=64, select_every=300,
                    notes=['day4 unused temporal buffer; all raw footprints role-local',
                           'CAL only one simulated day; empirical budget, no population certification',
                           'QUERY baseline used for evaluation only; never counterfactual model input',
                           'three sustained simulated scenarios, not independent daily fault events',
                           'post-injection recovery excluded; state need not recover immediately',
                           'occupancy is past recorded scheduler mode, not a trusted actuator command'])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2, ensure_ascii=False))
