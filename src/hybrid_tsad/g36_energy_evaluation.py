"""Fixed matched context-energy evaluation; healthy copies are posthoc only.

Reuses the frozen G36 metric definitions and previous paired time/mode masks.
No model fitting, score editing, threshold search, or new event boundaries.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .g36_data import FILES
from .g36_evaluation import evaluate, load_bank, confirm, spans
from .g36_paired_diagnostic import quadrants


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paired_summary(healthy_score, fault_score, threshold, mask):
    """Confirm on the full time axis BEFORE applying diagnostic strata."""
    healthy = confirm(np.asarray(healthy_score) > threshold)
    fault = confirm(np.asarray(fault_score) > threshold)
    mask = np.asarray(mask, bool)
    quad = quadrants(healthy, fault)
    n = int(mask.sum())
    row = {'n': n}
    for code, name in enumerate(['neither', 'healthy_only', 'fault_only', 'both']):
        count = int(((quad == code) & mask).sum())
        row[name + '_points'] = count
        row[name + '_rate'] = count / n if n else None
    row['longest_no_fault_alarm_minutes'] = 5 * max(
        [b-a for a, b in spans(mask & ~fault)] or [0])
    row['longest_no_exclusive_alarm_minutes'] = 5 * max(
        [b-a for a, b in spans(mask & ~(fault & ~healthy))] or [0])
    assert sum(row[k + '_points'] for k in ['neither', 'healthy_only', 'fault_only', 'both']) == n
    return row


def score_resolution(banks, output):
    """Read-only posthoc score support diagnosis; never a replacement score."""
    rows = []
    for method, folder in banks.items():
        for path in sorted(Path(folder).glob('*__*.npz')):
            with np.load(path, allow_pickle=False) as z:
                if 'normal_tail_count' not in z:
                    continue
                s, count = z['score'], z['normal_tail_count']
                d, observed = z['normal_distance'], z['observed_distance']
                assert np.array_equal(count, (d >= observed[:, None]).sum(1))
                run, role = path.stem.split('__')
                rows.append(dict(method=method, run=run, role=role, n=len(s),
                    distinct_scores=len(np.unique(s)), zero_tail_count=int((count == 0).sum()),
                    zero_tail_fraction=float((count == 0).mean()),
                    min_score=float(s.min()), max_score=float(s.max()),
                    observed_distance_mean=float(observed.mean()),
                    normal_distance_mean=float(d.mean()), bank_sha256=sha(path)))
    with Path(output).open('x', encoding='utf-8', newline='') as f:
        pd.DataFrame(rows).to_csv(f, index=False)


def run(data, bank_file, previous_pairs, output):
    banks = {k: Path(v) for k, v in json.loads(bank_file.read_text()).items()}
    inputs = [bank_file, data/'manifest.json', data/'scaler.npz']
    inputs += [folder/(r+'__'+role+'.npz') for folder in banks.values()
               for role, records in [('CAL', FILES[:1]), ('QUERY', FILES)] for r in records]
    inputs += [previous_pairs/(r+'__paired.npz') for r in FILES[1:]]
    before = {str(p): sha(p) for p in inputs}
    evaluate(data, banks, output)
    normal = pd.read_csv(output/'normal_false_alarms.csv', float_precision='round_trip').set_index('method')
    rows = []
    for r in FILES[1:]:
        # Reuse prior numeric/mode eligibility; no new selection from outcomes.
        with np.load(previous_pairs/(r+'__paired.npz'), allow_pickle=False) as z:
            idx = z['index'].copy()
            match = z['mode_footprint_match'].copy()
            switch = z['baseline_switch'].copy()
            current_match = z['mode_match'].copy()
        with np.load(data/(FILES[0]+'__QUERY.npz'), allow_pickle=False) as z:
            loc = np.searchsorted(z['index'], idx)
            assert np.array_equal(z['index'][loc], idx)
            occupied = z['mode'][loc] > .5
        steady = idx >= 2112
        strata = dict(all_injected=idx >= 2100, steady_all=steady,
            steady_aligned=steady & match,
            steady_aligned_non_switch=steady & match & ~switch,
            steady_aligned_occupied=steady & match & occupied,
            steady_aligned_unoccupied=steady & match & ~occupied,
            steady_current_mode_mismatch=steady & ~current_match,
            steady_current_match_history_mismatch=steady & current_match & ~match)
        for method, folder in banks.items():
            hi, hs, _, _ = load_bank(folder, data, FILES[0], 'QUERY')
            fi, fs, _, _ = load_bank(folder, data, r, 'QUERY')
            assert np.array_equal(idx, hi) and np.array_equal(idx, fi)
            threshold = float(normal.loc[method, 'threshold'])
            for stratum, mask in strata.items():
                rows.append(dict(method=method, run=r, stratum=stratum,
                    **paired_summary(hs, fs, threshold, mask)))
    pd.DataFrame(rows).to_csv(output/'paired_c3.csv', index=False)
    scenarios = pd.read_csv(output/'per_scenario.csv')
    summary = scenarios.groupby('method', sort=False).agg(
        macro_ap=('ap', 'mean'), macro_roc=('roc', 'mean'),
        macro_steady_c3=('steady_c3_coverage', 'mean'),
        worst_longest_c3_miss_minutes=('longest_c3_miss_minutes', 'max')).join(normal)
    summary.to_csv(output/'main_comparison.csv')
    assert all(sha(p) == h for p, h in before.items())
    (output/'INPUT_INTEGRITY.json').write_text(json.dumps(dict(input_sha256=before,
        inputs_unchanged=True, evaluator_sha256=sha(__file__),
        shared_metric_sha256=sha(Path(__file__).with_name('g36_evaluation.py')),
        healthy_copy='posthoc diagnostic only; never a detector input',
        query_used_for_fitting=False, normal_budget='unchanged CAL q99 higher, strict >',
        limits='one simulated normal CAL day; three correlated fault scenarios, not real confirmation'), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--banks', type=Path, required=True)
    p.add_argument('--previous-pairs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.data, a.banks, a.previous_pairs, a.output)
