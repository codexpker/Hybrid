"""Fixed evaluation of an event-supervised matched risk comparison.

No fitting on QUERY; use the existing complete-axis SWaT metric implementation.
The two new methods share training permissions. External no-label baselines do not.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .terminal_aligned_experiment import load, read, dump, sha
from .normality_density_evaluation import evaluate_series, table
from .alarm_complementarity import confirmed, longest_miss


CAL_AXIS = np.arange(446173, 495000, dtype=np.int64)
QUERY_AXIS = np.arange(608535, 944919, dtype=np.int64)
CAL_SELECTED = np.linspace(446173, 494999, 512, dtype=np.int64)


def extract(bank, role):
    """Explicit score and index contract; never silently flatten extra heads."""
    assert role in ('CAL', 'QUERY')
    index = np.asarray(bank['index'])
    score = np.asarray(bank['score'])
    assert index.ndim == score.ndim == 1 and index.shape == score.shape
    assert np.issubdtype(index.dtype, np.integer) and np.isfinite(score).all()
    expected = QUERY_AXIS if role == 'QUERY' else CAL_AXIS
    if role == 'CAL' and len(index) == len(CAL_SELECTED):
        expected = CAL_SELECTED
    assert np.array_equal(index, expected), f'{role} axis mismatch'
    for key in ('available_index', 'max_input_index'):
        if key in bank:
            assert np.array_equal(bank[key], index), f'{key} violates endpoint contract'
    return index, score


def threshold(cal_index, cal_score):
    pos = np.searchsorted(cal_index, CAL_SELECTED)
    assert np.array_equal(cal_index[pos], CAL_SELECTED)
    return float(np.quantile(cal_score[pos], .99, method='higher'))


def seal(workspace, output, config):
    cfg = read(config)
    assert cfg['primary_pair'] == ['Hybrid_F_tail25', 'Hybrid_F_mean']
    root = output/'evaluation'; root.mkdir(exist_ok=False)
    dest = root/'SWAT'; dest.mkdir(); (dest/'banks').mkdir()
    descriptors = []; sources = []
    for method in cfg['methods']:
        roles = {}
        for role in ('CAL', 'QUERY'):
            raw_path = Path(method[role])
            path = raw_path if raw_path.is_absolute() else workspace/raw_path
            z = load(path)
            index, score = extract(z, role)
            roles[role] = score
            roles[role+'_index'] = index
            sources.append(dict(method=method['name'], role=role,
                                path=str(path), sha256=sha(path)))
        tau = threshold(roles['CAL_index'], roles['CAL'])
        f = dest/'banks'/(method['name']+'.npz')
        np.savez_compressed(f, **roles, selected_CAL=CAL_SELECTED)
        descriptors.append(dict(method=method['name'], path=str(f), sha256=sha(f),
            threshold=tau, training_permission=method['training_permission']))
    dump(dest/'RAW_SEAL.json', dict(status='SEALED_BEFORE_QUERY_LABEL_JOIN',
        methods=descriptors, sources=sources, config_sha256=sha(config),
        implementation_sha256=sha(__file__),
        metric_implementation_sha256=sha(Path(__file__).with_name('normality_density_evaluation.py')),
        event_implementation_sha256=sha(Path(__file__).parent/'hybrid_v1/evaluation.py'),
        threshold='own original 512 CAL endpoints q99higher; strict greater-than',
        primary_pair=cfg['primary_pair'], current_scope='F-path development; no U robustness claim',
        historical_exposure=True, fixed_query_event_count=22, long_events=9,
        label_semantics='attack intervals, not certified physical-fault point truth'))
    dump(root/'ALL_DOMAINS_SEALED.json', dict(methods=descriptors,
        config_sha256=sha(config), QUERY_labels_used_for_selection=False))


def evaluate(workspace, output):
    import pandas as pd
    started = time.monotonic()
    root = output/'evaluation'; folder = root/'SWAT'
    read(root/'ALL_DOMAINS_SEALED.json')
    seal_record = read(folder/'RAW_SEAL.json')
    assert seal_record['implementation_sha256'] == sha(__file__)
    assert seal_record['metric_implementation_sha256'] == sha(Path(__file__).with_name('normality_density_evaluation.py'))
    evaluate_series(workspace, output, 'SWAT')
    summary = pd.read_csv(folder/'summary.csv')
    permissions = {r['method']: r['training_permission'] for r in seal_record['methods']}
    summary['training_permission'] = summary.method.map(permissions)
    summary['normal_budget_met'] = summary.normal_FPR <= .01
    summary['calibration_status'] = 'original independent CAL role; empirical QUERY check only'
    summary['event_scope'] = '22 events from one exposed recording; not 22 independent runs'
    table(root/'main_comparison.csv', summary.replace({np.nan: None}).to_dict('records'))
    event_frame = pd.read_csv(folder/'per_event.csv')
    eligible = event_frame[event_frame.main_eligible].copy()
    expected = set(eligible[eligible.method == 'Hybrid_F_mean'].event_id)
    assert len(expected) == 22
    steady_misses = []
    for desc in seal_record['methods']:
        bank = load(desc['path'])
        ix = bank['QUERY_index']; strict = bank['QUERY'] > desc['threshold']
        for stream, alarm in [('strict', strict), ('confirmed3', confirmed(ix, strict))]:
            frame = eligible[(eligible.method == desc['method']) & (eligible.stream == stream)]
            assert set(frame.event_id) == expected
            for _, e in frame.iterrows():
                if e.total_points > 512:
                    lo = int(e.local_start)+512-int(ix[0]); hi = int(e.local_end)-int(ix[0])
                    steady_misses.append(dict(method=desc['method'], stream=stream,
                        event_id=e.event_id, interior_points=hi-lo,
                        interior_longest_miss=longest_miss(alarm[lo:hi])))
    table(root/'long_event_interior_misses.csv', steady_misses)
    paired = []
    for stream in ('strict','confirmed3'):
        a = eligible[(eligible.method == 'Hybrid_F_mean') & (eligible.stream == stream)].set_index('event_id')
        b = eligible[(eligible.method == 'Hybrid_F_tail25') & (eligible.stream == stream)].set_index('event_id')
        for eid in sorted(expected):
            row = dict(event_id=eid, stream=stream,
                interpretation='paired descriptive event result; not independent-run inference')
            for key in ('coverage','steady_coverage','longest_miss','delay_points'):
                x, y = a.loc[eid,key], b.loc[eid,key]
                row['mean_'+key] = None if pd.isna(x) else float(x)
                row['tail_'+key] = None if pd.isna(y) else float(y)
                row['delta_'+key] = None if pd.isna(x) or pd.isna(y) else float(y-x)
            paired.append(row)
    table(root/'paired_events.csv', paired)
    dump(root/'FINAL_RECEIPT.json', dict(status='COMPLETED', CPU_seconds=time.monotonic()-started,
        source_sha256=sha(__file__), no_threshold_search=True,
        no_retraining=True, all_query_events_retained=True))


def main():
    p = argparse.ArgumentParser(); p.add_argument('phase',choices=['seal','evaluate'])
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--config',type=Path)
    a=p.parse_args()
    if a.phase == 'seal': seal(a.workspace.resolve(),a.output.resolve(),a.config.resolve())
    else: evaluate(a.workspace.resolve(),a.output.resolve())


if __name__ == '__main__':
    main()
