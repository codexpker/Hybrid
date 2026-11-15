"""Fixed G36 evaluation: continuous raw axes, no threshold fitted on QUERY."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .g36_data import FILES


def spans(mask):
    changes = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def confirm(mask):
    mask = np.asarray(mask, bool)
    out = np.zeros_like(mask)
    out[2:] = mask[2:] & mask[1:-1] & mask[:-2]
    return out


def max_gap(mask, allowed):
    return max([b-a for a, b in spans(np.asarray(allowed) & ~np.asarray(mask))] or [0])


def load_bank(folder, data, run, role):
    path = folder / f'{run}__{role}.npz'
    with np.load(path) as z:
        index, score = z['index'].copy(), z['score'].copy()
    with np.load(data / f'{run}__{role}.npz') as z:
        loc = np.searchsorted(z['index'], index)
        assert np.array_equal(z['index'][loc], index)
        assert np.array_equal(index, z['index'][51:]), (path, index.shape)
        label, mode = z['label'][loc].copy(), z['mode'][loc].copy()
    assert score.shape == index.shape and np.isfinite(score).all()
    return index, score, label, mode


def evaluate(data, banks, output):
    output.mkdir(parents=True, exist_ok=False)
    rows, normal_rows, pooled_rows = [], [], []
    for method, folder in banks.items():
        _, cal, _, _ = load_bank(folder, data, FILES[0], 'CAL')
        threshold = float(np.quantile(cal, .99, method='higher'))
        nidx, ns, _, nm = load_bank(folder, data, FILES[0], 'QUERY')
        na = ns > threshold
        nc = confirm(na)
        switch = np.zeros(len(nidx), bool)
        for pos in np.flatnonzero(nm[1:] != nm[:-1]) + 1:
            switch[max(0, pos-6):min(len(switch), pos+7)] = True
        normal_rows.append(dict(method=method, threshold=threshold, cal_n=len(cal),
            cal_fpr=float(np.mean(cal > threshold)), normal_n=len(ns),
            normal_false_points=int(na.sum()), normal_fpr=float(na.mean()),
            normal_c3_fpr=float(nc.mean()), normal_false_segments=len(spans(na)),
            normal_c3_segments=len(spans(nc)),
            normal_false_segments_per_1000=1000*len(spans(na))/len(na),
            switching_n=int(switch.sum()), switching_fpr=float(na[switch].mean()),
            other_normal_fpr=float(na[~switch].mean())))
        all_scores, all_labels = [ns], [np.zeros(len(ns), np.int8)]
        hits, c3hits, predicted, c3predicted = 0, 0, len(spans(na)), len(spans(nc))
        for run in FILES[1:]:
            idx, s, y, mode = load_bank(folder, data, run, 'QUERY')
            assert np.array_equal(idx, nidx)
            a, c = s > threshold, confirm(s > threshold)
            positive = y.astype(bool)
            steady = positive & (idx >= 2112)
            occupied = steady & (mode > .5)
            onset = np.flatnonzero(positive)[0]
            hit, c3hit = bool(np.any(a & positive)), bool(np.any(c & positive))
            hits += hit
            c3hits += c3hit
            predicted += len(spans(a))
            c3predicted += len(spans(c))
            sy = np.r_[np.zeros(len(ns), np.int8), y]
            ss = np.r_[ns, s]
            rows.append(dict(method=method, run=run, threshold=threshold,
                ap=average_precision_score(sy, ss), roc=roc_auc_score(sy, ss),
                any_hit=int(hit), c3_hit=int(c3hit),
                first_delay_minutes=(int(np.flatnonzero(a & positive)[0])-onset)*5 if hit else None,
                c3_delay_minutes=(int(np.flatnonzero(c & positive)[0])-onset)*5 if c3hit else None,
                abnormal_n=int(positive.sum()), strict_coverage=float(a[positive].mean()),
                c3_coverage=float(c[positive].mean()), steady_n=int(steady.sum()),
                steady_coverage=float(a[steady].mean()), steady_c3_coverage=float(c[steady].mean()),
                occupied_steady_n=int(occupied.sum()), occupied_steady_coverage=float(a[occupied].mean()),
                occupied_steady_c3_coverage=float(c[occupied].mean()),
                longest_miss_minutes=max_gap(a, positive)*5,
                longest_c3_miss_minutes=max_gap(c, positive)*5,
                alarm_segments=len(spans(a)), c3_segments=len(spans(c)),
                prefault_n=int((~positive).sum()), prefault_false_points=int(a[~positive].sum())))
            all_scores.append(s)
            all_labels.append(y)
        def event_row(hits_, pred_):
            precision = hits_/pred_ if pred_ else 0.
            recall = hits_/3
            f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
            return precision, recall, f1
        p, r, f = event_row(hits, predicted)
        cp, cr, cf = event_row(c3hits, c3predicted)
        pooled_rows.append(dict(method=method,
            pooled_ap=average_precision_score(np.concatenate(all_labels), np.concatenate(all_scores)),
            pooled_roc=roc_auc_score(np.concatenate(all_labels), np.concatenate(all_scores)),
            any_hit=hits, c3_hit=c3hits, true_events=3, predicted_events=predicted,
            event_precision=p, event_recall=r, event_f1=f,
            c3_predicted_events=c3predicted, c3_event_precision=cp, c3_event_recall=cr, c3_event_f1=cf))
    pd.DataFrame(rows).to_csv(output / 'per_scenario.csv', index=False)
    pd.DataFrame(normal_rows).to_csv(output / 'normal_false_alarms.csv', index=False)
    pd.DataFrame(pooled_rows).to_csv(output / 'pooled_events.csv', index=False)
    (output / 'protocol.json').write_text(json.dumps(dict(
        thresholds='CAL only q.99 higher; strict >; no QUERY tuning',
        point_ap='per scenario includes the single matched normal baseline; pooled includes baseline once',
        event_precision='one-to-one per independent run axis; fragmented alarms are unmatched predictions',
        c3='causal third and subsequent consecutive hits; never retrospective backfill',
        status='simulation mechanism only; correlated scenarios, one normal day CAL; no real confirmation'),
        indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--banks', type=Path, required=True, help='JSON map method to directory')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    evaluate(args.data, {k: Path(v) for k, v in json.loads(args.banks.read_text()).items()}, args.output)
