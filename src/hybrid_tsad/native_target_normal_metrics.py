"""Fixed-axis evaluation of normal-target decoder training and its initial states."""
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd

from . import native_full_window_metrics as prior
from .native_v_conditioning_runner import read, dump, sha
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .native_event_rank_readout import asset

OUT = Path('results/native_target_normal_20261124_v1')
KEYS = ('init_official', 'init_source_copy', 'init_source_all',
        'target_official', 'target_source_copy', 'target_source_all')
METRICS = list(prior.METRICS)
NAMES = dict(init_official='Official initialization, no target updates',
             init_source_copy='Source reconstruction initialization, no target updates',
             init_source_all='All-position source-supervised initialization, no target updates',
             target_official='Official initialization + target-normal training',
             target_source_copy='Source reconstruction initialization + target-normal training',
             target_source_all='All-position source supervision + target-normal training',
             LEFT='Complete LEFT')
CONTRASTS = (
    ('target_source_all', 'target_source_copy', 'source-supervised minus reconstruction initialization after target training'),
    ('target_source_all', 'target_official', 'source-supervised minus official initialization after target training'),
    ('init_source_all', 'init_source_copy', 'source-supervised minus reconstruction initialization before target training'),
    ('init_source_all', 'init_official', 'source-supervised minus official initialization before target training'),
    ('target_official', 'init_official', 'target training effect, official initialization'),
    ('target_source_copy', 'init_source_copy', 'target training effect, source reconstruction initialization'),
    ('target_source_all', 'init_source_all', 'target training effect, source-supervised initialization'),
    ('target_source_all', 'LEFT', 'complete source-supervised target-trained detector minus LEFT'),
)


def jobs(w):
    return [(d, e) for d, e in prior.jobs(w) if d in ('PSM', 'MSL')]


def psm_external(w, axis):
    binding = read(w/'preparations/native_supervised_mlp_20261123_v1/PSM_SWAT_COMMON_AXIS.json')
    panel = next(p for p in binding['panels'] if p['dataset'] == 'PSM' and p['track'] == 'offline-readout')
    for row in panel['rows']:
        if row['method'] not in ('CATCH W192', 'PCA residual'):
            continue
        path = asset(w, row['path'], row['sha256'])
        with np.load(path) as z:
            ix = z[row['index_key']].reshape(-1)-row['index_offset']
            take = np.searchsorted(ix, axis)
            np.testing.assert_array_equal(ix[take], axis)
            score = z[row['score_key']].reshape(-1)[take]
        name = (f"CATCH W192 seed{row['archived_seed']}"
                if row['method'] == 'CATCH W192' else row['method'])
        yield name, score, row


def entity(args):
    w, d, e = args
    start = time.monotonic()
    p = w/OUT/'readout'/d/(e+'.npz')
    rec = read(p.with_suffix('.json'))
    assert sha(p) == rec['sha256']
    dest = w/OUT/'metrics_parts'/d/(e+'.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        old = read(dest)
        assert old['bank_sha256'] == rec['sha256']
        return old
    old_path = w/prior.OUT/'metrics_parts'/d/(e+'.json')
    archived = read(old_path)
    ref = next(r for r in archived['rows'] if r['axis'] == 'common' and r['method'] == 'LEFT')
    axis = np.arange(ref['first'], ref['stop'])
    assert len(axis) == ref['points']
    y = prior.truth(w, d, e)
    with np.load(p) as z:
        np.testing.assert_array_equal(z['keys'], KEYS)
        ix, values = z['index'], z['V']
        np.testing.assert_array_equal(ix, np.arange(len(y)//100*100))
        np.testing.assert_array_equal(z['available_index'], (ix//100+1)*100-1)
        np.testing.assert_array_equal(z['window_start'], ix//100*100)
        np.testing.assert_array_equal(z['window_end_exclusive'], (ix//100+1)*100)
        assert values.shape == (len(ix), 6) and np.isfinite(values).all()
    take = np.searchsorted(ix, axis)
    np.testing.assert_array_equal(ix[take], axis)
    metric = load_author_metric(w/prior.AUTHOR)
    rows = []
    for j, key in enumerate(KEYS):
        for name, indices, score in [('common', axis, values[take, j]),
                                     ('native_full', ix, values[:, j])]:
            m, _ = continuous_metrics(y[indices], score.astype(float), metric)
            rows.append(dict(domain=d, entity=e, method=key, axis=name,
                             first=int(indices[0]), stop=int(indices[-1]+1), points=len(indices),
                             **{k: m[k] for k in METRICS}))
    rows.extend({k: v for k, v in r.items() if k != 'readout'} for r in archived['rows']
                if r['method'] == 'LEFT' and r['axis'] == 'common')
    bindings = []
    if d == 'PSM':
        for name, score, row in psm_external(w, axis):
            m, _ = continuous_metrics(y[axis], score.astype(float), metric)
            rows.append(dict(domain=d, entity=e, method=name, axis='common',
                             first=int(axis[0]), stop=int(axis[-1]+1), points=len(axis),
                             **{k: m[k] for k in METRICS}))
            bindings.append(row)
    result = dict(domain=d, entity=e, bank_sha256=rec['sha256'],
                  previous_axis_and_LEFT_sha256=sha(old_path),
                  external_bindings=bindings, rows=rows,
                  worker_wall_seconds=time.monotonic()-start)
    dump(dest, result)
    return result


def table(head, rows):
    return '\n'.join(['| '+' | '.join(head)+' |', '|'+'|'.join(['---']*len(head))+'|',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def summarize(w, results):
    root = w/OUT
    frame = pd.DataFrame([r for result in results for r in result['rows']])
    frame.to_csv(root/'entity_metrics.csv', index=False)
    macro = frame.groupby(['domain', 'axis', 'method'], sort=False)[METRICS].mean().reset_index()
    macro.to_csv(root/'macro.csv', index=False)
    common = frame[frame.axis == 'common'].set_index(['domain', 'entity', 'method'])
    differences, distributions, effects = [], [], []
    for d in ('PSM', 'MSL'):
        for method, ref, label in CONTRASTS:
            a = common.loc[d].xs(method, level='method')[METRICS]
            b = common.loc[d].xs(ref, level='method')[METRICS]
            delta = a-b
            assert len(delta) == (1 if d == 'PSM' else 27)
            differences.extend(dict(domain=d, entity=e, method=method, reference=ref,
                                    contrast=label, **v.to_dict()) for e, v in delta.iterrows())
            effects.append(dict(domain=d, method=method, reference=ref, contrast=label,
                                **delta.mean().to_dict()))
            for k in METRICS:
                v = delta[k]
                distributions.append(dict(domain=d, method=method, reference=ref,
                                          contrast=label, metric=k, macro_delta=v.mean(),
                                          median_delta=v.median(), wins=int((v > 0).sum()),
                                          ties=int((v == 0).sum()), losses=int((v < 0).sum()),
                                          entities=len(v)))
    pd.DataFrame(differences).to_csv(root/'paired_entity.csv', index=False)
    pd.DataFrame(distributions).to_csv(root/'paired_distribution.csv', index=False)
    pd.DataFrame(effects).to_csv(root/'paired_macro.csv', index=False)
    main = macro[macro.axis == 'common']
    text = ['# Target-normal training, seed2025', '',
            table(['Domain', 'Complete method', *METRICS],
                  [[r.domain, NAMES.get(r.method, r.method),
                    *[f'{getattr(r,k):.6f}' for k in METRICS]] for r in main.itertuples()]), '',
            'All six rows per task use the new shared target-normal preprocessing. Each target-trained model uses only normal target windows and normal reconstruction-MSE checkpoint selection, including epoch0. The three initializations retain their different official/source-label histories. This is a single project-training seed, not a three-seed result. LEFT and CATCH retain native offline windows and their own training/scaler permissions. Equal timestamps do not imply equal context or training permissions. No independent-CAL operating metrics are asserted.', '',
            '## Initialization and training effects', '',
            table(['Domain', 'Contrast', *METRICS],
                  [[r['domain'], r['contrast'], *[f'{r[k]:+.6f}' for k in METRICS]] for r in effects]), '',
            'Complete native axes and common axes remain separate in entity_metrics.csv and macro.csv. All27 MSL entities are evaluated separately before equal macro aggregation. Entity wins/ties/losses and median differences are in paired_distribution.csv; they are not independent training repetitions.']
    (root/'TABLES.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    dump(root/'METRIC_BINDINGS.json', dict(objects=len(results),
         entities=[{k: v for k, v in r.items() if k != 'rows'} for r in results],
         aggregation='entity first; equal domain macro; single seed2025',
         VUS='pinned LEFT evaluator windowSize100 thre250',
         operations='NA: no newly certified independent CAL'))


def run(w):
    start = time.monotonic()
    pending, active, completed = set(jobs(w)), {}, []
    assert len(pending) == 28
    with ProcessPoolExecutor(max_workers=6) as pool:
        while pending or active:
            for d, e in sorted(pending):
                if len(active) >= 6:
                    break
                if (w/OUT/'readout'/d/(e+'.json')).exists():
                    active[pool.submit(entity, (w, d, e))] = (d, e)
                    pending.remove((d, e))
            if active:
                ready, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
                for f in ready:
                    d, e = active.pop(f)
                    completed.append(f.result())
                    print(d, e, len(completed), 28, flush=True)
            else:
                time.sleep(5)
    summarize(w, completed)
    dump(w/OUT/'METRIC_COST.json', dict(driver_elapsed_seconds=time.monotonic()-start,
         worker_wall_seconds=sum(r['worker_wall_seconds'] for r in completed),
         workers=6, GPU_used=False))


if __name__ == '__main__':
    argparse.ArgumentParser().parse_args()
    run(Path.cwd())
