"""Evaluate sealed LEFT score banks and compatible cached public comparators."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np

from .left_decoder_context import ARMS
from .left_decoder_context_run import DOMAINS, OUT, SWAT_PREP
from .left_native_swat_metrics import LABEL_SHA, continuous_metrics, load_author_metric
from .left_swat_author import dump, read, sha
from .native_event_rank_readout import LABEL_SHA as PSM_LABEL_SHA

AXES = {'PSM': (132481, 220225), 'SWAT': (495000, 944856)}
NAMES = {'native': 'Native LEFT', 'observation': 'Observation-conditioned LEFT',
         'prototype': 'Prototype-conditioned LEFT'}
METRICS = ('AP','ROC_AUC','VUS_PR','VUS_ROC')
MANIFEST = Path('preparations/native_supervised_mlp_20261123_v1/PSM_SWAT_COMMON_AXIS.json')
AUTHOR = Path('preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py')


def truth(w, domain):
    lo, hi = AXES[domain]
    if domain == 'PSM':
        path = w/'data/common_psm_20261028_v1/test_label.npy'
        assert sha(path) == PSM_LABEL_SHA
        return np.load(path,allow_pickle=False).reshape(-1)[:hi-lo]
    path = w/SWAT_PREP/'SWAT_LABEL_METADATA.npz'
    assert sha(path) == LABEL_SHA
    with np.load(path,allow_pickle=False) as data:
        return data['label'][lo:hi]


def one(args):
    w, domain, arm = args
    out = w/OUT/domain/arm
    receipt = read(out/'SCORE_RECEIPT.json')
    train = read(out/'TRAIN_RECEIPT.json')
    bank = out/'raw_scores.npz'
    assert sha(bank) == receipt['bank_sha256']
    assert train['selected']['sha256'] == receipt['checkpoint_sha256']
    if (out/'METRICS.json').exists():
        record = read(out/'METRICS.json')
        assert record['score_sha256'] == receipt['bank_sha256']
        return domain, record
    lo, hi = AXES[domain]
    with np.load(bank,allow_pickle=False) as z:
        index = np.arange(lo,hi)
        np.testing.assert_array_equal(z['absolute_time_index'],index)
        np.testing.assert_array_equal(z['available_index'],lo+((index-lo)//192+1)*192-1)
        score = z['canonical_score']
        np.testing.assert_array_equal(z['per_channel'].mean(axis=-1),score)
    start = time.monotonic()
    result, _ = continuous_metrics(truth(w,domain),score,load_author_metric(w/AUTHOR))
    record = dict(method=arm, **result, score_sha256=receipt['bank_sha256'],
                  validation_loss=train['selected']['loss'], selected_epoch=train['selected']['epoch'],
                  updates=train['updates'], **train['parameters'],
                  train_seconds=train['train_seconds'], scoring_seconds=receipt['scoring_seconds'],
                  metric_seconds=time.monotonic()-start,
                  permission='target normal FIT / normal SELECT; prefix scaler; W192 offline')
    dump(out/'METRICS.json', record)
    print(domain,arm,flush=True)
    return domain, record


def external(w, domain):
    panel = next(p for p in read(w/MANIFEST)['panels']
                 if p['dataset']==domain and p['track']=='offline-readout')
    lo,hi = AXES[domain]
    axis = np.arange(lo,hi) - (lo if domain=='PSM' else 0)
    assert panel['axis']==[int(axis[0]), int(axis[-1]+1)]
    for row in panel['rows']:
        if row['method']=='LEFT native W192':
            continue
        path = w/row['path']
        assert sha(path)==row['sha256']
        with np.load(path,allow_pickle=False) as z:
            idx = z[row['index_key']].reshape(-1)-row['index_offset']
            indices = np.searchsorted(idx,axis)
            np.testing.assert_array_equal(idx[indices],axis)
            assert np.isfinite(z[row['score_key']].reshape(-1)[indices]).all()
        seed = row.get('archived_seed')
        name = row['method'] + (f' seed{seed}' if seed is not None else '')
        yield dict(method=name, **{k:row[k] for k in METRICS}, permission=row['permission'],
                   score_sha256=row['sha256'], reused=True)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])


def main():
    w = Path.cwd()
    root = w/OUT
    assert (root/'ALL_SCORES_READY').exists()
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, ((w,d,a) for d in DOMAINS for a in ARMS)))
    domains, main_rows, deltas, params = {}, [], [], []
    for domain in DOMAINS:
        rows = [row for d,row in results if d==domain]
        diff = [dict(reference=row['method'], **{k:rows[2][k]-row[k] for k in METRICS}) for row in rows[:2]]
        configs = [read(root/domain/a/'CONFIG.json') for a in ARMS]
        for key in ('common_initial_state_sha256','normal_values_sha256','scaler_sha256','configuration'):
            assert all(c[key]==configs[0][key] for c in configs)
        traces = [read(root/domain/a/'TRAIN_RECEIPT.json')['validation'] for a in ARMS]
        for group in zip(*traces):
            assert len({r['cpu_rng_sha256'] for r in group})==1
        for row in rows:
            params.append([domain,NAMES[row['method']],row['registered'],row['updated'],row['active_modulation'],
                           row['selected_epoch'],row['updates'],f"{row['train_seconds']:.3f}",f"{row['scoring_seconds']:.3f}"])
        rows.extend(external(w,domain))
        if domain=='SWAT':
            rows.extend(dict(method=name, permission='No compatible full-axis cache', **dict.fromkeys(METRICS))
                        for name in ('CATCH W192','PCA residual'))
        domains[domain] = dict(rows=rows,deltas=diff,absolute_axis=AXES[domain])
        for row in rows:
            main_rows.append([domain,NAMES.get(row['method'],row['method']),
                *[f'{row[k]:.6f}' if row[k] is not None else 'NA' for k in METRICS],row['permission'],
                f"{row['train_seconds']:.2f} / {row['scoring_seconds']:.2f}" if 'train_seconds' in row else 'NA'])
        deltas.extend([domain,'Prototype-conditioned minus '+NAMES[r['reference']],
                       *[f'{r[k]:+.6f}' for k in METRICS]] for r in diff)
    (root/'TABLES.md').write_text('\n\n'.join([
        '### Complete public benchmark on common scored points',
        table(['Task','Complete method',*METRICS,'Training and context permissions','Train / readout seconds'],main_rows),
        '### Matched structural ablation',
        table(['Task','Contrast',*METRICS],deltas),
        '### Parameters and observed cost',
        table(['Task','Method','Registered parameters','Adam state parameters','Active modulation parameters',
               'Selected epoch','Updates','Train seconds','Readout seconds'],params)])+'\n',encoding='utf8')
    dump(root/'METRICS.json',dict(domains=domains, metric_seconds=time.monotonic()-start,
        operations='NA: independent full-axis calibration not established',
        metric_definition='sklearn AP and ROC; LEFT VUS windowSize=100 thre=250'))


if __name__ == '__main__':
    main()
