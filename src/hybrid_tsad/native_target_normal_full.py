"""Readout and matched comparison for the full-parameter target-normal track."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import torch

from . import native_target_normal as train
from . import native_target_normal_readout as prior_readout
from . import native_target_normal_metrics as prior
from . import native_full_window_metrics as axes
from .native_position_aligned_training import values
from .native_v_conditioning import state_hashes, sha
from .native_v_conditioning_runner import read, dump, setup
from .left_native_swat_metrics import continuous_metrics, load_author_metric

OUT = train.FULL_OUT
KEYS = tuple(f'full_{arm}' for arm in train.ARMS)
DECODER_KEYS = ('target_official', 'target_source_copy', 'target_source_all')
INITIAL_NAMES = ('Official initialization', 'Source reconstruction initialization',
                 'All-position source-supervised initialization')
NAMES = {**prior.NAMES, **{k: f'{name} + full-parameter target-normal training'
                          for k, name in zip(KEYS, INITIAL_NAMES)}}
NAMES.update({k: f'{name} + decoder-only target-normal training'
              for k, name in zip(DECODER_KEYS, INITIAL_NAMES)})


def selected_models(w, domain, device):
    root = w / OUT
    seal = read(root / 'ALL_MODELS_FROZEN.json')
    assert seal['status'] == 'COMPLETED_BEFORE_TEST'
    models, weights = [], []
    for arm in train.ARMS:
        receipt = seal['domains'][domain][arm]
        path = root / receipt['checkpoint_relative']
        assert sha(path) == receipt['checkpoint_sha256']
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert state['epoch'] == receipt['selected_epoch']
        model, _ = train.load_initial(w, arm, device)
        model.load_state_dict(state['model'], strict=True)
        models.append(model.eval().requires_grad_(False))
        weights.append(receipt['checkpoint_sha256'])
    return models, weights


@torch.inference_mode()
def score(w, lane, lanes):
    device = setup()
    start = time.monotonic()
    jobs = prior.jobs(w)
    root = w / OUT / 'readout'
    root.mkdir(exist_ok=True, parents=True)
    loaded = {}
    for domain, entity in jobs[lane::lanes]:
        if domain not in loaded:
            loaded[domain] = selected_models(w, domain, device)
        models, weights = loaded[domain]
        folder = root / domain
        folder.mkdir(exist_ok=True, parents=True)
        bank = folder / f'{entity}.npz'
        receipt = bank.with_suffix('.json')
        if receipt.exists():
            cached = read(receipt)
            assert cached['sha256'] == sha(bank) and cached['weights_sha256'] == weights
            continue
        tick = time.monotonic()
        old = read(w / train.OUT / 'readout' / domain / f'{entity}.json')
        windows, binding = prior_readout.windows(w, domain, entity)
        assert binding == old['input_binding']
        before = [state_hashes(m) for m in models]
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        scores = np.empty((len(windows)*100, len(KEYS)), dtype=np.float32)
        for i, raw in enumerate(windows):
            x = torch.from_numpy(np.ascontiguousarray(raw)).to(device)[None]
            # The order, embeddings and encoder are now model-specific.
            for j, model in enumerate(models):
                v, _ = values(model, x)
                scores[i*100:(i+1)*100, j] = v.cpu().numpy()
        assert np.isfinite(scores).all()
        assert before == [state_hashes(m) for m in models]
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state(device))
        index = np.arange(len(scores))
        window_start = index//100*100
        tmp = bank.with_suffix('.part.npz')
        np.savez_compressed(tmp, keys=KEYS, V=scores, index=index, window_start=window_start,
                            window_end_exclusive=window_start+100, available_index=window_start+99)
        tmp.replace(bank)
        dump(receipt, dict(sha256=sha(bank), keys=KEYS, input_binding=binding, weights_sha256=weights,
                          native_B=1, copies=10, mask_seed=84551, correction=1, eval=True,
                          frozen_verified=True, RNG_restored=True, labels_read=False,
                          seconds=time.monotonic()-tick))
        print(domain, entity, len(scores), flush=True)
    dump(root/f'LANE_{lane}.json', dict(device=os.environ.get('CUDA_VISIBLE_DEVICES'),
         seconds=time.monotonic()-start, peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def entity_metrics(args):
    w, domain, entity = args
    start = time.monotonic()
    root = w / OUT
    bank = root/'readout'/domain/f'{entity}.npz'
    receipt = read(bank.with_suffix('.json'))
    assert sha(bank) == receipt['sha256']
    base_path = w/train.OUT/'metrics_parts'/domain/f'{entity}.json'
    base = read(base_path)
    common = next(row for row in base['rows'] if row['axis']=='common')
    axis = np.arange(common['first'], common['stop'])
    y = axes.truth(w, domain, entity)
    with np.load(bank) as data:
        np.testing.assert_array_equal(data['keys'], KEYS)
        index = data['index']
        scores = data['V']
        np.testing.assert_array_equal(index, np.arange(len(y)//100*100))
        np.testing.assert_array_equal(data['available_index'], index//100*100+99)
    np.testing.assert_array_equal(index[axis], axis)
    metric = load_author_metric(w/axes.AUTHOR)
    rows = list(base['rows'])
    for key, score in zip(KEYS, scores.T):
        for name, indices in [('common', axis), ('native_full', index)]:
            result, _ = continuous_metrics(y[indices], score[indices].astype(float), metric)
            rows.append(dict(domain=domain, entity=entity, method=key, axis=name,
                             first=int(indices[0]), stop=int(indices[-1]+1), points=len(indices),
                             **{k: result[k] for k in prior.METRICS}))
    result = dict(rows=rows, bank_sha256=receipt['sha256'], prior_metrics_sha256=sha(base_path),
                  worker_seconds=time.monotonic()-start)
    folder = root/'metrics_parts'/domain
    folder.mkdir(parents=True, exist_ok=True)
    dump(folder/f'{entity}.json', result)
    print(domain, entity, flush=True)
    return result


def summarize(w, results):
    root = w / OUT
    frame = pd.DataFrame(row for rec in results for row in rec['rows'])
    metrics = prior.METRICS
    frame.to_csv(root/'entity_metrics.csv', index=False)
    macro = frame.groupby(['domain', 'axis', 'method'], sort=False)[metrics].mean().reset_index()
    macro.to_csv(root/'macro.csv', index=False)
    pairs, distributions, effects = [], [], []
    for domain in train.DOMAINS:
        common = frame[(frame.domain == domain) & (frame.axis == 'common')]
        refs = list(zip(KEYS, DECODER_KEYS)) + [(KEYS[2], KEYS[1]), (KEYS[2], KEYS[0])]
        refs += [(key, ref) for key in (*KEYS, *DECODER_KEYS)
                 for ref in (('LEFT', 'PCA residual') if domain == 'PSM' else ('LEFT',))]
        for key, ref in refs:
            a = common[common.method == key].set_index('entity')[metrics]
            b = common[common.method == ref].set_index('entity')[metrics]
            assert a.index.is_unique and set(a.index) == set(b.index)
            delta = a-b
            tags = dict(domain=domain, method=key, reference=ref)
            pairs.extend(dict(**tags, entity=e, **r.to_dict()) for e, r in delta.iterrows())
            effects.append(dict(**tags, **delta.mean().to_dict()))
            distributions.extend(dict(**tags, metric=k, median_delta=v.median(),
                 wins=int((v > 0).sum()), ties=int((v == 0).sum()), losses=int((v < 0).sum()))
                 for k, v in delta.items())
    pd.DataFrame(pairs).to_csv(root/'paired_entity.csv', index=False)
    pd.DataFrame(effects).to_csv(root/'paired_macro.csv', index=False)
    pd.DataFrame(distributions).to_csv(root/'paired_distribution.csv', index=False)
    main = macro[macro.axis == 'common']
    display = lambda key: NAMES.get(key, key)
    (root/'TABLES.md').write_text('\n\n'.join([
        '# Public common-axis comparison',
        prior.table(['Task', 'Method', *metrics],
            [[r.domain, display(r.method), *[f'{getattr(r,k):.6f}' for k in metrics]]
             for r in main.itertuples()]),
        '# Training scope and initialization contrasts',
        prior.table(['Task', 'Method', 'Reference', *metrics],
            [[r['domain'], display(r['method']), display(r['reference']),
              *[f'{r[k]:+.6f}' for k in metrics]] for r in effects]),
    ])+'\n', encoding='utf8')


def evaluate(w):
    start = time.monotonic()
    root = w / OUT
    jobs = prior.jobs(w)
    # Evaluation starts only once each sealed score bank is present.
    while not all((root/'readout'/d/f'{e}.json').exists() for d,e in jobs):
        time.sleep(5)
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(entity_metrics, [(w,d,e) for d,e in jobs]))
    summarize(w, results)
    dump(root/'METRIC_COST.json', dict(objects=len(results), elapsed_seconds=time.monotonic()-start,
         worker_seconds=sum(r['worker_seconds'] for r in results), GPU_used=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('score', 'evaluate'))
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--lanes', type=int, default=3)
    args = parser.parse_args()
    if args.action == 'score':
        score(Path.cwd(), args.lane, args.lanes)
    else:
        evaluate(Path.cwd())
