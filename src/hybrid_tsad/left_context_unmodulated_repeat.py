"""Evaluate the fixed two-arm LEFT repetition on the original two public axes."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np

from . import left_context_unmodulated as previous
from . import left_decoder_context_metrics as prior
from .left_decoder_context_run import OUT, REPEAT
from .left_swat_author import dump, read, sha

ARMS = ('native', 'observation')
DOMAINS = ('PSM', 'SWAT')
PAIR_KEYS = tuple(f'{arm}_{seed}' for seed in (2025, 2026) for arm in ARMS)
LABELS = {
    **{f'native_{s}': f'Native LEFT, seed{s}' for s in (2025, 2026)},
    **{f'observation_{s}': f'Observation-assisted LEFT, detection off, seed{s}' for s in (2025, 2026)},
    'level': 'Pointwise level deviation',
}


def completed_run(w, domain, arm):
    folder = w/REPEAT/domain/arm
    config = read(folder/'CONFIG.json')
    train = read(folder/'TRAIN_RECEIPT.json')
    score = read(folder/'SCORE_RECEIPT.json')
    old = read(w/OUT/domain/arm/'CONFIG.json')
    mode = dict(native='none', observation='off')[arm]
    assert config['seed']==score['seed']==2026
    assert config['detection_modulation']==score['detection_modulation']==mode
    assert config['normal_select_modulation']==dict(native='none', observation='on')[arm]
    assert config['reused_check_sha256']==sha(w/OUT/domain/'CHECKS.json')
    assert config['common_initial_state_sha256']!=old['common_initial_state_sha256']
    assert all(config[k]==v for k,v in old.items() if k not in ('seed','common_initial_state_sha256'))
    assert train['status']==score['status']=='COMPLETED'
    assert train['selected']['sha256']==score['checkpoint_sha256']
    ckpt, = (folder/'model').rglob('checkpoint.pth')
    assert sha(ckpt)==score['checkpoint_sha256']
    assert not any(r['labels_read'] for r in (config,train,score))
    assert not train['test_values_read'] and score['state_unchanged']
    assert score['native_lambda']==config['curriculum']['maximum']>0
    validation = train['validation']
    expected = min(validation,key=lambda r:(r['validation_loss'], -r['epoch']))
    assert train['selected']['epoch']==expected['epoch']
    assert train['selected']['loss']==expected['validation_loss']
    assert all(r['updates']==r['step'] for r in validation)
    path = folder/'raw_scores.npz'
    assert sha(path)==score['bank_sha256']
    lo,hi = prior.AXES[domain]
    index = np.arange(lo,hi)
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['absolute_time_index'],index)
        np.testing.assert_array_equal(z['available_index'],lo+((index-lo)//192+1)*192-1)
        s = z['canonical_score']
        per_channel = z['per_channel']
        assert per_channel.shape==(len(index),config['channels'])
        np.testing.assert_array_equal(s,per_channel.mean(axis=1))
    return dict(config=config, train=train, score=score),s


def previous_pair(w, domain, old_panel):
    panel = old_panel['domains'][domain]
    lo,hi = prior.AXES[domain]
    assert panel['axis']==[lo,hi]
    old_keys = {'native_2025':'native', 'observation_2025':'observation_off'}
    old_banks, hashes = previous.scores(w,domain,keys=tuple(old_keys.values()))
    bank, bindings, rows = {}, {}, {}
    for new,old in old_keys.items():
        assert hashes[old]==panel['score_bindings'][old]
        bank[new] = old_banks[old]
        bindings[new] = hashes[old]
        rows[new] = panel['rows'][old]
    path = w/previous.ROOT/domain/'level.npz'
    assert sha(path)==panel['score_bindings']['level']
    with np.load(path,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['absolute_time_index'],np.arange(lo,hi))
        np.testing.assert_array_equal(z['available_index'],np.arange(lo,hi))
        bank['level'] = z['score']
    bindings['level'] = panel['score_bindings']['level']
    rows['level'] = panel['rows']['level']
    return bank, bindings, rows


def evaluate(w):
    start = time.perf_counter()
    root = w/REPEAT
    assert (root/'ALL_SCORES_READY').is_file()
    old_panel = read(w/previous.ROOT/'METRICS.json')
    runs, domains, tasks, keys = {}, {}, [], []
    for domain in DOMAINS:
        y = prior.truth(w,domain)
        bank, bindings, rows = previous_pair(w,domain,old_panel)
        runs[domain] = {}
        for arm in ARMS:
            receipt,s = completed_run(w,domain,arm)
            key = arm+'_2026'
            bank[key] = s
            bindings[key] = receipt['score']['bank_sha256']
            runs[domain][arm] = receipt
            tasks.append((w,y,s))
            keys.append((domain,key))
        a,b = (runs[domain][arm] for arm in ARMS)
        assert a['config']['common_initial_state_sha256']==b['config']['common_initial_state_sha256']
        overlap = list(zip(a['train']['validation'],b['train']['validation']))
        assert overlap
        for x,y_row in overlap:
            assert x['epoch']==y_row['epoch'] and x['cpu_rng_sha256']==y_row['cpu_rng_sha256']
        ext_rows,ext_banks = previous.archived_scores(w,domain)
        rows.update({row['method']:row for row in ext_rows})
        bank.update(ext_banks)
        domains[domain] = dict(y=y, rows=rows, banks=bank, bindings=bindings,
                               common_sampler_epochs=len(overlap))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for (domain,key), metrics in zip(keys,pool.map(previous.metric,tasks)):
            domains[domain]['rows'][key] = metrics
    final = {}
    for domain, d in domains.items():
        rows,bank,y = d['rows'], d['banks'], d['y']
        order = (*PAIR_KEYS,'level',*(k for k in rows if k not in LABELS))
        error = max(abs(value-rows[key][metric])
                    for key,s in bank.items() for value,metric in
                    zip(previous.ranking_check(y,s),('AP','ROC_AUC')))
        assert error<1e-12
        previous.plot(root,domain,y,{key:bank[key] for key in order},rows,names=LABELS,
                      dashed=('native_2025','observation_2025'))
        final[domain] = dict(rows={key:rows[key] for key in order},axis=prior.AXES[domain],
             score_bindings=d['bindings'], common_sampler_epochs=d['common_sampler_epochs'],
             paired_deltas=[dict(seed=s, **{k: rows[f'observation_{s}'][k]-rows[f'native_{s}'][k]
                                            for k in prior.METRICS}) for s in (2025,2026)],
             AP_ROC_rechecked_rows=len(bank), AP_ROC_max_error=error)
    dump(root/'METRICS.json',dict(complete=True,domains=final,
         new_training_runs=4, new_training_seed=2026,
         cpu_wall_seconds=time.perf_counter()-start, source_sha256=sha(__file__),
         VUS=dict(windowSize=100,thre=250), operations='NA'))


if __name__=='__main__':
    evaluate(Path.cwd())
