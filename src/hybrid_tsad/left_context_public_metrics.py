"""Five-benchmark panels for matched LEFT and archived external readers."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import native_event_space as nasa
from . import left_context_unmodulated as archived
from .left_context_unmodulated_repeat import completed_run, previous_pair
from .left_context_unmodulated import ranking_check
from .left_decoder_context_metrics import AUTHOR, AXES, METRICS, truth as archived_truth
from .left_decoder_context_run import PUBLIC, PUBLIC_ARMS, REPEAT
from .left_nasa_author import OUT as NASA_RUN, raw_paths
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_smd28_author import RUN as SMD_RUN, checked_data
from .left_swat_author import dump, read, sha
from .native_event_smd28 import records
from .native_rank_psm_extension import EXTERNAL as PSM_REFS
from .native_full_window_external import timercd
from .native_full_window_metrics import OUT as RADAR_BANKS, checked_bank, truth

DOMAINS = {'SMD': (28, 192, 38), 'MSL': (27, 96, 1), 'SMAP': (53, 192, 1)}
NATIVE = 'Native LEFT'
CANDIDATE = 'Observation-assisted LEFT (modulation off at detection)'
LEVEL = 'Pointwise level deviation'
NAMES = dict(zip(PUBLIC_ARMS, (NATIVE, CANDIDATE)))
LEFT_PERMISSIONS = 'Normal target FIT and SELECT, normal-prefix scaler includes SELECT, full-window offline'
LEVEL_PERMISSIONS = 'Zero training, identical LEFT normal-prefix scaler, current point only'
RADAR_PERMISSIONS = 'Frozen official pretrained TimeRadar, archived target FIT statistics, W100 B1 full-window offline project readout'
NASA_LABEL_SHA = '057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539'
SMAP_REFS = {'History Mean512': ('history_mean512', 'Strictly past 512 points, archived per-entity target FIT scaler'),
             'Persistence': ('persistence', 'Previous point, archived per-entity target FIT scaler'),
             'RoCA FULL-T': ('RoCA_FULL_T', 'Archived target FIT and CAL, causal endpoint project readout')}


def plan(w):
    smd = {r['id']: r for r in records(w)}
    return [(w, 'SMD', e, smd[e]) for e in sorted(smd)] + [
        (w, d, e, None) for d in ('MSL', 'SMAP') for e in nasa.entity_list(w, d)]


def locations(w, domain, entity, arm):
    root = w/PUBLIC/domain
    return root/entity/arm if domain == 'SMD' else root/arm


def run_receipt(w, domain, entity, arm):
    folder = locations(w, domain, entity, arm)
    train, scored = (read(folder/f'{name}.json') for name in ('TRAIN', 'SCORES'))
    ckpt, = (folder/'model').rglob('checkpoint.pth')
    assert sha(ckpt) == train['checkpoint_sha256']
    assert sha(folder/'scaler.npz') == train['scaler_sha256']
    assert train['seed'] == 2025 and train['arm'] == arm
    vals = train['loss_history']
    best = min(range(len(vals)), key=lambda i: (vals[i], -i))
    assert train['selected']['epoch'] == best+1
    assert train['selected']['val'] == vals[best]
    assert len(train['sampler_cpu_rng_sha256']) == len(vals) == train['epochs']
    assert train['updates'] == train['final_step']
    assert not scored.get('test_labels_read', scored.get('labels_read'))
    assert not train.get('test_read_in_training', train.get('test_values_or_labels_in_training'))
    assert scored['state_unchanged'] and scored['detection_modulation'] == ('none' if arm == 'native' else 'off')
    assert scored['native_lambda'] == (0.2 if domain == 'SMD' else 0.3)
    assert train['select_modulation'] == ('none' if arm == 'native' else 'on')
    return train, scored


def read_left(w, domain, entity, arm, score_receipt, count):
    folder = locations(w, domain, entity, arm)
    if domain == 'SMD':
        path = folder/'scores.npz'
        bank, source = score_receipt, score_receipt['input_sha256']
    else:
        path = folder/f'{entity}.npz'
        bank, = (b for b in score_receipt['banks'] if b['entity'] == entity)
        source = bank['input_sha256']
    assert sha(path) == bank['bank_sha256']
    _, window, channels = DOMAINS[domain]
    stop = (count//window)*window
    ix = np.arange(stop)
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['index'], ix)
        np.testing.assert_array_equal(z['available_index'], (ix//window+1)*window-1)
        score = z['score']
        assert score.shape == (stop,) and np.isfinite(score).all()
        if domain == 'SMD':
            assert z['per_channel'].shape == (stop, channels)
            np.testing.assert_array_equal(z['per_channel'].mean(axis=-1), score)
    return score, bank['bank_sha256'], source


def level_on_raw(w, domain, entity, raw, stop):
    scaler = locations(w, domain, entity, 'native')/'scaler.npz'
    with np.load(scaler, allow_pickle=False) as stats:
        x = ((raw[:stop]-stats['mean'])/stats['scale']).astype(np.float32)
    absolute = np.abs(x)
    score = np.where(absolute < 1, 0.5*x*x, absolute-0.5).mean(axis=1)
    assert np.isfinite(score).all()
    out = w/PUBLIC/'level'/domain
    out.mkdir(parents=True, exist_ok=True)
    path = out/f'{entity}.npz'
    if path.exists():
        with np.load(path, allow_pickle=False) as bank:
            np.testing.assert_array_equal(score, bank['score'])
            np.testing.assert_array_equal(np.arange(stop), bank['index'])
    else:
        np.savez_compressed(path, score=score, index=np.arange(stop), available_index=np.arange(stop))
    return score, sha(path), LEVEL_PERMISSIONS


def external(w, domain, entity, raw, record, labels, score_stop):
    path = w/RADAR_BANKS/domain/f'{entity}.npz'
    bank, info = checked_bank(path)
    data_id = info['input_binding']['input_sha256']
    if domain == 'SMD':
        role, = (r for r in record['roles'] if r['role'] == 'QUERY')
        assert data_id == role['sha256'] and role['bounds_half_open'] == [0, len(raw)]
    elif domain == 'MSL':
        assert data_id == sha(raw_paths(w, domain, 'test', entity))
    else:
        source = w/nasa.SMAP_QUERY/f'{entity}.npz'
        assert data_id == sha(source)
        np.testing.assert_array_equal(raw[:,0].astype(np.float32), nasa.load_npz(source).astype(np.float32))
    assert info['query_points'] == len(raw) and not info['labels_read']
    with bank as z:
        full_stop = len(raw)//100*100
        np.testing.assert_array_equal(z['index'], np.arange(full_stop))
        np.testing.assert_array_equal(z['available_index'], (z['index']//100+1)*100-1)
        assert z['keys'][0] == 'frozen_native_V'
        lo, hi = (512 if domain == 'SMAP' else 0), min(score_stop,full_stop)
        values = {'TimeRadar official V (project W100)': (z['V'][lo:hi,0], info['sha256'], RADAR_PERMISSIONS)}
    if domain == 'SMD':
        for name, score, permission, digest in timercd(w, domain, entity, np.arange(lo,hi)):
            values[name] = score, digest, permission
    if domain == 'SMAP':
        for name, (method, permission) in SMAP_REFS.items():
            source = ('roca_public_unified_evaluation_20261106_v1/entities'
                      if method.startswith('RoCA') else 'hybrid_v1_public_native_zs_20261105_v1/evaluation/entities')
            path = w/'results'/source/domain/entity/method/'calculation_inputs.npz'
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['index'], np.arange(lo,len(labels)))
                np.testing.assert_array_equal(z['available_index'], z['index'])
                np.testing.assert_array_equal(z['full_labels'], labels)
                assert z['valid'].all()
                values[name] = z['score'][:hi-lo].reshape(-1), sha(path), permission
    return lo, hi, values


def evaluate_entity(job):
    w, domain, entity, record = job
    label = truth(w, domain, entity)
    if domain == 'SMD':
        raw = checked_data(w, record, 'test')
        input_id = record['source'][1]['sha256']
    else:
        path = raw_paths(w, domain, 'test', entity)
        raw = np.load(path, allow_pickle=False)[:,0].astype(np.float64)[:,None]
        input_id = sha(path)
    assert len(raw) == len(label) and np.isin(label, [0, 1]).all()
    data = {}
    for arm in PUBLIC_ARMS:
        root = locations(w, domain, entity, arm)
        receipt = read(root/'SCORES.json')
        score, digest, source = read_left(w, domain, entity, arm, receipt, len(label))
        assert source == input_id
        data[NAMES[arm]] = score, digest, LEFT_PERMISSIONS
    stop = len(data[NATIVE][0])
    data[LEVEL] = level_on_raw(w, domain, entity, raw, stop)
    lo, hi, references = external(w, domain, entity, raw, record, label, stop)
    panels = [('internal', 0, stop, data),
              ('external_common', lo, hi,
               {**{k:(s[lo:hi],h,p) for k,(s,h,p) in data.items()}, **references})]
    metric = load_author_metric(w/AUTHOR)
    rows, err, checked = [], 0.0, 0
    for panel, first, end, scores in panels:
        y = label[first:end]
        has_both_classes = np.unique(y).size == 2
        for method, (s, identity, permission) in scores.items():
            assert s.shape == y.shape and np.isfinite(s).all()
            if has_both_classes:
                result,_ = continuous_metrics(y, s, metric)
                result = {k:result[k] for k in METRICS}
                assert all(np.isfinite(v) for v in result.values())
                ap, roc = ranking_check(y,s)
                err = max(err,abs(ap-result['AP']),abs(roc-result['ROC_AUC']))
                checked += 1
            else:
                result = dict.fromkeys(METRICS, None)
            rows.append(dict(domain=domain, entity=entity, panel=panel, method=method,
                 start=first, stop=end, points=end-first, positives=int(y.sum()),
                 tail_points=len(label)-end, cache_sha256=identity, permissions=permission, **result))
    assert err < 1e-12
    return rows, checked, err


def aggregate(rows):
    data = pd.DataFrame(rows)
    by = ['domain','panel','method','permissions']
    grouped = data.groupby(by, sort=False)
    macro = grouped[list(METRICS)].mean().reset_index()
    macro['entities'] = grouped['entity'].size().to_numpy()
    macro['points'] = grouped['points'].sum().to_numpy()
    macro['eligible_entities'] = grouped['AP'].count().to_numpy()
    return macro


def pair_distributions(rows):
    data = pd.DataFrame(rows).query("panel=='internal'")
    detailed, summary = [], []
    for domain in DOMAINS:
        block = data[data.domain==domain].set_index(['method','entity'])
        for reference in (NATIVE, LEVEL):
            ref = block.loc[reference]
            candidate = block.loc[CANDIDATE]
            assert ref.index.equals(candidate.index)
            diff = candidate[list(METRICS)]-ref[list(METRICS)]
            for entity, row in diff.iterrows():
                detailed.append(dict(domain=domain, entity=entity, reference=reference,
                      start=int(ref.loc[entity,'start']), stop=int(ref.loc[entity,'stop']), **row.to_dict()))
            for key in METRICS:
                values = diff[key].dropna()
                summary.append(dict(domain=domain, reference=reference, metric=key,
                    mean_delta=values.mean(), median_delta=values.median(), min_delta=values.min(),
                    max_delta=values.max(), wins=int((values>0).sum()), ties=int((values==0).sum()),
                    losses=int((values<0).sum()), eligible=len(values), entities=len(diff)))
    return detailed, summary


def prior_rows(w):
    previous = read(w/REPEAT/'METRICS.json')
    rows = []
    for domain, panel in previous['domains'].items():
        assert panel['axis'] == list(AXES[domain])
        for key, value in panel['rows'].items():
            base, sep, version = key.partition('_')
            if base in NAMES and sep:
                method, seed, permission = NAMES[base], int(version), LEFT_PERMISSIONS
            elif key == 'level':
                method, seed, permission = LEVEL, '', LEVEL_PERMISSIONS
            else:
                method, seed, permission = key, '', value['permission']
            rows.append(dict(domain=domain, panel='archived_full', method=method,
                  permissions=permission, seed=seed, entities=1, eligible_entities=1,
                  points=AXES[domain][1]-AXES[domain][0], **{k:value[k] for k in METRICS}))
    return rows



def psm_history_panel(w):
    previous = read(w/archived.ROOT/'METRICS.json')
    banks, identities, _ = previous_pair(w, 'PSM', previous)
    for arm in PUBLIC_ARMS:
        info, score = completed_run(w,'PSM',arm)
        key = f'{arm}_2026'
        banks[key] = score
        identities[key] = info['score']['bank_sha256']
    lo, hi = 512, AXES['PSM'][1]-AXES['PSM'][0]
    panel = {}
    for arm in PUBLIC_ARMS:
        for seed in (2025,2026):
            key = f'{arm}_{seed}'
            panel[NAMES[arm],seed] = banks[key][lo:hi], identities[key], LEFT_PERMISSIONS
    panel[LEVEL,''] = banks['level'][lo:hi], identities['level'], LEVEL_PERMISSIONS
    path, digest = PSM_REFS['HistoryMean512']
    assert sha(w/path) == digest
    with np.load(w/path,allow_pickle=False) as z:
        assert z['index'][0] == lo and z['index'][-1] >= hi-1
        np.testing.assert_array_equal(z['index'][:hi-lo], np.arange(lo,hi))
        panel['History Mean512',''] = z['score'][:hi-lo].reshape(-1), digest, 'Strictly past 512 points, archived target normal scaler'
    label = archived_truth(w, 'PSM')[lo:hi]
    metric = load_author_metric(w/AUTHOR)
    rows, max_error = [], 0.0
    for (method,seed), (score,identity,permission) in panel.items():
        result,_ = continuous_metrics(label, score, metric)
        ap, roc = ranking_check(label,score)
        max_error = max(max_error,abs(ap-result['AP']),abs(roc-result['ROC_AUC']))
        rows.append(dict(domain='PSM', panel='history_common', method=method,
                  permissions=permission, seed=seed, entities=1, eligible_entities=1,
                  points=len(label), start=lo, stop=hi, cache_sha256=identity,
                  **{k:result[k] for k in METRICS}))
    assert max_error < 1e-12
    return rows, max_error

def run(w):
    start = time.perf_counter()
    root = w/PUBLIC
    assert (root/'ALL_SCORES_READY').is_file()
    assert sha(w/nasa.MSL/'labels.csv') == NASA_LABEL_SHA
    jobs = plan(w)
    assert len(jobs) == 108
    keys = {('SMD',entity) for _,domain,entity,_ in jobs if domain=='SMD'}
    keys.update((domain,domain) for domain in ('MSL','SMAP'))
    receipt = read(root/'QUEUE_RESULTS.json')
    assert receipt['completed']==60
    assert {(r['domain'],r['entity'],r['arm']) for r in receipt['jobs']} == {
        (d,e,a) for d,e in keys for a in PUBLIC_ARMS}
    costs = []
    for domain,entity in sorted(keys):
        run_data = [run_receipt(w, domain, entity, a) for a in PUBLIC_ARMS]
        (a, _), (b, _) = run_data
        for key in ('common_initial_state_sha256','scaler_sha256','configs','check_sha256'):
            assert a[key] == b[key]
        overlap = min(a['epochs'],b['epochs'])
        assert a['sampler_cpu_rng_sha256'][:overlap] == b['sampler_cpu_rng_sha256'][:overlap]
        for arm, (t, s) in zip(PUBLIC_ARMS, run_data):
            scores = s.get('scored', 0) if domain=='SMD' else sum(b['scored_points'] for b in s['banks'])
            costs.append(dict(domain=domain, task=entity, arm=arm, device=t['device'],
                  train_seconds=t['train_seconds'], score_seconds=s['score_seconds'], points=scores,
                  epochs=t['epochs'], selected_epoch=t['selected']['epoch'], updates=t['updates'],
                  selected_step=t['selected']['step'], final_step=t['final_step'],
                  common_sampler_epochs=overlap, checkpoint_sha256=t['checkpoint_sha256'],
                  scaler_sha256=t['scaler_sha256'],common_initial_state_sha256=t['common_initial_state_sha256']))
    rows, checked, max_error = [], 0, 0.0
    with ProcessPoolExecutor(max_workers=6) as pool:
        for entity, n, error in pool.map(evaluate_entity,jobs):
            rows.extend(entity)
            checked += n
            max_error = max(max_error,error)
    history, history_error = psm_history_panel(w)
    checked += len(history)
    max_error = max(max_error,history_error)
    macro = aggregate(rows).assign(seed=lambda d:d.method.map({NATIVE:2025,CANDIDATE:2025}))
    details, summary = pair_distributions(rows)
    pd.DataFrame(rows).to_csv(root/'entity_metrics.csv',index=False)
    pd.DataFrame(details).to_csv(root/'entity_deltas.csv',index=False)
    pd.DataFrame(summary).to_csv(root/'paired_distribution.csv',index=False)
    pd.concat([pd.DataFrame(prior_rows(w)+history),macro],ignore_index=True).to_csv(root/'methods.csv',index=False)
    pd.DataFrame(costs).to_csv(root/'costs.csv',index=False)
    dump(root/'EVALUATION.json', dict(complete=True, scored_entities=108, observed_jobs=60,
         VUS=dict(windowSize=100,thre=250), AP_ROC_recomputed_rows=checked, max_AP_ROC_error=max_error,
         train_A40_seconds=sum(x['train_seconds'] for x in costs), score_A40_seconds=sum(x['score_seconds'] for x in costs),
         process_seconds=sum(x['elapsed_seconds'] for x in receipt['jobs']),
         queue_wall_seconds=receipt['wall_seconds'], evaluation_CPU_seconds=time.perf_counter()-start,
         calculator_sha256=sha(__file__), archived_two_domain_evaluation_sha256=sha(w/REPEAT/'METRICS.json'),
         operation_metrics='NA'))
