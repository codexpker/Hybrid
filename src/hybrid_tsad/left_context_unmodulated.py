"""Two fixed PSM readouts and CPU evaluation of conditioned training."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, roc_curve

from . import left_decoder_context_metrics as prior
from .left_decoder_context import model_class
from .left_decoder_context_run import OUT, PSM, SWAT_PREP, Windows, environment, fingerprint
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_swat_author import dump, read, sha

ROOT = Path('results/left_context_unmodulated_20261124_v1')
SWAT_OFF = Path(str(OUT) + '_swat_modulation_off')
ARMS = ('observation', 'prototype')
FORMAL = ('native', 'observation_off', 'prototype_off', 'level')
NAMES = {
    'native': 'Native LEFT',
    'observation_off': 'Observation-conditioned training, detection off',
    'prototype_off': 'Prototype-conditioned training, detection off',
    'observation_on': 'Observation-conditioned training, detection on',
    'prototype_on': 'Prototype-conditioned training, detection on',
    'level': 'Pointwise level deviation',
}
CONTRASTS = (
    ('observation_off', 'observation_on'),
    ('prototype_off', 'prototype_on'),
    ('observation_off', 'native'),
    ('prototype_off', 'native'),
    ('prototype_off', 'observation_off'),
    ('observation_off', 'level'),
    ('prototype_off', 'level'),
)


def standardized(w, domain, arm):
    lo, _ = prior.AXES[domain]
    old = w/OUT/domain/arm
    if domain == 'PSM':
        raw = np.load(w/PSM/'test.npy', allow_pickle=False)
    else:
        raw = np.load(w/SWAT_PREP/'data/ALL_VALUES.npy', mmap_mode='r', allow_pickle=False)[lo:]
    assert hashlib.sha256(raw.tobytes()).hexdigest() == read(old/'SCORE_RECEIPT.json')['raw_test_values_sha256']
    scaler = old/'SCALER.npz'
    assert sha(scaler) == read(old/'CONFIG.json')['scaler_sha256']
    x = np.array(raw, dtype=np.float64, order='F', copy=True)
    with np.load(scaler, allow_pickle=False) as s:
        x -= s['mean']
        x /= s['scale']
    return x, sha(scaler)


def off_ms(self, tokens, z_t, z_f, m_t, m_f, lam, batch, channels):
    assert tokens.shape[0] == batch*channels
    return self.ms_decoder(tokens, context=None)


@torch.inference_mode()
def score_psm(w):
    official, configs, _ = environment(w, 'PSM')
    cfg = official.Configs(str(configs/'PSM/model_configs_Left.json'))
    lo, hi = prior.AXES['PSM']
    root = w/ROOT/'PSM'
    root.mkdir(parents=True, exist_ok=False)
    for arm in ARMS:
        old = w/OUT/'PSM'/arm
        train = read(old/'TRAIN_RECEIPT.json')
        ckpt, = (old/'model').rglob('checkpoint.pth')
        digest = sha(ckpt)
        assert digest == train['selected']['sha256'] == read(old/'SCORE_RECEIPT.json')['checkpoint_sha256']
        assert sha(configs/'PSM/model_configs_Left.json') == read(old/'CONFIG.json')['configuration']['model_configs_Left.json']
        x, scaler_sha = standardized(w, 'PSM', arm)
        model = model_class(official.Left, arm)(cfg).cuda()
        load = model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=False)
        assert not load.missing_keys and all(k.rsplit('.', 1)[-1] in ('total_ops','total_params')
                                            for k in load.unexpected_keys)
        model.step = train['final_step']
        model.eval()
        before = fingerprint(model.state_dict())
        lam = model._curriculum_lambda()
        assert lam == read(old/'CONFIG.json')['curriculum']['maximum'] > 0
        model._condition_ms = MethodType(off_ms, model)
        loader = DataLoader(Windows(x, 192), batch_size=128, shuffle=False)
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = [model.infer(batch.cuda(),None,None,None)[0].cpu().numpy() for batch,_ in loader]
        torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        assert before == fingerprint(model.state_dict()) and model.step == train['final_step']
        assert all(not m.training for m in model.modules()) and model._curriculum_lambda() == lam
        per_channel = np.concatenate(outputs).reshape(-1,25)
        assert per_channel.shape == (hi-lo,25) and np.isfinite(per_channel).all()
        folder = root/arm
        folder.mkdir()
        bank = folder/'raw_scores.npz'
        index = np.arange(lo,hi)
        np.savez_compressed(bank, per_channel=per_channel, canonical_score=per_channel.mean(axis=1),
                            absolute_time_index=index, available_index=lo+((index-lo)//192+1)*192-1)
        dump(folder/'RECEIPT.json', dict(bank_sha256=sha(bank), checkpoint_sha256=digest,
             scaler_sha256=scaler_sha, native_lambda=lam, state_and_step_unchanged=True,
             points=len(index), full_readout_seconds=seconds, labels_read=False))
        print(arm, seconds, flush=True)


def scores(w, domain, *, keys=('native','observation_on','prototype_on','observation_off','prototype_off')):
    lo, hi = prior.AXES[domain]
    idx = np.arange(lo,hi)
    fingerprints = {}
    bank = {}
    for key in keys:
        arm = key.split('_')[0]
        raw_reference = w/OUT/domain/arm
        ref = read(raw_reference/'SCORE_RECEIPT.json')
        if key.endswith('_off'):
            folder = (w/SWAT_OFF if domain=='SWAT' else w/ROOT/domain)/arm
            receipt = read(folder/'RECEIPT.json')
            assert receipt['checkpoint_sha256'] == ref['checkpoint_sha256']
            assert receipt['scaler_sha256'] == read(raw_reference/'CONFIG.json')['scaler_sha256']
            assert receipt['state_and_step_unchanged']
        else:
            folder, receipt = raw_reference, ref
        path = folder/'raw_scores.npz'
        assert sha(path) == receipt['bank_sha256'] and not receipt['labels_read']
        fingerprints[key] = receipt['bank_sha256']
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['absolute_time_index'],idx)
            np.testing.assert_array_equal(z['available_index'],lo+((idx-lo)//192+1)*192-1)
            value = z['canonical_score']
            np.testing.assert_array_equal(value,z['per_channel'].mean(axis=1))
            bank[key] = value
    return bank, fingerprints


def level(w, domain):
    lo, hi = prior.AXES[domain]
    x, scaler_sha = standardized(w, domain, 'native')
    x = x[:hi-lo].astype(np.float32)
    absolute = np.abs(x)
    elementwise = np.where(absolute<1, .5*x*x, absolute-.5)
    result = elementwise.mean(axis=1)
    # Check the fixed pointwise definition using the independent library kernel.
    check = torch.from_numpy(x[:4096])
    lib = torch.nn.functional.smooth_l1_loss(check, torch.zeros_like(check), reduction='none')
    np.testing.assert_array_equal(lib.numpy(), elementwise[:len(check)])
    path = w/ROOT/domain/'level.npz'
    np.savez_compressed(path, score=result, absolute_time_index=np.arange(lo,hi),
                        available_index=np.arange(lo,hi))
    return result, dict(bank_sha256=sha(path), scaler_sha256=scaler_sha,
                        checked_first_points=len(check))


def ranking_check(y, score):
    order = np.argsort(-score, kind='stable')
    labels, sorted_scores = y[order], score[order]
    ends = np.flatnonzero(np.r_[np.diff(sorted_scores)!=0, True])
    tp = labels.cumsum(dtype=np.int64)[ends]
    fp = ends + 1 - tp
    ap = (np.diff(np.r_[0,tp])*tp/(tp+fp)).sum()/tp[-1]
    auc = ((np.r_[0,tp[:-1]]+tp)*np.diff(np.r_[0,fp])).sum()/(2*tp[-1]*fp[-1])
    return ap, auc


def metric(task):
    w, y, score = task
    result, _ = continuous_metrics(y, score, load_author_metric(w/prior.AUTHOR))
    return {k: result[k] for k in prior.METRICS}


def archived_scores(w, domain):
    manifest = read(w/prior.MANIFEST)
    panel = next(row for row in manifest['panels']
                 if row['dataset']==domain and row['track']=='offline-readout')
    axis = np.arange(*panel['axis'])
    # The comparator metadata returned by prior.external already validates the
    # archived hashes and confirms complete coverage of this exact axis.
    rows = list(prior.external(w,domain))
    raw_rows = [r for r in panel['rows'] if r['method']!='LEFT native W192']
    return rows, {
        item['method']: read_archive_curve(w, raw, axis)
        for item, raw in zip(rows, raw_rows)
    }


def read_archive_curve(w, row, axis):
    with np.load(w/row['path'], allow_pickle=False) as z:
        idx = (np.arange(*row['input_scored_axis']) if row['index_key'] is None
               else z[row['index_key']].reshape(-1)-row['index_offset'])
        take = np.searchsorted(idx, axis)
        np.testing.assert_array_equal(idx[take], axis)
        return z[row['score_key']].reshape(-1)[take]


def plot(root, domain, y, bank, rows, *, names=None, dashed=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (pr, roc) = plt.subplots(1,2,figsize=(9,4))
    palette = dict(native='#222222', observation='#0072B2', prototype='#D55E00',
                   mismatched='#D55E00', level='#009E73')
    external = ['#5F9EA0','#7C4A87','#88913E','#94663A','#A94D89','#636A95']
    handles = []
    ext_id = 0
    for key, values in bank.items():
        row = rows[key]
        prefix = key.split('_')[0]
        if prefix in palette:
            color, ls = palette[prefix], '--' if key.endswith('_on') or key in dashed else '-'
        else:
            color, ls = external[ext_id], ':'
            ext_id += 1
        precision, recall, _ = precision_recall_curve(y,values)
        fpr, tpr, _ = roc_curve(y,values)
        pr.plot(recall,precision,color=color,ls=ls,lw=1.15)
        roc.plot(fpr,tpr,color=color,ls=ls,lw=1.15)
        name = (names or NAMES).get(key,key).replace('-conditioned training, detection',' conditioning,')
        handles.append(Line2D([],[],color=color,ls=ls,label=f"{name} | {row['AP']:.3f} / {row['ROC_AUC']:.3f}"))
    pr.axhline(y.mean(),color='#aaa',ls=':',lw=.8)
    roc.plot([0,1],[0,1],color='#aaa',ls=':',lw=.8)
    pr.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Precision–recall')
    roc.set(xlabel='False positive rate',ylabel='True positive rate',xlim=(0,1),ylim=(0,1),title='ROC')
    fig.suptitle(f'{domain if domain=="PSM" else "SWaT"} · shared offline evaluation axis')
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=7.6, title='Method | AP / ROC-AUC',title_fontsize=8)
    bottom = (.33 if domain=='PSM' else .25) + (.04 if any(k.startswith('mismatched_') for k in bank) else 0)
    fig.tight_layout(rect=(0,bottom,1,.94))
    for suffix in ('svg','png'):
        fig.savefig(root/f'{domain.lower()}_pr_roc.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def evaluate(w):
    start = time.perf_counter()
    domains = ('PSM','SWAT')
    archived = read(w/OUT/'METRICS.json')
    root = w/ROOT
    by_domain = {}
    tasks = []
    keys = []
    for domain in domains:
        (root/domain).mkdir(parents=True, exist_ok=True)
        y = prior.truth(w,domain)
        bank, bindings = scores(w,domain)
        bank['level'], level_receipt = level(w,domain)
        bindings['level'] = level_receipt['bank_sha256']
        ext, external_banks = archived_scores(w,domain)
        rows = {}
        for ref in archived['domains'][domain]['rows'][:3]:
            key = ref['method'] if ref['method']=='native' else ref['method']+'_on'
            assert ref['score_sha256']==bindings[key]
            rows[key] = {k:ref[k] for k in prior.METRICS}
        rows.update({r['method']:r for r in ext})
        for key in ('observation_off','prototype_off','level'):
            tasks.append((w,y,bank[key]))
            keys.append((domain,key))
        by_domain[domain] = dict(rows=rows, bank={**bank,**external_banks}, y=y,
                                 score_bindings=bindings, level=level_receipt)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for (domain,key), result in zip(keys,pool.map(metric,tasks)):
            by_domain[domain]['rows'][key] = result
    final = {}
    for domain, entry in by_domain.items():
        rows, bank, y = entry['rows'], entry['bank'], entry['y']
        errors = [abs(got-rows[k][m]) for k,s in bank.items()
                  for got,m in zip(ranking_check(y,s),('AP','ROC_AUC'))]
        assert max(errors)<1e-12, max(errors)
        order = (*FORMAL,'observation_on','prototype_on',*(k for k in rows if k not in NAMES))
        plot(root,domain,y,{k:bank[k] for k in order},rows)
        final[domain] = dict(rows=rows, axis=prior.AXES[domain],
             score_bindings=entry['score_bindings'], level=entry['level'],
             contrasts=[dict(method=m, reference=r, **{k:rows[m][k]-rows[r][k] for k in prior.METRICS})
                        for m,r in CONTRASTS],
             independent_AP_ROC_max_error=max(errors), AP_ROC_rechecked_rows=len(errors)//2)
    # Match the previously reported SWaT fixed-checkpoint intervention.
    old = read(w/SWAT_OFF/'ANALYSIS.json')
    for arm in ARMS:
        for metric_key in ('AP','ROC_AUC'):
            key = arm+'_off'
            assert abs(final['SWAT']['rows'][key][metric_key]-old['rows'][key][metric_key])<1e-14
    dump(root/'METRICS.json', dict(domains=final, cpu_wall_seconds=time.perf_counter()-start,
         source_sha256=sha(__file__), holdout_note='Fixed after SWaT intervention, before PSM off readout',
         VUS=dict(windowSize=100,thre=250), operations='NA', complete=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('score_psm','evaluate'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    (score_psm if args.action=='score_psm' else evaluate)(Path.cwd())
