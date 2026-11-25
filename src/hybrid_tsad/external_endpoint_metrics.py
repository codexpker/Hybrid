"""Entity-first common-axis tables for endpoint and offline external detectors."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import native_event_space as nasa
from .external_frozen_bank import OUT, DADA_SHA, RCD_UNI, cached_rcd
from .left_context_unmodulated import ranking_check, read_archive_curve
from .left_decoder_context_metrics import MANIFEST
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_swat_author import dump, read, sha
from .native_event_rank_readout import asset
from .native_full_window_metrics import AUTHOR, METRICS, causal, left, truth
from .native_rank_psm_extension import EXTERNAL as PSM_REFS
from .timercd_fixed import ARMS, window_layout

DADA = 'DADA official V'
RCD_B = 'TimeRCD official weights, FIT-only adapter'
RCD_A = 'TimeRCD official whole-QUERY normalization'
LEFT = 'Complete LEFT'
HISTORY = 'History Mean512'
PCA = 'PCA residual'
CANDIDATE = 'Source-supervised endpoint V'
COPY = 'Source reconstruction continuation'
NASA_LABEL_SHA = '057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539'
TIMING = {
    DADA:'full W100, window-end', RCD_B:'full W5000 capped by length, actual window-end',
    RCD_A:'full W5000 capped by length, end of QUERY for statistics',
    LEFT:'full W96 MSL or W192 elsewhere, window-end', HISTORY:'past W512 and current value, pointwise',
    PCA:'current value only, pointwise',
}
BASE = ('official_V', 'official_E', 'official_V_plus_E')
SEEDS = (2025,2026,2027)


def new_bank(w, row, method, checkpoint, protocol_sha):
    p = w/OUT/row['domain']/row['entity']/f'{method}.npz'
    info = read(p.with_suffix('.json'))
    assert sha(p)==info['bank_sha256'] and info['checkpoint_sha256']==checkpoint and not info['labels_read']
    assert info['protocol_sha256']==protocol_sha and info['raw_assets_sha256']==row['raw_assets_sha256']
    with np.load(p,allow_pickle=False) as z:
        if method=='DADA':
            stop = row['test_points']//100*100
            index = np.arange(stop)
            np.testing.assert_array_equal(z['index'],index)
            np.testing.assert_array_equal(z['available_index'],index//100*100+99)
            return (index,z['score'],info['bank_sha256']),info
        idx = np.arange(row['test_points'])
        np.testing.assert_array_equal(z['index'],idx)
        np.testing.assert_array_equal(z['logit'],z['head_logits'][:,1]-z['head_logits'][:,0])
        assert z['valid'].all() and info['stats']['normalization_count']==1
        layout = window_layout(idx)
        ends = layout['window_end_index'] if method.endswith('B_fit_only') else np.full(len(idx),idx[-1])
        np.testing.assert_array_equal(z['available_index'],ends)
        np.testing.assert_array_equal(z['window_start_index'],layout['window_start_index'])
        return (idx,z['logit'],info['bank_sha256']),info


def causal_refs(w,d,e):
    ix, raw, bindings = causal(w,d,e)
    data = {}
    for k in BASE:
        data[k,0] = (ix,raw[k],bindings[0]['sha256'])
    for s in SEEDS:
        for alias,key in ((COPY,f'{s}_copy'),(CANDIDATE,f'{s}_point_weighted')):
            data[alias,s] = (ix,raw[key],bindings[int(s!=2025)]['sha256'])
    return data


def psm_cached(w, axis, methods):
    manifest = read(w/MANIFEST)
    panel, = (p for p in manifest['panels'] if p['dataset']=='PSM' and p['track']=='offline-readout')
    out = {}
    for row in panel['rows']:
        if row['method'] not in methods:
            continue
        asset(w,row['path'],row['sha256'])
        s = read_archive_curve(w,row,axis)
        init = row.get('archived_seed')
        name = row['method'] if init is None else f"{row['method']} seed{init}"
        out[name,0] = (axis,s,row['sha256'])
    return out


def history_ref(w, d, e, y):
    if d=='PSM':
        rel,digest = PSM_REFS['HistoryMean512']
        p = asset(w,rel,digest)
    else:
        p = w/'results/hybrid_v1_public_native_zs_20261105_v1/evaluation/entities/SMAP'/e/'history_mean512/calculation_inputs.npz'
        digest = sha(p)
    with np.load(p,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['index'],np.arange(512,len(y)))
        np.testing.assert_array_equal(z['full_labels'].reshape(-1),y)
        assert z['valid'].all()
        return z['index'],z['score'].reshape(-1),digest


def evaluate_entity(args):
    w,row,protocol_sha = args
    d,e,n = (row[k] for k in ('domain','entity','test_points'))
    y = truth(w,d,e)
    assert len(y)==n and np.isin(y,(0,1)).all()
    base = causal_refs(w,d,e)
    ix,left_score,left_identity = left(w,d,e)
    readouts = {**base, (LEFT,0):(ix,left_score,left_identity['sha256'])}
    dada, dada_receipt = new_bank(w,row,'DADA',DADA_SHA['pytorch_model.bin'],protocol_sha)
    readouts[DADA,0] = dada
    receipts = {DADA:dada_receipt['seconds']}
    if d in ('PSM','SMD'):
        cache = cached_rcd(w,e,n)
        for arm in ARMS:
            bank,digest = cache[arm]
            assert digest==row['TimeRCD_QUERY_sha256'][arm]
            name = RCD_A if arm==ARMS[0] else RCD_B
            idx = np.arange(n)
            ends = np.full(n,n-1) if arm==ARMS[0] else window_layout(idx)['window_end_index']
            np.testing.assert_array_equal(bank['available_index'],ends)
            readouts[name,0] = (bank['index'],bank['score'],digest)
    else:
        for arm,name in zip(ARMS,(RCD_A,RCD_B)):
            readouts[name,0], r = new_bank(w,row,'TimeRCD_'+arm,RCD_UNI,protocol_sha)
            receipts[name] = r['seconds']
            if arm==ARMS[1]:
                assert r['stats']['mean']==row['TimeRCD_fit_mean']
                assert r['stats']['std_ddof0_safe']==row['TimeRCD_fit_std']
    off = np.arange(*row['offline_axis'])
    assert off[-1]+1 == min(len(ix),len(dada[0]))
    if d=='PSM':
        readouts.update(psm_cached(w,off,('CATCH W192',PCA)))
    panels = [('causal',np.arange(*row['causal_axis']),base),
              ('common_offline',off,{k:v for k,v in readouts.items() if k[0]!=RCD_A}),
              ('whole_query_statistic',off,{(RCD_A,0):readouts[RCD_A,0]})]
    if 'history_axis' in row:
        hist = np.arange(*row['history_axis'])
        h = {**base,(HISTORY,0):history_ref(w,d,e,y)}
        if d=='PSM':
            h.update(psm_cached(w,hist,(PCA,)))
        panels.append(('causal_history',hist,h))
    scorer = load_author_metric(w/AUTHOR)
    rows,errors = [],[]
    for panel,axis,methods in panels:
        label = y[axis]
        assert np.unique(label).size==2
        for (name,seed),(index,score,digest) in methods.items():
            select = np.searchsorted(index,axis)
            np.testing.assert_array_equal(index[select],axis)
            s = score[select]
            assert np.isfinite(s).all()
            result,_ = continuous_metrics(label,s,scorer)
            ap,roc = ranking_check(label,s)
            errors.extend((abs(ap-result['AP']),abs(roc-result['ROC_AUC'])))
            timing = TIMING.get(name,'full W192, window-end' if name.startswith('CATCH') else 'causal W100 endpoint')
            rows.append(dict(domain=d,entity=e,panel=panel,method=name,seed=seed,start=int(axis[0]),
                stop=int(axis[-1])+1,points=len(axis),positives=int(label.sum()),timing=timing,bank_sha256=digest,
                **{k:result[k] for k in METRICS}))
    assert max(errors)<1e-12
    return rows, max(errors),receipts


def summaries(frame):
    grouped = frame.groupby(['domain','panel','method','seed','timing'],sort=False)
    metrics = grouped[list(METRICS)].mean()
    metrics['entities'] = grouped['entity'].size()
    metrics['points'] = grouped['points'].sum()
    paired, dist = [], []
    for domain,panel in frame[['domain','panel']].drop_duplicates().itertuples(index=False,name=None):
        data = frame[(frame.domain==domain)&(frame.panel==panel)]
        for seed in SEEDS:
            cand = data[(data.method==CANDIDATE)&(data.seed==seed)].set_index('entity')
            if cand.empty:
                continue
            refs = (('official_V',0),(COPY,seed)) if panel=='causal' else (
                (DADA,0),(RCD_B,0),(LEFT,0)) if panel=='common_offline' else ((HISTORY,0),(PCA,0))
            for method,ref_seed in refs:
                ref = data[(data.method==method)&(data.seed==ref_seed)].set_index('entity')
                if ref.empty:
                    continue
                assert cand.index.equals(ref.index)
                for dim in ('start','stop','points'):
                    assert cand[dim].equals(ref[dim])
                delta = cand[list(METRICS)]-ref[list(METRICS)]
                for entity,r in delta.iterrows():
                    paired.append(dict(domain=domain,panel=panel,entity=entity,seed=seed,reference=method,**r.to_dict()))
                for k in METRICS:
                    s = delta[k]
                    dist.append(dict(domain=domain,panel=panel,seed=seed,reference=method,metric=k,
                        mean_delta=s.mean(),median_delta=s.median(),wins=int((s>0).sum()),ties=int((s==0).sum()),
                        losses=int((s<0).sum()),entities=len(s)))
    return metrics.reset_index(),pd.DataFrame(paired),pd.DataFrame(dist)


def evaluate(w):
    start = time.perf_counter()
    root = w/OUT
    assert (root/'ALL_EXTERNAL_SCORES_READY').is_file()
    protocol = root/'AXES_AND_INPUTS.json'
    identity = sha(protocol)
    conf = read(protocol)
    assert not conf['labels_read'] and sha(w/nasa.MSL/'labels.csv')==NASA_LABEL_SHA
    records = conf['records']
    for method,n in (('DADA',109),('TimeRCD',80)):
        receipts = [read(f) for f in sorted(root.glob(f'{method}_LANE_*.json'))]
        got = [tuple(t) for r in receipts for t in r['records']]
        required = {(r['domain'],r['entity']) for r in records if method=='DADA' or r['domain'] in ('MSL','SMAP')}
        assert len(got)==n and set(got)==required
        assert all(r['zero_updates'] and r['model_unchanged'] and r['protocol_sha256']==identity for r in receipts)
    rows, costs = [], []
    err = 0.0
    with ProcessPoolExecutor(max_workers=12) as pool:
        for row,result in zip(records,pool.map(evaluate_entity,((w,r,identity) for r in records))):
            values,delta,used = result
            rows.extend(values)
            costs.extend(dict(domain=row['domain'],entity=row['entity'],method=m,seconds=s) for m,s in used.items())
            err=max(err,delta)
    frame = pd.DataFrame(rows)
    macro,pairs,dist = summaries(frame)
    frame.to_csv(root/'entity_metrics.csv',index=False)
    macro.to_csv(root/'domain_metrics.csv',index=False)
    pairs.to_csv(root/'entity_pairs.csv',index=False)
    dist.to_csv(root/'paired_distribution.csv',index=False)
    pd.DataFrame(costs).to_csv(root/'inference_cost.csv',index=False)
    dump(root/'EVALUATION.json',dict(complete=True,protocol_sha256=identity,scored_entities=len(records),
        metric_rows=len(frame),AP_ROC_max_abs_recheck=err,CPU_wall_seconds=time.perf_counter()-start,
        metric=dict(VUS_windowSize=100,VUS_thre=250),score_files=dict(new_DADA=109,new_TimeRCD=160,reused_TimeRCD=58),
        train_updates=0,source_sha256=sha(__file__),calibrated_new_methods=False))


if __name__=='__main__':
    evaluate(Path.cwd())
