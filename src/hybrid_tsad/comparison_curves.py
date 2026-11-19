"""CPU-only common-axis cached curves with fixed, accepted CAL operating points.

Optional extra manifest: {"models": [{"domain": "SWAT", "method": "MLP_TT_A",
"query": "results/.../QUERY.npz", "query_sha256": "...",
"metrics": "results/.../METRICS.json", "metrics_sha256": "...",
"metrics_method": "MLP_TT_A"}, ...]}. Paths are workspace-relative.
No threshold is selected or changed here. No model or torch import is needed.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

OLD_METRICS = 'results/candi_native_a_metrics_20261118_v1/METRICS.json'
OLD_SHA = '57e3e8957d7ad53406107ce731d0b3ebc33b35c5e45d6911907ae74d9bdcda98'
AXES = {'SWAT': (608535, 944856), 'PSM': (22656, 87744)}
METHODS = ('LEFT_noU', 'LEFT_TT_A', 'LEFT_R_A', 'MLP_noU', 'CANDI_A', 'MLP_TT_A', 'MLP_R_A')
STYLES = (('#666666', '-'), ('#0072B2', '-'), ('#009E73', '-'),
          ('#CC79A7', '--'), ('#D55E00', '-.'), ('#E69F00', ':'), ('#332288', '--'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def checked(path, expected, inputs):
    value = sha(path)
    if value != expected:
        raise ValueError(f'Changed accepted cache: {path}')
    inputs[str(path)] = value
    return path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def points(labels, scores, threshold):
    alarm = scores > threshold
    tp = int(np.sum(alarm & (labels == 1)))
    fp = int(np.sum(alarm & (labels == 0)))
    return dict(TP=tp, FP=fp, recall=tp / int(np.sum(labels == 1)),
                FPR=fp / int(np.sum(labels == 0)), precision=tp / max(tp + fp, 1))


def load_models(workspace, extras=None):
    # Reuse the accepted LEFT identity reader, without rerunning evaluation.
    from .left_response_preservation_evaluation import baseline
    w = Path(workspace).resolve()
    inputs = {}
    old = read(checked(w / OLD_METRICS, OLD_SHA, inputs))
    root = w / 'results/candi_native_a_20261118_v1'
    seal = read(checked(root / 'GLOBAL_RAW_BANK_SEAL.json', old['identities']['global_seal_sha256'], inputs))
    ids = {b['relative'].replace('\\', '/'): b['sha256'] for b in seal['banks']}
    models, labels = {}, {}
    for domain, axis in AXES.items():
        task = read(checked(root / domain / 'TASK.json', old['identities'][domain]['task_sha256'], inputs))
        if tuple(task['score_axes']['QUERY']) != axis:
            raise ValueError('Changed accepted QUERY axis')
        d = task['labels']
        p = checked(w / d['relative'], d['sha256'], inputs)
        labels[domain] = (np.load(p)['label'] if p.suffix == '.npz' else np.load(p)).reshape(-1)[slice(*axis)]
        if not np.isin(labels[domain], [0, 1]).all():
            raise ValueError('Binary evaluation labels required')
        _, left = baseline(w, domain)
        # baseline() verifies every reused LEFT score hash. Record the same identities.
        desc = old['reused_LEFT_metrics'][domain]
        inputs[str(w / desc['relative'])] = desc['sha256']
        models[domain] = {}
        for native, display in [('noU', 'LEFT_noU'), ('TT', 'LEFT_TT_A'), ('R', 'LEFT_R_A')]:
            row = next(r for r in old['rows'] if r['domain'] == domain and r['method'] == display and r['stream'] == 'strict')
            models[domain][display] = (left[native], row)
        for native, display, key in [('noU', 'MLP_noU', 'CANDI_noU'), ('CANDI', 'CANDI_A', 'CANDI_A')]:
            relative = f'{domain}/{native}/QUERY.npz'
            p = checked(root / relative, ids[relative], inputs)
            score = endpoint_bank(p, axis)
            row = next(r for r in old['rows'] if r['domain'] == domain and r['method'] == key and r['stream'] == 'strict')
            models[domain][display] = (score, row)
    if extras is not None:
        inputs[str(extras)] = sha(extras)
        for item in read(extras)['models']:
            domain, name = item['domain'], item['method']
            if name not in METHODS[5:] or name in models[domain]:
                raise ValueError('Only new plain MLP TT/R rows may be merged once')
            p = checked(w / item['query'], item['query_sha256'], inputs)
            metric = read(checked(w / item['metrics'], item['metrics_sha256'], inputs))
            row = next(r for r in metric['rows'] if r['domain'] == domain
                       and r['method'] == item.get('metrics_method', name) and r['stream'] == 'strict')
            models[domain][name] = (endpoint_bank(p, AXES[domain]), row)
        if any(set(group) != set(METHODS) for group in models.values()):
            raise ValueError('Final comparison requires all seven methods in both domains')
    return models, labels, inputs


def endpoint_bank(path, axis):
    with np.load(path, allow_pickle=False) as z:
        ix = np.arange(*axis)
        if (not np.array_equal(z['absolute_time_index'], ix)
                or not np.array_equal(z['available_index'], ix)
                or not np.array_equal(z['window_start'], ix - 9)
                or not np.array_equal(z['window_end_exclusive'], ix + 1)):
            raise ValueError('MLP requires actual length10 endpoint windows on the common axis')
        scores = z['canonical_score'].copy()
    if scores.shape != ix.shape or not np.isfinite(scores).all():
        raise ValueError('Incomplete MLP scores')
    return scores


def render(workspace, output, extras=None):
    start = time.perf_counter()
    models, labels, inputs = load_models(workspace, extras)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    all_points = []
    for domain, group in models.items():
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
        y = labels[domain]
        arrays, outside = {}, []
        for name, (color, linestyle) in zip(METHODS, STYLES):
            if name not in group:
                continue
            scores, row = group[name]
            if len(scores) != len(y) or not np.isfinite(scores).all():
                raise ValueError('Unmatched score/label shape')
            op = points(y, scores, row['threshold'])
            if op['TP'] != row['TP'] or op['FP'] != row['FP']:
                raise ValueError(f'{name}: fixed CAL operating point differs from accepted metrics')
            ap, auc = float(average_precision_score(y, scores)), float(roc_auc_score(y, scores))
            if abs(ap-row['AP']) > 1e-12 or abs(auc-row['ROC_AUC']) > 1e-12:
                raise ValueError('Accepted ranking no longer matches cache')
            precision, recall, pt = precision_recall_curve(y, scores)
            fpr, tpr, rt = roc_curve(y, scores, drop_intermediate=False)
            for key, value in dict(precision=precision, recall=recall, PR_threshold=pt,
                                   FPR=fpr, TPR=tpr, ROC_threshold=rt).items():
                arrays[f'{name}_{key}'] = value
            kwargs = dict(color=color, linestyle=linestyle, lw=1.6)
            axes[0].plot(recall, precision, label=f'{name} (AP {ap:.3f})', **kwargs)
            for ax in axes[1:]:
                ax.plot(fpr, tpr, label=f'{name} (AUC {auc:.3f})', **kwargs)
            axes[0].scatter(op['recall'], op['precision'], s=40, facecolors='white', edgecolors=color, zorder=4)
            for ax in axes[1:]:
                ax.scatter(op['FPR'], op['recall'], s=40, facecolors='white', edgecolors=color, zorder=4)
            if op['FPR'] > .05:
                outside.append(f"{name}: FPR {100*op['FPR']:.2f}%, recall {100*op['recall']:.2f}%")
            all_points.append(dict(domain=domain, model=name, threshold=row['threshold'],
                                   AP=ap, ROC_AUC=auc, CAL_points=512, CAL_exceedances=row['CAL_exceedances'],
                                   scored_points=len(y), normal_points=int(np.sum(y==0)), attack_points=int(np.sum(y==1)),
                                   **op, low_FPR_panel_outside=op['FPR']>.05,
                                   readout='LEFT retrospective wholewindow192' if name.startswith('LEFT') else 'MLP length10 actual endpoint'))
        axes[0].set(title='Complete precision–recall', xlabel='Recall', ylabel='Precision', xlim=(0,1), ylim=(0,1.02))
        for ax, limit, title in [(axes[1],1,'Complete ROC'),(axes[2],.05,'ROC: normal FPR 0–5%')]:
            ax.set(title=title,xlabel='Normal false-positive rate',ylabel='Recall / true-positive rate',xlim=(0,limit),ylim=(0,1.02))
        for ax in axes:
            ax.grid(alpha=.2); ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.legend(loc='lower left',fontsize=7)
        fig.suptitle(f'{domain} seed2025 | natural A | {len(group)} cached methods | own fixed CAL operating points')
        note = 'Open circles: original own-model CAL q99/higher/>; no threshold selection. Mixed CAL; PSM CAL all attack-labeled.'
        note += '\nLEFT: retrospective wholewindow192; MLP/CANDI: length10 endpoint. Permissions/training exposure differ.'
        note += '\nOutside low-FPR panel: ' + ('; '.join(outside) if outside else 'none')
        fig.text(.5,.015,note,ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.12,1,.94))
        for ext in ('png','pdf','svg'):
            fig.savefig(output/f'{domain}_PR_ROC.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
        np.savez_compressed(output/f'{domain}_CURVE_ARRAYS.npz',**arrays,QUERY_indices=np.arange(*AXES[domain]))
    with (output/'operating_points.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_points[0]));writer.writeheader();writer.writerows(all_points)
    record=dict(status='FINAL7_CACHE_CURVES_PASS' if extras else 'PREP5_CACHE_CURVES_READY',
                domains={d:dict(axis=AXES[d],methods=list(g)) for d,g in models.items()},
                input_sha256=inputs,cpu_wall_seconds=time.perf_counter()-start,
                inference=False,training=False,threshold_search=False,
                checks='exact TP/FP, AP/ROC-AUC vs accepted metrics; endpoint bank axes/availability; fixed CAL thresholds referenced, not refit')
    (output/'CURVE_RECEIPT.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    manifest={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    (output/'OUTPUT_SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extras',type=Path)
    args=parser.parse_args()
    print(json.dumps(render(args.workspace,args.output,args.extras),indent=2))
