"""Fixed current-block evolution study; reuse existing raw external banks.

No model selection or score definition is performed here. All CAL thresholds
are sealed before labels are opened. Historical data and artifacts are read-only.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from .terminal_aligned_experiment import read, load, sha, dump, evaluator
from .normality_density_evaluation import new_bank, table, evaluate_series, labels

HISTORY = 'hybrid_normality_density_20261115_v1'
VARIANTS = {'full':'Hybrid_evolution', 'time_only':'Hybrid_time_only',
            'no_evolution':'Hybrid_no_evolution'}


def aligned_values(bank, axis):
    """One-record cached baseline, never interpolate or bridge missing points."""
    idx = np.asarray(bank['index']).reshape(-1)
    score = np.asarray(bank['score']).reshape(-1)
    pos = np.searchsorted(idx, axis)
    if np.any(pos >= len(idx)) or not np.array_equal(idx[pos], axis):
        raise ValueError('Historical baseline does not cover the fixed axis')
    return score[pos]


def threshold(cal, axis, domain):
    if domain == 'TEP':
        return float(np.quantile(cal, .99, method='higher')), None
    selected = np.linspace(106720,132480,512,dtype=np.int64) if domain=='PSM' else np.linspace(446173,494999,512,dtype=np.int64)
    pos = np.searchsorted(axis, selected)
    if np.any(pos >= len(axis)) or not np.array_equal(axis[pos],selected):
        raise ValueError('Required CAL positions absent')
    return float(np.quantile(cal[pos], .99, method='higher')), selected


def seal(workspace, output, config):
    cfg = read(config)
    if cfg['seed'] != 2025 or cfg['variants'] != list(VARIANTS):
        raise ValueError('Unregistered comparison')
    root=output/'evaluation'; root.mkdir(exist_ok=False)
    for domain in ('TEP','PSM','SWAT'):
        folder=root/domain; folder.mkdir(); (folder/'banks').mkdir()
        roles=('CAL','QUERY','NORMAL_TEST') if domain=='TEP' else ('CAL','QUERY')
        axes={}; identities={}; sources=[]; entries={}
        for variant,name in VARIANTS.items():
            entry={}
            for role in roles:
                z, source=new_bank(output,domain,variant,role)
                axes.setdefault(role,z['index'])
                ids=np.column_stack([z['fault'][:,0],z['run'][:,0]])
                identities.setdefault(role,ids)
                assert np.array_equal(axes[role],z['index'])
                assert np.array_equal(identities[role],ids)
                entry[role]=z['raw_score'] if domain=='TEP' else z['raw_score'][0]
                sources.append(source)
            entries[name]=entry
        old=workspace/'results'/HISTORY/'evaluation'/domain
        oldseal=read(old/'RAW_SEAL.json')
        keep=['LEFT_native','HistoryMean100_matched' if domain=='TEP' else 'HistoryMean192_matched','FIT_level_matched']
        if domain=='TEP': keep+=['LEFT_target9','Ridge_H','Ridge_C','Simple_native_RidgeC_half']
        for name in keep:
            desc=next(d for d in oldseal['methods'] if d['method']==name)
            f=old/'banks'/f'{name}.npz'; assert sha(f)==desc['sha256']
            z=load(f)
            for role in roles: assert np.array_equal(z[role+'_index'],axes[role][0])
            if domain=='TEP':
                assert np.array_equal(z['QUERY_ids'],identities['QUERY'])
                assert np.array_equal(z['NORMAL_TEST_ids'],identities['NORMAL_TEST'])
            entries[name]={role:z[role] for role in roles}
            sources.append(dict(path=str(f),sha256=sha(f),reused=True))
        if domain=='SWAT':
            entries['HistoryMean512']={}
            for role in roles:
                f=workspace/'results/left_multievent_mixed_u_20261111_v1/HistoryMean512'/f'{role}.npz'
                entries['HistoryMean512'][role]=aligned_values(load(f),axes[role][0])
                sources.append(dict(path=str(f),sha256=sha(f),reused=True))
        descriptors=[]
        for name,entry in entries.items():
            tau,selected=threshold(entry['CAL'],axes['CAL'][0],domain)
            extra={role+'_index':axes[role][0] for role in roles}
            if domain=='TEP': extra.update(QUERY_ids=identities['QUERY'],NORMAL_TEST_ids=identities['NORMAL_TEST'])
            if selected is not None: extra['selected_CAL']=selected
            f=folder/'banks'/f'{name}.npz'; np.savez_compressed(f,**entry,**extra)
            descriptors.append(dict(method=name,path=str(f),sha256=sha(f),threshold=tau))
        dump(folder/'RAW_SEAL.json',dict(status='SEALED_BEFORE_LABEL_EVALUATION',methods=descriptors,
             sources=sources,config_sha256=sha(config),implementation_sha256=sha(__file__)))
        dump(folder/'metric_definition.json',dict(
            primary='mean two-sided predictive mixture Rosenblatt evidence of current four-observation block',
            temporal='block-previous history predicts mode weights; current within-block coordinates used only in joint Rosenblatt factorization; no future beyond t',
            threshold='original own legal CAL q99 higher; strict >; unchanged confirmed3',
            scope='historically exposed single-seed development; no trusted-context or U/F claim',
            permissions='all neural target normal FIT/SELECT; no anomaly training labels; TEP Ridge scores target9 whereas Hybrid and LEFT_native score all3',
            long_interior='TEP561..960; PSM/SWAT after first512 from original boundary; not physical steady-state truth'))
    dump(root/'ALL_DOMAINS_SEALED.json',dict(status='SEALED',labels_opened=False,config_sha256=sha(config)))


def external_psm(workspace, output):
    from sklearn.metrics import average_precision_score, roc_auc_score
    truth,_=labels(workspace,'PSM'); folder=output/'evaluation/PSM'
    axis=np.arange(512,87744); rows=[]
    for d in read(folder/'RAW_SEAL.json')['methods']:
        z=load(d['path']); assert np.array_equal(z['QUERY_index'][:len(axis)],axis)
        score=z['QUERY'][:len(axis)]
        rows.append(dict(method=d['method'],AP=float(average_precision_score(truth[axis],score)),
            ROC=float(roc_auc_score(truth[axis],score)),points=len(axis),
            inference='causal at endpoint; Hybrid scores last4 observed points',
            training='target normal FIT/SELECT for learned models; original FIT/CAL for simple models',
            source_sha256=d['sha256']))
    prior=workspace/'results/hybrid_v1_psm_U_scope2000_fixed_evaluation_20261107_v1'
    for name,permission in [('TimeRadar_official_native','official pretrained; target pre-test normalization; native offline future-window access'),
                            ('HistoryMean512_ZS','no target fitting; source scales and calibration; strictly past512')]:
        f=prior/name/'common_axis_inputs.npz'; z=load(f)
        assert np.array_equal(z['index'],axis) and z['valid'].all()
        rows.append(dict(method=name,AP=float(average_precision_score(truth[axis],z['score'])),
            ROC=float(roc_auc_score(truth[axis],z['score'])),points=len(axis),inference=permission,
            training=permission,source_sha256=sha(f)))
    table(folder/'external_common_axis.csv',rows)


def evaluate(workspace,output):
    start=time.monotonic(); read(output/'evaluation/ALL_DOMAINS_SEALED.json')
    evaluator(workspace,output).evaluate_tep()
    for d in ('PSM','SWAT'): evaluate_series(workspace,output,d)
    external_psm(workspace,output)
    dump(output/'evaluation/FINAL_EVALUATION_RECEIPT.json',dict(status='COMPLETED',seconds=time.monotonic()-start,
         implementation_sha256=sha(__file__),no_QUERY_parameter_selection=True))


def main():
    import torch
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['seal','evaluate'])
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    a=parser.parse_args()
    if a.phase=='seal':
        if a.config is None: parser.error('--config required')
        seal(a.workspace.resolve(),a.output.resolve(),a.config.resolve())
    else: evaluate(a.workspace.resolve(),a.output.resolve())


if __name__=='__main__': main()
