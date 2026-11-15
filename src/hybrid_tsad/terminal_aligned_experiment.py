"""Bounded cache-only endpoint experiment, with immutable historical inputs.

Uses the frozen shared-cycle evaluator rather than a new metric definition.
Preparation reads no QUERY labels; evaluation is a separate explicit phase.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .terminal_aligned_cycle import terminal_evidence

HISTORY = 'hybrid_native_predictive_cycle_20261113_v1'
JOBS = ('TEP_native', 'TEP_full', 'TEP_history_only', 'PSM_native', 'PSM_full')
EVALUATOR_SHA = 'f65223a67014d571f2eb175ca01e9933b06b12cdb6f8a35834bcc3a51134cb0a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def prepare(workspace, output, config):
    started = time.time()
    frozen = read(config)
    assert frozen['seed'] == 2025 and frozen['jobs'] == list(JOBS)
    if output.exists():
        # Only a pre-computation missing-asset recovery may reuse the directory.
        assert read(output/'FIXED_EXPERIMENT.json') == frozen
        assert all(p.name == 'FIXED_EXPERIMENT.json' or p.name.startswith('PREPARATION_FAILURE_') for p in output.iterdir())
    else:
        output.mkdir(parents=True)
        dump(output / 'FIXED_EXPERIMENT.json', frozen)
    prep, history = workspace/'preparations'/HISTORY, workspace/'results'/HISTORY
    gate = read(prep/'DATA_GATE.json')
    provenance, selections, counts = [], [], []
    for job in JOBS:
        domain, mode = job.split('_', 1)
        source = history/job
        trace = read(source/'train/TRAIN_TRACE.json')
        criterion = 'native' if mode == 'native' else 'joint'
        chosen = min((r for r in trace if r['eligible']), key=lambda r: r['SELECT'][criterion])
        checkpoint = source/'train/selected.pth'
        receipt = read(source/'score/SCORE_RECEIPT.json')
        train_receipt = read(source/'train/TRAIN_RECEIPT.json')
        assert sha(checkpoint) == receipt['selected_PT_sha256'] == train_receipt['selected_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['epoch'] == chosen['epoch'], 'Matching selected checkpoint unavailable; do not substitute'
        assert state['python_step'] == chosen['python_step'] == receipt['selected_python_step']
        initial = read(source/'train/INITIAL.json')
        cfg = read(prep/'configs'/f'{domain}.json')
        scaler = read(prep/f'{domain}_FIT_ONLY_SCALER.json')
        assert sha(prep/'configs'/f'{domain}.json') == initial['config_sha256']
        assert sha(prep/f'{domain}_FIT_ONLY_SCALER.json') == initial['scaler_sha256']
        assert sha(prep/'data'/f'{domain}_DRAWPLAN.npz') == initial['plan_sha256']
        for role in ('FIT', 'SELECT'):
            e = next(r for r in gate['roles'] if r['domain'] == domain and r['role'] == role)
            assert sha(prep/'data'/f'{domain}_{role}.npz') == e['sha256']
            assert any(r['role'] == role and r['sha256'] == e['sha256'] for r in train_receipt['source_roles'])
        selections.append(dict(job=job,criterion=criterion,epoch=state['epoch'],step=state['python_step'],
                               selected_loss=chosen['SELECT'][criterion],checkpoint_sha256=sha(checkpoint),
                               train_trace_sha256=sha(source/'train/TRAIN_TRACE.json'),
                               training_updates=train_receipt['updates'],new_training=False,
                               plan_sha256=initial['plan_sha256'],config_sha256=initial['config_sha256'],
                               scaler_sha256=initial['scaler_sha256']))
        del state
        dest = output/job/'score'; dest.mkdir(parents=True)
        roles = ('CAL','QUERY','NORMAL_TEST') if domain == 'TEP' else ('CAL','QUERY')
        for role in roles:
            file = source/'score'/f'{role}.npz'
            original_sha = sha(file)
            assert original_sha == next(r['sha256'] for r in receipt['outputs'] if r['role'] == role)
            z = load(file)
            rawfile = prep/'data'/f'{domain}_{role}.npz'
            original = next(r for r in gate['roles'] if r['domain'] == domain and r['role'] == role)
            assert sha(rawfile) == original['sha256']
            assert any(r['role'] == role and r['sha256'] == original['sha256'] for r in receipt['roles'])
            raw = load(rawfile)
            offset = raw['original_start'][:,None]
            loc = z['index'] - offset - (1 if domain == 'TEP' else 0)
            assert (loc >= cfg['seq_len']).all() and (loc < raw['values'].shape[1]).all()
            assert np.all(np.diff(z['index'], axis=1) == 1)
            assert np.array_equal(z['run'][:,0],raw['run']) and np.array_equal(z['fault'][:,0],raw['fault'])
            actual = raw['values'][np.arange(len(loc))[:,None], loc]
            assert np.array_equal(actual, z['target_all_original'])
            standardized = np.float32((actual-np.asarray(scaler['mean']))/np.asarray(scaler['scale']))
            assert np.array_equal(standardized,z['target_standardized'])
            assert np.allclose(z['prediction'],z['time_prediction_standardized']*np.asarray(scaler['scale'])+np.asarray(scaler['mean']),rtol=0,atol=0)
            availability = load(source/'score'/f'{role}.availability.npz')
            assert np.array_equal(availability['available_index'],z['index'])
            if mode != 'native':
                assert np.array_equal(availability['prediction_issue_index'],z['index']-1)
                assert np.array_equal(availability['prediction_max_input_index'],z['index']-1)
                z['legacy_raw_score'] = z['raw_score'].copy()
                z['legacy_channel_score'] = z['channel_score'].copy()
                with torch.no_grad():
                    evidence, parts = terminal_evidence(
                        *[torch.from_numpy(z[k]) for k in ('time_prediction_standardized','freq_prediction_standardized','MS_prediction_standardized','target_standardized')],
                        alpha_cycle=cfg['alpha_cycle_l'],alpha_ms=cfg['alpha_ms_l'],alpha_cons=cfg['alpha_cons_l'])
                # Preserve every channel's components before task-specific reduction.
                z['channel_score'] = evidence.numpy()
                for k,v in parts.items(): z['terminal_'+k] = v.numpy()
                channels = cfg.get('prediction_channels')
                z['raw_score'] = z['channel_score'][...,channels].mean(-1) if channels is not None else z['channel_score'].mean(-1)
                z['all_channel_mean'] = z['channel_score'].mean(-1)
                assert z['raw_score'].shape == z['index'].shape and np.isfinite(z['raw_score']).all()
                np.savez_compressed(dest/f'{role}.npz', **z)
            else:
                shutil.copy2(file,dest/f'{role}.npz')
            shutil.copy2(source/'score'/f'{role}.availability.npz',dest/f'{role}.availability.npz')
            provenance.append(dict(job=job,role=role,source=str(file),source_sha256=original_sha,
                                   derived=str(dest/f'{role}.npz'),derived_sha256=sha(dest/f'{role}.npz'),
                                   checkpoint_sha256=receipt['selected_PT_sha256'],raw_sha256=original['sha256'],
                                   availability_sha256=sha(dest/f'{role}.availability.npz')))
            counts.append(dict(job=job,role=role,series=len(z['index']),points=int(z['index'].size)))
    dump(output/'REUSE_AND_ALIGNMENT.json',dict(status='PASS',selections=selections,banks=provenance,
         counts=counts,unit='FIT-standardized; SmoothL1 beta1',labels_opened=False,
         metadata_only_fault_run_identity=True,QUERY_statistics_fitted=False,new_training=False,
         new_forward=False,GPU_hours=0,seconds=time.time()-started,source_sha256=sha(__file__)))


def evaluator(workspace, output):
    repo = Path(__file__).resolve().parents[2]
    file = repo/'review_artifacts/code/preparations'/HISTORY/'fixed_evaluate.py'
    assert sha(file) == EVALUATOR_SHA, 'Do not rewrite historical metric identity'
    spec = importlib.util.spec_from_file_location('frozen_shared_cycle_metrics',file)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.R, module.P, module.O = workspace, workspace/'preparations'/HISTORY, output
    original_read = module.read
    def portable_read(path):
        data = original_read(path)
        if Path(path).name == 'EVALUATION.ACTUAL.json':
            name = str(data['truth']['path']).replace('\\','/')
            for anchor in ('preparations/', 'results/'):
                if anchor in name:
                    data['truth']['path'] = str(workspace / name[name.index(anchor):]); break
        return data
    module.read = portable_read
    return module


def seal_and_evaluate(workspace, output):
    started = time.time()
    reuse = read(output/'REUSE_AND_ALIGNMENT.json'); assert reuse['status'] == 'PASS'
    for b in reuse['banks']:
        assert sha(b['source']) == b['source_sha256']
        assert sha(b['derived']) == b['derived_sha256']
    m = evaluator(workspace,output)
    m.cpu_tests()
    m.seal_tep(); m.seal_psm()
    for domain in ('TEP','PSM'):
        folder=output/'evaluation'/domain
        seal=read(folder/'RAW_SEAL.json')
        # Same Hybrid weights, historical raw readout, same CAL/QUERY axes.
        source=workspace/'results'/HISTORY/'evaluation'/domain
        old=read(source/'RAW_SEAL.json')
        entry=next(x for x in old['methods'] if x['method']=='Hybrid_full')
        original=source/'banks/Hybrid_full.npz'; assert sha(original)==entry['sha256']
        dest=folder/'banks/Hybrid_legacy_same_weights.npz'; shutil.copy2(original,dest)
        oldbank=load(original)
        for role in ('CAL','QUERY') + (('NORMAL_TEST',) if domain=='TEP' else ()):
            arr=load(output/f'{domain}_full'/'score'/f'{role}.npz')['legacy_raw_score']
            if domain=='PSM': arr=arr[0]
            assert np.array_equal(oldbank[role],arr)
        cal=oldbank['CAL']
        if domain=='PSM': cal=cal[oldbank['selected_CAL']-oldbank['CAL_index'][0]]
        tau=float(np.quantile(cal,.99,method='higher'))
        assert tau==entry['threshold']
        seal['methods'].append(dict(method='Hybrid_legacy_same_weights',path=str(dest),sha256=sha(dest),threshold=tau))
        seal['current_code_sha256']=sha(__file__)
        seal['score_revision']='terminal-aligned-v1; native loss selection; legacy same weights retained'
        # These are fresh output manifests, finalized BEFORE label evaluation.
        (folder/'RAW_SEAL.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
        protocol=read(folder/'metric_definition.json')
        protocol['primary']='Hybrid terminal SmoothL1 .5(time+frequency)+.5MS+.1consistency; native scoring unchanged'
        protocol['selection']='native: native SELECT; full/history_only: joint SELECT; cached same argmin checkpoints'
        protocol['method_names']={'Hybrid_full':'endpoint-aligned Hybrid','Hybrid_self_only':'endpoint-aligned without conditions','Hybrid_legacy_same_weights':'historical readout on identical full weights','Hybrid_nativeoff':'native reconstruction readout on identical full weights'}
        (folder/'metric_definition.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # No label-based decisions occur between sealing and evaluation.
    m.evaluate_tep(); m.evaluate_psm()
    supplement_events(output)
    verify_unchanged_controls(workspace,output)
    dump(output/'COMPUTE_RECEIPT.json',dict(status='COMPLETED',seconds=time.time()-started,GPU_hours=0,
         new_training=False,new_forward=False,seed=2025,metric_implementation_sha256=EVALUATOR_SHA,
         source_sha256=sha(__file__),QUERY_selection=False))


def supplement_events(output):
    """Expose any-hit separately; never replace one-to-one counts."""
    import pandas as pd
    folder=output/'evaluation/TEP'
    frame=pd.read_csv(folder/'per_fault_run.csv')
    frame['any_hit'] = frame.detected_anomaly_points > 0
    frame['any_hit_equals_one_to_one_this_single_event_run'] = frame.any_hit == (frame.M>0)
    assert frame.any_hit_equals_one_to_one_this_single_event_run.all()
    frame.to_csv(folder/'per_fault_run_with_anyhit.csv',index=False)
    summary=pd.read_csv(folder/'summary.csv')
    summary['any_hit_events']=summary.M
    summary['any_hit_recall']=summary.event_R
    summary['any_hit_scope']='single fault event per run; no equivalence assumed for multievent PSM'
    summary.to_csv(folder/'summary_with_anyhit.csv',index=False)


def verify_unchanged_controls(workspace, output):
    import pandas as pd
    rows=[]
    for domain in ('TEP','PSM'):
        old=pd.read_csv(workspace/'results'/HISTORY/'evaluation'/domain/'summary.csv')
        new=pd.read_csv(output/'evaluation'/domain/'summary.csv')
        methods=['LEFT_native','P0','MeanFIT']+(['Ridge_H','Ridge_C','Simple_native_RidgeC_half'] if domain=='TEP' else [])
        for name in methods+['Hybrid_legacy_same_weights']:
            previous='Hybrid_full' if name=='Hybrid_legacy_same_weights' else name
            a=old[old.method==previous].sort_values(['scope','stream']).reset_index(drop=True)
            b=new[new.method==name].sort_values(['scope','stream']).reset_index(drop=True)
            assert len(a)==len(b)>0
            columns=[c for c in a if c!='method' and c in b]
            pd.testing.assert_frame_equal(a[columns],b[columns],check_exact=True)
            rows.append(dict(domain=domain,method=name,rows=len(a),status='EXACT_METRIC_REPRODUCTION'))
    dump(output/'UNCHANGED_CONTROLS.json',rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['prepare','evaluate'])
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--config',type=Path)
    a=ap.parse_args(); torch.set_num_threads(2)
    if a.phase=='prepare':
        if a.config is None: ap.error('--config required for prepare')
        prepare(a.workspace.resolve(),a.output.resolve(),a.config)
    else: seal_and_evaluate(a.workspace.resolve(),a.output.resolve())


if __name__=='__main__':
    main()
