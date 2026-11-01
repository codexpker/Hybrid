"""Separately released CPU-only evaluation of 80 frozen LEFT score banks.

Default preview reads config metadata only. This module never calls a model,
opens features/checkpoints, generates a run, or chooses a threshold/checkpoint.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

from .. import left_normal_scores as score
from .. import weak_lag as wl


def dump(path, obj):
    import numpy as np
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj, indent=2, default=convert, allow_nan=False)+'\n')


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def validate_truth_references(refs):
    expected = [(s, i) for s in ('LIN', 'WEAK') for i in range(8)]
    identities = []
    for ref in refs:
        path = Path(ref['path']); scene, rid = ref['scenario'], ref['run_id']
        if (scene not in ('LIN', 'WEAK') or ref['role'] != 'DEV' or
                type(rid) is not int or rid not in range(8) or
                tuple(path.parts[-3:-1]) != (scene, 'DEV') or
                path.name != f'run{rid:02d}_truth.json' or
                not re.fullmatch('[0-9a-f]{64}', ref['sha256'])):
            raise PermissionError('Only exact LIN/WEAK DEV16 truth references admitted')
        identities.append((scene, rid))
    if identities != expected:
        raise PermissionError('All 16 unique ordered DEV truth identities required')


def validate_config(cfg):
    if (cfg['operation'] != 'LEFT_WL_C2_LABEL_EVALUATION' or
            cfg['source_manifest_sha256'] != score.MANIFEST_SHA or cfg['seed'] != 2025 or
            cfg['bootstrap_repeats'] != 10000 or cfg['max_seconds'] != 300 or
            cfg['max_rss_GiB'] != 2 or cfg['gpu_budget'] != 0):
        raise ValueError('Frozen CPU evaluation protocol/budget required')
    validate_truth_references(cfg['truths'])
    if Path(cfg['cpu_all_layers']['path']).name != 'all_layers.csv':
        raise PermissionError('Only pinned original CPU all_layers.csv admitted')
    reports = cfg['score_reports']
    if [r['scenario'] for r in reports] != ['LIN','WEAK']:
        raise PermissionError('Two ordered, separately completed scene reports required')
    for row in reports:
        if (Path(row['path']).name != 'report.json' or Path(row['config']['path']).name != 'config.json' or
                not all(re.fullmatch('[0-9a-f]{64}',row[k]) for k in ('checkpoint_sha256','scaler_sha256'))):
            raise PermissionError('Expected fixed-model/scaler identities and score report/config references required')
    return cfg


def require_release(config_path, release_path):
    cfg = validate_config(json.loads(Path(config_path).read_text()))
    release = json.loads(Path(release_path).read_text())
    if (cfg['status'] != 'FROZEN' or release.get('status') != 'APPROVED' or
            release.get('approved_by') != 'root' or release.get('allow_truth') is not True or
            release.get('operation') != cfg['operation'] or
            release.get('config_sha256') != score.sha(config_path) or
            release.get('module_sha256') != score.sha(__file__) or
            release.get('score_module_sha256') != score.sha(score.__file__) or
            release.get('weak_lag_module_sha256') != score.sha(wl.__file__)):
        raise PermissionError('Exact root CPU-evaluation release required')
    for item in (cfg['score_seal'], cfg['cpu_all_layers']):
        if not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise PermissionError('Frozen seal and original CPU metric-table SHA required')
    # Only the seal is read now. No truth or metric table body before the seal.
    if score.sha(cfg['score_seal']['path']) != cfg['score_seal']['sha256']:
        raise ValueError('Score seal SHA mismatch')
    seal = score.require_evaluation_seal(cfg['score_seal']['path'])
    return cfg, seal


def validate_evaluation_inputs(cfg, seal):
    """Close the complete80-bank label gate before any truth/metric-table IO.

    Hash equality alone is insufficient: N/S0 may share content, but each bank
    must be a unique resolved path and physical file tied to its actual report.
    """
    seal_rows = {}; resolved = set(); physical = set()
    for row in seal['banks']:
        ref=row['reference'];key=(ref['scenario'],ref['role'],ref['run_id'],ref['variant'])
        score.validate_reference(ref,ref['scenario'],ref['role'])
        path=Path(row['path']).resolve(strict=True);stat=path.stat()
        inode=(stat.st_dev,stat.st_ino)
        if (not path.is_file() or path.suffix != '.npz' or str(path) in resolved or
                inode in physical or key in seal_rows):
            raise PermissionError('Bank must be a unique real NPZ path and physical file; aliases denied')
        resolved.add(str(path));physical.add(inode);seal_rows[key]=row
    expected={(s,r,i,v) for s in ('LIN','WEAK') for r in ('CAL','DEV') for i in range(8)
              for v in (('N',) if r=='CAL' else ('A','N','S0','SB'))}
    if len(seal_rows)!=80 or set(seal_rows)!=expected:
        raise PermissionError('Exactly all80 unique score identities required')
    banks={};provenance=[]
    for item in cfg['score_reports']:
        scene=item['scenario'];report_path=Path(item['path']).resolve(strict=True)
        config_path=Path(item['config']['path']).resolve(strict=True)
        if (score.sha(report_path)!=item['sha256'] or score.sha(config_path)!=item['config']['sha256'] or
                config_path.parent!=report_path.parent):
            raise ValueError('Completed score report/config hash or run directory mismatch')
        report=json.loads(report_path.read_text());scfg=json.loads(config_path.read_text())
        if (report.get('status')!='Completed' or report.get('label_files_read') is not False or
                report.get('unscored_points')!=0 or len(report.get('banks',[]))!=40 or
                report.get('config_sha256')!=item['config']['sha256'] or
                report.get('module_sha256')!=score.sha(score.__file__) or
                scfg.get('status')!='FROZEN' or scfg.get('scenario')!=scene or scfg.get('seed')!=2025 or
                scfg.get('operation')!='LEFT_WL_C2_FIXED_SCORE' or
                scfg.get('source_manifest_sha256')!=score.MANIFEST_SHA):
            raise PermissionError('Incomplete/incompatible scene score report or frozen config')
        for key in ('checkpoint','scaler'):
            if (report.get(key)!=scfg.get(key) or
                    scfg[key]['sha256']!=item[key+'_sha256']):
                raise ValueError('Completed report model/scaler provenance differs from fixed expected identity')
        if (report.get('model_config',{}).get('enc_in')!=2 or
                report.get('model_config',{}).get('seq_len')!=192 or
                report.get('state_before')!=report.get('state_after') or
                report.get('state_before',{}).get('step')!=0 or
                not report.get('state_before',{}).get('state_sha256') or
                not all(re.fullmatch('[0-9a-f]{64}',v) for v in report['state_before']['state_sha256'].values()) or
                not report.get('state_before',{}).get('modes') or
                any(report['state_before']['modes'].values())):
            raise ValueError('C2 model/registered-state freeze evidence missing or changed')
        snapshot=report_path.parent/'source'
        if score.sha(snapshot/'left_normal_scores.py')!=report['module_sha256']:
            raise ValueError('Actual scoring-module snapshot SHA differs')
        for rel,digest in scfg['source_hashes'].items():
            if score.sha(snapshot/rel)!=digest:
                raise ValueError('Actual official-source snapshot differs from frozen config')
        loaded=report.get('loaded_modules',{})
        from ..left_baseline import MODEL_FILES
        if not set(MODEL_FILES)<=set(loaded):
            raise ValueError('Missing actual official-model module provenance')
        for rel,row in loaded.items():
            if (row.get('sha256')!=scfg['source_hashes'].get(rel) or
                    Path(row['path'])!=(Path(scfg['source_root'])/rel)):
                raise ValueError('Actually loaded official module differs from declared source identity')
        expected_refs={}
        for role in ('CAL','DEV'):
            score.validate_bank(scfg['banks'][role],scene,role)
            for ref in scfg['banks'][role]:expected_refs[(scene,role,ref['run_id'],ref['variant'])]=ref
        seen=set()
        for row in report['banks']:
            ref=row['reference'];key=(ref['scenario'],ref['role'],ref['run_id'],ref['variant'])
            if key in seen or ref!=expected_refs.get(key) or key not in seal_rows:
                raise ValueError('Completed report bank reference differs, duplicates, or crosses scene')
            seen.add(key);sealed=seal_rows[key]
            expected_name=f"{ref['role']}_run{ref['run_id']:02d}_{ref['variant']}.npz"
            if (row['file']!=expected_name or sealed['reference']!=ref or row['sha256']!=sealed['sha256'] or
                    Path(sealed['path']).resolve()!=(report_path.parent/expected_name).resolve() or
                    row.get('raw_shape')!=[128,192,2] or row.get('dtype')!='float32'):
                raise ValueError('Seal reference→file→hash/shape mapping differs from actual score report')
            banks[key]=load_score_bank(sealed)
        if seen!=set(expected_refs):raise ValueError('Completed report omits required bank')
        cal=score.calibrate_frozen([dict(reference=expected_refs[(scene,'CAL',i,'N')],
            **{r:banks[(scene,'CAL',i,'N')][r] for r in score.READOUTS}) for i in range(8)],scene)
        if cal!=report.get('calibration'):raise ValueError('Frozen report CAL threshold provenance differs')
        provenance.extend([{'path':str(report_path),'sha256':item['sha256']},
                           {'path':str(config_path),'sha256':item['config']['sha256']}])
    if set(banks)!=expected:raise ValueError('All80 structured banks must pass before labels open')
    return banks,provenance


def load_score_bank(row):
    import numpy as np
    if score.sha(row['path']) != row['sha256']:
        raise ValueError('Frozen score bank bytes differ')
    with np.load(row['path'], allow_pickle=False) as stored:
        raw = stored['channel_scores']
        reduced = score.reduce_raw(raw)
        coordinates = score.coordinates()
        for key, expected in dict(reduced, **coordinates).items():
            if key not in stored or stored[key].dtype != expected.dtype or not np.array_equal(stored[key], expected):
                raise ValueError(f'Frozen score field/coordinate differs: {key}')
    return reduced


def validate_truth(truth, scene, rid):
    import numpy as np
    if (truth['scenario'], truth['role'], truth['run_id']) != (scene, 'DEV', rid):
        raise PermissionError('Truth contents disagree with frozen identity')
    start, end = truth['event_start'], truth['event_end']
    if (type(start) is not int or type(end) is not int or
            not 24*score.H <= start <= 92*score.H or end-start != 12*score.H):
        raise ValueError('Expected original 12H event interval')
    lag = np.asarray(truth['normal_lag'])
    if lag.shape != (score.N,) or not np.issubdtype(lag.dtype, np.integer) or (lag < 0).any():
        raise ValueError('Complete integer normal-lag path required')
    if set(truth['labels']) != {'A', 'N', 'S0', 'SB'}:
        raise ValueError('All four original variant labels required')
    for v in ('A', 'N', 'S0', 'SB'):
        labels = np.asarray(truth['labels'][v])
        oracle = np.zeros(score.N, dtype=np.int8)
        if v == 'A': oracle[start:end] = 1
        if v == 'SB': oracle[start:start+score.H] = 1; oracle[end-score.H:end] = 1
        if labels.shape != oracle.shape or not np.array_equal(labels, oracle):
            raise ValueError('Original event/sham label semantics differ')
    return truth


def load_truth(ref):
    if score.sha(ref['path']) != ref['sha256']:
        raise ValueError('Frozen truth SHA mismatch')
    return validate_truth(json.loads(Path(ref['path']).read_text()), ref['scenario'], ref['run_id'])


def binary_counts(alarms, labels, mask):
    import numpy as np
    alarms, labels, mask = np.asarray(alarms, bool), np.asarray(labels), np.asarray(mask, bool)
    if alarms.shape != (128,) or labels.shape != (128,) or mask.shape != (128,):
        raise ValueError('Complete endpoint counts required')
    tp = int(np.count_nonzero(mask & alarms & (labels == 1)))
    fp = int(np.count_nonzero(mask & alarms & (labels == 0)))
    fn = int(np.count_nonzero(mask & ~alarms & (labels == 1)))
    tn = int(np.count_nonzero(mask & ~alarms & (labels == 0)))
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'alarm_points': tp+fp,
            'precision_numerator': tp, 'precision_denominator': tp+fp,
            'recall_numerator': tp, 'recall_denominator': tp+fn,
            'FPR_numerator': fp, 'FPR_denominator': fp+tn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}


def cpu_core_ap(rows):
    """Pinned original CSV only; retain NA and require each of 8 run IDs."""
    expected = {(s, m, r, i) for s in ('LIN', 'WEAK') for i in range(8)
                for m, r in [('MULTI191','last'), ('PEAK191','last'), ('SYNC191','last'),
                             ('MULTI64','mean128'), ('SYNC64','mean128')]}
    ap = {}
    for row in rows:
        if row['scenario'] not in ('LIN', 'WEAK') or row['variant'] != 'A' or row['layer'] != 'CORE':
            continue
        key = (row['scenario'], row['model'], row['readout'], int(row['run_id']))
        if key not in expected: continue
        if key in ap: raise ValueError('Duplicate original CPU CORE AP row')
        raw = row['AP']; value = None if raw in ('', None, 'None') else float(raw)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Invalid original CPU AP; NA cannot become zero')
        ap[key] = value
    if set(ap) != expected:
        raise ValueError('Missing one or more required CPU CORE paired run identities')
    return ap


def macro_summary(layers, events, matched):
    """Macro AP/ROC/FPR plus explicit micro count ratios; never pooled AP."""
    def groups(rows, fields):
        result = {}
        for row in rows:
            result.setdefault(tuple(row[f] for f in fields), []).append(row)
        return result
    output = {'layers': [], 'events': [], 'matched_normal': []}
    fields = ('scenario', 'readout', 'variant', 'layer')
    for key, rows in groups(layers, fields).items():
        if sorted(r['run_id'] for r in rows) != list(range(8)):
            raise ValueError('Macro summary requires all eight independent runs')
        entry = dict(zip(fields, key)); entry['runs'] = 8
        for metric in ('AP', 'ROC_AUC', 'normal_FPR'):
            values = [r[metric] for r in rows if r[metric] is not None]
            entry[metric+'_macro'] = sum(values)/len(values) if values else None
            entry[metric+'_valid_runs'] = len(values)
            entry[metric+'_min'] = min(values) if values else None
            entry[metric+'_max'] = max(values) if values else None
        for count in ('points', 'positive_points', 'normal_points', 'TP', 'FP', 'FN', 'TN', 'alarm_points'):
            entry[count] = sum(r[count] for r in rows)
        entry['normal_FPR_micro_numerator'] = entry['FP']
        entry['normal_FPR_micro_denominator'] = entry['normal_points']
        entry['normal_FPR_micro'] = entry['FP']/entry['normal_points'] if entry['normal_points'] else None
        output['layers'].append(entry)
    fields = ('scenario', 'readout', 'variant')
    for key, rows in groups(events, fields).items():
        expected_events = 8 if key[2] == 'A' else 16
        if (len(rows) != expected_events or
                any(sum(r['run_id'] == i for r in rows) != expected_events//8 for i in range(8))):
            raise ValueError('Missing original event or run in event summary')
        hits = sum(int(r['hit']) for r in rows)
        delays = [r['delay_points'] for r in rows if r['hit']]
        output['events'].append(dict(zip(fields,key), events=len(rows), hits=hits,
            event_recall_numerator=hits, event_recall_denominator=len(rows), event_recall=hits/len(rows),
            right_censored=sum(int(r['right_censored']) for r in rows),
            delay_hit_only_mean=sum(delays)/len(delays) if delays else None,
            delay_hit_only_count=len(delays), delay_unit='run-index points; retrospective window localization'))
    for key, rows in groups(matched, fields).items():
        if sorted(r['run_id'] for r in rows) != list(range(8)):
            raise ValueError('Matched interval summary requires all eight runs')
        hits = sum(int(r['matched_normal_interval_hit']) for r in rows)
        output['matched_normal'].append(dict(zip(fields,key), intervals=8, false_hits=hits,
            false_hit_numerator=hits, false_hit_denominator=8, false_hit_rate=hits/8))
    return output


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value*1024)
    except ImportError:
        import psutil
        mem = psutil.Process().memory_info()
        return int(getattr(mem, 'peak_wset', mem.rss))


def evaluate(config_path, release_path):
    import numpy as np
    started = time.monotonic()
    cfg, seal = require_release(config_path, release_path)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise PermissionError('CPU evaluation requires CUDA_VISIBLE_DEVICES empty')
    out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    report = {'status':'Running', 'started_unix':time.time(), 'seed':2025,
              'truth_files_read':[], 'forward_calls':0, 'GPU_budget':0,
              'config_sha256':score.sha(config_path), 'inputs':[],
              'versions':{'python':sys.version, 'numpy':np.__version__},
              'scope':'single-seed synthetic-control DEV exploration; not independent confirmation'}
    def check_budget():
        if time.monotonic()-started > 300: raise TimeoutError('CPU300s budget exhausted; no retry')
        if peak_rss_bytes() > 2*1024**3: raise MemoryError('CPU2GiB peak RSS budget exhausted; no retry')
    try:
        # Structural/provenance failure now produces a Failed receipt with zero
        # truth reads. No metric-table body or truth is opened before this gate.
        banks,provenance=validate_evaluation_inputs(cfg,seal)
        report['inputs'].extend(provenance)
        check_budget()
        import sklearn
        report['versions']['scikit-learn'] = sklearn.__version__
        for source, name in [(config_path,'config.json'), (release_path,'release.json'),
                             (cfg['score_seal']['path'],'score_seal.json'),
                             (__file__,'evaluation_source.py'), (score.__file__,'score_source.py'),
                             (wl.__file__,'weak_lag_source.py')]:
            shutil.copyfile(source,out/name)
            report['inputs'].append({'path':str(source),'sha256':score.sha(source)})
        check_budget()
        if score.sha(cfg['cpu_all_layers']['path']) != cfg['cpu_all_layers']['sha256']:
            raise ValueError('Pinned CPU all_layers.csv differs')
        shutil.copyfile(cfg['cpu_all_layers']['path'],out/'original_cpu_all_layers.csv')
        if score.sha(out/'original_cpu_all_layers.csv') != cfg['cpu_all_layers']['sha256']:
            raise ValueError('Original CPU metric-table snapshot SHA differs')
        with (out/'original_cpu_all_layers.csv').open(newline='') as handle:
            ap = cpu_core_ap(list(csv.DictReader(handle)))
        report['inputs'].append(cfg['cpu_all_layers'])
        for row in seal['banks']:
            report['inputs'].append({'path':row['path'],'sha256':row['sha256']})
            check_budget()
        calibration = {}
        for scene in ('LIN', 'WEAK'):
            records = [dict(reference=next(r['reference'] for r in seal['banks'] if
                (r['reference']['scenario'],r['reference']['role'],r['reference']['run_id'],r['reference']['variant']) ==
                (scene,'CAL',i,'N')), **{r:banks[(scene,'CAL',i,'N')][r] for r in score.READOUTS}) for i in range(8)]
            calibration[scene] = score.calibrate_frozen(records,scene)
        dump(out/'fixed_calibration.json',calibration)
        layers, events, matched, controls = [], [], [], []
        for ref in cfg['truths']:
            check_budget()
            # All 80 score banks were checked above; only now open a pinned truth.
            report['truth_files_read'].append({'path':ref['path'],'sha256':ref['sha256']})
            truth = load_truth(ref); scene,rid=ref['scenario'],ref['run_id']
            layer = wl.strata(np.asarray(truth['normal_lag']),truth['event_start'],truth['event_end'])
            ctl = score.control_equivalence(banks[(scene,'DEV',rid,'N')],banks[(scene,'DEV',rid,'S0')],
                                             banks[(scene,'DEV',rid,'SB')],layer['CORE'])
            controls.append(dict(scenario=scene,run_id=rid,**ctl))
            dump(out/f'{scene}_run{rid:02d}_strata.json',layer)
            for variant in ('A','N','S0','SB'):
                result = score.evaluate_fixed(banks[(scene,'DEV',rid,variant)],truth,variant,
                                              calibration[scene],seal_path=cfg['score_seal']['path'])
                for readout,detail in result.items():
                    identity=dict(scenario=scene,model='LEFT',readout=readout,run_id=rid,variant=variant,seed=2025)
                    for row in detail['layers']:
                        counts=binary_counts(detail['alarm'],detail['label_at_endpoint'],layer[row['layer']])
                        if counts['FPR_denominator'] != row['normal_points']:
                            raise ValueError('Metric normal denominator disagrees')
                        layers.append(dict(identity,**row,**counts,threshold=calibration[scene][readout]['threshold']))
                        if variant=='A' and row['layer']=='CORE': ap[(scene,'LEFT',readout,rid)]=row['AP']
                    events.extend(dict(identity,**row) for row in detail['events'])
                    if detail['matched_normal_interval_hit'] is not None:
                        matched.append(dict(identity,matched_normal_interval_hit=detail['matched_normal_interval_hit'],
                                            start=truth['event_start'],end_exclusive=truth['event_end'],length=12*score.H))
                    np.savez_compressed(out/f'{scene}_run{rid:02d}_{variant}_{readout}_evaluation.npz',
                        score=banks[(scene,'DEV',rid,variant)][readout],
                        y_score=banks[(scene,'DEV',rid,variant)]['y_'+readout],
                        alarm=detail['alarm'],labels=detail['label_at_endpoint'],endpoint=layer['endpoint'],
                        threshold=calibration[scene][readout]['threshold'])
                check_budget()
        bootstrap,draws = score.paired_core_ap_comparisons(ap)
        if (len(layers),len(events),len(matched),len(controls)) != (896,96,64,16):
            raise ValueError('Complete strata/event/matched/control row coverage differs')
        write_csv(out/'all_layers.csv',layers); write_csv(out/'all_events.csv',events)
        write_csv(out/'matched_normal_intervals.csv',matched)
        dump(out/'macro_summary.json',macro_summary(layers,events,matched))
        dump(out/'controls.json',controls); dump(out/'paired_bootstrap.json',bootstrap)
        np.save(out/'bootstrap_run_draws.npy',draws,allow_pickle=False)
        check_budget()
        for ref in cfg['truths']:
            if score.sha(ref['path']) != ref['sha256']: raise ValueError('Truth changed during evaluation')
        if score.sha(cfg['cpu_all_layers']['path']) != cfg['cpu_all_layers']['sha256']:
            raise ValueError('Original CPU table changed during evaluation')
        score.require_evaluation_seal(cfg['score_seal']['path'])
        if score.sha(cfg['score_seal']['path']) != cfg['score_seal']['sha256']:
            raise ValueError('Score seal changed during evaluation')
        report.update(status='Completed',truth_count=16,score_bank_count=80,comparisons=12,
                      layer_rows=896,event_rows=96,matched_interval_rows=64,control_rows=16,
                      bootstrap_repeats=10000,all_runs_retained=True,threshold_search=False,
                      label_permission='only after complete80-bank seal',point_adjustment=False)
    except BaseException as exc:
        report.update(status='Failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        elapsed=time.monotonic()-started
        report.update(wall_seconds=elapsed,ended_unix=time.time(),process_cpu_job_hours=elapsed/3600)
        try: report['peak_rss_bytes']=peak_rss_bytes()
        except Exception as exc: report['peak_rss_unavailable']=type(exc).__name__
        dump(out/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--release')
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    if args.execute:
        if not args.release: parser.error('--execute requires exact root --release')
        evaluate(args.config,args.release)
    else:
        cfg=validate_config(json.loads(Path(args.config).read_text()))
        print(json.dumps({'status':'PREVIEW_ONLY','config_status':cfg['status'],'truth_references':16,
                          'expected_frozen_banks':80,'body_data_read':False,'forward_calls':0,
                          'bootstrap_repeats':10000,'CPU_seconds':300,'RSS_GiB':2,'GPU_budget':0}))


if __name__=='__main__': main()
