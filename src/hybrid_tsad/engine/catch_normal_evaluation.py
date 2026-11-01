"""CATCH-only fixed evaluation helpers. No model or truth IO at import time.

The 80 physical banks and their report/config provenance must pass before a
separately released caller may open truth or historical metric-table bodies.
LEFT utilities below are pure fixed-grid/count/summary kernels only.
"""
import json
from pathlib import Path
import re

from .. import catch_normal_scores as score
from .. import weak_lag as wl
from . import left_normal_evaluation as kernels
from .left_normal_evaluation import (binary_counts, macro_summary,
    validate_truth_references, validate_truth, cpu_core_ap, write_csv, dump,
    load_truth, peak_rss_bytes)

OPERATION = 'CATCH_WL_C2_LABEL_EVALUATION'


def load_score_bank(row):
    """Recompute coordinates/readouts; native point version stays canonical."""
    import numpy as np
    if score.base.sha(row['path']) != row['sha256']:
        raise ValueError('Frozen raw CATCH NPZ differs')
    with np.load(row['path'], allow_pickle=False) as stored:
        required = {'input', 'reconstruction', 'time_error', 'frequency_error',
                    'native_score', 'canonical_score', 'last', 'mean128', 'y_last', 'y_mean128'}
        if set(stored.files) != required | set(score.coordinates()):
            raise ValueError('Exact CATCH raw score schema required')
        arrays = {k: stored[k].copy() for k in stored.files}
    for key in ('input','reconstruction','time_error','frequency_error'):
        x = arrays[key]
        if x.shape != (128,192,2) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('Complete finite native float32 CATCH tensor required')
    for key in ('native_score','canonical_score','last','mean128','y_last','y_mean128'):
        shape = (24576,) if key=='native_score' else ((128,192) if key=='canonical_score' else (128,))
        if arrays[key].shape != shape or arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ValueError('Complete finite native float32 CATCH score required')
    if not np.array_equal(arrays['native_score'], arrays['canonical_score'].reshape(-1)):
        raise ValueError('Native continuous point score is not canonical')
    for key, expected in score.coordinates().items():
        if arrays[key].dtype != expected.dtype or not np.array_equal(arrays[key], expected):
            raise ValueError('Actual original coordinate differs: '+key)
    expected = dict(last=arrays['canonical_score'][:,-1],
                    mean128=arrays['canonical_score'][:,64:].mean(axis=1))
    y = arrays['time_error'][:,:,1]+np.float32(.5)*arrays['frequency_error'][:,:,1]
    expected.update(y_last=y[:,-1], y_mean128=y[:,64:].mean(axis=1))
    if any(not np.array_equal(arrays[k], v) for k,v in expected.items()):
        raise ValueError('Frozen readout differs from original canonical/diagnostic arrays')
    # Independent arithmetic diagnostic; never replace the Torch score/version.
    independent = arrays['time_error'].mean(-1)+np.float32(.5)*arrays['frequency_error'].mean(-1)
    if not np.allclose(independent, arrays['canonical_score'], rtol=2e-6, atol=2e-6):
        raise ValueError('Independent channelmean formula disagrees with native score')
    time_error = np.square(arrays['input']-arrays['reconstruction'])
    if not np.allclose(time_error, arrays['time_error'], rtol=2e-6, atol=2e-6):
        raise ValueError('Independent reconstruction MSE disagrees')
    return arrays


def validate_score_inputs(cfg, seal):
    """No truth/old metric table IO. Reject aliases before any raw NPZ opens."""
    expected = {(s,r,i,v) for s in ('LIN','WEAK') for r in ('CAL','DEV') for i in range(8)
                for v in (('N',) if r=='CAL' else ('A','N','S0','SB'))}
    if (seal.get('status')!='APPROVED' or seal.get('approved_by')!='root'
        or seal.get('operation')!=OPERATION or seal.get('allow_truth') is not True
        or seal.get('source_manifest_sha256')!=score.MANIFEST_SHA
        or seal.get('score_module_sha256')!=score.base.sha(score.__file__)):
        raise PermissionError('Root CATCH-specific raw80 seal required')
    rows={}; paths=set(); inodes=set()
    for row in seal['banks']:
        ref=row['reference'];scene,role=ref['scenario'],ref['role']
        from ..left_normal_scores import validate_reference
        validate_reference(ref,scene,role)
        key=(scene,role,ref['run_id'],ref['variant'])
        path=Path(row['path']).resolve(strict=True); stat=path.stat(); inode=(stat.st_dev,stat.st_ino)
        if (not path.is_file() or path.suffix!='.npz' or key in rows or str(path) in paths or inode in inodes):
            raise PermissionError('Every bank needs unique identity/path/physical NPZ; aliases denied')
        rows[key]=row; paths.add(str(path)); inodes.add(inode)
    if set(rows)!=expected or len(rows)!=80:
        raise PermissionError('Exact80 physical CATCH bank identities required')
    if [r['scenario'] for r in cfg['score_reports']]!=['LIN','WEAK']:
        raise PermissionError('Ordered two-scene CATCH score reports required')
    banks={}
    for item in cfg['score_reports']:
        scene=item['scenario']; rp=Path(item['path']).resolve(strict=True)
        cp=Path(item['config']['path']).resolve(strict=True)
        if (rp.name!='report.json' or cp.name!='config.json' or rp.parent!=cp.parent
            or score.base.sha(rp)!=item['sha256'] or score.base.sha(cp)!=item['config']['sha256']):
            raise ValueError('Completed CATCH report/config actual directory/hash differs')
        report=json.loads(rp.read_text()); scfg=score.preview(cp)
        if (report.get('status')!='Completed' or report.get('operation')!=score.OPERATION
            or report.get('label_files_read') is not False or report.get('unscored_points')!=0
            or report.get('config_sha256')!=item['config']['sha256'] or scfg.get('status')!='FROZEN'
            or scfg['scenario']!=scene or report.get('adapter_hashes')!=scfg['adapter_hashes']
            or scfg['adapter_hashes']['catch_normal_scores.py']!=score.base.sha(score.__file__)
            or report.get('state_before')!=report.get('state_after')
            or not re.fullmatch('[0-9a-f]{64}',report.get('state_before',''))
            or report.get('eval_gumbel_preserved') is not True or len(report.get('banks',[]))!=40):
            raise PermissionError('CATCH completed score identity/state evidence differs')
        for key in ('checkpoint','scaler'):
            if report.get(key)!=scfg[key] or scfg[key]['sha256']!=item[key+'_sha256']:
                raise ValueError('CATCH best/scaler provenance differs')
        for key,value in score.DERIVED.items():
            if report.get('model_config',{}).get(key)!=value: raise ValueError('Derived native CATCH config differs')
        for key,value in score.PARAMS.items():
            if report.get('model_config',{}).get(key)!=value: raise ValueError('Native CATCH params differ')
        for name,digest in scfg['adapter_hashes'].items():
            if score.base.sha(rp.parent/'source'/name)!=digest: raise ValueError('Adapter snapshot differs')
        for rel,digest in scfg['source_hashes'].items():
            if score.base.sha(rp.parent/'source'/rel)!=digest: raise ValueError('Author snapshot differs')
        loaded=report.get('loaded_modules',{})
        if 'ts_benchmark.baselines.catch.CATCH' not in loaded: raise ValueError('Missing native imported-module evidence')
        for row in loaded.values():
            rel=str(Path(row['path']).relative_to(Path(scfg['source_root']))).replace('\\','/')
            if scfg['source_hashes'].get(rel)!=row['sha256']: raise ValueError('Actually imported source differs')
        refs={(scene,r,ref['run_id'],ref['variant']):ref for r in ('CAL','DEV') for ref in scfg['banks'][r]}
        seen=set(); paired_rng={}
        for row in report['banks']:
            ref=row['reference']; key=(ref['scenario'],ref['role'],ref['run_id'],ref['variant'])
            if key in seen or refs.get(key)!=ref or rows.get(key,{}).get('reference')!=ref:
                raise ValueError('Actual score reference duplicates/crosses declared bank')
            seen.add(key); sealed=rows[key]
            name=f"{ref['role']}_run{ref['run_id']:02d}_{ref['variant']}.npz"
            if (row['file']!=name or Path(sealed['path']).resolve()!=(rp.parent/name).resolve()
                or row['sha256']!=sealed['sha256'] or row.get('raw_shape')!=[128,192,2]
                or row.get('dtype')!='float32' or row.get('native_score_calls')!=1
                or row.get('model_forward_calls')!=1
                or row.get('seed')!=score.bank_seed(scene,ref['role'],ref['run_id'])):
                raise ValueError('Raw CATCH file/call/seed mapping differs')
            rng=(row.get('rng_before'),row.get('rng_after'))
            if any(not isinstance(r,dict) or set(r)!={'python','numpy','torch','cuda'}
                   or any(not re.fullmatch('[0-9a-f]{64}',v) for v in r.values()) for r in rng):
                raise ValueError('Complete actual RNG identity required')
            pair=(ref['role'],ref['run_id'])
            if pair in paired_rng and paired_rng[pair]!=rng:
                raise ValueError('Variant changed paired random stream')
            paired_rng[pair]=rng
            banks[key]=load_score_bank(sealed)
        if seen!=set(refs): raise ValueError('Incomplete actual CATCH scene report')
        calibration=score.calibrate_frozen([dict(reference=refs[(scene,'CAL',i,'N')],
            **{r:banks[(scene,'CAL',i,'N')][r] for r in score.READOUTS}) for i in range(8)],scene)
        if report.get('calibration')!=calibration: raise ValueError('CATCH CAL-only threshold differs')
    if set(banks)!=expected: raise ValueError('Incomplete80 raw bank validation')
    return banks


def paired_comparisons(ap):
    """Exactly16 predeclared comparisons, all8 runs, all scenes/readouts kept."""
    diffs={}
    for scene in ('LIN','WEAK'):
        specs=[('last',m,'last') for m in ('MULTI191','PEAK191','SYNC191')]
        specs += [('mean128',m,'mean128') for m in ('MULTI64','SYNC64')]
        specs += [('mean128','CATCH','last'),('last','LEFT','last'),('mean128','LEFT','mean128')]
        for own,other,readout in specs:
            pairs=[(ap[(scene,'CATCH',own,i)],ap[(scene,other,readout,i)]) for i in range(8)]
            diffs[f'{scene}_CATCH.{own}_minus_{other}.{readout}']=[a-b if a is not None and b is not None else None for a,b in pairs]
    return wl.paired_bootstrap(diffs,repeats=10000)


def evaluate_run(banks, truth, thresholds):
    """Pure post-seal kernel. Caller owns separate truth permission/reader."""
    import numpy as np
    scene,rid=truth['scenario'],truth['run_id']
    validate_truth(truth,scene,rid)
    strata=wl.strata(np.asarray(truth['normal_lag']),truth['event_start'],truth['event_end'])
    normal=banks[(scene,'DEV',rid,'N')]
    fields=('input','reconstruction','time_error','frequency_error','canonical_score','last','mean128','y_last','y_mean128')
    controls={k:dict(S0_full_equal=bool(np.array_equal(normal[k],banks[(scene,'DEV',rid,'S0')][k])),
        SB_CORE_equal=bool(np.array_equal(normal[k][strata['CORE']],banks[(scene,'DEV',rid,'SB')][k][strata['CORE']]))) for k in fields}
    if not all(all(v.values()) for v in controls.values()): raise ValueError('Fixed CATCH sham equivalence failed')
    layers=[]; events=[]; matched=[]
    for variant in ('A','N','S0','SB'):
        raw=banks[(scene,'DEV',rid,variant)]
        for readout in score.READOUTS:
            threshold=thresholds[readout]['threshold']
            detail=wl.evaluate_readout(raw[readout],np.asarray(truth['labels'][variant]),strata,
                threshold,truth['event_start'],truth['event_end'],variant)
            identity=dict(scenario=scene,model='CATCH',readout=readout,run_id=rid,variant=variant,seed=2025)
            for row in detail['layers']:
                counts=binary_counts(detail['alarm'],detail['label_at_endpoint'],strata[row['layer']])
                if counts['FPR_denominator']!=row['normal_points']: raise ValueError('Normal denominator differs')
                layers.append(dict(identity,**row,**counts,threshold=threshold))
            events.extend(dict(identity,**row) for row in detail['events'])
            if detail['matched_normal_interval_hit'] is not None:
                matched.append(dict(identity,matched_normal_interval_hit=detail['matched_normal_interval_hit'],
                    start=truth['event_start'],end_exclusive=truth['event_end'],length=2304))
    return dict(layers=layers,events=events,matched=matched,controls=controls)


def validate_config(cfg):
    if (cfg.get('operation')!=OPERATION or cfg.get('source_manifest_sha256')!=score.MANIFEST_SHA
        or cfg.get('seed')!=2025 or cfg.get('bootstrap_repeats')!=10000
        or cfg.get('max_seconds')!=300 or cfg.get('max_rss_GiB')!=2 or cfg.get('gpu_budget')!=0):
        raise PermissionError('Fixed CATCH CPU protocol required')
    validate_truth_references(cfg['truths'])
    for key in ('score_seal','cpu_all_layers','left_all_layers'):
        if not re.fullmatch('[0-9a-f]{64}',cfg[key]['sha256']):
            raise PermissionError('Actual frozen seal/table identity required; no pending SHA execution')
    return cfg


def require_release(config_path, release_path):
    cfg=validate_config(json.loads(Path(config_path).read_text()))
    release=json.loads(Path(release_path).read_text())
    if (cfg.get('status')!='FROZEN' or release.get('status')!='APPROVED'
        or release.get('approved_by')!='root' or release.get('operation')!=OPERATION
        or release.get('allow_truth') is not True or release.get('config_sha256')!=score.base.sha(config_path)
        or release.get('module_sha256')!=score.base.sha(__file__)
        or release.get('score_module_sha256')!=score.base.sha(score.__file__)
        or release.get('weak_lag_module_sha256')!=score.base.sha(wl.__file__)
        or release.get('left_evaluation_kernel_sha256')!=score.base.sha(kernels.__file__)):
        raise PermissionError('Separate root CATCH label release required')
    if score.base.sha(cfg['score_seal']['path'])!=cfg['score_seal']['sha256']:
        raise ValueError('Raw80 seal identity differs')
    return cfg,json.loads(Path(cfg['score_seal']['path']).read_text())


def left_core_ap(rows):
    import math
    expected={(s,'LEFT',r,i) for s in ('LIN','WEAK') for r in ('last','mean128') for i in range(8)}
    ap={}
    for row in rows:
        if row['variant']!='A' or row['layer']!='CORE':continue
        key=(row['scenario'],row['model'],row['readout'],int(row['run_id']))
        if key not in expected:continue
        if key in ap:raise ValueError('Duplicate original LEFT paired identity')
        value=None if row['AP'] in ('',None,'None') else float(row['AP'])
        if value is not None and (not math.isfinite(value) or not 0<=value<=1):raise ValueError('Invalid original LEFT AP')
        ap[key]=value
    if set(ap)!=expected:raise ValueError('All32 original LEFT same-readout pairs required')
    return ap


def evaluate(config_path, release_path):
    import csv
    import os
    import shutil
    import time
    import numpy as np
    cfg,seal=require_release(config_path,release_path)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only evaluation needs empty CVD')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report=dict(status='Running',operation=OPERATION,truth_files_read=[],forward_calls=0,GPU_budget=0,
        config_sha256=score.base.sha(config_path),module_sha256=score.base.sha(__file__),
        scope='single-seed synthetic-control DEV exploration; not independent confirmation')
    def budget():
        if time.monotonic()-started>300:raise TimeoutError('CPU budget exhausted')
        if peak_rss_bytes()>2*1024**3:raise MemoryError('CPU RSS budget exhausted')
    try:
        # This is the only label-opening boundary: all actual80 physical raw
        # structures/configs/reports must pass before any truth/old CSV body.
        banks=validate_score_inputs(cfg,seal);budget()
        ap={}
        for key,parser in [('cpu_all_layers',cpu_core_ap),('left_all_layers',left_core_ap)]:
            ref=cfg[key]
            if score.base.sha(ref['path'])!=ref['sha256']:raise ValueError('Pinned historical table differs')
            with Path(ref['path']).open(newline='') as handle:ap.update(parser(list(csv.DictReader(handle))))
            shutil.copyfile(ref['path'],out/(key+'.csv'))
        calibration={}
        for scene in ('LIN','WEAK'):
            calibration[scene]=score.calibrate_frozen([dict(reference=next(row['reference'] for row in seal['banks']
                if (row['reference']['scenario'],row['reference']['role'],row['reference']['run_id'],row['reference']['variant'])==(scene,'CAL',i,'N')),
                **{r:banks[(scene,'CAL',i,'N')][r] for r in score.READOUTS}) for i in range(8)],scene)
        layers=[];events=[];matched=[];controls=[]
        for ref in cfg['truths']:
            budget();report['truth_files_read'].append(ref)
            truth=load_truth(ref)
            result=evaluate_run(banks,truth,calibration[ref['scenario']])
            layers.extend(result['layers']);events.extend(result['events']);matched.extend(result['matched'])
            controls.append(dict(scenario=ref['scenario'],run_id=ref['run_id'],checks=result['controls']))
        if (len(layers),len(events),len(matched),len(controls))!=(896,96,64,16):
            raise ValueError('Full896/96/64/16 coverage required')
        for row in layers:
            if row['variant']=='A' and row['layer']=='CORE':
                ap[(row['scenario'],'CATCH',row['readout'],row['run_id'])]=row['AP']
        bootstrap,draws=paired_comparisons(ap)
        write_csv(out/'all_layers.csv',layers);write_csv(out/'all_events.csv',events)
        write_csv(out/'matched_normal_intervals.csv',matched)
        dump(out/'fixed_calibration.json',calibration);dump(out/'controls.json',controls)
        dump(out/'macro_summary.json',macro_summary(layers,events,matched))
        dump(out/'paired_bootstrap.json',bootstrap);np.save(out/'bootstrap_run_draws.npy',draws,allow_pickle=False)
        for path,name in [(config_path,'config.json'),(release_path,'release.json'),
                          (cfg['score_seal']['path'],'score_seal.json'),(__file__,'evaluation_source.py'),
                          (score.__file__,'score_source.py'),(wl.__file__,'weak_lag_source.py')]:
            shutil.copyfile(path,out/name)
        budget();report.update(status='Completed',layers=len(layers),events=len(events),
            matched_intervals=len(matched),controls=len(controls),paired_comparisons=len(bootstrap),
            raw_seal_sha256=cfg['score_seal']['sha256'])
    except BaseException as exc:
        report.update(status='Failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,peak_rss_bytes=peak_rss_bytes())
        dump(out/'report.json',report)
    return report


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--release');parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.execute:
        if not args.release:parser.error('Separate root label release required')
        evaluate(args.config,args.release)
    else:
        cfg=json.loads(Path(args.config).read_text());validate_truth_references(cfg['truths'])
        print(json.dumps(dict(status='METADATA_PREVIEW_ONLY',operation=OPERATION,
            arrays_read=False,labels_read=False,metric_table_bodies_read=False)))


if __name__=='__main__':main()
