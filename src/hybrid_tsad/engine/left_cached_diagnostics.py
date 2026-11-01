"""Post-hoc diagnostics of cached LEFT scores; no model/features/truth reader."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from .statistics import ap_cache, weighted_ap
from .. import left_normal_scores as fixed

CHANNELS=('original_mean','x_only','y_only')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def endpoint_ap(labels, scores):
    """Unweighted tie-group AP via existing statistics; single-class remains NA."""
    y,s=np.asarray(labels),np.asarray(scores)
    if y.ndim!=1 or y.shape!=s.shape or not np.isfinite(s).all() or not np.isin(y,[0,1]).all():
        raise ValueError('Finite aligned binary labels and scores required')
    if not len(y) or len(np.unique(y))!=2:return None
    return weighted_ap(ap_cache(y,s,np.zeros(len(y),np.int64)),np.ones(1,np.int64))


def channel_readouts(raw):
    reduced=fixed.reduce_raw(raw)
    return {'last':{'original_mean':reduced['last'],'x_only':raw[:,-1,0],'y_only':raw[:,-1,1]},
            'mean128':{'original_mean':reduced['mean128'],'x_only':raw[:,64:,0].mean(axis=1),
                       'y_only':raw[:,64:,1].mean(axis=1)}}


def expected_inputs(cfg):
    result=[]
    for scene in ('LIN','WEAK'):
        for role in ('CAL','DEV'):
            for rid in range(8):
                for variant in (('N',) if role=='CAL' else ('A','N','S0','SB')):
                    result.append(dict(kind='raw',scenario=scene,role=role,run_id=rid,variant=variant,
                        path=str(Path(cfg['score_roots'][scene])/f'{role}_run{rid:02d}_{variant}.npz')))
    for scene in ('LIN','WEAK'):
        for rid in range(8):
            result.append(dict(kind='strata',scenario=scene,run_id=rid,
                path=str(Path(cfg['evaluation_root'])/f'{scene}_run{rid:02d}_strata.json')))
            for readout in fixed.READOUTS:
                result.append(dict(kind='evaluation',scenario=scene,run_id=rid,readout=readout,
                    path=str(Path(cfg['evaluation_root'])/f'{scene}_run{rid:02d}_A_{readout}_evaluation.npz')))
    return result


def validate_config(cfg):
    if (cfg['operation']!='LEFT_CACHED_CHANNEL_ACCUMULATION_DIAGNOSTICS' or
            cfg['scope']!='EXPLICIT_POST_HOC_EXPLANATION' or cfg['max_seconds']!=120 or
            cfg['max_rss_GiB']!=1 or cfg['gpu_budget']!=0):
        raise ValueError('Only fixed post-hoc CPU120s/1GiB/GPU0 diagnostics admitted')
    expected=expected_inputs(cfg)
    if len(cfg['inputs'])!=128:raise PermissionError('Exactly80 raw +16 strata +32 evaluation caches required')
    for declared,actual in zip(cfg['inputs'],expected):
        if {k:declared.get(k) for k in actual}!=actual or not isinstance(declared.get('sha256'),str) or len(declared['sha256'])!=64:
            raise PermissionError('Only enumerated cached score/strata/evaluation files admitted')
    return cfg


def rss_bytes():
    try:
        import resource
        v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(v if sys.platform=='darwin' else v*1024)
    except ImportError:
        import psutil
        m=psutil.Process().memory_info();return int(getattr(m,'peak_wset',m.rss))


def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def json_write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def analyze(config_path):
    started=time.monotonic();cfg=validate_config(json.loads(Path(config_path).read_text()))
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only: CUDA_VISIBLE_DEVICES must be empty')
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    report=dict(status='Running',scope=cfg['scope'],authorization=cfg['authorization'],
        started_unix=time.time(),inputs=[],model_or_forward_calls=0,feature_arrays_read=0,original_truth_read=0,
        thresholds_computed=0,ROC_computed=False,
        ROC_reason='Reused statistics module has AP only; no additional audited pure-NumPy ROC function used',
        version={'python':sys.version,'numpy':np.__version__},config_sha256=sha(config_path),
        source_sha256=sha(__file__),statistics_sha256=sha(Path(__file__).with_name('statistics.py')),
        fixed_score_module_sha256=sha(fixed.__file__))
    def budget():
        if time.monotonic()-started>120:raise TimeoutError('CPU120s diagnostic budget exhausted; no retry')
        if rss_bytes()>1024**3:raise MemoryError('CPU1GiB diagnostic peak RSS exhausted; no retry')
    try:
        shutil.copyfile(config_path,out/'config.json');source=out/'source';source.mkdir()
        for p in (Path(__file__),Path(__file__).with_name('statistics.py'),Path(fixed.__file__)):
            shutil.copyfile(p,source/p.name)
        cached={};layers={};evaluations={}
        for ref in cfg['inputs']:
            budget()
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Pinned cache changed before diagnostics')
            scene,rid=ref['scenario'],ref['run_id']
            if ref['kind']=='raw':
                with np.load(ref['path'],allow_pickle=False) as z:
                    raw=z['channel_scores'];reduced=fixed.reduce_raw(raw)
                    for k,v in dict(reduced,**fixed.coordinates()).items():
                        if z[k].dtype!=v.dtype or not np.array_equal(z[k],v):
                            raise ValueError('Saved raw map/readout/coordinates inconsistent')
                if ref['role']=='DEV' and ref['variant'] in ('A','N'):
                    cached[(scene,rid,ref['variant'])]=channel_readouts(raw)
            elif ref['kind']=='strata':
                layer=json.loads(Path(ref['path']).read_text())
                if (not np.array_equal(layer['endpoint'],fixed.coordinates()['endpoint']) or
                        not np.array_equal(layer['input_start'],fixed.coordinates()['window_start'])):
                    raise ValueError('Existing strata do not align with frozen endpoints')
                masks={k:np.asarray(layer[k]) for k in ('ALL','CORE')}
                if any(m.shape!=(128,) or m.dtype!=np.bool_ for m in masks.values()) or not masks['ALL'].all():
                    raise ValueError('Exact existing boolean128 strata masks required')
                layers[(scene,rid)]=masks
            else:
                with np.load(ref['path'],allow_pickle=False) as z:
                    labels=z['labels'];endpoint=z['endpoint'];original=z['score'];yscore=z['y_score']
                if (labels.shape!=(128,) or not np.isin(labels,[0,1]).all() or
                        not np.array_equal(endpoint,fixed.coordinates()['endpoint'])):
                    raise ValueError('Existing evaluation label endpoints are invalid')
                readout=ref['readout'];from_raw=cached[(scene,rid,'A')][readout]
                if not np.array_equal(original,from_raw['original_mean']) or not np.array_equal(yscore,from_raw['y_only']):
                    raise ValueError('Existing B evaluation differs from A cached scores')
                evaluations[(scene,rid,readout)]=labels
            report['inputs'].append(ref)
        ap_rows=[];delta_rows=[];accumulation=[]
        for scene in ('LIN','WEAK'):
            for rid in range(8):
                mask=layers[(scene,rid)]
                labels=evaluations[(scene,rid,'last')]
                if not np.array_equal(labels,evaluations[(scene,rid,'mean128')]):
                    raise ValueError('Readouts do not share the same frozen labels')
                positive_core=mask['CORE']&(labels==1);count=int(positive_core.sum())
                vectors={'endpoint':fixed.coordinates()['endpoint'][positive_core]}
                run_ap={}
                for readout in fixed.READOUTS:
                    for channel in CHANNELS:
                        a=cached[(scene,rid,'A')][readout][channel]
                        n=cached[(scene,rid,'N')][readout][channel]
                        for layer,select in mask.items():
                            yy,ss=labels[select],a[select];value=endpoint_ap(yy,ss)
                            run_ap[(readout,channel,layer)]=value
                            ap_rows.append(dict(scenario=scene,run_id=rid,readout=readout,channel=channel,
                                variant='A',layer=layer,points=int(select.sum()),positive_points=int(yy.sum()),
                                negative_points=int((yy==0).sum()),positive_fraction=float(yy.mean()) if len(yy) else None,
                                AP=value,NA_reason=None if value is not None else 'EMPTY_OR_SINGLE_CLASS'))
                        difference=a[positive_core].astype(np.float64)-n[positive_core].astype(np.float64)
                        vectors[readout+'_'+channel]=difference
                        delta_rows.append(dict(scenario=scene,run_id=rid,readout=readout,channel=channel,
                            layer='CORE_POSITIVE_ENDPOINTS',endpoints=count,
                            mean_A_minus_matched_N=float(difference.mean()) if count else None,
                            median_A_minus_matched_N=float(np.median(difference)) if count else None,
                            mean_A=float(a[positive_core].mean(dtype=np.float64)) if count else None,
                            mean_N=float(n[positive_core].mean(dtype=np.float64)) if count else None,
                            positive_differences=int((difference>0).sum()),negative_differences=int((difference<0).sum()),
                            zero_differences=int((difference==0).sum()),NA_reason=None if count else 'NO_CORE_POSITIVE_ENDPOINT'))
                np.savez_compressed(out/f'{scene}_run{rid:02d}_CORE_positive_paired_differences.npz',**vectors)
                for channel in CHANNELS:
                    for layer in ('ALL','CORE'):
                        last=run_ap[('last',channel,layer)];mean=run_ap[('mean128',channel,layer)]
                        accumulation.append(dict(scenario=scene,run_id=rid,channel=channel,layer=layer,
                            AP_last=last,AP_mean128=mean,AP_mean128_minus_last=mean-last if last is not None and mean is not None else None))
                budget()
        if (len(ap_rows),len(delta_rows),len(accumulation))!=(192,96,96):raise ValueError('Run coverage differs')
        summary=[]
        for scene in ('LIN','WEAK'):
            for readout in fixed.READOUTS:
                for channel in CHANNELS:
                    for layer in ('ALL','CORE'):
                        group=[r for r in ap_rows if (r['scenario'],r['readout'],r['channel'],r['layer'])==(scene,readout,channel,layer)]
                        values=[r['AP'] for r in group if r['AP'] is not None]
                        summary.append(dict(scenario=scene,readout=readout,channel=channel,layer=layer,runs=8,
                            valid_runs=len(values),AP_macro=float(np.mean(values)) if values else None,
                            AP_min=min(values) if values else None,AP_max=max(values) if values else None,
                            positive_fraction_macro=float(np.mean([r['positive_fraction'] for r in group]))))
        csv_write(out/'run_AP.csv',ap_rows);csv_write(out/'run_CORE_positive_deltas.csv',delta_rows)
        csv_write(out/'run_accumulation_AP_differences.csv',accumulation);csv_write(out/'AP_summary.csv',summary)
        for ref in cfg['inputs']:
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Pinned cache changed during diagnostics')
        budget();report.update(status='Completed',raw_banks_checked=80,strata_checked=16,evaluation_caches_checked=32,
            AP_rows=192,paired_delta_rows=96,accumulation_rows=96,all_runs_retained=True,
            inference='post-hoc explanation only; no model/readout/channel/threshold selection; not independent confirmation')
    except BaseException as exc:
        report.update(status='Failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report.update(wall_seconds=time.monotonic()-started,ended_unix=time.time())
        try:report['peak_rss_bytes']=rss_bytes()
        except Exception as exc:report['peak_rss_unavailable']=type(exc).__name__
        json_write(out/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.execute:analyze(args.config)
    else:
        validate_config(json.loads(Path(args.config).read_text()))
        print(json.dumps({'status':'PREVIEW_ONLY','cached_references':128,'body_data_read':False,
                          'scope':'EXPLICIT_POST_HOC_EXPLANATION'}))


if __name__=='__main__':main()
