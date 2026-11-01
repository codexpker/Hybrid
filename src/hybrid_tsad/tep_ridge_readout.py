"""Fixed Ridge/ZMAX normal-CAL inference; no fit, label or fault-data path."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import time
from .tep_normal_ridge import feature_order,COLUMNS,FAMILIES

METHODS=FAMILIES+('ZMAX',)
ROOT='/home/a312/tsad_research/'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def indexed_design(z,target,spec):
    """Pure single-run geometry, never masquerades CAL as training role."""
    import numpy as np
    if target not in range(3):raise ValueError('Three fixed targets only')
    if z.shape!=(500,3) or z.dtype!=np.float64 or not np.isfinite(z).all():raise ValueError('Finite float64 standardized run required')
    for f in spec:
        if (f['channel'] not in range(3) or f['name']!=COLUMNS[f['channel']]
                or f['lag'] not in range(65) or (f['channel']==target and f['lag']==0)):
            raise PermissionError('No current-target, future or unknown channel inputs')
    t=np.arange(64,500,dtype=np.int64)
    return np.column_stack([z[t-f['lag'],f['channel']] for f in spec]),z[t,target],t


def validate_selection(selection):
    if selection['chosen']!={f:100. for f in FAMILIES}:raise PermissionError('Do not change accepted selected alphas')
    features=selection['features']
    if set(features)!={f'{f}_target{t}' for f in FAMILIES for t in range(3)}:raise ValueError('Exact twelve stored feature lists required')
    for family in FAMILIES:
        for target in range(3):
            spec=features[f'{family}_target{target}'];peaks={}
            if family=='PEAK':
                if len(spec)!=66:raise ValueError('Exact PEAK feature count')
                for f in spec[64:]:
                    if f['channel']==target or f['lag'] not in range(1,65):raise ValueError('Fixed past-only PEAK feature')
                    peaks[target,f['channel']]=f['lag']
            if spec!=feature_order(family,target,peaks):raise ValueError('Stored feature order differs from original family geometry')
    return features


def aligned_readouts(raw,*,first_raw_t):
    import numpy as np
    if first_raw_t not in (0,64) or raw.shape!=(500-first_raw_t,) or not np.isfinite(raw).all():raise ValueError('Complete single-run score axis required')
    ends=np.arange(99,500,dtype=np.int64)
    last=raw[ends-first_raw_t].copy()
    mean32=np.asarray([raw[t-31-first_raw_t:t+1-first_raw_t].mean() for t in ends],dtype=np.float64)
    return dict(end_index_0based=ends,last=last,mean32=mean32)


def score_run(raw_values,scaler,selection,weights,*,role,run_id):
    import numpy as np
    if role!='CAL' or run_id not in range(81,101):raise PermissionError('Only normal CAL81..100; no fit/query role')
    features=validate_selection(selection)
    expected_fit=[['TEP_REACTOR','FIT',i] for i in list(range(1,21))+list(range(51,61))]
    if (scaler['fit_ids']!=expected_fit or scaler['fit_points']!=15000 or scaler['ddof']!=0
            or scaler['SELECT_used_for_fit'] is not False or tuple(scaler['columns'])!=COLUMNS):raise PermissionError('Accepted FIT-only scaler required')
    mean=np.asarray(scaler['mean'],np.float64);scale=np.asarray(scaler['scale'],np.float64)
    if raw_values.shape!=(500,3) or raw_values.dtype!=np.float64 or not np.isfinite(raw_values).all():raise ValueError('Original raw float64 normal run required')
    if mean.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid fixed scaler')
    z=(raw_values-mean)/scale;arrays=dict(original_values=raw_values,FIT_z=z,prediction_t=np.arange(64,500,dtype=np.int64))
    for family in FAMILIES:
        predictions=[]
        for target in range(3):
            spec=features[f'{family}_target{target}'];key=f'{family}_a100_target{target}'
            coef=np.asarray(weights[key+'_coef']);intercept=np.asarray(weights[key+'_intercept'])
            if coef.dtype!=np.float64 or coef.shape!=(len(spec),) or intercept.dtype!=np.float64 or intercept.shape!=() or not np.isfinite(coef).all() or not np.isfinite(intercept):raise ValueError('Saved selected coefficients/intercept shape or dtype differs')
            x,y,t=indexed_design(z,target,spec);predictions.append(x@coef+float(intercept))
        pred=np.column_stack(predictions);signed=pred-z[64:];absolute=np.abs(signed)
        score=absolute.mean(axis=1)
        if not np.isfinite(pred).all() or not np.isfinite(score).all():raise FloatingPointError('Nonfinite fixed prediction/residual')
        arrays.update({family+'_prediction_z':pred,family+'_signed_residual_z':signed,
                       family+'_absolute_residual_z':absolute,family+'_point_score':score})
        for key,value in aligned_readouts(score,first_raw_t=64).items():arrays[family+'_'+key]=value
    zmax=np.abs(z).max(axis=1);arrays['ZMAX_point_score']=zmax
    for key,value in aligned_readouts(zmax,first_raw_t=0).items():arrays['ZMAX_'+key]=value
    return arrays


def calibrate(banks,*,coefficient_sha256,scaler_sha256):
    import numpy as np
    if len(banks)!=20 or [b['run_id'] for b in banks]!=list(range(81,101)):raise ValueError('All20 distinct ordered normal runs required')
    dist={};thresholds={}
    for bank in banks:
        if bank['role']!='CAL' or bank['coefficient_sha256']!=coefficient_sha256 or bank['scaler_sha256']!=scaler_sha256:raise PermissionError('Do not mix role/model/scaler pools')
        if set(bank['methods'])!=set(METHODS):raise ValueError('All five distinct methods required')
        for method,readouts in bank['methods'].items():
            if readouts['method']!=method:raise PermissionError('Do not mix different method scores')
            if not np.array_equal(readouts['end_index_0based'],np.arange(99,500)):raise ValueError('Reset original endpoint axis in each run')
            for key in ('last','mean32'):
                v=readouts[key]
                if v.shape!=(401,) or v.dtype!=np.float64 or not np.isfinite(v).all():raise ValueError('Complete finite float64 readout required')
    for method in METHODS:
        for key in ('last','mean32'):
            name=method+'_'+key;dist[name]=np.concatenate([b['methods'][method][key] for b in banks])
            thresholds[name]=float(np.sort(dist[name],kind='stable')[7940])
    dist.update(run_id=np.repeat(np.arange(81,101),401),end_index_0based=np.tile(np.arange(99,500),20))
    return dist,dict(n=8020,rank_1based=7941,comparison='strict_greater',thresholds=thresholds,
        coefficient_sha256=coefficient_sha256,scaler_sha256=scaler_sha256,
        limitation='Normal point calibration, not event FPR or fault detection evidence')


def preview(path):
    c=read_json(path)
    if (c.get('operation')!='TEP_RIDGE_NORMAL_CALIBRATION' or c.get('run_ids')!=list(range(81,101))
            or c.get('methods')!=list(METHODS) or c.get('selected_alpha')!={f:100. for f in FAMILIES}
            or c.get('max_seconds')!=120 or c.get('max_AS_GiB')!=4
            or c.get('quantile_rank_1based')!=7941 or c.get('comparison')!='strict_greater'
            or c.get('output')!=ROOT+'results/tep_ridge_normal_CAL_20261102_v1'):
        raise PermissionError('Only fixed normal Ridge/ZMAX calibration')
    for key in ('ridge_acceptance','selection','scaler','coefficients','normal_metadata','contract'):
        ref=c.get(key)
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise PermissionError('Exact reference required')
        p=ref['path']
        if '\\' in p or not PurePosixPath(p).is_absolute() or not p.startswith(ROOT) or '..' in PurePosixPath(p).parts:raise PermissionError('POSIX workspace references only')
    return c


def execute(path,release_path):
    c=preview(path);release=read_json(release_path)
    expected=dict(status='PRIMARY_RELEASED_TEP_RIDGE_NORMAL_CALIBRATION',approved_by='root',
        config_sha256=sha(path),helper_hashes=c['helper_hashes'],allow_normal_CAL81_100=True,
        allow_FIT_SELECT=False,allow_fit=False,allow_fault_query=False,allow_labels=False)
    if any(release.get(k)!=v for k,v in expected.items()):raise PermissionError('Exact CAL-only root release before parameter/array body')
    if set(c['helper_hashes'])!={'__init__.py','tep_normal_ridge.py','tep_ridge_readout.py'}:raise ValueError('Exact source closure')
    for name,digest in c['helper_hashes'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('Source identity changed')
    import os,resource
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')):raise PermissionError('Single-thread CPU-only environment')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    def ref(key,json_body=True):
        r=c[key]
        if sha(r['path'])!=r['sha256']:raise ValueError('Reference changed: '+key)
        return read_json(r['path']) if json_body else Path(r['path'])
    receipt=ref('ridge_acceptance');selection=ref('selection');scaler=ref('scaler');metadata=ref('normal_metadata');ref('contract',False)
    if (receipt['status']!='PRIMARY_ACCEPTED_LIMITED_NORMAL_RIDGE_FIT_SELECT'
            or receipt['coefficients_sha256']!=c['coefficients']['sha256'] or receipt['model_selection_sha256']!=c['selection']['sha256']
            or receipt['scaler_sha256']!=c['scaler']['sha256'] or receipt['selected_alpha']!={f:100. for f in FAMILIES}):raise PermissionError('Accepted selected model/scaler identity mismatch')
    validate_selection(selection)
    coefficients=ref('coefficients',False)
    rows=[r for r in metadata['records'] if r['role']=='cal']
    if [r['simulationRun'] for r in rows]!=list(range(81,101)):raise ValueError('Original CAL81..100 metadata required')
    for r in rows:
        if r['faultNumber']!=0 or r['shape']!=[500,3] or r['dtype']!='float64' or tuple(r['column_names'])!=COLUMNS or r['raw_sha256']!='12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940':raise PermissionError('Original normal source identity required')
    import numpy as np
    from threadpoolctl import threadpool_limits
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report=dict(status='RUNNING',config_sha256=sha(path),release_sha256=sha(release_path),no_fit=True,GPU_hours=0,banks=[])
    pools=[]
    try:
        selected_keys=[f'{family}_a100_target{target}_{field}' for family in FAMILIES for target in range(3) for field in ('coef','intercept')]
        with np.load(coefficients,allow_pickle=False) as w:weights={k:w[k].copy() for k in selected_keys}
        selected_parameters={f'{family}_a100_target{target}_{field}':weights[f'{family}_a100_target{target}_{field}']
            for family in FAMILIES for target in range(3) for field in ('coef','intercept')}
        np.savez(out/'selected_parameters.npz',**selected_parameters)
        with threadpool_limits(limits=1):
            for row in rows:
                p=Path(row['array_path'])
                expected_path=ROOT+'results/tep_reactor_normal_readiness_20261101_v1/derived/cal/run%03d.npy'%row['simulationRun']
                if str(p)!=expected_path or sha(p)!=row['sha256']:raise ValueError('Exact normal CAL array differs')
                values=np.load(p,allow_pickle=False)
                arrays=score_run(values,scaler,selection,weights,role='CAL',run_id=row['simulationRun'])
                bank_path=out/('run%03d.npz'%row['simulationRun']);np.savez(bank_path,**arrays)
                methods={f:dict(method=f,**{key:arrays[f+'_'+key] for key in ('last','mean32','end_index_0based')}) for f in METHODS}
                pools.append(dict(run_id=row['simulationRun'],role='CAL',coefficient_sha256=c['coefficients']['sha256'],scaler_sha256=c['scaler']['sha256'],methods=methods))
                report['banks'].append(dict(run_id=row['simulationRun'],array_source=row,path=str(bank_path),sha256=sha(bank_path)))
                if time.monotonic()-start>120:raise TimeoutError('120 second contract exceeded')
        dist,calibration=calibrate(pools,coefficient_sha256=c['coefficients']['sha256'],scaler_sha256=c['scaler']['sha256'])
        np.savez(out/'calibration_scores.npz',**dist);calibration['distribution_sha256']=sha(out/'calibration_scores.npz')
        (out/'calibration.json').write_text(json.dumps(calibration,indent=2,allow_nan=False),encoding='utf-8')
        report.update(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',calibration=calibration,
            feature_order=selection['features'],selected_alpha=selection['chosen'],coefficient_source=c['coefficients'],scaler_source=c['scaler'])
    except BaseException:
        import traceback
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc());raise
    finally:
        report.update(seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--release');a=p.parse_args()
    if a.execute:
        if not a.release:raise PermissionError('Root release required')
        print(json.dumps(execute(a.config,a.release),indent=2))
    else:preview(a.config);print('PREPARATION_ONLY_NO_PARAMETER_OR_ARRAY_BODY')

if __name__=='__main__':main()
