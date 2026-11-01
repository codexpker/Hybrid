"""Frozen deterministic Ridge/ZMAX opaque query scoring; no fit or evaluation."""
import argparse,json,time
from pathlib import Path,PurePosixPath
from .tep_ridge_readout import validate_selection,sha,read_json,METHODS,FAMILIES,COLUMNS
from .normal_b1_readout import validate_query_manifest
ROOT='/home/a312/tsad_research/'


def score_values(values,scaler,selection,weights):
    """Pure500/960 numeric geometry. Production role/identity is checked outside."""
    import numpy as np
    n=len(values)
    if values.shape!=(n,3) or n not in (500,960) or values.dtype!=np.float64 or not np.isfinite(values).all():raise ValueError('Complete original float64 3-channel series')
    expected_fit=[['TEP_REACTOR','FIT',i] for i in list(range(1,21))+list(range(51,61))]
    if (scaler['fit_ids']!=expected_fit or scaler['fit_points']!=15000 or scaler['ddof']!=0
            or scaler['SELECT_used_for_fit'] is not False or tuple(scaler['columns'])!=COLUMNS):raise PermissionError('Original FIT-only scaler')
    mean=np.asarray(scaler['mean'],np.float64);scale=np.asarray(scaler['scale'],np.float64)
    if mean.shape!=(3,) or scale.shape!=(3,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():raise ValueError('Invalid fixed scaler')
    z=(values-mean)/scale;features=validate_selection(selection);t=np.arange(64,n,dtype=np.int64);ends=np.arange(99,n,dtype=np.int64)
    arrays=dict(original_values=values,FIT_z=z,prediction_t=t)
    def add_readouts(method,raw,offset):
        arrays[method+'_end_index_0based']=ends.copy();arrays[method+'_last']=raw[ends-offset].copy()
        arrays[method+'_mean32']=np.asarray([raw[e-31-offset:e+1-offset].mean() for e in ends],dtype=np.float64)
    for family in FAMILIES:
        preds=[]
        for target in range(3):
            spec=features[f'{family}_target{target}']
            for f in spec:
                if f['channel'] not in range(3) or f['lag'] not in range(65) or f['name']!=COLUMNS[f['channel']] or (f['channel']==target and f['lag']==0):raise PermissionError('No own-current/future/unknown feature')
            x=np.column_stack([z[t-f['lag'],f['channel']] for f in spec]);key=f'{family}_a100_target{target}'
            coef=np.asarray(weights[key+'_coef']);intercept=np.asarray(weights[key+'_intercept'])
            if coef.shape!=(len(spec),) or coef.dtype!=np.float64 or intercept.shape!=() or intercept.dtype!=np.float64 or not np.isfinite(coef).all() or not np.isfinite(intercept):raise ValueError('Selected parameter schema')
            preds.append(x@coef+float(intercept))
        pred=np.column_stack(preds);signed=pred-z[64:];absolute=np.abs(signed);raw=absolute.mean(1)
        if not np.isfinite(pred).all() or not np.isfinite(raw).all():raise FloatingPointError('Nonfinite fixed scores')
        arrays.update({family+'_prediction_z':pred,family+'_signed_residual_z':signed,family+'_absolute_residual_z':absolute,family+'_point_score':raw})
        add_readouts(family,raw,64)
    raw=np.abs(z).max(1);arrays['ZMAX_point_score']=raw;add_readouts('ZMAX',raw,0)
    return arrays


def preview(path):
    c=read_json(path)
    if (c.get('operation')!='TEP_RIDGE_FIXED_QUERY_READOUT' or c.get('entity_ids')!=['entity_%04d'%i for i in range(1,401)]
            or c.get('methods')!=list(METHODS) or c.get('max_seconds')!=300 or c.get('max_AS_GiB')!=4
            or c.get('output')!=ROOT+'results/tep_ridge_fixed_QUERY_20261102_v1'):
        raise PermissionError('Fixed deterministic query contract only')
    return c


def require_release(path,release_path,c):
    if c.get('binding_state')!='EXACT_REFERENCES_BOUND' or any(c.get(k) is None for k in ('input_acceptance','input_manifest','calibration_acceptance','calibration')):
        raise PermissionError('Input and RidgeCAL acceptance remain unbound')
    for key in ('ridge_acceptance','selection','scaler','coefficients','contract','input_acceptance','input_manifest','calibration_acceptance','calibration'):
        ref=c.get(key)
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise PermissionError('Exact refs required')
        p=ref['path']
        if '\\' in p or not p.startswith(ROOT) or '..' in PurePosixPath(p).parts or 'sealed_identity' in PurePosixPath(p).parts:raise PermissionError('No sealed/relative/nonworkspace refs')
    r=read_json(release_path)
    exact=dict(status='PRIMARY_RELEASED_TEP_RIDGE_FIXED_QUERY_READOUT',approved_by='root',config_sha256=sha(path),
        helper_hashes=c['helper_hashes'],allow_opaque_query_inputs=True,allow_fit=False,allow_labels=False,allow_sealed_identity=False)
    if any(r.get(k)!=v for k,v in exact.items()):raise PermissionError('Exact CPU query release required')
    return r


def execute(path,release_path):
    c=preview(path);require_release(path,release_path,c)
    if set(c['helper_hashes'])!={'__init__.py','tep_normal_ridge.py','tep_ridge_readout.py','normal_b1_readout.py','tep_ridge_query.py'}:raise ValueError('Exact pure helper closure')
    for name,digest in c['helper_hashes'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('Source changed')
    def ref(key,json_body=True):
        p=c[key]
        if sha(p['path'])!=p['sha256']:raise ValueError('Pinned reference changed: '+key)
        return read_json(p['path']) if json_body else Path(p['path'])
    fit=ref('ridge_acceptance');selection=ref('selection');scaler=ref('scaler');inputs=ref('input_acceptance');manifest=ref('input_manifest')
    cal_receipt=ref('calibration_acceptance');cal=ref('calibration');ref('contract',False)
    if (fit['status']!='PRIMARY_ACCEPTED_LIMITED_NORMAL_RIDGE_FIT_SELECT' or fit['coefficients_sha256']!=c['coefficients']['sha256']
            or fit['model_selection_sha256']!=c['selection']['sha256'] or fit['scaler_sha256']!=c['scaler']['sha256']):raise PermissionError('Actual accepted parameter refs')
    if inputs.get('status')!='PRIMARY_ACCEPTED_LIMITED_TEP_FAULTY_INPUTS' or inputs.get('input_manifest_sha256')!=c['input_manifest']['sha256']:raise PermissionError('Accepted opaque inputs required')
    if (cal_receipt.get('status')!='PRIMARY_ACCEPTED_LIMITED_RIDGE_NORMAL_CALIBRATION'
            or cal_receipt.get('calibration_json_sha256')!=c['calibration']['sha256']
            or cal['coefficient_sha256']!=c['coefficients']['sha256'] or cal['scaler_sha256']!=c['scaler']['sha256']
            or cal['n']!=8020 or cal['rank_1based']!=7941 or cal['comparison']!='strict_greater'
            or set(cal['thresholds'])!={m+'_'+r for m in METHODS for r in ('last','mean32')}):raise PermissionError('Actual fixed10 CAL thresholds required')
    rows=validate_query_manifest(manifest);validate_selection(selection)
    import os,resource
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise PermissionError('CPU-only single thread')
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    import numpy as np
    from threadpoolctl import threadpool_limits
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();report=dict(status='RUNNING',banks=[],GPU_hours=0,no_fit=True,no_label_evaluation=True)
    compact=[]
    try:
        with np.load(ref('coefficients',False),allow_pickle=False) as w:
            weights={f'{f}_a100_target{t}_{k}':w[f'{f}_a100_target{t}_{k}'].copy() for f in FAMILIES for t in range(3) for k in ('coef','intercept')}
        np.savez(out/'selected_parameters.npz',**weights)
        with threadpool_limits(limits=1):
            for row in rows:
                p=Path(row['path'])
                if sha(p)!=row['sha256']:raise ValueError('Input bank changed')
                with np.load(p,allow_pickle=False) as z:
                    if set(z.files)!={'values','sample_1based','source_row_0based'}:raise PermissionError('Numerical input fields only')
                    values=z['values'].copy();sample=z['sample_1based'].copy();source=z['source_row_0based'].copy()
                if values.shape!=(960,3) or not np.array_equal(sample,np.arange(1,961)) or source.dtype!=np.int64 or source.shape!=(960,) or (source<0).any() or not (np.diff(source)>0).all():raise ValueError('Original opaque query axis differs')
                arrays=score_values(values,scaler,selection,weights);arrays.update(sample_1based=sample,source_row_0based=source)
                p=out/(row['entity_id']+'.npz');np.savez(p,**arrays)
                report['banks'].append(dict(entity_id=row['entity_id'],path=str(p),sha256=sha(p),input_sha256=row['sha256']))
                compact.append({m+'_'+r:arrays[m+'_'+r].copy() for m in METHODS for r in ('last','mean32')})
                if time.monotonic()-start>300:raise TimeoutError('300 second contract')
        np.savez(out/'compact_scores.npz',entity_id=np.asarray([r['entity_id'] for r in rows]),end_index_0based=np.tile(np.arange(99,960),(400,1)),
                 **{k:np.stack([d[k] for d in compact]) for k in compact[0]})
        report.update(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',config_sha256=sha(path),release_sha256=sha(release_path),
            source_refs=c,feature_order=selection['features'],calibration_thresholds=cal['thresholds'],compact_sha256=sha(out/'compact_scores.npz'))
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
        if not a.release:raise PermissionError('Root query release required')
        print(json.dumps(execute(a.config,a.release),indent=2))
    else:preview(a.config);print('PENDING_INPUT_AND_CAL_ACCEPTANCE_NO_PARAMETER_OR_QUERY_ARRAY_ACCESS')

if __name__=='__main__':main()
