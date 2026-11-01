"""Fixed TEP normal-only Ridge baselines; no CAL/fault/label scoring path."""
import argparse
import hashlib
import json
from pathlib import Path

COLUMNS=('xmeas_9','xmeas_21','xmv_10')
FAMILIES=('SELF','SYNC','PEAK','MULTI')
ALPHAS=(.1,1.,10.,100.)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def choose_alpha(losses):
    import math
    if set(losses)!=set(ALPHAS) or not all(math.isfinite(v) for v in losses.values()):
        raise ValueError('All four finite normal SELECT losses required')
    return min(ALPHAS,key=lambda a:(losses[a],-a))


def choose_peak(correlations):
    import math
    if set(correlations)!=set(range(1,65)) or not all(math.isfinite(v) for v in correlations.values()):
        raise ValueError('Complete finite past-lag profile required')
    return min(range(1,65),key=lambda lag:(-abs(correlations[lag]),lag))


def feature_order(family,target,peaks=None):
    if family not in FAMILIES or target not in range(3):raise ValueError('Fixed family/target required')
    spec=[{'channel':target,'name':COLUMNS[target],'lag':lag} for lag in range(1,65)]
    for other in range(3):
        if other==target or family=='SELF':continue
        lags=range(65) if family=='MULTI' else [0] if family=='SYNC' else [peaks[(target,other)]]
        spec.extend({'channel':other,'name':COLUMNS[other],'lag':int(lag)} for lag in lags)
    return spec


def design(runs,target,spec):
    """Per-run t64..end grid; never concatenate histories across runs."""
    import numpy as np
    if target not in range(3) or not runs:raise ValueError('Nonempty three-target runs required')
    from .normal_run_data import check_runs
    check_runs(runs,runs[0].role)
    for f in spec:
        if (f['channel'] not in range(3) or f['name']!=COLUMNS[f['channel']]
            or f['lag'] not in range(65) or (f['channel']==target and f['lag']==0)):
            raise PermissionError('Current target/future/derived input is forbidden')
    X=[];Y=[];ids=[];times=[]
    for run in runs:
        if tuple(run.columns)!=COLUMNS or len(run.values)<65:raise ValueError('Fixed columns and complete past required')
        t=np.arange(64,len(run.values),dtype=np.int64)
        X.append(np.column_stack([run.values[t-f['lag'],f['channel']] for f in spec]))
        Y.append(run.values[t,target]);ids.append(np.full(len(t),run.run_id,np.int64));times.append(t)
    return np.concatenate(X),np.concatenate(Y),np.concatenate(ids),np.concatenate(times)


def peak_profile(fit):
    import numpy as np
    if any(r.role!='FIT' for r in fit):raise PermissionError('PEAK lags use only FIT')
    peaks={};rows=[]
    for target in range(3):
        for other in range(3):
            if target==other:continue
            correlations={}
            for lag in range(1,65):
                x,y,_,_=design(fit,target,[{'channel':other,'name':COLUMNS[other],'lag':lag}])
                x=x[:,0];x=x-x.mean();y=y-y.mean()
                denom=float(np.sqrt(np.dot(x,x)*np.dot(y,y)))
                if not np.isfinite(denom) or denom<=0:
                    raise ValueError(f'Undefined FIT Pearson on legal t>=64 grid: target={COLUMNS[target]}, other={COLUMNS[other]}, lag={lag}; HOLD, no lag selection')
                correlations[lag]=float(np.dot(x,y)/denom)
            selected=choose_peak(correlations);peaks[target,other]=selected
            rows.extend(dict(target=COLUMNS[target],other=COLUMNS[other],lag=l,pearson=v,
                             absolute_pearson=abs(v),selected=l==selected) for l,v in correlations.items())
    return peaks,rows


def normalize_runs(runs,scaler):
    import numpy as np
    from .normal_run_data import NormalRun,check_runs
    check_runs(runs,runs[0].role)
    mean=np.asarray(scaler['mean'],np.float64);scale=np.asarray(scaler['scale'],np.float64)
    if (tuple(scaler['columns'])!=COLUMNS or mean.shape!=(3,) or scale.shape!=(3,)
        or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any()
        or scaler['ddof']!=0 or any(x[1]!='FIT' or x[0]!=runs[0].domain for x in scaler['fit_ids'])):
        raise PermissionError('Fixed positive finite FIT-only accepted scaler required')
    if runs[0].role=='FIT' and [list(r.identity) for r in runs]!=scaler['fit_ids']:
        raise PermissionError('Scaler and FIT identities differ')
    return [NormalRun(r.domain,r.role,r.run_id,r.columns,(r.values-mean)/scale) for r in runs]


def fit_select(fit,select):
    """Already standardized normal runs. Mature sklearn SSE Ridge, no new solver."""
    import numpy as np
    from sklearn.linear_model import Ridge
    if any(r.role!='FIT' for r in fit) or any(r.role!='SELECT' for r in select):
        raise PermissionError('Only independent normal FIT/SELECT roles')
    peaks,profile=peak_profile(fit);selection=[];models={};predictions={};chosen={};features={}
    coordinates=None
    for family in FAMILIES:
        matrices=[]
        for target in range(3):
            spec=feature_order(family,target,peaks);xf,yf,ir,tr=design(fit,target,spec);xv,yv,iv,tv=design(select,target,spec)
            matrices.append((xf,yf,xv,yv));features[f'{family}_target{target}']=spec
            current=(ir,tr,iv,tv)
            if coordinates is None:coordinates=current
            else:
                for a,b in zip(coordinates,current):np.testing.assert_array_equal(a,b)
        losses={}
        for alpha in ALPHAS:
            pv=[];pf=[]
            for target,(xf,yf,xv,yv) in enumerate(matrices):
                model=Ridge(alpha=alpha,fit_intercept=True,solver='cholesky').fit(xf,yf)
                if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_):
                    raise FloatingPointError('Nonfinite Ridge coefficients')
                key=f'{family}_a{alpha:g}_target{target}'
                models[key]={'coef':model.coef_.copy(),'intercept':float(model.intercept_)}
                pf.append(model.predict(xf));pv.append(model.predict(xv))
                if not np.isfinite(pf[-1]).all() or not np.isfinite(pv[-1]).all():
                    raise FloatingPointError('Nonfinite normal prediction')
                np.testing.assert_allclose(model.predict(xv),xv@model.coef_+model.intercept_,rtol=1e-12,atol=1e-12)
            pv=np.column_stack(pv);pf=np.column_stack(pf);yv=np.column_stack([m[3] for m in matrices])
            losses[alpha]=float(np.mean((pv-yv)**2))
            selection.append(dict(family=family,alpha=alpha,SELECT_MSE=losses[alpha],selected=False))
            predictions[f'{family}_a{alpha:g}_SELECT']=pv
            predictions[f'{family}_a{alpha:g}_FIT']=pf
        chosen[family]=choose_alpha(losses)
        yf=np.column_stack([m[1] for m in matrices])
        predictions[f'{family}_selected_FIT_residual']=predictions[f'{family}_a{chosen[family]:g}_FIT']-yf
        predictions[f'{family}_selected_SELECT_residual']=predictions[f'{family}_a{chosen[family]:g}_SELECT']-yv
        for row in selection:
            if row['family']==family:row['selected']=row['alpha']==chosen[family]
    ir,tr,iv,tv=coordinates
    predictions.update(FIT_target=np.column_stack([design(fit,t,feature_order('SELF',t))[1] for t in range(3)]),
        SELECT_target=np.column_stack([design(select,t,feature_order('SELF',t))[1] for t in range(3)]),
        FIT_run_id=ir,FIT_t=tr,SELECT_run_id=iv,SELECT_t=tv)
    return dict(chosen=chosen,models=models,features=features,selection=selection,peak_profile=profile,predictions=predictions)


def preview(path):
    c=read_json(path)
    if (c.get('operation')!='TEP_NORMAL_RIDGE_FIT_SELECT' or c.get('columns')!=list(COLUMNS)
        or c.get('families')!=list(FAMILIES) or c.get('alphas')!=list(ALPHAS)
        or c.get('max_lag')!=64 or c.get('solver')!='cholesky' or c.get('fit_intercept') is not True
        or c.get('max_seconds')!=300 or c.get('max_GiB')!=4):
        raise PermissionError('Only fixed normal Ridge preparation')
    if set(c['adapter_hashes'])!={'__init__.py','left_baseline.py','normal_run_data.py','left_normal_public.py','tep_normal_ridge.py'}:
        raise ValueError('Exact five-helper source closure required')
    for n,digest in c['adapter_hashes'].items():
        if sha(Path(__file__).parent/n)!=digest:raise ValueError('Changed helper identity: '+n)
    for key in ['normal_metadata','scaler','normal_training_acceptance','normal_source_acceptance','contract']:
        ref=c[key]
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Changed normal metadata reference: '+key)
    from .left_normal_public import validate_normal_metadata
    metadata=validate_normal_metadata(read_json(c['normal_metadata']['path']))
    receipt=read_json(c['normal_training_acceptance']['path'])
    if (receipt.get('status')!='PRIMARY_ACCEPTED_LIMITED_TEP_NORMAL_TRAINING'
        or receipt['scaler']['sha256']!=c['scaler']['sha256']):raise PermissionError('Accepted FIT scaler identity required')
    source=read_json(c['normal_source_acceptance']['path'])
    if (source.get('status')!='PRIMARY_ACCEPTED_LIMITED_NORMAL_ASSETS'
        or source.get('source_manifest_sha256')!=metadata['accepted_asset_manifest_sha256']):
        raise PermissionError('Accepted normal-source metadata identity required')
    if Path(c['output']).exists():raise FileExistsError(c['output'])
    return c,metadata


def execute(path,release_path):
    c,metadata=preview(path);release=read_json(release_path)
    expected={'status':'PRIMARY_RELEASED_TEP_NORMAL_RIDGE_FIT_SELECT','approved_by':'root',
        'config_sha256':sha(path),'adapter_hashes':c['adapter_hashes'],'allow_FIT_SELECT':True,
        'allow_CAL':False,'allow_Faulty_Testing':False,'allow_query_labels':False}
    if any(release.get(k)!=v for k,v in expected.items()):raise PermissionError('Separate exact root normal-fit release required')
    import os
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):
        raise RuntimeError('CPU single-thread/CVD-empty environment required')
    import time
    import traceback
    import numpy as np
    import sklearn
    from threadpoolctl import threadpool_limits
    from .left_normal_public import load_normal_arrays
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report={'status':'RUNNING','sklearn':sklearn.__version__,'solver':'Ridge(fit_intercept=True,solver=cholesky)',
        'objective':'sum squared errors + alpha*L2(coefficients), intercept unpenalized',
        'roles_read':['FIT','SELECT'],'CAL_fault_query_labels_read':False,'source_config_sha256':sha(path),
        'scaler':c['scaler'],'deterministic_no_fake_seed_replicates':True}
    try:
        runs=load_normal_arrays(metadata,released=True);scaler=read_json(c['scaler']['path'])
        fit=normalize_runs(runs['FIT'],scaler);select=normalize_runs(runs['SELECT'],scaler)
        with threadpool_limits(limits=1):result=fit_select(fit,select)
        weights={key+'_'+field:np.asarray(value,np.float64) for key,row in result['models'].items() for field,value in row.items()}
        np.savez(out/'coefficients.npz',**weights);np.savez(out/'normal_predictions.npz',**result.pop('predictions'))
        result.pop('models')
        (out/'model_selection.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        report['status']='COMPLETED_PENDING_INDEPENDENT_REVIEW'
    except BaseException:
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',traceback=traceback.format_exc());raise
    finally:
        report['seconds']=time.monotonic()-start
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--release');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.execute:
        if not a.release:p.error('Exact root release required')
        execute(a.config,a.release)
    else:
        preview(a.config);print(json.dumps({'status':'METADATA_PREVIEW_ONLY_NO_REAL_ARRAY_OR_MODEL_FIT'}))


if __name__=='__main__':main()
