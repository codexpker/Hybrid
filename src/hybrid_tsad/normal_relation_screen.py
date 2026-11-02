"""NumPy-only normal-train relation screen. Real data require exact root release.

Pure functions accept synthetic arrays. No query/label/perturbation/model module.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

OP='PUBLIC_NORMAL_RELATION_SCREEN'
RAW_SHA='ecebd2933ccdc1126e2abbfb538762dbde16410cebc77257a7b96e1e1fbcf753'
FIT=(0,13669);SELECT=(13861,18226);SHAPE=(28479,38)
LAM=1e-3

def ridge_fit(x,y,lam=LAM):
    """Minimize mean squared error + lambda*||coef||²; intercept unpenalized."""
    import numpy as np
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(len(x),) or len(x)==0 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Finite nonempty ridge geometry')
    center=x.mean(0);target_mean=float(y.mean());xc=x-center
    coef=np.linalg.solve(xc.T@xc/len(x)+lam*np.eye(x.shape[1]),xc.T@(y-target_mean)/len(x))
    return dict(coef=coef,feature_center=center,target_mean=target_mean,
        intercept=target_mean-float(center@coef),lambda_mse=lam)

def predict(x,model):return x@model['coef']+model['intercept']

def rank_candidates(rows):
    return sorted(rows,key=lambda r:(-r['abs_corr'],r['target'],r['source'],r['lag']))

def design(values,target,source,lag,family):
    import numpy as np
    t=np.arange(40,len(values),dtype=np.int64)
    own=8 if family=='AR8' else 32
    features=[(target,i) for i in range(1,own+1)]
    if family=='SYNC':features.append((source,0))
    elif family=='LAG':features.append((source,lag))
    elif family=='MULTI':features.extend((source,i) for i in range(1,33))
    elif family not in ('AR8','AR32'):raise ValueError('Fixed five families only')
    if target==source or not 1<=lag<=32 or any(ch==target and delay==0 or delay<0 for ch,delay in features):raise PermissionError('No current target/future or self pair')
    x=np.column_stack([values[t-delay,ch] for ch,delay in features])
    return x,values[t,target],t,features

def normal_screen(fit,select,*,select_origin=13861,budget=lambda:None):
    import numpy as np
    fit=np.asarray(fit,dtype=np.float64);select=np.asarray(select,dtype=np.float64)
    if fit.ndim!=2 or select.ndim!=2 or fit.shape[1]!=select.shape[1] or min(len(fit),len(select))<=40 or not np.isfinite(fit).all() or not np.isfinite(select).all():raise ValueError('Finite role-isolated normal matrices')
    mean=fit.mean(0);scale=fit.std(0,ddof=0);active=np.flatnonzero(scale>1e-12)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():raise ValueError('Nonfinite FIT statistics')
    safe=np.where(scale>1e-12,scale,1.)
    z=(fit-mean)/safe;v=(select-mean)/safe
    result=dict(status='HOLD',reason=None,FIT_points=len(fit),SELECT_points=len(select),
        scaler=dict(mean=mean,scale=safe,raw_std=scale,ddof=0,FIT_only=True),
        excluded_low_variance_channels=np.flatnonzero(scale<=1e-12).tolist(),duplicates=[],innovation_models={},
        AP=None,ROC_AUC=None,Aff_F1=None,Range_AUC=None,VUS_PR=None,event_recall=None,detection_delay=None)
    duplicate=set()
    for i,source in enumerate(active):
        for target in active[i+1:]:
            slope=float(z[:,source]@z[:,target]/(z[:,source]@z[:,source]))
            residual=z[:,target]-slope*z[:,source]
            ratio=float(np.var(residual)/np.var(z[:,target]))
            if ratio<=1e-10:
                duplicate.add((int(source),int(target)));duplicate.add((int(target),int(source)))
                result['duplicates'].append(dict(source=int(source),target=int(target),affine_slope_standardized=slope,residual_variance_ratio=ratio))
        budget()
    innovation=np.full_like(z,np.nan)
    t8=np.arange(8,len(z))
    for ch in active:
        x=np.column_stack([z[t8-k,ch] for k in range(1,9)])
        model=ridge_fit(x,z[t8,ch]);innovation[t8,ch]=z[t8,ch]-predict(x,model)
        result['innovation_models'][str(ch)]=model;budget()
    # Common targets40..FITend-1; lag32 sources start8, never prefix NaNs.
    t=np.arange(40,len(z));y=innovation[t][:,active];yc=y-y.mean(0);yv=np.sum(yc*yc,0)
    candidates=[]
    for lag in range(1,33):
        x=innovation[t-lag][:,active];xc=x-x.mean(0);xv=np.sum(xc*xc,0)
        den=np.sqrt(yv[:,None]*xv[None,:])
        corr=np.divide(yc.T@xc,den,out=np.full_like(den,np.nan),where=den>0)
        for iy,target in enumerate(active):
            for ix,source in enumerate(active):
                if source==target or (int(source),int(target)) in duplicate or not np.isfinite(corr[iy,ix]):continue
                candidates.append(dict(target=int(target),source=int(source),lag=lag,corr=float(corr[iy,ix]),abs_corr=abs(float(corr[iy,ix]))))
        budget()
    candidates=rank_candidates(candidates)
    result['candidates']=candidates;result['top10']=candidates[:10]
    if not candidates:
        result['reason']='NO_ELIGIBLE_FINITE_NON_DUPLICATE_PAIR';return result
    chosen=dict(candidates[0]);result['selected_pair']=chosen
    target,source,lag=chosen['target'],chosen['source'],chosen['lag']
    models={};predictions={};metrics=[];block_positions=np.array_split(np.arange(len(v)-40),4)
    for family in ['AR8','AR32','SYNC','LAG','MULTI']:
        x,y,t,features=design(z,target,source,lag,family);model=ridge_fit(x,y);model['features']=features;models[family]=model
        xs,ys,ts,_=design(v,target,source,lag,family);prediction=predict(xs,model);predictions[family]=prediction
        for block,pos in [('ALL',np.arange(len(ys)))]+[(str(i+1),p) for i,p in enumerate(block_positions)]:
            error=prediction[pos]-ys[pos]
            metrics.append(dict(model=family,block=block,n=len(pos),MAE=float(np.abs(error).mean()),MSE=float(np.mean(error**2)),RMSE=float(np.sqrt(np.mean(error**2)))))
        budget()
    lookup={(m['model'],m['block']):m for m in metrics};comparisons=[]
    for alternative,base in [(a,b) for a in ['AR8','AR32','SYNC','LAG','MULTI'] for b in ['AR32','SYNC'] if a!=b]:
        for block in ['ALL','1','2','3','4']:
            a=lookup[alternative,block]['MSE'];b=lookup[base,block]['MSE']
            comparisons.append(dict(comparison=alternative+'-'+base,block=block,delta_MSE=a-b,
                relative_MSE_reduction=None if b==0 else (b-a)/b))
    global_gain=next(m['relative_MSE_reduction'] for m in comparisons if m['comparison']=='MULTI-AR32' and m['block']=='ALL')
    improved=sum(lookup['MULTI',str(i)]['MSE']<lookup['AR32',str(i)]['MSE'] for i in range(1,5))
    # SELECT innovations use only FIT-fitted univariate AR8, not SELECT fitting.
    innovations=np.full_like(v,np.nan);st=np.arange(8,len(v))
    for ch in (source,target):
        sx=np.column_stack([v[st-k,ch] for k in range(1,9)])
        innovations[st,ch]=v[st,ch]-predict(sx,result['innovation_models'][str(ch)])
    select_corr=[]
    for name,pos in [('ALL',np.arange(len(v)-40))]+[(str(i+1),p) for i,p in enumerate(block_positions)]:
        tt=ts[pos];a=innovations[tt,target];b=innovations[tt-lag,source]
        std_a=float(np.std(a));std_b=float(np.std(b))
        tol_a=64*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(a))))
        tol_b=64*np.finfo(np.float64).eps*max(1.,float(np.max(np.abs(b))))
        degenerate_a=std_a<=tol_a;degenerate_b=std_b<=tol_b
        corr=None if degenerate_a or degenerate_b else float(np.corrcoef(a,b)[0,1])
        den=float(np.sqrt(np.sum((a-a.mean())**2)*np.sum((b-b.mean())**2)))
        select_corr.append(dict(block=name,n=len(tt),corr=corr,corr_denominator=den,
            target_std=std_a,source_std=std_b,target_std_tolerance=tol_a,source_std_tolerance=tol_b,
            target_degenerate=degenerate_a,source_degenerate=degenerate_b,
            degenerate=degenerate_a or degenerate_b,
            direction='NA' if corr is None else 'positive' if corr>0 else 'negative' if corr<0 else 'zero'))
    result.update(status='NORMAL_RELATION_SCREEN_COMPLETED',reason=None,models=models,metrics=metrics,
        comparisons=comparisons,SELECT_pair_corr=select_corr,
        SELECT_corr_std_across_four_blocks=float(np.std([x['corr'] for x in select_corr[1:] if x['corr'] is not None],ddof=0)) if any(x['corr'] is not None for x in select_corr[1:]) else None,
        selection_t_0based=ts+select_origin,SELECT_target_z=ys,SELECT_target_original=select[ts,target],predictions_z=predictions,
        predictions_original={k:a*safe[target]+mean[target] for k,a in predictions.items()},
        absolute_residual_z={k:np.abs(a-ys) for k,a in predictions.items()},
        squared_residual_z={k:(a-ys)**2 for k,a in predictions.items()},
        absolute_residual_original={k:np.abs(a-ys)*safe[target] for k,a in predictions.items()},
        squared_residual_original={k:(a-ys)**2*safe[target]**2 for k,a in predictions.items()},
        pilot_relation_usable=global_gain is not None and global_gain>=.05 and improved>=3,
        improved_blocks=improved,scientific_decision='HOLD_NOT_ANOMALY_DETECTION_GO',
        input_footprint=dict(common_target_start_within_role=40,max_own_lag=32,max_source_lag=32,
            candidate_innovation_own_AR=8,earliest_candidate_raw_within_FIT=0,
            SELECT_history_entirely_within_role=True,SELECT_block_history_may_cross_blocks=True),
        AP=None,Aff_F1=None,VUS_PR=None,
        note='Correlated time blocks, not independent runs; current other source is observable, not causal/exogenous.')
    return result

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jsonable(value):
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    if hasattr(value,'tolist'):return value.tolist()
    return value

def load_roles_only(path,*,shape=SHAPE,fit_range=FIT,select_range=SELECT):
    """Scan full line/field counts; convert numbers ONLY in allowed two roles."""
    import numpy as np
    fit=[];select=[];count=0
    with Path(path).open('r',encoding='utf-8') as handle:
        for i,line in enumerate(handle):
            fields=line.strip().split(',')
            if len(fields)!=shape[1]:raise ValueError('Raw field-count/shape differs')
            if fit_range[0]<=i<fit_range[1]:fit.append([float(x) for x in fields])
            elif select_range[0]<=i<select_range[1]:select.append([float(x) for x in fields])
            count+=1
    if count!=shape[0]:raise ValueError('Raw row-count differs')
    fit=np.asarray(fit,np.float64);select=np.asarray(select,np.float64)
    if fit.shape!=(fit_range[1]-fit_range[0],shape[1]) or select.shape!=(select_range[1]-select_range[0],shape[1]):raise ValueError('Role footprint differs')
    return fit,select

def execute(config_path,release_path,trusted):
    c=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if c.get('binding_state')!='BOUND_NORMAL_RELATION_SCREEN':raise PermissionError('PENDING before raw/numerical access')
    if trusted is None or len(trusted)!=64 or digest(release_path)!=trusted:raise PermissionError('External root trusted releaseSHA')
    r=json.loads(Path(release_path).read_text(encoding='utf-8'))
    if (r.get('status')!='PRIMARY_RELEASED_NORMAL_RELATION_SCREEN' or r.get('approved_by')!='root' or r.get('operation')!=OP
            or r.get('config_sha256')!=digest(config_path) or r.get('module_sha256')!=digest(__file__)
            or r.get('allow_FIT_SELECT_only') is not True or r.get('allow_CAL_pool_statistics') is not False):raise PermissionError('Exact normal-only root release')
    if (c.get('operation')!=OP or c.get('raw_sha256')!=RAW_SHA or c.get('shape')!=list(SHAPE)
            or c.get('FIT')!=list(FIT) or c.get('SELECT')!=list(SELECT) or c.get('max_seconds')!=120):raise PermissionError('Fixed public source/roles only')
    if digest(c['contract']['path'])!=c['contract']['sha256']:raise PermissionError('Contract bytes differ')
    out=Path(c['output'])
    if not out.is_absolute() or out.name!='public_normal_relation_screen_20261102_v1' or out.parent.name!='results':raise PermissionError('Dedicated absolute result output required')
    if out.exists():raise FileExistsError('Fresh output, no automatic retry')
    if any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']):raise PermissionError('Single-thread CPU shell required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU only')
    wall=time.monotonic();cpu=time.process_time()
    def budget():
        if time.monotonic()-wall>120 or time.process_time()-cpu>120:raise TimeoutError('Fixed wall/CPU120s; no reduced family')
    if digest(c['raw_path'])!=RAW_SHA:raise PermissionError('Raw source identity differs')
    import numpy as np
    fit,select=load_roles_only(c['raw_path']);budget()
    result=normal_screen(fit,select,select_origin=SELECT[0],budget=budget)
    out.mkdir(parents=False,exist_ok=False)
    arrays={}
    if 'models' in result:
        arrays={k:result[k] for k in ['selection_t_0based','SELECT_target_z','SELECT_target_original']}
        for field in ['predictions_z','predictions_original','absolute_residual_z','squared_residual_z','absolute_residual_original','squared_residual_original']:
            arrays.update({field+'_'+k:v for k,v in result[field].items()})
    np.savez(out/'normal_screen_arrays.npz',**arrays)
    (out/'result.json').write_text(json.dumps(jsonable(result),indent=2,allow_nan=False),encoding='utf-8')
    for name,rows in [('candidates',result['candidates']),('metrics',result.get('metrics',[])),
                      ('comparisons',result.get('comparisons',[])),('SELECT_pair_corr',result.get('SELECT_pair_corr',[]))]:
        if rows:
            with (out/(name+'.csv')).open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'execution.json').write_text(json.dumps(dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',config_sha256=digest(config_path),module_sha256=digest(__file__),raw_sha256=RAW_SHA,
        python=sys.version,numpy=np.__version__,deterministic_no_optimizer_seed=True,
        elapsed_seconds=time.monotonic()-wall,CPU_seconds=time.process_time()-cpu,GPU_hours=0,
        resource_scope='wall/CPU callbacks sampled between kernels; root external hard timeout/RSS supervisor required',
        CAL_or_pool_statistics=False,query_or_labels_loaded=False)),encoding='utf-8')
    files=[dict(path=p.name,sha256=digest(p),bytes=p.stat().st_size) for p in sorted(out.iterdir())]
    (out/'MANIFEST.json').write_text(json.dumps(dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',files=files),indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--release');p.add_argument('--trusted-release-sha256');a=p.parse_args()
    if not a.execute:print(json.dumps(dict(status='PREVIEW_NO_DATA_ACCESS',config=json.loads(Path(a.config).read_text(encoding='utf-8')))));return
    execute(a.config,a.release,a.trusted_release_sha256)

if __name__=='__main__':main()
