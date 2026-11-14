"""One fixed FIT-only nearest-context prediction-skill criterion.

Normal-reference retrieval and prediction-skill weighting are known ideas.
The frozen Ridge FIT residuals here are in-sample, not OOF confidence.
No current target, residual, label, or future input enters the online key.
"""
import numpy as np
WINDOW=512


def past_context(values):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or len(values)<=WINDOW or not np.isfinite(values).all():
        raise ValueError('finite [time,channel] run longer than512 required')
    zero=np.zeros((1,values.shape[1]),np.float64)
    sums=np.concatenate((zero,np.cumsum(values,axis=0)),axis=0)
    squares=np.concatenate((zero,np.cumsum(values*values,axis=0)),axis=0)
    mean=(sums[WINDOW:-1]-sums[:-WINDOW-1])/WINDOW
    second=(squares[WINDOW:-1]-squares[:-WINDOW-1])/WINDOW
    std=np.sqrt(np.maximum(second-mean*mean,0.))
    return np.concatenate((mean,std),axis=1),mean


def build_references(runs,predictions):
    if len(runs)!=len(predictions)or not len(runs):raise ValueError('paired original runs required')
    keys=[];skills=[];records=[];arx_errors=[];mean_errors=[]
    for run_number,(values,prediction)in enumerate(zip(runs,predictions)):
        values=np.asarray(values,np.float64);prediction=np.asarray(prediction,np.float64)
        if prediction.shape!=values[WINDOW:].shape:raise ValueError('strictpast target axis mismatch')
        context,history_mean=past_context(values)
        for anchor in range(WINDOW,len(values)-WINDOW+1,WINDOW):
            lo=anchor-WINDOW;hi=anchor
            mse_arx=float(np.mean((values[anchor:anchor+WINDOW]-prediction[lo:hi])**2))
            mse_mean=float(np.mean((values[anchor:anchor+WINDOW]-history_mean[lo:hi])**2))
            skill=float(np.clip(1-mse_arx/max(mse_mean,1e-12),0,1))
            keys.append(context[lo]);skills.append(skill);arx_errors.append(mse_arx);mean_errors.append(mse_mean)
            records.append(dict(run=run_number,anchor=anchor,key_start=anchor-WINDOW,key_end_exclusive=anchor,outcome_start=anchor,outcome_end_exclusive=anchor+WINDOW))
    keys=np.asarray(keys,np.float64);skills=np.asarray(skills,np.float64)
    if not len(keys):raise ValueError('at least one full512 key plus512 outcome required')
    mean=keys.mean(0);std=keys.std(0);scale=np.where(std<=1e-6,1.,std)
    return dict(keys=keys,skills=skills,key_mean=mean,key_scale=scale,standardized_keys=(keys-mean)/scale,
        reference_ARX_MSE=np.asarray(arx_errors),reference_H512_MSE=np.asarray(mean_errors),records=records)


def match_context(context,references,block=256):
    context=np.asarray(context,np.float64)
    standardized=(context-references['key_mean'])/references['key_scale']
    choices=np.empty(len(context),np.int64);distance=np.empty(len(context),np.float64)
    for start in range(0,len(context),block):
        end=min(start+block,len(context));diff=standardized[start:end,None,:]-references['standardized_keys'][None,:,:]
        d=np.sum(diff*diff,axis=2);nearest=np.argmin(d,axis=1)
        choices[start:end]=nearest;distance[start:end]=d[np.arange(end-start),nearest]
    return references['skills'][choices],choices,distance,standardized


def branch_status(skills,calibrated_relation):
    if not np.any(np.asarray(skills)>0):return 'DISABLED_ALL_FIT_SKILLS_ZERO'
    if not np.any(np.asarray(calibrated_relation)>0):return 'DISABLED_ALL_CAL_GATED_SCORES_ZERO'
    return 'ENABLED'


def conditional_readout(left,point,rho,tau_left,tau_cond,status):
    if tau_left<=0 or not np.isfinite(tau_left):raise ValueError('positive frozen LEFT threshold required')
    left_ratio=np.asarray(left,dtype=np.float64)/np.float64(tau_left)
    raw_conditioned=np.asarray(point,dtype=np.float64)*np.asarray(rho,dtype=np.float64)
    effective_conditioned=raw_conditioned if status=='ENABLED'else np.zeros_like(raw_conditioned)
    effective_tau=max(float(tau_cond),1e-12)
    score=np.maximum(left_ratio,effective_conditioned/effective_tau)
    return raw_conditioned,effective_conditioned,score
