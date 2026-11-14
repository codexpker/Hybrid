"""Fixed alarm-level OR; preserves each branch's causal three-point confirmation.

No continuous score fusion, model fitting, conditional gate, or label-based
operating-point selection is performed here. Calibration is exploratory when
the caller's normal CAL reused SELECT; disjoint event matching is separate
from overlapping any-hit signal evidence.
"""
import numpy as np


def confirmed(index, alarm, count=3):
    index=np.asarray(index);alarm=np.asarray(alarm,dtype=bool)
    if index.ndim!=1 or alarm.shape!=index.shape or not len(index) or np.any(np.diff(index)<=0):
        raise ValueError('ordered nonempty time axis required')
    out=np.zeros(len(index),bool);run=0
    for i in range(len(index)):
        if i and index[i]!=index[i-1]+1:run=0
        run=run+1 if alarm[i] else 0
        out[i]=run>=count
    return out


def fixed_policies(index, query_scores, calibration_scores):
    names=('LEFT_native','ARX_point','ARX_accumulated')
    if set(query_scores)!=set(names) or set(calibration_scores)!=set(names):
        raise ValueError('exactly three frozen raw branches required')
    components={};taus={}
    for name in names:
        c=np.asarray(calibration_scores[name]);q=np.asarray(query_scores[name])
        if c.shape!=(512,) or q.shape!=np.asarray(index).shape or not np.isfinite(c).all() or not np.isfinite(q).all():
            raise ValueError('finite common512 CAL and full QUERY scores required')
        taus[name]={str(level):float(np.quantile(c,level,method='higher'))for level in(.99,.995)}
        components[name]={}
        for level in(.99,.995):
            strict=q>taus[name][str(level)]
            components[name][str(level)]={'strict':strict,'confirmed3':confirmed(index,strict)}
    policies={name:dict(components[name]['0.99'],components=[name],cal_quantile=.99)for name in names}
    for name in names[1:]:
        left=components['LEFT_native']['0.995'];right=components[name]['0.995']
        policies['LEFT_OR_'+name]=dict(strict=left['strict']|right['strict'],confirmed3=left['confirmed3']|right['confirmed3'],components=['LEFT_native',name],cal_quantile=.995)
    return policies,components,taus


def point_counts(labels, scored, alarm):
    y=np.asarray(labels)==1;scored=np.asarray(scored,bool);alarm=np.asarray(alarm,bool)
    tp=int((scored&alarm&y).sum());fp=int((scored&alarm&~y).sum());fn=int((scored&~alarm&y).sum());tn=int((scored&~alarm&~y).sum())
    ratio=lambda a,b:float(a/b)if b else None
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,Precision=ratio(tp,tp+fp),Recall=ratio(tp,tp+fn),F1=ratio(2*tp,2*tp+fp+fn))


def longest_miss(alarm):
    best=run=0
    for hit in np.asarray(alarm,bool):
        run=0 if hit else run+1;best=max(best,run)
    return int(best)
