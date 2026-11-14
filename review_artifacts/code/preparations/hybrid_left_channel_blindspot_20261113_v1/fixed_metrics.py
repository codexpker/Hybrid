"""Existing event definitions reused for the fixed blindspot comparison."""
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R/'hybrid_ad/src'))
from hybrid_tsad.hybrid_v1 import evaluation as k
from hybrid_tsad.alarm_complementarity import confirmed,longest_miss,point_counts

def rolling_max64(score):
    from collections import deque
    q=deque();out=np.empty(len(score),float)
    for i,x in enumerate(score):
        while q and q[0]<i-63:q.popleft()
        while q and score[q[-1]]<=x:q.pop()
        q.append(i);out[i]=score[q[0]]
    return out

def evaluate(index,score,cal,truth,steady_after=192,hard_normal=None,long_min_length=512):
    index=np.asarray(index,int);score=np.asarray(score,float);truth=np.asarray(truth,np.int8)
    assert np.array_equal(index,np.arange(index[0],index[-1]+1))and len(score)==len(index)
    assert np.isfinite(score).all()and np.isfinite(cal).all()and index[-1]<len(truth)
    tau=float(np.quantile(cal,.99,method='higher'));scored=np.zeros(len(truth),bool);scored[index]=True
    events=k._coverage(truth,scored,0);ranking=dict(AP=float(average_precision_score(truth[index],score))if len(np.unique(truth[index]))==2 else None,ROC=float(roc_auc_score(truth[index],score))if len(np.unique(truth[index]))==2 else None)
    rows=[];per_event=[]
    for stream,bits in [('strict',score>tau),('confirmed3',confirmed(index,score>tau))]:
        alarm=np.zeros(len(truth),bool);alarm[index]=bits
        view,_=k._event_view(truth,scored,alarm,np.arange(len(truth)),events,0,'main_eligible');normal,_=k._normal(truth,scored,alarm,0)
        look={e['event_id']:e for e in view['events']};covers=[];steady=[];sp=sn=0;miss=[];delays=[]
        for e in events:
            lo,hi=e['local_start'],e['local_end'];b=alarm[lo:hi];info=look[e['event_id']];v=None
            if e['main_eligible']:
                covers.append(float(b.mean()));miss.append(longest_miss(b))
                if info['delay_points']is not None:delays.append(info['delay_points'])
                if hi-lo>long_min_length:
                    v=alarm[lo+steady_after:hi];steady.append(float(v.mean()));sp+=int(v.sum());sn+=len(v)
            per_event.append(dict(stream=stream,**e,**{a:b for a,b in info.items()if a!='event_id'},coverage=float(b.mean()),longest_miss_points=longest_miss(b),steady_coverage=float(v.mean())if v is not None else None))
        if hard_normal is not None:
            hn=np.asarray(hard_normal,bool)&scored&(~truth.astype(bool));normal['hard_normal_points']=int(hn.sum());normal['hard_normal_false_points']=int((alarm&hn).sum());normal['hard_normal_FPR']=float(alarm[hn].mean())if hn.any()else None
        rows.append(dict(stream=stream,long_min_length=long_min_length,steady_after=steady_after,threshold=tau,**ranking,**point_counts(truth,scored,alarm),**view['metrics'],**{a:b for a,b in normal.items()if not isinstance(b,list)},event_macro_coverage=float(np.mean(covers))if covers else None,long_event_count=len(steady),steady_macro_coverage=float(np.mean(steady))if steady else None,steady_pooled_coverage=sp/sn if sn else None,max_event_miss=max(miss)if miss else None,hit_only_delay_median=float(np.median(delays))if delays else None,uncertainty='single continuous series; events not independent runs'))
    return rows,per_event

def tests():
    assert np.array_equal(rolling_max64(np.array([3.,1.,2.])),[3,3,3])
    x=np.arange(100.,0.,-1);m=rolling_max64(x);assert m[63]==100 and m[64]==99
    ix=np.arange(512,1536);y=np.zeros(1536,np.int8);y[700:1400]=1;s=np.zeros(len(ix));s[(ix>=700)&(ix<1400)]=2
    rows,events=evaluate(ix,s,np.zeros(512),y);assert rows[1]['TP']==698 and rows[1]['long_event_count']==1
    zero,_=evaluate(ix,np.ones(len(ix)),np.zeros(512),np.zeros(1536,np.int8));assert zero[0]['AP']is None and zero[0]['ROC']is None and zero[0]['FP']==len(ix)
    print('FIXED_METRICS_CPU_PASS')
if __name__=='__main__':tests()

