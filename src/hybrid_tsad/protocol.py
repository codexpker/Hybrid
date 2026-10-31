"""Frozen forecast-block index contract; labels returned only to evaluator."""
import json
import numpy as np
from pathlib import Path

class QueryAccessor:
    def __init__(self,manifest):
        self.path=Path(manifest);self.meta=json.loads(self.path.read_text());self.cache={}
    def block(self,record,origin,include_labels=False):
        assert origin in record['origins']
        p=record['array_path']
        if p not in self.cache:self.cache[p]=np.load(p,mmap_mode='r')
        a=self.cache[p];h=record['history'];f=record['horizon']
        assert origin-h>=record['local_start']+record['guard']
        assert origin+f<=record['local_end_exclusive']-record['guard']
        out={'history':a[origin-h:origin].copy(),'target':a[origin:origin+f].copy(),
             'time_index':np.arange(origin,origin+f),'entity':record['entity'],'role':record['role']}
        if include_labels:
            y=np.load(record['label_path'],mmap_mode='r');out['labels']=y[origin:origin+f].copy()
        return out

def guarded_metric_eligibility(labels):
    """Do not calculate AP or ROC when a query collection has only one class."""
    u=np.unique(labels)
    return {'AP_ROC_eligible':bool(np.array_equal(u,[0,1])),
            'reason':None if np.array_equal(u,[0,1]) else 'SINGLE_CLASS_SCORED_COLLECTION'}

def score_block(prediction,target,history_scale,source_error_scale):
    prediction=np.asarray(prediction);target=np.asarray(target);history_scale=np.asarray(history_scale)
    if prediction.shape!=target.shape or target.ndim<2:raise ValueError('Expected equal [...,time,channels] shapes')
    expected=target.shape[:-2]+(1,target.shape[-1])
    if history_scale.shape not in (expected,(target.shape[-1],)):raise ValueError('History scale must be per-channel, constant over forecast time')
    if not np.isfinite(history_scale).all() or (history_scale<=0).any():raise ValueError('History scale must be finite and positive')
    scalar=np.asarray(source_error_scale)
    if scalar.ndim!=0 or not np.isfinite(scalar) or scalar<=0:raise ValueError('Calibration scale must be finite positive scalar')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('Missing target needs explicit valid-time exclusion; do not fill normal')
    return (np.abs(prediction-target)/history_scale).mean(-1)/source_error_scale
