"""Fixed CPU controls for the legal-role RoCA comparison; no labels or fitting.

Consume the exact neural arm's FIT scaler and CAL endpoints.  Root fixed the
short history at W-1, matching the RoCA arrived [t-W+1,t] footprint.  The sole
static control is mean(z_t**2), not a window or method search.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

from .hybrid_v1.data import Run
from .hybrid_v1.baselines import baseline_record
from .hybrid_v1.scoring import weighted_quantile,apply_calibration

METHODS=('STATIC_FIT_DISTANCE','SHORT_HISTORY_MEAN','SHORT_PERSISTENCE')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,x):
    with Path(path).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def standardized_run(run,mean,scale):
    """Same float64 arithmetic ->float32 as RoCA FitScaler.transform; no fit."""
    mean=np.asarray(mean,np.float64);scale=np.asarray(scale,np.float64)
    x=np.asarray(run.values,np.float64)
    if (mean.shape!=(x.shape[1],) or scale.shape!=mean.shape or (scale<=0).any()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or not np.isfinite(x).all() or not run.observed.all()):
        raise ValueError('Exact FIT scaler and finite observed fixed channel inputs required')
    return Run(run.domain,run.entity,run.run,run.role,((x-mean)/scale).astype(np.float32),run.observed.copy(),run.start)

def fixed_endpoint_score(standardized,index,window,method):
    if method not in METHODS or window not in (16,32):raise PermissionError('Only root-fixed W16/W32 controls')
    if not isinstance(index,(int,np.integer)):raise ValueError('Integer original endpoint required')
    local=int(index)-standardized.start
    if standardized.role not in ('CAL','QUERY') or local<window-1 or local>=len(standardized.values):
        raise PermissionError('Endpoint crosses one original CAL/QUERY role boundary')
    if not standardized.observed[local-window+1:local+1].all():raise ValueError('No new missing-data policy')
    if method=='STATIC_FIT_DISTANCE':
        # Float64 reduction of the SAME archived float32 standardized input.
        score=float(np.square(standardized.values[local].astype(np.float64)).mean())
        actual=[int(index),int(index)+1]
    else:
        name='history_mean512' if method=='SHORT_HISTORY_MEAN' else 'persistence'
        r=baseline_record(standardized,local,method=name,history=window-1)
        if not r['valid']:raise ValueError('Invalid original baseline endpoint')
        score=r['r_pred'];actual=r['footprint']
    return dict(domain=standardized.domain,entity=standardized.entity,run=standardized.run,role=standardized.role,
        index=int(index),available_index=int(index),score=score,valid=True,
        RoCA_information_bounds=[int(index)-window+1,int(index)+1],actual_input_bounds=actual,
        history=window-1 if method!='STATIC_FIT_DISTANCE' else None,method=method)

def fixed_calibration(records):
    if not records or any(r['role']!='CAL' or not r['valid'] for r in records):
        raise PermissionError('Nonempty fixed CAL scores only')
    score=np.array([r['score'] for r in records],np.float64)
    if not np.isfinite(score).all():raise ValueError('Finite original CAL scores required')
    # The original inverse empirical CDF q99; never a QUERY quantile.
    return dict(threshold=weighted_quantile(score,np.ones(len(score))/len(score),.99),CAL_points=len(score),
        rule='weighted inverse empirical CDF; first CDF>=.99; strict score>tau',target_fit_scaler_reused=True)

def write_score_array(directory,records,calibration,provenance):
    """NPZ index/score/availability is the existing generic evaluator interface."""
    if not records:raise ValueError('Empty score bank')
    stream=tuple(records[0][k] for k in ('method','domain','entity','run','role'))
    if stream[-1] not in ('CAL','QUERY') or any(tuple(r[k] for k in ('method','domain','entity','run','role'))!=stream for r in records):
        raise PermissionError('One fixed method/original entity/run/role per score bank')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    index=np.array([r['index'] for r in records],np.int64);score=np.array([r['score'] for r in records],np.float64)
    if (np.diff(index)<=0).any() or not np.isfinite(score).all():raise ValueError('Ordered finite original endpoint scores')
    np.savez_compressed(directory/'scores.npz',index=index,available_index=index.copy(),score=score,
        valid=np.ones(len(index),bool),alarm=score>calibration['threshold'])
    dump(directory/'BANK.json',dict(status='SEALED_RAW_NO_LABELS',count=len(index),first_index=int(index[0]),
        last_index=int(index[-1]),score_sha256=sha(directory/'scores.npz'),calibration=calibration,
        method=records[0]['method'],domain=records[0]['domain'],entity=records[0]['entity'],run=records[0]['run'],
        first_record=records[0],last_record=records[-1],provenance=provenance,labels_read=False,new_neural_forward=False))
    return dict(path=str(directory/'scores.npz'),sha256=sha(directory/'scores.npz'),count=len(index))

def original_long_CAL(run,indices,method,original_source_calibration):
    """New target CAL only; do not rescore existing original-scale H512 QUERY.

    This is a separate long-history track.  It deliberately does NOT apply the
    RoCA FIT scaler: existing QUERY was scored in original source-score units.
    """
    if run.role!='CAL' or method not in ('history_mean512','persistence'):
        raise PermissionError('Existing long-history target CAL only')
    output=[]
    for index in indices:
        r=baseline_record(run,int(index)-run.start,method=method,history=512)
        s=apply_calibration(r,original_source_calibration)
        if not s['valid']:raise ValueError('Invalid original long-history CAL score')
        output.append(dict(r,score=s['score'],valid=True))
    return output
