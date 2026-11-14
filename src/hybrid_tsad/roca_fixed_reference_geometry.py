"""Single frozen stage-localization probe, never a layer/k/metric search.

Unit-L2, same-entity same-checkpoint nominal FIT64, mean 5 nearest squared
Euclidean distances. Invalid zero vectors stay UNKNOWN. No QUERY fitting.
"""
import numpy as np

LAYERS=('CNN_flatten','encoder_h0','encoder_h1','encoder_h2','q','qr')
K=5
EPS=1e-12

def exact_identity_ties(features,labels,distance):
    """Exact raw-byte/unit-byte and radial-score ties, NOT tolerance clustering."""
    x=np.ascontiguousarray(features);y=np.asarray(labels);s=np.asarray(distance,np.float64)
    def describe(a):
        a=np.ascontiguousarray(a)
        if a.ndim==1:a=a[:,None]
        keys=a.view(np.dtype((np.void,a.dtype.itemsize*a.shape[1]))).reshape(-1)
        _,codes=np.unique(keys,return_inverse=True);n=int((y==0).sum());p=int((y==1).sum())
        normal=np.bincount(codes[y==0],minlength=int(codes.max())+1);anomaly=np.bincount(codes[y==1],minlength=len(normal))
        return dict(normal_unique=int((normal>0).sum()),anomaly_unique=int((anomaly>0).sum()),
            normal_largest_tie_fraction=float(normal.max()/n) if n else None,
            anomaly_largest_tie_fraction=float(anomaly.max()/p) if p else None,
            normal_anomaly_equal_pair_fraction=float(np.dot(normal.astype(float),anomaly)/(n*p)) if n*p else None)
    return dict(raw_row_bytes=describe(x),unit_float64_row_bytes=describe(normalized(x)[0]),
        radial_distance_float64_bytes=describe(s),
        interpretation='Exact stored bytes/unit computation/radial-score ties differ; no angular tolerance or score quantization search')

def normalized(features):
    x=np.asarray(features,np.float64)
    if x.ndim!=2:raise ValueError('Explicit N,D features required')
    finite=np.isfinite(x).all(1);norm=np.linalg.norm(np.where(np.isfinite(x),x,0),axis=1)
    valid=finite & (norm>EPS)
    u=np.zeros_like(x);u[valid]=x[valid]/norm[valid,None]
    return u,valid

def fixed_five_nn(reference,query,*,reference_role,block_points=1024,normalization='unit'):
    if reference_role!='FIT':raise PermissionError('Only frozen nominal FIT reference, never QUERY/CAL')
    if len(reference)!=64:raise PermissionError('Original fixed FIT64 rows must remain, invalid rows not replaced')
    if normalization not in ('unit','raw'):raise PermissionError('Only fixed raw-amplitude/unit normalization contrast')
    r,rv=normalized(reference);q,qv=normalized(query)
    if normalization=='raw':r=np.asarray(reference,np.float64);q=np.asarray(query,np.float64)
    if r.shape[1]!=q.shape[1]:raise ValueError('Same stage/checkpoint coordinate shape required')
    score=np.full(len(q),np.nan);valid=np.zeros(len(q),bool)
    if rv.sum()<K:
        return dict(score=score,valid=valid,reference_valid=int(rv.sum()),reason='FEWER_THAN5_VALID_FIT_REFERENCES')
    r=r[rv]
    for start in range(0,len(q),block_points):
        stop=min(start+block_points,len(q));ok=qv[start:stop]
        z=q[start:stop][ok]
        # Canonical float64 direct differences: norm/dot cancellation near
        # identical low-spread vectors would otherwise create false ties.
        distance=np.empty((len(z),len(r)),np.float64)
        for j,reference in enumerate(r):
            delta=z-reference
            distance[:,j]=np.square(delta).sum(1)
        selected=np.partition(distance,K-1,axis=1)[:,:K].mean(1)
        positions=np.flatnonzero(ok)+start;score[positions]=selected;valid[positions]=True
    return dict(score=score,valid=valid,reference_valid=int(rv.sum()),reason=None,
        k=K,metric='mean5_nearest_squared_Euclidean',normalization=normalization,normal_reference='nominal_FIT_not_certified_pure')
