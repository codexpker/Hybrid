"""Reusable independent normal-run windows; no implicit data discovery or IO."""
from dataclasses import dataclass


@dataclass
class NormalRun:
    domain: str
    role: str
    run_id: int
    columns: tuple
    values: object

    @property
    def identity(self):return (self.domain,self.role,self.run_id)


def check_runs(runs,role):
    import numpy as np
    if role not in ('FIT','SELECT') or not runs or any(r.role!=role for r in runs):
        raise PermissionError('Only explicit FIT/SELECT normal-run inputs are admitted')
    if len({r.identity for r in runs})!=len(runs):raise ValueError('Duplicate run identity')
    reference=(runs[0].domain,runs[0].columns)
    for r in runs:
        if ((r.domain,r.columns)!=reference or len(set(r.columns))!=len(r.columns)
            or r.values.dtype!=np.float64 or r.values.ndim!=2
            or r.values.shape[1]!=len(r.columns) or not np.isfinite(r.values).all()):
            raise ValueError('Finite float64 runs with identical declared column order required')


def fit_scaler(runs):
    import numpy as np
    check_runs(runs,'FIT')
    values=np.concatenate([r.values for r in runs],axis=0)
    mean=values.mean(axis=0);scale=values.std(axis=0,ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Positive finite FIT-only scales required; no imputation')
    return dict(mean=mean,scale=scale,ddof=0,columns=runs[0].columns,
                fit_ids=[r.identity for r in runs],fit_points=len(values))


class IndependentNormalWindows:
    def __init__(self,runs,scaler,flag,*,window):
        import numpy as np
        if flag not in ('train','val'):raise PermissionError('No CAL/query/test/label loader')
        role='FIT' if flag=='train' else 'SELECT';check_runs(runs,role)
        if not isinstance(window,int) or window<=0 or any(len(r.values)<window for r in runs):
            raise ValueError('Complete positive windows within every original run required')
        if (not scaler.get('fit_ids') or any(x[1]!='FIT' for x in scaler['fit_ids'])
            or tuple(scaler['columns'])!=runs[0].columns
            or any(x[0]!=runs[0].domain for x in scaler['fit_ids'])):
            raise PermissionError('Scaler provenance must be same-domain FIT-only')
        if flag=='train' and scaler['fit_ids']!=[r.identity for r in runs]:
            raise PermissionError('FIT scaler and training run list differ')
        mean=np.asarray(scaler['mean']);scale=np.asarray(scaler['scale'])
        if mean.shape!=(len(runs[0].columns),) or scale.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('Invalid channel-wise normalization')
        self.runs,self.window,self.flag=runs,window,flag
        self.step=1 if flag=='train' else window
        self.normalized=[(r.values-mean)/scale for r in runs]
        self.cumulative=np.r_[0,np.cumsum([1+(len(r.values)-window)//self.step for r in runs])]

    def __len__(self):return int(self.cumulative[-1])

    def footprint(self,index):
        import numpy as np
        if not 0<=index<len(self):raise IndexError(index)
        position=int(np.searchsorted(self.cumulative,index,side='right')-1)
        start=(index-int(self.cumulative[position]))*self.step
        run=self.runs[position]
        return dict(domain=run.domain,role=run.role,run_id=run.run_id,run_position=position,
                    start=start,end_exclusive=start+self.window,input_end_index=start+self.window-1,
                    sample_end_1based=start+self.window)

    def __getitem__(self,index):
        import numpy as np
        f=self.footprint(index)
        values=np.asarray(self.normalized[f['run_position']][f['start']:f['end_exclusive']],dtype=np.float32)
        return values,np.zeros((self.window,1),np.float32)


def inference_footprints(points,*,window,inference_batch_size=1):
    """Metadata-only B1 endpoint availability; no scores/arrays/labels."""
    if inference_batch_size!=1:
        raise PermissionError('Principal public readout must use B1, no future-batch backfill')
    if not isinstance(points,int) or not isinstance(window,int) or window<=0 or points<window:
        raise ValueError('A complete fixed run is required')
    return [dict(start=start,end_exclusive=start+window,endpoint=start+window-1,
                 available_index=start+window-1,inference_batch_size=1,
                 mean32_available_index=start+window-1)
            for start in range(points-window+1)]
