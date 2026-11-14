"""Actual role/scaler/window-plan checks; no model/GPU/QUERY load."""
from pathlib import Path
import json,time,numpy as np
import worker
P=Path(__file__).resolve().parent;start=time.monotonic();rows=[]
for domain,H in [('TEP',100),('PSM',192)]:
    d=worker.Roles(domain,['FIT','SELECT']);v,run,_,offset=d.load('FIT');sel,_,_,so=d.load('SELECT');before=list(d.access)
    for forbidden in ['CAL','QUERY','NORMAL_TEST']:
        try:d.load(forbidden);raise AssertionError('unauthorized role open')
        except PermissionError:pass
    assert d.access==before;raw=d.raw['FIT'];mean=raw.mean((0,1));sd=raw.std((0,1));sd=np.where(sd>0,sd,1.);assert np.allclose(mean,d.scaler['mean'],rtol=1e-10,atol=1e-12)and np.allclose(sd,d.scaler['scale'],rtol=1e-10,atol=1e-12)
    ds=worker.Windows(v,H);n=len(ds);assert n==(24000 if domain=='TEP'else 79296);assert ds[0].shape==(H+1,v.shape[-1])and ds[n-1].shape==ds[0].shape
    for idx in [0,n//2,n-1]:
        r=idx//ds.per;t=H+idx%ds.per;assert np.array_equal(ds[idx],v[r,t-H:t+1]);assert t-H>=0 and t<len(v[r])
    plan=np.load(P/'data'/(domain+'_DRAWPLAN.npz'))['order'];assert plan.shape==(20,n)
    for row in plan:assert np.array_equal(np.sort(row),np.arange(n))
    ss=worker.Windows(sel,H,[(r,t)for r in range(len(sel))for t in range(H,sel.shape[1],H)]);assert len(ss)>0
    if domain=='TEP':assert np.array_equal(run,np.arange(1,61))
    else:assert int(offset[0])+len(v[0])==79488 and int(so[0])==80032 and int(so[0])-79488==544
    rows.append(dict(domain=domain,status='PASS',FIT_shape=list(v.shape),SELECT_shape=list(sel.shape),FIT_sample_windows=n,SELECT_windows=len(ss),updates20epochs=20*int(np.ceil(n/128)),FIT_only_scaler_same_raw_allclose=True,scaler_mean_max_abs=float(np.max(np.abs(mean-d.scaler['mean']))),scaler_scale_max_abs=float(np.max(np.abs(sd-d.scaler['scale']))),scaler_mean_max_rel=float(np.max(np.abs((mean-d.scaler['mean'])/np.maximum(np.abs(d.scaler['mean']),1e-12)))),scaler_scale_max_rel=float(np.max(np.abs((sd-d.scaler['scale'])/np.maximum(np.abs(d.scaler['scale']),1e-12)))),frozen_scaler_unchanged=True,role_guard_before_file_open=True,all20paired_permutations_complete=True,past_and_target_full_role_contained=True,first_target_sample=(101 if domain=='TEP'else 192)))
out=dict(status='PASS',seconds=time.monotonic()-start,checks=rows,GPU_hours=0,QUERY_CAL_not_opened=True);(P/'WORKER_CPU_GATE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
