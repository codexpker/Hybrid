"""Existing public originals -> separate physical FIT/SELECT/CAL/TEST files."""
from pathlib import Path
import json,hashlib,time,os,gc
import numpy as np,pyreadr
P=Path(__file__).resolve().parent;R=P.parents[1];D=P/'data';D.mkdir(exist_ok=False);start=time.monotonic();rows=[]
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(domain,role,v,run,offset,fault=None):
    assert v.ndim==3 and np.isfinite(v).all();dest=D/(domain+'_'+role+'.npz');np.savez_compressed(dest,values=v,run=np.asarray(run),fault=np.zeros(len(v),int)if fault is None else np.asarray(fault),original_start=np.asarray(offset),observed=np.ones(v.shape,bool));rows.append(dict(domain=domain,role=role,path=str(dest),sha256=sha(dest),shape=list(v.shape),runs=list(map(int,run)),original_start=list(map(int,np.atleast_1d(offset)))))
def scaler(domain,v,cols):
    mean=v.mean((0,1));sd=v.std((0,1));scale=np.where(sd>0,sd,1.);dest=P/(domain+'_FIT_ONLY_SCALER.json');dest.write_text(json.dumps(dict(mean=mean.tolist(),scale=scale.tolist(),raw_std=sd.tolist(),columns=cols,fit_role='FIT only',shape=list(v.shape)),indent=2));return sha(dest)
rawpath=R/'data/tep_rieth2017_v1/TEP_FaultFree_Training.RData';df=next(iter(pyreadr.read_r(str(rawpath)).values()));assert df.shape==(250000,55);cols=['xmeas_9','xmv_10','xmeas_21'];allrun=[]
for run in range(1,101):
    q=df[df.simulationRun==run].sort_values('sample');assert np.array_equal(q['sample'],np.arange(1,501))and(q.faultNumber==0).all();allrun.append(q[cols].to_numpy(np.float64))
v=np.stack(allrun);del df,allrun;gc.collect();scaler('TEP',v[:60],cols)
for role,lo,hi in [('FIT',0,60),('SELECT',60,80),('CAL',80,100)]:save('TEP',role,v[lo:hi],range(lo+1,hi+1),np.zeros(hi-lo,int))
queryroot=R/'results/tep_relation_confirmation_20261113_v1/data'
for source,role in [('faulty','QUERY'),('normal','NORMAL_TEST')]:
    with np.load(queryroot/(source+'.npz'))as z:save('TEP',role,z['values'],z['fault_run'][:,1],np.zeros(len(z['values']),int),z['fault_run'][:,0])
prefix=R/'preparations/left_baseline_build_20261101_v1/normal_prefix_R_v1/prefix.npy';assert sha(prefix)=='f722bce0b19324d85c7b5691efe73453eb56022e867ccc1ba47bb3406787df38';v=np.load(prefix);assert v.shape==(132481,25)and np.isfinite(v).all()
scaler('PSM',v[None,:79488],['channel_'+str(i)for i in range(25)])
for role,lo,hi in [('FIT',0,79488),('SELECT',80032,105984),('CAL',106528,132481)]:save('PSM',role,v[None,lo:hi],[0],[lo])
query=R/'data/common_psm_raw_20261029_v1/test.npy';assert sha(query)=='6592518615d690ab153592b6dd0220c424441a0d4ee7bac87c3f89fe942893f5';v=np.load(query);assert v.shape==(87841,25);save('PSM','QUERY',v[None],[0],[0])
plans=[]
for domain,H,shape in [('TEP',100,(60,500,3)),('PSM',192,(1,79488,25))]:
    n=shape[0]*(shape[1]-H);rng=np.random.default_rng(2025);order=np.stack([rng.permutation(n).astype(np.int32)for _ in range(20)]);f=D/(domain+'_DRAWPLAN.npz');np.savez_compressed(f,order=order);plans.append(dict(domain=domain,path=str(f),sha256=sha(f),shape=list(order.shape),optimizer_updates=20*int(np.ceil(n/128))))
gate=dict(status='PHYSICAL_ROLES_READY',pid=os.getpid(),seconds=time.monotonic()-start,roles=rows,plans=plans,source_files=[dict(path=str(rawpath),sha256=sha(rawpath)),dict(path=str(prefix),sha256=sha(prefix)),dict(path=str(query),sha256=sha(query))],train_process_permissions=['FIT','SELECT'],no_labels_or_test_statistics_used=True)
(P/'DATA_GATE.json').write_text(json.dumps(gate,indent=2));print(json.dumps(dict(status='READY',roles=len(rows),seconds=gate['seconds'],plans=plans)),flush=True)
