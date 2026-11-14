from pathlib import Path
import sys,json,hashlib,numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'hybrid_ad/src'))
from hybrid_tsad.hybrid_v1 import controlled_mechanism as g
from hybrid_tsad.hybrid_v1 import controlled_wave as w
D=P/'controlled';D.mkdir(exist_ok=False);N=2560;BURN=4096;EVENT=(768,1536);NS='BLINDSPOT_FIXED_CONTROL_20261113'
def pair(r):return g.generate_pair('cross_mixed_sign',run_id=r,role=NS,n=N,burn_in=BURN,event=EVENT,d0=12,d1=20)
entries={};meta={}
def save(role,arrays,ids):
 f=D/(role+'.npz');np.savez_compressed(f,values=np.stack(arrays));entries[role]=dict(path=str(f.relative_to(P)).replace('\\','/'),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),shape=[len(arrays),N,3],run_ids=ids)
for role,ids in [('FIT',list(range(10000,10008))),('SELECT',[20000,20001]),('CAL',[30000,30001])]:save(role,[pair(r)[0]['N']['values']for r in ids],ids)
fit=np.concatenate([pair(r)[0]['N']['values']for r in range(10000,10008)]);sigma=fit.astype(float).std(0)
alltruth={}
for case in ['subset_offset','relation_break','normal_sync','normal_drift','no_coupling','sham']:
 arrays=[];ys=[]
 for r in range(90000,90005):
  p,t=pair(r);x=p['N']['values'].copy();s,e=EVENT;idx=t['retained_indices'];y=np.zeros(N,np.int8)
  if case=='subset_offset':x[s:e,1]+=2*sigma[1];y[s:e]=1
  elif case=='relation_break':
   driver,eps,seed=w._ar1('cross_mixed_sign',NS,r,'break_driver',len(t['latents']['u']));x[s:e,1]=.3*driver[idx[s:e]-12]+np.sqrt(.91)*t['latents']['zplus'][idx[s:e]];x[s:e,2]=-.3*driver[idx[s:e]-12]+np.sqrt(.91)*t['latents']['zminus'][idx[s:e]];y[s:e]=1
  elif case=='normal_sync':
   delta=np.zeros(len(t['latents']['u']));delta[idx[s]:idx[e-1]+1]=1.;x[:,0]+=delta[idx];x[:,1]+=.3*delta[idx-12];x[:,2]-=.3*delta[idx-12]
  elif case=='normal_drift':x+=np.arange(N)[:,None]/(N-1)
  elif case=='no_coupling':x=np.column_stack([t['latents'][k][idx]for k in ['u','zplus','zminus']]).astype(np.float32)
  arrays.append(x);ys.append(y)
 save(case,arrays,list(range(90000,90005)));alltruth[case]=np.stack(ys)
np.savez_compressed(D/'TRUTH.npz',**alltruth)
(P/'controlled'/'ARRAYS.json').write_text(json.dumps(entries,indent=2))
base=json.loads((R/'preparations/hybrid_native_predictive_cycle_20261113_v1/configs/TEP.json').read_text());(D/'MODEL.json').write_text(json.dumps(base,indent=2))
(D/'DATA_DEFINITION.json').write_text(json.dumps(dict(n=N,burn=BURN,H=100,event=EVENT,fully_window_steady=[868,1536],source_namespace=NS,generator=str(Path(g.__file__)),generator_sha256=hashlib.sha256(Path(g.__file__).read_bytes()).hexdigest(),equation='u,zplus,zminus independent AR1=.8lag1+.6epsilon; yplus=.3u[t-12]+sqrt(.91)zplus; yminus=-.3u[t-12]+sqrt(.91)zminus',subset='target1 +2 FITsigma only768:1536',sigma=sigma.tolist(),normal_sync='driver +1 on event, propagate original +/- .3 at delay12 including recovery tail',normal_drift='all3 linear0to1',negative_scope='sync/drift/uncoupled system are OOD normal alternatives absent originalFIT; not certified training-invariance; uncoupled removes +/- coupling across whole run, not sameinput contradictory labels',seed_rule='old seed_for(family,namespace,runid,stream)',no_test_selection=True),indent=2))
print('CONTROLLED_ARRAYS_READY',len(entries))
