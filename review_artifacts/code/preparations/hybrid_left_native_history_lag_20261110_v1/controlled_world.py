"""Pre-fixed lag256 controlled world; no measured test-label model choice."""
import json,time,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
CASES=('self_delay','cross_delay','joint_delay','cross_direction','normal_drift','normal_sync','independent_shift','sham')
POSITIVE=set(CASES[:4]);EVENT=(3072,5120)
def world(seed,n,case='normal'):
 rng=np.random.default_rng(seed);burn=32768;N=burn+n;eps,u,eta,w=rng.normal(size=(4,N));z=np.zeros(N)
 for t in range(256,N):
  at=t-burn;active=EVENT[0]<=at<EVENT[1];lag=128 if active and case in('self_delay','joint_delay')else 256
  z[t]=.7*z[t-lag]+np.sqrt(.51)*eps[t]
 ts=np.arange(N);active=(ts-burn>=EVENT[0])&(ts-burn<EVENT[1]);lag=np.where(active&np.isin(case,['cross_delay','joint_delay']),128,256);sign=np.where(active&(case=='cross_direction'),-1.,1.)
 y=.7*sign*u[np.maximum(0,ts-lag)]+np.sqrt(.51)*eta
 if case=='independent_shift':w=np.where(active,w[np.maximum(0,ts-128)],w)
 latent=np.stack((z,u,y,w),axis=-1)[burn:]
 # Same bounded drift/synchronous-normal family appears in every normal FIT.
 amplitude=rng.uniform(.1,.3);phase=rng.uniform(0,2*np.pi);states=rng.choice([-.25,0.,.25],size=(N+1023)//1024)
 mode=0 if case=='normal_drift' else 1 if case=='normal_sync' else seed%3
 sine=amplitude*np.sin(2*np.pi*ts/2048+phase);level=states[ts//1024]
 mu=(sine if mode==0 else level if mode==1 else sine+level)[burn:]
 values=(latent+mu[:,None]).astype(np.float32);labels=np.zeros(n,np.int8)
 if case in POSITIVE:labels[EVENT[0]:EVENT[1]]=1
 return values,labels,dict(seed=seed,case=case,normal_context_mode=['sine','states','sine+states'][mode],latent_mean=latent.mean(0).tolist(),latent_std=latent.std(0).tolist(),observed_mean=values.mean(0).tolist(),observed_std=values.std(0).tolist(),drift_amplitude=amplitude,normal_state_levels=[-.25,0,.25],steady_interval=[3584,5120]if case in POSITIVE else None)
def generate():
 start=time.monotonic();out=P/'controlled_data';out.mkdir(exist_ok=False);rows=[]
 for role,count,n,seed0 in [('FIT',4,8192,10000),('SELECT',2,4096,20000),('CAL',2,4096,30000)]:
  for i in range(count):
   x,y,meta=world(seed0+i,n);path=out/(role+'_'+str(i)+'.npz');np.savez_compressed(path,values=x);rows.append(dict(role=role,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),meta=meta))
 for case in CASES:
  for i in range(5):
   x,y,meta=world(90000+i,8192,case);path=out/(case+'_'+str(i)+'.npz');np.savez_compressed(path,values=x);truth=out/(case+'_'+str(i)+'_truth.npz');np.savez_compressed(truth,labels=y);rows.append(dict(role='QUERY',path=str(path),truth=str(truth),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),meta=meta))
 manifest=dict(theory='z[t]=.7z[t-256]+sqrt(.51)eps; y[t]=.7u[t-256]+sqrt(.51)eta; u,w IID; all latent marginal N0,1; burn32768; test relation changes delay128 or cross sign; observed same normal shared drift/state family',case_seeds_paired=True,independent_test_worlds=5,roles=rows,event=[3072,5120],transition=[3072,3584],steady=[3584,5120],no_test_rescaling=True,seconds=time.monotonic()-start)
 (P/'CONTROLLED_DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(status='CONTROLLED_DATA_GENERATED',arrays=len(rows),seconds=manifest['seconds'])),flush=True)
if __name__=='__main__':generate()

