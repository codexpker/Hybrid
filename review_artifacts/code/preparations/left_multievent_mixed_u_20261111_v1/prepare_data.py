"""Existing SWaT CSV to isolated roles and fixed real-event replacement plan."""
import hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
P=Path(__file__).resolve().parent;R=P.parents[1];start=time.monotonic();fixed=json.loads((P/'FIXED_EXPERIMENT.json').read_text());mapping=json.loads((P/'OFFICIAL_31_CHANNEL_MAPPING.json').read_text())['mapping'];names=[x['raw_name']for x in mapping if not x['removed']];positions={n:i for i,n in enumerate(names)}
source=R/'data/SWAT.csv';assert source.stat().st_size==976006181;D=P/'data';D.mkdir(exist_ok=False);wide=np.lib.format.open_memmap(D/'ALL_VALUES.npy',mode='w+',dtype=np.float64,shape=(944919,31));count=np.zeros(31,np.int64)
for chunk in pd.read_csv(source,chunksize=1000000):
    for name,g in chunk.groupby('cols',sort=False):
        if name in positions:
            k=positions[name];ix=g['date'].to_numpy(np.int64)-1;v=g['data'].to_numpy(np.float64);assert np.isfinite(v).all();wide[ix,k]=v;count[k]+=len(ix)
assert np.all(count==944919);wide.flush();sourcefit=wide[0:296020];sc=StandardScaler().fit(sourcefit);scaler=dict(mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),var=sc.var_.tolist(),FIT=[0,296020],FIT_only=True,retained_raw_names=names)
(P/'FIT_ONLY_SCALER.json').write_text(json.dumps(scaler,indent=2));roles=[]
for role,key in [('FIT','source_FIT'),('U_FIT','U_FIT'),('SELECT','SELECT'),('CAL','CAL'),('DONOR','DONOR'),('QUERY','QUERY')]:
    lo,hi=fixed[key];f=D/(role+'.npz');np.savez_compressed(f,values=np.asarray(wide[lo:hi]),observed=np.ones((hi-lo,31),bool));roles.append(dict(role=role,shape=[hi-lo,31],original_start=lo,end=hi,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
(P/'INPUT_QUALIFICATION.json').write_text(json.dumps(dict(roles=roles,original_csv_sha256='a8bad2c5f770dbd04e7ad44f58f565caf1c4fe1a68ab62d2d8625adb4d4cdceb',channels=names,no_label_in_role_arrays=True),indent=2))
# Only the separate plan builder sees public labels. Learner receives numerical windows.
with np.load(P/'SWAT_LABEL_METADATA.npz')as z:label=z['label'].astype(bool)
donlo,donhi=fixed['DONOR'];y=label[donlo:donhi];edges=np.diff(np.r_[False,y,False].astype(np.int8));es=np.flatnonzero(edges==1)+donlo;ee=np.flatnonzero(edges==-1)+donlo;assert len(es)==13 and np.all(ee-es>=192)
bad=y.copy()
for lo,hi in zip(es,ee):bad[max(0,lo-donlo-544):min(len(bad),hi-donlo+544)]=True
cs=np.r_[0,np.cumsum(bad)];normalstarts=np.flatnonzero(cs[192:]-cs[:-192]==0)+donlo;assert len(normalstarts)>0
rng=np.random.default_rng(2025);base=rng.integers(0,98673-192+1,size=(2000,128));r=np.random.default_rng(84551);selected=r.choice(256000,size=25600,replace=False);event_ids=np.arange(25600)%13;r.shuffle(event_ids);bd=np.full((2000,128),-1,np.int64);cd=bd.copy();event=np.full_like(bd,-1);mask=np.zeros_like(bd,bool)
for flat,e in zip(selected,event_ids):
    row,col=divmod(int(flat),128);b=int(r.integers(es[e],ee[e]-192+1));p=int(np.searchsorted(normalstarts,b));candidates=normalstarts[max(0,p-1):min(len(normalstarts),p+1)];c=int(min(candidates,key=lambda t:(abs(int(t)-b),int(t))));assert label[b:b+192].all()and not label[c:c+192].any();bd[row,col]=b-donlo;cd[row,col]=c-donlo;event[row,col]=e;mask[row,col]=True
np.savez_compressed(P/'PAIRED_PLAN.npz',base_starts=base,B_donor_starts=bd,C_donor_starts=cd,event_id=event,replacement_mask=mask,selected_flat_order=selected,donor_event_starts=es,donor_event_ends=ee,normal_legal_starts=normalstarts)
out=dict(status='PASS',updates=2000,batch_size=128,base_FIT_footprint=fixed['U_FIT'],DONOR_footprint=fixed['DONOR'],QUERY_footprint=fixed['QUERY'],guard544=True,window_exposures=256000,replaced_exposures=int(mask.sum()),unique_true_events=13,event_presentations=np.bincount(event_ids,minlength=13).tolist(),positive_time_exposures=int(mask.sum())*192,potential_positive_time_channel_exposures=int(mask.sum())*192*31,physical_affected_channels='UNKNOWN',C_distance_to_every_positive_ge544=True,C_nearest_start_tie_earlier=True,no_QUERY_donor=True,no_labels_in_model_roles=True,plan_sha256=hashlib.sha256((P/'PAIRED_PLAN.npz').read_bytes()).hexdigest(),scaler_sha256=hashlib.sha256((P/'FIT_ONLY_SCALER.json').read_bytes()).hexdigest(),seconds=time.monotonic()-start)
(P/'DATA_PLAN_GATE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
