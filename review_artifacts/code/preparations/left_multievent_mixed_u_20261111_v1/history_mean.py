"""Fixed H512 comparator; only past values predict the current observation."""
import hashlib,json,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name/'HistoryMean512';O.mkdir(parents=True,exist_ok=False);start=time.monotonic()
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load(f):
    with np.load(f)as z:return z['values'].astype(np.float64)
def residual(x):
    cs=np.concatenate([np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)],axis=0);pred=(cs[512:-1]-cs[:-513])/512;return x[512:]-pred
rng=np.random.default_rng(2025);fixture=rng.normal(size=(1024,31));early=residual(fixture);changed=fixture.copy();changed[700:]+=20;assert np.array_equal(early[:188],residual(changed)[:188]);target=fixture.copy();target[600]+=2;assert np.allclose(residual(target)[88]-early[88],2.)
sc=json.loads((P/'FIT_ONLY_SCALER.json').read_text());mean=np.asarray(sc['mean']);std=np.asarray(sc['scale']);cal=(load(P/'data/CAL.npz')-mean)/std;cr=residual(cal);selected=np.linspace(446173,494999,512,dtype=np.int64);scale=np.maximum(np.median(np.abs(cr[selected-446173]),axis=0),1e-6);banks=[]
for role,values,offset in [('CAL',cal,445661),('QUERY',(load(P/'data/QUERY.npz')-mean)/std,608023)]:
    res=residual(values);score=(np.abs(res)/scale).mean(axis=1);ix=np.arange(offset+512,offset+len(values));f=O/(role+'.npz');np.savez_compressed(f,index=ix,score=score,residual=res,available_index=ix,valid=np.ones(len(ix),bool));banks.append(dict(role=role,path=str(f),sha256=sha(f),points=len(ix)))
tau=float(np.quantile((np.abs(cr[selected-446173])/scale).mean(axis=1),.99,method='higher'));(O/'CALIBRATION.json').write_text(json.dumps(dict(selected_index=selected.tolist(),perchannel_residual_scale=scale.tolist(),threshold=tau,q=.99,method='higher',source_scaler_sha256=sha(P/'FIT_ONLY_SCALER.json')),indent=2));(O/'RECEIPT.json').write_text(json.dumps(dict(status='COMPLETED',banks=banks,normal_future_input_permission='same normalCAL/sourceFIT stats, strictpast512 predictor',no_label_read=True,CPU_future_target_exclusion_test=True,seconds=time.monotonic()-start),indent=2));print(json.dumps(dict(status='H512_COMPLETE',seconds=time.monotonic()-start)))
