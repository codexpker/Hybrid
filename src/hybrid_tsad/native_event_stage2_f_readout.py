"""Fixed F endpoint scores and matched operating evaluation; no label access during readout."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch_frft.frft_module import ifrft
from .native_event_stage2_f import *
from .native_v_conditioning import WEIGHT_SHA

@torch.inference_mode()
def score(w,domain):
    device=setup(12);root=w/OUT/domain
    folder,task=target_task(w,domain)
    models=[];weights=[]
    for arm in ARMS:
        m=initial(w,arm,device).requires_grad_(False)
        models.append(m)
        if arm=='official': weights.append(WEIGHT_SHA)
        else: weights.append(read(w/BASE/'MODELS_FROZEN.json')['arms'][arm]['checkpoint_sha256'])
    for arm in ARMS:
        d=root/arm;r=read(d/'ENDPOINT.json');ck=asset(d,'endpoint.pt',r['checkpoint_sha256'])
        m=initial(w,arm,device);m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model']);models.append(m.requires_grad_(False));weights.append(sha(ck))
    names=[f'{a}_{p}' for p in ('Z','F') for a in ARMS]
    orders=[];models[0].net.register_forward_hook(lambda _m,_a,o:orders.append(o.squeeze()))
    for role in ('CAL','QUERY'):
        dest=root/role;dest.mkdir(exist_ok=True)
        ix=role_index(w,folder,task,role);x,_,origin,_=role_values(w,folder,task,role)
        reuse=None
        if domain=='PSM':
            cache=w/'results/native_e_vs_v_psm_20261123_v1/banks.npz'
            receipt=read(cache.with_name('SCORES.json'));assert sha(cache)==receipt['sha256']
            with np.load(cache) as z:
                assert receipt['scaler_sha256']==task['scaler_sha256']
                assert np.array_equal(z['index'][ix-99],ix)
                reuse=z['query'][ix-99][:,[0,2,3]]
        else:
            cache=w/'results/native_e_vs_v_smd28_20261123_v1/machine-3-2.npz'
            receipt=read(cache.with_suffix('.json'));assert sha(cache)==receipt['sha256']
            with np.load(cache) as z:
                oldix=z['cal_index'] if role=='CAL' else z['query_index']
                where=np.searchsorted(oldix,ix)
                if receipt['scaler_sha256']==task['scaler_sha256'] and (where<len(oldix)).all() and np.array_equal(oldix[where],ix):
                    reuse=z['cal' if role=='CAL' else 'query'][where][:,[0,2,3]]
        values=torch.from_numpy(x).to(device)
        chunks=[];cost=0
        for lo in range(0,len(ix),2048):
            path=dest/f'part_{lo:05}.npz';meta=path.with_suffix('.json')
            if meta.exists():
                r=read(meta);assert sha(path)==r['sha256'] and r['weights']==weights
                with np.load(path) as z:chunks.append(z['score'])
                cost+=r['seconds'];continue
            tic=time.monotonic();out=torch.empty((min(2048,len(ix)-lo),6),device=device)
            for j,end in enumerate(ix[lo:lo+2048]):
                t=int(end)-origin;assert t>=99
                win=values[t-99:t+1][None];orders.clear()
                with fixed_native_rng(device):enc,copies=models[0](win,norm=0,mask_mode='c',copies=10)
                if reuse is not None:out[j,:3]=torch.as_tensor(reuse[lo+j],device=device)
                indices=list(range(3,6)) if reuse is not None else list(range(6))
                decoded=ifrft(torch.stack([models[k].decoder(enc) for k in indices]),orders[0],dim=-1).abs()
                for n,k in enumerate(indices):
                    cp=decoded[n].reshape(10,1,x.shape[1],100).permute(0,1,3,2)
                    if lo==j==0:
                        with fixed_native_rng(device):_,direct=models[k](win,norm=0,mask_mode='c',copies=10)
                        torch.testing.assert_close(cp,direct,rtol=0,atol=0)
                    out[j,k]=models[k].cal_anomaly_score(win,cp,None,1)[0,99]
            ar=out.cpu().numpy();assert np.isfinite(ar).all()
            np.savez_compressed(path,index=ix[lo:lo+2048],score=ar)
            seconds=time.monotonic()-tic;cost+=seconds;chunks.append(ar)
            dump(meta,dict(sha256=sha(path),weights=weights,seconds=seconds,gpu=os.environ['CUDA_VISIBLE_DEVICES']))
            print(domain,role,lo+len(ar),len(ix),flush=True)
        bank=dest/'SCORES.npz';np.savez_compressed(bank,index=ix,window_start=ix-99,available_index=ix,keys=names,score=np.vstack(chunks))
        dump(dest/'SCORES.json',dict(sha256=sha(bank),weights=weights,seconds=cost,scaler_sha256=task['scaler_sha256'],reused_Z=reuse is not None,labels_read=False))

def evaluate(w):
    root=w/OUT;methods=[];events=[];metric=load_author_metric(asset(w,'preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py',METRIC_SHA))
    for domain in SUPPORT:
        folder,task=target_task(w,domain)
        if domain=='PSM':y=np.load(asset(w,'data/common_psm_20261028_v1/test_label.npy',LABEL_SHA)).reshape(-1)
        else:
            r=read(folder/'EVALUATION_BINDING.json');y=np.loadtxt(asset(w,r['relative'],r['sha256']),delimiter=',',dtype=int).reshape(-1)
        banks=[]
        for role in ('CAL','QUERY'):
            d=root/domain/role;r=read(d/'SCORES.json')
            with np.load(asset(d,'SCORES.npz',r['sha256'])) as z:banks.append((z['index'],z['score'],z['keys']))
        (ci,cal,keys),(qi,query,_) = banks
        for j,name in enumerate(keys):
            arm,phase=name.rsplit('_',1);tau=exact_cal(ci,cal[:,j],ci)
            rank,_=continuous_metrics(y[qi],query[:,j],metric)
            rows,details,_,_=evaluate_score(qi,query[:,j],y,tau,qi,query_start=int(qi[0]),query_stop=int(qi[-1]+1),original_length=len(y),expected_events=ELIGIBLE_EVENTS[domain],ranking=rank)
            causal_timing(rows,details)
            for row in rows:
                e=[{k:'' if v is None else str(v) for k,v in r.items()} for r in details if r['stream']==row['stream']]
                row.update(group_operating([row],e));methods.append(dict(domain=domain,initialization=arm,phase=phase,**row,**rank))
            events.extend(dict(domain=domain,initialization=arm,phase=phase,**r) for r in details)
    write_csv(root/'methods.csv',methods);write_csv(root/'events.csv',events)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['score','evaluate']);p.add_argument('--domain');a=p.parse_args()
    if a.action=='score':score(Path.cwd(),a.domain)
    else:evaluate(Path.cwd())
