"""Source-selected position-aligned decoder evaluation on the fixed W100 grid."""
import argparse
import fcntl
import os
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from . import native_full_window_readout as prior
from .native_v_conditioning import load_native,sha,state_hashes,fixed_native_rng
from .native_v_conditioning_runner import read,dump,setup

OUT=Path('results/native_position_aligned_20261124_v1')
ARMS=('copy','terminal','all_positions')


def models(w):
    root=w/OUT;seal=read(root/'ALL_MODELS_FROZEN.json')
    base=load_native(w,'cuda');initial=base.state_dict();nets=[];digests=[]
    for arm in ARMS:
        r=seal['arms'][arm];path=root/r['checkpoint_relative']
        assert sha(path)==r['checkpoint_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['update']==r['selected_step']
        m=deepcopy(base);m.load_state_dict(saved['model'],strict=True);m.eval().requires_grad_(False)
        assert all(torch.equal(initial[k],v) for k,v in m.state_dict().items() if not k.startswith('decoder.'))
        nets.append(m);digests.append(sha(path))
    old,oldkeys,oldsha=prior.models(w)
    reuse=[]
    for m in nets:
        same=[i for i,n in enumerate(old) if all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items())]
        reuse.append(same[0] if same else None)
    del old
    return base,nets,digests,reuse


@torch.inference_mode()
def score(w,lane,lanes):
    from torch_frft.frft_module import ifrft
    setup(12);base,nets,digests,reuse=models(w)
    identity=[state_hashes(m) for m in nets]
    order=[];base.net.register_forward_hook(lambda _m,_a,x:order.append(x.squeeze()))
    root=w/OUT/'readout';root.mkdir(exist_ok=True,parents=True)
    active=[i for i,j in enumerate(reuse) if j is None]
    tasks=prior.jobs(w)
    for domain,entity,n in tasks[lane::lanes]+tasks:
        dest=root/domain;dest.mkdir(parents=True,exist_ok=True)
        bank=dest/(entity+'.npz');receipt=bank.with_suffix('.json')
        with bank.with_suffix('.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if receipt.exists():
                r=read(receipt);assert sha(bank)==r['sha256'] and r['weights_sha256']==digests
                continue
            tic=time.monotonic();oldbank=w/prior.OUT/domain/(entity+'.npz');oldr=read(oldbank.with_suffix('.json'))
            assert sha(oldbank)==oldr['sha256']
            windows,binding=prior.windows(w,domain,entity)
            assert binding==oldr['input_binding'] and n==binding['original_points']
            with np.load(oldbank) as z:
                ix=z['index'];assert np.array_equal(ix,np.arange(n//100*100))
                values=np.empty((len(ix),3),np.float32)
                for k,j in enumerate(reuse):
                    if j is not None:values[:,k]=z['V'][:,j]
            for b,window in enumerate(windows) if active else []:
                x=torch.from_numpy(np.ascontiguousarray(window)).cuda()[None];order.clear()
                with fixed_native_rng('cuda'):encoded,_=base(x,norm=0,mask_mode='c',copies=10)
                decoded=ifrft(torch.stack([nets[k].decoder(encoded) for k in active]),order[0],dim=-1).abs()
                for j,k in enumerate(active):
                    copies=decoded[j].reshape(10,1,x.shape[-1],100).permute(0,1,3,2)
                    if b==0:
                        with fixed_native_rng('cuda'):_,direct=nets[k](x,norm=0,mask_mode='c',copies=10)
                        torch.testing.assert_close(copies,direct,rtol=0,atol=0)
                    values[b*100:(b+1)*100,k]=nets[k].cal_anomaly_score(x,copies,None,1)[0].cpu().numpy()
            assert identity==[state_hashes(m) for m in nets] and np.isfinite(values).all()
            tmp=bank.with_suffix('.part.npz')
            np.savez_compressed(tmp,V=values,keys=ARMS,index=ix,window_start=ix//100*100,window_end_exclusive=(ix//100+1)*100,available_index=(ix//100+1)*100-1)
            tmp.replace(bank)
            dump(receipt,dict(domain=domain,entity=entity,sha256=sha(bank),keys=ARMS,weights_sha256=digests,reused_old_columns=reuse,prior_bank_sha256=oldr['sha256'],input_binding=binding,points=len(ix),omitted_tail=n-len(ix),seconds=time.monotonic()-tic,device=os.environ['CUDA_VISIBLE_DEVICES'],B=1,copies=10,mask_seed=84551,correction=1,frozen_verified=True,query_labels_read=False))
            print(domain,entity,len(ix),round(time.monotonic()-tic,2),flush=True)
    dump(root/f'LANE_{lane}.json',dict(weights_sha256=digests,reused_old_columns=reuse,frozen_verified=identity==[state_hashes(m) for m in nets]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lane',type=int,default=0);p.add_argument('--lanes',type=int,default=4);a=p.parse_args();score(Path.cwd(),a.lane,a.lanes)
