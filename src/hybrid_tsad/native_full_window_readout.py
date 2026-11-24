"""Fixed native TimeRadar full-window V/E readout over all public test entities."""
import argparse
import fcntl
import os
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from . import native_event_smd28 as smd
from . import native_event_space as nasa
from .native_e_vs_v_psm import inputs
from .native_v_conditioning import fixed_native_rng,sha,state_hashes,SOURCE_SHA,WEIGHT_SHA,actual_mask_bits
from .native_v_conditioning_runner import read,dump,setup

OUT=Path('results/native_full_window_20261124_v1')
MODELS=['frozen_native_V','2025_copy','2025_point_weighted','2026_copy','2026_point_weighted','2027_copy','2027_point_weighted','2025_E']


def models(w):
    nets,keys,digests=smd.load(w)
    d=w/'results/native_e_vs_v_20261123_v1/train/point_weighted_e'
    rec=read(d/'SELECTION.json');pt=d/'selected.pt';assert sha(pt)==rec['selected_checkpoint_sha256']
    saved=torch.load(pt,map_location='cpu',weights_only=True)
    e=deepcopy(nets[0]);e.load_state_dict(saved['model'],strict=True)
    nets.append(e);keys.append('2025_E');digests.append(sha(pt))
    assert keys==MODELS
    original=nets[0].state_dict()
    for net in nets:
        net.eval().requires_grad_(False)
        assert all(torch.equal(original[k],v) for k,v in net.state_dict().items() if not k.startswith('decoder.'))
    return nets,keys,digests


def jobs(w):
    rows=[('PSM','PSM',87841)]
    for domain in ['MSL','SMAP']:
        rows.extend((domain,e,read(w/nasa.OUT/domain/(e+'.json'))['query_size']) for e in nasa.entity_list(w,domain))
    rows.extend(('SMD',r['id'],r['query_points']) for r in smd.records(w))
    assert len(rows)==109
    return sorted(rows,key=lambda r:r[2],reverse=True)


def windows(w,domain,entity):
    if domain=='PSM':
        values,_,sc=inputs(w)
        n=len(values)
        return values[:n//100*100].reshape(-1,100,values.shape[1]),dict(scaler_sha256=sc,input_sha256=sha(w/'data/common_psm_20261028_v1/test.npy'),original_points=n)
    if domain=='SMD':
        r=next(r for r in smd.records(w) if r['id']==entity)
        values,origin=smd.data(w,r,'QUERY');assert origin==0
        role=next(b for b in r['roles'] if b['role']=='QUERY')
        assert sha(w/r['scaler']['path'])==r['scaler']['sha256']
        n=len(values)
        return values[:n//100*100].reshape(-1,100,38),dict(scaler_sha256=r['scaler']['sha256'],input_sha256=role['sha256'],original_points=n,source_binding_sha256=smd.BINDING_SHA)
    data,_,qi,binding=nasa.read_roles(w,domain,entity)
    n=binding['query_size'];assert len(data[1])==n-99
    win=data[1][::100][:n//100,:,None]
    assert win.shape==(n//100,100,1)
    r=read(w/nasa.OUT/domain/(entity+'.json'))
    assert all(r[k]==v for k,v in binding.items())
    path=w/(nasa.MSL/'test'/(entity+'.npy') if domain=='MSL' else nasa.SMAP_QUERY/(entity+'.npz'))
    return win,dict(**binding,input_sha256=sha(path),prior_receipt_sha256=sha(w/nasa.OUT/domain/(entity+'.json')),original_points=n)


def full_scores(model,x,copies):
    v=model.cal_anomaly_score(x,copies,None,1)[0]
    e=(copies.mean(dim=0)-x).square().mean(dim=-1)[0]
    assert v.shape==e.shape==(100,)
    return v,e


@torch.inference_mode()
def reconstruct(nets,x,order):
    from torch_frft.frft_module import ifrft
    order.clear()
    with fixed_native_rng('cuda'):
        encoded,official=nets[0](x,norm=0,mask_mode='c',copies=10)
    projected=torch.stack([net.decoder(encoded) for net in nets[1:]])
    decoded=ifrft(projected,order[0],dim=-1).abs()
    copies=[official]+[decoded[k].reshape(10,1,x.shape[-1],100).permute(0,1,3,2) for k in range(len(nets)-1)]
    return copies


@torch.inference_mode()
def check(w):
    setup(12)
    nets,keys,digests=models(w)
    before=[state_hashes(n) for n in nets]
    from . import native_event_rank_training as src
    folder,task,_,_,_=src.source_roles(w)
    normal=src.nominal_swat_input(w,folder,task,'FIT')
    plan=src.load_plan(w/src_output())
    x=next(src.windows(normal,plan['swat_fit'][0,:1],'cuda'))
    order=[];hook=nets[0].net.register_forward_hook(lambda _m,_a,o:order.append(o.squeeze()))
    cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
    copies=reconstruct(nets,x,order)
    errors=[]
    for k,net in enumerate(nets):
        with fixed_native_rng('cuda'):
            _,direct=net(x,norm=0,mask_mode='c',copies=10)
        torch.testing.assert_close(copies[k],direct,rtol=0,atol=0)
        v,e=full_scores(net,x,copies[k])
        terminal=net.cal_anomaly_score(x,direct,None,1)[0,99]
        torch.testing.assert_close(v[99],terminal,rtol=0,atol=0)
        errors.append(float((v[99]-terminal).abs()))
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
    assert before==[state_hashes(n) for n in nets]
    hook.remove()
    root=w/OUT;root.mkdir(exist_ok=True,parents=True)
    dump(root/'CHECKS.json',dict(status='PASSED',keys=keys,weights_sha256=digests,official_source_sha256=SOURCE_SHA,official_weight_sha256=WEIGHT_SHA,mask_seed=84551,mask_bits=actual_mask_bits(x.shape[-1],'cuda').tolist(),native_B=1,copies=10,correction=1,eval=True,normal_window_end=int(plan['swat_fit'][0,0]),same_window_endpoint99_max_error=max(errors),grouped_inverse_bit_exact_all_eight=True,full_native_vectors=True,frozen_state_verified=True,RNG_restored=True,query_labels_read=False,source_labels_read_for_role_binding=True))


def src_output():return Path('results/native_event_rank_psm_20261122_v1')


@torch.inference_mode()
def score(w,lane,lanes):
    setup(12)
    root=w/OUT;check_rec=read(root/'CHECKS.json');assert check_rec['status']=='PASSED'
    nets,keys,digests=models(w);assert digests==check_rec['weights_sha256']
    initial=[state_hashes(n) for n in nets]
    order=[];nets[0].net.register_forward_hook(lambda _m,_a,o:order.append(o.squeeze()))
    work=jobs(w)
    for domain,entity,n in work[lane::lanes]+work:
        d=root/domain;d.mkdir(exist_ok=True,parents=True)
        bank=d/(entity+'.npz');receipt=bank.with_suffix('.json')
        with bank.with_suffix('.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if receipt.exists():
                r=read(receipt);assert sha(bank)==r['sha256'] and r['weights_sha256']==digests
                continue
            t=time.monotonic();window,binding=windows(w,domain,entity)
            assert binding['original_points']==n
            V=np.empty((len(window)*100,len(keys)),np.float32);E=np.empty_like(V)
            for j,ar in enumerate(window):
                x=torch.from_numpy(np.ascontiguousarray(ar)).cuda()[None]
                copies=reconstruct(nets,x,order)
                for k,(net,cp) in enumerate(zip(nets,copies)):
                    if j==0:
                        with fixed_native_rng('cuda'):_,direct=net(x,norm=0,mask_mode='c',copies=10)
                        torch.testing.assert_close(cp,direct,rtol=0,atol=0)
                    v,e=full_scores(net,x,cp)
                    V[j*100:(j+1)*100,k]=v.cpu().numpy();E[j*100:(j+1)*100,k]=e.cpu().numpy()
            assert np.isfinite(V).all() and np.isfinite(E).all()
            assert initial==[state_hashes(net) for net in nets]
            ix=np.arange(len(V));start=ix//100*100;available=start+99
            tmp=bank.with_suffix('.part.npz')
            np.savez_compressed(tmp,keys=keys,V=V,E=E,index=ix,window_start=start,window_end_exclusive=available+1,available_index=available)
            tmp.replace(bank)
            dump(receipt,dict(dataset=domain,entity=entity,keys=keys,weights_sha256=digests,sha256=sha(bank),query_points=n,score_points=len(ix),scored_windows=len(window),omitted_tail=n-len(ix),input_binding=binding,available='containing window endpoint',offline=True,window=100,stride=100,eval=True,native_B=1,copies=10,mask_seed=84551,correction=1,seconds=time.monotonic()-t,device=os.environ['CUDA_VISIBLE_DEVICES'],peak_reserved_bytes=torch.cuda.max_memory_reserved(),frozen_state_verified=True,labels_read=False))
            print(domain,entity,len(ix),round(time.monotonic()-t,3),flush=True)
    dump(root/f'LANE_{lane}.json',dict(frozen_state_verified=initial==[state_hashes(n) for n in nets],GPU=os.environ['CUDA_VISIBLE_DEVICES']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['check','score']);p.add_argument('--lane',type=int,default=0);p.add_argument('--lanes',type=int,default=4);a=p.parse_args()
    if a.action=='check':check(Path.cwd())
    else:score(Path.cwd(),a.lane,a.lanes)
