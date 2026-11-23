"""Fixed single-event target-supervised F control for three native initializations."""
import argparse
import os
import time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from .native_event_stage2_readout import target_task, ELIGIBLE_EVENTS
from .native_event_rank_readout import asset, LABEL_SHA, METRIC_SHA
from .native_event_rank import native_endpoint_and_copy_loss, event_pair_loss
from .native_v_conditioning import sha, state_hashes, fixed_native_rng
from .native_v_conditioning_runner import read, dump, role_values, role_index, setup, seed
from .native_v_replay import decoder_model, optimizer
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .native_v_conditioning_evaluation import exact_cal, causal_timing
from .timercd_swat_evaluation import evaluate_score
from .native_event_space_summary import group_operating
from .left_swat_replay_repeats import write_csv

OUT=Path('results/native_event_stage2_f_20261123_v1')
BASE=Path('results/native_event_rank_psm_20261122_v1')
ARMS=('official','copy','point_weighted')
SUPPORT={'PSM':((3248,3439),(92,604)), 'machine-3-2':((3080,3917),(161,673))}

def initial(w,arm,device):
    m=decoder_model(w,device)
    if arm!='official':
        r=read(w/BASE/'MODELS_FROZEN.json')['arms'][arm]
        ck=asset(w/BASE,r['checkpoint_relative'],r['checkpoint_sha256'])
        m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model'],strict=True)
    return m

def support(w,d):
    root=w/OUT/d
    with np.load(root/'SUPPORT.npz') as z:
        return {k:z[k] for k in z.files}

def prepare(w):
    out=w/OUT;out.mkdir(exist_ok=False,parents=True)
    for domain,(event,normal) in SUPPORT.items():
        folder,task=target_task(w,domain)
        x,_,origin,rec=role_values(w,folder,task,'U_FIT')
        assert origin==0 and event[1]<=len(x) and normal[1]<=len(x)
        if domain=='PSM':
            labels=np.load(asset(w,'data/common_psm_20261028_v1/test_label.npy',LABEL_SHA)).reshape(-1)[:len(x)]
        else:
            b=read(folder/'EVALUATION_BINDING.json')
            labels=np.loadtxt(asset(w,b['relative'],b['sha256']),delimiter=',',dtype=int).reshape(-1)[:len(x)]
        assert labels[event[0]:event[1]].all() and not labels[event[0]-1] and not labels[event[1]]
        assert not labels[normal[0]:normal[1]].any()
        runs=np.convolve(1-labels,np.ones(512,dtype=int),'valid')
        assert int(np.flatnonzero(runs==512)[0])==normal[0]
        neg=np.arange(normal[0]+99,normal[1]);pos=np.arange(event[0]+99,event[1])
        assert len(neg)==413 and len(pos)==(92 if domain=='PSM' else 738)
        strata=np.array_split(pos,3)
        rng=[np.random.default_rng(s) for s in np.random.SeedSequence(2025).spawn(3)]
        plan=dict(copy=rng[0].choice(neg,size=(240,32)),positive=np.column_stack([rng[1].choice(g,size=(240,5)) for g in strata]),negative=rng[2].choice(neg,size=(240,15)),weights=np.array([len(g) for g in strata])[None])
        d=out/domain;d.mkdir()
        np.savez_compressed(d/'PLAN.npz',**plan)
        # Learning receives only two fixed, non-concatenated support blocks.
        np.savez_compressed(d/'SUPPORT.npz',normal=x[slice(*normal)],positive=x[slice(*event)])
        dump(d/'SUPPORT.json',dict(domain=domain,event=event,normal=normal,positive_ends=len(pos),negative_ends=len(neg),source_record=rec,scaler_sha256=task['scaler_sha256'],u_fit=task['roles']['U_FIT'],plan_sha256=sha(d/'PLAN.npz'),support_sha256=sha(d/'SUPPORT.npz'),label_budget='one complete public abnormal event plus 512 normal points; construction uses public truth',learning_has_only_support_arrays=True))
    dump(out/'CONFIG.json',dict(seed=2025,steps=240,warmup=24,optimizer='fresh source AdamW configuration',copy_batch=32,positive_per_step=15,negative_pair_per_step=15,pair_margin=1,pair_coefficient=1,selection='fixed endpoint; step0 reference only',updates='decoder only',score='native endpoint V B1 ten fixed complementary masks norm0 correction1',source_pair_replay=False))

def one(m,arrays,end,positive,meta,device):
    key='positive' if positive else 'normal'
    bounds=meta['event'] if positive else meta['normal']
    t=int(end)-bounds[0]
    assert 99<=t<len(arrays[key])
    return native_endpoint_and_copy_loss(m,torch.from_numpy(arrays[key][t-99:t+1]).to(device)[None])

def train(w,domain,arm):
    tic=time.monotonic();device=setup();seed(2025)
    root=w/OUT/domain;meta=read(root/'SUPPORT.json');arrays=support(w,domain)
    assert sha(root/'SUPPORT.npz')==meta['support_sha256'] and sha(root/'PLAN.npz')==meta['plan_sha256']
    with np.load(root/'PLAN.npz') as z:plan={k:z[k] for k in z.files}
    # Shared official-initial scale. Calculating it identically avoids file races.
    official=initial(w,'official',device)
    with torch.inference_mode():
        values=np.array([float(one(official,arrays,e,False,meta,device)[0]) for e in range(meta['normal'][0]+99,meta['normal'][1])])
    scale=max(1e-8,float(np.median(values)));del official
    m=initial(w,arm,device);before=state_hashes(m);opt=optimizer(m)
    initial_grad=None;trace=[]
    for step in range(240):
        opt.zero_grad(set_to_none=True)
        lr=1e-4*min((step+1)/24,1)
        for g in opt.param_groups:g['lr']=lr
        losses=[]
        for e in plan['copy'][step]:
            _,loss=one(m,arrays,e,False,meta,device);(loss/32).backward();losses.append(float(loss.detach()))
        # Match the established pair loss's per-draw mean rather than score averaging.
        pos=torch.stack([one(m,arrays,e,True,meta,device)[0] for e in plan['positive'][step]]).reshape(3,5).T.reshape(5,3)
        neg=torch.stack([one(m,arrays,e,False,meta,device)[0] for e in plan['negative'][step]]).reshape(5,3)
        pair=event_pair_loss(pos,neg,scale,phase_sizes=np.repeat(plan['weights'],5,axis=0));pair.backward()
        norms=[p.grad.norm() for p in m.parameters() if p.requires_grad and p.grad is not None]
        grad=float(torch.linalg.vector_norm(torch.stack(norms)))
        assert np.isfinite(grad) and grad>0 and all(p.grad is None for p in m.parameters() if not p.requires_grad)
        opt.step();trace.append(dict(step=step+1,copy=np.mean(losses),pair=float(pair.detach()),gradient=grad))
        if step==0 or (step+1)%40==0:print(domain,arm,step+1,round(time.monotonic()-tic,1),flush=True)
    after=state_hashes(m)
    assert all(before[k]==v for k,v in after.items() if not k.startswith('decoder.'))
    assert all(not s.training for s in m.modules())
    d=root/arm;d.mkdir(exist_ok=False)
    torch.save(dict(model=m.state_dict(),update=240,arm=arm,domain=domain),d/'endpoint.pt')
    dump(d/'TRACE.json',trace)
    dump(d/'ENDPOINT.json',dict(domain=domain,arm=arm,checkpoint_sha256=sha(d/'endpoint.pt'),scale=scale,scale_count=len(values),plan_sha256=meta['plan_sha256'],support_sha256=meta['support_sha256'],seconds=time.monotonic()-tic,gpu=os.environ['CUDA_VISIBLE_DEVICES'],peak_reserved=torch.cuda.max_memory_reserved(),copy_exposures=240*32,positive_exposures=240*15,negative_pair_exposures=240*15,frozen_state_verified=True,step=240))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--domain');p.add_argument('--arm');a=p.parse_args()
    if a.action=='prepare':prepare(Path.cwd())
    else:train(Path.cwd(),a.domain,a.arm)
