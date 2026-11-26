"""Dimension-specific native LEFT source checkpoints and matched public transfer."""
import argparse
import hashlib
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .left_decoder_context_run import (OUT as NATIVE, WINDOW, Windows, environment, fingerprint,
                                       model_class, read, dump, sha)
from .intervention_reference import source_records
from .trajectory_experiment import target

OUT = Path("results/left_multisource_20261126_v1")
WIDTH = {"PSM":25, "SWAT":31}
SOURCE_STEPS = 4000
SOURCE_SELECT = (3200, 3600, 4000)
U_STEPS = 2000
BATCH = 32


def initialize(w, domain, ema_selector=None, weighted_reductions=None):
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    torch.set_num_threads(1)
    official, cfgs, _ = environment(w,domain)
    cfg = official.Configs(str(cfgs/domain/"model_configs_Left.json"))
    model = model_class(official.Left,"native",ema_selector=ema_selector,
                        weighted_reductions=weighted_reductions)(cfg).cuda()
    assert model.enc_in == WIDTH[domain] and model.seq_len == WINDOW
    return model


def target_inputs(w,d):
    """Use the byte-bound normal prefix and the already fitted author's scaler."""
    root=w/NATIVE/d/"native"
    cfg=read(root/"CONFIG.json")
    assert cfg["seed"]==2025 and sha(root/"SCALER.npz")==cfg["scaler_sha256"]
    raw=np.asfortranarray(np.concatenate([target(w,d,"FIT"),target(w,d,"SELECT")]),dtype=np.float64)
    digest=hashlib.sha256(raw.tobytes()).hexdigest()
    assert digest==cfg["normal_values_sha256"]
    scaler=StandardScaler()
    with np.load(root/"SCALER.npz") as z:
        scaler.mean_,scaler.var_,scaler.scale_=z["mean"],z["var"],z["scale"]
    scaler.n_features_in_=raw.shape[1]
    scaler.n_samples_seen_=len(raw)
    return scaler.transform(raw),cfg["FIT"][1],scaler,digest


def source_batch(records, width, batch, rng):
    x = np.empty((batch,WINDOW,width),dtype=np.float32)
    for j in range(batch):
        group = records[j%2]
        record = group[rng.integers(len(group))]
        start = rng.integers(len(record)-WINDOW+1)
        columns = rng.choice(record.shape[1],width,replace=False)
        x[j] = record[start:start+WINDOW][:,columns]
    return x


def update(model,opt,x):
    opt.zero_grad(set_to_none=True)
    step = model.step
    loss,_ = model(torch.from_numpy(x).cuda(),None,None,None)
    loss.backward()
    opt.step()
    assert model.step == step+1
    value = loss.detach().item()
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite LEFT loss")
    return value


def save(model,dest):
    torch.save(dict(state=model.state_dict(),step=model.step),dest/"selected.pt")


def restore(w,model,d):
    path=w/OUT/d/"source"/"selected.pt"
    receipt=read(path.parent/"receipt.json")
    assert sha(path)==receipt["sha256"]
    ck=torch.load(path,map_location="cpu",weights_only=True)
    model.load_state_dict(ck["state"],strict=True)
    model.step=ck["step"]
    assert model.step==receipt["selected"] and fingerprint(model.state_dict())==receipt["fingerprint"]
    return receipt


def source(w,d,dest):
    model=initialize(w,d)
    pools=source_records(w)
    rng=np.random.default_rng(2025)
    select_rng=np.random.default_rng(2026)
    panel=[torch.from_numpy(source_batch(pools["select"],WIDTH[d],BATCH,select_rng)).cuda() for _ in range(32)]
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    curve=[]
    best=float("inf")
    active=plateau=0
    loss_sum=0.
    for step in range(1,SOURCE_STEPS+1):
        model.train()
        loss_sum+=update(model,opt,source_batch(pools["fit"],WIDTH[d],BATCH,rng))
        lam=model._curriculum_lambda()
        active+=int(lam>0)
        plateau+=int(lam>=model.lambda_max)
        if step%400==0:
            print(d,step,loss_sum/400,lam,flush=True)
            row=dict(step=step,fit_loss=loss_sum/400,lam=lam)
            loss_sum=0.
            if step in SOURCE_SELECT:
                assert lam==model.lambda_max
                model.eval()
                with torch.no_grad():
                    risk=float(np.mean([model(x,None,None,None)[0].item() for x in panel]))
                assert np.isfinite(risk)
                row["select_loss"]=risk
                if risk<best:
                    best=risk
                    selected=step
                    selected_fp=fingerprint(model.state_dict())
                    save(model,dest)
            curve.append(row)
    return dict(updates=SOURCE_STEPS,selected=selected,fingerprint=selected_fp,curve=curve,
        active_updates=active,full_strength_updates=plateau,source_exposure=SOURCE_STEPS*BATCH,
        tep_exposure=SOURCE_STEPS*BATCH//2,smd_exposure=SOURCE_STEPS*BATCH//2,target_exposure=0,
        select_windows=32*BATCH,warmup=model.warmup_steps,ramp=model.ramp_steps,lambda_max=model.lambda_max,
        shapes={name:{n:list(p.shape) for n,p in getattr(model,name).named_parameters()} for name in ("Et","Ef","Dt","Df")})


def mixed(w,d,arm,dest):
    model=initialize(w,d)
    parent=None
    if arm!="random_u":
        parent=restore(w,model,d)["sha256"]
    start_step=model.step
    _,_,scaler,_=target_inputs(w,d)
    data=scaler.transform(target(w,d,"U")).astype(np.float32)
    source_fit=source_records(w)["fit"] if arm=="replay" else None
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    r1,r2,rs=[np.random.default_rng(s) for s in np.random.SeedSequence(2025).spawn(3)]
    total=0.
    trace=[]
    ix=np.arange(WINDOW)
    model.train()
    for step in range(1,U_STEPS+1):
        first=data[r1.integers(len(data)-WINDOW+1,size=BATCH//2)[:,None]+ix]
        if arm=="replay":
            other=source_batch(source_fit,WIDTH[d],BATCH//2,rs)
        else:
            other=data[r2.integers(len(data)-WINDOW+1,size=BATCH//2)[:,None]+ix]
        total+=update(model,opt,np.concatenate([first,other]))
        if step%400==0:
            trace.append(dict(step=step,loss=total/400,lam=model._curriculum_lambda()))
            total=0.
    save(model,dest)
    return dict(updates=U_STEPS,selected=U_STEPS,initial_step=start_step,model_step=model.step,parent=parent,
        source_exposure=U_STEPS*BATCH//2 if arm=="replay" else 0,
        target_exposure=U_STEPS*BATCH//2 if arm=="replay" else U_STEPS*BATCH,curve=trace)


def target_normal(w,d,dest):
    """Use the archived scratch trainer, normal data order, epoch budget and early stopping."""
    official,cfgs,_=environment(w,d)
    model=initialize(w,d)
    parent=restore(w,model,d)
    values,cut,scaler,raw_hash=target_inputs(w,d)
    baseline=w/NATIVE/d/"native"
    archived=read(baseline/"CONFIG.json")
    assert archived["normal_values_sha256"]==raw_hash
    assert archived["FIT"]==[0,cut] and archived["SELECT"]==[cut,len(values)]
    with np.load(baseline/"SCALER.npz") as z:
        np.testing.assert_array_equal(z["mean"],scaler.mean_)
        np.testing.assert_array_equal(z["scale"],scaler.scale_)
    trace=[]
    selected=[]
    orig=official.EarlyStopping
    class Stop(orig):
        def save_checkpoint(self,loss,m,path):
            super().save_checkpoint(loss,m,path)
            selected.append((len(trace),m.step,float(loss)))
    class Experiment(official.Exp_Anomaly_Detection):
        def _acquire_device(self):
            return torch.device("cuda:0")
        def _build_model(self):
            return model
        def _get_data(self,flag,step=None):
            assert flag in ("train","val")
            training=flag=="train"
            ds=Windows(values[:cut] if training else values[cut:],1 if training else WINDOW)
            return ds,DataLoader(ds,batch_size=128,shuffle=training,drop_last=False,num_workers=0)
        def vali(self,loader):
            loss=super().vali(loader)
            assert np.isfinite(loss)
            trace.append(dict(epoch=len(trace)+1,loss=float(loss),step=model.step))
            return loss
    official.EarlyStopping=Stop
    args=SimpleNamespace(data=d,data_origin="DADA",model_class="Left",gpu=0,use_gpu=True,
        gpu_type="cuda",use_multi_gpu=False,configs_path=str(cfgs),model_save_path=str(dest/"model"),
        results_save_path=str(dest/"unused"))
    try:
        exp=Experiment(args,id=0)
        exp.train()
    finally:
        official.EarlyStopping=orig
    completed=model.step
    epoch,model.step,loss=selected[-1]
    save(model,dest)
    count=cut-WINDOW+1
    return dict(parent=parent["sha256"],selected=epoch,selected_step=model.step,
        updates=len(trace)*int(np.ceil(count/128)),final_step=completed,curve=trace,
        source_exposure=0,target_exposure=len(trace)*count,selected_loss=loss)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("workspace",type=Path)
    p.add_argument("domain",choices=WIDTH)
    p.add_argument("arm",choices=("source","N","ordinary","random_u","replay"))
    args=p.parse_args()
    w,d,arm=args.workspace,args.domain,args.arm
    dest=w/OUT/d/arm
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/"receipt.json").exists():
        raise FileExistsError(dest)
    start=time.monotonic()
    if arm=="source":
        result=source(w,d,dest)
    elif arm=="N":
        result=target_normal(w,d,dest)
    else:
        result=mixed(w,d,arm,dest)
    dump(dest/"receipt.json",dict(domain=d,arm=arm,seed=2025,width=WIDTH[d],window=WINDOW,
         seconds=time.monotonic()-start,sha256=sha(dest/"selected.pt"),labels_read=False,
         device=torch.cuda.get_device_name(),peak_bytes=torch.cuda.max_memory_allocated(),**result))


if __name__=="__main__":
    main()
