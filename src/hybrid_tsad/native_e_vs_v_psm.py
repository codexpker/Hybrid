"""Recover endpoint E from archived copies; score only the new decoder on PSM."""
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from .native_rank_psm_extension import AXIS, EXTERNAL, READOUT_SEAL, RUN
from .native_rank_psm_repeats import SUFFIX
from .native_event_rank_readout import BASE, TASK_SHA, TARGET_SHA, LABEL_SHA, METRIC_SHA, asset
from .native_v_conditioning_runner import read, dump, setup, role_index, sha
from .native_v_conditioning import load_native, fixed_native_rng
from .left_native_swat_metrics import load_author_metric, continuous_metrics
from .left_swat_replay_repeats import write_csv
from .native_v_conditioning_evaluation import exact_cal, causal_timing
from .native_event_space_summary import group_operating
from .timercd_swat_evaluation import evaluate_score

OUT = Path("results/native_e_vs_v_psm_20261123_v1")
CHECK = Path("results/native_e_vs_v_20261123_v1/train/point_weighted_e")
EXT = Path("results/native_rank_psm_extension_20261122_v1")
KEYS = ["official_V","official_E","copy_V","Vtrain_V","Etrain_E","official_V_plus_E","Vtrain_E","Etrain_V"]
HASHES = {"frozen_native_V":"83021c4f2175a1d10934073bb0b1b42874f0b807061f7d3b101c02d428a44c9a",
          "copy":"316830d354dc01392bbde204bbaa7fd7f5cf9515dbc82f80ddc07b3ce04b8b26",
          "point_weighted":"cf092a69cc11cee01269f16758f6204a7a023231ceb618e2ef420bf1dd49f955"}


def inputs(w):
    d = w/BASE/"PSM"
    task = read(asset(d,"TASK.json",TASK_SHA))
    raw = np.load(asset(w,task["data"]["target_test"]["relative"],TARGET_SHA))
    with np.load(asset(d,"SCALER.npz",task["scaler_sha256"])) as sc:
        x = ((raw.astype(np.float64)-sc["mean"])/sc["std"]).astype("float32")
    return x,role_index(w,d,task,"CAL"),task["scaler_sha256"]


def cached(w):
    d=w/OUT
    d.mkdir(exist_ok=True,parents=True)
    x,cal,sc=inputs(w)
    oldseal=read(asset(w/RUN,"READOUT_FROZEN.json",READOUT_SEAL))
    columns={}
    digests={}
    for arm,vkey,ekey in [("frozen_native_V","official_V","official_E"),("copy","copy_V",None),("point_weighted","Vtrain_V","Vtrain_E")]:
        with np.load(asset(w/EXT,arm+".npz",HASHES[arm])) as z:
            np.testing.assert_array_equal(z["index"],AXIS)
            columns[vkey]=z["score"]
            if ekey is None:
                continue
            e=np.full(len(AXIS),np.nan)
            def recover(idx,copies):
                assert copies.shape==(len(idx),10,25) and np.isnan(e[idx-99]).all()
                np.testing.assert_allclose(copies.astype("float64").var(1,ddof=1).mean(1),columns[vkey][idx-99],rtol=1e-5,atol=1e-7)
                e[idx-99]=np.square(copies.mean(axis=1,dtype=np.float32)-x[idx]).mean(axis=1,dtype=np.float32)
            recover(z["new_index"],z["new_terminal_copies"])
        for role,idx in [("QUERY",SUFFIX),("CAL",cal)]:
            if arm=="point_weighted":
                ref=oldseal["banks"][arm][role]
                rec=read(asset(w,ref["relative"],ref["sha256"]))
            else:
                rec=read(w/BASE/"cache/PSM"/role/"SEALED.json")
            with np.load(asset(w,rec["relative"],rec["sha256"])) as b:
                np.testing.assert_array_equal(b["absolute_time_index"],idx)
                np.testing.assert_array_equal(b["window_start"],idx-99)
                np.testing.assert_array_equal(x[idx],b["current_normalized"])
                recover(idx,b["reconstruction_terminal"])
            digests[f"{arm}_{role}"]=rec["sha256"]
        assert np.isfinite(e).all()
        columns[ekey]=e
    columns["official_V_plus_E"]=columns["official_V"]+columns["official_E"]
    p=d/"historical.npz"
    np.savez_compressed(p,**columns,index=AXIS,window_start=AXIS-99,available_index=AXIS,cal_index=cal)
    dump(d/"CACHE.json",dict(sha256=sha(p),input_sha256=TARGET_SHA,scaler_sha256=sc,
         reconstruction_sources=digests,full_V_sources=HASHES,labels_read=False))


@torch.inference_mode()
def score(w):
    d=w/OUT
    x,cal,sc=inputs(w)
    rec=read(w/CHECK/"SELECTION.json")
    pt=asset(w/CHECK,"selected.pt",rec["selected_checkpoint_sha256"])
    base=load_native(w,"cuda")
    state=torch.load(pt,map_location="cpu",weights_only=True)
    assert state["update"]==rec["selected_step"]
    cached_state=all(torch.equal(v.cpu(),state["model"][k]) for k,v in base.state_dict().items())
    hist=read(d/"CACHE.json")
    tic=time.monotonic()
    data=asset(d,"historical.npz",hist["sha256"])
    with np.load(data) as z:
        scores={k:z[k] for k in KEYS if k in z}
    if cached_state:
        vals=np.stack([scores["official_E"],scores["official_V"]],axis=1)
    else:
        base.load_state_dict(state["model"],strict=True)
        all_x=torch.from_numpy(x).cuda()
        parts=[]
        for lo in range(0,len(AXIS),2048):
            ix=AXIS[lo:lo+2048]; p=d/f"e_{lo:05}.npz"; meta=p.with_suffix(".json")
            if meta.exists():
                r=read(meta)
                assert r["weight"]==sha(pt)
                with np.load(asset(d,p.name,r["sha256"])) as z:
                    np.testing.assert_array_equal(z["index"],ix)
                    parts.append(z["scores"])
                continue
            out=torch.empty((len(ix),2),device="cuda")
            for j,t in enumerate(ix):
                window=all_x[t-99:t+1].unsqueeze(0)
                with fixed_native_rng("cuda"):
                    _,copies=base(window,norm=0,mask_mode="c",copies=10)
                out[j,0]=(copies.mean(dim=0)[0,99]-window[0,99]).square().mean()
                out[j,1]=base.cal_anomaly_score(window,copies,None,1)[0,99]
            ar=out.cpu().numpy()
            assert np.isfinite(ar).all()
            np.savez_compressed(p,scores=ar,index=ix,available_index=ix,window_start=ix-99)
            dump(meta,dict(weight=sha(pt),sha256=sha(p)))
            parts.append(ar)
            print(ix[-1],round(time.monotonic()-tic,1),flush=True)
        vals=np.vstack(parts)
    scores.update(Etrain_E=vals[:,0],Etrain_V=vals[:,1])
    p=d/"banks.npz"
    np.savez_compressed(p,keys=KEYS,query=np.stack([scores[k] for k in KEYS],axis=1),
                        index=AXIS,available_index=AXIS,window_start=AXIS-99,cal_index=cal)
    dump(d/"SCORES.json",dict(sha256=sha(p),scaler_sha256=sc,weight=sha(pt),step=state["update"],
        reused_official=cached_state,seconds=time.monotonic()-tic,gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),labels_read=False))


def evaluate(w):
    d=w/OUT
    y=np.load(asset(w,"data/common_psm_20261028_v1/test_label.npy",LABEL_SHA)).reshape(-1)
    metric=load_author_metric(asset(w,"preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py",METRIC_SHA))
    ranks,external,ops,events=[],[],[],[]
    with np.load(asset(d,"banks.npz",read(d/"SCORES.json")["sha256"])) as z:
        ci=z["cal_index"]
        assert y[ci].sum()==len(ci)==512
        np.testing.assert_array_equal(z["index"],AXIS)
        scores=z["query"]
    with (w/EXT/"PSM_RANKING.csv").open() as f:
        old_rows=list(csv.DictReader(f))
    archived={(r["method"],int(r["axis_start"])):r for r in old_rows}
    original={"official_V":"frozen_native_V","copy_V":"copy","Vtrain_V":"point_weighted"}
    for j,name in enumerate(KEYS):
        s=scores[:,j]
        for start,rows in ((99,ranks),(512,external)):
            keep=AXIS>=start
            if name in original:
                r=archived[original[name],start]
                assert int(r["points"])==keep.sum() and r["score_sha256"]==HASHES[original[name]]
                m={k:float(r[k]) for k in ("AP","ROC_AUC","AUC_PR_trapezoidal","VUS_PR","VUS_ROC")}
            else:
                m,_=continuous_metrics(y[AXIS[keep]],s[keep],metric)
            rows.append(dict(dataset="PSM",method=name,points=int(keep.sum()),axis_start=start,**m))
        tau=exact_cal(ci,s[ci-99],ci)
        rr,ee,_,_=evaluate_score(SUFFIX,s[SUFFIX-99],y,tau,SUFFIX,query_start=23016,query_stop=87744,
                                 original_length=len(y),expected_events=(48,5),ranking=None)
        causal_timing(rr,ee)
        for row in rr:
            details=[{k:"" if v is None else str(v) for k,v in e.items()} for e in ee if e["stream"]==row["stream"]]
            row.update(group_operating([row],details))
            ops.append(dict(dataset="PSM",method=name,**row))
        events.extend(dict(dataset="PSM",method=name,**e) for e in ee)
    for name in ("LEFT W192 endpoint","HistoryMean512"):
        r=archived[name,512]
        external.append({k:r[k] for k in external[0]})
    write_csv(d/"ranking_macro.csv",ranks)
    write_csv(d/"external_axis.csv",external)
    write_csv(d/"operating.csv",ops)
    write_csv(d/"events.csv",events)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["cached","score","evaluate"]);args=p.parse_args()
    if args.action=="score": setup(12)
    {"cached":cached,"score":score,"evaluate":evaluate}[args.action](Path.cwd())
