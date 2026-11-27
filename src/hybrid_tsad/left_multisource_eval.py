"""Sealed LEFT readout banks, full test and isolated role evaluations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import torch

from .left_multisource import OUT, NATIVE, WIDTH, WINDOW, target_inputs, initialize, read, sha, dump
from .trajectory_experiment import target, ROLES
from .trajectory_evaluation import (labels, independent_check, continuous_metrics, load_author_metric,
    evaluate_score, PSM_AXES, SW_AXIS, PSM_CAL, SW_CAL, save_rows)
from .left_decoder_context_metrics import AUTHOR

AXES={"PSM":(0,87744),"SWAT":(495000,944856)}
MIXED={"PSM":PSM_AXES["QUERY"],"SWAT":SW_AXIS}


@torch.no_grad()
def bank(w,d,arm,role,*,root=OUT):
    dest=w/root/d/"banks"
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/(arm+"_"+role+".npz")
    if path.with_suffix(".json").exists():
        assert sha(path)==read(path.with_suffix(".json"))["sha256"]
        return
    start=time.monotonic()
    checkpoint=w/root/d/arm
    receipt=read(checkpoint/"receipt.json")
    assert sha(checkpoint/"selected.pt")==receipt["sha256"]
    model=initialize(w,d).eval()
    ck=torch.load(checkpoint/"selected.pt",map_location="cpu",weights_only=True)
    # Official N training profiles FLOPs with thop. Stored counters have no model role.
    params={k:v for k,v in ck["state"].items() if k.split(".")[-1] not in ("total_ops","total_params")}
    model.load_state_dict(params,strict=True)
    step=model.step=ck["step"]
    _,_,scaler,_=target_inputs(w,d)
    x=scaler.transform(target(w,d,role)).astype(np.float32)
    lo,hi=ROLES[d][role]
    anchor=AXES[d][0]
    first=anchor+((lo-anchor+WINDOW-1)//WINDOW)*WINDOW
    n=(hi-first)//WINDOW
    windows=x[first-lo:first-lo+n*WINDOW].reshape(n,WINDOW,WIDTH[d])
    scores=[]
    for index in range(0,n,128):
        batch=torch.from_numpy(np.ascontiguousarray(windows[index:index+128])).cuda()
        raw=model.infer(batch,None,None,None)[0].cpu().numpy()
        scores.append(raw.mean(axis=-1).reshape(-1))
    s=np.concatenate(scores)
    idx=np.arange(first,first+len(s))
    assert np.isfinite(s).all() and model.step==step
    np.savez_compressed(path,index=idx,available=np.repeat(idx[WINDOW-1::WINDOW],WINDOW),score=s)
    dump(path.with_suffix(".json"),dict(start=first,stop=int(idx[-1]+1),points=len(s),role=role,
        seconds=time.monotonic()-start,sha256=sha(path),checkpoint_sha256=receipt["sha256"],step=step))


def read_score(w,d,arm,role,*,root=OUT):
    path=w/root/d/"banks"/(arm+"_"+role+".npz")
    assert sha(path)==read(path.with_suffix(".json"))["sha256"]
    with np.load(path) as z:
        return z["index"],z["score"],z["available"]


def evaluate(data):
    w,d,arm,role=data
    if arm=="scratch_N":
        base=w/NATIVE/d/"native"
        b=base/"raw_scores.npz"
        assert sha(b)==read(base/"SCORE_RECEIPT.json")["bank_sha256"]
        with np.load(b) as z:
            shift=132481 if d=="PSM" else 0
            i,s,v=z["absolute_time_index"]-shift,z["canonical_score"],z["available_index"]-shift
    else:
        i,s,v=read_score(w,d,arm,role)
    y=labels(w,d)
    lo,hi=(AXES if role=="TEST" else MIXED)[d]
    hit=(i>=lo)&(i<hi)
    i,s,v=i[hit],s[hit],v[hit]
    np.testing.assert_array_equal(i,np.arange(lo,hi))
    rank,_=continuous_metrics(y[i],s,load_author_metric(w/AUTHOR))
    np.testing.assert_allclose(independent_check(y[i],s),[rank["AP"],rank["ROC_AUC"]],rtol=0,atol=1e-11)
    ops,events=[],[]
    if role=="QUERY":
        ci,cs,_=read_score(w,d,arm,"CAL")
        p=PSM_CAL if d=="PSM" else SW_CAL
        idx=np.searchsorted(ci,p)
        np.testing.assert_array_equal(ci[idx],p)
        tau=float(np.quantile(cs[idx],.99,method="higher"))
        ops,events,_,_=evaluate_score(i,s,y,tau,v,query_start=lo,query_stop=hi,
                  original_length=len(y),expected_events=None,ranking=rank)
        for row in [*ops,*events]:
            row.update(domain=d,arm=arm,cal_attack=int(y[p].sum()),cal_count=len(p))
    return dict(domain=d,arm=arm,role=role,start=lo,stop=hi,**rank),ops,events


def run(w,d,mode):
    public=mode=="public"
    role="TEST" if public else "QUERY"
    arms=("source","N","scratch_N") if public else ("source","ordinary","replay","random_u")
    for arm in arms:
        if arm=="scratch_N":
            continue
        for r in (("TEST",) if public else ("CAL","QUERY")):
            bank(w,d,arm,r)
    with ProcessPoolExecutor(max_workers=len(arms)) as pool:
        rows=list(pool.map(evaluate,[(w,d,a,role) for a in arms]))
    dest=w/OUT/d
    save_rows(dest/(mode+".csv"),[r[0] for r in rows])
    if not public:
        save_rows(dest/"operations.csv",[v for _,op,_ in rows for v in op])
        save_rows(dest/"events.csv",[v for _,_,ev in rows for v in ev])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("workspace",type=Path)
    parser.add_argument("domain",choices=WIDTH)
    parser.add_argument("mode",choices=("public","mixed"))
    args=parser.parse_args()
    run(args.workspace,args.domain,args.mode)
