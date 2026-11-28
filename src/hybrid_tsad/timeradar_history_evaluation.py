"""Exact B1 endpoint readout and fixed metrics for the history model."""
import argparse
from pathlib import Path
import time
import json
import os

import numpy as np
import torch
from torch_frft.frft_module import ifrft

from .timeradar_history_experiment import model, forward
from .native_v_conditioning import load_native, sha, fixed_native_rng
from .trajectory_experiment import ROLES, target, dump
from .left_multisource import target_inputs
from .left_multisource_eval import AXES, MIXED, read_score
from .trajectory_evaluation import labels, continuous_metrics, independent_check, load_author_metric, save_rows
from .left_decoder_context_metrics import AUTHOR


def bounds(d,stage):
    lo,hi=AXES[d]
    if stage=="U":
        lo=max(MIXED[d][0],ROLES[d]["QUERY"][0]+611)
    else:
        lo+=611
    return lo,hi


def score(w,out,d,arm,stage,part=0,parts=1):
    dest=out/d/"scores"/f"{stage}_{arm}"
    dest.mkdir(parents=True,exist_ok=True)
    _score(w,out,d,arm,stage,part,parts)


def _score(w,out,d,arm,stage,part=0,parts=1):
    dest=out/d/"scores"/f"{stage}_{arm}"
    dest.mkdir(parents=True,exist_ok=True)
    finished=dest/f"part{part}.json"
    if finished.exists():
        return
    started=time.monotonic()
    torch.set_num_threads(1)
    m=model(w,arm)
    ck=out/("source" if stage=="Z" else d)/arm/"selected.pt"
    receipt=json.loads(ck.with_name("receipt.json").read_text())
    assert sha(ck)==receipt["sha256"]
    state=torch.load(ck,map_location="cpu",weights_only=True)
    m.load_state_dict(state["model"])
    m.eval()
    original=load_native(w,"cuda") if arm=="native" and stage=="Z" else None
    _,_,sc,scaler_binding=target_inputs(w,d)
    values=torch.from_numpy(sc.transform(target(w,d,"TEST")).astype(np.float32)).cuda()
    origin=ROLES[d]["TEST"][0]
    lo,hi=bounds(d,stage)
    lo=lo+(hi-lo)*part//parts
    hi0=bounds(d,stage)[1]
    hi=bounds(d,stage)[0]+(hi0-bounds(d,stage)[0])*(part+1)//parts
    for chunk,start in enumerate(range(lo,hi,1000)):
        path=dest/f"part{part}_chunk{chunk}.npz"
        if path.with_suffix(".json").exists():
            assert sha(path)==json.loads(path.with_suffix(".json").read_text())["sha256"]
            continue
        claim=path.with_suffix(".claim")
        try:
            descriptor=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(descriptor,"w") as f: f.write(str(os.getpid()))
        idx=np.arange(start,min(hi,start+1000))
        scores=[];official=[]
        tick=time.monotonic()
        with torch.no_grad():
            for t in idx:
                j=int(t)-origin
                current=values[j-99:j+1][None]
                history=values[j-611:j-99][None]
                rep,copies=m(history,current,history_start=int(t)-611,current_start=int(t)-99,
                              bypass=arm!="history")
                scores.append(copies.var(0,correction=1).mean(-1)[0,-1].item())
                if original is not None:
                    # Frozen encoder and order are identical; only decoder differs.
                    order=m.native.net(current[0].T.mean(0)).squeeze()
                    raw=ifrft(original.decoder(rep),order,dim=-1).abs()
                    y=raw.reshape(10,1,current.shape[2],100).permute(0,1,3,2)
                    if chunk==0 and t==idx[0]:
                        with fixed_native_rng(current.device):
                            _, independent=original(current,norm=0,mask_mode="c",copies=10)
                        torch.testing.assert_close(y,independent,rtol=0,atol=0)
                    official.append(y.var(0,correction=1).mean(-1)[0,-1].item())
        assert np.isfinite(scores).all()
        np.savez_compressed(path,index=idx,score=np.asarray(scores),official=np.asarray(official),
                            available=idx,history_start=idx-611,current_start=idx-99)
        dump(path.with_suffix(".json"),dict(sha256=sha(path),seconds=time.monotonic()-tick,points=len(idx)))
        claim.unlink()
        if chunk%10==0: print("READ",d,arm,stage,part,len(idx),time.monotonic()-tick,flush=True)
    expected_chunks=(hi-lo+999)//1000
    if not all((dest/f"part{part}_chunk{k}.json").is_file() for k in range(expected_chunks)):
        return
    dump(finished,dict(checkpoint=sha(ck),domain=d,arm=arm,stage=stage,part=part,parts=parts,
                       start=lo,stop=hi,seconds=time.monotonic()-started,scoring_B=1,
                       normal_values_sha256=scaler_binding,
                       native_official_shared_encoder=original is not None))
    print("SCORED",d,arm,stage,part,flush=True)


def bank(out,d,arm,stage):
    dest=out/d/"scores"/f"{stage}_{arm}"
    receipts=[json.loads(p.read_text()) for p in dest.glob("part*.json") if "_chunk" not in p.name]
    assert receipts and len(receipts)==receipts[0]["parts"]
    rows=[]
    for p in dest.glob("*.npz"):
        assert sha(p)==json.loads(p.with_suffix(".json").read_text())["sha256"]
        with np.load(p) as z: rows.append({k:z[k].copy() for k in z.files})
    order=np.argsort(np.concatenate([z["index"] for z in rows]))
    data={k:np.concatenate([z[k] for z in rows]) for k in ("index","score","available","history_start","current_start")}
    data={k:v[order] for k,v in data.items()}
    expected=np.arange(*bounds(d,stage))
    np.testing.assert_array_equal(data["index"],expected)
    official=np.concatenate([z["official"] for z in rows])
    if len(official): data["official"]=official[order]
    return data


def evaluate(w,out):
    results=[]
    for d in ("PSM","SWAT"):
        y=labels(w,d)
        for arm in ("native","history"):
            for stage in ("Z","U"):
                data=bank(out,d,arm,stage)
                panels=("public","QUERY") if stage=="Z" else ("QUERY",)
                for panel in panels:
                    lo,hi=bounds(d,"U" if panel=="QUERY" else "Z")
                    mask=(data["index"]>=lo)&(data["index"]<hi)
                    idx=data["index"][mask]
                    np.testing.assert_array_equal(idx,np.arange(lo,hi))
                    methods={"score":arm}
                    if "official" in data: methods["official"]="official"
                    for col,name in methods.items():
                        metrics,_=continuous_metrics(y[idx],data[col][mask],load_author_metric(w/AUTHOR))
                        ap,roc=independent_check(y[idx],data[col][mask])
                        np.testing.assert_allclose([ap,roc],[metrics["AP"],metrics["ROC_AUC"]],atol=1e-11,rtol=0)
                        results.append(dict(domain=d,method=name,stage=stage,panel=panel,
                            start=lo,stop=hi,points=len(idx),**metrics))
        # Native LEFT scores cover the same ticks but retain offline W192 permission.
        for panel in ("public","QUERY"):
            role="TEST" if panel=="public" else "QUERY"
            i,s,_=read_score(w,d,"source",role)
            lo,hi=bounds(d,"Z" if panel=="public" else "U")
            take=(i>=lo)&(i<hi)
            idx=i[take]
            np.testing.assert_array_equal(idx,np.arange(lo,hi))
            metrics,_=continuous_metrics(y[idx],s[take],load_author_metric(w/AUTHOR))
            results.append(dict(domain=d,method="multisource_LEFT",stage="Z",panel=panel,
                 start=lo,stop=hi,points=len(idx),**metrics))
    save_rows(out/"metrics.csv",results)


def worker(w,out):
    while True:
        pending=False
        for stage in ("Z","U"):
            for d in ("SWAT","PSM"):
                for arm in ("history","native"):
                    dest=out/d/"scores"/f"{stage}_{arm}"
                    if (dest/"part0.json").exists(): continue
                    pending=True
                    checkpoint=out/("source" if stage=="Z" else d)/arm
                    if (checkpoint/"receipt.json").is_file():
                        score(w,out,d,arm,stage)
        if not pending:return
        time.sleep(5)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("score","evaluate","worker"))
    p.add_argument("workspace",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--domain",choices=("PSM","SWAT"))
    p.add_argument("--arm",choices=("native","history"))
    p.add_argument("--stage",choices=("Z","U"))
    p.add_argument("--part",type=int,default=0)
    p.add_argument("--parts",type=int,default=1)
    a=p.parse_args()
    if a.action=="evaluate": evaluate(a.workspace,a.output)
    elif a.action=="worker": worker(a.workspace,a.output)
    else: score(a.workspace,a.output,a.domain,a.arm,a.stage,a.part,a.parts)
