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
from .left_multisource import target_inputs, NATIVE
from .left_multisource_eval import AXES, MIXED, read_score
from .trajectory_evaluation import labels, continuous_metrics, independent_check, load_author_metric, save_rows
from .left_decoder_context_metrics import AUTHOR


def bounds(d,stage):
    lo,hi=AXES[d]
    if stage in ("U","F"):
        lo=max(MIXED[d][0],ROLES[d]["QUERY"][0]+611)
    else:
        lo+=611
    return lo,hi


def score_name(stage,arm,initial="source"):
    return f"{stage}_{arm}"+("_official_initialization" if stage=="F" and initial=="official" else "")


def score(w,out,d,arm,stage,part=0,parts=1,initial="source"):
    dest=out/d/"scores"/score_name(stage,arm,initial)
    dest.mkdir(parents=True,exist_ok=True)
    if stage!="F" and (out/"paired_encoder_reuse.json").is_file():
        assert part==0 and parts==1
        _paired_score(w,out,d,stage)
    else:
        _score(w,out,d,arm,stage,part,parts,initial)


def _paired_score(w,out,d,stage):
    """Reuse only the identical frozen encoder; each decoder/history stays distinct.

    Calls the complete history forward with its original encoder's known output.
    Its hooks, conditioner, native transforms and decoder still run. Exact copy
    parity is checked against an independent, uncached history forward.
    """
    roots={arm:out/("source" if stage=="Z" else d)/arm for arm in ("native","history")}
    if not all((root/"receipt.json").is_file() for root in roots.values()): return
    dests={arm:out/d/"scores"/f"{stage}_{arm}" for arm in roots}
    if all((p/"part0.json").is_file() for p in dests.values()): return
    torch.set_num_threads(1)
    models={arm:model(w,arm) for arm in roots}
    digests={}
    for arm,m in models.items():
        ck=roots[arm]/"selected.pt"
        digests[arm]=sha(ck)
        assert digests[arm]==json.loads((roots[arm]/"receipt.json").read_text())["sha256"]
        m.load_state_dict(torch.load(ck,map_location="cpu",weights_only=True)["model"])
        m.eval()
        dests[arm].mkdir(parents=True,exist_ok=True)
    native,history=models["native"],models["history"]
    common=native.native.state_dict()
    assert all(torch.equal(v,history.native.state_dict()[k]) for k,v in common.items() if not k.startswith("decoder."))
    official=load_native(w,"cuda") if stage=="Z" else None
    _,_,sc,binding=target_inputs(w,d)
    values=torch.from_numpy(sc.transform(target(w,d,"TEST")).astype(np.float32)).cuda()
    assert torch.isfinite(values).all()
    origin=ROLES[d]["TEST"][0]
    lo,hi=bounds(d,stage)
    queue=out/d/"scores"/("paired_"+stage)
    queue.mkdir(parents=True,exist_ok=True)
    checked_native=False; checked_history=False
    for chunk,start in enumerate(range(lo,hi,1000)):
        paths={arm:p/f"part0_chunk{chunk}.npz" for arm,p in dests.items()}
        missing=[arm for arm,p in paths.items() if not p.with_suffix(".json").exists()]
        if not missing: continue
        claim=queue/f"{chunk}.claim"
        try: descriptor=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError: continue
        with os.fdopen(descriptor,"w") as f: f.write(str(os.getpid()))
        assert not any(p.with_suffix(".claim").exists() for p in paths.values())
        idx=np.arange(start,min(hi,start+1000))
        scores={arm:[] for arm in missing}; official_scores=[]
        tick=time.monotonic()
        with torch.no_grad():
            for t in idx:
                j=int(t)-origin
                current=values[j-99:j+1][None]
                past=values[j-611:j-99][None]
                meta=dict(history_start=int(t)-611,current_start=int(t)-99)
                rep,yn=native(past,current,**meta,bypass=True)
                if "native" in missing:
                    scores["native"].append(yn.var(0,correction=1).mean(-1)[0,-1])
                    if official is not None:
                        order=native.native.net(current[0].T.mean(0)).squeeze()
                        raw=ifrft(official.decoder(rep),order,dim=-1).abs()
                        yo=raw.reshape(10,1,current.shape[2],100).permute(0,1,3,2)
                        official_scores.append(yo.var(0,correction=1).mean(-1)[0,-1])
                        if not checked_native:
                            with fixed_native_rng(current.device):
                                _,independent=official(current,norm=0,mask_mode="c",copies=10)
                            torch.testing.assert_close(yo,independent,rtol=0,atol=0)
                            checked_native=True
                if "history" in missing:
                    previous=history.native.encoder.forward
                    history.native.encoder.forward=lambda *args,**kwargs:rep
                    try: _,yh=history(past,current,**meta)
                    finally: history.native.encoder.forward=previous
                    if not checked_history:
                        _,independent=history(past,current,**meta)
                        torch.testing.assert_close(yh,independent,rtol=0,atol=0)
                        dump(queue/f"parity_{os.getpid()}_{chunk}.json",dict(index=int(t),
                             full_copies_bitwise_equal=True,native_nondecoder_equal=True,checkpoints=digests))
                        checked_history=True
                    scores["history"].append(yh.var(0,correction=1).mean(-1)[0,-1])
        elapsed=time.monotonic()-tick
        for arm in missing:
            numeric=torch.stack(scores[arm]).cpu().numpy().astype(np.float64)
            assert np.isfinite(numeric).all()
            extra=(torch.stack(official_scores).cpu().numpy().astype(np.float64)
                   if arm=="native" and official is not None else np.array([]))
            path=paths[arm]
            np.savez_compressed(path,index=idx,score=numeric,official=extra,available=idx,
                                history_start=idx-611,current_start=idx-99)
            dump(path.with_suffix(".json"),dict(sha256=sha(path),points=len(idx),
                 seconds=elapsed/len(missing),seconds_are_shared_pair_apportionment=True,
                 joint_chunk_seconds=elapsed,physical_GPU=os.environ.get("CUDA_VISIBLE_DEVICES"),
                 frozen_encoder_reused=True))
        dump(queue/f"{chunk}.json",dict(seconds=elapsed,columns=missing,points=len(idx),
             physical_GPU=os.environ.get("CUDA_VISIBLE_DEVICES"),frozen_encoder_forwards=len(idx)))
        claim.unlink()
        if chunk%10==0: print("PAIRED_READ",d,stage,chunk,len(idx),elapsed,flush=True)
    count=(hi-lo+999)//1000
    if all((dest/f"part0_chunk{k}.json").is_file() for dest in dests.values() for k in range(count)):
        for arm,dest in dests.items():
            dump(dest/"part0.json",dict(checkpoint=digests[arm],domain=d,arm=arm,stage=stage,
                 part=0,parts=1,start=lo,stop=hi,scoring_B=1,normal_values_sha256=binding,
                 frozen_encoder_reuse=True,pair_ledger=str(queue.relative_to(out))))
        print("PAIRED_SCORED",d,stage,flush=True)


def _score(w,out,d,arm,stage,part=0,parts=1,initial="source"):
    dest=out/d/"scores"/score_name(stage,arm,initial)
    dest.mkdir(parents=True,exist_ok=True)
    finished=dest/f"part{part}.json"
    if finished.exists():
        return
    started=time.monotonic()
    torch.set_num_threads(1)
    m=model(w,arm)
    if stage=="F":
        ck=out/d/score_name(stage,arm,initial)/"selected.pt"
    else:
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


def bank(out,d,arm,stage,initial="source"):
    dest=out/d/"scores"/score_name(stage,arm,initial)
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
    results=[];axes=[]
    for d in ("PSM","SWAT"):
        y=labels(w,d)
        public=bounds(d,"Z");query=bounds(d,"U")
        axes.append(dict(domain=d,released_TEST_role=list(ROLES[d]["TEST"]),
            old_public_axis=list(AXES[d]),public_axis=list(public),
            QUERY_role=list(ROLES[d]["QUERY"]),legacy_mixed_axis=list(MIXED[d]),QUERY_axis=list(query),
            public_left_warmup_points=public[0]-ROLES[d]["TEST"][0],
            public_left_warmup_positive_points=int(y[ROLES[d]["TEST"][0]:public[0]].sum()),
            right_tail_excluded_for_existing_common_axis=ROLES[d]["TEST"][1]-public[1],
            QUERY_additional_left_crop_from_legacy=query[0]-MIXED[d][0]))
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
                            start=lo,stop=hi,points=len(idx),positive_points=int(y[idx].sum()),**metrics))
        # Native LEFT scores cover the same ticks but retain offline W192 permission.
        for panel in ("public","QUERY"):
            role="TEST" if panel=="public" else "QUERY"
            i,s,_=read_score(w,d,"source",role)
            lo,hi=bounds(d,"Z" if panel=="public" else "U")
            take=(i>=lo)&(i<hi)
            idx=i[take]
            np.testing.assert_array_equal(idx,np.arange(lo,hi))
            metrics,_=continuous_metrics(y[idx],s[take],load_author_metric(w/AUTHOR))
            np.testing.assert_allclose(independent_check(y[idx],s[take]),
                [metrics["AP"],metrics["ROC_AUC"]],atol=1e-11,rtol=0)
            results.append(dict(domain=d,method="multisource_LEFT",stage="Z",panel=panel,
                 start=lo,stop=hi,points=len(idx),positive_points=int(y[idx].sum()),**metrics))
            # Target-normal LEFT is a stronger existing background reference.
            # Its scaler is exactly the scaler already bound by target_inputs.
            root=w/NATIVE/d/"native"
            path=root/"raw_scores.npz"
            assert sha(path)==json.loads((root/"SCORE_RECEIPT.json").read_text())["bank_sha256"]
            with np.load(path) as z:
                external_index=z["absolute_time_index"]-(132481 if d=="PSM" else 0)
                hit=(external_index>=lo)&(external_index<hi)
                np.testing.assert_array_equal(external_index[hit],idx)
                values=z["canonical_score"][hit]
            metrics,_=continuous_metrics(y[idx],values,load_author_metric(w/AUTHOR))
            np.testing.assert_allclose(independent_check(y[idx],values),
                [metrics["AP"],metrics["ROC_AUC"]],atol=1e-11,rtol=0)
            results.append(dict(domain=d,method="target_normal_LEFT",stage="N",panel=panel,
                 start=lo,stop=hi,points=len(idx),positive_points=int(y[idx].sum()),**metrics))
    save_rows(out/"metrics.csv",results)
    dump(out/"axis_manifest.json",axes)


def evaluate_F(w,out,initial="source"):
    rows=[]
    for d in ("PSM","SWAT"):
        y=labels(w,d)
        for arm in ("native","history"):
            data=bank(out,d,arm,"F",initial)
            idx=data["index"]
            metrics,_=continuous_metrics(y[idx],data["score"],load_author_metric(w/AUTHOR))
            np.testing.assert_allclose(independent_check(y[idx],data["score"]),
                [metrics["AP"],metrics["ROC_AUC"]],atol=1e-11,rtol=0)
            lo,hi=bounds(d,"F")
            rows.append(dict(domain=d,method=arm,initial=initial,stage="F",panel="QUERY",
                             start=lo,stop=hi,points=len(idx),**metrics))
    save_rows(out/("F_metrics"+("_official_initialization" if initial=="official" else "")+".csv"),rows)


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
    p.add_argument("action",choices=("score","evaluate","worker","evaluate-F"))
    p.add_argument("workspace",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--domain",choices=("PSM","SWAT"))
    p.add_argument("--arm",choices=("native","history"))
    p.add_argument("--stage",choices=("Z","U","F"))
    p.add_argument("--initial",choices=("source","official"),default="source")
    p.add_argument("--part",type=int,default=0)
    p.add_argument("--parts",type=int,default=1)
    a=p.parse_args()
    if a.action=="evaluate": evaluate(a.workspace,a.output)
    elif a.action=="evaluate-F":evaluate_F(a.workspace,a.output,a.initial)
    elif a.action=="worker": worker(a.workspace,a.output)
    else: score(a.workspace,a.output,a.domain,a.arm,a.stage,a.part,a.parts,a.initial)
