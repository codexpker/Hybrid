"""Read the sensor-zero NASA folds with three frozen decoders; retain both endpoint evidences."""
import argparse
import ast
import csv
import fcntl
import os
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import torch

from . import native_event_space as nasa
from .left_swat_replay_repeats import write_csv
from .native_event_space_summary import RANKS, group_operating, read_csv
from .native_v_conditioning import fixed_native_rng, sha, WEIGHT_SHA
from .native_v_conditioning_runner import dump, read

ROOT = Path("results/native_e_vs_v_20261123_v1")
OUT = ROOT/"space"
METHODS = ("official_V","official_E","copy_V","Vtrain_V","Etrain_E","official_V_plus_E","Vtrain_E","Etrain_V")
OLD = dict(zip(nasa.ARMS,("official_V","copy_V","Vtrain_V")))


def models(w):
    initial, _, vtrain = nasa.models(w)
    root = w/ROOT/"train/point_weighted_e"
    meta = read(root/"SELECTION.json")
    ck = root/"selected.pt"
    assert meta["status"] == "COMPLETED" and sha(ck) == meta["selected_checkpoint_sha256"]
    e = torch.load(ck,map_location="cpu",weights_only=True)
    assert e["update"] == meta["selected_step"] and e["arm"] == "point_weighted_e"
    etrain = deepcopy(initial)
    etrain.load_state_dict(e["model"],strict=True)
    assert all(torch.equal(v,initial.state_dict()[k]) for k,v in etrain.state_dict().items()
               if not k.startswith("decoder.") or e["update"] == 0)
    identity = read(w/nasa.SOURCE_RUN/"MODELS_FROZEN.json")["arms"]
    return [initial,vtrain,etrain], [WEIGHT_SHA,identity["point_weighted"]["checkpoint_sha256"],sha(ck)],e["update"]


@torch.inference_mode()
def score(w,lane,lanes):
    from torch_frft.frft_module import ifrft
    nets,weights,estep=models(w)
    initial = nets[0]
    order = []
    initial.net.register_forward_hook(lambda _m,_a,o:order.append(o.squeeze()))
    jobs=[]
    for domain in ("MSL","SMAP"):
        for entity in nasa.entity_list(w,domain):
            old=w/nasa.OUT/domain/f"{entity}.npz"
            info=read(old.with_suffix(".json"))
            jobs.append((info["hash_identical_windows"],domain,entity,old,info))
    jobs.sort(reverse=True)
    for _,domain,entity,old,info in jobs[lane::lanes]+jobs:
        dest=w/OUT/domain
        dest.mkdir(parents=True,exist_ok=True)
        bank=dest/f"{entity}.npz"
        receipt=bank.with_suffix(".json")
        with bank.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if receipt.exists():
                rec=read(receipt)
                assert rec["sha256"]==sha(bank) and rec["weights"]==weights
                continue
            start=time.monotonic()
            windows,ci,qi,binding=nasa.read_roles(w,domain,entity)
            assert sha(old)==info["sha256"] and all(info[k]==v for k,v in binding.items())
            with np.load(old) as z:
                np.testing.assert_array_equal(z["cal_index"],ci)
                np.testing.assert_array_equal(z["query_index"],qi)
                prior=np.concatenate((z["cal"],z["query"]))
            assert list(nasa.ARMS)==info["arms"]
            flat=np.ascontiguousarray(np.concatenate(windows))
            _,first,inv=np.unique(flat.view("V400").ravel(),return_index=True,return_inverse=True)
            assert len(first)==info["hash_identical_windows"]
            unique=torch.from_numpy(flat[first]).cuda()[:,:,None]
            vals=torch.empty((len(first),3,2),device="cuda")
            for i in range(len(first)):
                x=unique[i:i+1]
                order.clear()
                with fixed_native_rng("cuda"):
                    encoded,base=initial(x,norm=0,mask_mode="c",copies=10)
                angle=order[0]
                for j,net in enumerate(nets):
                    if j==2 and estep==0:
                        vals[i,j]=vals[i,0]
                        continue
                    if j==0:
                        outputs=base
                    else:
                        outputs=ifrft(net.decoder(encoded),angle,dim=-1).abs().reshape(10,1,1,100).permute(0,1,3,2)
                        if i==0:
                            with fixed_native_rng("cuda"):
                                _,direct=net(x,norm=0,mask_mode="c",copies=10)
                            torch.testing.assert_close(outputs,direct,rtol=0,atol=0)
                    vals[i,j,0]=net.cal_anomaly_score(x,outputs,None,1)[0,99]
                    vals[i,j,1]=(outputs.mean(dim=0)[0,99]-x[0,99]).square().mean()
            evidence=vals.cpu().numpy()[inv.reshape(-1)]
            np.testing.assert_array_equal(evidence[:,0,0],prior[:,0])
            np.testing.assert_array_equal(evidence[:,1,0],prior[:,2])
            all_scores=np.column_stack((prior[:,0],evidence[:,0,1],prior[:,1],prior[:,2],evidence[:,2,1],
                            prior[:,0]+evidence[:,0,1],evidence[:,1,1],evidence[:,2,0]))
            assert np.isfinite(all_scores).all()
            tmp=bank.with_suffix(".part.npz")
            np.savez_compressed(tmp,keys=METHODS,cal_index=ci,query_index=qi,cal_window_start=ci-99,
                query_window_start=qi-99,cal_available_index=ci,query_available_index=qi,
                cal=all_scores[:len(ci)],query=all_scores[len(ci):])
            tmp.replace(bank)
            dump(receipt,dict(dataset=domain,entity=entity,**binding,hash_identical_windows=len(first),
                sha256=sha(bank),prior_sha256=info["sha256"],weights=weights,e_selected_step=estep,
                cal_points=len(ci),query_points=len(qi),V_identity="bit-exact official/V-trained on all cached points",
                sensor=0,device=os.environ.get("CUDA_VISIBLE_DEVICES"),seconds=time.monotonic()-start,
                label_read=False))
            print(domain,entity,len(first),round(time.monotonic()-start,1),flush=True)


def evaluate(w):
    root=w/OUT
    assert all((root/d/f"{e}.json").exists() for d in ("MSL","SMAP") for e in nasa.entity_list(w,d))
    ranking,oper,banks=[],[],[]
    new=[(i,m) for i,m in enumerate(METHODS) if m not in OLD.values()]
    for d in ("MSL","SMAP"):
        p=root/d
        ids=nasa.entity_list(w,d)
        with (w/nasa.MSL/"labels.csv").open() as f:
            records={r["chan_id"]:r for r in csv.DictReader(f) if r["spacecraft"]==d}
        fn=partial(nasa.entity_metrics,method_columns=new)
        rr,ee=[],[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for methods,events in pool.map(fn,[(w,p,d,e,records[e]) for e in ids]):
                rr.extend(methods)
                ee.extend(events)
        for name,rows in (("methods",rr),("events",ee)):
            rows.extend(dict(r,method=OLD[r["method"]]) for r in read_csv(w/nasa.OUT/d/f"{name}.csv"))
            write_csv(p/f"{name}.csv",rows)
        rr,ee=read_csv(p/"methods.csv"),read_csv(p/"events.csv")
        assert len(rr)==len(ids)*len(METHODS)*2
        for m in METHODS:
            strict=[r for r in rr if r["method"]==m and r["stream"]=="strict"]
            assert sorted(r["entity"] for r in strict)==ids
            ranking.append(dict(dataset=d,method=m,entities=len(ids),
                **{k:np.mean([float(r[k]) for r in strict]) for k in RANKS}))
            for s in ("strict","confirmed3"):
                op=group_operating([r for r in rr if r["method"]==m and r["stream"]==s],
                                   [r for r in ee if r["method"]==m and r["stream"]==s])
                oper.append(dict(dataset=d,method=m,stream=s,**op))
        banks.extend(read(p/f"{e}.json") for e in ids)
    for name,rows in (("ranking_macro",ranking),("operating",oper),("banks",banks)):
        write_csv(root/f"{name}.csv",rows)


def common_entity(args):
    workspace, entity, raw = args
    bank = workspace / OUT / "SMAP" / f"{entity}.npz"
    assert sha(bank) == read(bank.with_suffix(".json"))["sha256"]
    truth = np.zeros(int(raw["num_values"]), dtype=int)
    for start, stop in ast.literal_eval(raw["anomaly_sequences"]):
        truth[start:stop+1] = 1
    metric = nasa.load_author_metric(workspace / "preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py")
    rows = []
    with np.load(bank) as scores:
        query = scores["query_index"]
        keep = query >= 512
        np.testing.assert_array_equal(query[keep], np.arange(512, len(truth)))
        for col, method in enumerate(METHODS):
            if method in OLD.values():
                continue
            values, _ = nasa.continuous_metrics(truth[query[keep]], scores["query"][keep, col].astype(np.float64), metric)
            rows.append(dict(entity=entity, method=method, points=int(keep.sum()), **{k: values[k] for k in RANKS}))
    return rows


def common(workspace):
    ref = workspace / "preparations/native_e_vs_v_space_20261123_v1/SMAP_COMMON_MACRO.csv"
    assert sha(ref) == "539c649ad45946a059d9defbd682558a4bd1c038bd4f87c2dcfda85b036bbf39"
    old = read_csv(ref)
    by_method = {r["method"]: r for r in old}
    for r in old:
        r["method"] = OLD.get(r["method"], r["method"])
    ids = nasa.entity_list(workspace, "SMAP")
    with (workspace / nasa.MSL / "labels.csv").open() as f:
        records = {r["chan_id"]: r for r in csv.DictReader(f) if r["spacecraft"] == "SMAP"}
    assert len(ids) == 53
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows = [row for group in pool.map(common_entity, [(workspace, e, records[e]) for e in ids]) for row in group]
    parent_method = {"official_E": "frozen_native_V", "official_V_plus_E": "frozen_native_V",
                     "Vtrain_E": "point_weighted", "Etrain_V": "point_weighted", "Etrain_E": "point_weighted"}
    table = old.copy()
    for method, parent in parent_method.items():
        matched = [r for r in rows if r["method"] == method]
        assert len(matched) == 53 and sum(r["points"] for r in matched) == 400481
        table.append(dict(by_method[parent], method=method, **{k: np.mean([r[k] for r in matched]) for k in RANKS}))
    write_csv(workspace / OUT / "SMAP"/"common_methods.csv",rows)
    write_csv(workspace / OUT / "SMAP"/"common_macro.csv",table)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["score","evaluate","common"])
    parser.add_argument("--lane",type=int,default=0)
    parser.add_argument("--lanes",type=int,default=2)
    args=parser.parse_args()
    if args.action=="score":
        score(Path.cwd(),args.lane,args.lanes)
    elif args.action=="common":
        common(Path.cwd())
    else:
        evaluate(Path.cwd())
