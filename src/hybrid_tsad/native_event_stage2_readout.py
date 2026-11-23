"""Frozen native endpoint scoring for the fixed source-supervised Stage2-U comparison."""
import argparse
import fcntl
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .native_event_rank_readout import asset, LABEL_SHA, METRIC_SHA
from .native_v_conditioning import fixed_native_rng, load_native, sha
from .native_v_conditioning_runner import dump, read, role_index, role_values, setup
from .native_v_conditioning_evaluation import causal_timing, exact_cal
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_swat_replay_repeats import write_csv
from .native_event_space_summary import group_operating
from .timercd_swat_evaluation import evaluate_score


OUT = Path("results/native_event_stage2_u_20261123_v1")
PRIOR = Path("results/native_event_rank_psm_20261122_v1")
INIT_SHA = "ec572d31d5612ff0d7ebdbe58b133ec631a2430700e3bd449907307de798ddf1"
DOMAINS = ("PSM", "machine-3-2", "machine-3-3")
TASK_SHA = dict(zip(DOMAINS, (
    "fba30d9560eb35769d24d8649d592fe2e845813ae772395e845881a18cfaddc7",
    "b1781413e5a94ff9d7a1884118c3a1f8bed090f2c7777aee5e8b7a3ba78ce9c2",
    "85f41928b8c2d8bfd6eb979e52ea521015ea4ef73f05b87de6cc56863782e463")))
ARMS = ("noU", "replay_copy", "replay_pair", "source_pair")
PART = 2048


def target_task(w, entity):
    base = ("timeradar_native_v_replay_20261120_v1" if entity == "PSM" else
            "timeradar_target_mean_smd_20261120_v1")
    folder = w / "results" / base / entity
    return folder, read(asset(folder, "TASK.json", TASK_SHA[entity]))


def load_models(w, entity):
    base = load_native(w, "cuda")
    base.eval().requires_grad_(False)
    original = base.state_dict()
    weights = {}
    models = {}
    for arm in ARMS:
        if arm == "noU":
            path = asset(w / PRIOR, "train/point_weighted/selected.pt", INIT_SHA)
        else:
            folder = w / OUT / "train" / ("SHARED" if arm == "source_pair" else entity) / arm
            ref = read(folder / "ENDPOINT.json")
            path = asset(folder, "endpoint.pt", ref["checkpoint_sha256"])
        weights[arm] = sha(path)
        if entity == "PSM" and arm == "noU":
            continue
        data = torch.load(path, map_location="cuda", weights_only=True)
        if arm != "noU":
            assert data["update"] == 2000 and data["arm"] == arm
        state = data["model"]
        assert set(state) == set(original)
        assert all(torch.equal(original[k], state[k]) for k in original if not k.startswith("decoder."))
        m = deepcopy(base)
        m.load_state_dict(state, strict=True)
        models[arm] = m.eval().requires_grad_(False)
    return weights, models


def reused_psm(w, role, index, task):
    seal = read(w / PRIOR / "READOUT_FROZEN.json")["banks"]["point_weighted"][role]
    info = read(asset(w, seal["relative"], seal["sha256"]))
    assert info["scaler_sha256"] == task["scaler_sha256"] and info["checkpoint_sha256"] == INIT_SHA
    with np.load(asset(w, info["relative"], info["sha256"])) as z:
        for key, expected in (("absolute_time_index",index),("available_index",index),
                              ("window_start",index-99),("window_end_exclusive",index+1)):
            np.testing.assert_array_equal(z[key],expected)
        v = z["V"].reshape(-1)
        np.testing.assert_array_equal(z["canonical_score"], v.astype(float))
    return v, info["sha256"]


@torch.inference_mode()
def score(w, lane, lanes):
    from torch_frft.frft_module import ifrft
    setup(12)
    root = w / OUT
    global_seal = root / "ALL_MODELS_FROZEN.json"
    assert global_seal.exists()
    seal_sha = sha(global_seal)
    for entity in DOMAINS:
        folder, task = target_task(w, entity)
        weights, models = load_models(w, entity)
        names = list(models)
        base = models[names[0]]
        order = []
        hook = base.net.register_forward_hook(lambda _m,_a,value: order.append(value.squeeze()))
        for role in ["CAL", "QUERY"]:
            index = role_index(w,folder,task,role)
            x, _, origin, rec = role_values(w,folder,task,role)
            assert sha(w / rec["relative"]) == rec["sha256"]
            inputs = torch.from_numpy(x).cuda()
            target = root / "banks" / entity / role
            target.mkdir(parents=True,exist_ok=True)
            starts = list(range(0,len(index),PART))
            for lo in starts[lane::lanes] + starts:
                part = target / f"part_{lo:05}.npz"
                idx = index[lo:lo+PART]
                with part.with_suffix(".lock").open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    manifest = part.with_suffix(".json")
                    if manifest.exists():
                        info = read(manifest)
                        assert info["sha256"] == sha(part) and info["weight_sha256"] == weights
                        continue
                    tic = time.monotonic()
                    vals = torch.empty((len(idx),len(names)),device="cuda")
                    for i, end in enumerate(idx):
                        t = int(end)-origin
                        assert 99 <= t < len(inputs)
                        window = inputs[t-99:t+1].unsqueeze(0)
                        order.clear()
                        with fixed_native_rng("cuda"):
                            enc, primary = base(window,norm=0,mask_mode="c",copies=10)
                        vals[i,0] = base.cal_anomaly_score(window,primary,None,1)[0,99]
                        stacked = torch.stack([models[k].decoder(enc) for k in names[1:]])
                        decoded = ifrft(stacked,order[0],dim=-1).abs()
                        for j,k in enumerate(names[1:],1):
                            copies = decoded[j-1].reshape(10,1,x.shape[1],100).permute(0,1,3,2)
                            if i == 0 and lo == 0:
                                with fixed_native_rng("cuda"):
                                    _, full = models[k](window,norm=0,mask_mode="c",copies=10)
                                torch.testing.assert_close(full,copies,rtol=0,atol=0)
                            vals[i,j] = models[k].cal_anomaly_score(window,copies,None,1)[0,99]
                    values = vals.cpu().numpy()
                    assert np.isfinite(values).all()
                    tmp = part.with_suffix(".part.npz")
                    np.savez_compressed(tmp,index=idx,keys=names,V=values)
                    tmp.replace(part)
                    dump(manifest,dict(sha256=sha(part),weight_sha256=weights,seal_sha256=seal_sha,
                        elapsed_seconds=time.monotonic()-tic,device=os.environ.get("CUDA_VISIBLE_DEVICES"),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),labels_read=False))
                    print(entity,role,lo,len(idx),round(time.monotonic()-tic,1),flush=True)
        hook.remove()


def merge(w):
    root = w / OUT
    banks = []
    for entity in DOMAINS:
        folder, task = target_task(w,entity)
        for role in ["CAL", "QUERY"]:
            index = role_index(w,folder,task,role)
            target = root / "banks" / entity / role
            parts = []
            expected = list(ARMS[1:] if entity=="PSM" else ARMS)
            weights = None
            cost = 0.
            seal = sha(root/"ALL_MODELS_FROZEN.json")
            for lo in range(0,len(index),PART):
                p = target / f"part_{lo:05}.npz"
                info = read(p.with_suffix(".json"))
                assert sha(p) == info["sha256"] and info["seal_sha256"] == seal
                assert weights is None or info["weight_sha256"] == weights
                weights = info["weight_sha256"]
                cost += info["elapsed_seconds"]
                with np.load(p) as z:
                    np.testing.assert_array_equal(z["index"],index[lo:lo+PART])
                    np.testing.assert_array_equal(z["keys"],expected)
                    parts.append(z["V"].copy())
            score = np.concatenate(parts,axis=0)
            prior = None
            if entity == "PSM":
                v, prior = reused_psm(w,role,index,task)
                score = np.column_stack([v,score])
            assert score.shape == (len(index),4) and np.isfinite(score).all()
            bank = target / "SCORES.npz"
            np.savez_compressed(bank,keys=list(ARMS),index=index,available_index=index,window_start=index-99,V=score)
            banks.append(dict(domain=entity,role=role,relative=bank.relative_to(w).as_posix(),sha256=sha(bank),
                index_sha256=task["indexes"][role]["sha256"],scaler_sha256=task["scaler_sha256"],
                weights=weights,noU_archive_sha256=prior,points=len(index),device_seconds=cost,
                source_seal_sha256=seal))
    dump(root/"SCORES_FROZEN.json",dict(banks=banks,labels_read=False))


def evaluate(w):
    root = w / OUT
    banks = read(root / "SCORES_FROZEN.json")["banks"]
    metric = load_author_metric(asset(w,"preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py",METRIC_SHA))
    methods, events = [], []
    for entity in DOMAINS:
        folder, task = target_task(w,entity)
        if entity=="PSM":
            truth = np.load(asset(w,"data/common_psm_20261028_v1/test_label.npy",LABEL_SHA)).reshape(-1)
        else:
            rec = read(folder/"EVALUATION_BINDING.json")
            truth = np.loadtxt(asset(w,rec["relative"],rec["sha256"]),delimiter=",",dtype=int).reshape(-1)
        arrays = {}
        for rec in banks:
            if rec["domain"] != entity:
                continue
            with np.load(asset(w,rec["relative"],rec["sha256"])) as z:
                ix = z["index"]
                np.testing.assert_array_equal(ix,role_index(w,folder,task,rec["role"]))
                np.testing.assert_array_equal(z["keys"],ARMS)
                arrays[rec["role"]] = (ix,z["V"].astype(float))
        ci, cal = arrays["CAL"]
        qi, query = arrays["QUERY"]
        start,stop = (23016,87744) if entity=="PSM" else (8957,23703)
        np.testing.assert_array_equal(qi,np.arange(start,stop))
        if entity=="PSM":
            assert truth[ci].sum()==512
        for j, arm in enumerate(ARMS):
            tau = exact_cal(ci,cal[:,j],ci)
            rank, _ = continuous_metrics(truth[qi],query[:,j],metric)
            rows, details, _, _ = evaluate_score(qi,query[:,j],truth,tau,qi,query_start=start,query_stop=stop,
                original_length=len(truth),expected_events=None)
            causal_timing(rows,details)
            for row in rows:
                ss = [{k: "" if v is None else str(v) for k,v in e.items()} for e in details if e["stream"]==row["stream"]]
                row.update(group_operating([row],ss))
                methods.append(dict(row, **rank,domain=entity,arm=arm,CAL_points=512,
                    CAL_exceedances=int((cal[:,j]>tau).sum()),CAL_selected_attack_points=512 if entity=="PSM" else None,
                    axis_start=start,axis_stop=stop))
            events.extend(dict(e,domain=entity,arm=arm) for e in details)
    write_csv(root/"methods.csv",methods)
    write_csv(root/"events.csv",events)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("score","merge","evaluate"))
    p.add_argument("--lane",type=int,default=0)
    p.add_argument("--lanes",type=int,default=1)
    args=p.parse_args()
    if args.action=="score":
        score(Path.cwd(),args.lane,args.lanes)
    elif args.action=="merge":
        merge(Path.cwd())
    else:
        evaluate(Path.cwd())
