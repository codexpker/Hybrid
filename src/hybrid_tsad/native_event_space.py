"""Frozen entity-local Telemanom sensor-0 extension of the native variance fold."""
import argparse
import ast
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .native_v_conditioning import fixed_native_rng, load_native, sha
from .native_v_conditioning_evaluation import causal_timing
from .native_v_conditioning_runner import dump, read
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .timercd_swat_evaluation import evaluate_score

ARMS = ("frozen_native_V", "copy", "point_weighted")
MSL = Path("data/native_event_msl_v1")
SMAP = Path("data/roca_smap_target_roles_20261106_v1")
SMAP_QUERY = Path("data/hybrid_v1_smap_native_zs_20261105_v1")
SOURCE_RUN = Path("results/native_event_rank_psm_20261122_v1")
OUT = Path("results/native_event_space_20261122_v1")


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        if not z["observed"].all():
            raise ValueError(f"Missing samples in {path}")
        return z["values"][:, 0].astype("float64")


def read_roles(workspace, dataset, entity):
    if dataset == "SMAP":
        base = workspace / SMAP / entity
        cal = load_npz(base / "CAL.npz")
        source = workspace / "data/roca_common_fit_scalers_20261106_v1" / f"SMAP__{entity}.npz"
        with np.load(source) as z:
            mean, scale = float(z["mean"][0]), float(z["scale"][0])
        fit_count = len(load_npz(base / "FIT.npz"))
        origin = fit_count + 15
        anchors = [row["index"] for row in read(workspace / SMAP / "CAL32_PER_ENTITY.PLAN.json")
                   if row["entity"] == entity]
        query = load_npz(workspace / SMAP_QUERY / f"{entity}.npz")
    else:
        train = np.load(workspace / MSL / "train" / f"{entity}.npy")[:, 0].astype(np.float32).astype("float64")
        query = np.load(workspace / MSL / "test" / f"{entity}.npy")[:, 0].astype(np.float32).astype("float64")
        fit_count = int(len(train) * .6)
        origin = fit_count + 15
        cal = train[origin:]
        anchors = np.linspace(origin + 15, len(train)-1, 32, dtype=np.int64)
        scaler = StandardScaler().fit(train[:fit_count, None])
        mean, scale = float(scaler.mean_[0]), float(scaler.scale_[0])
    cal_ends = np.array([int(t)-origin for t in anchors if int(t)-origin >= 99], dtype=np.int64)
    assert len(set(cal_ends)) == len(cal_ends) and len(cal_ends) > 0 and cal_ends.max() < len(cal)
    query_ends = np.arange(99, len(query))
    windows = []
    for values, ends in ((cal,cal_ends), (query,query_ends)):
        assert len(ends) and np.isfinite(values).all()
        x = np.asarray((values - mean) / scale, dtype=np.float32)
        windows.append(np.lib.stride_tricks.sliding_window_view(x,100)[ends-99])
    return windows, cal_ends + origin, query_ends, dict(fit_count=fit_count, mean=mean, scale=scale,
            cal_origin=origin, cal_stop=origin+len(cal), query_size=len(query))


def entity_list(w, dataset):
    group = w / (SMAP_QUERY if dataset == "SMAP" else MSL / "test")
    ids = sorted(p.stem for p in group.glob("*.npz" if dataset == "SMAP" else "*.npy"))
    if len(ids) != (53 if dataset == "SMAP" else 27):
        raise ValueError("Mismatch with fixed native entity list")
    return ids


def models(w):
    from copy import deepcopy
    base = load_native(w, "cuda")
    seal_path = w / SOURCE_RUN / "MODELS_FROZEN.json"
    assert sha(seal_path) == "12a86647125d392b75149e2e60e5277ac162409e0b6ea145709a36a4add0598d"
    all_models = [base]
    for arm in ARMS[1:]:
        receipt = read(seal_path)["arms"][arm]
        ck = w / SOURCE_RUN / receipt["checkpoint_relative"]
        assert sha(ck) == receipt["checkpoint_sha256"]
        model = deepcopy(base)
        model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["model"])
        assert all(torch.equal(value,base.state_dict()[key]) for key,value in model.state_dict().items()
                   if not key.startswith("decoder."))
        all_models.append(model)
    return all_models


def score(w, dataset, lane):
    import fcntl
    from torch_frft.frft_module import ifrft
    start = time.monotonic()
    out = w / OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    ids = entity_list(w, dataset)
    selection = ids[lane::3]
    nets = models(w)
    base = nets[0]
    order = []
    base.net.register_forward_hook(lambda mod, args, val: order.append(val.squeeze()))
    with torch.inference_mode():
        for entity in selection:
            bank = out / f"{entity}.npz"
            with (out / f"{entity}.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if bank.exists() and bank.with_suffix(".json").exists():
                    assert sha(bank) == read(bank.with_suffix(".json"))["sha256"]
                    continue
                t0 = time.monotonic()
                windows, ci, qi, binding = read_roles(w,dataset,entity)
                flat = np.ascontiguousarray(np.concatenate(windows))
                _, first, inverse = np.unique(flat.view("V400").ravel(), return_index=True, return_inverse=True)
                single = torch.from_numpy(flat[first]).to("cuda")[:,:,None]
                result = torch.empty((len(first),3), device="cuda")
                for i in range(len(first)):
                    x = single[i:i+1]
                    order.clear()
                    with fixed_native_rng("cuda"):
                        encoded, copies = base(x,norm=0,mask_mode="c",copies=10)
                    angle = order[0]
                    result[i,0] = base.cal_anomaly_score(x,copies,None,1)[0,99]
                    for j, model in enumerate(nets[1:],1):
                        decoded = ifrft(model.decoder(encoded),angle,dim=-1).abs().reshape(10,1,1,100).permute(0,1,3,2)
                        value = model.cal_anomaly_score(x,decoded,None,1)[0,99]
                        if i == 0:
                            with fixed_native_rng("cuda"):
                                _, direct = model(x,norm=0,mask_mode="c",copies=10)
                            torch.testing.assert_close(decoded,direct,atol=0,rtol=0)
                        result[i,j] = value
                values = result.cpu().numpy()[inverse.reshape(-1)]
                assert np.isfinite(values).all()
                np.savez_compressed(bank, cal_index=ci, query_index=qi, cal=values[:len(ci)], query=values[len(ci):])
                dump(bank.with_suffix(".json"),dict(dataset=dataset, entity=entity, **binding, arms=ARMS,
                     hash_identical_windows=len(first), cal_points=len(ci), query_points=len(qi), sha256=sha(bank),
                     seconds=time.monotonic()-t0, masked_B1=True, label_read=False))
                print(dataset, entity, len(first), round(time.monotonic()-t0,1), flush=True)
    print("Completed lane",lane,dataset,time.monotonic()-start,flush=True)


def operating_metrics(index, scores, truth, tau, ranking):
    from .hybrid_v1.evaluation import _coverage, _event_view, _normal
    from .alarm_complementarity import confirmed, point_counts
    valid = np.zeros(len(truth), dtype=bool)
    valid[index] = True
    intervals = _coverage(truth, valid, 0)
    if any(event["main_eligible"] for event in intervals):
        rows, events, _, _ = evaluate_score(index,scores,truth,tau,index,query_start=99,query_stop=len(truth),
                      original_length=len(truth),expected_events=None,ranking=ranking)
        causal_timing(rows,events)
        return rows, events
    rows, details = [], []
    for stream, flags in (("strict", scores>tau), ("confirmed3", confirmed(index,scores>tau))):
        bits = np.zeros(len(truth), dtype=bool)
        bits[index] = flags
        view, _ = _event_view(truth,valid,bits,np.arange(len(truth)),intervals,0,"main_eligible")
        normal, _ = _normal(truth,valid,bits,0)
        rows.append(dict(stream=stream,threshold=tau,**ranking,**point_counts(truth,valid,bits),
                         **view["metrics"],**{k:v for k,v in normal.items() if not isinstance(v,list)},
                         mode="CAUSAL_ENDPOINT",scored_points=len(index),long_event_count=0,long_steady_total_points=0,
                         event_macro_coverage=None,event_point_coverage=None,worst_event_coverage=None,
                         long_steady_macro_coverage=None,long_steady_point_coverage=None,max_event_miss_points=None,
                         delay_hit_only_median=None))
        for event, entry in zip(intervals,view["events"]):
            details.append(dict(event,**{k:v for k,v in entry.items() if k not in event},
                                stream=stream,coverage=None,longest_miss_points=None,steady_after512_coverage=None,
                                steady_longest_miss_points=None,observed_alarm_points=int(bits[event["local_start"]:event["local_end"]].sum()),
                                timing_status="CAUSAL_ENDPOINT",observation_status="UNOBSERVED_OR_INELIGIBLE"))
    return rows, details


def metrics(w, dataset):
    root = w / OUT / dataset
    ids = entity_list(w,dataset)
    assert all((root/f"{k}.json").is_file() for k in ids)
    with (w/MSL/"labels.csv").open() as f:
        records = {row["chan_id"]:row for row in csv.DictReader(f) if row["spacecraft"]==dataset}
    metric = load_author_metric(w/"preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py")
    rows,events = [], []
    for entity in ids:
        raw = records[entity]
        truth = np.zeros(int(raw["num_values"]), dtype=int)
        for start,stop in ast.literal_eval(raw["anomaly_sequences"]):
            truth[start:stop+1] = 1
        bank = root / f"{entity}.npz"
        info = read(bank.with_suffix(".json"))
        assert len(truth)==info["query_size"] and info["sha256"]==sha(bank)
        with np.load(bank) as z:
            qi = z["query_index"]
            np.testing.assert_array_equal(qi,np.arange(99,len(truth)))
            for i,arm in enumerate(ARMS):
                ranked = z["query"][:,i]
                values,_ = continuous_metrics(truth[qi],ranked,metric)
                tau = float(np.quantile(z["cal"][:,i],.99,method="higher"))
                rr,ee=operating_metrics(qi,ranked,truth,tau,values)
                for r in rr:
                    rows.append(dict(r,**values,entity=entity,dataset=dataset,method=arm,cal_points=info["cal_points"]))
                for e in ee:
                    events.append(dict(e,entity=entity,dataset=dataset,method=arm))
    from .left_swat_replay_repeats import write_csv
    write_csv(root/"methods.csv",rows)
    write_csv(root/"events.csv",events)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=["score","metrics"])
    p.add_argument("dataset",choices=["MSL","SMAP"])
    p.add_argument("--lane",type=int,choices=[0,1,2])
    p.add_argument("--workspace",type=Path,default=Path.cwd())
    args = p.parse_args()
    if args.action=="score":
        score(args.workspace,args.dataset,args.lane)
    else:
        metrics(args.workspace,args.dataset)
