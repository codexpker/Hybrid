"""Frozen head-only seed repeats on all eligible NASA sensor-0 entities."""
import argparse
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

from . import native_event_space as previous
from .left_swat_replay_repeats import write_csv
from .native_event_space_summary import RANKS, group_operating, read_csv
from .native_v_conditioning import fixed_native_rng, load_native, sha
from .native_v_conditioning_runner import dump, read

OUT = Path("results/native_event_space_repeat_20261122_v1")
SOURCE = Path("results/native_event_rank_seed_repeats_20261122_v1")
SEEDS = (2026, 2027)
METHODS = ("copy", "point_weighted")
CASES = tuple((s, m) for s in SEEDS for m in METHODS)


def read_models(w):
    """Bind selected tensors to their source-only seal; identify reusable official columns."""
    publication = read(w / SOURCE / "ALL_MODELS_FROZEN.json")
    assert publication["status"] == "COMPLETED" and publication["target_read"] is False
    initial = load_native(w, "cpu")
    nets, keys, digests = [], [], []
    for seed, method in CASES:
        root = w / SOURCE / f"seed{seed}"
        seal_path = root / "MODELS_FROZEN.json"
        assert sha(seal_path) == publication["seeds"][str(seed)]["sha256"]
        seal = read(seal_path)
        receipt = seal["arms"][method]
        assert seal["seed"] == seed and seal["status"] == "COMPLETED"
        ck = root / receipt["checkpoint_relative"]
        assert sha(ck) == receipt["checkpoint_sha256"]
        assert sha(root / receipt["selection_relative"]) == receipt["selection_sha256"]
        saved = torch.load(ck, map_location="cpu", weights_only=True)
        assert saved["update"] == receipt["selected_step"]
        model = deepcopy(initial)
        model.load_state_dict(saved["model"], strict=True)
        step0 = saved["update"] == 0
        for name, value in model.state_dict().items():
            if step0 or not name.startswith("decoder."):
                assert torch.equal(value, initial.state_dict()[name]), name
        keys.append(f"{seed}_{method}")
        digests.append(receipt["checkpoint_sha256"])
        nets.append(None if step0 else model.cuda().eval())
    return nets, keys, digests


@torch.inference_mode()
def score(w, lane, lanes):
    from torch_frft.frft_module import ifrft

    nets, keys, digests = read_models(w)
    active = [j for j, model in enumerate(nets) if model is not None]
    base = nets[active[0]] if active else None
    orders = []
    if active:
        base.net.register_forward_hook(lambda module, args, out: orders.append(out.squeeze()))
    jobs = []
    for dataset in ("MSL", "SMAP"):
        for entity in previous.entity_list(w, dataset):
            old = w / previous.OUT / dataset / f"{entity}.npz"
            info = read(old.with_suffix(".json"))
            jobs.append((info["hash_identical_windows"], dataset, entity, old, info))
    # Each lane starts at a different high-work entity and steals remaining unlocked work.
    jobs.sort(reverse=True)
    for _, dataset, entity, old, info in jobs[lane::lanes] + jobs:
        root = w / OUT / dataset
        root.mkdir(parents=True, exist_ok=True)
        bank = root / f"{entity}.npz"
        receipt_file = bank.with_suffix(".json")
        with bank.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if receipt_file.exists():
                receipt = read(receipt_file)
                assert receipt["checkpoint_sha256"] == digests and receipt["sha256"] == sha(bank)
                continue
            started = time.monotonic()
            assert sha(old) == info["sha256"] and info["arms"][0] == "frozen_native_V"
            windows, cal, query, binding = previous.read_roles(w, dataset, entity)
            assert all(info[k] == v for k, v in binding.items())
            flat = np.ascontiguousarray(np.concatenate(windows))
            _, first, inverse = np.unique(flat.view("V400").ravel(), return_index=True, return_inverse=True)
            assert len(first) == info["hash_identical_windows"]
            values = np.empty((len(flat),len(keys)), dtype=np.float32)
            with np.load(old) as z:
                np.testing.assert_array_equal(cal, z["cal_index"])
                np.testing.assert_array_equal(query, z["query_index"])
                for j in range(len(keys)):
                    if j not in active:
                        values[:,j] = np.concatenate((z["cal"][:,0], z["query"][:,0]))
            if active:
                single = torch.from_numpy(flat[first]).cuda()[:,:,None]
                result = torch.empty((len(first),len(active)), device="cuda")
                for i in range(len(first)):
                    x = single[i:i+1]
                    orders.clear()
                    with fixed_native_rng("cuda"):
                        encoded, copies = base(x,norm=0,mask_mode="c",copies=10)
                    order = orders[0]
                    for column, j in enumerate(active):
                        model = nets[j]
                        if column:
                            decoded = ifrft(model.decoder(encoded),order,dim=-1).abs().reshape(10,1,1,100).permute(0,1,3,2)
                            if i == 0:
                                with fixed_native_rng("cuda"):
                                    _, direct = model(x,norm=0,mask_mode="c",copies=10)
                                torch.testing.assert_close(decoded,direct,atol=0,rtol=0)
                        else:
                            decoded = copies
                        result[i,column] = model.cal_anomaly_score(x,decoded,None,1)[0,99]
                values[:,active] = result.cpu().numpy()[inverse.reshape(-1)]
            assert np.isfinite(values).all()
            tmp = bank.with_suffix(".part.npz")
            np.savez_compressed(tmp,cal_index=cal,query_index=query,cal=values[:len(cal)],query=values[len(cal):])
            tmp.replace(bank)
            dump(receipt_file, dict(dataset=dataset,entity=entity,**binding,arms=keys,checkpoint_sha256=digests,
                 reused_frozen=[keys[j] for j in range(len(keys)) if j not in active],
                 inputs_archive_sha256=info["sha256"],hash_identical_windows=len(first),
                 cal_points=len(cal),query_points=len(query),sha256=sha(bank),seconds=time.monotonic()-started,
                 masked_B1=True, label_read=False,device_slot=os.environ.get("CUDA_VISIBLE_DEVICES"),
                 device_name=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            print(dataset,entity,len(first),round(time.monotonic()-started,1),flush=True)


def metrics(w, dataset):
    root = w / OUT / dataset
    ids = previous.entity_list(w,dataset)
    info = read(root/f"{ids[0]}.json")
    keys = info["arms"]
    frozen = info["reused_frozen"]
    for entity in ids:
        binding = read(root/f"{entity}.json")
        assert all(binding[k] == info[k] for k in ("checkpoint_sha256","arms","reused_frozen"))
    with (w/previous.MSL/"labels.csv").open() as handle:
        records = {row["chan_id"]: row for row in csv.DictReader(handle) if row["spacecraft"] == dataset}
    columns = [(j, key) for j,key in enumerate(keys) if key not in frozen]
    methods, events = [], []
    if columns:
        args = [(w,root,dataset,e,records[e]) for e in ids]
        with ProcessPoolExecutor(max_workers=4) as pool:
            fn = partial(previous.entity_metrics,method_columns=columns)
            for rr,ee in pool.map(fn,args):
                methods.extend(rr)
                events.extend(ee)
    for filename, output in (("methods.csv",methods), ("events.csv",events)):
        if frozen:
            old = [row for row in read_csv(w / previous.OUT / dataset / filename) if row["method"] == "frozen_native_V"]
            for key in frozen:
                output.extend(dict(row,method=key) for row in old)
        for row in output:
            seed,method = row["method"].split("_",1)
            row.update(seed=int(seed), method=method)
        write_csv(root / filename,output)


def summarize(w):
    root = w / OUT
    summary, ops, entities, ledger = [], [], [], []
    for dataset in ("MSL","SMAP"):
        ids = previous.entity_list(w,dataset)
        rows = read_csv(root/dataset/"methods.csv")
        events = read_csv(root/dataset/"events.csv")
        assert len(rows) == len(ids)*len(CASES)*2
        for entity in ids:
            info = read(root / dataset / f"{entity}.json")
            ledger.append({k:info[k] for k in ("entity","dataset","sha256","inputs_archive_sha256","cal_points",
                           "query_points","device_slot","seconds","peak_allocated_bytes","hash_identical_windows","reused_frozen","checkpoint_sha256")})
        for seed, method in CASES:
            filtered = [r for r in rows if r["seed"] == str(seed) and r["method"]==method]
            ranked = [r for r in filtered if r["stream"]=="strict"]
            assert sorted(r["entity"] for r in ranked) == ids
            for r in ranked:
                entities.append({k:r[k] for k in ("dataset","entity","method","seed",*RANKS)})
            values = {k:np.array([float(r[k]) for r in ranked]) for k in RANKS}
            summary.append(dict(dataset=dataset,method=method,seed=seed,entities=len(ranked),
                                **{k:v.mean() for k,v in values.items()},**{k+"_median":np.median(v) for k,v in values.items()}))
            for stream in ("strict","confirmed3"):
                group = [r for r in filtered if r["stream"]==stream]
                details = [e for e in events if e["method"]==method and e["seed"]==str(seed) and e["stream"]==stream]
                ops.append(dict(dataset=dataset,method=method,seed=seed,stream=stream,**group_operating(group,details)))
    for filename, content in (("ranking_macro",summary),("operating",ops),("entity_ranking",entities),("banks",ledger)):
        write_csv(root/f"{filename}.csv",content)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action",choices=["score","metrics","summarize"])
    p.add_argument("--dataset",choices=["MSL","SMAP"])
    p.add_argument("--lane",type=int,default=0)
    p.add_argument("--lanes",type=int,default=3)
    args = p.parse_args()
    if args.action=="score":
        score(Path.cwd(),args.lane,args.lanes)
    elif args.action=="metrics":
        metrics(Path.cwd(),args.dataset)
    else:
        summarize(Path.cwd())
