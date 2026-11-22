"""Score seven source-sealed TimeRadar detectors on the 28 separate SMD test records."""
import argparse
import fcntl
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import native_event_space as original
from .native_event_space_repeat import read_models
from .native_v_conditioning import fixed_native_rng, sha, WEIGHT_SHA
from .native_v_conditioning_runner import dump, read, setup

OUT = Path("results/native_event_smd28_20261123_v1")
BINDING = Path("preparations/timeradar_smd_full_20261116_v1/data_binding_full28.json")
BINDING_SHA = "501184abd4dbd09e9d18a60cc84c88de05eb77f088f706286bb111b28fb51c73"
ENTITIES = tuple(f"machine-{g}-{i}" for g, n in ((1, 8), (2, 9), (3, 11)) for i in range(1, n + 1))


def records(w):
    assert sha(w / BINDING) == BINDING_SHA
    rows = read(w / BINDING)["records"]
    assert len(rows) == 28 and {r["id"] for r in rows} == set(ENTITIES)
    return rows


def data(w, r, role):
    binding, = (b for b in r["roles"] if b["role"] == role)
    assert binding["entity"] == r["id"] and binding["run"] == {"CAL":"train","QUERY":"test"}[role]
    assert sha(w / binding["path"]) == binding["sha256"]
    lo, hi = binding["bounds_half_open"]
    with np.load(w / binding["path"]) as z:
        x = z["x"]
        np.testing.assert_array_equal(z["index"], np.arange(lo, hi))
    assert x.shape == (hi-lo, 38) and x.dtype == np.float32 and np.isfinite(x).all()
    assert r["channel_order"] == [f"feature_{i}" for i in range(38)]
    return x, lo


def load(w):
    first = original.models(w)
    rest, labels, digests = read_models(w)
    assert all(m is not None for m in rest)
    seal = read(w / original.SOURCE_RUN / "MODELS_FROZEN.json")["arms"]
    return (first + rest, ["frozen_native_V","2025_copy","2025_point_weighted"] + labels,
            [WEIGHT_SHA] + [seal[m]["checkpoint_sha256"] for m in original.ARMS[1:]] + digests)


@torch.inference_mode()
def score(w, lane, lanes, grouped_inverse=False):
    from torch_frft.frft_module import ifrft
    setup(12)
    nets, keys, weights = load(w)
    base = nets[0]
    order = []
    base.net.register_forward_hook(lambda _m, _a, value: order.append(value.squeeze()))
    root = w / OUT
    root.mkdir(parents=True, exist_ok=True)
    jobs = sorted(records(w), key=lambda r: r["query_points"], reverse=True)
    for r in jobs[lane::lanes] + jobs:
        rid = r["id"]
        bank = root / f"{rid}.npz"
        with bank.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            receipt = bank.with_suffix(".json")
            if receipt.exists():
                assert sha(bank) == read(receipt)["sha256"]
                continue
            assert sha(w / r["scaler"]["path"]) == r["scaler"]["sha256"]
            start = time.monotonic()
            cx, lo = data(w,r,"CAL")
            qx, qlo = data(w,r,"QUERY")
            assert qlo == 0 and len(qx) == r["query_points"]
            ci = np.linspace(99,len(cx)-1,512,dtype=np.int64)
            assert len(np.unique(ci)) == 512
            qi = np.arange(99,len(qx))
            time_indices = np.r_[ci,qi + len(cx)]
            joined = torch.from_numpy(np.concatenate((cx,qx))).cuda()
            parts = root / "parts" / rid
            parts.mkdir(parents=True, exist_ok=True)
            scores = np.empty((len(time_indices),len(keys)),dtype=np.float32)
            for offset in range(0,len(scores),1024):
                ends = time_indices[offset:offset+1024]
                part = parts / f"{offset:05d}.npy"
                if part.exists():
                    scores[offset:offset+len(ends)] = np.load(part)
                    continue
                values = torch.empty((len(ends), len(keys)), device="cuda")
                for i, t in enumerate(ends):
                    x = joined[t-99:t+1].unsqueeze(0)
                    order.clear()
                    with fixed_native_rng("cuda"):
                        encoded, native = base(x,norm=0,mask_mode="c",copies=10)
                    values[i,0] = base.cal_anomaly_score(x,native,None,1)[0,99]
                    angle = order[0]
                    # The extra dimension enumerates decoder checkpoints on ONE window.
                    # Its FRFT order is the same scalar for every checkpoint.
                    if grouped_inverse:
                        projected = torch.stack([model.decoder(encoded) for model in nets[1:]])
                        copies = ifrft(projected,angle,dim=-1).abs()
                    for j, model in enumerate(nets[1:],1):
                        reconstructed = copies[j-1] if grouped_inverse else ifrft(model.decoder(encoded),angle,dim=-1).abs()
                        decoded = reconstructed.reshape(10,1,38,100).permute(0,1,3,2)
                        if i == 0 and offset == 0:
                            with fixed_native_rng("cuda"):
                                _, direct = model(x,norm=0,mask_mode="c",copies=10)
                            torch.testing.assert_close(decoded,direct,rtol=0,atol=0)
                        values[i,j] = model.cal_anomaly_score(x,decoded,None,1)[0,99]
                local = values.cpu().numpy()
                assert np.isfinite(local).all()
                tmp = part.with_suffix(".part.npy")
                np.save(tmp,local)
                tmp.replace(part)
                scores[offset:offset+len(ends)] = local
                print(rid,offset+len(ends),len(scores),round(time.monotonic()-start,1),flush=True)
            np.savez_compressed(bank,keys=keys,cal_index=ci+lo,query_index=qi,cal=scores[:len(ci)],query=scores[len(ci):])
            dump(receipt,dict(entity=rid,keys=keys,weights_sha256=weights,sha256=sha(bank),source_binding_sha256=BINDING_SHA,
                       scaler_sha256=r["scaler"]["sha256"],cal_points=len(ci),cal_first=int(ci[0]+lo),
                       cal_last=int(ci[-1]+lo),query_points=len(qi),query_start=99,query_stop=len(qx),
                       grouped_inverse=grouped_inverse,gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),seconds=time.monotonic()-start,
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),labels_read=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane",type=int,default=0)
    parser.add_argument("--lanes",type=int,default=2)
    parser.add_argument("--grouped-inverse",action="store_true")
    args = parser.parse_args()
    score(Path.cwd(),args.lane,args.lanes,args.grouped_inverse)
