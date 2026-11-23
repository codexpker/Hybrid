"""Paired causal endpoint E/V benchmarks on all 28 physically separate SMD machines."""
import argparse
import fcntl
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from . import native_event_smd28 as old
from . import native_event_space
from .native_v_conditioning import fixed_native_rng, sha
from .native_v_conditioning_runner import dump, read, setup


OUT = Path("results/native_e_vs_v_smd28_20261123_v1")
E_DIR = Path("results/native_e_vs_v_20261123_v1/train/point_weighted_e")
KEYS = ["official_V", "official_E", "copy_V", "Vtrain_V", "Etrain_E",
        "official_V_plus_E", "Vtrain_E", "Etrain_V"]
OLD_COLS = [0, 2, 3]
FRESH_COLS = [1, 4, 5, 6, 7]


def load(w):
    base, _, v = native_event_space.models(w)
    path = w/E_DIR/"selected.pt"
    e_sha = read(w/E_DIR/"SELECTION.json")["selected_checkpoint_sha256"]
    assert sha(path) == e_sha
    e = deepcopy(base)
    e.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
    assert all(torch.equal(base.state_dict()[k], p) for k, p in e.state_dict().items()
               if not k.startswith("decoder."))
    return base, v, e, e_sha


def terminal(model, x, copies):
    return (model.cal_anomaly_score(x, copies, None, 1)[0, 99],
            (copies[:, :, 99].mean(dim=0) - x[:, 99]).square().mean())


@torch.inference_mode()
def score(w, lane, lanes):
    from torch_frft.frft_module import ifrft
    setup(12)
    base, v, e, e_sha = load(w)
    orders = []
    base.net.register_forward_hook(lambda _m, _a, value: orders.append(value.squeeze()))
    root = w/OUT
    root.mkdir(parents=True, exist_ok=True)
    jobs = sorted(old.records(w), key=lambda r: r["query_points"], reverse=True)
    for rec in jobs[lane::lanes] + jobs:
        entity = rec["id"]
        bank = root/f"{entity}.npz"
        with bank.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if bank.with_suffix(".json").exists():
                assert sha(bank) == read(bank.with_suffix(".json"))["sha256"]
                continue
            tic = time.monotonic()
            cx, co = old.data(w, rec, "CAL")
            qx, qo = old.data(w, rec, "QUERY")
            assert qo == 0 and sha(w/rec["scaler"]["path"]) == rec["scaler"]["sha256"]
            oldbank = w/old.OUT/f"{entity}.npz"
            receipt = read(oldbank.with_suffix(".json"))
            assert sha(oldbank) == receipt["sha256"] and receipt["scaler_sha256"] == rec["scaler"]["sha256"]
            with np.load(oldbank) as archived:
                ci, qi = archived["cal_index"], archived["query_index"]
                np.testing.assert_array_equal(ci, np.linspace(99, len(cx)-1, 512, dtype=np.int64)+co)
                np.testing.assert_array_equal(qi, np.arange(99,len(qx)))
                assert archived["keys"].tolist()[:3] == ["frozen_native_V", "2025_copy", "2025_point_weighted"]
                prior = np.concatenate([archived["cal"][:,:3], archived["query"][:,:3]])
            n = len(prior)
            result = np.empty((n, len(KEYS)), np.float32)
            result[:,OLD_COLS] = prior
            values = torch.from_numpy(np.concatenate((cx, qx))).cuda()
            ends = np.concatenate((ci-co, qi+len(cx)))
            parts = root / "parts" / entity
            parts.mkdir(parents=True, exist_ok=True)
            for offset in range(0, n, 1024):
                ix = ends[offset:offset+1024]
                path = parts / f"{offset:05}.npy"
                if path.exists():
                    out = np.load(path)
                else:
                    values_out = torch.empty((len(ix),7),device="cuda")
                    for j,t in enumerate(ix):
                        x = values[t-99:t+1].unsqueeze(0)
                        orders.clear()
                        with fixed_native_rng("cuda"):
                            enc, native = base(x, norm=0, mask_mode="c", copies=10)
                        order = orders[0]
                        rv, re = terminal(base, x, native)
                        projected = torch.stack([model.decoder(enc) for model in (v,e)])
                        reconstructed = ifrft(projected,order,dim=-1).abs()
                        out_scores = []
                        for k,model in enumerate((v,e)):
                            copies = reconstructed[k].reshape(10,1,38,100).permute(0,1,3,2)
                            if offset == j == 0:
                                with fixed_native_rng("cuda"):
                                    _, direct = model(x, norm=0, mask_mode="c", copies=10)
                                torch.testing.assert_close(copies, direct, rtol=0, atol=0)
                            out_scores.extend(terminal(model, x, copies))
                        vv, ve, ev, ee = out_scores
                        values_out[j] = torch.stack((re, ee, rv+re, ve, ev, rv, vv))
                    out = values_out.cpu().numpy()
                    assert np.isfinite(out).all()
                    np.testing.assert_array_equal(out[:,-2:], prior[offset:offset+len(ix)][:,[0,2]])
                    tmp = path.with_suffix(".part.npy")
                    np.save(tmp,out)
                    tmp.replace(path)
                    print(entity,offset+len(ix),n,round(time.monotonic()-tic,1),flush=True)
                np.testing.assert_array_equal(out[:,-2:], prior[offset:offset+len(ix)][:,[0,2]])
                result[offset:offset+len(ix),FRESH_COLS] = out[:,:5]
            np.testing.assert_array_equal(result[:,5], result[:,0]+result[:,1])
            tmp = bank.with_suffix(".part.npz")
            np.savez_compressed(tmp,keys=KEYS, cal=result[:len(ci)], query=result[len(ci):],
                cal_index=ci,query_index=qi,cal_window_start=ci-99,query_window_start=qi-99,
                cal_available_index=ci,query_available_index=qi)
            tmp.replace(bank)
            dump(bank.with_suffix(".json"),dict(entity=entity,sha256=sha(bank), prior_bank_sha256=receipt["sha256"],
                old_weights_sha256=receipt["weights_sha256"][:3],E_weights_sha256=e_sha,source_binding_sha256=old.BINDING_SHA,
                scaler_sha256=rec["scaler"]["sha256"],cal_points=len(ci),query_points=len(qi),
                query_start=int(qi[0]),query_stop=int(qi[-1])+1,exact_agreement_old_V_every_window=True,
                labels_read=False,GPU=os.environ["CUDA_VISIBLE_DEVICES"],seconds=time.monotonic()-tic,
                peak_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lane",type=int,required=True)
    p.add_argument("--lanes",type=int,default=4)
    args = p.parse_args()
    score(Path.cwd(),args.lane,args.lanes)
