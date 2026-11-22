"""Extend three locked PSM causal V banks to every legal 100-point test window."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .native_event_rank_readout import BASE, TASK_SHA, TARGET_SHA, LABEL_SHA, METRIC_SHA, asset
from .native_v_conditioning import load_native, fixed_native_rng, WEIGHT_SHA, state_hashes
from .native_v_conditioning_runner import dump, read, sha, setup
from .left_swat_replay_repeats import write_csv


RUN = "results/native_event_rank_psm_20261122_v1"
MODEL_SEAL = "12a86647125d392b75149e2e60e5277ac162409e0b6ea145709a36a4add0598d"
READOUT_SEAL = "e4b944d4458fd5b5132b5950ad3a21c537a27802580da04544395c26bbe73b0d"
ARMS = ("frozen_native_V", "copy", "point_weighted")
AXIS = np.arange(99, 87841)
EXTERNAL = {
    "LEFT W192 endpoint": ("results/left_causal_endpoint_20261109_v1/EVALUATION/full_causal__LEFT_R_CAUSAL_ENDPOINT.npz",
                          "76f61cb23917f40c1b9ebbcbf8396459cba730688969963a91458e4495bd8600"),
    "HistoryMean512": ("results/hybrid_v1_psm_Z1_fixed_evaluation_20261102_v1/history_mean512/calculation_inputs.npz",
                       "f1c792db8c65d95783c6ac183f5bbca22de3621164384d70c0aac4e40c6a423a"),
}


def scores(workspace, out):
    prior = workspace / RUN
    models = read(asset(prior, "MODELS_FROZEN.json", MODEL_SEAL))
    banks = read(asset(prior, "READOUT_FROZEN.json", READOUT_SEAL))
    base = workspace / BASE / "PSM"
    task = read(asset(base, "TASK.json", TASK_SHA))
    raw = np.load(asset(workspace, task["data"]["target_test"]["relative"], TARGET_SHA), allow_pickle=False)
    with np.load(asset(base, "SCALER.npz", task["scaler_sha256"])) as scaler:
        x = ((raw.astype(np.float64) - scaler["mean"]) / scaler["std"]).astype(np.float32)
    assert x.shape == (87841, 25) and np.isfinite(x).all()
    device = setup(12)
    receipts = {}
    for arm in ARMS:
        receipt = out / (arm + ".json")
        if receipt.exists():
            r = read(receipt)
            asset(out, r["file"], r["sha256"])
            receipts[arm] = r
            continue
        start = time.monotonic()
        model = load_native(workspace, device)
        digest = WEIGHT_SHA
        if arm != ARMS[0]:
            rec = models["arms"][arm]
            digest = rec["checkpoint_sha256"]
            state = torch.load(asset(prior, rec["checkpoint_relative"], digest), map_location=device, weights_only=True)
            assert state["arm"] == arm and state["update"] == rec["selected_step"]
            model.load_state_dict(state["model"], strict=True)
        initial = state_hashes(model)
        score = np.full(len(x), np.nan)
        reused = {}
        for role in ("CAL","QUERY"):
            if arm == ARMS[0]:
                r = read(workspace / BASE / "banks/PSM/A" / role / "RECEIPT.json")
                expected = {"CAL":"67ad82292b11cd8fd09469a3f272930ea75ebc067afef2cf8c6e24ea5f53b450",
                            "QUERY":"26e54dd3f0e74093c091356fe242efa0bd0c6ad5f37877298032eeb8210e965e"}[role]
                assert r["sha256"] == expected
            else:
                ref = banks["banks"][arm][role]
                r = read(asset(workspace, ref["relative"], ref["sha256"]))
                assert r["checkpoint_sha256"] == digest
            with np.load(asset(workspace, r["relative"], r["sha256"])) as z:
                index = z["absolute_time_index"]
                assert np.array_equal(index, z["available_index"]) and np.isnan(score[index]).all()
                score[index] = z["canonical_score"]
            reused[role] = {"count": len(index), "sha256": r["sha256"]}
        missing = AXIS[np.isnan(score[AXIS])]
        assert len(missing) == len(AXIS) - 64728 - 512
        copies = np.empty((len(missing),10,25),dtype=np.float32)
        with torch.inference_mode():
            for i, t in enumerate(missing):
                window = torch.from_numpy(x[t-99:t+1]).unsqueeze(0).to(device)
                with fixed_native_rng(device):
                    _, y = model(window, norm=0, mask_mode="c", copies=10)
                    score[t] = model.cal_anomaly_score(window, y, anomaly_criterion=None, L=1)[0,99].item()
                copies[i] = y[:,0,99].cpu().numpy()
        np.testing.assert_allclose(score[missing], copies.astype(np.float64).var(1,ddof=1).mean(1),rtol=1e-5,atol=1e-7)
        assert initial == state_hashes(model) and np.isfinite(score[AXIS]).all()
        path = out / (arm+".npz")
        np.savez_compressed(path,index=AXIS,available_index=AXIS,window_start=AXIS-99,score=score[AXIS],
                            new_index=missing,new_terminal_copies=copies)
        r = dict(file=path.name,sha256=sha(path),weight_sha256=digest,reused=reused,
                 new_points=len(missing),points=len(AXIS),scaler_sha256=task["scaler_sha256"],
                 elapsed_device_seconds=time.monotonic()-start, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 model_seal=MODEL_SEAL, previous_bank_seal=READOUT_SEAL, target_labels_read=False)
        dump(receipt,r)
        receipts[arm]=r
        print(arm,r["elapsed_device_seconds"],flush=True)
    dump(out/"SCORES_FROZEN.json", receipts)


def evaluate(workspace, out):
    receipts=read(out/"SCORES_FROZEN.json")
    assert set(receipts)==set(ARMS)
    labels=np.load(asset(workspace,"data/common_psm_20261028_v1/test_label.npy",LABEL_SHA)).reshape(-1)
    metric_path=asset(workspace,"preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py",METRIC_SHA)
    rows=[]
    for name in (*ARMS,*EXTERNAL):
        if name in receipts:
            r=receipts[name]
            path=asset(out,r["file"],r["sha256"])
            starts=(99,512)
        else:
            path=asset(workspace,*EXTERNAL[name])
            starts=(512,)
        with np.load(path) as bank:
            index, score=bank["index"],bank["score"].reshape(-1)
        for start in starts:
            keep=(index>=start)&(index<87841)
            axis=index[keep]
            np.testing.assert_array_equal(axis,np.arange(start,87841))
            m,_=continuous_metrics(labels[axis],score[keep],load_author_metric(metric_path))
            rows.append(dict(dataset="PSM",method=name,axis_start=start,axis_stop=87841,points=len(axis),
                             positives=int(labels[axis].sum()),score_sha256=sha(path),**m))
    write_csv(out/"PSM_RANKING.csv",rows)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("score","evaluate"))
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    (scores if args.action=="score" else evaluate)(args.workspace.resolve(),args.output.resolve())
