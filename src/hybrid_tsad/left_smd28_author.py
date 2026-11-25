"""Entity-isolated SMD28 benchmark using the upstream LEFT train and infer paths."""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .left_swat_author import dump, read, sha

RUN = Path("results/left_smd28_author_20261123_v1")
PREP = Path("preparations/left_smd28_author_20261123_v1")
OLD = Path("preparations/timeradar_smd_full_20261116_v1")
REV = "3fadb4811797075e233076b9faceb0d8ec59b697"
CONFIG = {"train_configs.json":"0a6b4cc8bb0b53f8ed63bfd1ad71741a15114ab7b458827b819054a90bbf6995",
          "model_configs_Left.json":"dd5aaf0238ea3e1f10bd5d944b6baacd53067eba8759db081c29e36620bc3bd4",
          "model_configs_0.json":"ac14696992d5a005a444fac0c75cce172a94b2d5138006593d004a5105268d54"}


def bindings(w):
    records = read(w/OLD/"data_binding_full28.json")["records"]
    assert len({r["id"] for r in records}) == len(records) == 28
    return {r["id"]:r for r in records}


def checked_data(w, record, split):
    item = record["source"][{"train":0, "test":1}[split]]
    p = Path(item["path"])
    if not p.is_file():
        p = w/PREP/"raw"/f"{record['id']}_{split}.txt"
    assert p.name == f"{record['id']}_{split}.txt" and sha(p) == item["sha256"]
    a = np.asfortranarray(np.loadtxt(p, delimiter=",", dtype=np.float64))
    assert a.shape == (record["train_points" if split == "train" else "query_points"],38)
    assert np.isfinite(a).all()
    return a


def run_entity(w, entity, arm=None, *, check_only=False):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    if arm is not None:
        from .left_decoder_context_run import (PUBLIC, fingerprint, freeze_public_detection,
                                               public_kind, public_model_choices, sampler_fingerprint)
        check_dir = w/PUBLIC/'SMD'/'observation_check'
        output = check_dir if check_only else w/PUBLIC/'SMD'/entity/arm
    else:
        output = w/RUN/entity
    output.mkdir(parents=arm is not None, exist_ok=False)
    start = time.monotonic()
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory)
    upstream = w/"preparations/left_baseline_build_20261101_v1/upstream"
    sys.path[:0] = [str(upstream), str(upstream.parent/"deps_thop_transport2_v1/site")]
    import exp.exp_anomaly_detection as official
    configs = w/PREP/"configs"
    assert {k:sha(configs/"SMD"/k) for k in CONFIG} == CONFIG
    record = bindings(w)[entity]
    normal = checked_data(w, record, "train")
    scaler = StandardScaler().fit(normal)
    norm = scaler.transform(normal)
    train_end = int(len(norm)*.8)
    if arm is not None:
        previous = w/RUN/entity
        ref = read(previous/"TRAIN.json")
        assert ref["upstream"] == REV and ref["seed"] == 2025 and ref["configs"] == CONFIG
        assert ref["fit"] == [0, train_end] and ref["validation"] == [train_end, len(norm)]
        assert ref["scaler"] == [0, len(norm)] and ref["train_sha256"] == record["source"][0]["sha256"]
        with np.load(previous/"scaler.npz", allow_pickle=False) as old:
            for key in ("mean", "scale", "var"):
                np.testing.assert_array_equal(getattr(scaler, key+"_"), old[key])
        kind = public_kind(w, official, arm)
    np.savez(output/"scaler.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)
    selections, losses = [], []
    sampler = []
    train_steps = 0

    class Windows(Dataset):
        def __init__(self, values, stride):
            self.values, self.stride = values, stride

        def __len__(self):
            return (len(self.values)-192)//self.stride+1

        def __getitem__(self, i):
            a = i*self.stride
            return self.values[a:a+192].astype(np.float32), np.zeros((192,1),np.float32)

    class Stop(official.EarlyStopping):
        def save_checkpoint(self, val_loss, model, path):
            super().save_checkpoint(val_loss, model, path)
            selections.append(dict(epoch=len(losses), val=float(val_loss), step=model.step))

    official.EarlyStopping = Stop

    class Experiment(official.Exp_Anomaly_Detection):
        def _acquire_device(self):
            return torch.device("cuda:0")

        def _build_model(self):
            return super()._build_model() if arm is None else kind(self.hyper_args)

        def _get_data(self, flag, step=None):
            nonlocal train_steps
            assert flag in ("train","val")
            ds = Windows(norm[:train_end] if flag=="train" else norm[train_end:],192 if step is None else step)
            dl = DataLoader(ds,batch_size=128,shuffle=flag=="train",drop_last=False,num_workers=0)
            if flag=="train":
                train_steps = len(dl)
            return ds, dl

        def vali(self, loader):
            value = float(super().vali(loader))
            assert np.isfinite(value)
            losses.append(value)
            if arm is not None:
                sampler.append(sampler_fingerprint())
                dump(output/"PROGRESS.json", dict(epoch=len(losses), validation_loss=value,
                     updates=train_steps*len(losses), cpu_rng_sha256=sampler[-1]))
            return value

    exp = Experiment(SimpleNamespace(data="SMD", data_origin="DADA", model_class="Left", gpu=0,
        use_gpu=True, gpu_type="cuda", use_multi_gpu=False, configs_path=str(configs),
        model_save_path=str(output/"model"),results_save_path=str(output/"unused")),id=0)
    metadata = {}
    if arm is not None:
        sample = np.stack([norm[:192], norm[192:384]])
        choices = public_model_choices(exp, arm, output, check_dir/'CHECKS.json', sample,
                                       check_only=check_only)
        if check_only:
            return
        metadata.update(choices, sampler_cpu_rng_sha256=sampler)
    exp.train()
    pt = Path(exp.model_save_path)/"checkpoint.pth"
    assert pt.is_file() and min(losses) == selections[-1]["val"]
    info = dict(entity=entity,seed=2025,upstream=REV,worker_sha256=sha(__file__),configs=CONFIG,
        device=os.environ.get("CUDA_VISIBLE_DEVICES"),hardware=torch.cuda.get_device_name(),
        fit=[0,train_end], validation=[train_end,len(norm)], scaler=[0,len(norm)],
        train_sha256=record["source"][0]["sha256"],scaler_sha256=sha(output/"scaler.npz"),
        loss_history=losses, checkpoints=selections, epochs=len(losses), selected=selections[-1],
        updates=train_steps*len(losses), final_step=exp.model.step,
        checkpoint_sha256=sha(pt), train_seconds=time.monotonic()-start,
        train_peak_reserved=torch.cuda.max_memory_reserved(), test_read_in_training=False, **metadata)
    if arm is not None:
        info["hardware"] = torch.cuda.get_device_name()
    dump(output/"TRAIN.json",info)
    torch.cuda.reset_peak_memory_stats()
    tic = time.monotonic()
    test = checked_data(w,record,"test")
    ds = Windows(scaler.transform(test),192)
    scoring_metadata = {}
    if arm is not None:
        lam, state, step = freeze_public_detection(exp.model, arm)
        scoring_metadata.update(detection_modulation=metadata["detection_modulation"], native_lambda=lam)
    else:
        exp.model.eval()
    scores = []
    with torch.inference_mode():
        for x,_ in DataLoader(ds,batch_size=128,shuffle=False):
            score = exp.model.infer(x.cuda(),None,None,None)[0]
            assert score.shape == x.shape and torch.isfinite(score).all()
            scores.append(score.cpu().numpy())
    if arm is not None:
        assert fingerprint(exp.model.state_dict()) == state and exp.model.step == step
        assert exp.model._curriculum_lambda() == lam
        scoring_metadata["state_unchanged"] = True
    raw = np.concatenate(scores,axis=0).reshape(-1,38)
    score = raw.mean(axis=1)
    bank = output/"scores.npz"
    np.savez_compressed(bank,score=score,index=np.arange(len(score)), per_channel=raw,
                        available_index=np.repeat(np.arange(191,len(score),192),192))
    dump(output/"SCORES.json",dict(entity=entity,bank_sha256=sha(bank),query_points=len(test),
        scored=len(score), omitted_tail=len(test)-len(score), input_sha256=record["source"][1]["sha256"],
        mode="original nonoverlap full-window, offline", labels_read=False,calibrated=False,
        score_seconds=time.monotonic()-tic,score_peak_reserved=torch.cuda.max_memory_reserved(),
        seconds=time.monotonic()-start, **scoring_metadata))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action",choices=["entity","lane"])
    parser.add_argument("--entity")
    parser.add_argument("--arm", choices=["native", "observation"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--lane",type=int,choices=[0,1])
    args = parser.parse_args()
    w = Path.cwd()
    if args.action == "entity":
        if args.check and args.arm != "observation":
            parser.error("--check requires the observation arm")
        run_entity(w,args.entity,args.arm,check_only=args.check)
        return
    if args.arm is not None or args.check:
        parser.error("The historical lane command is only for the archived native runs")
    for entity in sorted(bindings(w))[args.lane::2]:
        out=w/RUN/entity
        if (out/"SCORES.json").exists():
            assert sha(out/"scores.npz")==read(out/"SCORES.json")["bank_sha256"]
            continue
        with (w/RUN/f"{entity}.log").open("x") as log:
            subprocess.run([sys.executable,"-u","-m",__spec__.name,"entity","--entity",entity],
                           check=True,stdout=log,stderr=subprocess.STDOUT,timeout=3600)


if __name__ == "__main__":
    main()
