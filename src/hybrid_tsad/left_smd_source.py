"""SMD-held-out LEFT pretraining from two normal public domains."""

import argparse
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from . import left_multisource as base
from .left_smd28_author import PREP, CONFIG
from .trajectory_experiment import TEP, checked

OUT = Path("results/left_smd_reliability_20261126_v1")
HAI = Path("data/hai_2103_official")
REVISION = "2a814cebc9a66b06c9e5cd545e2d72e65d383737"
BLOBS = (
    "2940eeeb2018c68fc01586da77986a5771b12585",
    "4810e976c30ff9a15c2cadb69319bd7e9fe922e7",
    "207e4b9f4c157505d5cb4a66e8ebe84d03575dab",
)
LABELS = ["attack", "attack_P1", "attack_P2", "attack_P3"]
TEP_DIGEST = "2c4e24dc41b14a7a521de71a8bbac87cef3788fcf46c9dd321aab5a14f38c698"


def git_blob(path):
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as file:
        for buf in iter(lambda: file.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def prepare(w):
    dest = w / OUT / "inputs"
    dest.mkdir(exist_ok=True, parents=True)
    shared = StandardScaler()
    schema = None
    records = {}
    for i in (1, 3, 2):
        name = f"train{i}"
        path = w / HAI / (name + ".csv.gz")
        blob = git_blob(path)
        assert blob == BLOBS[i - 1], (path.name, blob)
        frame = pd.read_csv(path)
        assert not frame.isna().any().any()
        dates = pd.to_datetime(frame.pop("time"))
        assert np.all(np.diff(dates.to_numpy(dtype="datetime64[s]")).astype(int) == 1)
        assert (frame[LABELS] == 0).all().all()
        frame.drop(columns=LABELS, inplace=True)
        assert len(frame.columns) == 79 and frame.columns.is_unique
        if schema is None:
            schema = list(frame.columns)
        assert list(frame.columns) == schema
        x = frame.to_numpy(dtype=np.float64)
        assert np.isfinite(x).all()
        if i in (1, 3):
            scaler = StandardScaler().fit(x)
            shared.partial_fit(x)
            fit_id = name
        else:
            scaler = shared
            fit_id = "train1+train3 (point pooled)"
        np.savez(dest / (name + "_scaler.npz"), mean=scaler.mean_, scale=scaler.scale_)
        normalized = dest / (name + ".npy")
        np.save(normalized, scaler.transform(x).astype(np.float32))
        records[name] = dict(
            rows=len(x),
            columns=len(schema),
            first=str(dates.iloc[0]),
            last=str(dates.iloc[-1]),
            role="SELECT" if i == 2 else "FIT",
            blob=blob,
            sha256=base.sha(path),
            array_sha256=base.sha(normalized),
            scaler_fit=fit_id,
            scaler_sha256=base.sha(dest / (name + "_scaler.npz")),
            missing=0,
            nonsecond_intervals=0,
            nonzero_label_entries=0,
        )
    base.dump(
        dest / "receipt.json",
        dict(
            revision=REVISION,
            records=records,
            features=schema,
            excluded_columns=["time", *LABELS],
        ),
    )


def source_pools(w):
    root = w / OUT / "inputs"
    meta = base.read(root / "receipt.json")

    def hai(i):
        path = root / f"train{i}.npy"
        return checked(path, meta["records"][f"train{i}"]["array_sha256"])

    # Previously archived 52-column TEP normal array is already standardized
    # with the pooled FIT runs 1..20. SELECT runs 61..65 use that same scaler.
    tep = checked(w / TEP, TEP_DIGEST)[0]
    assert tep.shape == (25, 500, 52)
    return {
        "fit": [list(tep[:20]), [hai(1), hai(3)]],
        "select": [list(tep[20:]), [hai(2)]],
    }


def initialize(w, weighted_reductions=None):
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(
        20 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    official, _, _ = base.environment(w, "PSM")
    cfgs = w / PREP / "configs" / "SMD"
    assert {key: base.sha(cfgs / key) for key in CONFIG} == CONFIG
    cfg = official.Configs(str(cfgs / "model_configs_Left.json"))
    model = base.model_class(
        official.Left, "native", weighted_reductions=weighted_reductions
    )(cfg).cuda()
    assert model.enc_in == 38 and model.seq_len == 192
    return model


def restore(w, model):
    folder = w / OUT / "source"
    meta = base.read(folder / "receipt.json")
    assert base.sha(folder / "selected.pt") == meta["sha256"]
    checkpoint = torch.load(
        folder / "selected.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint["state"], strict=True)
    model.step = checkpoint["step"]
    assert (
        model.step == meta["selected"]
        and base.fingerprint(model.state_dict()) == meta["fingerprint"]
    )
    return meta


def train(w):
    dest = w / OUT / "source"
    if (dest / "receipt.json").exists():
        restore(w, initialize(w))
        return
    dest.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    result = base.train_source(
        initialize(w), source_pools(w), 38, dest, "SMD", ("tep", "hai")
    )
    base.dump(
        dest / "receipt.json",
        dict(
            **result,
            seed=2025,
            window=192,
            width=38,
            sha256=base.sha(dest / "selected.pt"),
            seconds=time.monotonic() - start,
            device=torch.cuda.get_device_name(),
            peak_bytes=torch.cuda.max_memory_allocated(),
            fit_domains=["TEP FaultFree Training", "HAI 21.03"],
            excluded_domain="SMD",
            hai_receipt_sha256=base.sha(w / OUT / "inputs/receipt.json"),
            tep_array_sha256=TEP_DIGEST,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("stage", choices=("prepare", "train"))
    args = p.parse_args()
    (prepare if args.stage == "prepare" else train)(args.workspace)
