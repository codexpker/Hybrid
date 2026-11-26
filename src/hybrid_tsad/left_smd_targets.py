"""Native 38-channel public normal transfer and an isolated SMD mixed-target protocol."""

import argparse
from functools import lru_cache
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from . import left_multisource as base
from . import left_smd28_author as smd
from .left_decoder_context_run import PUBLIC, Windows, environment
from .left_reliability import smooth, multiscale, set_weights

OUT = Path("results/left_smd_reliability_20261126_v1")
W = 192
STEPS = 2000


def roles(record):
    n, q = record["train_points"], record["query_points"]
    a, b = int(0.6 * n), int(0.8 * n)
    return dict(
        scaler_fit=[0, a],
        CAL=[b + 544, n],
        U=[0, q // 3],
        QUERY=[q // 3 + 544, q],
        TEST=[0, q],
        public_scaler=[0, n],
        public_train=[0, b],
        public_select=[b, n],
        mixed_records=dict(CAL="train", U="test", QUERY="test"),
    )


def normal_scaler(w, rec, public):
    x = smd.checked_data(w, rec, "train")
    bounds = roles(rec)["public_scaler" if public else "scaler_fit"]
    sc = StandardScaler().fit(x[slice(*bounds)])
    if public:
        p = w / PUBLIC / "SMD" / rec["id"] / "native"
        old = smd.read(p / "TRAIN.json")
        assert (
            old["train_sha256"] == rec["source"][0]["sha256"]
            and old["configs"] == smd.CONFIG
        )
        assert old["scaler"] == bounds and old["fit"] == roles(rec)["public_train"]
        assert old["validation"] == roles(rec)["public_select"] and old["seed"] == 2025
        assert smd.sha(p / "scaler.npz") == old["scaler_sha256"]
        with np.load(p / "scaler.npz") as z:
            for key in ("mean", "var", "scale"):
                np.testing.assert_allclose(
                    z[key], getattr(sc, key + "_"), rtol=2e-12, atol=1e-15
                )
                # The archived coefficients define the native public comparator.
                setattr(sc, key + "_", z[key].copy())
    return x, sc


def init(w, weighted=False):
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    torch.set_num_threads(1)
    official, _, _ = environment(w, "PSM")
    configs = w / smd.PREP / "configs" / "SMD"
    assert {k: smd.sha(configs / k) for k in smd.CONFIG} == smd.CONFIG
    cfg = official.Configs(str(configs / "model_configs_Left.json"))
    model = base.model_class(
        official.Left,
        "native",
        weighted_reductions=(smooth, multiscale) if weighted else None,
    )(cfg).cuda()
    assert model.seq_len == W and model.enc_in == 38
    return model


def arr_hash(x):
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def source_path(w):
    return w / OUT / "source"


def restore_path(model, dest):
    row = smd.read(dest / "receipt.json")
    assert smd.sha(dest / "selected.pt") == row["sha256"]
    ckpt = torch.load(dest / "selected.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(
        {
            k: v
            for k, v in ckpt["state"].items()
            if k.split(".")[-1] not in ("total_ops", "total_params")
        },
        strict=True,
    )
    model.step = ckpt["step"]
    return row["sha256"]


@torch.no_grad()
def reliability(model, data, starts):
    model.eval()
    step, state = model.step, base.fingerprint(model.state_dict())
    uniq, ix = np.unique(starts, return_inverse=True)
    scores = np.empty((len(uniq), W), dtype=np.float32)
    for i in range(0, len(uniq), 128):
        windows = data[uniq[i : i + 128, None] + np.arange(W)]
        scores[i : i + len(windows)] = (
            model.infer(torch.from_numpy(windows).cuda(), None, None, None)[0]
            .mean(-1)
            .cpu()
            .numpy()
        )
    assert model.step == step and state == base.fingerprint(model.state_dict())
    values = scores[ix.reshape(starts.shape)]
    r = (rankdata(values, method="average").reshape(values.shape) - 0.5) / values.size
    raw = 1.5 - r
    weights = (raw / raw.mean(axis=(1, 2), keepdims=True)).astype(np.float32)
    np.testing.assert_allclose(weights.mean((1, 2)), 1, atol=2e-7, rtol=0)
    return weights


@lru_cache(maxsize=1)
def fit_sources(w):
    from .left_smd_source import source_pools

    return source_pools(w)["fit"]


def adapt(w, rec, arm, dest):
    model = init(w, arm == "window")
    parent = None if arm == "random_u" else restore_path(model, source_path(w))
    initial = model.step
    r1, r2, rs = [
        np.random.default_rng(s) for s in np.random.SeedSequence(2025).spawn(3)
    ]
    extra = np.random.default_rng(np.random.SeedSequence([2025, 1126, 4]))
    shared = arm == "source_only"
    replay = arm in ("window", "replay", "source_only")
    sources = fit_sources(w) if replay else None
    weights = None
    if not shared:
        _, sc = normal_scaler(w, rec, False)
        bounds = roles(rec)["U"]
        x = sc.transform(smd.checked_data(w, rec, "test")[slice(*bounds)]).astype(
            np.float32
        )
        starts = r1.integers(len(x) - W + 1, size=(STEPS, 16))
        if arm == "window":
            # The frozen teacher uses unmodified native reductions. Isolate
            # its initialization from the student's training RNG trajectory.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                teacher = init(w)
                assert restore_path(teacher, source_path(w)) == parent
                weights = reliability(teacher, x, starts)
                del teacher
            np.save(dest / "target_window_weights.npy", weights.mean(2))
    curve = []
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for s in range(STEPS):
        first = (
            base.source_batch(sources, 38, 16, extra)
            if shared
            else x[starts[s, :, None] + np.arange(W)]
        )
        second = (
            base.source_batch(sources, 38, 16, rs)
            if replay
            else x[r2.integers(len(x) - W + 1, size=16)[:, None] + np.arange(W)]
        )
        if weights is not None:
            set_weights(model, torch.from_numpy(weights[s]).cuda(), False)
        curve.append(base.update(model, opt, np.concatenate([first, second])))
    base.save(model, dest)
    np.save(dest / "fit_loss.npy", curve)
    return dict(
        updates=STEPS,
        parent=parent,
        initial_step=initial,
        selected_step=model.step,
        target_exposure=0 if shared else STEPS * (16 if replay else 32),
        source_exposure=STEPS * (32 if shared else 16 if replay else 0),
        shared_target_plan=None if shared else arr_hash(starts),
        weights_sha256=None
        if weights is None
        else smd.sha(dest / "target_window_weights.npy"),
    )


def normal(w, rec, dest):
    """Same upstream trainer/window order as the archived scratch arm."""
    official, _, _ = environment(w, "PSM")
    model = init(w)
    parent = restore_path(model, source_path(w))
    initial = model.step
    data, sc = normal_scaler(w, rec, True)
    x, cut = sc.transform(data), roles(rec)["public_train"][1]
    curve, selected, sampler = [], [], []
    scratch = smd.read(w / PUBLIC / "SMD" / rec["id"] / "native" / "TRAIN.json")
    previous = official.EarlyStopping

    class Stop(previous):
        def save_checkpoint(self, value, m, path):
            super().save_checkpoint(value, m, path)
            selected.append((len(curve), m.step, float(value)))

    class Experiment(official.Exp_Anomaly_Detection):
        def _acquire_device(self):
            return torch.device("cuda:0")

        def _build_model(self):
            return model

        def _get_data(self, flag, step=None):
            assert flag in ("train", "val")
            train = flag == "train"
            ds = Windows(x[:cut] if train else x[cut:], 1 if train else W)
            return ds, DataLoader(
                ds, batch_size=128, shuffle=train, drop_last=False, num_workers=0
            )

        def vali(self, loader):
            val = float(super().vali(loader))
            assert np.isfinite(val)
            curve.append(val)
            sampler.append(arr_hash(torch.get_rng_state().numpy()))
            return val

    official.EarlyStopping = Stop
    args = SimpleNamespace(
        data="SMD",
        data_origin="DADA",
        model_class="Left",
        gpu=0,
        use_gpu=True,
        gpu_type="cuda",
        use_multi_gpu=False,
        configs_path=str(w / smd.PREP / "configs"),
        model_save_path=str(dest / "model"),
        results_save_path=str(dest / "unused"),
    )
    try:
        exp = Experiment(args, id=0)
        exp.train()
    finally:
        official.EarlyStopping = previous
    final = model.step
    epoch, model.step, value = selected[-1]
    assert value == min(curve)
    common = min(len(sampler), len(scratch["sampler_cpu_rng_sha256"]))
    assert sampler[:common] == scratch["sampler_cpu_rng_sha256"][:common]
    base.save(model, dest)
    return dict(
        parent=parent,
        initial_step=initial,
        updates=final - initial,
        epochs=len(curve),
        target_exposure=(cut - W + 1) * len(curve),
        selected=epoch,
        selected_step=model.step,
        loss_history=curve,
        sampler_cpu_rng_sha256=sampler,
        scratch_score_sha256=smd.read(
            w / PUBLIC / "SMD" / rec["id"] / "native" / "SCORES.json"
        )["bank_sha256"],
    )


def train(w, entity, arm):
    dest = w / OUT / entity / arm if arm != "source_only" else w / OUT / "source_only"
    dest.mkdir(exist_ok=True, parents=True)
    if (dest / "receipt.json").is_file():
        assert smd.read(dest / "receipt.json")["sha256"] == smd.sha(
            dest / "selected.pt"
        )
        return dest
    assert arm != "source"
    rec = None if arm == "source_only" else smd.bindings(w)[entity]
    tic = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    info = normal(w, rec, dest) if arm == "N" else adapt(w, rec, arm, dest)
    smd.dump(
        dest / "receipt.json",
        dict(
            entity="shared" if arm == "source_only" else entity,
            arm=arm,
            seed=2025,
            seconds=time.monotonic() - tic,
            sha256=smd.sha(dest / "selected.pt"),
            device=torch.cuda.get_device_name(),
            peak_bytes=torch.cuda.max_memory_reserved(),
            **info,
        ),
    )
    return dest


@torch.no_grad()
def bank(w, rec, arm, role):
    dest = w / OUT / rec["id"] / "banks"
    dest.mkdir(exist_ok=True, parents=True)
    path = dest / (arm + "_" + role + ".npz")
    receipt = path.with_suffix(".json")
    if receipt.is_file():
        assert smd.sha(path) == smd.read(receipt)["sha256"]
        return
    tic = time.monotonic()
    model = init(w).eval()
    ck = w / OUT / arm if arm in ("source", "source_only") else dest.parent / arm
    parent = restore_path(model, ck)
    step, state = model.step, base.fingerprint(model.state_dict())
    rr = roles(rec)
    train_values, sc = normal_scaler(w, rec, role == "TEST")
    values = train_values if role == "CAL" else smd.checked_data(w, rec, "test")
    lo, hi = rr[role]
    start, stop = (lo + W - 1) // W * W, hi // W * W
    windows = sc.transform(values[start:stop]).astype(np.float32).reshape(-1, W, 38)
    scores = []
    for i in range(0, len(windows), 128):
        b = torch.from_numpy(np.ascontiguousarray(windows[i : i + 128])).cuda()
        scores.append(
            model.infer(b, None, None, None)[0].mean(-1).cpu().numpy().ravel()
        )
    assert model.step == step and base.fingerprint(model.state_dict()) == state
    score = np.concatenate(scores)
    index = np.arange(start, stop)
    available = (index // W + 1) * W - 1
    if role == "CAL":
        pick = np.linspace(0, len(index) - 1, 512, dtype=np.int64)
        assert len(np.unique(pick)) == 512
        index, available, score = index[pick], available[pick], score[pick]
    assert (
        np.isfinite(score).all()
        and np.all((available >= index) & (available < hi))
        and np.all(available - W + 1 >= lo)
    )
    np.savez_compressed(path, index=index, available=available, score=score)
    smd.dump(
        receipt,
        dict(
            role=role,
            arm=arm,
            start=start,
            stop=stop,
            points=len(score),
            checkpoint_sha256=parent,
            sha256=smd.sha(path),
            seconds=time.monotonic() - tic,
            labels_read=False,
            model_step=step,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("entity")
    parser.add_argument(
        "arm",
        choices=(
            "source_only",
            "source",
            "N",
            "random_u",
            "ordinary",
            "replay",
            "window",
        ),
    )
    parser.add_argument("--shared-train", action="store_true")
    args = parser.parse_args()
    w, entity, arm = args.workspace, args.entity, args.arm
    torch.cuda.set_per_process_memory_fraction(
        20 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    if args.shared_train:
        assert arm == "source_only"
        train(w, None, arm)
        return
    rec = smd.bindings(w)[entity]
    dest = w / OUT / entity
    dest.mkdir(parents=True, exist_ok=True)
    rr = roles(rec)
    if (dest / "roles.json").exists():
        assert rr == smd.read(dest / "roles.json")
    else:
        smd.dump(dest / "roles.json", rr)
    if arm not in ("source", "source_only"):
        train(w, entity, arm)
    for role in (
        ("TEST", "CAL", "QUERY")
        if arm == "source"
        else ("TEST",)
        if arm == "N"
        else ("CAL", "QUERY")
    ):
        bank(w, rec, arm, role)


if __name__ == "__main__":
    main()
