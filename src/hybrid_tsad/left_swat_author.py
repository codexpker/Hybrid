"""Unmodified LEFT author train loop and non-overlapping SWaT raw-score readout.

Only data access is substituted: training loads the original numerical normal
prefix, and TEST values are read only after official early stopping/reload.
No test labels or author threshold-selection entry points are called.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def dump(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.workspace, args.output
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import Dataset, DataLoader
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(20 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    upstream = root / "preparations/left_baseline_build_20261101_v1/upstream"
    sys.path.insert(0, str(root / "preparations/left_baseline_build_20261101_v1/deps_thop_transport2_v1/site"))
    sys.path.insert(0, str(upstream))
    official = importlib.import_module("exp.exp_anomaly_detection")
    old = root / "preparations/left_multievent_mixed_u_20261111_v1"
    values_path = old / "data/ALL_VALUES.npy"
    values = np.load(values_path, mmap_mode="r", allow_pickle=False)
    assert values.shape == (944919, 31) and values.dtype == np.float64
    mapping = read(old / "OFFICIAL_31_CHANNEL_MAPPING.json")["mapping"]
    names = [entry["raw_name"] for entry in mapping if not entry["removed"]]
    assert len(names) == 31 and names == read(old / "INPUT_QUALIFICATION.json")["channels"]
    # Recover exactly the column-oriented author dataframe's numeric layout.
    normal = np.array(values[:495000], dtype=np.float64, order="F", copy=True)
    assert normal.flags.f_contiguous and np.isfinite(normal).all()
    checked_roles = []
    for role in read(old / "INPUT_QUALIFICATION.json")["roles"]:
        if role["end"] <= 495000:
            path = old / "data" / (role["role"] + ".npz")
            assert sha(path) == role["sha256"]
            with np.load(path) as z:
                assert np.array_equal(normal[role["original_start"]:role["end"]], z["values"])
            checked_roles.append(dict(role=role["role"], sha256=role["sha256"]))
    scaler = StandardScaler().fit(normal)
    normalized = scaler.transform(normal)
    np.savez_compressed(output / "SCALER.npz", mean=scaler.mean_, var=scaler.var_,
                        scale=scaler.scale_, n_samples_seen=scaler.n_samples_seen_)
    source = dict(source_csv_sha256="a8bad2c5f770dbd04e7ad44f58f565caf1c4fe1a68ab62d2d8625adb4d4cdceb",
         ALL_VALUES_sha256=sha(values_path), ALL_VALUES_shape=list(values.shape), dtype=str(values.dtype),
         channels=names, train=[0, 396000], SELECT=[396000, 495000], TEST=[495000, 944919],
         scaler_fit=[0, 495000], normal_prefix_layout="float64 Fortran: author column dataframe layout",
         checked_normal_role_sha256=checked_roles, labels_read=False, TEST_values_read_during_training=False,
         worker_sha256=sha(__file__), source_files={str(p.relative_to(upstream)):sha(p) for p in
            [upstream/"exp/exp_anomaly_detection.py", upstream/"data_provider/data_provider.py",
             upstream/"models/Left/LEFT.py", upstream/"models/Left/Tools.py",
             upstream/"models/Left/Utils.py", upstream/"models/Left/EncDec.py",
             upstream/"models/Left/Attention_Blocks.py"]},
         configs={str(p.relative_to(args.configs)):sha(p) for p in sorted(args.configs.rglob("*.json"))})
    dump(output / "SOURCE_BINDING.json", source)
    events = []
    selections = []
    updates = [0]

    class Windows(Dataset):
        def __init__(self, data, origin, stride):
            self.data, self.origin, self.stride = data, origin, stride
        def __len__(self):
            return (len(self.data) - 192) // self.stride + 1
        def __getitem__(self, index):
            a = index * self.stride
            return np.float32(self.data[a:a+192]), np.zeros((192, 1), np.float32)

    native_stopping = official.EarlyStopping
    class RecordingStopping(native_stopping):
        def save_checkpoint(self, loss, model, path):
            super().save_checkpoint(loss, model, path)
            selections.append(dict(epoch=len([e for e in events if e["kind"] == "SELECT"]),
                 SELECT_loss=float(loss), python_step=model.step, updates=updates[0],
                 checkpoint_sha256=sha(Path(path)/"checkpoint.pth")))
    official.EarlyStopping = RecordingStopping

    class Isolated(official.Exp_Anomaly_Detection):
        def _acquire_device(self):
            return torch.device("cuda:0")
        def _get_data(self, flag, step=None):
            if flag not in ("train", "val"):
                raise PermissionError("TEST forbidden in training loader")
            stride = 192 if step is None else step
            data, origin = (normalized[:396000], 0) if flag == "train" else (normalized[396000:], 396000)
            ds = Windows(data, origin, stride)
            events.append(dict(kind="loader", flag=flag, stride=stride, windows=len(ds),
                 first=[origin, origin+192], last=[origin+(len(ds)-1)*stride, origin+(len(ds)-1)*stride+192]))
            return ds, DataLoader(ds, batch_size=self.train_configs.batch_size,
                                  shuffle=flag == "train", drop_last=False, num_workers=0)
        def _select_optimizer(self):
            optimizer = super()._select_optimizer()
            def checked_step(opt, _args, _kwargs):
                updates[0] += 1
                grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
            optimizer.register_step_post_hook(checked_step)
            return optimizer
        def vali(self, loader):
            value = super().vali(loader)
            assert np.isfinite(value)
            events.append(dict(kind="SELECT", loss=float(value), python_step=self.model.step,
                               updates=updates[0]))
            dump(output / ("epoch%02d.json" % len([e for e in events if e["kind"] == "SELECT"])), events[-1])
            return value
        def test(self, **kwargs):
            raise PermissionError("Only fixed raw-score stage allowed")
        def evaluate_spot(self, **kwargs):
            raise PermissionError("No label-dependent author threshold evaluation")

    exp = Isolated(SimpleNamespace(data="SWAT", data_origin="DADA", model_class="Left", gpu=0,
        use_gpu=True, gpu_type="cuda", use_multi_gpu=False, configs_path=str(args.configs),
        model_save_path=str(output/"model"), results_save_path=str(output/"unused_results")), id=0)
    expected = dict(train_epochs=20, batch_size=128, optim="adam", learning_rate=.0001,
                    lradj="type1", patience=3)
    assert read(args.configs/"SWAT/train_configs.json") == expected
    train_start = time.monotonic()
    try:
        exp.train()
    finally:
        dump(output/"TRAIN_EVENTS.json", dict(events=events, selected_checkpoints=selections))
    pt = Path(exp.model_save_path)/"checkpoint.pth"
    assert selections[-1]["checkpoint_sha256"] == sha(pt)
    final_step = exp.model.step
    train_seconds = time.monotonic()-train_start
    dump(output/"TRAIN_RECEIPT.json", dict(status="COMPLETED", checkpoint_sha256=sha(pt),
         updates=updates[0], selected=selections[-1], final_python_step=final_step,
         train_seconds=train_seconds, allocated=torch.cuda.max_memory_allocated(),
         reserved=torch.cuda.max_memory_reserved(), labels_read=False, TEST_read=False))
    print(json.dumps(dict(event="TRAIN_COMPLETED", updates=updates[0], selected=selections[-1],
                          train_seconds=train_seconds)), flush=True)
    # This is the first actual TEST value read. No labels are opened.
    test = scaler.transform(np.array(values[495000:], dtype=np.float64, order="F", copy=True))
    windows = Windows(test, 495000, 192)
    assert len(windows) == 2343
    loader = DataLoader(windows, batch_size=128, shuffle=False, drop_last=False, num_workers=0)
    loaded = exp.model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True), strict=False)
    assert not loaded.missing_keys and all(key.endswith(("total_ops", "total_params"))
                                           for key in loaded.unexpected_keys)
    exp.model.eval()
    before = {k:v.detach().cpu().clone() for k,v in exp.model.state_dict().items()}
    scores = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for batch, _ in loader:
            raw = exp.model.infer(batch.float().cuda(), None, None, None)[0]
            assert torch.isfinite(raw).all()
            scores.append(raw.cpu().numpy())
    per_channel = np.concatenate(scores).reshape(-1, 31)
    canonical = np.mean(per_channel, axis=-1)
    assert canonical.dtype == np.float32 and len(canonical) == 449856
    assert exp.model.step == final_step and all(torch.equal(before[k], v.cpu()) for k,v in exp.model.state_dict().items())
    bank = output/"raw_scores.npz"
    np.savez_compressed(bank, canonical_score=canonical, per_channel=per_channel,
                        absolute_time_index=np.arange(495000, 944856),
                        available_index=np.repeat(np.arange(495000+191, 944856, 192), 192),
                        valid=np.ones(449856, bool))
    dump(output/"RAW_SCORE_RECEIPT.json", dict(status="COMPLETED", bank_sha256=sha(bank),
         checkpoint_sha256=sha(pt), labels_read=False, threshold_fit=False, mode="native offline full window",
         score="original float32 channel mean", windows=2343, seq_len=192, stride=192,
         scored=[495000, 944856], input=[495000, 944919], omitted_tail=63,
         final_python_step=final_step, model_state_and_step_unchanged=True,
         score_allocated=torch.cuda.max_memory_allocated(), score_reserved=torch.cuda.max_memory_reserved(),
         total_worker_seconds=time.monotonic()-start,
         UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(event="RAW_SCORE_COMPLETED", bank_sha256=sha(bank))), flush=True)


if __name__ == "__main__":
    main()
