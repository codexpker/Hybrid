"""Thin external LEFT adapter. Upstream model/source/data remain outside this package.

Preview uses stdlib only. Execution requires a separately approved config.
R-only score requires separate query permission and never reads labels.
"""
import argparse
import ast
import hashlib
import importlib
import json
import random
import signal
import struct
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


COMMIT = "3fadb4811797075e233076b9faceb0d8ec59b697"
PSM_SHA = "38e446209e82df94b82c11aa43d87e3671da86167ed794c1b7be5eb91fc0e8ac"
R_CHECKPOINT_SHA = "e07a50bde4a470b79590b6392f4cb883f715e6205a78c308293f522f5656d64c"
R_SCALER_SHA = "41da46ca635f3053d8970f5fe1eb0565117a0f69be92590cac2092d9bb50bfb3"
FEATURES = ["feature_%d" % i for i in range(25)]
MODEL_FILES = ["models/Left/%s.py" % n for n in
               ["LEFT", "Tools", "Utils", "EncDec", "Attention_Blocks"]]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(path):
    """Identity/shape preparation only; do not import torch or load arrays."""
    cfg = read_json(path)
    if cfg.get("operation") not in ("profile", "train", "score"):
        raise ValueError("Only profile/train/R-score operations supported; no evaluation")
    if cfg.get("track") != "R":
        raise ValueError("C track has not been frozen")
    if cfg.get("upstream_commit") != COMMIT:
        raise ValueError("Unexpected official revision")
    if cfg.get("seed") != 2025:
        raise ValueError("First development seed is fixed at 2025")
    root = Path(cfg["source_root"]).resolve()
    hashes = cfg["source_hashes"]
    required = MODEL_FILES + ["configs/PSM/model_configs_Left.json"]
    if cfg["operation"] == "train":
        required += ["exp/exp_anomaly_detection.py"] + [
            "models/CrossAD/%s.py" % n for n in
            ["Basic_CrossAD", "Attention_Blocks", "EncDec", "Context_Blocks"]]
        required += ["configs/PSM/model_configs_0.json", "configs/PSM/train_configs.json"]
    for name in required:
        if name not in hashes or file_sha(root / name) != hashes[name]:
            raise ValueError("Official source mismatch: " + name)
    model_cfg = read_json(root / "configs/PSM/model_configs_Left.json")
    if (model_cfg["seq_len"], model_cfg["enc_in"]) != (192, 25):
        raise ValueError("Expected unchanged PSM model configuration")
    output = Path(cfg["output"]).resolve()
    if output.exists():
        raise FileExistsError(output)
    if cfg["operation"] == "profile":
        if cfg.get("batch_size") != 128 or cfg.get("steps") != 3:
            raise ValueError("Profile requires exactly three B128 steps")
        if cfg.get("data") is not None:
            raise ValueError("Synthetic profile cannot reference real data")
        if cfg.get("max_seconds") != 180 or cfg.get("max_allocated_GiB") != 20:
            raise ValueError("Profile resource contract mismatch")
    elif cfg["operation"] == "train":
        for key in ("array", "metadata", "array_sha256", "metadata_sha256"):
            if not cfg.get("data", {}).get(key):
                raise ValueError("Train prefix requires exact identity: " + key)
        if cfg.get("max_allocated_GiB") != 20:
            raise ValueError("Training memory contract mismatch")
    else:
        validate_score_metadata(cfg)
    return cfg, root, output


def require_approval(config_path, cfg):
    approval = read_json(cfg["approval_path"])
    expected = "PRIMARY_RELEASED_LEFT_BASELINE_" + cfg["operation"].upper()
    if approval.get("status") != expected or approval.get("config_sha256") != file_sha(config_path):
        raise PermissionError("Operation is not released for this exact config")
    if approval.get("adapter_sha256") != file_sha(__file__):
        raise PermissionError("Adapter bytes are not released")
    if approval.get("allow_query") is not (cfg["operation"] == "score") or approval.get("allow_labels") is not False:
        raise PermissionError("Separate exact query permission required; labels always forbidden")
    if approval.get("max_seconds") != cfg["max_seconds"]:
        raise PermissionError("Timeout contract mismatch")
    return approval


def npy_header_only(path):
    """Read only the NPY header, not a single query data value."""
    with Path(path).open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            raise ValueError("Not an NPY array")
        version = tuple(f.read(2))
        if version == (1, 0):
            length = struct.unpack("<H", f.read(2))[0]
        elif version in ((2, 0), (3, 0)):
            length = struct.unpack("<I", f.read(4))[0]
        else:
            raise ValueError("Unsupported NPY header version")
        if length > 65536:
            raise ValueError("Oversized NPY header")
        header = ast.literal_eval(f.read(length).decode("utf-8" if version == (3, 0) else "latin1"))
        header["data_offset"] = f.tell()
    return header


def query_metadata(query):
    if set(query) != {"array", "metadata", "array_sha256", "metadata_sha256"}:
        raise ValueError("Only query feature array and metadata references accepted; no label path")
    if Path(query["array"]).name != "test.npy":
        raise ValueError("Only the raw test feature array is accepted")
    if file_sha(query["metadata"]) != query["metadata_sha256"]:
        raise ValueError("Query metadata identity mismatch")
    meta = read_json(query["metadata"])
    if (meta.get("source_sha256") != PSM_SHA or meta.get("feature_names") != FEATURES or
            meta.get("raw_values_not_normalized") is not True or meta.get("test_shape") != [87841, 25] or
            meta.get("array_sha256", {}).get("test.npy") != query["array_sha256"]):
        raise ValueError("Raw query source/schema mismatch")
    header = npy_header_only(query["array"])
    if header.get("shape") != (87841, 25) or header.get("descr") != "<f8":
        raise ValueError("Expected float64 query feature array [87841,25]")
    if Path(query["array"]).stat().st_size != header["data_offset"] + 87841*25*8:
        raise ValueError("NPY file length mismatch")
    return meta, header  # array bytes/hash are not read here; only on released score execution


def validate_score_metadata(cfg):
    if cfg.get("batch_size") != 128 or cfg.get("max_seconds") != 180 or cfg.get("max_allocated_GiB") != 30:
        raise ValueError("R score batch/resource contract mismatch")
    for name, fixed in [("checkpoint", R_CHECKPOINT_SHA), ("scaler", R_SCALER_SHA)]:
        ref = cfg.get(name, {})
        if ref.get("sha256") != fixed or file_sha(ref["path"]) != fixed:
            raise ValueError("Frozen R identity mismatch: " + name)
    acceptance = cfg["training_acceptance"]
    if file_sha(acceptance["path"]) != acceptance["sha256"]:
        raise ValueError("Training acceptance identity mismatch")
    accepted = read_json(acceptance["path"])
    if (accepted.get("status") != "PRIMARY_ACCEPTED_R_TRAIN_AND_FROZEN_CHECKPOINT" or
            accepted["checkpoint"]["sha256"] != R_CHECKPOINT_SHA or accepted["scaler"]["sha256"] != R_SCALER_SHA):
        raise ValueError("Training/checkpoint not accepted")
    keys = cfg.get("expected_thop_keys", [])
    if len(keys) != 86 or len(set(keys)) != 86 or not all(k.endswith(("total_ops", "total_params")) for k in keys):
        raise ValueError("Exact 86 expected thop statistic keys must be declared")
    query_metadata(cfg["query"])


def transform_query_with_saved_scaler(values, scaler_record):
    """Restore author R statistics, never fit on query. Preserve float64 F layout."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    if values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != 25 or not np.isfinite(values).all():
        raise ValueError("Invalid query feature values; no implicit imputation")
    if scaler_record.get("track") != "R" or scaler_record.get("fit_interval") != [0, 132481]:
        raise ValueError("Unexpected scaler fit role")
    stats = scaler_record["statistics"]
    scaler = StandardScaler()
    for source, target in [("mean", "mean_"), ("var", "var_"), ("scale", "scale_")]:
        a = np.asarray(stats[source], dtype=np.float64)
        if a.shape != (25,) or not np.isfinite(a).all():
            raise ValueError("Invalid scaler statistics")
        setattr(scaler, target, a)
    if (scaler.scale_ <= 0).any() or (scaler.var_ < 0).any() or stats["n_samples_seen"] != 132481:
        raise ValueError("Invalid scale or sample count")
    scaler.n_samples_seen_ = stats["n_samples_seen"]
    scaler.n_features_in_ = 25
    return scaler.transform(np.asfortranarray(values))


class FixedQueryWindows:
    def __init__(self, normalized):
        self.values = normalized

    def __len__(self):
        return (len(self.values)-192)//192 + 1

    def __getitem__(self, index):
        import numpy as np
        if not 0 <= index < len(self):
            raise IndexError(index)
        return np.float32(self.values[index*192:(index+1)*192])


def author_numpy_scores(chunks):
    """Exact Exp.test concatenate -> reshape -> np.array -> channel mean."""
    import numpy as np
    if not chunks or any(a.dtype != np.float32 or a.ndim != 3 or a.shape[1:] != (192, 25) or
                         not np.isfinite(a).all() for a in chunks):
        raise ValueError("Invalid raw per-channel score chunks")
    energy = np.concatenate(chunks, axis=0)
    flat = energy.reshape(-1, energy.shape[-1])
    flat = np.array(flat)  # retain the author's explicit copy/order behavior
    canonical = np.mean(flat, axis=-1)  # retain NumPy float32 reduction; no torch.mean replacement
    return energy, canonical


def query_coordinates(total=87841, window=192, batch=128, base=132481):
    import numpy as np
    if (total, window, batch, base) != (87841, 192, 128, 132481):
        raise ValueError("Only the frozen R coverage/availability protocol supported")
    count = total//window
    starts = base + np.arange(count, dtype=np.int64)*window
    ends = starts+window
    batch_ends = base + np.minimum((np.arange(count)//batch+1)*batch, count)*window - 1
    mask = np.arange(total) < count*window
    return {"absolute_time_index": np.arange(base, base+count*window, dtype=np.int64),
            "window_start": starts, "window_end_exclusive": ends,
            "window_input_end": ends-1, "window_batch_end": batch_ends,
            "point_input_end": np.repeat(ends-1, window),
            "point_batch_end": np.repeat(batch_ends, window),
            "point_window_id": np.repeat(np.arange(count, dtype=np.int64), window),
            "channel_index": np.arange(25, dtype=np.int64), "scored_mask": mask}


def load_expected_checkpoint(model, state, expected):
    if len(expected) != 86 or len(set(expected)) != 86 or not all(k.endswith(("total_ops", "total_params")) for k in expected):
        raise ValueError("Expected thop key identity invalid")
    unexpected = set(state)-set(model.state_dict())
    if unexpected != set(expected):
        raise ValueError("Unexpected checkpoint keys differ from exact thop list")
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or set(result.unexpected_keys) != set(expected):
        raise ValueError("Missing scientific state or unexpected non-thop keys")
    return {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


def load_prefix(array_path, meta_path):
    """Load a hash-pinned raw train prefix, never a combined/query array."""
    import numpy as np
    p = Path(array_path)
    if p.name not in ("prefix.npy", "train.npy"):
        raise ValueError("Only explicit raw normal-prefix filenames accepted")
    meta = read_json(meta_path)
    if meta.get("source_sha256") != PSM_SHA or meta.get("feature_names") != FEATURES:
        raise ValueError("Source SHA/column order mismatch")
    if meta.get("raw_values_not_normalized") is not True:
        raise ValueError("Already-normalized or unverified cache refused")
    expected = meta.get("array_sha256", {}).get(p.name)
    if not expected or file_sha(p) != expected:
        raise ValueError("Raw prefix array SHA mismatch")
    array = np.load(p, allow_pickle=False)
    if array.shape != (132481, 25) or array.dtype != np.float64 or not np.isfinite(array).all():
        raise ValueError("Prefix shape or finite-value contract failed; no implicit imputation")
    return array, meta


def prepare_prefix(array_path, meta_path, output):
    """Independent derivative with source lineage. No scaler fit or model call."""
    import numpy as np
    array, meta = load_prefix(array_path, meta_path)
    out = Path(output)
    out.mkdir(exist_ok=False)
    np.save(out / "prefix.npy", array, allow_pickle=False)
    result = {"source_sha256": PSM_SHA, "feature_names": FEATURES,
              "prefix_shape": list(array.shape), "raw_values_not_normalized": True,
              "array_sha256": {"prefix.npy": file_sha(out / "prefix.npy")},
              "parent_array": {"path": str(Path(array_path).resolve()), "sha256": file_sha(array_path)},
              "parent_metadata": {"path": str(Path(meta_path).resolve()), "sha256": file_sha(meta_path)},
              "query_or_labels_read": False, "scaler_fitted": False, "dtype": str(array.dtype),
              "finite_values": True, "imputation_in_this_adapter": False,
              "original_csv_value_equivalence": "UNVERIFIED; source lineage metadata only"}
    (out / "meta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


class NormalPrefixDataset:
    """Author R scaler/splits and float32 windows; dummy labels, no query branch."""
    def __init__(self, prefix, flag, win_size=192, step=192):
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        if flag not in ("train", "val", "init"):
            raise PermissionError("No query/test/labels loader in training adapter")
        if prefix.ndim != 2 or len(prefix) < win_size or not np.isfinite(prefix).all():
            raise ValueError("Invalid prefix")
        if prefix.dtype != np.float64:
            raise ValueError("R preprocessing must not cast the raw prefix to float32")
        # Official read_data builds a dict-of-columns pandas frame, then drops
        # date/label and converts to_numpy: float64 Fortran layout before fit.
        # Preserve this reduction layout rather than fitting the C-order cache.
        parsed = np.asfortranarray(prefix)
        self.scaler = StandardScaler().fit(parsed)
        full = self.scaler.transform(parsed)
        end = int(len(prefix) * .8)
        self.offset = 0 if flag in ("train", "init") else end
        self.values = full[:end] if flag == "train" else full[end:] if flag == "val" else full
        self.win_size, self.step, self.flag = win_size, step, flag

    def __len__(self):
        return max(0, (len(self.values) - self.win_size) // self.step + 1)

    def footprint(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = self.offset + index * self.step
        return start, start + self.win_size

    def __getitem__(self, index):
        import numpy as np
        self.footprint(index)
        start = index * self.step
        return (np.float32(self.values[start:start+self.win_size]),
                np.zeros((self.win_size, 1), dtype=np.float32))


def import_official(root, name):
    sys.path.insert(0, str(root))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != (root / (name.replace(".", "/") + ".py")).resolve():
        raise RuntimeError("Wrong external module imported: " + name)
    return module


def model_runtime(cfg, root):
    import numpy as np
    import torch
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.cuda.set_device(0)
    cap = cfg["max_allocated_GiB"] * 1024**3
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., cap/total), device=0)
    torch.cuda.reset_peak_memory_stats(0)  # includes subsequent model construction
    module = import_official(root, "models.Left.LEFT")
    loaded = {}
    for rel in MODEL_FILES:
        mod = sys.modules[rel[:-3].replace("/", ".")]
        actual = Path(mod.__file__).resolve()
        if actual != (root / rel).resolve() or file_sha(actual) != cfg["source_hashes"][rel]:
            raise RuntimeError("Loaded official identity mismatch")
        loaded[rel] = {"path": str(actual), "sha256": file_sha(actual)}
    return torch, module.Left, loaded


def profile(cfg, root, report):
    torch, Left, loaded = model_runtime(cfg, root)
    report["loaded_modules"] = loaded
    model = Left(SimpleNamespace(**read_json(root/"configs/PSM/model_configs_Left.json"))).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    report["parameter_count"] = sum(p.numel() for p in model.parameters())
    report["steps"] = []
    for i in range(3):
        x = torch.randn(128, 192, 25, device="cuda:0")
        torch.cuda.synchronize()
        t = time.monotonic()
        opt.zero_grad()
        loss, _ = model(x, None, None, None)
        assert loss.ndim == 0 and torch.isfinite(loss)
        torch.cuda.synchronize()
        fwd = time.monotonic()-t
        loss.backward()
        grad = [p.grad for p in model.parameters() if p.grad is not None]
        assert grad and all(torch.isfinite(g).all() for g in grad)
        assert any(bool((g != 0).any()) for g in grad)
        torch.cuda.synchronize()
        backward = time.monotonic()-t-fwd
        opt.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic()-t
        report["steps"].append({"index": i, "batch": 128, "loss": float(loss.detach()),
                                 "forward_seconds": fwd, "backward_and_grad_checks_seconds": backward,
                                 "whole_update_seconds": elapsed, "self_step": model.step})
        print(json.dumps(report["steps"][-1]), flush=True)
        if torch.cuda.max_memory_allocated() > 20*1024**3:
            raise MemoryError("20GiB allocated limit exceeded")
    report["coverage"] = {"warmup1000": "INCOMPLETE", "good_any_EMA": "INCOMPLETE"}
    report["checkpoint_saved"] = False
    return torch


def train(cfg, root, output, report):
    """Official train loop with the sole data-permission substitution and logging."""
    for kind in ("array", "metadata"):
        if file_sha(cfg["data"][kind]) != cfg["data"][kind + "_sha256"]:
            raise ValueError("Released train input mismatch: " + kind)
    prefix, meta = load_prefix(cfg["data"]["array"], cfg["data"]["metadata"])
    torch, _, loaded = model_runtime(cfg, root)
    official = import_official(root, "exp.exp_anomaly_detection")
    from torch.utils.data import DataLoader
    report["loaded_modules"] = loaded
    report["data_identity"] = {"array_sha256": file_sha(cfg["data"]["array"]),
                               "metadata_sha256": file_sha(cfg["data"]["metadata"]), "source_sha256": PSM_SHA}
    events = []

    class TrainOnly(official.Exp_Anomaly_Detection):
        def _basic(self):
            def snapshot():
                return {"torch_rng_sha256": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                        "cuda_rng_sha256": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                        "python_model_step": self.model.step,
                        "modes": {k: m.training for k, m in self.model.named_modules()},
                        "state_sha256": {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                         for k, v in self.model.state_dict().items()}}
            before = snapshot()
            try:
                return super()._basic()  # preserve dummy RNG/thop behavior; never silently skip
            finally:
                events.append({"kind": "official_basic", "before": before, "after": snapshot()})

        def _get_data(self, flag, step=None):
            ds = NormalPrefixDataset(prefix, flag, self.model_configs.seq_len,
                                     self.model_configs.seq_len if step is None else step)
            events.append({"kind": "loader", "flag": flag, "windows": len(ds),
                           "first": ds.footprint(0), "last": ds.footprint(len(ds)-1),
                           "percentage": 1., "shuffle": flag == "train", "workers": 0,
                           "drop_last": False, "raw_dtype": str(prefix.dtype),
                           "scaler_input_layout": "float64 Fortran, author dict-dataframe layout",
                           "scaler_fit_interval": [0, len(prefix)],
                           "scaler": {"mean": ds.scaler.mean_.tolist(), "var": ds.scaler.var_.tolist(),
                                      "scale": ds.scaler.scale_.tolist(),
                                      "n_samples_seen": int(ds.scaler.n_samples_seen_)}})
            return ds, DataLoader(ds, batch_size=self.train_configs.batch_size,
                                  shuffle=flag == "train", drop_last=False)

        def _select_optimizer(self):
            opt = super()._select_optimizer()
            updates = [0]
            def log_step(optimizer, _args, _kwargs):
                updates[0] += 1
                if updates[0] == 1 or updates[0] % 100 == 0:
                    grads = [p.grad for p in self.model.parameters() if p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads):
                        raise FloatingPointError("Nonfinite gradient; stop without retry")
                    events.append({"kind": "optimizer_step", "optimizer_step": updates[0],
                                   "python_model_step": self.model.step,
                                   "lr": [g["lr"] for g in optimizer.param_groups],
                                   "gradient_l2": float(torch.stack([g.detach().float().norm() for g in grads]).norm())})
            opt.register_step_post_hook(log_step)
            return opt

        def vali(self, loader):
            value = super().vali(loader)
            events.append({"kind": "normal_validation", "loss": float(value), "model_step": self.model.step})
            return value

        def test(self, **kwargs):
            raise PermissionError("Query/test not authorized by train operation")

        def evaluate_spot(self, **kwargs):
            raise PermissionError("Label/score evaluation disabled")

    args = SimpleNamespace(data="PSM", data_origin="DADA", model_class="Left", gpu=0, use_gpu=True,
                           gpu_type="cuda", use_multi_gpu=False, configs_path=str(root/"configs"),
                           model_save_path=str(output/"model"), results_save_path=str(output/"unused_results"))
    exp = TrainOnly(args, id=0)
    original_train = read_json(root/"configs/PSM/train_configs.json")
    assert original_train == {"train_epochs": 20, "batch_size": 128, "optim": "adam",
                              "learning_rate": 1e-4, "lradj": "type2", "patience": 3}
    try:
        exp.train()  # unchanged _basic/thop, train/vali/lr/early stop; no run.py auto-test
    finally:
        (output/"normal_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    report["final_python_step"] = exp.model.step
    report["validation_events"] = events
    report["checkpoint_policy"] = "official best state_dict; no query access; step/optimizer not in official checkpoint"
    return torch


def score_state(model):
    return {"step": model.step, "modes": {k: m.training for k, m in model.named_modules()},
            "state_sha256": {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                             for k, v in model.state_dict().items()}}


def require_frozen_score_state(before, after):
    if before != after or any(after["modes"].values()):
        raise RuntimeError("Scoring changed model parameters/buffers/step/modes")


def score(cfg, root, output, report):
    """One separately released R readout; original infer and NumPy mean, no labels."""
    import numpy as np
    # Metadata/header were checked in preview; actual query bytes only after release.
    query = cfg["query"]
    meta, header = query_metadata(query)
    if file_sha(query["array"]) != query["array_sha256"]:
        raise ValueError("Query array identity mismatch")
    raw = np.load(query["array"], allow_pickle=False)
    report["query_values_read"] = True
    normalized = transform_query_with_saved_scaler(raw, read_json(cfg["scaler"]["path"]))
    windows = FixedQueryWindows(normalized)
    assert len(windows) == 457
    torch, Left, loaded = model_runtime(cfg, root)
    from torch.utils.data import DataLoader
    model = Left(SimpleNamespace(**read_json(root/"configs/PSM/model_configs_Left.json"))).cuda()
    state = torch.load(cfg["checkpoint"]["path"], map_location="cpu", weights_only=True)
    report["checkpoint_load"] = load_expected_checkpoint(model, state, cfg["expected_thop_keys"])
    model.eval()
    before = score_state(model)
    assert not any(before["modes"].values())
    loader = DataLoader(windows, batch_size=128, shuffle=False, drop_last=False, num_workers=0)
    report["loaded_modules"] = loaded
    report["batches"] = []
    report["rng_before_loader"] = {"torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                                   "cuda": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
    chunks = []
    with torch.no_grad():
        for batch_id, values in enumerate(loader):
            torch.cuda.synchronize()
            t = time.monotonic()
            tensor = values.float().to("cuda:0")
            channel_scores, _ = model.infer(tensor, None, None, None)
            chunk = channel_scores.detach().cpu().numpy()
            torch.cuda.synchronize()
            if chunk.shape != tuple(values.shape) or chunk.dtype != np.float32 or not np.isfinite(chunk).all():
                raise ValueError("Invalid official infer scores")
            chunks.append(chunk)
            report["batches"].append({"id": batch_id, "shape": list(chunk.shape),
                                       "inference_and_cpu_copy_seconds": time.monotonic()-t,
                                       "numpy_strides": list(chunk.strides),
                                       "numpy_C": bool(chunk.flags.c_contiguous),
                                       "numpy_F": bool(chunk.flags.f_contiguous)})
    after = score_state(model)
    report["rng_after_loader"] = {"torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                                  "cuda": hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()}
    require_frozen_score_state(before, after)
    energy, canonical = author_numpy_scores(chunks)
    if energy.shape != (457, 192, 25) or canonical.shape != (87744,):
        raise ValueError("Unexpected full R score coverage")
    coordinates = query_coordinates()
    np.savez_compressed(output/"raw_scores.npz", channel_scores=energy, canonical_score=canonical,
                        channel_names=np.asarray(FEATURES), **coordinates)
    # Verify serializing raw arrays did not alter the canonical NumPy reduction.
    with np.load(output/"raw_scores.npz", allow_pickle=False) as stored:
        recomputed = np.mean(np.array(stored["channel_scores"].reshape(-1, 25)), axis=-1)
        if not np.array_equal(recomputed, stored["canonical_score"]):
            raise ValueError("Saved raw/canonical NumPy reduction mismatch; no replacement")
    if file_sha(cfg["checkpoint"]["path"]) != cfg["checkpoint"]["sha256"]:
        raise RuntimeError("Checkpoint bytes changed")
    report.update(state_before=before, state_after=after, query_header=header,
                  checkpoint_sha256=cfg["checkpoint"]["sha256"], scaler_sha256=cfg["scaler"]["sha256"],
                  query_sha256=query["array_sha256"], raw_shape=list(energy.shape),
                  canonical_shape=list(canonical.shape), canonical_dtype=str(canonical.dtype),
                  query_interval=[132481, 220322], scored_interval=[132481, 220225],
                  unscored_interval=[220225, 220322], unscored_points=97,
                  earliest_first_window_input_end=132672, first_batch_input_end=157056,
                  score_formula="original infer; NumPy concatenate->reshape(-1,25)->np.array->mean(axis=-1), float32",
                  source_version="existing raw cache lineage, not certified author-identical CSV",
                  availability="retrospective window localization; no causal/early-warning claim",
                  physical_clock="absolute row indices; physical minute mapping not certified",
                  label_files_read=False, thresholds_fitted=False, detection_metrics_computed=False,
                  exact_serialized_reduction=True)
    return torch


def run(config_path, execute=False):
    cfg, root, output = validate_config(config_path)
    if not execute:
        return {"status": "PREVIEW_ONLY", "operation": cfg["operation"], "output": str(output),
                "config_sha256": file_sha(config_path), "query_allowed": False,
                "requires_query_permission": cfg["operation"] == "score", "executed": False}
    require_approval(config_path, cfg)  # reject pending before any torch/data import
    start = time.monotonic()
    def timed_out(_signum, _frame):
        raise TimeoutError("Whole operation time budget exceeded")
    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(cfg["max_seconds"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "operation": cfg["operation"], "config_sha256": file_sha(config_path),
              "adapter_sha256": file_sha(__file__), "query_values_read": False, "query_labels_read": False}
    torch = None
    try:
        if cfg["operation"] == "profile":
            torch = profile(cfg, root, report)
        elif cfg["operation"] == "train":
            torch = train(cfg, root, output, report)
        else:
            torch = score(cfg, root, output, report)
        report["status"] = "COMPLETED_PENDING_REVIEW"
        return report
    except BaseException:
        report["status"] = "FAILED_NO_AUTOMATIC_RETRY"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        if torch is None:
            torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            props = torch.cuda.get_device_properties(0)
            report["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "device": props.name, "uuid": str(getattr(props, "uuid", "UNVERIFIED")),
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                 "peak_reset": "once before model construction"}
        report["whole_operation_seconds"] = time.monotonic()-start
        (output/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, args.execute), indent=2))


if __name__ == "__main__":
    main()
