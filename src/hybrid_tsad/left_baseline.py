"""Thin external LEFT adapter. Upstream model/source/data remain outside this package.

Preview uses stdlib only. Execution requires a separately approved config. No query
loader is reachable: score and the unfrozen C protocol are explicitly refused.
"""
import argparse
import hashlib
import importlib
import json
import random
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


COMMIT = "3fadb4811797075e233076b9faceb0d8ec59b697"
PSM_SHA = "38e446209e82df94b82c11aa43d87e3671da86167ed794c1b7be5eb91fc0e8ac"
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
    if cfg.get("operation") not in ("profile", "train"):
        raise ValueError("score/query unsupported; only profile or train")
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
    else:
        for key in ("array", "metadata", "array_sha256", "metadata_sha256"):
            if not cfg.get("data", {}).get(key):
                raise ValueError("Train prefix requires exact identity: " + key)
        if cfg.get("max_allocated_GiB") != 20:
            raise ValueError("Training memory contract mismatch")
    return cfg, root, output


def require_approval(config_path, cfg):
    approval = read_json(cfg["approval_path"])
    expected = "PRIMARY_RELEASED_LEFT_BASELINE_" + cfg["operation"].upper()
    if approval.get("status") != expected or approval.get("config_sha256") != file_sha(config_path):
        raise PermissionError("Operation is not released for this exact config")
    if approval.get("adapter_sha256") != file_sha(__file__):
        raise PermissionError("Adapter bytes are not released")
    if approval.get("allow_query", True) or approval.get("allow_labels", True):
        raise PermissionError("This adapter never accepts query or label access")
    if approval.get("max_seconds") != cfg["max_seconds"]:
        raise PermissionError("Timeout contract mismatch")
    return approval


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


def run(config_path, execute=False):
    cfg, root, output = validate_config(config_path)
    if not execute:
        return {"status": "PREVIEW_ONLY", "operation": cfg["operation"], "output": str(output),
                "config_sha256": file_sha(config_path), "query_allowed": False, "executed": False}
    require_approval(config_path, cfg)  # reject pending before any torch/data import
    start = time.monotonic()
    def timed_out(_signum, _frame):
        raise TimeoutError("Whole operation time budget exceeded")
    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(cfg["max_seconds"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "operation": cfg["operation"], "config_sha256": file_sha(config_path),
              "adapter_sha256": file_sha(__file__), "query_or_labels_read": False}
    torch = None
    try:
        torch = profile(cfg, root, report) if cfg["operation"] == "profile" else train(cfg, root, output, report)
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
