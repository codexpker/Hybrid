"""Shared, explicit-action development runner for the endpoint candidate.

No tensor library, data, weights or labels are loaded by preview/import.
This is from-scratch engineering development, not official reproduction.
"""
import argparse
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

MODES = ("tf", "lag", "concat", "gated")
_EXECUTION_ACTION = ContextVar("candidate_execution_action", default=None)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=encode, allow_nan=False)


def protocol_hash(config):
    # Release signatures can be added after raw score production without
    # changing the scientific configuration or checkpoint identity.
    value = {k: v for k, v in config.items() if k != "release"}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def resolve(value, config_path):
    value = os.path.expandvars(os.path.expanduser(str(value)))
    if "$" in value or "PENDING" in value:
        raise ValueError("Unresolved path")
    p = Path(value)
    return p if p.is_absolute() else Path(config_path).parent / p


def validate_config(config):
    if config.get("schema_version") != 1 or config.get("development_only") is not True:
        raise ValueError("Explicit development schema required")
    model, train, data = config["model"], config["training"], config["data"]
    if model["horizon"] != 1 or model["history"] < 1:
        raise ValueError("This runner supports positive history, target1 only")
    if config.get("tf_checkpoint") is not None:
        raise ValueError("First development recipe is explicitly from_scratch")
    if any(type(train[k]) is not int or train[k] < 1 for k in
           ("updates", "batch_size", "validate_every", "select_windows")):
        raise ValueError("Positive integer training dimensions required")
    if train["warmup"] != 0 or any(not np.isfinite(train[k]) or train[k] <= 0
                                  for k in ("lr", "grad_clip")):
        raise ValueError("Finite training settings, zero warmup required")
    if not np.isfinite(train["weight_decay"]) or train["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    previous = 0
    for name in ("FIT", "SELECT", "CAL"):
        start, end = data["roles"][name]
        if (type(start) is not int or type(end) is not int or start < previous
                or end > data["train"]["shape"][0] or end - start <= model["history"]):
            raise ValueError("Nonoverlapping, role-contained complete windows required")
        if name != "FIT" and start - previous < data["guard"]:
            raise ValueError("Role guard too short")
        previous = end
    if data["train"]["shape"][1] != data["test"]["shape"][1]:
        raise ValueError("Channel dimensions differ")
    if config["calibration"] != {"quantile": 0.99, "method": "higher", "strict": True,
                                  "confirmation_points": 3}:
        raise ValueError("Pinned q99 strict>, three-point current confirmation required")


def role_targets(bounds, history):
    start, end = bounds
    if any(type(x) is not int for x in (start, end, history)) or start < 0 or history < 1 or end <= start + history:
        raise ValueError("No complete role-contained windows")
    return np.arange(start + history, end, dtype=np.int64)


def fit_medians(array, fit_bounds):
    a = np.asarray(array)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.number) or np.isinf(a).any():
        raise ValueError("Numeric 2D input without infinity required")
    fit = a[slice(*fit_bounds)]
    if not len(fit) or np.isnan(fit).all(axis=0).any():
        raise ValueError("All-missing FIT channel")
    return np.nanmedian(fit, axis=0).astype(np.float64)


def impute(array, medians):
    a = np.asarray(array)
    m = np.asarray(medians)
    if a.ndim != 2 or m.shape != (a.shape[1],) or not np.isfinite(m).all() or np.isinf(a).any():
        raise ValueError("Invalid imputation input")
    result = np.where(np.isnan(a), m[None, :], a).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite after imputation")
    return result


def sample_targets(targets, updates, batch_size, seed):
    # This RNG is independent of Torch initialization/masking and mode order.
    return np.random.default_rng(seed).choice(targets, (updates, batch_size), replace=True)


def windows(array, targets, history):
    targets = np.asarray(targets)
    if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer) or np.any(targets < history) or np.any(targets >= len(array)):
        raise ValueError("Invalid window target coordinates")
    return np.stack([array[t-history:t] for t in targets]), array[targets, None, :]


def calibrate(scores):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Finite nonempty normal CAL scores required")
    return float(np.quantile(scores, 0.99, method="higher"))


def confirmed_alarm(index, scores, threshold, count=3):
    index, scores = np.asarray(index), np.asarray(scores)
    if (index.ndim != 1 or scores.shape != index.shape or not len(index)
            or not np.issubdtype(index.dtype, np.integer) or np.any(np.diff(index) <= 0)
            or not np.isfinite(scores).all() or not np.isfinite(threshold)
            or type(count) is not int or count < 1):
        raise ValueError("Ordered finite score stream required")
    alarm = np.zeros(len(scores), dtype=bool)
    run = 0
    for i in range(len(scores)):
        if i and index[i] != index[i-1] + 1:
            run = 0
        run = run + 1 if scores[i] > threshold else 0
        alarm[i] = run >= count
    return alarm


def copy_common_state(model, master):
    """Copy exact same-name/shape tensors; concat-specific tensors stay seeded."""
    own = model.state_dict()
    copied = []
    for key, value in own.items():
        if key in master and value.shape == master[key].shape:
            if value.dtype != master[key].dtype:
                raise ValueError("Common initialization dtype mismatch")
            own[key] = master[key].detach().clone()
            copied.append(key)
    model.load_state_dict(own, strict=True)
    return copied


def state_identities(state):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in state.items()}


def _load_array(spec, config_path):
    p = resolve(spec["path"], config_path)
    if sha256(p) != spec["sha256"]:
        raise ValueError("Data SHA256 mismatch")
    if spec.get("format") == "csv":
        import csv
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f))
        drop = spec.get("drop_columns", [])
        if len(set(header)) != len(header) or any(name not in header for name in drop):
            raise ValueError("CSV columns missing or duplicated")
        columns = [i for i, name in enumerate(header) if name not in drop]
        if not columns or (spec.get("value_columns") is not None and
                           [header[i] for i in columns] != spec["value_columns"]):
            raise ValueError("CSV value columns mismatch")
        array = np.genfromtxt(p, delimiter=",", skip_header=1, usecols=columns,
                              dtype=np.float64, ndmin=2, encoding="utf-8-sig")
        if len(spec["shape"]) == 1 and array.shape[1] == 1:
            array = array[:, 0]
        if list(array.shape) != spec["shape"]:
            raise ValueError("Data shape mismatch")
        return array
    loaded = np.load(p, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            array = loaded[spec["key"]].copy()
        finally:
            loaded.close()
    else:
        array = loaded
    if list(array.shape) != spec["shape"]:
        raise ValueError("Data shape mismatch")
    return array


def authorize(config, action, config_path, mode=None):
    validate_config(config)
    release = config.get("release") or {}
    if (release.get("protocol_sha256") != protocol_hash(config)
            or action not in release.get("approved_actions", [])
            or not release.get("release_id") or (mode and mode not in release.get("modes", []))):
        raise ValueError("Explicit matching action/mode release required")
    sources = release.get("source_sha256", {})
    required = ("candidate_experiment.py", "candidate_tf_lag.py", "candidate_autoformer_lag.py")
    if not all(name in sources and sha256(Path(__file__).parent / name) == sources[name] for name in required):
        raise ValueError("Runner/model source release mismatch")
    for name, expected in sources.items():
        if sha256(Path(__file__).parent / name) != expected:
            raise ValueError("Additional dependency source release mismatch")
    if config["limits"]["max_seconds"] <= 0 or config["limits"]["max_cuda_bytes"] <= 0:
        raise ValueError("Positive explicit resource limits required")


def _torch_setup(config):
    import torch
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(min(1., config["limits"]["max_cuda_bytes"]/props.total_memory), device)
    return torch


def _out(config, config_path):
    return resolve(config["output_directory"], config_path)


def _receipt(config, config_path, stage):
    if stage["protocol_sha256"] != protocol_hash(config):
        raise ValueError("Previous stage protocol mismatch")
    if stage["source_sha256"] != config["release"]["source_sha256"]:
        approved = config["release"].get("parent_source_compatibility", {})
        parent, current = stage["source_sha256"], config["release"]["source_sha256"]
        differences = {name for name in set(parent) | set(current) if parent.get(name) != current.get(name)}
        if (approved.get("approved_by") != "root" or not approved.get("approval_id")
                or approved.get("parent_source_sha256") != parent
                or set(approved.get("allowed_old_stages", [])) != {"prepare", "train"}
                or stage.get("stage") not in approved["allowed_old_stages"]
                or len(parent) != 10 or set(parent) != set(current)
                or differences != {"candidate_experiment.py"}
                or parent.get("candidate_experiment.py") != "c38123ef639c3adf43c7b3ad6aa0232587f1538c265aee9477121c5a9f076a59"):
            raise ValueError("Previous stage source mismatch; exact root-approved resource-only parent required")
    for name, expected in stage["files"].items():
        if sha256(_out(config, config_path) / name) != expected:
            raise ValueError("Previous stage artifact mismatch")


def _save_receipt(config, config_path, directory, metadata):
    out = _out(config, config_path)
    metadata.update(protocol_sha256=protocol_hash(config), config_sha256=sha256(config_path),
                    source_sha256=config["release"]["source_sha256"],
                    files={str(p.relative_to(out)): sha256(p) for p in directory.rglob("*") if p.is_file()})
    json_write(directory / "receipt.json", metadata)


def _read_receipt(config, config_path, relative):
    stage = json.loads((_out(config, config_path) / relative / "receipt.json").read_text())
    _receipt(config, config_path, stage)
    return stage


def prepare(config, config_path):
    """Only train values; materialize shared sampling and explicit step0 states."""
    from .candidate_tf_lag import candidate_from_config
    torch = _torch_setup(config)
    out = _out(config, config_path)
    directory = out / "prepare"
    directory.mkdir(parents=True, exist_ok=False)
    raw = _load_array(config["data"]["train"], config_path)
    medians = fit_medians(raw, config["data"]["roles"]["FIT"])
    impute(raw, medians)  # Reject inf/float32 overflow in all train roles.
    np.save(directory / "medians.npy", medians)
    h, tr = config["model"]["history"], config["training"]
    indices = {role: role_targets(bounds, h) for role, bounds in config["data"]["roles"].items()}
    schedule = sample_targets(indices["FIT"], tr["updates"], tr["batch_size"], config["sampling_seed"])
    np.savez(directory / "indices.npz", **indices, schedule=schedule,
             SELECT_validation=indices["SELECT"][:tr["select_windows"]])
    torch.manual_seed(config["seed"])
    master = candidate_from_config(config_path, mode="gated").cpu()
    master_state = {k: v.detach().clone() for k, v in master.state_dict().items()}
    torch.save(master_state, directory / "common_init_bank.pt")
    identities = {}
    for mode in MODES:
        torch.manual_seed(config["seed"])
        model = candidate_from_config(config_path, mode=mode).cpu()
        copied = copy_common_state(model, master_state)
        torch.save(model.state_dict(), directory / (mode + "_step0.pt"))
        identities[mode] = {"copied_keys": copied, "tensor_sha256": state_identities(model.state_dict()),
                            "parameter_counts": model.parameter_counts()}
    json_write(directory / "initialization.json", identities)
    _save_receipt(config, config_path, directory, {"stage": "prepare", "torch": torch.__version__,
        "sampling_seed": config["sampling_seed"], "from_scratch": True,
        "randomness_note": "Same sample indices/common tensors; mode mask RNG trajectories not asserted identical"})


def _tensors(torch, array, targets, history, device):
    x, y = windows(array, targets, history)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def _resources(torch, device, started, config):
    max_seconds = config["limits"]["max_seconds"]
    action = _EXECUTION_ACTION.get()
    release = config.get("release") or {}
    override = release.get("action_limits", {}).get(action) if action is not None else None
    if override is not None:
        if (action != "infer" or action not in release.get("approved_actions", [])
                or not release.get("release_id") or override.get("approved_by") != "root"
                or not override.get("approval_id") or set(override) != {"approved_by", "approval_id", "max_seconds"}
                or type(override["max_seconds"]) not in (int, float)
                or not np.isfinite(override["max_seconds"]) or not 0 < override["max_seconds"] <= 4800):
            raise ValueError("Explicit root-approved infer-only wall override required")
        max_seconds = override["max_seconds"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = max(torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))
        if peak > min(config["limits"]["max_cuda_bytes"], 20 * 2**30):
            raise RuntimeError("CUDA memory budget exceeded")
    else:
        peak = 0
    elapsed = time.monotonic() - started
    if elapsed > max_seconds:
        raise RuntimeError("Wall time budget exceeded")
    return elapsed, peak


def train(config, config_path, mode):
    from .candidate_tf_lag import candidate_from_config, candidate_objective
    _read_receipt(config, config_path, "prepare")
    torch = _torch_setup(config)
    device = torch.device(config["device"])
    out = _out(config, config_path)
    directory = out / ("train_" + mode)
    directory.mkdir(exist_ok=False)
    raw = _load_array(config["data"]["train"], config_path)
    array = impute(raw, np.load(out / "prepare/medians.npy", allow_pickle=False))
    with np.load(out / "prepare/indices.npz", allow_pickle=False) as z:
        schedule, select = z["schedule"].copy(), z["SELECT_validation"].copy()
    model = candidate_from_config(config_path, mode=mode).to(device)
    initial = torch.load(out / ("prepare/" + mode + "_step0.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(initial, strict=True)
    tr, h = config["training"], config["model"]["history"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=tr["lr"], weight_decay=tr["weight_decay"])
    # Reset masks after model creation; validation uses and restores RNG state.
    torch.manual_seed(config["mask_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["mask_seed"])
        torch.cuda.reset_peak_memory_stats(device)
    torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
               directory / "mask_rng_start.pt")
    started, best, selected = time.monotonic(), float("inf"), None
    losses, validations = [], []
    with (directory / "steps.jsonl").open("x", encoding="utf-8") as log:
        for step, targets in enumerate(schedule, 1):
            model.train()
            x, y = _tensors(torch, array, targets, h, device)
            optimizer.zero_grad(set_to_none=True)
            loss = candidate_objective(model(x), y)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            elapsed, peak = _resources(torch, device, started, config)
            row = {"step": step, "loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"],
                   "grad_norm_before_clip": float(grad), "elapsed_seconds": elapsed, "peak_cuda_bytes": peak}
            log.write(json.dumps(row) + "\n"); log.flush()
            losses.append(row["loss"])
            if step % tr["validate_every"] == 0 or step == len(schedule):
                model.eval()
                total, n = 0.0, 0
                devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    torch.manual_seed(config["validation_seed"])
                    for start in range(0, len(select), tr["batch_size"]):
                        ix = select[start:start+tr["batch_size"]]
                        x, y = _tensors(torch, array, ix, h, device)
                        value = float(candidate_objective(model(x), y))
                        if not np.isfinite(value):
                            raise RuntimeError("Nonfinite normal validation")
                        total += value * len(ix); n += len(ix)
                        _resources(torch, device, started, config)
                mse = total / n
                validations.append({"step": step, "normal_SELECT_MSE": mse, "windows": n})
                if mse < best:
                    best, selected = mse, step
                    torch.save(model.state_dict(), directory / "selected.pt")
    torch.save(model.state_dict(), directory / "final.pt")
    torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}, directory / "final_training_state.pt")
    final = model.state_dict()
    changed = {k: float((v.detach().cpu() - initial[k]).abs().max()) for k, v in final.items()}
    json_write(directory / "parameter_changes.json", changed)
    json_write(directory / "validation.json", validations)
    np.savez(directory / "loss_curve.npz", step=np.arange(1, len(losses)+1), loss=losses)
    elapsed, peak = _resources(torch, device, started, config)
    _save_receipt(config, config_path, directory, {"stage": "train", "mode": mode,
        "selected_step": selected, "normal_SELECT_MSE": best, "updates": len(schedule),
        "elapsed_seconds": elapsed, "peak_cuda_bytes": peak, "torch": torch.__version__,
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "changed_tensors": sum(v > 0 for v in changed.values()), "labels_opened": False})


def infer(config, config_path, mode):
    from .candidate_tf_lag import candidate_from_config, residual_scores
    _read_receipt(config, config_path, "prepare")
    training = _read_receipt(config, config_path, "train_" + mode)
    torch = _torch_setup(config)
    device = torch.device(config["device"])
    out = _out(config, config_path)
    directory = out / ("infer_" + mode)
    directory.mkdir(exist_ok=False)
    model = candidate_from_config(config_path, mode=mode).to(device)
    model.load_state_dict(torch.load(out / ("train_"+mode+"/selected.pt"), map_location=device, weights_only=True), strict=True)
    model.eval()
    medians = np.load(out / "prepare/medians.npy", allow_pickle=False)
    h, batch = config["model"]["history"], config["training"]["batch_size"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for role, spec in (("CAL", config["data"]["train"]), ("TEST", config["data"]["test"])):
        array = impute(_load_array(spec, config_path), medians)
        bounds = config["data"]["roles"]["CAL"] if role == "CAL" else [0, len(array)]
        indices = role_targets(bounds, h)
        predictions, errors, scores, targets = [], [], [], []
        torch.manual_seed(config["inference_seed"])
        with torch.no_grad():
            for start in range(0, len(indices), batch):
                ix = indices[start:start+batch]
                x, y = _tensors(torch, array, ix, h, device)
                output = model(x)
                residual = residual_scores(output, y)
                predictions.append(output["prediction"].cpu().numpy()[:, 0])
                errors.append(residual["per_channel"].cpu().numpy()[:, 0])
                scores.append(residual["continuous"].cpu().numpy()[:, 0])
                targets.append(y.cpu().numpy()[:, 0])
                _resources(torch, device, started, config)
        np.savez(directory / (role + ".npz"), index=indices, input_start=indices-h,
                 input_end_inclusive=indices-1, available_index=indices,
                 prediction=np.concatenate(predictions), target=np.concatenate(targets),
                 per_channel_error=np.concatenate(errors), score=np.concatenate(scores))
    with np.load(directory / "CAL.npz", allow_pickle=False) as z:
        threshold = calibrate(z["score"])
    json_write(directory / "calibration.json", {"threshold": threshold, **config["calibration"],
               "normal_CAL_only": True, "test_labels_opened": False})
    elapsed, peak = _resources(torch, device, started, config)
    _save_receipt(config, config_path, directory, {"stage": "infer", "mode": mode,
        "selected_step": training["selected_step"], "elapsed_seconds": elapsed,
        "peak_cuda_bytes": peak, "labels_opened": False})


def profile_infer(config, config_path, mode):
    """Fixed first 128 legal TEST windows, no labels and no range selection."""
    from .candidate_tf_lag import candidate_from_config, residual_scores
    _read_receipt(config, config_path, "prepare")
    _read_receipt(config, config_path, "train_" + mode)
    torch = _torch_setup(config)
    device = torch.device(config["device"])
    out = _out(config, config_path)
    directory = out / ("profile_infer_" + mode)
    directory.mkdir(exist_ok=False)
    model = candidate_from_config(config_path, mode=mode).to(device)
    model.load_state_dict(torch.load(out / ("train_"+mode+"/selected.pt"), map_location=device, weights_only=True), strict=True)
    model.eval()
    medians = np.load(out / "prepare/medians.npy", allow_pickle=False)
    array = impute(_load_array(config["data"]["test"], config_path), medians)
    h, batch = config["model"]["history"], config["training"]["batch_size"]
    all_indices = role_targets([0, len(array)], h)
    indices = all_indices[:128]
    if len(indices) != 128:
        raise ValueError("Fixed cost profile requires first 128 legal TEST windows")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    rows = []
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            ix = indices[start:start+batch]
            x, y = _tensors(torch, array, ix, h, device)
            residual = residual_scores(model(x), y)
            rows.append(residual["continuous"].cpu().numpy()[:, 0])
            _resources(torch, device, started, config)
    seconds, peak = _resources(torch, device, started, config)
    np.savez(directory / "fixed_prefix.npz", index=indices, score=np.concatenate(rows))
    cal_points = len(role_targets(config["data"]["roles"]["CAL"], h))
    _save_receipt(config, config_path, directory, {"stage": "profile_infer", "mode": mode,
        "windows": 128, "elapsed_compute_seconds": seconds, "peak_cuda_bytes": peak,
        "estimated_CAL_plus_TEST_compute_seconds": seconds/128*(cal_points+len(all_indices)),
        "estimate_note": "Fixed-prefix estimate, includes first forward overhead; not a guaranteed upper bound; excludes loading",
        "labels_opened": False, "range_selection": "first128_legal_TEST_windows_fixed_in_source"})


def evaluate_arrays(index, score, full_labels, threshold, *, affiliation=None):
    """Reuse shared ranking/event evaluation; confirmation is a separate fixed stream."""
    from .engine.evaluation import evaluate_point_ranking, evaluate_fixed_scores
    labels = np.asarray(full_labels)
    if labels.ndim != 1 or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Binary full test labels required")
    index = np.asarray(index)
    if np.any(index < 0) or np.any(index >= len(labels)):
        raise ValueError("Label coordinates out of bounds")
    mask = np.zeros(len(labels), bool); mask[index] = True
    ranking = evaluate_point_ranking(index, score, labels, mask, query_start=0)
    coverage = [{"event_id": e["event_id"], "start": e["start"], "end_exclusive": e["end_exclusive"],
                 "eligible_complete": e["complete"], "scored_points": e["scored_points"],
                 "total_points": e["total_points"]} for e in ranking["event_coverage"]]
    streams = {"strict_point": np.asarray(score) > threshold,
               "confirmed3_current": confirmed_alarm(index, score, threshold)}
    result = {"ranking": ranking["metrics"], "event_coverage": ranking["event_coverage"], "streams": {}}
    for name, alarm in streams.items():
        y = labels[index]
        if len(np.unique(y)) != 2:
            result["streams"][name] = {"status": "NA_SINGLE_CLASS"}
            continue
        # Shared helper is threshold-only; pass the already-fixed binary alarm,
        # then discard its irrelevant binary ranking fields (ranking above uses raw).
        row = evaluate_fixed_scores(index, y, alarm.astype(float), 0.5, coverage, affiliation=affiliation)
        for key in ("AP", "ROC", "AP_over_positive_fraction", "threshold"):
            row["metrics"].pop(key, None)
        m = row["metrics"]
        tp, fp, fn = m["TP"], m["FP"], m["FN"]
        m.update(point_Recall=tp/(tp+fn) if tp+fn else None,
                 point_F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                 normal_exposure_points=int((y == 0).sum()),
                 normal_false_alarm_points=fp, raw_threshold=threshold)
        for event in row["events"]:
            event["delay_censored"] = event["complete"] and not event["any_hit"]
            event["censor_after_points"] = event["total_points"] if event["delay_censored"] else None
        row.pop("complete_event_valid")
        result["streams"][name] = row
    result["Range_VUS"] = {"status": "NOT_EVALUATED_OFFICIAL_DEPENDENCY_AND_FIXED_BUFFER_REQUIRED"}
    return result


def _clean_json(value):
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def evaluate(config, config_path):
    out = _out(config, config_path)
    approved = config["release"].get("approved_raw_sha256", {})
    # Validate all four modes' complete raw artifacts BEFORE opening any label.
    for mode in MODES:
        _read_receipt(config, config_path, "infer_" + mode)
        for role in ("CAL", "TEST"):
            name = "infer_" + mode + "/" + role + ".npz"
            if name not in approved or sha256(out / name) != approved[name]:
                raise ValueError("All four raw score banks require explicit hash approval before labels")
    affiliation, provenance = None, None
    if config.get("official_affiliation_identity"):
        from .engine.official_affiliation import load_official_affiliation
        affiliation, provenance = load_official_affiliation(resolve(config["official_affiliation_identity"], config_path))
    labels = _load_array(config["evaluation_labels"], config_path)
    directory = out / "evaluate"
    directory.mkdir(exist_ok=False)
    for mode in MODES:
        threshold = json.loads((out / ("infer_"+mode+"/calibration.json")).read_text())["threshold"]
        with np.load(out / ("infer_"+mode+"/TEST.npz"), allow_pickle=False) as z:
            result = evaluate_arrays(z["index"], z["score"], labels, threshold, affiliation=affiliation)
        result["affiliation_provenance"] = provenance
        json_write(directory / (mode+".json"), _clean_json(result))
    _save_receipt(config, config_path, directory, {"stage": "evaluate", "labels_opened": True,
                  "scope": "Single-seed short-budget engineering development; no novelty/generalization claim"})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def block(name):
    for prefix, label in (("tf.encoder.", "tf_encoder"), ("tf.net.", "tf_order"),
            ("tf.real_input_embed.", "tf_embed"), ("tf.imag_input_embed.", "tf_embed"),
            ("tf_project.", "tf_projection"), ("lag.embed.", "lag_embed"),
            ("lag.self_operator.", "lag_self"), ("lag.cross_operator.", "lag_cross"),
            ("lag.merge.", "lag_merge"), ("gate.", "fusion"),
            ("concat.", "fusion"), ("head.", "head")):
        if name.startswith(prefix):
            return label
    return "other"


def profile(config_path):
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if (cfg["scope"] != "MODEL-FIRST-TF-LAG-GPU-L0-20261102" or cfg["seed"] != 20261102
            or cfg["shape"] != [2, 100, 25] or cfg["updates_per_mode"] != 3
            or cfg["modes"] != ["tf", "lag", "concat", "gated"]):
        raise ValueError("Profile must match the fixed root L0 contract")
    for item in cfg["sources"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("Profile source/config/contract identity mismatch: " + item["path"])
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "execution_config.json", cfg)
    import os
    import torch
    from .candidate_tf_lag import candidate_from_config, candidate_objective, residual_scores, forecast_coordinates
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
        raise ValueError("This approved profile requires the checked physical GPU0")
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    if str(props.uuid).removeprefix("GPU-") != cfg["gpu_uuid"].removeprefix("GPU-"):
        raise ValueError("Physical GPU UUID mismatch")
    budget = 20 * 2**30
    torch.cuda.set_per_process_memory_fraction(min(1., budget / props.total_memory), 0)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    history = torch.randn(*cfg["shape"], device="cuda")
    target = torch.randn(2, 1, 25, device="cuda")
    torch.save({"history": history.cpu(), "target": target.cpu(),
                "coordinates": forecast_coordinates(0, 100, 1)}, output / "synthetic_inputs.pt")
    rows = []
    start = time.monotonic()
    for mode in cfg["modes"]:
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        model = candidate_from_config(cfg["candidate_config"], mode=mode).to("cuda").train()
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        initial = {n: p.detach().clone() for n, p in params.items()}
        optimizer = torch.optim.AdamW(params.values(), lr=1e-3, weight_decay=0.)
        steps = []
        for step in range(1, 4):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_start = time.monotonic()
            before = {n: p.detach().clone() for n, p in params.items()}
            result = model(history)
            loss = candidate_objective(result, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite profile loss")
            loss.backward()
            gradients = {}
            for name, p in params.items():
                if p.grad is not None:
                    if not bool(torch.isfinite(p.grad).all()):
                        raise RuntimeError("Nonfinite gradient: " + name)
                    label = block(name)
                    gradients[label] = gradients.get(label, 0.) + float(p.grad.detach().double().square().sum())
            norm = torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
            optimizer.step()
            changes = {}
            for name, p in params.items():
                label = block(name)
                changes[label] = max(changes.get(label, 0.), float((p.detach() - before[name]).abs().max()))
            torch.cuda.synchronize()
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            if max(allocated, reserved) > budget or time.monotonic() - start > 180:
                raise RuntimeError("L0 profile memory/time budget exceeded")
            item = {"step": step, "loss": float(loss.detach()), "gradnorm_before_clip": float(norm),
                    "block_gradnorms": {k: v**.5 for k, v in gradients.items()}, "block_parameter_maxdelta": changes,
                    "seconds_with_observation": time.monotonic() - step_start,
                    "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved, "lr": 1e-3}
            steps.append(item)
            print(json.dumps({"mode": mode, **item}), flush=True)
        model.eval()
        rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        with torch.no_grad():
            first = model(history)
            score = residual_scores(first, target)
            changed_target_score = residual_scores(first, target + 123.)
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state_all(rng_cuda)
            second = model(history)
        if not torch.equal(first["prediction"], second["prediction"]):
            raise RuntimeError("Fixed-RNG target-external forecast replay failed")
        if torch.equal(score["continuous"], changed_target_score["continuous"]):
            raise RuntimeError("Changed target must affect residual score")
        final_changes = {}
        for name, p in params.items():
            label = block(name)
            final_changes[label] = max(final_changes.get(label, 0.), float((p.detach() - initial[name]).abs().max()))
        required = {"head"} | ({"tf_encoder"} if mode != "lag" else set())
        required |= {"lag_self", "lag_cross"} if mode != "tf" else set()
        if not all(final_changes.get(k, 0.) > 0 for k in required):
            raise RuntimeError("Main active block had no parameter update")
        torch.save({"prediction": first["prediction"].cpu(), "target": target.cpu(),
            "prediction_normalized": first["prediction_normalized"].cpu(),
            "history_mean": first["history_mean"].cpu(), "history_scale": first["history_scale"].cpu(),
            "per_channel_residual_squared": score["per_channel"].cpu(),
            "continuous_score": score["continuous"].cpu(),
            "synthetic_state_not_research_checkpoint": {k: v.cpu() for k, v in model.state_dict().items()}},
            output / (mode + "_synthetic_output.pt"))
        rows.append({"mode": mode, "parameters": model.parameter_counts(), "steps": steps,
                     "block_total_maxdelta": final_changes, "fixed_rng_forecast_bitexact": True,
                     "target_external": True, "loaded_class": type(model).__module__ + "." + type(model).__name__})
        del model, optimizer, params, initial, before, first, second, result
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    report = {"status": "L0_COMPLETED_NOT_DETECTION_EVIDENCE", "rows": rows,
        "seconds": time.monotonic() - start, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu_name": props.name, "gpu_uuid": str(props.uuid), "expected_nvidia_uuid": cfg["gpu_uuid"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "peak_reset_count": 1,
        "optimizer_updates": 12, "synthetic_only": True, "real_data_or_pretrained_loaded": False,
        "seed": cfg["seed"], "source_hashes": cfg["sources"], "stage_GPU_hours_requires_OS_receipt": True}
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"]}), flush=True)
    return report




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", choices=("preview", "prepare", "train", "profile_infer", "infer", "evaluate"), default="preview")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--execute", action="store_true", help="Legacy fixed synthetic profile only")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if config.get("scope") == "MODEL-FIRST-TF-LAG-GPU-L0-20261102":
        if args.action != "preview" or args.mode is not None:
            parser.error("Legacy profile accepts --config [--execute] only")
        if args.execute:
            profile(args.config)
        else:
            print(json.dumps(config))
        return
    if args.execute:
        parser.error("Development actions require --action; --execute is legacy profile only")
    validate_config(config)
    if args.action == "preview":
        print(json.dumps({"action": "preview", "protocol_sha256": protocol_hash(config),
                          "data_opened": False, "weights_opened": False,
                          "release_present": bool(config.get("release")), "modes": MODES}))
        return
    if args.action in ("train", "profile_infer", "infer") and args.mode is None:
        parser.error("train/infer require explicit --mode")
    authorize(config, args.action, args.config, args.mode)
    token = _EXECUTION_ACTION.set(args.action)
    print(json.dumps({"execution_action": args.action, "release_id": config["release"]["release_id"],
                      "base_limits": config["limits"],
                      "action_limit_override": config["release"].get("action_limits", {}).get(args.action)}), flush=True)
    try:
        if args.action in ("train", "profile_infer", "infer"):
            globals()[args.action](config, args.config, args.mode)
        else:
            globals()[args.action](config, args.config)
    finally:
        _EXECUTION_ACTION.reset(token)


if __name__ == "__main__":
    main()
