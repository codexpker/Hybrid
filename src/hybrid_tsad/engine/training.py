"""Package-native extraction of the frozen source-wave optimizer/evaluation loop.

No task-specific driver. The production recipe remains 10000 x 64; helper
functions also permit small synthetic CPU equivalence checks, not research runs.
"""
import copy
import json
import math
from pathlib import Path
import time

import numpy as np

from ..config import OFFICIAL_SHA, read_json, sha256
from .identity import SCORE_SPACE
from .source import calibrate_source, read_plan


def learning_rate(step):
    return .0003 * min(step / 500, 1) * (1 if step <= 500 else .5 * (1 + math.cos(math.pi * (step - 500) / 9500)))


def source_batch(sampler, plan, rows, role, start, *, device, history=400, n=64):
    import torch
    if history not in (100, 400):
        raise ValueError("Only contract histories 100/400 are supported")
    x, y = [], []
    for j, b, c in zip(plan[role + "_record"][start:start + n], plan[role + "_origin"][start:start + n],
                       plan[role + "_channel"][start:start + n]):
        r, b, c = rows[int(j)], int(b), int(c)
        assert b - history >= r["local_start"] and b + 5 <= r["local_end_exclusive"]
        ar = sampler.array(r)
        x.append(ar[b - history:b, c])
        y.append(ar[b:b + 5, c])
    return torch.tensor(np.stack(x)[:, :, None], device=device), torch.tensor(np.stack(y)[:, :, None], device=device)


def evaluate_source(model, sampler, plan, rows, role, objective, cfg, *, device, history=400):
    import torch
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    model.eval()
    losses = []
    arrays = {k: [] for k in ["prediction", "target", "history_scale", "valid_target"]}
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg["mask_seed"])
        for start in range(0, len(plan[role + "_record"]), 64):
            x, y = source_batch(sampler, plan, rows, role, start, device=device, history=history)
            output = model(x)
            loss, _ = objective(output, y)
            assert torch.isfinite(loss)
            losses.append(float(loss))
            for k, value in [("prediction", output["prediction"]), ("target", y),
                             ("history_scale", output["scale"]), ("valid_target", torch.isfinite(y))]:
                arrays[k].append(value.cpu().numpy())
    return float(np.mean(losses)), {k: np.concatenate(v) for k, v in arrays.items()}


def optimizer_step(model, optimizer, x, y, objective, step):
    import torch
    model.train()
    lr = learning_rate(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    output = model(x)
    loss, parts = objective(output, y)
    assert torch.isfinite(loss)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1))
    assert np.isfinite(gradient_norm)
    optimizer.step()
    return loss, parts, lr, gradient_norm


def capture_step_state(model, optimizer, step, *, device):
    """Extra engineering sidecar; legacy checkpoints are kept unchanged."""
    import torch
    device = torch.device(device)
    numpy_state = np.random.get_state()
    return copy.deepcopy({"schema_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "model_history": model.history, "model_mode": getattr(model, "mode", None),
                          "step": step, "torch_cpu_rng": torch.get_rng_state(),
                          "torch_device_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                          "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]})


def restore_step_state(model, optimizer, state, *, device):
    import torch
    if state["schema_version"] != 1:
        raise ValueError("Unknown engineering step-state schema")
    if state["model_history"] != model.history or state["model_mode"] != getattr(model, "mode", None):
        raise ValueError("Step-state history/mode differs from the receiving model")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_cpu_rng"])
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_device_rng"], device)
    value = state["numpy_rng"]
    np.random.set_state((value[0], np.asarray(value[1], dtype="uint32"), *value[2:]))
    return state["step"]


def validate_training_assets(paths, job):
    """Check the frozen recipe, plan identity, data roles and all original hashes."""
    from ..data import canonical_domain
    cfg, inputs = read_json(paths["wave_config"]), read_json(paths["source_inputs"])
    expected = dict(steps=10000, batch=64, lr=.0003, weight_decay=.01, warmup=500, validate_every=500)
    if any(cfg.get(k) != v for k, v in expected.items()) or cfg.get("query_evaluation") is not False:
        raise ValueError("Native train requires the unchanged frozen 10000x64 optimizer recipe")
    history = cfg.get("history")
    if history not in (100, 400) or cfg.get("horizon", 5) != 5:
        raise ValueError("Frozen history must be 100/400 and horizon5")
    spec = cfg["jobs"][job]
    if not {"sources", "mode", "plan"} <= set(spec) or set(spec) - {"sources", "mode", "plan", "seed"}:
        raise ValueError("Unknown frozen job fields")
    sources = spec["sources"]
    if (not isinstance(sources, list) or not sources or any(s not in ("PSM", "SMD_SUBSET", "TEP") for s in sources)
            or len(set(sources)) != len(sources)):
        raise ValueError("Source domains must be unique supported public-loader domains")
    target = canonical_domain(cfg["target"])
    if spec["mode"] not in ("temporal", "tf", "hybrid") or target in [canonical_domain(s) for s in spec["sources"]]:
        raise ValueError("Invalid source mode or target exclusion")
    if read_json(paths["entry_gate"]).get("status") != "PASS":
        raise ValueError("Frozen source entry gate has not passed")
    if Path(inputs["manifest"]).resolve() != paths["source_manifest"] or Path(inputs["official"]).resolve() != paths["official"]:
        raise ValueError("Source input paths differ from approved native inputs")
    if sha256(paths["official"]) != OFFICIAL_SHA:
        raise ValueError("Official source identity changed")
    if any(sha256(p) != h for p, h in inputs["hashes"].items()):
        raise ValueError("Original source input hash mismatch")
    rows = read_json(paths["rows"])
    manifest = read_json(paths["source_manifest"])
    required_hashes = {paths["official"], paths["source_manifest"]} | {Path(r["source"]).resolve() for r in rows}
    if any(r["dataset"] == "TEP" for r in rows):
        required_hashes.add(Path(manifest.get("tep_cache_path", paths["source_manifest"].parent / "tep_normal_runs_1_100.npy")).resolve())
    if not required_hashes <= {Path(p).resolve() for p in inputs["hashes"]}:
        raise ValueError("Raw source/cache files are missing frozen input hashes")
    expected_rows = [r for r in manifest["records"] if r["dataset"] in spec["sources"] and canonical_domain(r["dataset"]) != target]
    if rows != expected_rows or any(r["history"] != history or r["horizon"] != 5 for r in rows):
        raise ValueError("Plan rows differ from the frozen source manifest/history")
    plan = read_plan(paths["plan"])
    for role, total in (("fit", 640000), ("select", 256), ("cal", 256)):
        for suffix, dtype in (("record", "uint16"), ("origin", "uint32"), ("channel", "uint8")):
            value = plan[role + "_" + suffix]
            if value.shape != (total,) or value.dtype != np.dtype(dtype):
                raise ValueError("Frozen compact plan exposure/dtype changed")
        for ri in np.unique(plan[role + "_record"]):
            if int(ri) >= len(rows) or rows[int(ri)]["role"] != role:
                raise ValueError("Plan role/record mismatch")
            r = rows[int(ri)]
            selected = plan[role + "_record"] == ri
            origins, channels = plan[role + "_origin"][selected], plan[role + "_channel"][selected]
            if (not np.isin(origins, r["forecast_origins"]).all() or np.any(channels >= len(r["columns"]))
                    or np.any(origins.astype("int64") - history < r["local_start"])
                    or np.any(origins.astype("int64") + 5 > r["local_end_exclusive"])):
                raise ValueError("Plan footprint/channel differs from frozen source rows")
        domains = np.asarray([rows[int(i)]["dataset"] for i in plan[role + "_record"]])
        expected_domains = np.asarray([spec["sources"][(n // 64) % len(spec["sources"])] for n in range(total)])
        if not np.array_equal(domains, expected_domains):
            raise ValueError("Source domain order differs from the frozen 64-window alternating plan")
    return cfg, inputs, spec, rows, plan


def run_native_training(paths, output, *, job, device, operation_digest):
    """Real fixed-budget production loop. This function is never run by tests."""
    import torch
    from ..data import ManifestSampler
    from ..model import HybridPredictor, objective
    cfg, inputs, spec, rows, plan = validate_training_assets(paths, job)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    input_hashes = {name: sha256(path) for name, path in paths.items()}
    for name in ("contract", "approval", "wave_config", "source_inputs", "source_manifest", "rows", "entry_gate"):
        if name in paths:
            snapshot = output / "frozen_inputs" / (name + paths[name].suffix)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(paths[name].read_bytes())
    package_root = Path(__file__).resolve().parents[1]
    code_hashes = {}
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root)
        destination = output / "implementation/src/hybrid_tsad" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        code_hashes[str(relative).replace("\\", "/")] = sha256(source)
    (output / "source_provenance.json").write_text(json.dumps({
        "operation_manifest_sha256": operation_digest, "package_code_sha256": code_hashes,
        "external_inputs": {name: {"path": str(path), "sha256": input_hashes[name]} for name, path in paths.items()},
        "official_sha256": sha256(paths["official"]), "upstream_copied": False}, indent=2))
    torch.set_num_threads(2)
    seed = spec.get("seed", cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    history = cfg["history"]
    model = HybridPredictor(paths["official"], mode=spec["mode"], history=history).to(device)
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0003, weight_decay=.01)
    sampler = ManifestSampler(inputs["manifest"], exclude_dataset=cfg["target"])
    curve, best, stale = [], float("inf"), 0
    started = time.time()
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, 10001):
        x, y = source_batch(sampler, plan, rows, "fit", (step - 1) * 64, device=device, history=history)
        loss, parts, lr, gradient_norm = optimizer_step(model, optimizer, x, y, objective, step)
        if step % 25 == 0:
            row = dict(step=step, loss=float(loss.detach()), future=float(parts["future"].detach()),
                       tf=float(parts["tf"].detach()), lr=lr, gradient_norm=gradient_norm, seconds=time.time()-started)
            curve.append(row)
        if step % 500 == 0:
            val, arrays = evaluate_source(model, sampler, plan, rows, "select", objective, cfg, device=device, history=history)
            if val < best:
                best, stale, best_step = val, 0, step
                torch.save(model.state_dict(), output / "best.pt")
            else:
                stale += 1
            row.update(source_select_loss=val, best_step=best_step, stale=stale, patience_warning=stale >= 4)
            (output / "curve.json").write_text(json.dumps(curve, indent=2))
            np.savez_compressed(output / f"select_step{step}.npz", **arrays)
            print(job, step, val, flush=True)
        if is_cuda:
            assert torch.cuda.max_memory_allocated(device) <= 20 * 1024**3
    torch.save(model.state_dict(), output / "step10000.pt")
    torch.save({"optimizer": optimizer.state_dict(), "step": 10000}, output / "optimizer_step10000.pt")
    for role in ("select", "cal"):
        _, arrays = evaluate_source(model, sampler, plan, rows, role, objective, cfg, device=device, history=history)
        np.savez_compressed(output / f"{role}_primary.npz", **arrays)
    with np.load(output / "cal_primary.npz") as bank:
        scores, calibration = calibrate_source(bank)
    np.savez_compressed(output / "cal_scores.npz", **scores)
    calibration.update(calibration_schema_version=2, role="source_cal", score_space=SCORE_SPACE, history=history,
                       source_bank_sha256=sha256(output / "cal_primary.npz"), operation_manifest_sha256=operation_digest)
    (output / "source_calibration.json").write_text(json.dumps(calibration, indent=2, allow_nan=False))
    delta = {n: float((p.detach().cpu() - initial[n]).abs().max()) for n, p in model.named_parameters()}
    assert all(delta[n] == 0 for n, p in model.named_parameters() if not p.requires_grad)
    report_spec = dict(spec)
    if "seed" in report_spec and report_spec["seed"] != seed:
        raise ValueError("Frozen job seed differs from the actual training seed")
    report_spec.pop("seed", None)
    checkpoint_meta = dict(checkpoint_sha256=sha256(output / "step10000.pt"), history=history, horizon=5,
                           mode=report_spec["mode"], copies=2, temporal_depth=4, step=10000, seed=seed,
                           source_domains=list(report_spec["sources"]), excluded_target=cfg["target"],
                           source_manifest_sha256=sha256(paths["source_manifest"]), shared_plan_sha256=sha256(paths["plan"]),
                           training_config_sha256=sha256(paths["wave_config"]), operation_manifest_sha256=operation_digest)
    checkpoint_meta["weight_sha256"] = checkpoint_meta["checkpoint_sha256"]
    (output / "checkpoint_metadata.json").write_text(json.dumps(checkpoint_meta, indent=2))
    if any(sha256(package_root / name) != h for name, h in code_hashes.items()):
        raise ValueError("Package code changed during the frozen source run")
    if (any(sha256(path) != input_hashes[name] for name, path in paths.items())
            or any(sha256(path) != h for path, h in inputs["hashes"].items())):
        raise ValueError("Frozen inputs changed during source training")
    summary = dict(status="SOURCE_10K_COMPLETE", job=job, **report_spec, **model.parameter_counts(), steps=10000,
                   exposure_windows=640000, seed=seed, best_step=best_step, primary_step=10000, history=history,
                   seconds=time.time()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if is_cuda else None,
                   plan_sha256=sha256(paths["plan"]), normal_error_scale=calibration["normal_error_scale"],
                   threshold99=calibration["threshold99"], parameter_changes=delta, query_evaluated=False,
                   original_source_unchanged=sha256(inputs["official"]) == inputs["hashes"][inputs["official"]],
                   operation_manifest_sha256=operation_digest)
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
