"""Shared public-manifest Stage1/Z/U/F runner; default preview, explicit execute.

Synthetic smoke uses the same model/objectives/sampler/save-load kernels and is
explicitly non-scientific. It never authorizes real data or GPU execution.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import Run, load_manifest
from .training import fit, adapt, infer, calibrate_route, logit_threshold
from .scoring import apply_calibration


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(clean(value), f, indent=2, ensure_ascii=False, allow_nan=False)


def construct(config):
    from . import HybridV1, HybridV1Config, OfficialSources
    import torch
    torch.manual_seed(config["seed"])
    import os
    settings = HybridV1Config(**config["model"])
    from .scoring import recipe
    active, _ = recipe(config.get("active_components"))
    if (not settings.tf_enabled and "r_tf" in active) or (not settings.ms_enabled and "r_ms" in active):
        raise ValueError("Disabled core branch must be excluded by the fixed active_components recipe")
    if (settings.history, settings.horizon) != (config["history"], config["horizon"]):
        raise ValueError("Core/tool history and horizon must agree")
    sources = {key: os.path.expandvars(value) for key, value in config["sources"].items()}
    if any("$" in value for value in sources.values()):
        raise ValueError("Bind existing official source paths before execution")
    return HybridV1(settings, OfficialSources(**sources)).to(config.get("device", "cpu"))


def synthetic_runs(seed=17, history=128, horizon=2):
    rng = np.random.default_rng(seed)
    runs = []
    # Each role has an explicit independent run; channels remain multivariate.
    for domain, channels in (("A", 1), ("B", 2), ("C", 3)):
        for role in ("FIT", "SELECT", "CAL", "QUERY", "SUPPORT", "NORMAL_SELECT", "NORMAL_CAL"):
            values = rng.normal(size=(history+horizon+8, channels)).astype(np.float32)
            observed = np.ones(values.shape, bool); observed[1, 0] = False
            if role == "SUPPORT":
                runs.append(Run(domain, domain, domain+"_SUPPORT_NORMAL", role, values.copy(), observed.copy()))
                values[history+1:history+3] += 2
            runs.append(Run(domain, domain, domain+"_"+role, role, values, observed))
    return runs


def options(config):
    return {**{k: config[k] for k in ("history", "horizon", "steps", "lr", "kind", "seed")}, "active_components": config.get("active_components")}


def save_model(path, model):
    import torch
    torch.save(model.state_dict(), path)


def load_model(config, path):
    import torch
    model = construct(config)
    model.load_state_dict(torch.load(path, map_location=config.get("device", "cpu"), weights_only=True), strict=True)
    return model


def save_runtime(output, model):
    """Actual imported dependencies and verified upstream bytes, not labels."""
    import importlib.metadata
    import inspect
    import platform
    import torch
    import torch_frft.frft_module as fractional
    location = Path(inspect.getsourcefile(fractional))
    save_json(output/"runtime.json", {"python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "torch_frft": importlib.metadata.version("torch-frft"),
        "installed_frft_source": {"path": str(location), "sha256": hashlib.sha256(location.read_bytes()).hexdigest()},
        "external_source_manifest": model.source_manifest,
        "temporal_class": type(model.temporal.encoder).__module__+"."+type(model.temporal.encoder).__name__,
        "MS_class": type(model.ms_filter.author).__module__+"."+type(model.ms_filter.author).__name__,
        "TF_class": type(model.tf_backbone.author).__module__+"."+type(model.tf_backbone.author).__name__,
        "parameter_counts_flags_not_all_active_paths": model.parameter_counts(),
        "core_config": model.config.to_dict()})


def score_records(records, calibration, event=False):
    output = []
    for record in records:
        row = dict(record)
        if event:
            value = row.get("event_logit", float("nan"))
            row.update(score=value, valid=bool(np.isfinite(value)),
                       alarm=bool(value > calibration["threshold"]) if np.isfinite(value) else None)
        else:
            row.update(apply_calibration(row, calibration))
        output.append(row)
    return output


def source_approval(config_path, release_path, trusted_sha, phase, output):
    """Small exact-source operation guard, before scientific data/model access."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not release_path or not trusted_sha:
        raise PermissionError("Exact root source-operation release required")
    raw = Path(release_path).read_bytes(); release = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != trusted_sha or release.get("status") != "APPROVED"
            or release.get("approved_by") != "root" or release.get("allowed_actions") != [phase]
            or release.get("config_sha256") != hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
            or cfg.get("operation") != "HYBRID_V1_SOURCE_STAGE1"
            or phase not in {"source_train", "calibrate_source"}):
        raise PermissionError("Pending/wrong root source operation")
    own = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    if release.get("source_sha256") != own:
        raise PermissionError("Shared source closure changed")
    for key in ("manifest", "plans"):
        if hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest() != cfg[key+"_sha256"]:
            raise PermissionError("Fixed source manifest/plan changed")
    if Path(output) != Path(cfg["outputs"][phase]) or Path(output).exists():
        raise PermissionError("Exact fresh source output required")
    if cfg.get("permissions") != {"source_FIT_SELECT": True, "source_CAL": True, "PSM": False,
                                   "query_or_labels": False, "adaptation": False}:
        raise PermissionError("Source-only role permissions required")
    if phase == "calibrate_source":
        for key in ("checkpoint", "train_receipt"):
            if hashlib.sha256(Path(cfg[key]).read_bytes()).hexdigest() != cfg[key+"_sha256"]:
                raise PermissionError("Accepted selected source checkpoint/receipt changed")
    return {"metadata_preflight": "PASS", "phase": phase, "data_opened": False, "GPU_queried": False}


def source_action(config, phase, output):
    """Train and source-CAL are separate actions; both reuse the shared kernels."""
    import time
    began = time.monotonic()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save_json(output/"config.json", config)
    plan = json.loads(Path(config["plans"]).read_text(encoding="utf-8"))
    if plan["history"] != config["history"] or plan["horizon"] != config["horizon"] or plan["seed"] != config["seed"]:
        raise ValueError("Fixed source plan configuration mismatch")
    roles = {"FIT", "SELECT"} if phase == "source_train" else {"CAL"}
    runs = load_manifest(config["manifest"], roles=roles)
    if {r.domain for r in runs} != {"SMD", "Sunspot"} or any(r.role not in roles for r in runs):
        raise ValueError("Only the fixed SMD/Sunspot source roles are permitted")
    save_json(output/"input_runs.json", [{"domain": r.domain, "entity": r.entity, "run": r.run,
        "role": r.role, "start": r.start, "shape": list(r.values.shape),
        "values_sha256": hashlib.sha256(r.values.tobytes()).hexdigest(),
        "observed_sha256": hashlib.sha256(r.observed.tobytes()).hexdigest()} for r in runs])
    timings = {"source_load_and_identity_IO_seconds": time.monotonic()-began}
    import torch
    cuda = str(config.get("device", "cpu")).startswith("cuda")
    if cuda:
        limit = config["resources"]["training_GiB" if phase == "source_train" else "evaluation_GiB"]*1024**3
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.set_per_process_memory_fraction(min(1., limit/properties.total_memory), 0)
        torch.cuda.reset_peak_memory_stats(0)
        save_json(output/"GPU_START.json", {"name": properties.name, "uuid": str(properties.uuid),
            "allocator_limit_bytes": limit, "set_before_model_construction": True, "no_AMP": True})
    try:
        began = time.monotonic()
        model = construct(config) if phase == "source_train" else load_model(config, config["checkpoint"])
        if cuda: torch.cuda.synchronize()
        timings["model_construction_or_checkpoint_load_seconds"] = time.monotonic()-began
        began = time.monotonic()
        save_runtime(output, model)
        timings["runtime_metadata_IO_seconds"] = time.monotonic()-began
        if phase == "source_train":
            model, trace = fit(model, [r for r in runs if r.role == "FIT"], [r for r in runs if r.role == "SELECT"],
                **options(config), validate_every=config["validate_every"], sampling_plan=plan["FIT"],
                validation_plan=plan["SELECT"], weight_decay=config["weight_decay"], artifact_dir=output)
            began = time.monotonic()
            save_model(output/"source.pt", model); save_json(output/"source_training.json", trace)
            timings["final_result_IO_seconds"] = time.monotonic()-began
        else:
            records, cal_timings = [], []
            # Plans are domain-contiguous by preparation. Keep that exact order.
            domains = list(dict.fromkeys(entry["domain"] for entry in plan["CAL"]))
            if [entry["domain"] for d in domains for entry in plan["CAL"] if entry["domain"] == d] != [e["domain"] for e in plan["CAL"]]:
                raise ValueError("Source CAL plan must retain its fixed domain-contiguous order")
            for domain in domains:
                began = time.monotonic()
                part = infer(model, [r for r in runs if r.domain == domain], history=config["history"], horizon=config["horizon"],
                    seed=config["seed"], active_components=config.get("active_components"),
                    index_plan=[e for e in plan["CAL"] if e["domain"] == domain])
                cal_timings.append({"domain": domain, "windows": len(part), "wall_seconds": time.monotonic()-began})
                records.extend(part)
            began = time.monotonic()
            save_json(output/"source_CAL_timing.json", cal_timings)
            calibration = calibrate_route(records, "STAGE1", active_components=config.get("active_components"))
            save_json(output/"source_calibration_records.json", records)
            save_json(output/"source_calibration.json", calibration)
            timings["final_result_IO_seconds"] = time.monotonic()-began
    finally:
        save_json(output/"action_timing.json", timings)
        if cuda:
            save_json(output/"GPU_PEAK.json", {"allocated_bytes": torch.cuda.max_memory_allocated(0),
                "reserved_bytes": torch.cuda.max_memory_reserved(0), "resets": 1,
                "limit_bytes": limit})



def execute(config, phase, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save_json(output/"config.json", config)
    kwargs = options(config)
    synthetic = phase == "synthetic_smoke"
    runs = synthetic_runs(config["seed"], config["history"], config["horizon"]) if synthetic else load_manifest(config["manifest"])
    save_json(output/"input_runs.json", [{"domain": r.domain, "entity": r.entity, "run": r.run,
        "role": r.role, "start": r.start, "shape": list(r.values.shape),
        "values_sha256": hashlib.sha256(r.values.tobytes()).hexdigest(),
        "observed_sha256": hashlib.sha256(r.observed.tobytes()).hexdigest()} for r in runs])
    if phase in {"pretrain", "synthetic_smoke"}:
        source = construct(config)
        save_runtime(output, source)
        source, trace = fit(source, [r for r in runs if r.role == "FIT"],
                            [r for r in runs if r.role == "SELECT"], **kwargs)
        save_model(output/"source.pt", source); save_json(output/"source_training.json", trace)
        # Actual save/load path, not only an in-memory fixture branch.
        source = load_model(config, output/"source.pt")
        cal_records = infer(source, [r for r in runs if r.role == "CAL"],
                            history=config["history"], horizon=config["horizon"], seed=config["seed"], active_components=config.get("active_components"))
        source_cal = calibrate_route(cal_records, "STAGE1", active_components=config.get("active_components"))
        save_json(output/"source_calibration_records.json", cal_records)
        save_json(output/"source_calibration.json", source_cal)
        if not synthetic:
            return
    else:
        source = load_model(config, config["checkpoint"])
        save_runtime(output, source)
        source_cal = json.loads(Path(config["source_calibration"]).read_text())
    if phase in {"adapt", "synthetic_smoke"}:
        routes = ("Z", "U", "F") if synthetic else (config["route"],)
        for route in routes:
            pool = [r for r in runs if r.domain == "C"] if synthetic else runs
            support = config.get("support")
            if synthetic:
                h = config["history"]
                support = {"K": 1, "events": [{"event_id": "synthetic_E1", "entity": "C", "run": "C_SUPPORT", "start": h+1, "end": h+3}],
                    "normal": [{"matched_event_id": "synthetic_E1", "entity": "C", "run": "C_SUPPORT_NORMAL", "start": h+1, "end": h+3}],
                    "forbidden": [{"entity": r.entity, "run": r.run, "start": r.start, "end": r.start+len(r.values)}
                                  for r in pool if r.role in {"QUERY", "NORMAL_SELECT", "NORMAL_CAL"}]}
            model, trace = adapt(source, route,
                [r for r in pool if r.role == ("SUPPORT" if route == "F" else "FIT")],
                [r for r in pool if r.role == ("NORMAL_SELECT" if route == "F" else "SELECT")], support=support, **kwargs)
            save_model(output/(route+".pt"), model); save_json(output/(route+"_training.json"), trace)
            actual_route = "Z" if route == "F" and support["K"] == 0 else route
            records = [] if actual_route == "Z" else infer(model,
                            [r for r in pool if r.role == ("NORMAL_CAL" if actual_route == "F" else "CAL")],
                            history=config["history"], horizon=config["horizon"], seed=config["seed"], event_logits=actual_route == "F", active_components=config.get("active_components"))
            calibration = ({"threshold": logit_threshold(records), "normal_support_only": True, "score": "raw_event_logit"}
                           if actual_route == "F" else calibrate_route(records, actual_route, source_cal, config.get("active_components")))
            save_json(output/(route+"_calibration.json"), calibration)
            save_json(output/(route+"_calibration_records.json"), records)
            records = infer(model, [r for r in pool if r.role == "QUERY"], history=config["history"],
                            horizon=config["horizon"], seed=config["seed"], event_logits=actual_route == "F", active_components=config.get("active_components"))
            save_json(output/(route+"_scores.json"), score_records(records, calibration, event=actual_route == "F"))
        save_json(output/"STATUS.json", {"synthetic_non_scientific_results": synthetic,
                  "routes_executed": routes, "not_evidence_of_detection_benefit": True})
    elif phase == "infer":
        calibration = json.loads(Path(config["calibration"]).read_text())
        # A saved F-K0 branch is exactly Z, including later standalone inference.
        event_mode = config["route"] == "F" and calibration.get("score") == "raw_event_logit"
        records = infer(source, [r for r in runs if r.role == "QUERY"], history=config["history"],
                        horizon=config["horizon"], seed=config["seed"], event_logits=event_mode, active_components=config.get("active_components"))
        save_json(output/"scores.json", score_records(records, calibration, event=event_mode))


def evaluate(config, output):
    """Reuse shared evaluator after raw scores exist; label input is exclusive here."""
    from ..candidate_experiment import evaluate_arrays, _clean_json
    records = json.loads(Path(config["scores"]).read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for entry in config["label_runs"]:
        labels = np.load(entry["path"], allow_pickle=False)
        rows = [r for r in records if (r["entity"], r["run"]) == (entry["entity"], entry["run"]) and r["valid"]]
        indices = np.array([r["index"]-entry.get("original_start", 0) for r in rows], dtype=np.int64)
        result = evaluate_arrays(indices, np.array([r["score"] for r in rows]), labels, config["threshold"])
        save_json(output/(entry["entity"]+"_"+entry["run"]+".json"), _clean_json(result))


def l0_approval(config_path, release_path, trusted_sha, output):
    """Metadata-only guard. Rejected requests never import Torch/query CUDA."""
    from .config import HybridV1Config
    from .provenance import OfficialSources
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not release_path or not trusted_sha:
        raise PermissionError("Root exact synthetic L0 release and externally trusted SHA required")
    raw = Path(release_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != trusted_sha:
        raise PermissionError("Trusted root release SHA mismatch")
    release = json.loads(raw)
    if (release.get("status") != "APPROVED" or release.get("approved_by") != "root"
            or release.get("operation") != "HYBRID_V1_SYNTHETIC_GPU_L0"
            or release.get("allowed_actions") != ["synthetic_gpu_l0"]
            or release.get("config_sha256") != hashlib.sha256(Path(config_path).read_bytes()).hexdigest()):
        raise PermissionError("Pending/wrong-scope/config root release")
    if (cfg.get("operation") != "HYBRID_V1_SYNTHETIC_GPU_L0"
            or HybridV1Config(**cfg["model"]) != HybridV1Config()
            or cfg.get("cases") != [{"batch_size": 1, "channels": 3, "updates": 2},
                                     {"batch_size": 1, "channels": 55, "updates": 1}]
            or cfg.get("physical_gpu") != 0 or cfg.get("device") != "cuda:0"
            or cfg["resources"] != {"wall_seconds": 600, "kill_grace_seconds": 5,
                "training_allocated_and_reserved_limit_GiB": 20,
                "evaluation_allocated_and_reserved_limit_GiB": 30,
                "stage_GPUh_limit_including_cleanup": .25, "automatic_retry": False}
            or cfg["training"] != {"kind": "huber", "active_components": ["r_tf", "r_ms", "r_pred"],
                "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .01,
                "grad_clip": 1., "precision": "float32_no_AMP", "load_checkpoint": False,
                "retain_usable_weights": False}
            or cfg["permissions"] != {"synthetic_only": True, "real_data": False,
                "checkpoint_load": False, "labels_or_AD_metrics": False, "large_download": False}):
        raise PermissionError("Synthetic L0 recipe/scope differs from the bounded request")
    for field in ("cases", "resources", "permissions", "expected_gpu_uuid", "physical_gpu"):
        if release.get(field) != cfg[field]:
            raise PermissionError("Root release must bind " + field)
    own = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    if release.get("source_sha256") != own:
        raise PermissionError("Exact shared source closure changed")
    external = OfficialSources(**cfg["sources"]).manifest()
    if release.get("external_source_sha256") != {e["path"]: e["sha256"] for e in external}:
        raise PermissionError("External author source identity changed")
    if Path(output) != Path(cfg["output"]) or Path(output).exists():
        raise PermissionError("Synthetic L0 requires the exact fresh output")
    return {"metadata_preflight": "PASS", "source_sha256": own, "external_source_manifest": external,
            "torch_imported_by_guard": False, "GPU_queried_by_guard": False}


def synthetic_gpu_l0(config, output):
    """Bounded shared-core diagnostic, no real inputs or retained checkpoint.

    This is an engineering optimizer/telemetry loop around the SAME passes and
    objective kernels, not a replacement model or scientific training recipe.
    The caller must first pass l0_approval; parent timeout/OS wait remains external.
    """
    import datetime
    import os
    import subprocess
    import time
    started = time.monotonic()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    traces, evaluations = [], []
    report = {"start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "synthetic_only": True, "weights_retained": False, "GPUh_limit_including_parent_cleanup": .25}
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config["physical_gpu"]):
            raise RuntimeError("Explicit physical GPU0 visibility required")
        inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits"], text=True, timeout=10)
        rows = [r.split(", ") for r in inventory.strip().splitlines()]
        card = next(r for r in rows if int(r[0]) == config["physical_gpu"])
        if card[1] != config["expected_gpu_uuid"] or "A40" not in card[2]:
            raise RuntimeError("Physical GPU UUID/name mismatch")
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits"], text=True, timeout=10)
        if any(r.split(",")[0].strip() == card[1] for r in processes.splitlines()):
            raise RuntimeError("Approved GPU is occupied; no switching/retry")
        save_json(output/"GPU_PRECHECK.json", {"inventory": inventory, "compute_processes": processes})
        import torch
        from .training import passes, tensor_batch
        from .objectives import ssl_objective
        from .scoring import residual_components
        from .data import window
        torch.set_num_threads(1); torch.manual_seed(config["seed"])
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        actual_uuid = str(props.uuid).removeprefix("GPU-")
        if actual_uuid != config["expected_gpu_uuid"].removeprefix("GPU-"):
            raise RuntimeError("Logical cuda0 does not map to approved physical GPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(20*1024**3/props.total_memory)
        torch.cuda.reset_peak_memory_stats()
        model = construct(config)
        save_runtime(output, model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        rng = np.random.default_rng(config["seed"])
        peak = {"allocated": 0, "reserved": 0}
        resets = []
        def begin_phase(phase, limit):
            torch.cuda.set_per_process_memory_fraction(limit*1024**3/props.total_memory)
            torch.cuda.reset_peak_memory_stats()
            resets.append({"phase": phase, "limit_GiB": limit})
        def resources(limit):
            torch.cuda.synchronize()
            allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            peak["allocated"] = max(peak["allocated"], allocated)
            peak["reserved"] = max(peak["reserved"], reserved)
            if max(allocated, reserved) > limit*1024**3 or time.monotonic()-started > 600:
                raise RuntimeError("Synthetic diagnostic resource cap exceeded")
            return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
                    "elapsed_diagnostic_seconds": time.monotonic()-started}
        def bank(name, batch, f, r):
            values = {"history": batch["history"], "history_observed": batch["observed"],
                "reconstruction_context": batch["reconstruction_context"], "reconstruction_observed": batch["reconstruction_observed"],
                "future_targets": batch["future_targets"], "future_observed": batch["future_observed"],
                "prediction": f["prediction"], "prediction_normalized": f["prediction_normalized"],
                "tf_reconstruction": r["tf_reconstruction"], "tf_normalized": r["tf_normalized"],
                "ms_by_scale": r["ms_reconstruction_by_scale"], "ms_normalized_by_scale": r["ms_normalized_by_scale"],
                "hidden_mask": r["hidden_mask"], "visible_mask": r["visible_mask"], "copy_masks": r["transform_hidden_mask"],
                "order": r["order"], "mean": r["stats"]["mean"], "scale": r["stats"]["scale"],
                "forecast_mean": f["stats"]["mean"], "forecast_scale": f["stats"]["scale"],
                "tf_valid": r["tf_valid"], "ms_valid_by_scale": r["ms_valid_by_scale"], "forecast_valid": f["valid"]}
            np.savez_compressed(output/(name+".npz"), **{k: v.detach().cpu().numpy() for k, v in values.items()},
                index=np.array([batch["index"]]), available_index=np.array([batch["available_index"]]))
        for ci, case in enumerate(config["cases"]):
            channels = case["channels"]
            values = rng.normal(size=(config["model"]["history"]+config["model"]["horizon"]+1, channels)).astype(np.float32)
            run = Run("synthetic", "case"+str(ci), "generated", "FIT", values, np.ones_like(values, bool))
            batch = tensor_batch(window(run, config["model"]["history"], config["model"]["history"], config["model"]["horizon"]), "cuda:0")
            begin_phase("train_C"+str(channels), 20)
            for step in range(case["updates"]):
                seed = config["seed"]+100*ci+step
                model.train(); optimizer.zero_grad(set_to_none=True)
                before = {n: p.detach().clone() for n, p in model.named_parameters()}
                f, r = passes(model, batch, seed)
                objective = ssl_objective(f, r, batch, "huber")
                if not bool(objective["valid_windows"].all()) or not bool(torch.isfinite(objective["total"])):
                    raise RuntimeError("Invalid/nonfinite synthetic supervision")
                objective["total"].backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                block_grad = {}
                for n, p in model.named_parameters():
                    key = n.split(".")[0]
                    if p.grad is not None:
                        block_grad[key] = max(block_grad.get(key, 0.), float(p.grad.abs().max()))
                optimizer.step()
                deltas = {}
                for n, p in model.named_parameters():
                    key = n.split(".")[0]
                    deltas[key] = max(deltas.get(key, 0.), float((p.detach()-before[n]).abs().max()))
                del before
                row = {"channels": channels, "step_in_case": step+1, "mask_seed": seed,
                    "loss": float(objective["total"].detach()), "parts": {k: float(v.detach()) for k, v in objective["parts"].items()},
                    "valid_counts": {k: v.tolist() for k, v in objective["valid_counts"].items()},
                    "grad_norm_before_clip": float(grad), "gradient_max_after_clip_by_block": block_grad,
                    "parameter_delta_max_by_block": deltas, **resources(20)}
                traces.append(row); save_json(output/("train_C%d_step%d.json"%(channels, step+1)), row)
                bank("train_C%d_step%d"%(channels, step+1), batch, f, r)
            begin_phase("eval_C"+str(channels), 30)
            model.eval()
            with torch.no_grad():
                f, r = passes(model, batch, config["seed"]+100*ci+50)
                components = residual_components(f, r, batch)
                row = {"channels": channels, "mask_seed": config["seed"]+100*ci+50,
                    "raw_residual_components": {k: v.cpu().tolist() for k, v in components["per_channel"].items()},
                    "prediction_shape": list(f["prediction"].shape), "TF_shape": list(r["tf_reconstruction"].shape),
                    "MS_by_scale_shape": list(r["ms_reconstruction_by_scale"].shape), **resources(30)}
                evaluations.append(row); save_json(output/("eval_C%d.json"%channels), row)
                bank("eval_C%d"%channels, batch, f, r)
        report.update(status="COMPLETED_PENDING_REVIEW", optimizer_updates=len(traces), traces=traces,
            evaluations=evaluations, peak_allocator_bytes_across_resets=peak, allocator_resets=resets,
            GPU_name=props.name, GPU_uuid=str(props.uuid), actual_device="cuda:0", wall_seconds=time.monotonic()-started,
            active_parameter_counts=model.parameter_counts(), no_AD_metrics_or_labels=True,
            wall_seconds_scope="L0 function from precheck; external parent OS wait is authoritative whole-process cost")
        save_json(output/"report.json", report)
    except BaseException as error:
        report.update(status="FAILED_NO_RETRY", error=repr(error), completed_traces=traces,
                      wall_seconds=time.monotonic()-started)
        save_json(output/"failure.json", report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("pretrain", "source_train", "calibrate_source", "adapt", "infer", "evaluate", "synthetic_smoke", "synthetic_gpu_l0"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release")
    parser.add_argument("--trusted-release-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    guarded = args.phase in {"synthetic_gpu_l0", "source_train", "calibrate_source"}
    if args.preflight_only and not guarded:
        parser.error("--preflight-only is restricted to the synthetic_gpu_l0 release guard")
    if (args.release or args.trusted_release_sha256) and not guarded:
        parser.error("Synthetic L0 release arguments cannot authorize another operation")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps({"phase": args.phase, "preview": True, "data_opened": False})); return
    if args.phase == "synthetic_gpu_l0":
        result = l0_approval(args.config, args.release, args.trusted_release_sha256, args.output_dir)
        if args.preflight_only:
            print(json.dumps(result)); return
        synthetic_gpu_l0(config, args.output_dir)
    elif args.phase in {"source_train", "calibrate_source"}:
        result = source_approval(args.config, args.release, args.trusted_release_sha256, args.phase, args.output_dir)
        if args.preflight_only:
            print(json.dumps(result)); return
        source_action(config, args.phase, args.output_dir)
    elif args.phase == "evaluate":
        evaluate(config, args.output_dir)
    else:
        execute(config, args.phase, args.output_dir)
    output = Path(args.output_dir)
    save_json(output/"receipt.json", {"phase": args.phase,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        "synthetic_non_scientific_results": args.phase == "synthetic_smoke"})


if __name__ == "__main__":
    main()
