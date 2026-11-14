"""Exact-release A-stage runner. Default preview; no target/U/F/public phases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

OPERATIONS = {"l0": "HYBRID_V1_REFERENCE_CENTER_A_L0",
              "source_train": "HYBRID_V1_REFERENCE_CENTER_A_SOURCE_TRAIN",
              "source_calibrate": "HYBRID_V1_REFERENCE_CENTER_A_SOURCE_CALIBRATE"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(config_path, release_path, trusted):
    cfg = json.loads(Path(config_path).read_text())
    if not release_path or not trusted or sha(release_path) != trusted:
        raise PermissionError("New exact root A-stage release required")
    signed = json.loads(Path(release_path).read_text())
    phase = cfg.get("phase")
    if (phase not in OPERATIONS or signed.get("status") != "APPROVED" or signed.get("approved_by") != "root"
            or cfg.get("status") != "ROOT_BOUND" or cfg.get("operation") != OPERATIONS[phase]
            or signed.get("operation") != cfg["operation"] or signed.get("allowed_actions") != [phase]
            or signed.get("config_sha256") != sha(config_path) or signed.get("allow_query_labels") is not False
            or signed.get("allow_target_fit") is not False or signed.get("allow_public_query") is not False):
        raise PermissionError("Wrong/pending operation; A has no public or U/F entry")
    for name, expected in cfg["source_sha256"].items():
        path = Path(cfg["runtime_source_root"])/name
        if sha(path) != expected:
            raise PermissionError("A frozen source closure changed: "+name)
    if signed.get("source_sha256") != cfg["source_sha256"]:
        raise PermissionError("Root did not bind the exact source closure")
    for path, expected in cfg["external_source_sha256"].items():
        if sha(path) != expected:
            raise PermissionError("Official dependency bytes changed")
    if sha(cfg["installed_frft"]["path"]) != cfg["installed_frft"]["sha256"]:
        raise PermissionError("Installed FRFT bytes changed")
    if Path(cfg["output_dir"]).exists():
        raise FileExistsError("Fresh output required; no automatic scientific retry")
    if cfg.get("resume") is not False or (cfg["history"], cfg["horizon"]) != (512, 32):
        raise PermissionError("A retains formal H512/P32 and no resume")
    if cfg.get("active_components") != ["r_tf", "r_ms", "r_pred"]:
        raise PermissionError("A uses the original full .25/.25/.5 score recipe")
    if phase != "l0":
        for key in ("manifest", "plans"):
            if sha(cfg[key]) != cfg[key+"_sha256"]:
                raise PermissionError("Accepted source input/plan changed")
        if cfg.get("center_mode") not in {"window", "reference"} or cfg.get("seed") not in (2025, 2026, 2027):
            raise PermissionError("Only registered two arms/three seeds")
        if phase == "source_calibrate":
            for key in ("checkpoint", "train_receipt", "training_acceptance"):
                if sha(cfg[key]) != cfg[key+"_sha256"]:
                    raise PermissionError("Own accepted training dependency changed")
    return cfg, signed


def active_lease(cfg, release_sha, config_sha):
    """A trusted supervisor must claim its fixed lease before Torch/CUDA import."""
    ledger = json.loads(Path(cfg["budget_ledger"]).read_text())
    lease = ledger["leases"][cfg["lease_id"]]
    if (lease["state"] != "RUNNING" or lease["config_sha256"] != config_sha
            or lease["release_sha256"] != release_sha or lease["owner_pid"] != os.getppid()):
        raise PermissionError("No owned live fixed-budget lease")


def construct(cfg, arm):
    import torch
    from . import model as core
    from .config import HybridV1Config
    from .provenance import OfficialSources
    from .reference_center import model_class
    torch.manual_seed(cfg["seed"])
    settings = HybridV1Config.from_dict(cfg["model"])
    factory = core.HybridV1 if arm == "legacy" else model_class(core)
    kwargs = {} if arm == "legacy" else {"center_mode": arm}
    return factory(settings, OfficialSources(**cfg["sources"]), **kwargs).to(cfg["device"])


def state_identity(model):
    result = {}
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        result[name] = {"shape": list(array.shape), "dtype": str(array.dtype),
                        "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    return result


def exact(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or not torch.equal(a, b):
            raise AssertionError("Legacy/window numerical parity differs")
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a: exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): exact(x, y)
    else:
        assert a == b


def l0(cfg, out):
    import torch
    from . import cli, model as core, training, reference_center as ref
    from .data import Run
    from .objectives import ssl_objective
    kernels = ref.training_namespace(training)
    products = []
    for channels in cfg["technical_channels"]:
        # Entirely separate technical IID process: never formal AR1 run2/3/4.
        rng = np.random.default_rng(cfg["technical_data_seed"]+channels)
        values = rng.normal(size=(1088, channels)).astype(np.float32)
        observed = np.ones(values.shape, bool)
        observed[10:14, 0] = False; values[10:14, 0] = np.nan
        run = Run("A_TECHNICAL_IID", "C"+str(channels), "NOT_FORMAL_QUERY", "FIT", values, observed)
        batch = training.tensor_batch(ref.reference_window(run, 1024, 512, 32), cfg["device"])
        models = {arm: construct(cfg, arm).eval() for arm in ("legacy", "window", "reference")}
        states = {arm: state_identity(m) for arm, m in models.items()}
        assert states["legacy"] == states["window"] == states["reference"]
        folder = out/("C"+str(channels)); folder.mkdir()
        cli.save_json(folder/"initial_state_identity.json", states)
        timings = {}
        with torch.no_grad():
            outputs = {}
            for arm, model in models.items():
                torch.cuda.synchronize(); began = time.monotonic()
                outputs[arm] = (training if arm == "legacy" else kernels).passes(model, batch, cfg["seed"])
                torch.cuda.synchronize(); timings[arm+"_cold_eval_pass_seconds"] = time.monotonic()-began
            exact(outputs["legacy"], outputs["window"])
            exact(ssl_objective(*outputs["legacy"], batch, "huber"), ssl_objective(*outputs["window"], batch, "huber"))
            f, r = outputs["reference"]
            center = ref.visible_reference_statistics(batch["reference_values"], batch["reference_observed"])["mean"]
            exact(f["stats"]["mean"], center)
            for group in range(r["hidden_mask"].shape[1]):
                _, local = core.visible_normalize(batch["reconstruction_context"], r["visible_mask"][:, group])
                exact(r["stats"]["mean"][:, group], center)
                exact(r["stats"]["scale"][:, group], local["scale"])
            cli.save_json(folder/"raw_interface.json", {"forecast": f["prediction"].cpu().tolist(),
                "forecast_stats": {k:v.cpu().tolist() for k,v in f["stats"].items()},
                "reconstruction_stats": {k:v.cpu().tolist() for k,v in r["stats"].items()},
                "tf_endpoint": r["tf_reconstruction"][:, :, :, -1].cpu().tolist(),
                "ms_endpoint": r["ms_reconstruction_by_scale"][:, :, :, :, -1].cpu().tolist(),
                "legacy_window_exact": True, "normalization_scope": "fixed_role_prefix_center_and_visible_local_scale",
                "legacy_core_scope_interpretation": "visible_student_only describes local scale, reference center is earlier fixed prefix"})
        model = models["reference"]; model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(); began = time.monotonic()
        forecast, reconstruction = kernels.passes(model, batch, cfg["seed"])
        objective = ssl_objective(forecast, reconstruction, batch, "huber")
        assert bool(objective["valid_windows"].all()) and bool(torch.isfinite(objective["total"]))
        objective["total"].backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert bool(torch.isfinite(parameter.grad).all())
                block = name.split(".")[0]
                gradients[block] = max(gradients.get(block, 0.), float(parameter.grad.abs().max()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        timings["reference_one_update_seconds"] = time.monotonic()-began
        changed = state_identity(model) != states["reference"]
        assert changed
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "NOT_SCIENTIFIC_NOT_REUSED": True}, folder/"synthetic_update_bundle.pt")
        cli.save_runtime(folder, model)
        cli.save_json(folder/"reference_provenance.json", model.reference_provenance())
        products.append({"C": channels, "timings": timings, "legacy_window_exact": True,
                         "initial_states_equal": True, "updated": changed, "gradient_max_by_block": gradients,
                         "gradient_norm": float(norm), "loss": float(objective["total"].detach()),
                         "loss_parts": {k:float(v.detach()) for k,v in objective["parts"].items()},
                         "valid_counts": {k:v.cpu().tolist() for k,v in objective["valid_counts"].items()},
                         "NOT_SCIENTIFIC_NOT_REUSED": True})
        del models, outputs, model, optimizer, forecast, reconstruction, f, r
        torch.cuda.empty_cache()
    cli.save_json(out/"report.json", {"status": "COMPLETED_REFERENCE_CENTER_SYNTHETIC_L0", "products": products,
        "formal_inputs_or_source_arrays_read": False, "checkpoint_loaded": False,
        "reference_center": "fixed_role_prefix_center_and_visible_local_scale", "scientific_results": False})


def source_action(cfg, out):
    import torch
    from . import cli, data, training, reference_center as ref
    from .scoring import fit_calibration
    phase = cfg["phase"]
    roles = {"FIT", "SELECT"} if phase == "source_train" else {"CAL"}
    runs = data.load_manifest(cfg["manifest"], roles=roles)
    if {r.domain for r in runs} != {"SMD", "Sunspot"} or any(r.role not in roles for r in runs):
        raise ValueError("Only registered source roles")
    plans = json.loads(Path(cfg["plans"]).read_text())
    if (plans["seed"], plans["history"], plans["horizon"]) != (cfg["seed"], 512, 32):
        raise ValueError("Shared matched plan identity mismatch")
    for run in runs:
        # Enforce reference support/geometry even for the matched-window arm.
        ref.reference_window(run, 1024, 512, 32, arrived=phase != "source_train")
    cli.save_json(out/"input_runs.json", [{"domain":r.domain,"entity":r.entity,"run":r.run,"role":r.role,
        "start":r.start,"shape":list(r.values.shape),"values_sha256":hashlib.sha256(r.values.tobytes()).hexdigest(),
        "observed_sha256":hashlib.sha256(r.observed.tobytes()).hexdigest()} for r in runs])
    model = construct(cfg, cfg["center_mode"])
    kernels = ref.training_namespace(training)
    cli.save_runtime(out, model)
    cli.save_json(out/"reference_provenance.json", model.reference_provenance())
    if phase == "source_train":
        if cfg["steps"] != 8000 or cfg["validate_every"] != 100 or cfg["kind"] != "huber":
            raise ValueError("Fixed source update/selection/loss budget changed")
        model, report = kernels.fit(model, [r for r in runs if r.role=="FIT"], [r for r in runs if r.role=="SELECT"],
            history=512, horizon=32, steps=8000, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            kind="huber", seed=cfg["seed"], route="STAGE1", active_components=cfg["active_components"],
            validate_every=100, sampling_plan=plans["FIT"], validation_plan=plans["SELECT"], artifact_dir=out)
        cli.save_model(out/"source.pt", model)
        cli.save_json(out/"source_training.json", report)
    else:
        model.load_state_dict(torch.load(cfg["checkpoint"], map_location=cfg["device"], weights_only=True), strict=True)
        records = list(kernels.iter_infer(model, runs, history=512, horizon=32, seed=cfg["seed"],
                          active_components=cfg["active_components"], index_plan=plans["CAL"]))
        for row in records:
            row["replay"]["reference"]["normalization_scope"] = ("visible_local_center_and_scale" if cfg["center_mode"]=="window"
                else "fixed_role_prefix_center_and_visible_local_scale")
        cli.save_json(out/"source_calibration_records.json", records)
        cli.save_json(out/"source_calibration.json", fit_calibration(records, source=True, active_components=cfg["active_components"]))


def execute(cfg, config_path):
    import torch
    from . import cli
    out = Path(cfg["output_dir"]); out.mkdir(parents=True, exist_ok=False)
    with (out/"config.json").open("xb") as handle:
        handle.write(Path(config_path).read_bytes())
    torch.set_num_threads(1)
    limit = (20 if cfg["phase"] in {"l0", "source_train"} else 30)*1024**3
    properties = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1., limit/properties.total_memory), 0)
    torch.cuda.reset_peak_memory_stats(0)
    cli.save_json(out/"GPU_START.json", {"name":properties.name,"uuid":str(properties.uuid),"allocator_limit_bytes":limit,
                                      "set_before_model_construction":True,"no_AMP":True})
    try:
        (l0 if cfg["phase"]=="l0" else source_action)(cfg, out)
    finally:
        torch.cuda.synchronize()
        cli.save_json(out/"GPU_PEAK.json", {"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(), "allocator_limit_bytes":limit})
    files = {str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()}
    cli.save_json(out/"receipt.json", {"phase":cfg["phase"],"center_mode":cfg.get("center_mode"),
        "config_sha256":sha(config_path),"source_sha256":cfg["source_sha256"],
        "files":files,"scientific_training":cfg["phase"]=="source_train","query_labels_opened":False,
        "normalization_scope": "declared center arm; always original visible local variance"})


def main():
    parser = argparse.ArgumentParser()
    for key in ("config", "release", "trusted-release-sha256"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    cfg, signed = metadata(args.config, args.release, args.trusted_release_sha256)
    if not args.execute or args.preflight_only:
        print(json.dumps({"metadata_preflight":"PASS","operation":cfg["operation"],"Torch_imported":"torch" in sys.modules,"scientific_child_started":False}))
        return
    active_lease(cfg, args.trusted_release_sha256, sha(args.config))
    execute(cfg, args.config)


if __name__ == "__main__":
    main()
