"""Source-only public processed-domain glue; existing scientific kernels unchanged.

Real MSL, deterministic baselines and synthetic cost have separate operations.
No labels, target fitting, optimization, calibration fitting or resume interface.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .readout import (sha, named_binding_metadata, write_model_bank,
                      write_baseline_bank, _state, _rng, _restore, _rng_json)

MODEL = "HYBRID_V1_PUBLIC_SOURCE_ONLY_MODEL_READOUT"
BASELINE = "HYBRID_V1_PUBLIC_SOURCE_ONLY_BASELINE_READOUT"
PROFILE = "HYBRID_V1_PUBLIC_SYNTHETIC_C1_COST"
SCIENCE = {"checkpoint", "calibration", "training_config", "training_receipt",
           "calibration_config", "calibration_receipt"}
QUERY = {"query_manifest", "query_acceptance", "input_data"}
METHODS = ["persistence", "history_mean512"]
MSL_MANIFEST = "97d26b7fa3d9092fd78864fdfafe35ac8fbc756a9fdac808bd52c309b2b77e6d"
MSL_NPZ = "2c979d6a1259714a8ef88bf9f151cb8df2de7d7f620206819b3b7390b15195bc"
REFERENCE = "40e2247ed3049bf9d2303c380b2da55c1494c874ca38dd24652f741ed33ccc74"
FULL_SOURCE = {
    2025: ("c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812", "da239eac4c5ab1e76718cd651c48f6f89b8d5658b5b9b2a20cb400d88f83d585"),
    2026: ("591a43e7ecac6d77a1fbfa970dba078262381ec124c95be6607fe4836fbf2b93", "4464dea82b0c04cb79e9b4aec86983d5b86c1ea1846a4c87a81cc0e39ff085eb"),
    2027: ("af91f63875e75d3d4e4d0772e549cc062256fbec8ac25e353a7f198e946b7aa4", "8f23cf37e93a57890506dc9a73d8a2948b13327eee85434f3ea24b4d55f69145")}
BASELINE_CAL = {"persistence": "5513840e1454ef3cd84e373c510a336660d716d60971ad6e14b752e1c946d1cd",
                "history_mean512": "16b3a4959a08998ea2dedbf3c2373e7a95e1fae15607f21d065ce42d9825fb9d"}
SYNTHETIC = dict(rng_seed=84551, length=640, channels=1,
                 distribution="standard_normal_float32", observed="all_true")


def permissions(operation):
    return dict(query=operation != PROFILE, synthetic_profile=operation == PROFILE,
                labels=False, training=False, threshold_fit=False, target_fit=False,
                source_calibration_fit=False)


def source_identity():
    return {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}


def external_identity(cfg):
    if cfg["operation"] == BASELINE:
        return {}
    from .provenance import OfficialSources
    return {e["path"]: e["sha256"] for e in OfficialSources(**cfg["sources"]).manifest()}


def query_metadata(manifest, acceptance, assets):
    """Only the accepted processed C1 test axis; no array or label loading."""
    if (acceptance.get("status") != "ACCEPTED_LIMITED_PROCESSED_QUERY_INPUT"
            or acceptance.get("approved_by") != "root"
            or acceptance.get("dataset") != "MSL_AUTHOR_PROCESSED_COL0"
            or acceptance.get("manifest_sha256") != MSL_MANIFEST
            or acceptance.get("products", {}).get("query_inputs.npz") != MSL_NPZ
            or acceptance.get("shape") != [73729, 1]
            or acceptance.get("Run_original_start") != 0
            or acceptance.get("primary_feature_indices") != [0]
            or assets["query_manifest"]["sha256"] != MSL_MANIFEST
            or assets["input_data"]["sha256"] != MSL_NPZ):
        raise PermissionError("Accepted processed MSL C1 identity required")
    entries = manifest.get("runs", [])
    expected = dict(domain="MSL_AUTHOR_PROCESSED_COL0", entity="processed_anonymous_stream",
                    run="processed_test_axis", role="QUERY", path="query_inputs.npz",
                    format="npz", key="values", observed_key="observed", bounds=[0, 73729],
                    expected_shape=[73729, 1], sha256=MSL_NPZ, original_start=0)
    if entries != [expected]:
        raise PermissionError("Exact anonymous processed QUERY manifest required")
    relative = Path(assets["query_manifest"]["path"]).parent / expected["path"]
    if relative.resolve() != Path(assets["input_data"]["path"]).resolve():
        raise PermissionError("Manifest NPZ and bound input path differ")
    return expected


def validate_config(cfg):
    """Pure metadata checks, before reading assets or constructing any model."""
    op = cfg.get("operation")
    if op not in {MODEL, BASELINE, PROFILE} or cfg.get("permissions") != permissions(op):
        raise PermissionError("Explicit public operation and source-only permissions required")
    if (any(k in cfg for k in ("labels", "label_path", "label_runs", "evaluation_labels"))
            or cfg.get("resume") is not False
            or (cfg.get("history"), cfg.get("horizon"), cfg.get("chunk_size")) != (512, 32, 64)):
        raise PermissionError("No label/resume; fixed H512/P32/chunk64 required")
    if op == PROFILE:
        if (cfg.get("profile") != "SYNTHETIC_C1_COST128_V1" or cfg.get("synthetic") != SYNTHETIC
                or cfg.get("reference_points") != 3 or cfg.get("source_seed") != 2025
                or cfg.get("physical_gpu") != 0):
            raise PermissionError("Only fixed synthetic C1 seed2025 cost allowed")
        required = SCIENCE | {"asset_seal", "reference_training"}
        if cfg.get("resources") != dict(wall_seconds=175, kill_grace_seconds=5,
                evaluation_GiB=30, max_GPUh=.05, automatic_retry=False):
            raise PermissionError("Synthetic hard total180seconds/.05GPUh required")
    else:
        if (cfg.get("profile") != "MSL_PROCESSED_COL0_SOURCE_Z_V1" or "synthetic" in cfg
                or cfg.get("expected_records") != 73217):
            raise PermissionError("Real processed MSL must not use synthetic exemption")
        required = (SCIENCE | {"asset_seal"} | QUERY if op == MODEL else
                    QUERY | {"baseline_acceptance", "persistence_calibration", "history_mean512_calibration"})
        limits = cfg.get("resources", {})
        if (type(limits.get("wall_seconds")) is not int or limits["wall_seconds"] <= 0
                or limits.get("kill_grace_seconds") != 5 or limits.get("automatic_retry") is not False
                or (op == MODEL and limits.get("evaluation_GiB") != 30)
                or (op == BASELINE and limits.get("AS_GiB") != 8)):
            raise PermissionError("Explicit separately approved bounded public readout resources required")
    if set(cfg.get("assets", {})) != required:
        raise PermissionError("Exact operation-specific assets required")
    if op != BASELINE:
        from .config import HybridV1Config
        if (cfg.get("route") != "SOURCE_CONTROL" or cfg.get("source_seed") not in (2025, 2026, 2027)
                or cfg.get("inference_seed") != cfg["source_seed"]
                or cfg.get("device") != "cuda:0" or type(cfg.get("physical_gpu")) is not int
                or cfg["physical_gpu"] not in range(4)
                or not str(cfg.get("expected_gpu_uuid", "")).startswith("GPU-")
                or HybridV1Config(**cfg["model"]) != HybridV1Config()
                or cfg.get("active_components") != ["r_tf", "r_ms", "r_pred"]):
            raise PermissionError("Frozen full model, source seed, A40 mapping and recipe required")
        if tuple(cfg["assets"][k]["sha256"] for k in ("checkpoint", "calibration")) != FULL_SOURCE[cfg["source_seed"]]:
            raise PermissionError("Only original full three-seed own source checkpoints/CAL allowed")
    elif cfg.get("methods") != METHODS or cfg.get("device") != "cpu":
        raise PermissionError("Two deterministic CPU controls, each once, required")
    elif any(cfg["assets"][m+"_calibration"]["sha256"] != BASELINE_CAL[m] for m in METHODS):
        raise PermissionError("Original two deterministic source CALs required")
    if not cfg.get("output"):
        raise PermissionError("Exact fresh output required")
    return cfg


def admit(config_path, release_path, trusted):
    cfg = validate_config(json.loads(Path(config_path).read_text()))
    if not release_path or not trusted or sha(release_path) != trusted:
        raise PermissionError("Exact externally trusted root release required")
    release = json.loads(Path(release_path).read_text())
    op = cfg["operation"]
    if (release.get("status") != "APPROVED" or release.get("approved_by") != "root"
            or release.get("operation") != op or release.get("allowed_actions") != [op]
            or release.get("config_sha256") != sha(config_path)
            or release.get("resources") != cfg.get("resources")
            or release.get("allow_resume") is not False):
        raise PermissionError("Wrong public action/config/resource authority")
    own = source_identity()
    if cfg.get("stream_source_sha256") != own or release.get("source_sha256") != own:
        raise PermissionError("Immutable stream22 plus public adapter source required")
    if Path(cfg["output"]).exists():
        raise PermissionError("Fresh output required; no automatic resume")
    for asset in cfg["assets"].values():
        if sha(asset["path"]) != asset["sha256"]:
            raise PermissionError("Bound accepted asset bytes changed")
    if op != PROFILE:
        query_metadata(json.loads(Path(cfg["assets"]["query_manifest"]["path"]).read_text()),
                       json.loads(Path(cfg["assets"]["query_acceptance"]["path"]).read_text()), cfg["assets"])
    if op != BASELINE:
        seal = cfg["assets"]["asset_seal"]
        named_binding_metadata(cfg, json.loads(Path(seal["path"]).read_text()))
        binding = {k: cfg[k] for k in ("control_id", "route", "source_seed", "inference_seed",
                                     "physical_gpu", "expected_gpu_uuid")}
        if release.get("named_binding") != binding or release.get("asset_seal_sha256") != seal["sha256"]:
            raise PermissionError("Exact six-asset named lineage and GPU identity required")
    else:
        accepted = json.loads(Path(cfg["assets"]["baseline_acceptance"]["path"]).read_text())
        if accepted.get("approved_by") != "root" or accepted.get("status") != "ACCEPTED_LIMITED_BASELINE_SOURCE_CAL":
            raise PermissionError("Accepted source baseline CAL required")
        expected = {a["method"]: a["calibration_sha256"] for a in accepted["methods"]}
        if set(expected) != set(METHODS) or any(cfg["assets"][m+"_calibration"]["sha256"] != expected[m] for m in METHODS):
            raise PermissionError("Exact own deterministic source calibrations required")
    if op == PROFILE and cfg["assets"]["reference_training"]["sha256"] != REFERENCE:
        raise PermissionError("Actual old V4 reference required")
    if release.get("external_source_sha256") != external_identity(cfg):
        raise PermissionError("Actual official external source bytes differ")
    cfg["execution_config_sha256"] = sha(config_path)
    cfg["execution_release_sha256"] = trusted
    return cfg


def _device(cfg):
    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(cfg["physical_gpu"]):
        raise PermissionError("Signed physical-to-logical GPU mapping required")
    props = torch.cuda.get_device_properties(0)
    if "A40" not in props.name or str(props.uuid).removeprefix("GPU-") != cfg["expected_gpu_uuid"].removeprefix("GPU-"):
        raise PermissionError("Wrong A40 UUID")
    torch.cuda.set_per_process_memory_fraction(30*1024**3/props.total_memory)
    torch.cuda.reset_peak_memory_stats()
    return props


def _complete(path, count):
    manifest = json.loads((Path(path)/"manifest.json").read_text())
    if not manifest["complete"] or manifest["committed_records"] != count:
        raise ValueError("Incomplete fixed public query coverage")
    return sha(Path(path)/"manifest.json")


def execute(cfg):
    """Real source-only bank writing, invoked only after exact admission."""
    from .cli import load_model, save_json, save_runtime
    from .data import load_manifest
    op = cfg["operation"]
    if op == PROFILE:
        return synthetic_cost(cfg)
    props = _device(cfg) if op == MODEL else None
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic(); save_json(out/"config.json", cfg)
    runs = load_manifest(cfg["assets"]["query_manifest"]["path"], roles={"QUERY"})
    if len(runs) != 1 or runs[0].values.shape != (73729, 1) or runs[0].start != 0:
        raise ValueError("Actual accepted processed C1 run required")
    identity = dict(input_sha256=MSL_NPZ, config_sha256=cfg["execution_config_sha256"],
                    source_sha256=hashlib.sha256(json.dumps(cfg["stream_source_sha256"], sort_keys=True).encode()).hexdigest())
    banks = {}
    if op == MODEL:
        import torch
        model = load_model(dict(cfg, seed=cfg["source_seed"]), cfg["assets"]["checkpoint"]["path"])
        model.eval(); before = _state(model); before_rng = _rng(); save_runtime(out, model)
        cal = json.loads(Path(cfg["assets"]["calibration"]["path"]).read_text())
        identity.update(checkpoint_sha256=cfg["assets"]["checkpoint"]["sha256"],
                        calibration_sha256=cfg["assets"]["calibration"]["sha256"])
        result = write_model_bank(model, runs, cal, out/"bank", identity=identity,
            history=512, horizon=32, seed=cfg["inference_seed"], chunk_size=64, resume=False)
        if before != _state(model): raise ValueError("Inference changed model tensors")
        banks["bank"] = _complete(out/"bank", 73217)
        save_json(out/"point_times.json", result["timings"])
        save_json(out/"rng.json", dict(before=_rng_json(before_rng), after=_rng_json(_rng())))
        save_json(out/"resources.json", dict(peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(), GPU_name=props.name, GPU_uuid=str(props.uuid), model_state_unchanged=True))
    else:
        for method in METHODS:
            asset = cfg["assets"][method+"_calibration"]
            cal = json.loads(Path(asset["path"]).read_text())
            tagged = hashlib.sha256(("NO_TRAINED_CHECKPOINT::"+method+"::"+cfg["stream_source_sha256"]["baselines.py"]).encode()).hexdigest()
            item = dict(identity, checkpoint_sha256=tagged, calibration_sha256=asset["sha256"])
            write_baseline_bank(runs, method, cal, out/method/"bank", identity=item, history=512, chunk_size=64, resume=False)
            banks[method] = _complete(out/method/"bank", 73217)
    report = dict(status="COMPLETED_PENDING_INDEPENDENT_REVIEW", operation=op, banks=banks,
        points_per_bank=73217, profile=cfg["profile"], source_only=True, updates=0,
        labels_read=False, target_fit=False, threshold_fit=False, resume=False,
        model_seed=cfg.get("source_seed"), mask_seed_rule="source_seed + test_index" if op == MODEL else "deterministic; not training seeds",
        native_entity_or_physical_time_certified=False, function_wall_seconds=time.monotonic()-began)
    save_json(out/"report.json", report)
    return report


def synthetic_cost(cfg):
    """Fixed128 C1 writer points, then3 old-V4 references; restore main final RNG."""
    import torch
    import resource
    from .cli import load_model, save_json, save_runtime
    from .data import Run
    from .chunked_bank import _fingerprint
    props = _device(cfg); out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); save_json(out/"config.json", cfg)
    random.seed(2025); np.random.seed(2025)
    began = time.monotonic()
    model = load_model(dict(cfg, seed=2025), cfg["assets"]["checkpoint"]["path"])
    model.eval(); torch.cuda.synchronize(); load_seconds = time.monotonic()-began
    save_runtime(out, model)
    x = np.random.default_rng(84551).standard_normal((640, 1)).astype(np.float32)
    run = Run("SYNTHETIC_PROFILE", "C1", "rng84551", "QUERY", x, np.ones(x.shape, bool))
    cal = json.loads(Path(cfg["assets"]["calibration"]["path"]).read_text())
    identity = dict(input_sha256=hashlib.sha256(x.tobytes()+run.observed.tobytes()).hexdigest(),
        config_sha256=cfg["execution_config_sha256"], checkpoint_sha256=cfg["assets"]["checkpoint"]["sha256"],
        calibration_sha256=cfg["assets"]["calibration"]["sha256"],
        source_sha256=hashlib.sha256(json.dumps(cfg["stream_source_sha256"], sort_keys=True).encode()).hexdigest())
    before = _state(model); initial_rng = _rng(); began = time.monotonic()
    result = write_model_bank(model, [run], cal, out/"bank", identity=identity,
        history=512, horizon=32, seed=2025, chunk_size=64, capture_prefix=3, resume=False)
    main_seconds = time.monotonic()-began; final_rng = _rng(); bank_sha = _complete(out/"bank", 128)
    spec = importlib.util.spec_from_file_location("hybrid_tsad.hybrid_v1._public_cost_old_v4", cfg["assets"]["reference_training"]["path"])
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    plan = [dict(domain=run.domain, entity=run.entity, run=run.run, role="QUERY", index=t) for t in range(512, 515)]
    began = time.monotonic()
    try:
        _restore(initial_rng)
        reference = old.infer(model, [run], history=512, horizon=32, seed=2025, index_plan=plan)
        if ([_fingerprint(v) for v in reference] != [_fingerprint(v) for v in result["prefix"]]
                or _rng_json(_rng()) != _rng_json(result["prefix_rng"]) or before != _state(model)):
            raise ValueError("Fixed oldV4 raw/RNG/state parity failed")
    finally:
        _restore(final_rng)
    reference_seconds = time.monotonic()-began
    if before != _state(model): raise ValueError("Cost profile changed model tensors")
    save_json(out/"prefix_reference.json", dict(reference=reference, main=result["prefix"], exact=True))
    save_json(out/"rng.json", dict(before=_rng_json(initial_rng), after_first3=_rng_json(result["prefix_rng"]), after_main=_rng_json(final_rng)))
    save_json(out/"point_times.json", result["timings"])
    forward = np.array([p["forward_and_raw_pack_seconds"] for p in result["timings"]])
    append = np.array([p["score_and_bank_append_seconds"] for p in result["timings"]])
    amortized = (append.sum()+result["finalize_seconds"])/128
    report = dict(status="COMPLETED_PENDING_INDEPENDENT_REVIEW", main_points=128, reference_points=3,
        total_forward_points=131, seed=2025, channels=1, synthetic_only=True, real_query=False,
        labels_read=False, training=False, threshold_fit=False, bank_manifest_sha256=bank_sha,
        cold_model_load_seconds=load_seconds, main_writer_seconds=main_seconds, reference_seconds=reference_seconds,
        first_forward_seconds=float(forward[0]), steady_forward_P50=float(np.quantile(forward[1:], .5)),
        steady_forward_P95=float(np.quantile(forward[1:], .95)), steady_forward_max=float(forward[1:].max()),
        bank_amortized_seconds_per_point=float(amortized), bank_finalize_seconds=result["finalize_seconds"],
        bank_bytes=sum(p.stat().st_size for p in (out/"bank").iterdir() if p.is_file()),
        planning_points_per_model=73217, planning_three_models=219651,
        planning_2x_forward_plus_amortized_GPUh=3*(load_seconds+float(forward[0])+2*(float(forward[1:].max())+float(amortized))*73217)/3600,
        estimate_is_not_upper_bound=True, model_state_unchanged=True, raw_RNG_exact_first3=True,
        main_final_RNG_restored=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        GPU_name=props.name, GPU_uuid=str(props.uuid), function_wall_seconds=time.monotonic()-started)
    save_json(out/"report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--release")
    parser.add_argument("--trusted-release-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        validate_config(json.loads(Path(args.config).read_text()))
        print(json.dumps(dict(status="METADATA_PREVIEW_ONLY", input_arrays_opened=False, model_constructed=False)))
        return
    cfg = admit(args.config, args.release, args.trusted_release_sha256)
    print(json.dumps(execute(cfg)))


if __name__ == "__main__":
    main()
