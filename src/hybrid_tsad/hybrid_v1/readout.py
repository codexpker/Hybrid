"""Bounded readout glue. Shared model/scoring/bank kernels stay unchanged.

The current CLI authorizes only the named synthetic C25 cost profile. The
stream writer is reusable for a later separately authorized public readout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np

OPERATION = "HYBRID_V1_Z1_SYNTHETIC_C25_PROFILE"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_run(spec):
    from .data import Run
    if spec != {"rng_seed": 84551, "length": 640, "channels": 25,
                "distribution": "standard_normal_float32", "observed": "all_true"}:
        raise ValueError("Only the fixed synthetic C25 profile input is supported")
    x = np.random.default_rng(spec["rng_seed"]).standard_normal((640, 25)).astype(np.float32)
    return Run("SYNTHETIC_PROFILE", "C25", "rng84551", "QUERY", x, np.ones(x.shape, bool))


def admit(config_path, release_path, trusted, output):
    """Metadata guard before Torch/model/input generation; no query alias."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not release_path or not trusted:
        raise PermissionError("Exact root profile release required")
    raw = Path(release_path).read_bytes(); release = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != trusted or release.get("status") != "APPROVED"
            or release.get("approved_by") != "root" or release.get("operation") != OPERATION
            or release.get("allowed_actions") != ["synthetic_profile"]
            or release.get("config_sha256") != sha(config_path)):
        raise PermissionError("Pending/wrong root profile release")
    if (config.get("operation") != OPERATION or config.get("source_seed") != 2025
            or config.get("inference_seed") != 2025 or config.get("device") != "cuda:0"
            or config.get("physical_gpu") != 0 or config.get("history") != 512
            or config.get("horizon") != 32 or config.get("chunk_size") != 64
            or config.get("reference_points") != 3
            or config.get("permissions") != {"synthetic_profile": True, "query": False,
                "labels": False, "training": False, "source_calibration_fit": False}):
        raise PermissionError("Fixed profile shape/permissions changed")
    synthetic = config["synthetic"]
    if synthetic != {"rng_seed": 84551, "length": 640, "channels": 25,
                     "distribution": "standard_normal_float32", "observed": "all_true"}:
        raise PermissionError("Synthetic input changed")
    if config.get("resources") != {"wall_seconds":120,"kill_grace_seconds":5,"evaluation_GiB":30,
                                   "max_GPUh":.035,"min_disk_free_GiB":10,"automatic_retry":False}:
        raise PermissionError("Fixed profile resource request changed")
    if Path(output) != Path(config["output"]) or Path(output).exists():
        raise PermissionError("Exact fresh profile output required")
    own = {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}
    if config["stream_source_sha256"] != own or release.get("source_sha256") != own:
        raise PermissionError("Readout source closure changed")
    for asset in config["assets"].values():
        if sha(asset["path"]) != asset["sha256"]:
            raise PermissionError("Accepted asset identity changed")
    reference = config["assets"]["reference_training"]
    if reference["sha256"] != "40e2247ed3049bf9d2303c380b2da55c1494c874ca38dd24652f741ed33ccc74":
        raise PermissionError("Actual frozen V4 reference required")
    from .config import HybridV1Config
    from .provenance import OfficialSources
    if HybridV1Config(**config["model"]) != HybridV1Config():
        raise PermissionError("Default full model required")
    external = OfficialSources(**config["sources"]).manifest()
    if release.get("external_source_sha256") != {a["path"]: a["sha256"] for a in external}:
        raise PermissionError("Official source identity changed")
    return config


def _state(model):
    return {k: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in model.state_dict().items()}


def _rng():
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state().clone(), cuda=[s.clone() for s in torch.cuda.get_rng_state_all()])


def _restore(snapshot):
    import torch
    random.setstate(snapshot["python"]); np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["cpu"]); torch.cuda.set_rng_state_all(snapshot["cuda"])


def _rng_json(snapshot):
    n = snapshot["numpy"]
    return dict(python=repr(snapshot["python"]), numpy=[n[0], n[1].tolist(), *n[2:]],
                torch_cpu_hex=snapshot["cpu"].numpy().tobytes().hex(),
                torch_cuda_hex=[v.cpu().numpy().tobytes().hex() for v in snapshot["cuda"]])


def write_model_bank(model, runs, calibration, directory, *, identity, history,
                     horizon, seed, chunk_size=64, capture_prefix=0):
    """Same pointwise iterator and same score_records; storage/time only."""
    import torch
    from .chunked_bank import ChunkedBank, iter_pointwise_infer
    from .cli import score_records
    cuda = next(model.parameters()).device.type == "cuda"
    ranges = [dict(domain=r.domain, entity=r.entity, run=r.run,
                   start=r.start+history, end=r.start+len(r.values)) for r in runs]
    iterator = iter(iter_pointwise_infer(model, runs, history=history, horizon=horizon, seed=seed))
    prefix, prefix_rng, timings = [], None, []
    with ChunkedBank(directory, identity=identity, ranges=ranges,
                     threshold=calibration["threshold"], chunk_size=chunk_size) as bank:
        for index in range(sum(e["end"]-e["start"] for e in ranges)):
            began = time.monotonic(); record = next(iterator)
            if cuda: torch.cuda.synchronize()
            forward_seconds = time.monotonic()-began
            if index < capture_prefix:
                prefix.append(record)
                if index == capture_prefix-1: prefix_rng = _rng()
            began = time.monotonic()
            bank.append(score_records([record], calibration)[0])
            timings.append(dict(index=record["index"], forward_and_raw_pack_seconds=forward_seconds,
                                score_and_bank_append_seconds=time.monotonic()-began))
        if next(iterator, None) is not None:
            raise ValueError("Iterator exceeded declared arrived coverage")
        began = time.monotonic(); bank.finalize(); finalize_seconds = time.monotonic()-began
    return dict(prefix=prefix, prefix_rng=prefix_rng, timings=timings, finalize_seconds=finalize_seconds)


def synthetic_profile(config, output):
    import importlib.util
    import os
    import resource
    import shutil
    import torch
    from .cli import construct, save_json, save_runtime
    from .chunked_bank import _fingerprint
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise PermissionError("Fixed physical GPU0 mapping required")
    output = Path(output)
    if shutil.disk_usage(output.parent).free < 10*1024**3:
        raise RuntimeError("At least10GiB free disk required")
    output.mkdir(exist_ok=False)
    started = time.monotonic(); props = torch.cuda.get_device_properties(0)
    if "A40" not in props.name or str(props.uuid).removeprefix("GPU-") != config["expected_gpu_uuid"].removeprefix("GPU-"):
        raise RuntimeError("Wrong fixed A40")
    torch.cuda.set_per_process_memory_fraction(30*1024**3/props.total_memory)
    torch.cuda.reset_peak_memory_stats()
    save_json(output/"config.json", config)
    random.seed(config["source_seed"]); np.random.seed(config["source_seed"])
    began = time.monotonic(); model = construct(dict(config, seed=config["source_seed"]))
    state = torch.load(config["assets"]["checkpoint"]["path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True); model.eval(); del state
    torch.cuda.synchronize(); model_load_seconds = time.monotonic()-began
    save_runtime(output, model)
    calibration = json.loads(Path(config["assets"]["calibration"]["path"]).read_text())
    run = synthetic_run(config["synthetic"])
    identity = dict(input_sha256=hashlib.sha256(run.values.tobytes()+run.observed.tobytes()).hexdigest(),
        config_sha256=config["execution_config_sha256"], checkpoint_sha256=config["assets"]["checkpoint"]["sha256"],
        calibration_sha256=config["assets"]["calibration"]["sha256"],
        source_sha256=hashlib.sha256(json.dumps(config["stream_source_sha256"],sort_keys=True).encode()).hexdigest())
    before_state = _state(model); before_rng = _rng(); began = time.monotonic()
    result = write_model_bank(model, [run], calibration, output/"bank", identity=identity,
        history=512, horizon=32, seed=config["inference_seed"], chunk_size=64, capture_prefix=3)
    main_seconds = time.monotonic()-began; after_rng = _rng()
    assert before_state == _state(model), "Main inference changed model tensors"
    # Compare after the cold main path, then restore the exact main final RNG.
    spec = importlib.util.spec_from_file_location("hybrid_tsad.hybrid_v1._profile_old_v4",
                                               config["assets"]["reference_training"]["path"])
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    plan = [dict(domain=run.domain,entity=run.entity,run=run.run,role=run.role,index=t) for t in range(512,515)]
    began = time.monotonic()
    try:
        _restore(before_rng)
        reference = old.infer(model,[run],history=512,horizon=32,seed=config["inference_seed"],index_plan=plan)
        assert [_fingerprint(v) for v in reference] == [_fingerprint(v) for v in result["prefix"]]
        assert _rng_json(_rng()) == _rng_json(result["prefix_rng"]), "Old/new prefix RNG differs"
        assert before_state == _state(model), "Reference inference changed model tensors"
    finally:
        _restore(after_rng)
    reference_seconds = time.monotonic()-began
    save_json(output/"prefix_reference.json", dict(reference=reference,main=result["prefix"],exact=True))
    save_json(output/"rng.json", {k:_rng_json(v) for k,v in (("before",before_rng),("after_first3",result["prefix_rng"]),("after_main",after_rng))})
    save_json(output/"point_times.json", result["timings"])
    seconds = np.array([r["forward_and_raw_pack_seconds"]+r["score_and_bank_append_seconds"] for r in result["timings"]])
    forward = np.array([r["forward_and_raw_pack_seconds"] for r in result["timings"]])
    bank_amortized = (sum(r["score_and_bank_append_seconds"] for r in result["timings"])+result["finalize_seconds"])/128
    bank_bytes = sum(p.stat().st_size for p in (output/"bank").iterdir() if p.is_file())
    report = dict(status="COMPLETED_PENDING_INDEPENDENT_REVIEW",main_points=128,old_reference_points=3,
        total_pass_pairs=131,raw_exact_first3=True,model_state_unchanged=True,RNG_exact_first3_and_main_final_restored=True,
        first_point_seconds=float(seconds[0]),steady_P50_seconds=float(np.quantile(seconds[1:],.5)),
        steady_P95_seconds=float(np.quantile(seconds[1:],.95)),steady_max_seconds=float(seconds[1:].max()),
        model_load_seconds=model_load_seconds,main_with_bank_seconds=main_seconds,reference_seconds=reference_seconds,
        bank_finalize_seconds=result["finalize_seconds"],bank_bytes=bank_bytes,bytes_per_main_point=bank_bytes/128,
        estimated_three_seed_full_points=3*87329,
        planning_2x_max_GPUh=3*(model_load_seconds+2*float(seconds[1:].max())*87329)/3600,
        planning_2x_max_forward_plus_amortized_bank_GPUh=3*(model_load_seconds+float(seconds[0])+
            2*(float(forward[1:].max())+bank_amortized)*87329)/3600,
        steady_forward_P95_seconds=float(np.quantile(forward[1:],.95)),bank_amortized_seconds_per_point=bank_amortized,
        planning_2x_bank_bytes=2*bank_bytes/128*3*87329,
        estimate_is_not_throughput_upper_bound=True,real_input_missing_or_IO_may_differ=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        function_wall_seconds=time.monotonic()-started,GPU_name=props.name,GPU_uuid=str(props.uuid),
        no_optimizer_or_query_or_labels=True,source_runtime="old frozen V4 produced weights; accepted stream version for this profile")
    save_json(output/"report.json",report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--release");p.add_argument("--trusted-release-sha256")
    p.add_argument("--execute",action="store_true");p.add_argument("--preflight-only",action="store_true")
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(preview=True,only_supported_operation=OPERATION,model_or_data_opened=False)));return
    config=admit(args.config,args.release,args.trusted_release_sha256,args.output_dir)
    if args.preflight_only:
        print(json.dumps(dict(metadata_preflight="PASS",GPU_or_model_started=False)));return
    config["execution_config_sha256"]=sha(args.config)
    print(json.dumps(synthetic_profile(config,args.output_dir)))

if __name__ == "__main__": main()
