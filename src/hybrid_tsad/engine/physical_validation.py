"""Explicit original-preprocess backend and read-only execution evidence binding.

This module never selects a backend by comparing scores or trying alternatives.
CPU defaults are only for synthetic compatibility; real TEP requires provenance.
"""
from pathlib import Path
from datetime import datetime
import shlex

from ..config import read_json, sha256
from .readout_manifest import checked, record, load_readout_operation, entity_profile, validate_whitelist, native_key

CPU_BACKEND = "original_cpu_float32_v1"
CUDA_BACKEND = "original_cuda_a40_float32_v1"
CUDA_RUNTIME = {"torch_version": "2.8.0+cu128", "cuda_version": "12.8",
                "gpu_name": "NVIDIA A40", "compute_capability": [8, 6]}
ORIGINAL_A_CODE_IDENTITY_SHA = "41acaeb0c4e38948972e4c02321904335ae9e3c0330c773c11024a62261029e1"


def physical_validation_spec(value, profile):
    if value is None:
        if profile["name"] != "SYNTHETIC":
            raise ValueError("Real TEP physical validation requires an explicit approved backend")
        return {"backend": CPU_BACKEND, "device": "cpu"}
    if not isinstance(value, dict) or set(value) != {"backend", "device"}:
        raise ValueError("Physical validation requires exactly backend/device")
    backend, device = value["backend"], value["device"]
    if backend == CPU_BACKEND:
        if device != "cpu" or profile["name"] != "SYNTHETIC":
            raise ValueError("CPU default is only a synthetic engineering validation backend")
    elif backend == CUDA_BACKEND:
        if device != "cuda:0" or profile["channels"] != 52:
            raise ValueError("Original CUDA validation is fixed to cuda:0 B1x100x52 float32")
    else:
        raise ValueError("Unapproved physical validation backend; no fallback")
    return dict(value)


def validate_physical_runtime(spec):
    """Check the selected runtime before reading values; no forward or RNG call."""
    if spec["backend"] == CPU_BACKEND:
        return
    if spec["backend"] != CUDA_BACKEND:
        raise ValueError("Unapproved physical validation backend; no fallback")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("Required original CUDA validation backend unavailable; no CPU fallback")
    if (str(torch.__version__) != CUDA_RUNTIME["torch_version"]
            or torch.version.cuda != CUDA_RUNTIME["cuda_version"]
            or torch.cuda.get_device_name(spec["device"]) != CUDA_RUNTIME["gpu_name"]
            or list(torch.cuda.get_device_capability(spec["device"])) != CUDA_RUNTIME["compute_capability"]
            or torch.is_autocast_enabled("cuda")):
        raise ValueError("Physical validation runtime differs from the approved A40 float32 backend")


def bind_execution_evidence(evidence_path, shard_list_path, native_banks, catalog_path, query_path, profile, spec):
    """Validate original operation/launch/runtime records, never bank self-report.

    The concrete evidence schema is checked below. Missing or UNVERIFIED math
    provenance blocks execution. Historical OS wait-exit telemetry is not needed
    to establish which preprocessing formula produced a completed hashed bank.
    """
    if spec["backend"] == CPU_BACKEND:
        if evidence_path is not None:
            raise ValueError("Synthetic CPU validation does not accept production execution evidence")
        return None
    if evidence_path is None:
        raise ValueError("Original CUDA bank validation requires frozen execution evidence")
    return _bind_cuda_execution(evidence_path, shard_list_path, native_banks, catalog_path, query_path, profile)


def _bind_cuda_execution(evidence_path, shard_list_path, native_banks, catalog_path, query_path, profile):
    evidence_record=record(evidence_path)
    evidence=read_json(evidence_path)
    if (evidence.get("schema_version")!=1 or evidence.get("profile")!=profile
            or evidence.get("evidence_not_approval") is not True
            or evidence.get("status")!="CANDIDATE_NOT_PRIMARY_APPROVED_METADATA_DERIVATION_ONLY"):
        raise ValueError("Invalid frozen CUDA execution evidence; evidence is not an approval")
    if shard_list_path is None:raise ValueError("CUDA validation requires the original frozen shard list")
    source_root=Path(evidence["runtime_source_root"])
    identity_path=checked(evidence["original_code_identity"],evidence_path)
    if sha256(identity_path)!=ORIGINAL_A_CODE_IDENTITY_SHA:
        raise ValueError("Original inference code identity is not the audited A snapshot")
    identity=read_json(identity_path)
    expected_code={str((source_root.parent/name).resolve()):digest for name,digest in identity["files"].items()}
    actual_code={str(checked(r,evidence_path)):r["sha256"] for r in evidence["actual_runtime_files"]}
    if (len(actual_code)!=len(evidence["actual_runtime_files"]) or actual_code!=expected_code):
        raise ValueError("Executed source bytes do not match the original A identity")
    for ref in evidence["original_core_files"].values():
        if str(checked(ref,evidence_path)) not in expected_code or ref["sha256"]!=expected_code[str(checked(ref,evidence_path))]:
            raise ValueError("Original core provenance changed")
    prior=evidence["historical_versions"]["prior_frozen_record"]
    environment=read_json(checked(prior["environment"],evidence_path))
    prior_commands=read_json(checked(prior["commands"],evidence_path))
    if environment.get("torch")!=CUDA_RUNTIME["torch_version"] or environment.get("cuda")!=CUDA_RUNTIME["cuda_version"]:
        raise ValueError("Prior recorded environment does not match the approved validation runtime")
    # Prior environment is a reference, never re-labelled as per-job telemetry.
    if (evidence["historical_versions"].get("job_time_Torch")!="UNVERIFIED"
            or evidence["historical_versions"].get("job_time_CUDA")!="UNVERIFIED"):
        raise ValueError("Historical job versions must preserve the documented UNVERIFIED limitation")
    shard_refs=read_json(shard_list_path)
    expected_shards={str(checked(r,shard_list_path)):r["sha256"] for r in shard_refs}
    rows=evidence["shards"]
    if (len(expected_shards)!=len(shard_refs) or len(rows)!=len(expected_shards)
            or profile["name"]=="TEP100" and len(rows)!=5):
        raise ValueError("CUDA evidence must cover every original shard exactly once")
    entities=entity_profile(read_json(query_path),profile)
    seen_shards=set();seen_banks=set();seen_names=set()
    for row in rows:
        shard_path=checked(row["shard_manifest"],evidence_path)
        if (str(shard_path) in seen_shards or row["name"] in seen_names
                or expected_shards.get(str(shard_path))!=row["shard_manifest"]["sha256"]):
            raise ValueError("CUDA execution evidence has duplicate/foreign shard identities")
        seen_shards.add(str(shard_path));seen_names.add(row["name"])
        shard=read_json(shard_path,max_bytes=64*1024*1024)
        operation_path=checked(row["infer_operation"],evidence_path)
        cfg,paths,output,_,digest=load_readout_operation(operation_path)
        if (cfg["operation"]!="readout-infer" or cfg["parameters"]!={"profile":profile,"scope":"shard","device":"cuda:0"}
                or record(paths["catalog"])!=record(catalog_path) or record(paths["query_manifest"])!=record(query_path)
                or record(paths["approval"])!=row["approval"] or cfg["approval"]!=row["approval"]
                or record(paths["whitelist"])!=row["whitelist"] or cfg["inputs"]["whitelist"]!=row["whitelist"]
                or shard_path!=output/"shard_manifest.json"):
            raise ValueError("Original approved CUDA operation does not bind this exact shard/input/output")
        for name,ref in {"contract":cfg["contract"],"approval":cfg["approval"],**cfg["inputs"]}.items():checked(ref,operation_path)
        approval=read_json(checked(row["approval"],evidence_path))
        if (approval.get("status")!="APPROVED" or approval.get("operation_manifest_sha256")!=digest
                or approval.get("contract_sha256")!=cfg["contract"]["sha256"]):
            raise ValueError("Original inference approval does not bind its operation digest")
        if row["runtime"]!=evidence["original_code_identity"] or row["runtime_source_root"]!=str(source_root):
            raise ValueError("Launch source snapshot identity differs from the frozen code evidence")
        launch=read_json(checked(row["launch"],evidence_path))
        selector=row["launch_selector"]
        if selector!="top-level":
            if selector!="jobs."+row["name"]:raise ValueError("Unsupported launch selector")
            launch=launch["jobs"][row["name"]]
        command=launch["command"]
        if (len(command)<11 or command[:1]!=["timeout"] or command[3]!="env"
                or command[-6:]!=["-m","hybrid_tsad","readout","--config",str(operation_path),"--execute"]):
            raise ValueError("Actual launch command does not execute the approved readout operation")
        assignments=command[4:-7]
        env=dict(item.split("=",1) for item in assignments if "=" in item)
        if (len(env)!=len(assignments) or set(env)!={"CUDA_VISIBLE_DEVICES","PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS"}
                or env["PYTHONPATH"]!=str(source_root) or env["OMP_NUM_THREADS"]!="2" or env["OPENBLAS_NUM_THREADS"]!="2"
                or env["CUDA_VISIBLE_DEVICES"] not in ("0","1","2","3")
                or type(launch["GPU"]) is not int or launch["GPU"]!=int(env["CUDA_VISIBLE_DEVICES"])
                or row["physical_gpu_index_from_original_CUDA_VISIBLE_DEVICES"]!=launch["GPU"]
                or type(row["physical_gpu_index_from_original_CUDA_VISIBLE_DEVICES"]) is not int
                or command[-7] not in shlex.split(prior_commands["prepare"])
                or command[-7] not in shlex.split(prior_commands["verify"])):
            raise ValueError("Actual CUDA device/interpreter/source mapping is not bound to original launch evidence")
        if (row["backend_type"]!="cuda" or row["logical_device_from_original_operation"]!="cuda:0"
                or row["recorded_input_contract"]["shape"]!=[1,100,52]
                or row["recorded_input_contract"]["dtype"]!="float32" or row["recorded_input_contract"]["history"]!=100):
            raise ValueError("CUDA provenance input/backend contract changed")
        release_path=checked(row["release"],evidence_path);release=read_json(release_path)
        if row["release"]==row["approval"]:
            release_time=approval["approved_at_utc"]  # pilot has no separate release file
        else:
            if (release.get("status")!="REMAINING3597_RELEASED" or release["operation_digests"].get(row["name"])!=digest
                    or release.get("labels_released") is not False):
                raise ValueError("Original primary release does not cover this operation")
            release_time=release["released_at_utc"]
        if not (datetime.fromisoformat(approval["approved_at_utc"])<=datetime.fromisoformat(launch["launched_utc"])
                and datetime.fromisoformat(release_time)<=datetime.fromisoformat(launch["launched_utc"])):
            raise ValueError("Original approval/release does not precede launch")
        cells=validate_whitelist(read_json(paths["whitelist"]),entities,"shard")
        keys={native_key(c["seed"],c["mode"],c["entity"]) for c in cells}
        log=read_json(checked(row["stdout_log"],evidence_path),max_bytes=64*1024*1024)
        if (shard!=log or shard.get("status")!="SHARD_ONLY_LABELS_LOCKED" or shard.get("labels_opened") is not False
                or shard["profile"]!=profile or shard["whitelist"]!=cells or set(shard["banks"])!=keys
                or set(row["bank_keys"])!=keys or len(row["bank_keys"])!=len(keys) or row["bank_count"]!=len(keys)
                or shard["catalog_sha256"]!=sha256(catalog_path) or shard["query_manifest_sha256"]!=sha256(query_path)):
            raise ValueError("Original completion JSON/log/whitelist/bank population differs")
        metadata={r["key"]:r["metadata"] for r in row["bank_metadata_bindings"]}
        if set(metadata)!=keys or len(metadata)!=len(row["bank_metadata_bindings"]):
            raise ValueError("Execution evidence metadata coverage is incomplete/duplicate")
        for key in keys:
            if key in seen_banks or key not in native_banks:raise ValueError("Execution evidence bank population changed")
            seen_banks.add(key)
            if (metadata[key]!=shard["banks"][key]["metadata"]
                    or any(shard["banks"][key][part]!=native_banks[key][part] for part in ("bank","rng","metadata"))):
                raise ValueError("Sealed bank hashes differ from actual launched shard outputs")
    if seen_banks!=set(native_banks):raise ValueError("Execution evidence does not bind every native bank")
    if record(evidence_path)!=evidence_record:raise ValueError("Execution evidence changed during validation")
    return evidence_record
