"""Bounded readout glue. Shared model/scoring/bank kernels stay unchanged.

Each synthetic profile, model query, source-baseline CAL or baseline query
requires its own exact root release. No label input or evaluation is supported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np

OPERATION = "HYBRID_V1_Z1_SYNTHETIC_C25_PROFILE"
PRODUCTION_ACTIONS = {"HYBRID_V1_PSM_Z_MODEL_INFER":"model_infer",
                      "HYBRID_V1_PSM_Z_BASELINE_CAL":"baseline_calibrate",
                      "HYBRID_V1_PSM_Z_BASELINE_INFER":"baseline_infer"}
NAMED_OPERATION = "HYBRID_V1_PSM_NAMED_MODEL_INFER"


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
                     horizon, seed, chunk_size=64, capture_prefix=0, resume=False, active_components=None):
    """Same pointwise iterator and same score_records; storage/time only."""
    import torch
    from .chunked_bank import ChunkedBank, iter_pointwise_infer
    from .cli import score_records
    cuda = next(model.parameters()).device.type == "cuda"
    ranges = [dict(domain=r.domain, entity=r.entity, run=r.run,
                   start=r.start+history, end=r.start+len(r.values)) for r in runs]
    prefix, prefix_rng, timings = [], None, []
    with ChunkedBank(directory, identity=identity, ranges=ranges,
                     threshold=calibration["threshold"], chunk_size=chunk_size) as bank:
        if bank.manifest["committed_records"] and not resume:
            raise PermissionError("Existing prefix needs a separate root resume release")
        if bank.manifest["complete"]:
            return dict(prefix=[],prefix_rng=None,timings=[],finalize_seconds=0.,already_complete=True)
        plan = bank.remaining_plan() if resume else None
        extra = {} if active_components is None else {"active_components": active_components}
        iterator = iter(iter_pointwise_infer(model, runs, history=history, horizon=horizon, seed=seed,index_plan=plan,**extra))
        remaining = sum(e["end"]-e["start"] for e in ranges)-bank.manifest["committed_records"]
        for index in range(remaining):
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


def common_cal_plan(plans):
    """One deterministic source grid, not three independently trained baselines."""
    if set(plans)!={2025,2026,2027}:
        raise ValueError("All three accepted source plans required")
    points=plans[2025]["CAL"]
    if any(p["CAL"]!=points for p in plans.values()):
        raise ValueError("Source CAL positions differ across seeds")
    if len(points)!=1024 or {d:sum(e["domain"]==d for e in points) for d in ("SMD","Sunspot")}!={"SMD":512,"Sunspot":512}:
        raise ValueError("Exactly512 source CAL positions per domain required")
    if any(e["role"]!="CAL" for e in points):raise ValueError("CAL role required")
    return points


def query_manifest_metadata(value):
    """No numeric CSV read: reject train/history borrowing and label aliases."""
    if len(value.get("runs",[]))!=1:raise ValueError("One original PSM test run required")
    e=value["runs"][0]
    if (any(k in e for k in ("labels","label_path")) or e.get("role")!="QUERY" or e.get("domain")!="PSM"
        or e.get("bounds")!=[0,87841] or e.get("original_start",0)!=0 or e.get("expected_shape")!=[87841,25]
        or e.get("sha256")!="566bc50117d13f58793d830c14975c55f627cab63953265d7d6ce2dbcf2b0f80"
        or e.get("format")!="csv" or e.get("has_header") is not True
        or e.get("columns")!=["feature_%d"%i for i in range(25)]):
        raise ValueError("Exact label-free PSM test/columns/original axis required")
    return e


def model_binding_metadata(cfg,acceptance):
    seed=cfg.get("source_seed")
    if seed not in (2025,2026,2027) or cfg.get("inference_seed")!=seed or cfg.get("physical_gpu")!=seed-2025 or cfg.get("device")!="cuda:0":
        raise PermissionError("Wrong named seed/GPU")
    if acceptance.get("status")!="ACCEPTED_LIMITED_SOURCE_CAL":raise PermissionError("Accepted source assets required")
    allowed={(e["seed"],e["sha256"]) for e in acceptance["assets"]}
    if any((seed,cfg["assets"][k]["sha256"]) not in allowed for k in ("checkpoint","calibration")):
        raise PermissionError("Wrong seed/checkpoint/source calibration pairing")


def named_binding_metadata(cfg, seal):
    """Read receipt/config metadata only; root's exact seal attests acceptance.

    SOURCE_CONTROL uses existing source receipt files/phase. U and Z_TCAL use
    the actual stage2_adaptation receipt schema; target CAL never aliases source.
    """
    from .config import HybridV1Config
    from .scoring import recipe
    route=cfg.get("route"); seed=cfg.get("source_seed"); assets=cfg["assets"]
    if (route not in {"SOURCE_CONTROL","U_TCAL","Z_TCAL"} or seed not in (2025,2026,2027)
            or cfg.get("inference_seed")!=seed or type(cfg.get("physical_gpu")) is not int
            or cfg["physical_gpu"] not in range(4) or cfg.get("device")!="cuda:0"
            or not isinstance(cfg.get("expected_gpu_uuid"),str) or not cfg["expected_gpu_uuid"].startswith("GPU-")
            or not isinstance(cfg.get("control_id"),str) or not cfg["control_id"]):
        raise PermissionError("Explicit named route/seed/physical GPU/UUID required")
    if seal.get("status")!="ACCEPTED_NAMED_READOUT_ASSETS" or seal.get("approved_by")!="root":
        raise PermissionError("Accepted root named-asset seal required")
    matches=[e for e in seal.get("bindings",[]) if (e.get("control_id"),e.get("route"),e.get("source_seed"))
             ==(cfg["control_id"],route,seed)]
    bound_keys={"checkpoint","calibration","training_config","training_receipt","calibration_config","calibration_receipt"}
    if route=="U_TCAL": bound_keys.add("parent_source_checkpoint")
    if (len(matches)!=1 or set(matches[0].get("assets",{}))!=bound_keys
            or any(matches[0]["assets"][k]!=assets[k] for k in bound_keys)):
        raise PermissionError("Named checkpoint/CAL/config/receipt pairing is not sealed")
    values={k:json.loads(Path(assets[k]["path"]).read_text()) for k in
            ("training_config","training_receipt","calibration_config","calibration_receipt","calibration")}
    train,trained,calcfg,calibrated,cal=[values[k] for k in
        ("training_config","training_receipt","calibration_config","calibration_receipt","calibration")]
    for actual in (train,calcfg):
        if (actual.get("model")!=cfg.get("model") or actual.get("sources")!=cfg.get("sources")
                or actual.get("seed")!=seed or actual.get("history")!=512 or actual.get("horizon")!=32
                or list(recipe(actual.get("active_components"))[0])!=list(recipe(cfg.get("active_components"))[0])):
            raise PermissionError("Actual training/CAL config model, sources, seed or recipe differs")
    settings=HybridV1Config(**cfg["model"]); active,weights=recipe(cfg.get("active_components"))
    if ((settings.history,settings.horizon)!=(512,32) or (not settings.tf_enabled and "r_tf" in active)
            or (not settings.ms_enabled and "r_ms" in active) or cal.get("active_components")!=list(active)
            or cal.get("weights")!=weights or set(cal.get("scales",{}))!=set(active)
            or not all(np.isfinite(v) and v>0 for v in cal["scales"].values())
            or not np.isfinite(cal.get("threshold",float("nan")))):
        raise PermissionError("Model branch/accepted fixed CAL recipe mismatch")
    ckpt=assets["checkpoint"]["sha256"]
    def source_train_receipt():
        return (train.get("operation")=="HYBRID_V1_SOURCE_STAGE1" and trained.get("phase")=="source_train"
            and trained.get("synthetic_non_scientific_results") is False
            and trained.get("files",{}).get("source.pt")==ckpt
            and trained.get("files",{}).get("config.json")==assets["training_config"]["sha256"])
    if route=="SOURCE_CONTROL":
        good=(source_train_receipt() and calcfg.get("operation")=="HYBRID_V1_SOURCE_STAGE1"
            and calibrated.get("phase")=="calibrate_source" and cal.get("source_domain_equal_weight") is True
            and calibrated.get("synthetic_non_scientific_results") is False and cal.get("domains")==["SMD","Sunspot"]
            and calcfg.get("checkpoint_sha256")==ckpt
            and calcfg.get("train_receipt_sha256")==assets["training_receipt"]["sha256"]
            and calibrated.get("files",{}).get("source_calibration.json")==assets["calibration"]["sha256"]
            and calibrated.get("files",{}).get("config.json")==assets["calibration_config"]["sha256"])
    else:
        mode="U" if route=="U_TCAL" else "Z_TCAL"
        good=(calcfg.get("operation")=="HYBRID_V1_STAGE2_U" and calcfg.get("scope")=="public_target"
            and calcfg.get("mode")==mode and calibrated.get("phase")=="target_calibrate"
            and calibrated.get("status")=="COMPLETED" and calibrated.get("scope")=="public_target"
            and calibrated.get("mode")==mode and calibrated.get("calibrated_checkpoint_sha256")==ckpt
            and calibrated.get("updates")==0 and calibrated.get("target_fit") is False
            and calibrated.get("query_or_labels") is False and cal.get("domains")==["PSM"]
            and calibrated.get("target_calibration_contact") is True
            and calibrated.get("calibration_route_kernel")=="U/source=False"
            and cal.get("source_domain_equal_weight") is False
            and calibrated.get("manifest_sha256")==calcfg.get("manifest_sha256")
            and calibrated.get("plans_sha256")==calcfg.get("plans_sha256"))
        if route=="U_TCAL":
            parent=assets["parent_source_checkpoint"]["sha256"]
            good=good and (train.get("operation")=="HYBRID_V1_STAGE2_U" and train.get("mode")=="U"
                and train.get("scope")=="public_target" and trained.get("status")=="COMPLETED"
                and trained.get("phase")=="target_train" and trained.get("mode")=="U"
                and trained.get("scope")=="public_target" and trained.get("target_fit") is True
                and type(trained.get("updates")) is int and trained["updates"]>0
                and trained.get("adapted_checkpoint_sha256")==ckpt
                and trained.get("source_checkpoint_sha256")==train.get("source_checkpoint_sha256")==parent
                and calcfg.get("source_checkpoint_sha256")==calibrated.get("source_checkpoint_sha256")==parent
                and calcfg.get("adapted_checkpoint_sha256")==ckpt
                and calcfg.get("adapted_receipt_sha256")==assets["training_receipt"]["sha256"]
                and trained.get("manifest_sha256")==train.get("manifest_sha256")==calcfg.get("manifest_sha256")
                and trained.get("plans_sha256")==train.get("plans_sha256")==calcfg.get("plans_sha256"))
        else:
            good=good and source_train_receipt() and calcfg.get("source_checkpoint_sha256")==ckpt and calibrated.get("source_checkpoint_sha256")==ckpt
    if not good: raise PermissionError("Source/target receipt phase, mode or checkpoint/CAL lineage mismatch")
    return dict(control_id=cfg["control_id"],route=route,source_seed=seed,
                inference_seed=cfg["inference_seed"],physical_gpu=cfg["physical_gpu"],expected_gpu_uuid=cfg["expected_gpu_uuid"],
                target_calibration_contact=route!="SOURCE_CONTROL",pure_source_only_Z=route=="SOURCE_CONTROL")


def admit_named(config_path,release_path,trusted,output):
    """Separate operation: old full-only production guard is never weakened."""
    cfg=json.loads(Path(config_path).read_text())
    if not release_path or not trusted: raise PermissionError("Exact named-model root release required")
    raw=Path(release_path).read_bytes(); release=json.loads(raw)
    if (hashlib.sha256(raw).hexdigest()!=trusted or release.get("status")!="APPROVED"
            or release.get("approved_by")!="root" or release.get("operation")!=NAMED_OPERATION
            or cfg.get("operation")!=NAMED_OPERATION or release.get("allowed_actions")!=["named_model_infer"]
            or release.get("config_sha256")!=sha(config_path)):
        raise PermissionError("Pending/changed named operation release")
    if (cfg.get("permissions")!={"query":True,"labels":False,"training":False,"threshold_fit":False,"target_fit":False}
            or any(k in cfg for k in ("labels","label_path","label_runs","evaluation_labels"))
            or (cfg.get("history"),cfg.get("horizon"),cfg.get("chunk_size"))!=(512,32,64)):
        raise PermissionError("Named readout-only permissions/H512/P32/chunk64 required")
    own={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")}
    if cfg.get("stream_source_sha256")!=own or release.get("source_sha256")!=own:
        raise PermissionError("Named stream source changed")
    required={"query_manifest","asset_seal","checkpoint","calibration","training_config","training_receipt","calibration_config","calibration_receipt"}
    if cfg.get("route")=="U_TCAL": required.add("parent_source_checkpoint")
    if set(cfg["assets"])!=required: raise PermissionError("Only exact named-model asset set is supported")
    for asset in cfg["assets"].values():
        if sha(asset["path"])!=asset["sha256"]: raise PermissionError("Named asset bytes changed")
    query_manifest_metadata(json.loads(Path(cfg["assets"]["query_manifest"]["path"]).read_text()))
    named_binding_metadata(cfg,json.loads(Path(cfg["assets"]["asset_seal"]["path"]).read_text()))
    if release.get("asset_seal_sha256")!=cfg["assets"]["asset_seal"]["sha256"]:
        raise PermissionError("Root must bind accepted asset-seal SHA explicitly")
    binding={k:cfg[k] for k in ("control_id","route","source_seed","inference_seed","physical_gpu","expected_gpu_uuid")}
    if release.get("named_binding")!=binding: raise PermissionError("Wrong independent named seed/GPU/UUID binding")
    from .provenance import OfficialSources
    external=OfficialSources(**cfg["sources"]).manifest()
    if release.get("external_source_sha256")!={e["path"]:e["sha256"] for e in external}:
        raise PermissionError("Official source identity changed")
    if cfg.get("resources")!={"wall_seconds":21595,"kill_grace_seconds":5,"evaluation_GiB":30,"automatic_retry":False}:
        raise PermissionError("Named6h/30GiB existing resource contract required")
    resume=release.get("allow_resume",False)
    if type(resume) is not bool or Path(output)!=Path(cfg["output"]) or (Path(output).exists() and not resume):
        raise PermissionError("Exact fresh output or separately signed resume required")
    return cfg,resume


def production_named(cfg,output,resume=False):
    """Existing GPU resource contract and bank kernels, with explicit lineage."""
    import os,platform,resource,torch
    from .cli import load_model,save_json,save_runtime
    from .data import load_manifest
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(cfg["physical_gpu"]):
        raise PermissionError("Signed physical-to-logical mapping required")
    props=torch.cuda.get_device_properties(0)
    if str(props.uuid).removeprefix("GPU-")!=cfg["expected_gpu_uuid"].removeprefix("GPU-") or "A40" not in props.name:
        raise PermissionError("Wrong signed A40 UUID")
    torch.cuda.set_per_process_memory_fraction(30*1024**3/props.total_memory);torch.cuda.reset_peak_memory_stats()
    out=Path(output);out.mkdir(parents=True,exist_ok=resume)
    attempt=out/"executions"/cfg["execution_release_sha256"];attempt.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();save_json(attempt/"config.json",cfg)
    binding=named_binding_metadata(cfg,json.loads(Path(cfg["assets"]["asset_seal"]["path"]).read_text()))
    save_json(attempt/"named_identity.json",dict(binding,asset_seal_sha256=cfg["assets"]["asset_seal"]["sha256"]))
    runs=load_manifest(cfg["assets"]["query_manifest"]["path"],roles={"QUERY"})
    input_sha=query_manifest_metadata(json.loads(Path(cfg["assets"]["query_manifest"]["path"]).read_text()))["sha256"]
    code_sha=hashlib.sha256(json.dumps(cfg["stream_source_sha256"],sort_keys=True).encode()).hexdigest()
    model=load_model(dict(cfg,seed=cfg["source_seed"]),cfg["assets"]["checkpoint"]["path"])
    model.eval();before=_state(model);save_runtime(attempt,model)
    calibration=json.loads(Path(cfg["assets"]["calibration"]["path"]).read_text())
    identity=dict(input_sha256=input_sha,config_sha256=cfg["execution_config_sha256"],source_sha256=code_sha,
        checkpoint_sha256=cfg["assets"]["checkpoint"]["sha256"],calibration_sha256=cfg["assets"]["calibration"]["sha256"])
    result=write_model_bank(model,runs,calibration,out/"bank",identity=identity,history=512,horizon=32,
        seed=cfg["inference_seed"],chunk_size=64,resume=resume,active_components=cfg.get("active_components"))
    if before!=_state(model): raise ValueError("Named inference mutated model tensors")
    manifest=json.loads((out/"bank/manifest.json").read_text())
    if not manifest["complete"] or manifest["committed_records"]!=87329:
        raise ValueError("Incomplete original-axis PSM bank")
    save_json(attempt/"point_times.json",result["timings"])
    save_json(attempt/"resources.json",dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        GPU_name=props.name,GPU_uuid=str(props.uuid),model_state_unchanged=True))
    save_json(attempt/"report.json",dict(status="COMPLETED_PENDING_INDEPENDENT_REVIEW",action="named_model_infer",**binding,
        bank_manifest_sha256=sha(out/"bank/manifest.json"),labels_read=False,training=False,threshold_fit=False,resume=resume,
        Python=platform.python_version(),NumPy=np.__version__,CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        function_wall_seconds=time.monotonic()-began,tail31_retained=True))


def admit_production(config_path,release_path,trusted,output):
    cfg=json.loads(Path(config_path).read_text());action=PRODUCTION_ACTIONS.get(cfg.get("operation"))
    if not release_path or not trusted:raise PermissionError("Named root production release required")
    raw=Path(release_path).read_bytes();approved=json.loads(raw)
    if (hashlib.sha256(raw).hexdigest()!=trusted or approved.get("status")!="APPROVED"
        or approved.get("approved_by")!="root" or approved.get("operation")!=cfg.get("operation")
        or not action or approved.get("allowed_actions")!=[action] or approved.get("config_sha256")!=sha(config_path)):
        raise PermissionError("Pending/wrong production release")
    expected=dict(query=action!="baseline_calibrate",source_baseline_CAL=action=="baseline_calibrate",
                  labels=False,training=False,target_fit=False,source_model_CAL=False)
    if cfg.get("permissions")!=expected or any(k in cfg for k in ("labels","label_path","label_runs","evaluation_labels")):
        raise PermissionError("Exact label-free action permissions required")
    if cfg.get("history")!=512 or cfg.get("horizon")!=32 or cfg.get("chunk_size")!=64:
        raise PermissionError("Fixed H512/P32/chunk64 required")
    own={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")}
    if cfg.get("stream_source_sha256")!=own or approved.get("source_sha256")!=own:
        raise PermissionError("Production runtime identity changed")
    for asset in cfg["assets"].values():
        if sha(asset["path"])!=asset["sha256"]:raise PermissionError("Bound asset changed")
    if Path(output)!=Path(cfg["output"]):raise PermissionError("Exact output required")
    resume=approved.get("allow_resume",False)
    if type(resume) is not bool or (Path(output).exists() and not resume):
        raise PermissionError("Existing output requires separate root resume release")
    if action=="baseline_calibrate":
        if resume:raise PermissionError("Source baseline calibration is a one-time CPU action")
        if "query_manifest" in cfg["assets"]:raise PermissionError("No query input in source CAL action")
        common_cal_plan({s:json.loads(Path(cfg["assets"]["source_plan_%d"%s]["path"]).read_text()) for s in (2025,2026,2027)})
    else:
        query_manifest_metadata(json.loads(Path(cfg["assets"]["query_manifest"]["path"]).read_text()))
    if action=="model_infer":
        acceptance=json.loads(Path(cfg["assets"]["source_CAL_acceptance"]["path"]).read_text())
        model_binding_metadata(cfg,acceptance)
        from .config import HybridV1Config
        if HybridV1Config(**cfg["model"])!=HybridV1Config():raise PermissionError("Full frozen model required")
        from .provenance import OfficialSources
        external=OfficialSources(**cfg["sources"]).manifest()
        if approved.get("external_source_sha256")!={e["path"]:e["sha256"] for e in external}:
            raise PermissionError("Fixed official source identity required")
        if cfg["resources"]!={"wall_seconds":21595,"kill_grace_seconds":5,"evaluation_GiB":30,"automatic_retry":False}:
            raise PermissionError("Named6h including grace required")
    elif cfg.get("methods")!=["persistence","history_mean512"] or cfg.get("device")!="cpu":
        raise PermissionError("Only two deterministic CPU methods required")
    elif cfg.get("resources")!={"wall_seconds":120 if action=="baseline_calibrate" else 1800,
                               "kill_grace_seconds":5,"address_space_GiB":4,"automatic_retry":False}:
        raise PermissionError("Named CPU sourceCAL120/query1800+5/AS4GiB action required")
    return cfg,resume


def write_baseline_bank(runs,method,calibration,directory,*,identity,history=512,chunk_size=64,resume=False):
    from .baselines import iter_baselines
    from .chunked_bank import ChunkedBank
    from .cli import score_records
    ranges=[dict(domain=r.domain,entity=r.entity,run=r.run,start=r.start+history,end=r.start+len(r.values)) for r in runs]
    with ChunkedBank(directory,identity=identity,ranges=ranges,threshold=calibration["threshold"],chunk_size=chunk_size) as bank:
        if bank.manifest["committed_records"] and not resume:raise PermissionError("Root resume required")
        if bank.manifest["complete"]:return
        for raw in iter_baselines(runs,method=method,history=history,index_plan=bank.remaining_plan() if resume else None):
            bank.append(score_records([raw],calibration)[0])
        bank.finalize()


def production(cfg,output,resume=False):
    """Label-free orchestration only; no optimization or new scoring formula."""
    from .cli import save_json,score_records,save_runtime,load_model
    from .data import load_manifest
    import os,platform,resource
    out=Path(output);out.mkdir(parents=True,exist_ok=resume)
    attempt=out/"executions"/cfg["execution_release_sha256"];attempt.mkdir(parents=True,exist_ok=False)
    save_json(attempt/"config.json",cfg)
    began=time.monotonic();action=PRODUCTION_ACTIONS[cfg["operation"]]
    if action=="baseline_calibrate":
        from .baselines import iter_baselines,fit_source_baseline_calibration
        points=common_cal_plan({s:json.loads(Path(cfg["assets"]["source_plan_%d"%s]["path"]).read_text()) for s in (2025,2026,2027)})
        runs=load_manifest(cfg["assets"]["source_manifest"]["path"],roles={"CAL"})
        for method in cfg["methods"]:
            records=list(iter_baselines(runs,method=method,history=512,index_plan=points))
            calibration=fit_source_baseline_calibration(records,expected_domains={"SMD","Sunspot"})
            save_json(out/(method+"_source_records.json"),records)
            save_json(out/(method+"_source_calibration.json"),calibration)
    else:
        runs=load_manifest(cfg["assets"]["query_manifest"]["path"],roles={"QUERY"})
        input_sha=query_manifest_metadata(json.loads(Path(cfg["assets"]["query_manifest"]["path"]).read_text()))["sha256"]
        code_sha=hashlib.sha256(json.dumps(cfg["stream_source_sha256"],sort_keys=True).encode()).hexdigest()
        if action=="model_infer":
            if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(cfg["physical_gpu"]):
                raise PermissionError("Fixed physical-to-logical mapping required")
            import torch
            props=torch.cuda.get_device_properties(0)
            if str(props.uuid).removeprefix("GPU-")!=cfg["expected_gpu_uuid"].removeprefix("GPU-") or "A40" not in props.name:
                raise RuntimeError("Wrong fixed A40")
            torch.cuda.set_per_process_memory_fraction(30*1024**3/props.total_memory);torch.cuda.reset_peak_memory_stats()
            model=load_model(dict(cfg,seed=cfg["source_seed"],checkpoint=cfg["assets"]["checkpoint"]["path"]),cfg["assets"]["checkpoint"]["path"])
            model.eval();before=_state(model);save_runtime(attempt,model)
            calibration=json.loads(Path(cfg["assets"]["calibration"]["path"]).read_text())
            identity=dict(input_sha256=input_sha,config_sha256=cfg["execution_config_sha256"],source_sha256=code_sha,
                checkpoint_sha256=cfg["assets"]["checkpoint"]["sha256"],calibration_sha256=cfg["assets"]["calibration"]["sha256"])
            result=write_model_bank(model,runs,calibration,out/"bank",identity=identity,history=512,horizon=32,
                                   seed=cfg["inference_seed"],chunk_size=64,resume=resume)
            assert before==_state(model),"Z inference mutated model tensors"
            save_json(attempt/"resources.json",dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
                GPU_name=props.name,GPU_uuid=str(props.uuid),model_state_unchanged=True))
            save_json(attempt/"point_times.json",result["timings"])
        else:
            for method in cfg["methods"]:
                asset=cfg["assets"][method+"_calibration"]
                calibration=json.loads(Path(asset["path"]).read_text())
                # Tagged implementation identity, explicitly NOT a trained checkpoint.
                no_model=hashlib.sha256(("NO_TRAINED_CHECKPOINT::"+method+"::"+cfg["stream_source_sha256"]["baselines.py"]).encode()).hexdigest()
                identity=dict(input_sha256=input_sha,config_sha256=cfg["execution_config_sha256"],source_sha256=code_sha,
                              checkpoint_sha256=no_model,calibration_sha256=asset["sha256"])
                write_baseline_bank(runs,method,calibration,out/method/"bank",identity=identity,chunk_size=64,resume=resume)
            save_json(attempt/"baseline_identity.json",dict(learned_parameters=0,training_seeds=0,
                checkpoint_identity="Tagged NO_TRAINED_CHECKPOINT implementation digest; not weights"))
    banks={}
    if action!="baseline_calibrate":
        directories=[out/"bank"] if action=="model_infer" else [out/m/"bank" for m in cfg["methods"]]
        for directory in directories:
            manifest=json.loads((directory/"manifest.json").read_text())
            if not manifest["complete"] or manifest["committed_records"]!=87329:raise ValueError("Incomplete full PSM coverage")
            banks[str(directory.relative_to(out))]=sha(directory/"manifest.json")
    save_json(attempt/"report.json",dict(status="COMPLETED_PENDING_INDEPENDENT_REVIEW",action=action,banks=banks,
        source_weights_or_model_scorer_changed=False,labels_read=False,training=False,resume=resume,
        Python=platform.python_version(),NumPy=np.__version__,CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        function_wall_seconds=time.monotonic()-began,clock="model information available_index; durable commits batched64, not a deployed delivery-latency claim"))


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
        print(json.dumps(dict(preview=True,supported_operations=[OPERATION]+list(PRODUCTION_ACTIONS)+[NAMED_OPERATION],model_or_data_opened=False)));return
    operation=json.loads(Path(args.config).read_text()).get("operation")
    if operation==NAMED_OPERATION:
        config,resume=admit_named(args.config,args.release,args.trusted_release_sha256,args.output_dir)
    elif operation in PRODUCTION_ACTIONS:
        config,resume=admit_production(args.config,args.release,args.trusted_release_sha256,args.output_dir)
    else:
        config=admit(args.config,args.release,args.trusted_release_sha256,args.output_dir);resume=False
    if args.preflight_only:
        print(json.dumps(dict(metadata_preflight="PASS",GPU_or_model_started=False)));return
    config["execution_config_sha256"]=sha(args.config)
    if operation==NAMED_OPERATION:
        config["execution_release_sha256"]=args.trusted_release_sha256
        production_named(config,args.output_dir,resume)
    elif operation in PRODUCTION_ACTIONS:
        config["execution_release_sha256"]=args.trusted_release_sha256
        production(config,args.output_dir,resume)
    else:
        print(json.dumps(synthetic_profile(config,args.output_dir)))

if __name__ == "__main__": main()
