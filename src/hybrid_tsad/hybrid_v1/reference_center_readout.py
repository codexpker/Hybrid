"""Fixed Stage-A reference-center mechanism readout; no training or metrics.

Uses the accepted center model/training namespace, original point iterator,
controlled-wave same-forward anchor proxy/scorer and immutable bank writer.
This is a separate entry: the running source runner and its26 files stay intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import controlled_wave as wave
from . import controlled_mechanism as original


OPERATION="HYBRID_V1_REFERENCE_CENTER_A_FIXED_MECHANISM_READOUT"
ACTIVE=["r_tf","r_ms","r_pred"]
KINDS=("N","LEVEL","NORMAL_DRIFT","REFERENCE_POLLUTED_LEVEL")
ANCHORS=(1024,1151,1279,1280,1535,1791,1792,2175,2559,2560,2815,3071)
RESOURCES=dict(total_seconds_including_cleanup=840,TERM_seconds=835,cleanup_seconds=5,
    evaluation_allocator_GiB=30,automatic_retry=False,max_live_A_GPU_jobs=4)
DEPENDENCIES=("selected_checkpoint","training_config","training_receipt","training_acceptance",
    "own_source_calibration","calibration_config","calibration_receipt","calibration_acceptance",
    "input_manifest","input_receipt","input_acceptance")


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    with Path(path).open("x",encoding="utf-8") as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write("\n")


def cases_for_seed(seed):
    if type(seed) is not int or seed not in (2025,2026,2027):
        raise ValueError("One fixed process/model seed pair required")
    run=seed-2023
    return ["run%d_%s"%(run,kind) for kind in KINDS]


def arrived_plan(run):
    return [dict(domain=run.domain,entity=run.entity,run=run.run,role=run.role,
        index=run.start+t,footprint=[run.start+t-512,run.start+t+1])
        for t in range(1024,len(run.values))]


class SamePassArchiveModel(wave.AnchorModel):
    """Observe already computed reference stats while its context is live.

    Original AnchorModel owns all forecast/reconstruction calls, fixed route
    anchors and RNG checks. This extension copies an existing tiny tensor tree;
    it adds no normalizer, statistical estimate, forward or RNG draw.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reference_pending=None

    def reconstruct(self,*args,**kwargs):
        result=super().reconstruct(*args,**kwargs)
        if self.reference_pending is not None:
            raise ValueError("Previous same-pass metadata was not consumed")
        reference=self.model._reference_bound
        if reference is None:raise ValueError("Reference must be captured within the original live context")
        actual=int(kwargs["mask_seed"])
        self.reference_pending=dict(index=self.run.start+actual-self.seed,
            actual_mask_seed=actual,stats=original.detached(reference))
        return result

    def add_reference(self,record):
        metadata=self.reference_pending
        if metadata is None or metadata["index"]!=record["index"] or metadata["actual_mask_seed"]!=record["replay"]["actual_mask_seed"]:
            raise ValueError("Same-pass reference export/record index mismatch")
        record["replay"]["same_pass_reference_stats"]=metadata["stats"]
        record["replay"]["reference"]["normalization_scope"]=("visible_local_center_and_scale"
            if self.model.center_mode=="window" else "fixed_role_prefix_center_and_visible_local_scale")
        self.reference_pending=None
        return record

    def finish(self,expected):
        super().finish(expected)
        if self.reference_pending is not None:raise ValueError("Unconsumed same-pass stats")


def archive_run(model,run,directory,*,identity,calibration,seed,training_namespace,kernels,
                wanted=ANCHORS,rng=wave.neural_rng,same_rng=wave.same_neural_rng,
                anchor_validator=original.validate_lag_axes):
    """One real iterator over one original run; small tests inject fake kernels."""
    from .chunked_bank import iter_pointwise_infer
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    def sink(payload):
        anchor_validator(payload["forecast_lag"],model.config.to_dict(),run.values.shape[1])
        for axes in payload["reconstruction_lag"]:
            anchor_validator(axes,model.config.to_dict(),run.values.shape[1])
        original.save_bundle(directory,"ANCHOR_%d"%payload["index"],payload)
    proxy=SamePassArchiveModel(model,run,wanted,seed,sink,rng,same_rng)
    original.save_bundle(directory,"INITIAL_RNG",original.detached(rng()))
    plan=arrived_plan(run)
    rows=iter_pointwise_infer(proxy,[run],history=512,horizon=32,seed=seed,
        active_components=ACTIVE,index_plan=plan,infer_fn=training_namespace.iter_infer)
    def observed_rows():
        for record in rows:yield proxy.add_reference(record)
    spans=[dict(domain=run.domain,entity=run.entity,run=run.run,
                start=run.start+1024,end=run.start+len(run.values))]
    product=wave.consume_records(observed_rows(),directory/"bank",identity=identity,
        ranges=spans,calibration=calibration,active=ACTIVE,kernels=kernels)
    proxy.finish(len(plan))
    original.save_bundle(directory,"FINAL_RNG",original.detached(rng()))
    dump(directory/"ARCHIVE_IDENTITY.json",dict(case=run.run,original_start=run.start,
        scored_range=[run.start+1024,run.start+len(run.values)],records=len(plan),
        anchor_indices=list(wanted),forecast_calls=proxy.forecasts,reconstruct_calls=proxy.reconstructions,
        extra_forward_calls=0,reference_stats_from_original_live_context=True,
        route_capture="original same-pass AnchorModel at fixed archive indices; no extra inference"))
    return product


def bound_json(asset):
    if set(asset)!={"path","sha256"} or sha(asset["path"])!=asset["sha256"]:
        raise PermissionError("Accepted exact dependency identity required")
    return json.loads(Path(asset["path"]).read_text(encoding="utf-8"))


def check_dependencies(cfg,metadata):
    train=metadata["training_config"];trained=metadata["training_receipt"]
    calcfg=metadata["calibration_config"];calibrated=metadata["calibration_receipt"]
    cal=metadata["own_source_calibration"];assets=cfg["assets"]
    for current,phase in ((train,"source_train"),(calcfg,"source_calibrate")):
        if (current.get("phase")!=phase or current.get("center_mode")!=cfg["center_mode"]
                or current.get("seed")!=cfg["source_seed"] or current.get("history")!=512
                or current.get("horizon")!=32 or current.get("model")!=cfg["model"]
                or current.get("sources")!=cfg["sources"] or current.get("active_components")!=ACTIVE
                or current.get("external_source_sha256")!=cfg["external_source_sha256"]
                or current.get("installed_frft")!=cfg["installed_frft"]
                or any(cfg["source_sha256"].get(k)!=v for k,v in current.get("source_sha256",{}).items())):
            raise PermissionError("Own arm/seed/source scientific configuration mismatch")
    if (trained.get("phase")!="source_train" or calibrated.get("phase")!="source_calibrate"
            or trained.get("center_mode")!=cfg["center_mode"] or calibrated.get("center_mode")!=cfg["center_mode"]
            or trained.get("config_sha256")!=assets["training_config"]["sha256"]
            or calibrated.get("config_sha256")!=assets["calibration_config"]["sha256"]
            or trained.get("files",{}).get("source.pt")!=assets["selected_checkpoint"]["sha256"]
            or calibrated.get("files",{}).get("source_calibration.json")!=assets["own_source_calibration"]["sha256"]
            or calcfg.get("checkpoint_sha256")!=assets["selected_checkpoint"]["sha256"]
            or calcfg.get("train_receipt_sha256")!=assets["training_receipt"]["sha256"]
            or cal.get("valid_windows")!=1024):
        raise PermissionError("Only the accepted source-SELECT selected PT and its own CAL pairing")
    from . import scoring
    wave.validate_calibration(cal,ACTIVE,scoring)
    receipt=metadata["input_receipt"];manifest=metadata["input_manifest"]
    expected=[name for seed in (2025,2026,2027) for name in cases_for_seed(seed)]
    protocol=receipt.get("protocol",{})
    if ([r.get("run") for r in manifest.get("runs",[])]!=expected
            or [r.get("case_id") for r in receipt.get("inputs",[])]!=expected
            or receipt.get("input_count")!=12
            or receipt.get("input_manifest_sha256")!=assets["input_manifest"]["sha256"]
            or protocol.get("N")!=3072 or protocol.get("reference")!=[0,512]
            or protocol.get("scored")!=[1024,3072] or protocol.get("event")!=[1280,2560]
            or protocol.get("steady")!=[1792,2560] or protocol.get("run_seed_pairs")!=[[2,2025],[3,2026],[4,2027]]):
        raise PermissionError("Root-accepted exact twelve Stage-A original case axes required")
    for entry,row in zip(manifest["runs"],receipt["inputs"]):
        if (entry.get("sha256")!=row.get("sha256") or entry.get("path")!=row.get("path")
                or entry.get("expected_shape")!=[3072,1] or entry.get("bounds")!=[0,3072]
                or entry.get("original_start")!=0 or entry.get("role")!="QUERY"
                or "labels" in entry or "label_path" in entry):
            raise PermissionError("Input identity/clock/role changed")


def admit(config_path,release_path,trusted):
    cfg=json.loads(Path(config_path).read_text());signed=json.loads(Path(release_path).read_text())
    permissions=dict(mechanism_query=True,public_query=False,labels=False,training=False,
        calibration_fit=False,input_generation=False,evaluation=False,resume=False)
    if (sha(release_path)!=trusted or cfg.get("status")!="ROOT_BOUND"
            or signed.get("status")!="APPROVED" or signed.get("approved_by")!="root"
            or cfg.get("operation")!=OPERATION or signed.get("operation")!=OPERATION
            or signed.get("allowed_actions")!=["fixed_mechanism_readout"]
            or signed.get("config_sha256")!=sha(config_path)
            or cfg.get("permissions")!=permissions or signed.get("permissions")!=permissions
            or cfg.get("resources")!=RESOURCES or signed.get("resources")!=RESOURCES
            or cfg.get("source_seed")!=cfg.get("inference_seed")
            or cfg.get("case_ids")!=cases_for_seed(cfg.get("source_seed"))
            or cfg.get("center_mode") not in ("window","reference")
            or cfg.get("history")!=512 or cfg.get("horizon")!=32 or cfg.get("active_components")!=ACTIVE
            or cfg.get("anchor_indices")!=list(ANCHORS) or cfg.get("device")!="cuda:0"
            or type(cfg.get("physical_gpu")) is not int or cfg["physical_gpu"] not in range(4)
            or not isinstance(cfg.get("expected_gpu_uuid"),str)
            or cfg.get("output_dir")!=signed.get("output_dir") or Path(cfg["output_dir"]).exists()
            or signed.get("source_sha256")!=cfg.get("source_sha256")
            or signed.get("assets")!=cfg.get("assets")
            or signed.get("accepted_dependency_sha256")!={k:cfg["assets"][k]["sha256"] for k in DEPENDENCIES}):
        raise PermissionError("Exact accepted one-method/seed mechanism readout release required")
    if set(cfg["assets"])!=set(DEPENDENCIES):raise PermissionError("No extra/label assets admitted")
    source_root=Path(cfg["runtime_source_root"])
    if Path(__file__).resolve()!=(source_root/"hybrid_tsad/hybrid_v1/reference_center_readout.py").resolve():
        raise PermissionError("Actual imported reader differs from frozen runtime")
    for name,digest in cfg["source_sha256"].items():
        if sha(source_root/name)!=digest:raise PermissionError("Frozen reader/scientific source mismatch")
    for path,digest in cfg["external_source_sha256"].items():
        if sha(path)!=digest:raise PermissionError("Official bytes mismatch")
    if sha(cfg["installed_frft"]["path"])!=cfg["installed_frft"]["sha256"]:
        raise PermissionError("Installed FRFT bytes mismatch")
    # PT is only hashed here; never deserialized by a metadata or test call.
    checkpoint=cfg["assets"]["selected_checkpoint"]
    if sha(checkpoint["path"])!=checkpoint["sha256"]:raise PermissionError("Selected PT identity mismatch")
    metadata={key:bound_json(asset) for key,asset in cfg["assets"].items() if key!="selected_checkpoint"}
    check_dependencies(cfg,metadata)
    return cfg,metadata


def execute(cfg,metadata,*,config_path):
    # The existing per-job live lease must precede Torch/CUDA import.
    from . import reference_center_cli as source_cli
    source_cli.active_lease(cfg,cfg["execution_release_sha256"],sha(config_path))
    if os.environ.get("CUDA_VISIBLE_DEVICES")!=str(cfg["physical_gpu"]):
        raise PermissionError("Fixed physical GPU mapping required")
    import torch
    from . import cli,data,training,scoring,chunked_bank,reference_center
    properties=torch.cuda.get_device_properties(0)
    if "A40" not in properties.name or str(properties.uuid).removeprefix("GPU-")!=cfg["expected_gpu_uuid"].removeprefix("GPU-"):
        raise PermissionError("Fresh expected A40 UUID required")
    torch.set_num_threads(1);limit=30*1024**3
    torch.cuda.set_per_process_memory_fraction(min(1.,limit/properties.total_memory),0)
    torch.cuda.reset_peak_memory_stats(0)
    out=Path(cfg["output_dir"]);out.mkdir(parents=True,exist_ok=False)
    with (out/"CONFIG.json").open("xb") as handle:handle.write(Path(config_path).read_bytes())
    try:
        all_runs=data.load_manifest(cfg["assets"]["input_manifest"]["path"],roles={"QUERY"})
        runs={r.run:r for r in all_runs}
        if len(runs)!=12 or any(r.values.shape!=(3072,1) or r.start!=0 or not r.observed.all() for r in all_runs):
            raise ValueError("Complete original all-observed twelve controlled inputs required")
        model=source_cli.construct(dict(cfg,seed=cfg["source_seed"]),cfg["center_mode"])
        model.load_state_dict(torch.load(cfg["assets"]["selected_checkpoint"]["path"],map_location=cfg["device"],weights_only=True),strict=True)
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        before=source_cli.state_identity(model);dump(out/"MODEL_STATE_BEFORE.json",before)
        cli.save_runtime(out,model);dump(out/"reference_provenance.json",model.reference_provenance())
        namespaces=reference_center.training_namespace(training)
        cal=metadata["own_source_calibration"]
        input_entries={r["run"]:r for r in metadata["input_manifest"]["runs"]}
        source_digest=hashlib.sha256(json.dumps(cfg["source_sha256"],sort_keys=True).encode()).hexdigest()
        products=[]
        for case_id in cfg["case_ids"]:
            run=runs[case_id]
            identity=dict(input_sha256=input_entries[case_id]["sha256"],config_sha256=sha(config_path),
                checkpoint_sha256=cfg["assets"]["selected_checkpoint"]["sha256"],
                calibration_sha256=cfg["assets"]["own_source_calibration"]["sha256"],source_sha256=source_digest)
            products.append(archive_run(model,run,out/case_id,identity=identity,calibration=cal,
                seed=cfg["inference_seed"],training_namespace=namespaces,kernels=(cli,scoring,chunked_bank)))
        after=source_cli.state_identity(model)
        dump(out/"MODEL_STATE_AFTER.json",after)
        if before!=after:raise AssertionError("Frozen model state changed")
        dump(out/"RECEIPT.json",dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT",operation=OPERATION,
            center_mode=cfg["center_mode"],source_seed=cfg["source_seed"],products=products,
            bank_count=4,scalar_points=8192,source_selected_PT_and_own_CAL=True,
            source_checkpoint=cfg["assets"]["selected_checkpoint"],calibration=cfg["assets"]["own_source_calibration"],
            training=False,labels_opened=False,metrics=False,extra_forward_calls=0))
    finally:
        torch.cuda.synchronize()
        dump(out/"GPU_PEAK.json",dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),allocator_limit_bytes=limit))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--config",required=True)
    parser.add_argument("--release");parser.add_argument("--trusted-release-sha256")
    parser.add_argument("--execute",action="store_true");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="ROOT_PENDING_PREVIEW",assets_opened=False,Torch_imported="torch" in sys.modules,query=False)));return
    if not args.release or not args.trusted_release_sha256:raise PermissionError("Exact root readout release required")
    cfg,metadata=admit(args.config,args.release,args.trusted_release_sha256)
    cfg=dict(cfg,execution_release_sha256=args.trusted_release_sha256)
    execute(cfg,metadata,config_path=args.config)


if __name__=="__main__":main()
