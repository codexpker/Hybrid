"""Independent F event-logit execution. No fitting, labels or PSM route alias."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from . import event_query as helper

OPERATION=helper.READOUT_OPERATION
ACTION="event_model_infer"
ASSETS={"checkpoint","training_config","training_receipt","calibration","calibration_config","calibration_receipt",
        "parent_source_checkpoint","source_config","source_receipt","training_acceptance","calibration_acceptance",
        "query_manifest","query_input_acceptance"}


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,allow_nan=False)


def query_metadata(manifest):
    helper._no_annotation(manifest)
    if len(manifest.get("runs",[]))!=1:raise PermissionError("One independent QUERY run required")
    r=manifest["runs"][0]
    expected=dict(domain="SMD",entity="machine-2-1",run="machine-2-1_test",role="QUERY",format="npz",key="values",observed_key="observed",expected_shape=[6565,38],bounds=[0,6565],original_start=17129)
    if any(r.get(k)!=v for k,v in expected.items()) or manifest.get("geometry")!=helper.geometry():raise PermissionError("Exact native38 suffix query required; no prefix or borrowed warmup")
    if manifest.get("source_raw",{}).get("sha256")!=helper.RAW_SHA or manifest["source_raw"].get("numeric_bounds")!=[17129,23694]:raise PermissionError("Original raw/suffix identity differs")
    return r


def preflight(config,release,trusted):
    if not release or not trusted or helper.sha(release)!=trusted:raise PermissionError("Exact root event-query release required")
    cfg=read(config);signed=read(release)
    if (signed.get("status")!="APPROVED" or signed.get("approved_by")!="root" or signed.get("operation")!=OPERATION or cfg.get("operation")!=OPERATION
            or signed.get("allowed_actions")!=[ACTION] or signed.get("config_sha256")!=helper.sha(config) or signed.get("allow_labels") is not False):raise PermissionError("Pending/wrong F-only readout signature")
    if cfg.get("permissions")!={"query":True,"labels":False,"training":False,"threshold_fit":False,"target_fit":False}:raise PermissionError("Zero-update label-free query only")
    helper._no_annotation({k:v for k,v in cfg.items() if k!="permissions"})
    if set(cfg["assets"])!=ASSETS or cfg.get("chunk_size")!=64 or cfg.get("active_components")!=["r_tf","r_ms","r_pred"]:raise PermissionError("Exact F source/PT/CAL/input acceptance closure required")
    if Path(cfg["output"]).exists():raise FileExistsError("Fresh F query output; no resume/retry")
    base=Path(__file__).resolve().parents[2];own=cfg["source_sha256"]
    required={"event_readout","event_query","event_adaptation","event_data","target_data","chunked_bank","readout","training","cli","config","data","model","temporal","fractional","lag","multiview","objectives","scoring","provenance","__init__"}
    if signed.get("source_sha256")!=own or not {"hybrid_tsad/hybrid_v1/"+x+".py" for x in required}<=set(own):raise PermissionError("Complete immutable F/readout source closure required")
    pinned={"event_query.py":"327214150a7300d26d7691184064a175b874e6b7d6cf8b915444b4b5e8ad805d","event_adaptation.py":"e85ac61006aa85e48e7ba027656ef781e71961f3b431445d60685aa60e97f653","chunked_bank.py":"17a298cb5f7255ae9d429630ec4f07152a65c7711a2657bba1e6de6d9ec4d2b4"}
    for name,digest in own.items():
        p=(base/name).resolve()
        if base not in p.parents or p.suffix!=".py" or helper.sha(p)!=digest or (p.name in pinned and digest!=pinned[p.name]):raise PermissionError("F scientific/readout source identity changed")
    seal=cfg["asset_seal"]
    if helper.sha(seal["path"])!=seal["sha256"] or signed.get("asset_seal_sha256")!=seal["sha256"]:raise PermissionError("Exact root F asset seal required")
    # Hash every asset, including PT and query NPZ, without loading tensors/arrays.
    for asset in cfg["assets"].values():
        if "label" in Path(asset["path"]).name.lower():raise PermissionError("Annotation file cannot be a model/query asset")
        if not asset.get("sha256") or helper.sha(asset["path"])!=asset["sha256"]:raise PermissionError("Changed accepted F asset")
    a=cfg["assets"];docs={key:read(a[name]["path"]) for key,name in {"source_config":"source_config","F_train_receipt":"training_receipt","F_cal_config":"calibration_config","F_cal_receipt":"calibration_receipt","calibration":"calibration"}.items()}
    binding=helper.event_binding_metadata(cfg,docs,read(seal["path"]))
    source_receipt=read(a["source_receipt"]["path"]);traincfg=read(a["training_config"]["path"])
    if source_receipt.get("phase")!="source_train" or source_receipt.get("files",{}).get("source.pt")!=a["parent_source_checkpoint"]["sha256"] or source_receipt["files"].get("config.json")!=a["source_config"]["sha256"]:raise PermissionError("Actual parent source receipt/config mismatch")
    for key in ("model","sources","history","horizon","active_components"):
        if cfg.get(key)!=docs["source_config"].get(key) or traincfg.get(key)!=cfg.get(key) or docs["F_cal_config"].get(key)!=cfg.get(key):raise PermissionError("Actual full F architecture/recipe changed")
    for key in ("choice","source_checkpoint_sha256","manifest_sha256","plans_sha256","support_sha256"):
        if traincfg.get(key)!=docs["F_train_receipt"].get(key):raise PermissionError("Actual F saved training lineage differs")
    for key in ("training_acceptance","calibration_acceptance"):
        accepted=read(a[key]["path"])
        if accepted.get("approved_by")!="root" or not str(accepted.get("status","")).startswith("ACCEPTED"):raise PermissionError("F training and own CAL must be root accepted")
        case=next((x for x in accepted.get("cases",[]) if x.get("choice")==cfg["choice"]),{}) if "cases" in accepted else accepted
        if key=="training_acceptance":
            if case.get("adapted_checkpoint_sha256",case.get("checkpoint_sha256"))!=a["checkpoint"]["sha256"]:raise PermissionError("Root training acceptance names a different F checkpoint/choice")
        elif case.get("calibration_sha256")!=a["calibration"]["sha256"] or case.get("receipt_sha256")!=a["calibration_receipt"]["sha256"]:raise PermissionError("Root CAL acceptance names a different own calibration/receipt")
    m=read(a["query_manifest"]["path"]);r=query_metadata(m);qa=read(a["query_input_acceptance"]["path"])
    if (qa.get("approved_by")!="root" or qa.get("status")!="ACCEPTED_SMD_EVENT_QUERY_INPUTS" or qa.get("manifest_sha256")!=a["query_manifest"]["sha256"]
            or qa.get("NPZ_sha256")!=r["sha256"] or qa.get("numeric_bounds")!=[17129,23694] or qa.get("labels_read") is not False or helper.sha(r["path"])!=r["sha256"]):raise PermissionError("Accepted query-only derivation required")
    from .provenance import OfficialSources
    official=OfficialSources(**cfg["sources"]).manifest()
    if signed.get("external_source_sha256")!={x["path"]:x["sha256"] for x in official}:raise PermissionError("Official source identity changed")
    return cfg,signed,docs["calibration"],binding


def infer_loaded(model,runs,calibration,output,*,identity,history,horizon,seed):
    """Shared execution spine, also used by separately signed synthetic Torch test."""
    from .event_adaptation import state
    from .readout import _rng,_rng_json
    out=Path(output);out.mkdir(parents=True,exist_ok=False);model.eval()
    before=state(model);flags={n:p.requires_grad for n,p in model.named_parameters()};rng_before=_rng_json(_rng())
    write(out/"state_before.json",before);write(out/"RNG_before.json",rng_before);write(out/"parameter_flags_before.json",flags)
    helper.write_event_bank(model,runs,calibration,out/"bank",identity=identity,history=history,horizon=horizon,seed=seed,feature_dim=2*model.config.width+3)
    after=state(model);rng_after=_rng_json(_rng())
    after_flags={n:p.requires_grad for n,p in model.named_parameters()}
    write(out/"state_after.json",after);write(out/"RNG_after.json",rng_after);write(out/"parameter_flags_after.json",after_flags)
    if before!=after or flags!=after_flags:raise RuntimeError("Inference changed model parameters/buffers/flags")
    m=read(out/"bank/manifest.json")
    receipt=dict(status="COMPLETED_PENDING_RAW_ACCEPTANCE",operation=OPERATION,score_kind="raw_event_logit",updates=0,labels_read=False,
        records=m["committed_records"],bank_manifest_sha256=helper.sha(out/"bank/manifest.json"),model_state_unchanged=True,RNG_transition_recorded=True,RNG_unchanged=rng_before==rng_after)
    write(out/"receipt.json",receipt);return receipt


def execute(config,release,trusted):
    cfg,signed,cal,binding=preflight(config,release,trusted)
    lifecycle=Path(signed["lifecycle"]);cap_path=lifecycle/"GPU_CAP_BEFORE_MODEL.json"
    if os.environ.get("HYBRID_F_QUERY_CHILD")!=trusted or not cap_path.is_file():raise PermissionError("Owned capped resource child required")
    cap=read(cap_path)
    canonical=lambda v:str(v)[4:] if str(v).startswith("GPU-") else str(v)
    if (cap.get("release_sha256")!=trusted or cap.get("config_sha256")!=signed["config_sha256"] or cap.get("limit_bytes")!=30*1024**3
            or cap.get("set_before_model_construction") is not True or cap.get("physical_gpu")!=cfg["physical_gpu"]
            or canonical(cap.get("GPU_uuid"))!=canonical(cfg["expected_gpu_uuid"])):raise PermissionError("Verified pre-model 30GiB UUID-bound cap required")
    from .cli import load_model,save_runtime
    from .data import load_manifest
    a=cfg["assets"];weights={k:helper.sha(a[k]["path"]) for k in ("checkpoint","parent_source_checkpoint")}
    model=load_model(dict(cfg,seed=cfg["source_seed"]),a["checkpoint"]["path"])
    runs=load_manifest(a["query_manifest"]["path"],roles={"QUERY"})
    if len(runs)!=1 or runs[0].start!=17129 or runs[0].values.shape!=(6565,38):raise PermissionError("Loaded query-only axes changed")
    code_sha=hashlib.sha256(json.dumps(cfg["source_sha256"],sort_keys=True).encode()).hexdigest()
    identity=dict(input_sha256=helper.RAW_SHA,config_sha256=signed["config_sha256"],checkpoint_sha256=a["checkpoint"]["sha256"],calibration_sha256=a["calibration"]["sha256"],source_sha256=code_sha)
    result=infer_loaded(model,runs,cal,cfg["output"],identity=identity,history=512,horizon=32,seed=2025)
    out=Path(cfg["output"]);save_runtime(out,model);write(out/"config.json",cfg);write(out/"binding.json",binding)
    if any(helper.sha(a[k]["path"])!=v for k,v in weights.items()):raise RuntimeError("Source or F checkpoint file changed")
    if result["records"]!=6053:raise RuntimeError("Full6053 F query required, no tail truncation")
    write(out/"weights_post.json",dict(sha256=weights,unchanged=True));write(out/"execution_identity.json",dict(config_sha256=signed["config_sha256"],release_sha256=trusted,source_sha256=cfg["source_sha256"],asset_seal=cfg["asset_seal"]))
    (out/"event_readout.py").write_bytes(Path(__file__).read_bytes());return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("config","release","trusted-release-sha256"):p.add_argument("--"+k)
    p.add_argument("--execute",action="store_true");p.add_argument("--preflight-only",action="store_true");a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,model_or_data_opened=False)));return
    if a.preflight_only:preflight(a.config,a.release,a.trusted_release_sha256);print(json.dumps(dict(metadata_PASS=True,arrays_or_model_loaded=False)));return
    print(json.dumps(execute(a.config,a.release,a.trusted_release_sha256)))


if __name__=="__main__":main()
