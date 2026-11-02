"""Independent F query input/score helpers. No model loading or GPU launcher.

Real execution still needs separate root asset/data/OS releases. Old PSM
readout and F training/CAL modules are not modified or impersonated.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

DATA_OPERATION="HYBRID_V1_SMD_EVENT_QUERY_INPUTS"
READOUT_OPERATION="HYBRID_V1_SMD_EVENT_LOGIT_READOUT"
PROFILE="SMD21_F_EXPLORATORY_H512"
RAW_SHA="2d103661799271be958227907950a1c76803356442d60a4070671cc0abbefb20"
QUERY=(17129,23694)
CHOICES={"K1_choice0","K1_choice1","K1_choice2","K3_choice3"}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()


def geometry(start=17129,end=23694,history=512,horizon=32):
    if any(type(v) is not int for v in (start,end,history,horizon)) or not 0<=start<start+history<end or horizon<1:raise ValueError("Legal independent query geometry required")
    return dict(original_start=start,original_length=end-start,history=history,horizon=horizon,
        warmup=[start,start+history],scored=[start+history,end],records=end-start-history,
        mask_rule="base_seed + original_index - Run.start; role-local index",tail_retained=min(horizon-1,end-start-history))


def parse_query_rows(rows,*,total_rows,start,end,channels):
    """Text rows outside bounds are counted but NEVER numerically converted."""
    if not 0<=start<end<=total_rows or channels<1:raise ValueError("Explicit suffix shape required")
    selected=[];count=0
    for i,row in enumerate(rows):
        count=i+1
        if start<=i<end:
            if len(row)!=channels:raise ValueError("Native query channel count changed")
            selected.append(row)
    if count!=total_rows or len(selected)!=end-start:raise ValueError("Original row count/bounds changed")
    values=np.asarray(selected,dtype=np.float32)
    if np.isinf(values).any():raise ValueError("Existing Run forbids Inf; never repair silently")
    return values,np.isfinite(values)


def _write(path,value):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,allow_nan=False)


def prepare_inputs(cfg):
    """Root-guarded caller only. No label path, no prefix NPZ, one QUERY role."""
    raw=cfg["raw_csv"];out=Path(cfg["output"])
    if out.exists():raise FileExistsError("Fresh query input output required")
    if sha(raw["path"])!=raw["sha256"]:raise PermissionError("Original whole-file byte identity changed")
    with Path(raw["path"]).open(newline="",encoding="utf-8-sig") as f:
        values,observed=parse_query_rows(csv.reader(f),total_rows=raw["expected_shape"][0],start=cfg["query_bounds"][0],end=cfg["query_bounds"][1],channels=raw["expected_shape"][1])
    start,end=cfg["query_bounds"];out.mkdir(parents=True,exist_ok=False)
    npz=out/"query_values_observed.npz"
    np.savez_compressed(npz,values=values,observed=observed,original_index=np.arange(start,end,dtype=np.int64))
    run=dict(domain="SMD",entity="machine-2-1",run="machine-2-1_test",role="QUERY",path=str(npz.resolve()),sha256=sha(npz),
        format="npz",key="values",observed_key="observed",expected_shape=list(values.shape),bounds=[0,len(values)],original_start=start)
    manifest=dict(runs=[run],source_raw=dict(path=raw["path"],sha256=raw["sha256"],numeric_bounds=[start,end]),geometry=geometry(start,end,cfg["history"],cfg["horizon"]))
    _write(out/"query_manifest.json",manifest)
    receipt=dict(status="COMPLETED_PENDING_ACCEPTANCE",operation=DATA_OPERATION,labels_read=False,prefix_numeric_converted=False,raw_byte_SHA_verified=True,
        numeric_bounds=[start,end],shape=list(values.shape),manifest_sha256=sha(out/"query_manifest.json"),NPZ_sha256=sha(npz),model_or_fit=False)
    _write(out/"receipt.json",receipt);return receipt


def _no_annotation(value):
    if isinstance(value,dict):
        if any("label" in k.lower() or "ledger" in k.lower() for k in value):raise PermissionError("No annotation assets in query input metadata")
        for v in value.values():_no_annotation(v)
    elif isinstance(value,list):
        for v in value:_no_annotation(v)


def input_admit(config,release,trusted):
    cfg=json.loads(Path(config).read_text())
    if not release or not trusted or sha(release)!=trusted:raise PermissionError("Exact root query-input release required")
    signed=json.loads(Path(release).read_text());expected=dict(query_values=True,labels=False,prefix_F_values=False,model=False,GPU=False)
    if (cfg.get("operation")!=DATA_OPERATION or signed.get("operation")!=DATA_OPERATION or signed.get("status")!="APPROVED" or signed.get("approved_by")!="root"
            or signed.get("allowed_actions")!=["prepare_smd_event_query"] or signed.get("config_sha256")!=sha(config) or cfg.get("permissions")!=expected):raise PermissionError("Wrong/pending query-only input release")
    _no_annotation({k:v for k,v in cfg.items() if k!="permissions"})
    raw=cfg["raw_csv"]
    if (raw.get("sha256")!=RAW_SHA or raw.get("expected_shape")!=[23694,38] or raw.get("has_header") is not False or cfg.get("query_bounds")!=list(QUERY)
            or cfg.get("history")!=512 or cfg.get("horizon")!=32):raise PermissionError("Frozen native suffix query contract changed")
    if any("label" in str(v).lower() for v in (raw["path"],cfg["output"])):raise PermissionError("No label path alias")
    base=Path(__file__).resolve().parents[2];own=cfg["source_sha256"]
    if signed.get("source_sha256")!=own or "hybrid_tsad/hybrid_v1/event_query.py" not in own:raise PermissionError("Complete exact source binding required")
    for name,digest in own.items():
        f=(base/name).resolve()
        if base not in f.parents or f.suffix!=".py" or sha(f)!=digest:raise PermissionError("Query source closure changed")
    return cfg


def event_binding_metadata(cfg,documents,seal):
    """Pure metadata proof; does not hash/open PT or load a model. Not a release."""
    if (cfg.get("operation")!=READOUT_OPERATION or cfg.get("profile")!=PROFILE or cfg.get("score_kind")!="raw_event_logit" or cfg.get("choice") not in CHOICES
            or cfg.get("source_seed")!=2025 or cfg.get("inference_seed")!=2025 or cfg.get("history")!=512 or cfg.get("horizon")!=32):raise PermissionError("Explicit fixed F readout required")
    if seal!={"status":"ACCEPTED_SMD_EVENT_QUERY_ASSETS","approved_by":"root","assets":cfg["assets"],"choice":cfg["choice"],"score_kind":"raw_event_logit"}:raise PermissionError("Exact accepted F asset seal required")
    s,t,tc,cr,cal=[documents[k] for k in ("source_config","F_train_receipt","F_cal_config","F_cal_receipt","calibration")]
    if t.get("status")!="COMPLETED" or t.get("phase")!="target_event_train" or cr.get("status")!="COMPLETED" or cr.get("phase")!="target_event_calibrate":raise ValueError("F phases cannot masquerade as source/U")
    if (t.get("loaded_roles")!=["NORMAL_SELECT","SUPPORT"] or t.get("query_or_labels") is not False or s.get("seed")!=2025
            or cfg["assets"]["parent_source_checkpoint"]["sha256"]!="c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812"
            or cfg["assets"]["checkpoint"]["sha256"]==cfg["assets"]["parent_source_checkpoint"]["sha256"]):raise ValueError("Independent F adaptation/parent/roles required")
    for k in ("choice","source_checkpoint_sha256","manifest_sha256","plans_sha256","support_sha256"):
        if t.get(k)!=cr.get(k) or t.get(k)!=tc.get(k):raise ValueError("F CAL/training/parent lineage differs")
    for digest in [t.get(k) for k in ("source_checkpoint_sha256","manifest_sha256","plans_sha256","support_sha256","adapted_checkpoint_sha256")]+[cfg["assets"]["training_receipt"].get("sha256")]:
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):raise ValueError("Actual accepted F asset SHA cannot be pending")
    if (t["choice"]!=cfg["choice"] or t.get("adapted_checkpoint_sha256")!=cfg["assets"]["checkpoint"]["sha256"]
            or tc.get("adapted_checkpoint_sha256")!=t["adapted_checkpoint_sha256"] or tc.get("adapted_receipt_sha256")!=cfg["assets"]["training_receipt"]["sha256"]
            or t["source_checkpoint_sha256"]!=cfg["assets"]["parent_source_checkpoint"]["sha256"]
            or tc.get("model")!=s.get("model") or cfg.get("model")!=s.get("model")):raise ValueError("Actual F checkpoint/source architecture binding differs")
    if (cr.get("updates")!=0 or cr.get("loaded_roles")!=["NORMAL_CAL"] or cr.get("query_or_labels") is not False
            or cal.get("score_kind")!="event_logit" or cal.get("q")!=.99 or cal.get("method")!="higher" or cal.get("strict_greater") is not True
            or cr.get("score_kind")!="event_logit" or cr.get("threshold")!=cal.get("threshold") or not np.isfinite(cal.get("threshold",np.nan))):raise ValueError("Only own independent normal raw-logit CAL supported")
    return dict(score_kind="raw_event_logit",choice=cfg["choice"],threshold=cal["threshold"],seed_rule="2025 + original_index - 17129",geometry=geometry())


def event_scored_record(raw,threshold,feature_dim):
    features=np.asarray(raw.get("event_features"),dtype=float)
    if features.shape!=(1,feature_dim):raise ValueError("Exact full event feature shape required")
    value=raw.get("event_logit",float("nan"));features_ok=bool(np.isfinite(features).all());logit_ok=isinstance(value,(float,np.floating)) and bool(np.isfinite(value))
    valid=features_ok and logit_ok
    result=dict(raw,score=float(value) if valid else float("nan"),valid=valid,reason=None if valid else ("NONFINITE_EVENT_FEATURES" if not features_ok else "NONFINITE_EVENT_LOGIT"),score_kind="raw_event_logit")
    if valid:result["alarm"]=bool(value>threshold)
    else:result.pop("alarm",None)
    return result


def write_event_bank(model,runs,calibration,directory,*,identity,history=512,horizon=32,seed=2025,feature_dim=259,infer_fn=None):
    """Storage/scoring helper only; real caller must enforce root/PT/resource/state guards."""
    from .chunked_bank import ChunkedBank,iter_pointwise_infer
    runs=list(runs)
    if len(runs)!=1 or runs[0].role!="QUERY":raise PermissionError("One independently sliced QUERY Run only")
    if calibration.get("score_kind")!="event_logit" or calibration.get("method")!="higher" or calibration.get("q")!=.99 or calibration.get("strict_greater") is not True or not np.isfinite(calibration["threshold"]):raise ValueError("F CAL is not a residual/source threshold")
    if Path(directory).exists():raise FileExistsError("Fresh bank; no implicit retry/resume")
    ranges=[dict(domain=r.domain,entity=r.entity,run=r.run,start=r.start+history,end=r.start+len(r.values)) for r in runs]
    with ChunkedBank(directory,identity=identity,ranges=ranges,threshold=calibration["threshold"],chunk_size=64) as bank:
        rows=iter_pointwise_infer(model,runs,history=history,horizon=horizon,seed=seed,event_logits=True,active_components=["r_tf","r_ms","r_pred"],infer_fn=infer_fn)
        for raw in rows:
            start=next(r.start for r in runs if (r.domain,r.entity,r.run)==tuple(raw[k] for k in ("domain","entity","run")))
            if raw.get("available_index")!=raw["index"] or raw.get("replay",{}).get("actual_mask_seed")!=seed+raw["index"]-start:raise ValueError("F original index/role-local mask clock differs")
            bank.append(event_scored_record(raw,calibration["threshold"],feature_dim))
        bank.finalize()


def main(argv=None):
    p=argparse.ArgumentParser(description="Query-only data builder; no model execution CLI exists here")
    p.add_argument("--config",required=True);p.add_argument("--execute",action="store_true");p.add_argument("--release");p.add_argument("--trusted-release-sha256");a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,numeric_or_model_read=False)));return
    cfg=input_admit(a.config,a.release,a.trusted_release_sha256);print(json.dumps(prepare_inputs(cfg)))


if __name__=="__main__":main()
