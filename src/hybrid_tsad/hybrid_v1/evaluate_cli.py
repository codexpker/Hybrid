"""One CPU fixed-bank evaluation entry. All bank gates precede label I/O.

No model/data feature loader, optimizer, threshold search or inference routing.
The accepted numerical evaluator is reused without altering its definitions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

OPERATION="HYBRID_V1_PSM_Z1_FIXED_EVALUATION"
EVALUATOR_SHA="72491c52bed3b6f1787ee634ddb9c0ef95ac6eaa16f333c0b83c10e1b329a831"
STATISTICS_SHA="6c3fb03ab168ce28836433d23957cfc7767932f2953f6d8c98d4c58cbecf4b10"
STATISTICS_RECIPE=dict(enabled=True,block_points=4096,n_replicates=2000,rng_seed=99173)
LABEL_SHA="e5b6b0c716a9b3d52e3a1b06761527811f75a218790c37b8a5e806741aeb0676"
IDS={"model_seed2025","model_seed2026","model_seed2027","persistence","history_mean512"}
INPUT_SHA="566bc50117d13f58793d830c14975c55f627cab63953265d7d6ce2dbcf2b0f80"
REQUIRED_SOURCE={"hybrid_tsad/hybrid_v1/evaluate_cli.py","hybrid_tsad/hybrid_v1/evaluation.py",
    "hybrid_tsad/hybrid_v1/chunked_bank.py","hybrid_tsad/candidate_experiment.py","hybrid_tsad/config.py",
    "hybrid_tsad/engine/events.py","hybrid_tsad/engine/evaluation.py","hybrid_tsad/engine/statistics.py",
    "hybrid_tsad/engine/official_affiliation.py","hybrid_tsad/hybrid_v1/statistics.py"}
PERFORMANCE_DIRECTIONS={"/ranking/"+k:"minimum" for k in ("AP","trapezoid_PR_AUC","ROC_AUC")}
for _stream in ("strict_point","confirmed3_current"):
    _prefix="/streams/"+_stream
    PERFORMANCE_DIRECTIONS.update({_prefix+"/point/"+k:"minimum" for k in ("Precision","Recall","F1")})
    for _view in ("main_events","legacy_complete_only_events"):
        PERFORMANCE_DIRECTIONS.update({_prefix+"/"+_view+"/metrics/"+k:"minimum" for k in
            ("one_to_one_eventPrecision","one_to_one_eventRecall","one_to_one_eventF1","auxiliary_any_hit_recall")})
    PERFORMANCE_DIRECTIONS.update({_prefix+"/normal/"+k:"maximum" for k in
        ("normal_FPR","normal_alarm_fragments_per_1000_points","normal_block_any_alarm_fraction")})
    PERFORMANCE_DIRECTIONS.update({_prefix+"/affiliation/global_result/"+k:"minimum" for k in ("Aff_P","Aff_R","Aff_F1")})


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def admit(config_path,release_path,trusted):
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not release_path or not trusted:raise PermissionError("Exact root evaluation release required")
    if sha(release_path)!=trusted:raise PermissionError("Release bytes differ")
    signed=json.loads(Path(release_path).read_text())
    if (signed.get("status")!="APPROVED" or signed.get("approved_by")!="root"
        or signed.get("operation")!=OPERATION or signed.get("allowed_actions")!=["evaluate"]
        or signed.get("allow_labels") is not True or signed.get("config_sha256")!=sha(config_path)
        or cfg.get("operation")!=OPERATION):raise PermissionError("Pending/wrong evaluation release")
    if cfg.get("permissions")!={"labels":True,"evaluation":True,"training":False,"target_fit":False,"GPU":False}:
        raise PermissionError("Fixed CPU evaluation permissions required")
    if cfg.get("statistics")!=STATISTICS_RECIPE:raise PermissionError("Fixed unconditional AP statistics recipe required")
    if (cfg.get("history")!=512 or cfg.get("original_length")!=87841 or cfg.get("original_start")!=0
        or cfg.get("expected_records")!=87329 or cfg.get("stream")!=["PSM","anonymous_PSM","test_original"]):
        raise PermissionError("Frozen full PSM original axis required")
    labels=cfg["labels"]
    if (labels.get("sha256")!=LABEL_SHA or labels.get("shape")!=[87841] or labels.get("format")!="csv"
        or labels.get("drop_columns")!=["timestamp_(min)"] or labels.get("value_columns")!=["label"]):
        raise PermissionError("Unique original label version/schema required")
    own=cfg["source_sha256"];base=Path(__file__).resolve().parents[2]
    if (signed.get("source_sha256")!=own or own.get("hybrid_tsad/hybrid_v1/evaluation.py")!=EVALUATOR_SHA
        or own.get("hybrid_tsad/hybrid_v1/statistics.py")!=STATISTICS_SHA):
        raise PermissionError("Accepted independent evaluator identity required")
    if not REQUIRED_SOURCE.issubset(own):raise PermissionError("Complete evaluation helper closure required")
    for name,digest in own.items():
        path=(base/name).resolve()
        if not path.is_relative_to(base) or sha(path)!=digest:raise PermissionError("Evaluation source changed")
    # Explicit root seal is an immutable referenced document, not a new format.
    seal=cfg["raw_seal"]
    if not seal.get("sha256") or sha(seal["path"])!=seal["sha256"] or signed.get("raw_seal_sha256")!=seal["sha256"]:
        raise PermissionError("Five-bank root seal required")
    if any(e["identity"].get("input_sha256")!=INPUT_SHA for e in cfg["banks"]):raise PermissionError("One fixed original PSM input required")
    if cfg.get("affiliation_identity") is not None and sha(cfg["affiliation_identity"])!=cfg.get("affiliation_identity_sha256"):
        raise PermissionError("Exact official Aff identity required")
    if Path(cfg["output"]).exists():raise FileExistsError("Fresh evaluation output required")
    return cfg


def bank_gates(cfg):
    """Check ALL five immutable banks/calibrations before the label loader.

    Manifest/chunk hashes and contiguous counts are checked without decoding raw
    scores. evaluate_bank_arrays subsequently verifies the original raw axis.
    """
    entries=cfg["banks"]
    if len(entries)!=5 or {e["id"] for e in entries}!=IDS:raise ValueError("All3model+2baseline banks required")
    expected_range=[dict(zip(("domain","entity","run"),cfg["stream"]),start=cfg["history"],end=cfg["original_length"])]
    reports=[]
    for e in entries:
        if e["id"].startswith("model_seed"):
            if e.get("seed")!=int(e["id"][-4:]) or e.get("kind")!="model":raise ValueError("Named model seed required")
        elif e.get("kind")!="deterministic_baseline" or e.get("seed") is not None:
            raise ValueError("Deterministic baseline cannot become independent training seeds")
        directory=Path(e["directory"]);manifest=directory/"manifest.json"
        if not e.get("manifest_sha256") or sha(manifest)!=e["manifest_sha256"]:raise ValueError("Sealed manifest SHA differs")
        m=json.loads(manifest.read_text());contract=m["contract"]
        if (m.get("complete") is not True or m.get("committed_records")!=cfg["expected_records"]
            or contract.get("ranges")!=expected_range or contract.get("chunk_size")!=64
            or contract.get("confirmation_points")!=3 or contract.get("identity")!=e["identity"]):
            raise ValueError("Complete coverage/identity/clock differs")
        cal=e["calibration"]
        if sha(cal["path"])!=cal["sha256"] or cal["sha256"]!=e["identity"]["calibration_sha256"]:
            raise ValueError("Frozen source calibration SHA differs")
        c=json.loads(Path(cal["path"]).read_text())
        if contract["threshold"]!=c["threshold"] or not np.isfinite(c["threshold"]):
            raise ValueError("Bank must retain its own source-fixed threshold")
        position=0
        for chunk in m["chunks"]:
            path=directory/chunk["file"]
            if (path.parent!=directory or path.name!=chunk["file"] or chunk["start_position"]!=position
                or not 0<chunk["records"]<=64 or sha(path)!=chunk["sha256"]):
                raise ValueError("Chunk path/count/identity differs before label I/O")
            position+=chunk["records"]
        if position!=cfg["expected_records"]:raise ValueError("Incomplete committed chunks")
        reports.append(dict(id=e["id"],manifest_sha256=e["manifest_sha256"],records=position,threshold=c["threshold"]))
    return reports


def load_labels(spec):
    """The ONLY label file opener; caller must have passed all bank gates."""
    import csv
    if sha(spec["path"])!=spec["sha256"]:raise ValueError("Original label file bytes differ")
    with Path(spec["path"]).open(encoding="utf-8-sig",newline="") as f:header=next(csv.reader(f))
    names=spec["drop_columns"]+spec["value_columns"]
    if len(header)!=len(names) or set(header)!=set(names):raise ValueError("Exact label CSV header required")
    values=np.genfromtxt(spec["path"],delimiter=",",skip_header=1,usecols=[header.index("label")],ndmin=1)
    if list(values.shape)!=spec["shape"] or not np.isfinite(values).all() or not np.isin(values,[0,1]).all():
        raise ValueError("Full original finite binary labels required")
    return values.astype(np.int8)


def _leaves(value,prefix=""):
    if isinstance(value,dict):
        for k,v in value.items():yield from _leaves(v,prefix+"/"+k)
    elif isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value):yield prefix,float(value)


def aggregate(results):
    """No best-seed selection; unavailable metrics remain NA for the group."""
    selected={s:dict(ranking=results["model_seed%d"%s]["ranking"],streams=results["model_seed%d"%s]["streams"]) for s in (2025,2026,2027)}
    leaves={s:dict(_leaves(v)) for s,v in selected.items()};keys=sorted(set().union(*(set(x) for x in leaves.values())))
    summary={}
    for key in keys:
        values=[leaves[s].get(key) for s in (2025,2026,2027)]
        if any(x is None for x in values):
            summary[key]=dict(mean=None,SD=None,worst_seed=None,reason="NOT_AVAILABLE_FOR_ALL3SEEDS");continue
        # Report all numeric denominators too, but do not call counts performance.
        direction=PERFORMANCE_DIRECTIONS.get(key)
        idx=(int(np.argmin(values)) if direction=="minimum" else int(np.argmax(values))) if direction else None
        summary[key]=dict(mean=float(np.mean(values)),SD=float(np.std(values,ddof=1)),minimum=min(values),maximum=max(values),
                          worst_seed=(2025,2026,2027)[idx] if idx is not None else None,
                          worst_direction=direction or "NA_DENOMINATOR_OR_NONPERFORMANCE_FIELD")
    return dict(model_seeds=[2025,2026,2027],model_mean_SD=summary,
        deterministic_controls={k:dict(ranking=results[k]["ranking"],streams=results[k]["streams"]) for k in ("persistence","history_mean512")},
        Range_VUS="NOT_COMPUTED_BY_THIS_ENTRY; preserve numerical evaluator NA records",seed_selection=False)


def run_evaluation(cfg):
    if cfg.get("statistics")!=STATISTICS_RECIPE:raise ValueError("Frozen AP statistics cannot be switched after results")
    gates=bank_gates(cfg)  # Must precede every label SHA/header/body operation.
    labels=load_labels(cfg["labels"])
    from .evaluation import evaluate_bank_arrays,save_evaluation
    from .chunked_bank import read_records
    output=Path(cfg["output"]);output.mkdir(parents=True,exist_ok=False)
    results={};scalar_inputs={}
    for entry,gate in zip(cfg["banks"],gates):
        result=evaluate_bank_arrays(entry["directory"],labels,gate["threshold"],expected_manifest_sha256=entry["manifest_sha256"],
            original_length=cfg["original_length"],original_start=cfg["original_start"],warmup_points=cfg["history"],
            stream=cfg["stream"],affiliation_identity=cfg.get("affiliation_identity"))
        # Replay only scalar fields for the exact save_evaluation hash contract.
        rows=[(r["index"],r.get("score"),r["valid"],r["available_index"],r.get("reason")) for r in read_records(entry["directory"])]
        arrays=dict(index=np.array([r[0] for r in rows],np.int64),
            score=np.array([np.nan if r[1] is None else r[1] for r in rows],np.float64),valid=np.array([r[2] for r in rows],bool),
            available_index=np.array([r[3] for r in rows],np.int64))
        reasons=[r[4] for r in rows]
        save_evaluation(output/entry["id"],result,**arrays,full_labels=labels,invalid_reasons=reasons,
            provenance=dict(gates=gates,bank_id=entry["id"],label_file=cfg["labels"],source_sha256=cfg.get("source_sha256"),
                raw_seal=cfg.get("raw_seal"),replay_note="Scalar-bank second streaming read solely for saved-input identity; no model forward"))
        results[entry["id"]]=result
        scalar_inputs[entry["id"]]={k:arrays[k] for k in ("index","score","valid")}
        scalar_inputs[entry["id"]]["invalid_reasons"]=reasons
    # Use the five arrays already needed for save_evaluation: no third raw read.
    # Each method's full-coverage metrics above remain separate and unchanged.
    from .statistics import paired_ap_bootstrap,save_statistics
    sensitivity=paired_ap_bootstrap(scalar_inputs,labels,original_length=cfg["original_length"],
        original_start=cfg["original_start"],model_names=tuple("model_seed%d"%s for s in (2025,2026,2027)),
        baseline_names=("persistence","history_mean512"))
    stats_dir=output/"paired_AP_time_block_sensitivity"
    save_statistics(stats_dir,sensitivity,provenance=dict(raw_seal=cfg.get("raw_seal"),gates=gates,
        source_sha256=cfg.get("source_sha256"),label_file=cfg["labels"],
        scope="Common-valid original-grid paired AP sensitivity; not a score ensemble or event/Aff bootstrap"))
    summary=aggregate(results)
    summary["paired_AP_statistics"]=dict(directory=stats_dir.name,parameters=dict(STATISTICS_RECIPE),
        status=sensitivity["summary"]["status"],coverage=sensitivity["summary"]["coverage"],
        draw_counts=sensitivity["summary"]["draw_counts"],full_coverage_metrics_replaced=False,
        full_dataset_runtime_previously_measured=False)
    (output/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    (output/"BANK_GATES.json").write_text(json.dumps(gates,indent=2),encoding="utf-8")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--release");p.add_argument("--trusted-release-sha256")
    p.add_argument("--execute",action="store_true");p.add_argument("--preflight-only",action="store_true");a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,operation=OPERATION,label_or_bank_opened=False)));return
    cfg=admit(a.config,a.release,a.trusted_release_sha256)
    if a.preflight_only:
        print(json.dumps(dict(all_five_bank_gates=bank_gates(cfg),labels_opened=False)));return
    run_evaluation(cfg)


if __name__=="__main__":main()
