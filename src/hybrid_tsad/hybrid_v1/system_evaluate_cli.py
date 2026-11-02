"""Independent W1 seven-method fixed-bank profile; no model/inference/fit.

Accepted source-six/target-three modules and statistical definitions stay byte-identical.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from . import named_evaluate_cli as named
from .bank_recalibration import bound,sha,bank_metadata,code_gate
from .scoring import recipe
scalar_inputs=named.scalar_inputs
point_pair=named.point_pair
paired_events=named.paired_events
load_labels=named.load_labels
STATISTICS_NA=named.STATISTICS_NA
OPERATION="HYBRID_V1_SYSTEM_W1_FIXED_BANK_EVALUATION"
PROFILE="PSM_SYSTEM_W1_SEED2025"
CONTROLS={"tf_off_seed2025":"tf_off","causal_transformer_off_seed2025":"causal_transformer_off","self_lag_off_seed2025":"self_lag_off","cross_lag_off_seed2025":"cross_lag_off"}
FLAGS={"full":{},"tf_off":{"tf_enabled":False},"causal_transformer_off":{"temporal_enabled":False},"self_lag_off":{"self_lag_enabled":False},"cross_lag_off":{"cross_lag_enabled":False}}


def contract(cfg):
    if cfg.get("profile") not in {PROFILE,"SYNTHETIC_SYSTEM_W1"} or cfg.get("statistics")!=STATISTICS_NA:raise ValueError("Explicit W1 profile and honest NA statistics required")
    if any(k in cfg for k in ("model_seeds","seed_groups","model_mean_SD","ensemble","aggregate")):raise ValueError("Methods are not seeds/ensemble")
    entries=cfg["banks"];ids=[e["id"] for e in entries];expected=set(CONTROLS)|{"full_seed2025"}|named.BASELINES
    if len(entries)!=7 or set(ids)!=expected or len({str(Path(e["directory"]).resolve()) for e in entries})!=7:raise ValueError("Seven distinct fixed W1 banks required")
    for e in entries:
        baseline=e["id"] in named.BASELINES;control=e["id"] if baseline else ("full" if e["id"]=="full_seed2025" else CONTROLS[e["id"]])
        active=["r_pred"] if baseline else (["r_ms","r_pred"] if control=="tf_off" else ["r_tf","r_ms","r_pred"])
        if (e.get("control_id")!=control or e.get("calibration_origin")!="source" or e.get("active_components")!=active
                or e.get("kind")!=("deterministic_baseline" if baseline else "source_model")
                or e.get("training_seed")!=(None if baseline else 2025) or e.get("inference_seed")!=(None if baseline else 2025)
                or (not baseline and e.get("model_flags")!=FLAGS[control])):raise ValueError("W1 method/flags/seed/source recipe differs")
        if cfg["profile"]==PROFILE and (e["ranges"]!=[dict(zip(("domain","entity","run"),cfg["stream"]),start=512,end=87841)] or e["identity"]["input_sha256"]!=named.INPUT_SHA):raise ValueError("Original PSM arrived axis required")
    pairs=[dict(reference="full_seed2025",candidate=k,pairing="same_training_seed") for k in CONTROLS]+[dict(reference="full_seed2025",candidate=k,pairing="model_vs_deterministic") for k in ("persistence","history_mean512")]
    if cfg["comparisons"]!=pairs:raise ValueError("Retain all four controls plus both baselines, six pairs")
    if cfg["profile"]==PROFILE and (cfg.get("history")!=512 or cfg.get("original_length")!=87841 or cfg.get("original_start")!=0 or cfg.get("stream")!=["PSM","anonymous_PSM","test_original"] or cfg["labels"]["sha256"]!=named.LABEL_SHA):raise ValueError("Fixed W1 PSM input/label identity required")


def bank_gates(cfg):
    contract(cfg);named._not_label(cfg["raw_seal"]["path"],cfg);seal=bound(cfg["raw_seal"])
    if seal!={"status":"ACCEPTED_SYSTEM_W1_FIXED_BANKS","approved_by":"root","profile":cfg["profile"],"banks":cfg["banks"]}:raise PermissionError("Exact root W1 seal required before labels")
    reports=[]
    for e in cfg["banks"]:
        # Reject any label-path alias before metadata/hash or record I/O.
        for path in [str(Path(e["directory"])/"manifest.json"),e["calibration"]["path"]]+([e["calibration_receipt"]["path"]] if e["kind"]=="source_model" else []):named._not_label(path,cfg)
        mp=Path(e["directory"])/"manifest.json"
        if sha(mp)!=e["manifest_sha256"]:raise PermissionError("W1 manifest changed")
        manifest=json.loads(mp.read_text())
        for c in manifest["chunks"]:named._not_label(Path(e["directory"])/c["file"],cfg)
        m=bank_metadata(e);cal=bound(e["calibration"])
        if (e["identity"]["calibration_sha256"]!=e["calibration"]["sha256"] or cal.get("source_domain_equal_weight") is not True
                or cal.get("active_components")!=e["active_components"] or cal.get("weights")!=recipe(e["active_components"])[1]
                or cal.get("domains")!=["SMD","Sunspot"] or set(cal.get("scales",{}))!=set(e["active_components"])
                or any(not np.isfinite(v) or v<=0 for v in cal["scales"].values()) or not np.isfinite(cal.get("threshold",np.nan))
                or m["contract"]["threshold"]!=cal["threshold"]):raise ValueError("Each bank must retain its own exact source CAL/active weights")
        if e["kind"]=="source_model":
            receipt=bound(e["calibration_receipt"])
            if receipt.get("phase")!="calibrate_source" or receipt.get("files",{}).get("source_calibration.json")!=e["calibration"]["sha256"]:raise ValueError("No targetCAL/U receipt substitution")
        reports.append(dict(id=e["id"],threshold=cal["threshold"],manifest_sha256=e["manifest_sha256"],records=m["committed_records"]))
    if cfg.get("affiliation_identity"):
        named._not_label(cfg["affiliation_identity"],cfg)
        if sha(cfg["affiliation_identity"])!=cfg["affiliation_identity_sha256"]:raise PermissionError("Official Aff identity changed")
        from ..engine.official_affiliation import load_official_affiliation
        load_official_affiliation(cfg["affiliation_identity"])
    return reports


def admit(config,release,trusted):
    cfg=json.loads(Path(config).read_text())
    if not release or not trusted:raise PermissionError("Exact root W1 CPU release required")
    signed=bound(dict(path=release,sha256=trusted))
    if (signed.get("status")!="APPROVED" or signed.get("approved_by")!="root" or signed.get("operation")!=OPERATION or cfg.get("operation")!=OPERATION
            or signed.get("allowed_actions")!=["evaluate_system_w1_fixed_banks"] or signed.get("allow_labels") is not True
            or signed.get("config_sha256")!=sha(config) or signed.get("raw_seal_sha256")!=cfg["raw_seal"]["sha256"]):raise PermissionError("Changed/pending W1 release")
    if cfg.get("permissions")!={"labels":True,"evaluation":True,"training":False,"target_fit":False,"GPU":False}:raise PermissionError("CPU evaluation only")
    contract(cfg);code_gate(cfg,signed,{"hybrid_tsad/hybrid_v1/system_evaluate_cli.py","hybrid_tsad/hybrid_v1/bank_recalibration.py"})
    if Path(cfg["output"]).exists():raise FileExistsError("Fresh W1 output required")
    return cfg


def run_evaluation(cfg):
    if Path(cfg["output"]).exists():raise FileExistsError("Fresh named output required")
    gates=bank_gates(cfg);arrays=scalar_inputs(cfg)  # ALL seal/identity/axis checks precede labels.
    labels=load_labels(cfg["labels"])
    from .evaluation import evaluate_bank_arrays,save_evaluation
    output=Path(cfg["output"]);output.mkdir(parents=True,exist_ok=False);results={}
    for entry,gate in zip(cfg["banks"],gates):
        result=evaluate_bank_arrays(entry["directory"],labels,gate["threshold"],expected_manifest_sha256=entry["manifest_sha256"],
            original_length=cfg["original_length"],original_start=cfg["original_start"],warmup_points=cfg["history"],stream=cfg["stream"],affiliation_identity=cfg.get("affiliation_identity"))
        values=arrays[entry["id"]]
        save_evaluation(output/entry["id"],result,full_labels=labels,provenance=dict(named_descriptor=entry,raw_seal=cfg["raw_seal"],
            gates=gates,source_sha256=cfg.get("source_sha256"),labels=cfg["labels"],no_score_ensemble=True),**values)
        results[entry["id"]]=result
    pairs=[]
    for pair in cfg["comparisons"]:
        left,right=pair["reference"],pair["candidate"];folder=output/"pairs"/(left+"__vs__"+right);folder.mkdir(parents=True)
        point,common=point_pair(arrays[left],arrays[right],labels,cfg["original_start"])
        events=paired_events(results[left],results[right])
        with (folder/"paired_events.csv").open("x",newline="",encoding="utf-8") as f:
            keys=list(dict.fromkeys(k for row in events for k in row));writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(events)
        np.savez_compressed(folder/"common_point_inputs.npz",index=common,reference_score=arrays[left]["score"][np.searchsorted(arrays[left]["index"],common)],
            candidate_score=arrays[right]["score"][np.searchsorted(arrays[right]["index"],common)],full_labels=labels)
        summary=dict(pair,common_point_effect=point,statistics=dict(STATISTICS_NA),event_pairing="Original per-bank greedy statuses; no rematch or block-concatenated event/Aff CI")
        (folder/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False));pairs.append(summary)
    summary=dict(status="COMPLETED_PENDING_INDEPENDENT_REVIEW",profile=cfg["profile"],methods=[dict(descriptor=e,ranking=results[e["id"]]["ranking"],coverage=results[e["id"]]["coverage"],
        streams={s:dict(point=v["point"],normal=v["normal"],event_metrics=v["main_events"]["metrics"],affiliation=v["affiliation"],Range_VUS=v["Range_VUS"]) for s,v in results[e["id"]]["streams"].items()}) for e in cfg["banks"]],
        comparisons=pairs,statistics=dict(STATISTICS_NA),model_seed_mean=None,no_score_ensemble=True,Range_VUS="Preserve accepted core NA; no substitute formula")
    (output/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    (output/"BANK_GATES.json").write_text(json.dumps(gates,indent=2))
    (output/"CONFIG_AND_PROVENANCE.json").write_text(json.dumps(cfg,indent=2))
    (output/"system_evaluate_cli.py").write_bytes(Path(__file__).read_bytes())
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--release");p.add_argument("--trusted-release-sha256")
    p.add_argument("--execute",action="store_true");p.add_argument("--preflight-only",action="store_true");a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,labels_or_banks_opened=False)));return
    cfg=admit(a.config,a.release,a.trusted_release_sha256)
    if a.preflight_only:bank_gates(cfg);scalar_inputs(cfg);print(json.dumps(dict(PASS=True,labels_opened=False)));return
    run_evaluation(cfg)


if __name__=="__main__":main()
