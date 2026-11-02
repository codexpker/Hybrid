"""Explicit single-seed Z-S/Z-Tcal/U-Tcal fixed-bank CPU evaluation.

Reuses accepted metric/pair helpers, never changes source-six/five/statistics.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from . import named_evaluate_cli as named
from .bank_recalibration import bound,sha,code_gate,bank_metadata,calibration,checked_rows,recipe_identity
from .scoring import apply_calibration

OPERATION="HYBRID_V1_TARGET_NAMED_FIXED_BANK_EVALUATION"
METHODS={"Z_S_seed2025":("source","Z"),"Z_Tcal_seed2025":("target","Z_TCAL"),"U_Tcal_seed2025":("target","U")}


def contract(cfg):
    if cfg.get("profile") not in {"PSM_U200_SEED2025","SYNTHETIC_TARGET"} or cfg.get("statistics")!=named.STATISTICS_NA:raise ValueError("Fixed target profile/honest NA statistics required")
    if any(k in cfg for k in ("model_seeds","ensemble","aggregate","model_mean_SD","seed_groups")):raise ValueError("Three methods are not three seeds")
    entries=cfg["banks"]
    if len(entries)!=3 or {e["id"] for e in entries}!=set(METHODS) or len({str(Path(e["directory"]).resolve()) for e in entries})!=3:raise ValueError("Exact three distinct methods required")
    by={e["id"]:e for e in entries};z,t,u=[by[k] for k in METHODS]
    for e in entries:
        if (e.get("calibration_origin"),e.get("adaptation_mode"))!=METHODS[e["id"]] or e.get("training_seed")!=2025 or e.get("inference_seed")!=2025 or e.get("active_components")!=["r_tf","r_ms","r_pred"]:raise ValueError("Source/target, seed, mode or primary recipe differs")
        if e["identity"]["input_sha256"]!=z["identity"]["input_sha256"] or e["ranges"]!=z["ranges"]:raise ValueError("Common original input/axis required")
    if t["identity"]["checkpoint_sha256"]!=z["identity"]["checkpoint_sha256"] or u["parent_source_checkpoint_sha256"]!=z["identity"]["checkpoint_sha256"] or u["identity"]["checkpoint_sha256"]==z["identity"]["checkpoint_sha256"]:raise ValueError("Frozen Z-Tcal and adapted U lineage required")
    if t["receipt_expect"]["plans_sha256"]!=u["receipt_expect"]["plans_sha256"] or t["receipt_expect"]["manifest_sha256"]!=u["receipt_expect"]["manifest_sha256"]:raise ValueError("Same target CAL positions/input required")
    pairs=[dict(reference="Z_S_seed2025",candidate="Z_Tcal_seed2025",pairing="calibration_only"),dict(reference="Z_Tcal_seed2025",candidate="U_Tcal_seed2025",pairing="same_seed_equal_target_CAL"),dict(reference="Z_S_seed2025",candidate="U_Tcal_seed2025",pairing="combined_update_and_calibration")]
    if cfg["comparisons"]!=pairs:raise ValueError("Retain all three predeclared comparisons")
    if cfg["profile"]=="PSM_U200_SEED2025":
        expected=[dict(domain="PSM",entity="anonymous_PSM",run="test_original",start=512,end=87841)]
        if (cfg["history"]!=512 or cfg["original_length"]!=87841 or cfg["original_start"]!=0 or cfg["stream"]!=["PSM","anonymous_PSM","test_original"]
                or z["ranges"]!=expected or z["identity"]["input_sha256"]!=named.INPUT_SHA or cfg["labels"]["sha256"]!=named.LABEL_SHA):raise ValueError("Fixed PSM axis/label identity required")
    return by


def _no_label_alias(cfg):
    # Inspect descriptor strings before opening any asset; label path is never stat'ed.
    label=str(Path(cfg["labels"]["path"]).absolute()).casefold()
    def walk(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k=="labels":continue
                if k in {"path","directory","affiliation_identity"} and isinstance(x,str) and str(Path(x).absolute()).casefold()==label:raise PermissionError("Prelabel asset aliases labels")
                walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(cfg)


def gates(cfg):
    contract(cfg);_no_label_alias(cfg);seal=bound(cfg["raw_seal"])
    if seal!={"status":"ACCEPTED_TARGET_FIXED_BANKS","approved_by":"root","profile":cfg["profile"],"banks":cfg["banks"]}:raise PermissionError("Exact root method/CAL seal required before labels")
    reports=[]
    for e in cfg["banks"]:
        m=bank_metadata(e);c=calibration(e)
        if m["contract"]["threshold"]!=c["threshold"]:raise ValueError("Bank threshold differs from actual CAL")
        parent=None
        if e["id"]=="Z_Tcal_seed2025":
            overlay=e["overlay"];r=bound(overlay["receipt"]);bank_metadata(overlay["producer"])
            if (r.get("operation")!="HYBRID_V1_RAW_BANK_RECALIBRATION" or r.get("producer")!=overlay["producer"] or r.get("destination")!=overlay["destination"]
                    or r.get("manifest_sha256")!=e["manifest_sha256"] or overlay["destination"]["identity"]!=e["identity"]
                    or e["identity"]["config_sha256"]!=recipe_identity(overlay["producer"],overlay["destination"])
                    or overlay["producer"]["manifest_sha256"]!=next(x for x in cfg["banks"] if x["id"]=="Z_S_seed2025")["manifest_sha256"]):raise ValueError("Z-Tcal overlay producer lineage differs")
            parent=iter(checked_rows(overlay["producer"]))
        for position,row in enumerate(checked_rows(e)):
            raw=next(parent,None) if parent is not None else row
            if raw is None or raw["index"]!=row["index"]:raise ValueError("Overlay/original axis differs")
            if parent is not None and (row.get("producer_position")!=position or row.get("producer_manifest_sha256")!=e["overlay"]["producer"]["manifest_sha256"]):raise ValueError("Overlay original position differs")
            expected=apply_calibration(raw,c)
            if type(row.get("valid")) is not bool or row["valid"]!=expected["valid"] or (row["valid"] and row.get("score")!=expected["score"]) or (not row["valid"] and row.get("reason")!=expected["reason"]):raise ValueError("Stored scores/valid are not this raw plus this CAL")
            if parent is not None:
                from .chunked_bank import _fingerprint
                if _fingerprint({k:row.get(k) for k in e["active_components"]})!=_fingerprint({k:raw.get(k) for k in e["active_components"]}):raise ValueError("Overlay residual changed")
        if parent is not None and next(parent,None) is not None:raise ValueError("Producer tail dropped")
        reports.append(dict(id=e["id"],threshold=c["threshold"],calibration_origin=e["calibration_origin"],mode=e["adaptation_mode"]))
    if cfg.get("affiliation_identity"):
        if sha(cfg["affiliation_identity"])!=cfg["affiliation_identity_sha256"]:raise PermissionError("Official Aff identity changed")
        from ..engine.official_affiliation import load_official_affiliation
        load_official_affiliation(cfg["affiliation_identity"])
    return reports


def run(cfg):
    if Path(cfg["output"]).exists():raise FileExistsError("Fresh evaluation output required")
    reports=gates(cfg);arrays=named.scalar_inputs(cfg) # Every bank and scalar verified before label SHA/body.
    labels=named.load_labels(cfg["labels"])
    from .evaluation import evaluate_bank_arrays,save_evaluation
    output=Path(cfg["output"]);output.mkdir(parents=True,exist_ok=False);results={}
    for e,g in zip(cfg["banks"],reports):
        result=evaluate_bank_arrays(e["directory"],labels,g["threshold"],expected_manifest_sha256=e["manifest_sha256"],original_length=cfg["original_length"],original_start=cfg["original_start"],warmup_points=cfg["history"],stream=cfg["stream"],affiliation_identity=cfg.get("affiliation_identity"))
        save_evaluation(output/e["id"],result,full_labels=labels,provenance=dict(descriptor=e,raw_seal=cfg["raw_seal"],source_sha256=cfg.get("source_sha256")),**arrays[e["id"]]);results[e["id"]]=result
    pairs=[]
    for pair in cfg["comparisons"]:
        a,b=pair["reference"],pair["candidate"];folder=output/"pairs"/(a+"__vs__"+b);folder.mkdir(parents=True)
        effect,common=named.point_pair(arrays[a],arrays[b],labels,cfg["original_start"]);events=named.paired_events(results[a],results[b])
        with (folder/"paired_events.csv").open("x",newline="") as f:
            keys=list(dict.fromkeys(k for row in events for k in row));writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(events)
        np.savez_compressed(folder/"common_point_inputs.npz",index=common,reference_score=arrays[a]["score"][np.searchsorted(arrays[a]["index"],common)],candidate_score=arrays[b]["score"][np.searchsorted(arrays[b]["index"],common)],full_labels=labels)
        item=dict(pair,common_point_effect=effect,statistics=dict(named.STATISTICS_NA));(folder/"summary.json").write_text(json.dumps(item,indent=2,allow_nan=False));pairs.append(item)
    summary=dict(status="COMPLETED_PENDING_REVIEW",profile=cfg["profile"],methods=results,comparisons=pairs,statistics=dict(named.STATISTICS_NA),model_seed_mean=None,strong_equal_CAL_baseline_claim="NOT_SUPPORTED_NO_TARGET_CAL_BASELINE_ASSETS")
    (output/"SUMMARY.json").write_text(json.dumps(summary,indent=2,allow_nan=False));(output/"CONFIG.json").write_text(json.dumps(cfg,indent=2))
    (output/"BANK_GATES.json").write_text(json.dumps(reports,indent=2))
    (output/"target_evaluate_cli.py").write_bytes(Path(__file__).read_bytes());return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("config","release","trusted-release-sha256"):p.add_argument("--"+k)
    p.add_argument("--execute",action="store_true");p.add_argument("--preflight-only",action="store_true");a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,bank_or_labels_opened=False)));return
    cfg=json.loads(Path(a.config).read_text());signed=bound(dict(path=a.release,sha256=a.trusted_release_sha256))
    if (signed.get("status")!="APPROVED" or signed.get("approved_by")!="root" or signed.get("operation")!=OPERATION or cfg.get("operation")!=OPERATION or signed.get("allowed_actions")!=["evaluate_target_named_fixed_banks"] or signed.get("allow_labels") is not True or signed.get("config_sha256")!=sha(a.config) or signed.get("raw_seal_sha256")!=cfg["raw_seal"]["sha256"]):raise PermissionError("Exact root CPU evaluation release required")
    code_gate(cfg,signed,{"hybrid_tsad/hybrid_v1/target_evaluate_cli.py","hybrid_tsad/hybrid_v1/bank_recalibration.py","hybrid_tsad/hybrid_v1/named_evaluate_cli.py"})
    if a.preflight_only:gates(cfg);named.scalar_inputs(cfg);print(json.dumps(dict(PASS=True,labels_opened=False)));return
    run(cfg)


if __name__=="__main__":main()
