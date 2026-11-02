"""Root-sealed named fixed-bank CPU evaluation; no model/fit/inference.

Different controls are methods, never fake training seeds. Existing fixed-five
CLI/statistics stay unchanged. Named bootstrap CI is explicitly unimplemented.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np
from .evaluate_cli import sha,load_labels,EVALUATOR_SHA,STATISTICS_SHA,LABEL_SHA,INPUT_SHA,REQUIRED_SOURCE

OPERATION="HYBRID_V1_NAMED_FIXED_BANK_EVALUATION"
READER_SHA="17a298cb5f7255ae9d429630ec4f07152a65c7711a2657bba1e6de6d9ec4d2b4"
STATISTICS_NA=dict(status="NOT_COMPUTED",CI=None,reason="GENERIC_NAMED_PAIRWISE_BOOTSTRAP_NOT_IMPLEMENTED",
                   block_points=4096,n_replicates=2000,rng_seed=99173)
A1_MODELS={"full_seed2025":"full","A1_fusion_none_seed2025":"A1_fusion_none",
           "A1_fusion_feature_mean_seed2025":"A1_fusion_feature_mean",
           "A1_fusion_concat_linear_seed2025":"A1_fusion_concat_linear"}
BASELINES={"persistence","history_mean512"}


def named_contract(cfg):
    """Validate named methods/pairs without opening bank, CAL or label files."""
    if cfg.get("profile") not in {"PSM_A1_SOURCE_SEED2025","SYNTHETIC_NAMED"} or cfg.get("statistics")!=STATISTICS_NA:
        raise ValueError("Explicit named profile and honest unimplemented CI required")
    if any(k in cfg for k in ("model_seeds","seed_groups","model_mean_SD","ensemble","aggregate")):
        raise ValueError("No fake multi-seed means or score ensemble in named evaluation")
    entries=cfg["banks"];ids=[e["id"] for e in entries]
    if not entries or len(ids)!=len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+",s) for s in ids):
        raise ValueError("Unique safe named bank IDs required")
    if len({str(Path(e["directory"]).resolve()) for e in entries})!=len(entries):
        raise ValueError("One bank cannot be relabelled as several controls or seeds")
    lookup={e["id"]:e for e in entries};controls=[]
    for e in entries:
        controls.append(e["control_id"])
        if e.get("calibration_origin")!="source":
            raise ValueError("This first route only supports sealed source-calibrated banks; target rescoring is not implemented")
        if e["kind"]=="source_model":
            if (type(e.get("training_seed")) is not int or e["training_seed"]<0
                    or type(e.get("inference_seed")) is not int or e["inference_seed"]<0):
                raise ValueError("Real training/inference seed metadata required for each distinct method")
        elif (e["kind"]!="deterministic_baseline" or e.get("training_seed") is not None
              or e.get("inference_seed") is not None or e["control_id"] not in BASELINES):
            raise ValueError("A learned control cannot masquerade as deterministic or vice versa")
    if len(controls)!=len(set(controls)):raise ValueError("Unique method controls required; do not duplicate a bank as seeds")
    seen=set()
    for pair in cfg["comparisons"]:
        left,right=pair["reference"],pair["candidate"]
        if left==right or left not in lookup or right not in lookup or (left,right) in seen:
            raise ValueError("Unique explicit reference/candidate comparisons required")
        seen.add((left,right));a,b=lookup[left],lookup[right]
        if pair["pairing"]=="same_training_seed":
            if a["kind"]!=b["kind"] or a["kind"]!="source_model" or a["training_seed"]!=b["training_seed"] or a["inference_seed"]!=b["inference_seed"]:
                raise ValueError("Different training seeds are not same-seed system ablations")
        elif pair["pairing"]!="model_vs_deterministic" or a["kind"]!="source_model" or b["kind"]!="deterministic_baseline":
            raise ValueError("Baseline comparisons must explicitly declare deterministic status")
    if cfg["profile"]=="PSM_A1_SOURCE_SEED2025":
        if (set(ids)!=set(A1_MODELS)|BASELINES or len(entries)!=6 or cfg.get("history")!=512
                or cfg.get("original_length")!=87841 or cfg.get("original_start")!=0
                or cfg.get("stream")!=["PSM","anonymous_PSM","test_original"]):
            raise ValueError("Exact six-method PSM source2025 study required")
        for e in entries:
            if e["id"] in A1_MODELS:
                if e["kind"]!="source_model" or e["control_id"]!=A1_MODELS[e["id"]] or e["training_seed"]!=2025 or e["inference_seed"]!=2025:
                    raise ValueError("Full2025 and each distinct fusion2025 must retain actual identities")
            elif e["kind"]!="deterministic_baseline" or e["control_id"]!=e["id"]:
                raise ValueError("Original deterministic baseline identities required")
            expected=[dict(zip(("domain","entity","run"),cfg["stream"]),start=512,end=87841)]
            if e["ranges"]!=expected or e["identity"]["input_sha256"]!=INPUT_SHA:
                raise ValueError("Full original PSM input/arrived axis required")
        expected={("full_seed2025",name,"same_training_seed") for name in A1_MODELS if name!="full_seed2025"}
        expected|={("full_seed2025",name,"model_vs_deterministic") for name in BASELINES}
        if {(e["reference"],e["candidate"],e["pairing"]) for e in cfg["comparisons"]}!=expected:
            raise ValueError("All preregistered named comparisons must be retained")
    return lookup


def admit(config_path,release_path,trusted):
    cfg=json.loads(Path(config_path).read_text())
    if not release_path or not trusted or sha(release_path)!=trusted:
        raise PermissionError("Exact root named evaluation release required before labels")
    signed=json.loads(Path(release_path).read_text())
    if (signed.get("status")!="APPROVED" or signed.get("approved_by")!="root" or signed.get("operation")!=OPERATION
            or cfg.get("operation")!=OPERATION or signed.get("allowed_actions")!=["evaluate_named_fixed_banks"]
            or signed.get("allow_labels") is not True or signed.get("config_sha256")!=sha(config_path)):
        raise PermissionError("Pending/changed named CPU evaluation release")
    if cfg.get("permissions")!={"labels":True,"evaluation":True,"training":False,"target_fit":False,"GPU":False}:
        raise PermissionError("CPU evaluation-only permissions required")
    named_contract(cfg)
    own=cfg["source_sha256"];base=Path(__file__).resolve().parents[2]
    required=REQUIRED_SOURCE|{"hybrid_tsad/hybrid_v1/named_evaluate_cli.py","hybrid_tsad/hybrid_v1/scoring.py"}
    pinned={"hybrid_tsad/hybrid_v1/evaluation.py":EVALUATOR_SHA,"hybrid_tsad/hybrid_v1/statistics.py":STATISTICS_SHA,
            "hybrid_tsad/hybrid_v1/chunked_bank.py":READER_SHA,
            "hybrid_tsad/hybrid_v1/evaluate_cli.py":"4fe88b0e9b7792ee95845f087113e4e04813fc732e15257fa1a482bbbcd697cd",
            "hybrid_tsad/hybrid_v1/scoring.py":"a87517e1f4a02b8193ad1ffbc667a88710ec22c5db884a8f50d85b818961576d"}
    if signed.get("source_sha256")!=own or not required.issubset(own) or any(own.get(k)!=v for k,v in pinned.items()):
        raise PermissionError("Accepted core/reader/unchanged statistics and full helper closure required")
    for name,digest in own.items():
        path=(base/name).resolve()
        _not_label(path,cfg)
        if base not in path.parents or path.suffix!=".py" or sha(path)!=digest:raise PermissionError("Named evaluation source changed")
    if signed.get("raw_seal_sha256")!=cfg["raw_seal"]["sha256"]:
        raise PermissionError("Root must bind the exact named bank seal")
    if cfg["profile"]=="PSM_A1_SOURCE_SEED2025":
        labels=cfg["labels"]
        if (labels.get("sha256")!=LABEL_SHA or labels.get("shape")!=[87841] or labels.get("format")!="csv"
                or labels.get("drop_columns")!=["timestamp_(min)"] or labels.get("value_columns")!=["label"]):
            raise PermissionError("Original label identity/schema required; no label file opened yet")
    if Path(cfg["output"]).exists():raise FileExistsError("Fresh named output required")
    return cfg


def _not_label(path,cfg):
    # Lexical absolute comparison; do not stat/resolve/open the label file here.
    if os.path.normcase(os.path.abspath(path))==os.path.normcase(os.path.abspath(cfg["labels"]["path"])):
        raise PermissionError("A pre-label bank/seal/CAL/code asset cannot alias the label path")


def bank_gates(cfg):
    """Authenticate the exact root seal and ALL bank/CAL bytes before labels."""
    named_contract(cfg);seal=cfg["raw_seal"]
    _not_label(seal["path"],cfg)
    if not seal.get("sha256") or sha(seal["path"])!=seal["sha256"]:
        raise PermissionError("Missing/changed root named raw seal")
    root=json.loads(Path(seal["path"]).read_text())
    if (root.get("status")!="ACCEPTED_NAMED_FIXED_BANKS" or root.get("approved_by")!="root"
            or root.get("profile")!=cfg["profile"] or root.get("banks")!=cfg["banks"]):
        raise PermissionError("Every named bank, identity, calibration and semantic descriptor must be root-sealed")
    if cfg.get("affiliation_identity") is not None:
        _not_label(cfg["affiliation_identity"],cfg)
        if sha(cfg["affiliation_identity"])!=cfg.get("affiliation_identity_sha256"):
            raise PermissionError("Official affiliation identity changed")
        from ..engine.official_affiliation import load_official_affiliation
        load_official_affiliation(cfg["affiliation_identity"])
    reports=[]
    for e in cfg["banks"]:
        directory=Path(e["directory"]);mp=directory/"manifest.json"
        _not_label(mp,cfg)
        if not e.get("manifest_sha256") or sha(mp)!=e["manifest_sha256"]:raise ValueError("Named bank manifest changed")
        m=json.loads(mp.read_text());contract=m["contract"];ranges=e["ranges"];previous=cfg["original_start"]+cfg["history"];total=0
        for span in ranges:
            if ([span[k] for k in ("domain","entity","run")]!=cfg["stream"] or span["start"]<previous
                    or not span["start"]<span["end"]<=cfg["original_start"]+cfg["original_length"]):
                raise ValueError("Only ordered original ranges, no stitched streams or overlap")
            previous=span["end"];total+=span["end"]-span["start"]
        if (not total or m.get("complete") is not True or m.get("committed_records")!=total
                or contract.get("ranges")!=ranges or contract.get("identity")!=e["identity"]
                or contract.get("chunk_size")!=64 or contract.get("confirmation_points")!=3):
            raise ValueError("Complete named bank identity/count/clock differs")
        cal=e["calibration"]
        _not_label(cal["path"],cfg)
        if sha(cal["path"])!=cal["sha256"] or cal["sha256"]!=e["identity"]["calibration_sha256"]:
            raise ValueError("This bank does not contain scores from this calibration; rescoring not implemented")
        c=json.loads(Path(cal["path"]).read_text())
        if contract["threshold"]!=c["threshold"] or not np.isfinite(c["threshold"]) or c.get("source_domain_equal_weight") is not True:
            raise ValueError("Retain each model's actual fixed source calibration")
        if cfg["profile"]=="PSM_A1_SOURCE_SEED2025":
            from .scoring import recipe
            active=["r_pred"] if e["kind"]=="deterministic_baseline" else ["r_tf","r_ms","r_pred"]
            if c.get("active_components")!=active or c.get("weights")!=recipe(active)[1] or c.get("domains")!=["SMD","Sunspot"]:
                raise ValueError("Actual source-model/baseline CAL recipe/domain differs")
        position=0
        for chunk in m["chunks"]:
            path=directory/chunk["file"]
            _not_label(path,cfg)
            if (path.parent!=directory or path.name!=chunk["file"] or chunk["start_position"]!=position
                    or not 0<chunk["records"]<=64 or sha(path)!=chunk["sha256"]):
                raise ValueError("Named chunk path/count/hash differs before label I/O")
            position+=chunk["records"]
        if position!=total:raise ValueError("Named chunk records incomplete")
        reports.append(dict(id=e["id"],manifest_sha256=e["manifest_sha256"],records=total,threshold=c["threshold"]))
    return reports


def scalar_inputs(cfg):
    """Pre-label scalar replay: raw axis, bool/NA reasons and clock validated."""
    from .chunked_bank import read_records
    output={}
    for e in cfg["banks"]:
        expected=[i for span in e["ranges"] for i in range(span["start"],span["end"])];rows=[]
        for number,row in enumerate(read_records(e["directory"])):
            if (number>=len(expected) or [row[k] for k in ("domain","entity","run")]!=cfg["stream"]
                    or type(row.get("index")) is not int or type(row.get("available_index")) is not int
                    or row["index"]!=expected[number] or row.get("available_index")!=row["index"]
                    or type(row.get("valid")) is not bool):raise ValueError("Raw named indices/clock/valid differ before labels")
            score=row.get("score");valid=row["valid"];reason=row.get("reason")
            if valid and (score is None or not np.isfinite(score)):raise ValueError("Valid score must be finite")
            if not valid and (not isinstance(reason,str) or not reason):raise ValueError("Preserve explicit invalid reason")
            rows.append((row["index"],score,valid,row["available_index"],reason))
        if len(rows)!=len(expected):raise ValueError("Raw bank ends before declared range")
        output[e["id"]]=dict(index=np.array([a[0] for a in rows],np.int64),score=np.array([np.nan if a[1] is None else a[1] for a in rows],np.float64),
            valid=np.array([a[2] for a in rows],bool),available_index=np.array([a[3] for a in rows],np.int64),invalid_reasons=[a[4] for a in rows])
    return output


def point_pair(left,right,labels,start):
    """Observed AP difference on common valid ORIGINAL points; no bootstrap."""
    from ..engine.evaluation import evaluate_point_ranking
    common=np.intersect1d(left["index"][left["valid"]],right["index"][right["valid"]],assume_unique=True)
    mask=np.zeros(len(labels),bool);mask[common-start]=True
    result=dict(common_valid_points=len(common),reference_valid_points=int(left["valid"].sum()),candidate_valid_points=int(right["valid"].sum()),
        primary_full_coverage_replaced=False,CI=None,CI_reason=STATISTICS_NA["reason"],reference_AP=None,candidate_AP=None,AP_difference_candidate_minus_reference=None)
    if not len(common):result["NA_reason"]="NO_COMMON_VALID_POINTS"
    elif len(np.unique(labels[mask]))!=2:result["NA_reason"]="SINGLE_CLASS_COMMON_VALID_LABELS"
    else:
        try:
            values=[evaluate_point_ranking(common,a["score"][np.searchsorted(a["index"],common)],labels,mask,query_start=start)["metrics"]["AP"] for a in (left,right)]
            result.update(reference_AP=values[0],candidate_AP=values[1],AP_difference_candidate_minus_reference=values[1]-values[0],NA_reason=None)
        except ModuleNotFoundError as exc:result["NA_reason"]="RANKING_DEPENDENCY_UNAVAILABLE:"+str(exc.name)
    return result,common


def paired_events(reference,candidate):
    """Original-event join, preserving each bank's greedy match/censor result."""
    left=reference["event_coverage"];right=candidate["event_coverage"]
    if [(e["event_id"],e["start"],e["end_exclusive"]) for e in left]!=[(e["event_id"],e["start"],e["end_exclusive"]) for e in right]:
        raise ValueError("Pair event catalogues differ; no ID/offset repair")
    rows=[]
    for stream in ("strict_point","confirmed3_current"):
        for view in ("main_events","legacy_complete_only_events"):
            a=reference["streams"][stream][view]["events"];b=candidate["streams"][stream][view]["events"]
            expected=[e["event_id"] for e in left]
            if [e["event_id"] for e in a]!=expected or [e["event_id"] for e in b]!=expected:raise ValueError("Pair event row identity differs")
            for original,candidate_coverage,x,y in zip(left,right,a,b):
                both=x["eligible"] and y["eligible"]
                row=dict(stream=stream,view=view,event_id=original["event_id"],start=original["start"],end_exclusive=original["end_exclusive"],
                    boundary_censored=original["boundary_censored"],joint_eligible=both,
                    any_hit_difference=int(y["any_hit"])-int(x["any_hit"]) if both else None,
                    match_difference=int(y["one_to_one_matched"])-int(x["one_to_one_matched"]) if both else None,
                    delay_difference_points=y["delay_points"]-x["delay_points"] if x["delay_points"] is not None and y["delay_points"] is not None else None)
                for prefix,event,coverage in (("reference",x,original),("candidate",y,candidate_coverage)):
                    row.update({prefix+"_"+key:value for key,value in event.items() if key!="event_id"})
                    row.update({prefix+"_coverage_ratio":coverage["coverage_ratio"],prefix+"_scored_points":coverage["scored_points"]})
                rows.append(row)
    return rows


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
    (output/"named_evaluate_cli.py").write_bytes(Path(__file__).read_bytes())
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True);p.add_argument("--release");p.add_argument("--trusted-release-sha256")
    p.add_argument("--execute",action="store_true");p.add_argument("--preflight-only",action="store_true");a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,label_or_bank_opened=False,operation=OPERATION)));return
    cfg=admit(a.config,a.release,a.trusted_release_sha256)
    if a.preflight_only:
        gates=bank_gates(cfg);scalar_inputs(cfg);print(json.dumps(dict(gates=gates,labels_opened=False)));return
    run_evaluation(cfg)


if __name__=="__main__":main()
