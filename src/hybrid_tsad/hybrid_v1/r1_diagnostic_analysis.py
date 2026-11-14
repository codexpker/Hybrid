"""Fixed CPU description of frozen R1 outputs; no model, new loss or fitting.

All four complete raw directories are byte-sealed before numerical outputs or
the separate qualification ledger are read. Direction is SOURCE minus STUDENT.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

MODELS=("SOURCE_2025","U_A_2025","U_B_2025","U_C_2025")
TASKS=("tf","ms","pred")


def availability(measure):
    if measure.endswith("total") or measure.endswith("loss_pred") or measure=="delta_source_minus_student_pred" or measure=="D_pred_teacher_scale_MSE":
        return 31,"OFFLINE_P32_TASK_DIAGNOSTIC"
    return 0,"CURRENT_ARRIVED_T_EVIDENCE"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,allow_nan=False)


def scalar(value):
    a=np.asarray(value,dtype=np.float64)
    if a.size!=1:raise ValueError("One window/scalar required")
    v=float(a.reshape(-1)[0]);return v if np.isfinite(v) else None


def inverse_quantile(values,q):
    v=sorted(float(x) for x in values if x is not None and np.isfinite(x))
    return v[max(0,math.ceil(q*len(v))-1)] if v else None


def summarize(values):
    v=[float(x) for x in values if x is not None and np.isfinite(x)]
    return dict(n=len(v),mean=float(np.mean(v)) if v else None,median=float(np.median(v)) if v else None,
                min=min(v) if v else None,max=max(v) if v else None,
                positive_count=sum(x>0 for x in v),negative_count=sum(x<0 for x in v))


def write_csv(path,rows):
    if not rows:raise ValueError("Do not silently emit an empty analysis")
    keys=list(rows[0])
    with Path(path).open("x",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def deduplicate(rows):
    result={}
    for row in rows:
        # Same numerical unit with different truth is retained as an explicit
        # semantic ambiguity, never silently collapsed into a preferred label.
        key=(row["model"],row["unit"],row["label"],row["P32_class"],row["H512_class"])
        if key not in result:
            result[key]=dict(row,logical_alias_count=1,logical_aliases=row["sample_id"])
        else:
            result[key]["logical_alias_count"]+=1
            result[key]["logical_aliases"]+="|"+row["sample_id"]
    return list(result.values())


def verify_complete(raw,receipt_hash,plan,cfg,out):
    if sha(raw/"RECEIPT.json")!=receipt_hash:raise PermissionError("Changed accepted receipt")
    receipt=json.loads((raw/"RECEIPT.json").read_text(encoding="utf-8"))
    if (receipt.get("status")!="COMPLETED_PENDING_INDEPENDENT_AUDIT"
            or receipt.get("total_unique_model_window_units")!=344
            or receipt.get("labels_or_phase_read") is not False
            or [x.get("model") for x in receipt.get("models",[])]!=list(MODELS)):
        raise PermissionError("Four complete frozen raw outputs required before analysis")
    expected={u["forward_unit_id"] for u in plan["units"]}
    artifacts=[raw/"RECEIPT.json",raw/"CONFIG.json"]
    for model in MODELS:
        folder=raw/model
        for suffix in ("npz","json"):
            files=list((folder/"raw").glob("*."+suffix))
            if {p.stem for p in files}!=expected or len(files)!=86:
                raise ValueError("Incomplete, duplicate or unexpected raw output: "+model)
            artifacts+=files
        for name in ("STATE_BEFORE.json","STATE_AFTER.json","runtime.json","TASK_CHANGE_AND_DISCREPANCY.json","PROGRESS.jsonl"):
            if not (folder/name).is_file():raise FileNotFoundError(folder/name)
            artifacts.append(folder/name)
        before=json.loads((folder/"STATE_BEFORE.json").read_text(encoding="utf-8"))
        after=json.loads((folder/"STATE_AFTER.json").read_text(encoding="utf-8"))
        if before!=after:raise ValueError("Frozen state changed")
    seal=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(artifacts)]
    dump(out/"ALL_RAW_BYTE_SEAL_BEFORE_ANALYSIS.json",dict(accepted_receipt_sha256=receipt_hash,
        files=seal,raw_bundle_pairs=344,models=list(MODELS),numerical_task_outputs_opened=False,
        qualification_ledger_opened=False))
    saved=json.loads((raw/"CONFIG.json").read_text(encoding="utf-8"))
    if saved.get("numeric_plan",{}).get("sha256")!=cfg["numeric_plan"]["sha256"]:
        raise ValueError("Forward used a different numeric plan")
    return receipt


def read_rows(raw,plan,ledger):
    logical={r["sample_id"]:r for r in ledger if r["source_kind"]=="new_synthetic"}
    if len(logical)!=96:raise ValueError("Exactly96 registered synthetic logical windows required")
    units={u["forward_unit_id"]:u for u in plan["units"]};rows=[]
    for model in MODELS:
        changes=json.loads((raw/model/"TASK_CHANGE_AND_DISCREPANCY.json").read_text(encoding="utf-8"))
        if len(changes)!=86 or {x["forward_unit_id"] for x in changes}!=set(units):
            raise ValueError("Changed/missing task-change unit")
        for item in changes:
            unit=units[item["forward_unit_id"]]
            if (item["logical_alias_ids"]!=unit["logical_alias_ids"] or item["target"]!=unit["target"]
                    or item["mask_seed"]!=unit["mask_seed"] or item["same_masks_stats"] is not True):
                raise ValueError("Mismatched mask, index or alias")
            task=item["task_change"];kd=item["source_discrepancy"]
            values={}
            for key in TASKS+("consistency",):
                p=task["parts"][key]
                s=scalar(p["source"]) if p["valid"] else None
                u=scalar(p["student"]) if p["valid"] else None
                values["source_loss_"+key]=s;values["student_loss_"+key]=u
                values["delta_source_minus_student_"+key]=s-u if s is not None and u is not None else None
                values["legacy_student_minus_source_"+key]=p["student_minus_source"]
                values["valid_"+key]=p["valid"]
            s,u=scalar(task["source_total"]),scalar(task["student_total"])
            values.update(source_total=s,student_total=u,delta_source_minus_student_total=s-u if s is not None and u is not None else None)
            for key in ("D_pred_teacher_scale_MSE","D_T_feature_MSE","D_F_feature_MSE"):
                values[key]=kd[key]
            for key in ("r_tf","r_ms","r_pred","r_var"):
                values["source_current_"+key]=scalar(task["source_raw_residuals"][key])
                values["student_current_"+key]=scalar(task["student_raw_residuals"][key])
            for alias in unit["logical_alias_ids"]:
                r=logical[alias]
                if r["forward_unit_id"]!=unit["forward_unit_id"] or int(r["mask_seed"])!=unit["mask_seed"]:
                    raise ValueError("Ledger does not match frozen numeric plan")
                label=int(r["diagnostic_label"]) if r["diagnostic_label"] in ("0","1") and r["class_separation_eligible"]=="True" else None
                rows.append(dict(model=model,sample_id=alias,unit=unit["forward_unit_id"],family=r["family"],
                    run=int(r["run_id"]),variant=r["variant"],target=int(r["target"]),label=label,
                    phase=r["phase"],P32_class=r["P32_target_class"],P32_positive=int(r["P32_positive_points"]),
                    H512_class=r["H512_history_class"],H512_positive=int(r["H512_positive_points"]),
                    TF_last100_positive=int(r["reconstruction_last100_positive_points"]),
                    normal_by_generator=label==0 and r["variant"] in ("N","NORMAL_DRIFT","NORMAL_SCALE_SWITCH"),**values))
    return rows


def analyze(rows,cfg):
    unique=deduplicate(rows);families=sorted({r["family"] for r in rows});thresholds={}
    for family in families:
        pool=[r for r in unique if r["model"]=="SOURCE_2025" and r["family"]==family
              and r["run"]==9101 and r["normal_by_generator"]]
        thresholds[family]=dict(q=.8,threshold=inverse_quantile([r["source_total"] for r in pool],.8),
            n_unique=len(pool),reference_run=9101,strict_greater=True,
            qualification="Generated normal controls; difficulty only, not learned health labels")
    for r in rows:
        q=thresholds[r["family"]]["threshold"]
        r["normal_difficulty_q80_from_run9101"]=q
        r["high_source_error_normal_run9102"]=(r["run"]==9102 and r["normal_by_generator"]
            and q is not None and r["source_total"] is not None and r["source_total"]>q)
    unique=deduplicate(rows)
    measures=["source_total","student_total","delta_source_minus_student_total"]
    measures += [prefix+task for task in TASKS for prefix in ("source_loss_","student_loss_","delta_source_minus_student_")]
    measures += ["D_pred_teacher_scale_MSE","D_T_feature_MSE","D_F_feature_MSE"]
    measures += [side+"_current_"+component for side in ("source","student") for component in ("r_tf","r_ms","r_pred")]
    summaries=[]
    for model in MODELS:
        for family in families:
            for run in (9101,9102):
                base=[r for r in unique if (r["model"],r["family"],r["run"])==(model,family,run)]
                for layer in ("ALL_REGISTERED","GENERATED_NORMAL","NORMAL_CHANGE","ELIGIBLE_ABNORMAL","UNCERTAIN_RECOVERY","HIGH_SOURCE_ERROR_NORMAL"):
                    chosen=[r for r in base if layer=="ALL_REGISTERED" or
                        (layer=="GENERATED_NORMAL" and r["normal_by_generator"]) or
                        (layer=="NORMAL_CHANGE" and r["normal_by_generator"] and r["variant"] in ("NORMAL_DRIFT","NORMAL_SCALE_SWITCH")) or
                        (layer=="ELIGIBLE_ABNORMAL" and r["label"]==1) or
                        (layer=="UNCERTAIN_RECOVERY" and r["label"] is None) or
                        (layer=="HIGH_SOURCE_ERROR_NORMAL" and r["high_source_error_normal_run9102"])]
                    for measure in measures:
                        offset,clock=availability(measure)
                        summaries.append(dict(model=model,family=family,run=run,stratum=layer,measure=measure,
                            score_available_offset=offset,evidence_clock=clock,
                            eligible_rows=len(chosen),NA_reason=None if chosen else "EMPTY_FIXED_STRATUM",**summarize([r[measure] for r in chosen])))
                for pclass,hclass in sorted({(r["P32_class"],r["H512_class"]) for r in base}):
                    chosen=[r for r in base if (r["P32_class"],r["H512_class"])==(pclass,hclass)]
                    for measure in measures:
                        offset,clock=availability(measure)
                        summaries.append(dict(model=model,family=family,run=run,stratum="P32="+pclass+";H512="+hclass,measure=measure,
                            score_available_offset=offset,evidence_clock=clock,
                            eligible_rows=len(chosen),NA_reason=None,**summarize([r[measure] for r in chosen])))
    try:
        from sklearn.metrics import roc_auc_score,average_precision_score
        metric_dependency=None
    except (ImportError,AttributeError) as exc:
        roc_auc_score=average_precision_score=None;metric_dependency=type(exc).__name__+": "+str(exc)
    rank=[]
    for model in MODELS:
        for family in families:
            base=[r for r in unique if r["model"]==model and r["family"]==family and r["run"]==9102 and r["label"] is not None]
            for measure in measures:
                chosen=[r for r in base if r[measure] is not None and np.isfinite(r[measure])]
                y=[r["label"] for r in chosen];score=[r[measure] for r in chosen]
                reason="DEPENDENCY_UNAVAILABLE" if metric_dependency else ("SINGLE_CLASS_OR_EMPTY" if len(set(y))<2 else None)
                offset,clock=availability(measure)
                rank.append(dict(model=model,family=family,run=9102,measure=measure,
                    score_available_offset=offset,evidence_clock=clock,diagnostic_P32_label_mature_offset=31,
                    direction="HIGHER_VALUE_RANKED_AS_ABNORMAL_NO_SIGN_SEARCH",n=len(y),positive=sum(y),
                    AP=float(average_precision_score(y,score)) if reason is None else None,
                    AUROC=float(roc_auc_score(y,score)) if reason is None else None,
                    NA_reason=reason,dependency_detail=metric_dependency,
                    inference_scope="Sparse paired-anchor description; P32 labels mature at t+31; not online-event performance"))
    return rows,unique,thresholds,summaries,rank


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--raw-root",required=True);p.add_argument("--accepted-receipt-sha256",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();start=time.monotonic();cfg=json.loads(Path(a.config).read_text(encoding="utf-8"))
    if (cfg.get("analysis_version")!="R1_FIXED_TRANSFER_DIAG_V1" or cfg.get("normal_reference_quantile")!=.8
            or cfg.get("reference_run")!=9101 or cfg.get("diagnostic_run")!=9102
            or cfg.get("delta_direction")!="SOURCE_MINUS_STUDENT" or cfg.get("model_ids")!=list(MODELS)):
        raise PermissionError("Changed fixed R1 description protocol")
    for key in ("numeric_plan","qualification_ledger"):
        if sha(cfg[key]["path"])!=cfg[key]["sha256"]:raise ValueError("Changed frozen analysis input")
    if sha(__file__)!=cfg["analysis_source_sha256"]:raise ValueError("Changed analysis caller")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads(Path(cfg["numeric_plan"]["path"]).read_text(encoding="utf-8"))
    raw=Path(a.raw_root);verify_complete(raw,a.accepted_receipt_sha256,plan,cfg,out)
    with Path(cfg["qualification_ledger"]["path"]).open(encoding="utf-8",newline="") as f:ledger=list(csv.DictReader(f))
    rows=read_rows(raw,plan,ledger);rows,unique,thresholds,summaries,rank=analyze(rows,cfg)
    write_csv(out/"ALL_LOGICAL_VALUES.csv",rows);write_csv(out/"DEDUPLICATED_VALUES.csv",unique)
    write_csv(out/"ALL_TASK_STRATIFIED_SUMMARIES.csv",summaries)
    write_csv(out/"OFFLINE_P32_TASK_RANKING.csv",[r for r in rank if r["score_available_offset"]==31])
    write_csv(out/"CURRENT_T_EVIDENCE_RANKING.csv",[r for r in rank if r["score_available_offset"]==0])
    dump(out/"SOURCE_DIFFICULTY_THRESHOLDS.json",thresholds)
    dump(out/"RECEIPT.json",dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT",logical_rows=len(rows),
        expected_logical_rows=384,unique_analysis_rows=len(unique),actual_neural_forward=False,new_loss=False,
        calibration_or_anomaly_threshold_fit=False,normal_difficulty_fit_run=9101,diagnostic_run=9102,
        raw_receipt_sha256=a.accepted_receipt_sha256,analysis_source_sha256=sha(__file__),
        seconds=time.monotonic()-start,synthetic_transfer_only=True,independent_event_CI=False,
        original_M31_pollution_absorption_proven=False))
    print(json.dumps(dict(status="COMPLETE",logical_rows=len(rows),seconds=time.monotonic()-start)))


if __name__=="__main__":main()
