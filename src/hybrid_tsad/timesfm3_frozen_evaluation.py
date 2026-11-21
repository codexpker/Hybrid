"""Read-only evaluation of complete, frozen official TimesFM3 point forecasts.

The score is a project residual baseline, not an official anomaly head. No
model is imported, no forecast is recomputed and no QUERY cutoff is deployed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .native_v_conditioning_evaluation import (
    AXES, ACCEPTED_NATIVE_METRICS, ACCEPTED_NATIVE_METRICS_SHA,
    ACCEPTED_NATIVE_EVENTS, ACCEPTED_NATIVE_EVENTS_SHA,
    EXTERNAL_MANIFEST, EXTERNAL_MANIFEST_SHA,
    causal_timing, checked, exact_cal, load_bank, read, sha)

METHOD = "TimesFM3_frozen"
SOURCE = "918df02c9d366b2145bc5876135fa2131d3ed2a6"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
WEIGHT = "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da"
OPTIONS = dict(horizon=1, return_quantiles=True, use_symmetric_averaging=False,
               make_positive=False, sort_quantiles=True, use_znorm=False, padding_mode="none")
BASELINES = {"TimeRadar_V_A", "TimesFM_RAW_D", "HistoryMean512", "LEFT_noU", "LEFT_R",
             "MLP_noU", "MLP_R", "CANDI_A"}
PAIR_FIELDS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC", "normal_FPR", "Recall", "F1",
    "M", "auxiliary_any_hit_events", "one_to_one_eventF1", "event_point_coverage",
    "event_macro_coverage", "long_steady_macro_coverage", "long_steady_point_coverage",
    "max_event_miss_points", "normal_false_alarm_points", "normal_alarm_fragments")


def paired_rows(rows):
    """Same-axis descriptive differences; not a model/permission ablation."""
    pairs = []
    for domain in AXES:
        for stream in ("strict", "confirmed3"):
            new = next(r for r in rows if (r["domain"],r["method"],r["stream"]) == (domain,METHOD,stream))
            for name in ("TimeRadar_V_A", "TimesFM_RAW_D"):
                old = next(r for r in rows if (r["domain"],r["method"],r["stream"]) == (domain,name,stream))
                if new["QUERY_axis"] != old["QUERY_axis"]:
                    raise ValueError("Never subtract different QUERY axes")
                row = dict(domain=domain, stream=stream, contrast=METHOD+" minus "+name,
                    comparison="complete methods with distinct pretraining/architecture; not a weight-only ablation")
                row.update({k:None if new.get(k) is None or old.get(k) is None else new[k]-old[k] for k in PAIR_FIELDS})
                pairs.append(row)
    return pairs


def formula_gate(bank):
    index = np.asarray(bank["absolute_time_index"])
    pred, quant = np.asarray(bank["prediction_normalized"]), np.asarray(bank["quantiles_normalized"])
    actual = np.asarray(bank["actual_normalized"])
    if (index.ndim != 1 or not len(index) or np.any(np.diff(index) <= 0)
            or pred.ndim != 2 or pred.shape != actual.shape or pred.shape[0] != len(index)
            or quant.shape != (*pred.shape, 9)):
        raise ValueError("Actual joint multivariate predictions and nine quantiles per point/channel required")
    for key, wanted in (("available_index", index), ("history_start", index-512), ("history_end_exclusive", index)):
        np.testing.assert_array_equal(bank[key], wanted)
    np.testing.assert_allclose(bank["quantile_levels"], np.arange(1, 10)/10, rtol=0, atol=1e-15)
    if not np.isfinite(quant).all() or not np.isfinite(actual).all():
        raise ValueError("Keep finite official predictions and arrived observations")
    np.testing.assert_array_equal(pred, quant[..., 4])
    error = np.square(pred.astype(np.float64)-actual.astype(np.float64))
    np.testing.assert_array_equal(bank["squared_error_per_channel"], error)
    np.testing.assert_array_equal(bank["canonical_score"], error.mean(axis=1))
    if np.asarray(bank["canonical_score"]).shape != index.shape:
        raise ValueError("No scalar batch MSE or unsupervised quantile score substitution")
    return dict(points=len(index), channels=pred.shape[1], median_quantile_index=4,
                residual_precision="storedFP32 operands promotedFP64 before square/meanC", exact=True)


def evaluate_arrays(domain, cal, query, cal_points, labels, metric_path):
    from .left_native_swat_metrics import continuous_metrics, load_author_metric
    from .timercd_swat_evaluation import evaluate_score
    if domain not in AXES or not np.array_equal(query["absolute_time_index"], np.arange(*AXES[domain])):
        raise ValueError("Keep the full fixed public QUERY axis")
    checks = {"CAL": formula_gate(cal), "QUERY": formula_gate(query)}
    tau = exact_cal(cal["absolute_time_index"], cal["canonical_score"], cal_points)
    idx = query["absolute_time_index"]
    ranking, surface = continuous_metrics(labels[idx], query["canonical_score"], load_author_metric(metric_path))
    rows, events, _, _ = evaluate_score(idx, query["canonical_score"], labels, tau, idx,
        query_start=AXES[domain][0], query_stop=AXES[domain][1], original_length=len(labels),
        expected_events=None, ranking=ranking)
    rows, events = causal_timing(rows, events)
    for row in rows:
        row.update(ranking, domain=domain, method=METHOD, seed=None, QUERY_axis=list(AXES[domain]),
            CAL_points=512, CAL_selected_attack_points=int(labels[cal_points].sum()),
            CAL_exceedances=int(np.sum(cal["canonical_score"] > tau)),
            calibration="same512 legal mixed CAL; ownq99/higher/strict>; not a normal1%FPR certificate",
            history=512, forecast_horizon=1, all_variables_joint=True, training_updates=0,
            adaptation_updates=0, trainable_parameters=0, official_weights_used=True,
            source_commit=SOURCE, weight_revision=REVISION, weight_sha256=WEIGHT,
            score="project SOURCEFIT-standardized meanC squared official median forecast residual; no officialADhead",
            input_access="strict past512; no current or future observations/covariates",
            external_scaler="SOURCE_FIT only; one transform; official internal inverse retained",
            historically_exposed_development=True)
    for event in events:
        event.update(domain=domain, method=METHOD)
    return rows, events, surface, checks


def terminal_inputs(workspace, root):
    seal = read(root/"GLOBAL_RAW_BANK_SEAL.json")
    if (seal.get("status") != "COMPLETED" or seal.get("labels_read") is not False
            or seal.get("no_labels_decoded") is not True or seal.get("training") is not False
            or seal.get("adaptation") is not False or seal.get("weight_sha256") != WEIGHT
            or seal.get("source_commit") != SOURCE):
        raise ValueError("Need a complete label-free, zero-update official checkpoint seal")
    records = {(r["domain"], r["role"]): r for r in seal["banks"]}
    if len(seal["banks"]) != 4 or set(records) != {(d,r) for d in AXES for r in ("CAL", "QUERY")}:
        raise ValueError("Exactly four complete public CAL/QUERY banks")
    accepted = read(checked(workspace, dict(relative=ACCEPTED_NATIVE_METRICS, sha256=ACCEPTED_NATIVE_METRICS_SHA)))
    paths, tasks = {}, {}
    for domain in AXES:
        task = read(root/domain/"TASK.json"); tasks[domain] = task
        original_path = checked(workspace, dict(relative=task["source_TASK_relative"], sha256=task["source_TASK_sha256"]))
        old = read(original_path)
        if (task["source_commit"] != SOURCE or task["weight_revision"] != REVISION
                or task["weight_sha256"] != WEIGHT or task["predict_options"] != OPTIONS):
            raise ValueError("Frozen source, weights or inference options changed")
        for k in ("roles", "data", "channel_order", "scaler_sha256"):
            if task[k] != old[k]: raise ValueError("Previously fixed permissions/scaler changed: "+k)
        if sha(root/domain/"SCALER.npz") != old["scaler_sha256"]:
            raise ValueError("Source-only scaler identity changed")
        for role in ("CAL", "QUERY"):
            idx = np.load(checked(workspace, task["indexes"][role]), allow_pickle=False)
            np.testing.assert_array_equal(idx, np.load(checked(workspace, old["indexes"][role]), allow_pickle=False))
            descriptor = records[domain,role]
            path = checked(workspace, descriptor)
            receipt = read(checked(workspace, dict(relative=descriptor["receipt_relative"], sha256=descriptor["receipt_sha256"])))
            if receipt["labels_read"] or receipt["count"] != len(idx):
                raise ValueError("Readout permissions or point count changed")
            for part in receipt["parts"]:
                if (part["source_commit"] != SOURCE or part["weight_revision"] != REVISION or part["weight_sha256"] != WEIGHT
                        or part["predict_options"] != OPTIONS or part["current_target_in_model_input"]
                        or not part["all_parameters_frozen"] or not part["model_state_RNG_unchanged"] or part["labels_read"]
                        or part["scaler_sha256"] != task["scaler_sha256"]):
                    raise ValueError("Actual inference checkpoint, options, causal access or frozen state changed")
                checked(workspace, part)
            bank = load_bank(path); formula_gate(bank)
            np.testing.assert_array_equal(bank["absolute_time_index"], idx)
            old_row = next(r for r in accepted["rows"] if r["domain"] == domain and r["method"] == "TimeRadar_V_A" and r["stream"] == "strict")
            old_path = workspace/f"results/native_v_conditioning_20261119_v1/banks/{domain}/A/{role}/BANK.npz"
            if sha(old_path) != old_row["bank_"+role+"_sha256"]:
                raise ValueError("Reused nativeV identity changed")
            previous = load_bank(old_path)
            for k in ("absolute_time_index", "available_index", "actual_normalized", "actual_raw"):
                np.testing.assert_array_equal(bank[k], previous[k])
            paths[domain,role] = path
    return seal, tasks, paths, accepted


def event_overlap(index, left_score, left_tau, right_score, right_tau, labels, left_events, right_events):
    """Descriptive sets at original CAL operations, never a fitted OR detector."""
    from .alarm_complementarity import confirmed
    a, b = np.asarray(left_score)>left_tau, np.asarray(right_score)>right_tau
    normal = labels[index] == 0
    result = []
    for stream, aa, bb in (("strict", a, b), ("confirmed3", confirmed(index,a), confirmed(index,b))):
        le = {e["event_id"]: e for e in left_events if e["stream"] == stream and e["main_eligible"]}
        re = {e["event_id"]: e for e in right_events if e["stream"] == stream and e["main_eligible"]}
        if set(le) != set(re): raise ValueError("Shared full event qualification required")
        for key in le:
            if (le[key]["start"],le[key]["end_exclusive"]) != (re[key]["start"],re[key]["end_exclusive"]):
                raise ValueError("Event identity mismatch")
        row = dict(stream=stream, complete_events=len(le),
            normal_FP_both=int(np.sum(normal & aa & bb)), normal_FP_TimesFM_only=int(np.sum(normal & aa & ~bb)),
            normal_FP_TimeRadar_only=int(np.sum(normal & ~aa & bb)), normal_no_alarm=int(np.sum(normal & ~aa & ~bb)),
            use="descriptive independent CAL operations only; not a deployed OR or cooperative innovation")
        for kind in ("any_hit", "one_to_one_matched"):
            sets = {"both": [], "TimesFM_only": [], "TimeRadar_only": [], "neither": []}
            for eid in le:
                flag = (bool(le[eid][kind]),bool(re[eid][kind]))
                sets[{(True,True):"both",(True,False):"TimesFM_only",(False,True):"TimeRadar_only",(False,False):"neither"}[flag]].append(eid)
            for name, ids in sets.items(): row[kind+"_"+name] = len(ids); row[kind+"_"+name+"_IDs"] = ids
        result.append(row)
    return result


def run(workspace, root, output):
    start = time.perf_counter(); workspace,root,output = (Path(p).resolve() for p in (workspace,root,output))
    if output.exists(): raise FileExistsError(output)
    seal,tasks,paths,old = terminal_inputs(workspace,root)
    manifest = read(checked(workspace,dict(relative=EXTERNAL_MANIFEST,sha256=EXTERNAL_MANIFEST_SHA)))
    old_events = read(checked(workspace,dict(relative=ACCEPTED_NATIVE_EVENTS,sha256=ACCEPTED_NATIVE_EVENTS_SHA)))
    rows,events,low,overlap,surfaces,checks = [],[],[],[],{},{}
    from .native_v_conditioning_curves import low_fpr_entry
    for domain in AXES:
        desc = manifest["labels"][domain]; lp = checked(workspace,dict(relative=desc["path"],sha256=desc["sha256"]))
        labels = load_bank(lp)[desc["key"]] if lp.suffix == ".npz" else np.load(lp,allow_pickle=False).reshape(-1)
        cal,query = (load_bank(paths[domain,r]) for r in ("CAL","QUERY"))
        points = np.load(checked(workspace,tasks[domain]["indexes"]["CAL"]),allow_pickle=False)
        rr,ee,surface,check = evaluate_arrays(domain,cal,query,points,labels,workspace/manifest["author_metric"])
        for r in rr: r.update(bank_CAL_sha256=sha(paths[domain,"CAL"]),bank_QUERY_sha256=sha(paths[domain,"QUERY"]))
        rows.extend(rr);events.extend(ee);checks[domain]=check
        ll,arrays = low_fpr_entry(query["absolute_time_index"],query["canonical_score"],labels,rr,dict(domain=domain,method=METHOD,seed=None))
        low.extend(ll)
        for k,v in zip(("TPR","FPR","precision","windows"),surface):surfaces[domain+"_"+k]=v
        for k,v in arrays.items():surfaces[domain+"_LOW_"+k]=v
        base = next(r for r in old["rows"] if r["domain"]==domain and r["method"]=="TimeRadar_V_A" and r["stream"]=="strict")
        vb = load_bank(workspace/f"results/native_v_conditioning_20261119_v1/banks/{domain}/A/QUERY/BANK.npz")
        ve = [e for e in old_events if e["domain"]==domain and e["method"]=="TimeRadar_V_A"]
        overlap.extend(dict(r,domain=domain) for r in event_overlap(query["absolute_time_index"],query["canonical_score"],rr[0]["threshold"],
            vb["canonical_score"],base["threshold"],labels,ee,ve))
    rows.extend(dict(r,reused_from_METRICS_sha256=ACCEPTED_NATIVE_METRICS_SHA,
        comparison_scope="same QUERY axis; method-specific training/context/scaler/CAL permissions retained; not architecture-matched")
        for r in old["rows"] if r["method"] in BASELINES)
    paired = paired_rows(rows)
    events.extend(dict(e,reused_from_EVENTS_sha256=ACCEPTED_NATIVE_EVENTS_SHA) for e in old_events if e["method"]=="TimeRadar_V_A")
    output.mkdir(parents=True)
    from .left_swat_replay_repeats import write_csv
    write_csv(output/"methods.csv",rows);write_csv(output/"LOW_FPR_SUPPLEMENT.csv",low);write_csv(output/"EVENT_OVERLAP.csv",overlap)
    write_csv(output/"PAIRED_METHODS.csv",paired)
    np.savez_compressed(output/"NEW_VUS_AND_LOW_CURVES.npz",**surfaces)
    (output/"EVENTS.json").write_text(json.dumps(events,indent=2,allow_nan=False),encoding="utf-8")
    result=dict(status="COMPLETED",rows=rows,paired=paired,low_fpr=low,event_overlap=overlap,formula_checks=checks,
        raw_seal_sha256=sha(root/"GLOBAL_RAW_BANK_SEAL.json"),bank_manifest=seal["banks"],old_metrics_sha256=ACCEPTED_NATIVE_METRICS_SHA,
        old_VUS_recomputed=False,actual_GPU_hours=seal["actual_GPU_hours"],cpu_wall_seconds=time.perf_counter()-start,
        interpretation="Frozen complete official pretrained multivariate forecasts; project median residual baseline, no Hybrid innovation/contamination/coordination claim")
    (output/"METRICS.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    for name in ("workspace","root","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();r=run(a.workspace,a.root,a.out)
    print("Completed",len(r["rows"]),"fixed method/stream rows")
