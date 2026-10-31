"""One all-group label release, per-entity metrics/Aff, fixed run-cluster report."""
from pathlib import Path
import csv
import json

import numpy as np

from ..config import sha256
from .evaluation import evaluate_fixed_scores
from .multi_readout import verify_label_unlock
from .official_affiliation import load_official_affiliation
from .pipeline import _json_safe
from .readout_manifest import checked,json_write
from .run_statistics import run_cluster_statistics
from . import run_statistics as statistics_module
from . import official_affiliation as affiliation_module


def write_csv(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(key for row in rows for key in row))
    if not fields:raise ValueError("An empty CSV table requires an explicit schema")
    with Path(path).open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            clean=_json_safe(row)
            writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value for key,value in clean.items()})


def maximum_matching_count(predicted, truth):
    # Separate augmenting-path implementation for the synthetic test oracle.
    matched = {}
    def visit(i, seen):
        lo, hi = predicted[i]
        for j, (_, start, end) in enumerate(truth):
            if lo < end and hi > start and j not in seen:
                seen.add(j)
                if j not in matched or visit(matched[j], seen):
                    matched[j] = i
                    return True
        return False
    for i in range(len(predicted)):
        visit(i, set())
    return len(matched)

def evaluate_group(seal_path,catalog_path,query_path,affiliation_identity_path,output,*,profile):
    # This verification rehashes every raw bank, RNG, score and source identity;
    # no binary label file is opened on any failure before the group is complete.
    seal,entities=verify_label_unlock(seal_path,catalog_path,query_path,profile)
    callback,aff_provenance=load_official_affiliation(affiliation_identity_path)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fixed_inputs={str(seal_path):sha256(seal_path),str(catalog_path):seal["catalog"]["sha256"],
                  str(query_path):seal["query_manifest"]["sha256"],str(affiliation_identity_path):aff_provenance["identity_sha256"]}
    try:
        return _evaluate_released_group(seal,entities,callback,aff_provenance,output,seal_path,query_path,profile,fixed_inputs)
    except Exception as exc:
        json_write(output/"evaluation_failure.json",dict(status="FAILED_NO_COMPLETE_REPORT",error_type=type(exc).__name__,
            error=str(exc),source_seal_sha256=sha256(seal_path),labels_permission="released complete group; no complete metrics report committed"))
        raise


def _evaluate_released_group(seal,entities,callback,aff_provenance,output,seal_path,query_path,profile,fixed_inputs):
    source_seal_sha=sha256(seal_path)
    rows=[];event_rows=[];alarm_rows=[];aff_rows=[]
    for entity,q in entities.items():
        label_path=Path(q["label_path"])
        if sha256(label_path)!=q["label_sha256_from_frozen_parent"]:
            raise ValueError("Frozen original entity label identity mismatch")
        labels=np.load(label_path,mmap_mode="r",allow_pickle=False)
        expected=(np.arange(profile["points"])>=q["event_start_0based"]) & (np.arange(profile["points"])<q["event_end_exclusive"])
        if (labels.shape!=(profile["points"],) or not np.array_equal(labels,expected.astype(labels.dtype))
                or q["event_start_0based"]!=profile["event_start"] or q["event_end_exclusive"]!=profile["points"]):
            raise ValueError("Original binary event labels disagree with frozen entity metadata; never relabel")
        for key,spec in seal["scores"].items():
            if spec["entity"]!=entity:continue
            path=checked(spec["file"],seal_path)
            with np.load(path,allow_pickle=False) as bank:index,score=bank["time_index"].copy(),bank["score"].copy()
            y=labels[index]
            coverage=[dict(event_id=entity+"_event",start=q["event_start_0based"],end_exclusive=q["event_end_exclusive"],
                           eligible_complete=True,scored_points=int((y==1).sum()),total_points=q["event_end_exclusive"]-q["event_start_0based"])]
            result=evaluate_fixed_scores(index,y,score,spec["threshold"],coverage,affiliation=callback)
            coordinates=output/"coordinates"/(key+".npz");coordinates.parent.mkdir(parents=True,exist_ok=True)
            origins=np.asarray(q["origins"],dtype="int64")
            np.savez_compressed(coordinates,time_index=index,label=y,complete_event_valid=result.pop("complete_event_valid"),
                forecast_origin=origins,history_start=origins-100,history_end_exclusive=origins,target_end_exclusive=origins+5,
                original_entity_length=np.asarray(profile["points"]),source_label_sha256=np.asarray(q["label_sha256_from_frozen_parent"]))
            predicted=[(a["start"],a["end_exclusive"]) for a in result["alarms"]]
            truth=[(coverage[0]["event_id"],coverage[0]["start"],coverage[0]["end_exclusive"])]
            if sum(a["matched_event"] is not None for a in result["alarms"])!=maximum_matching_count(predicted,truth):
                raise ValueError("Greedy event count differs from independent maximum matching")
            base=dict(method=spec["method"],seed=spec["seed"],fault_id=q["fault_id"],run_id=q["run_id"],entity=entity)
            metrics=result["metrics"];pred=score>spec["threshold"]
            normal_hours=float((y==0).sum()*profile["sample_interval_minutes"]/60)
            metrics.update(point_Recall=metrics["TP"]/(metrics["TP"]+metrics["FN"]),
                one_to_one_event_recall=sum(a["matched_event"] is not None for a in result["alarms"])/len(coverage),
                delay_units="original sample points; delay_minutes uses frozen sample_interval_minutes",
                pre_fault_alarm=bool(pred[index<q["event_start_0based"]].any()),
                alarm_continuing_across_fault_start=any(a["start"]<q["event_start_0based"]<a["end_exclusive"] for a in result["alarms"]),
                normal_observation_hours=normal_hours,
                normal_alarm_fragments_per_hour=metrics["normal_alarm_fragments"]/normal_hours if normal_hours else float("nan"),
                normal_duration_reason=None if normal_hours else "NO_NORMAL_SCORED_OBSERVATION",
                point_Precision_reason=None if np.isfinite(metrics["point_Precision"]) else "NO_PREDICTED_POSITIVES",
                operating_condition=q.get("operating_condition"),
                operating_condition_reason=None if q.get("operating_condition") is not None else "NOT_PROVIDED_BY_FROZEN_QUERY_METADATA",
                first_scored_original_index=int(index[0]),scored_end_exclusive=int(index[-1])+1,
                source_score_sha256=spec["file"]["sha256"],source_label_sha256=q["label_sha256_from_frozen_parent"])
            rows.append({**base,**metrics})
            for e in result["events"]:
                e["delay_minutes"]=None if e["delay_sample_points"] is None else e["delay_sample_points"]*profile["sample_interval_minutes"]
                e.update(original_start=q["event_start_0based"],original_end_exclusive=q["event_end_exclusive"],
                    delay_reason=None if e["delay_sample_points"] is not None else "NOT_DETECTED" if e["complete"] else "INCOMPLETE_EVENT",
                    pre_fault_alarm=metrics["pre_fault_alarm"],alarm_continuing_across_fault_start=metrics["alarm_continuing_across_fault_start"],
                    delay_caveat="zero delay may be a continuing pre-fault alarm; consult crossing/pre-fault flags")
                event_rows.append({**base,**e})
            alarm_rows.extend({**base,**row} for row in result["alarms"])
            aff_rows.extend({**base,**row} for row in result["affinity"])
    statistics=run_cluster_statistics(rows,profile)
    # Recheck exactly the scored inputs used, without performing another fit or
    # changing any method/threshold after labels became available.
    if sha256(seal_path)!=source_seal_sha:raise ValueError("Source seal changed during fixed evaluation")
    if any(sha256(path)!=digest for path,digest in fixed_inputs.items()):
        raise ValueError("Frozen evaluation definition/official identity inputs changed during evaluation")
    if any(sha256(Path(aff_provenance["directory"])/name)!=digest for name,digest in aff_provenance["files"].items()):
        raise ValueError("Frozen official Affiliation implementation changed during evaluation")
    for spec in seal["scores"].values():checked(spec["file"],seal_path)
    for q in entities.values():
        if sha256(q["label_path"])!=q["label_sha256_from_frozen_parent"]:
            raise ValueError("Frozen labels changed during fixed evaluation")
    np.savez_compressed(output/"paired_run_bootstrap.npz",**statistics.pop("bootstrap_arrays"))
    write_csv(output/"entity_metrics.csv",rows)
    write_csv(output/"event_results.csv",event_rows)
    write_csv(output/"alarm_events.csv",alarm_rows,fields=["method","seed","fault_id","run_id","entity","block","start","end_exclusive","matched_event"])
    write_csv(output/"entity_affiliation.csv",aff_rows)
    for name,key in (("macro_summary","macro"),("entity_seed_mean","entity_seed_mean"),("fault_summary","per_fault"),
                     ("run_summary","per_run"),("worst_cases","worst"),("paired_run_intervals","intervals")):
        write_csv(output/(name+".csv"),statistics[key])
    seed_rows=[dict(method=row["method"],metric=row["metric"],seed=seed,value=row["per_seed"][seed])
               for row in statistics["macro"] for seed in (0,1,2)]
    write_csv(output/"seed_summary.csv",seed_rows)
    write_csv(output/"failed_entity_metrics.csv",statistics["failures"],fields=["method","seed","fault_id","run_id","AP","ROC","reason"])
    report=dict(status="FIXED_MULTI_ENTITY_EVALUATION_COMPLETE",metrics=rows,events=event_rows,alarms=alarm_rows,
                per_entity_Aff=aff_rows,statistics=statistics,overall_Aff_F1=None,
                overall_Aff_reason="NO_OFFICIAL_GLOBAL_DEFINITION_ACROSS_ENTITIES",affiliation=aff_provenance,
                source_seal_sha256=source_seal_sha,labels_use="metrics only after all9xentity banks and all18xentity scores sealed",
                threshold_search=False,point_adjustment=False,
                profile=profile,original_query_entities=len(entities),original_query_events=len(entities),
                event_row_semantics="one original event per entity repeated for each fixed method/seed; not independent events",
                evaluation_code_sha256={Path(path).name:sha256(path) for path in (__file__,statistics_module.__file__,affiliation_module.__file__)})
    json_write(output/"evaluation.json",_json_safe(report))
    files={str(path.relative_to(output)).replace("\\","/"):sha256(path) for path in output.rglob("*") if path.is_file()}
    json_write(output/"evaluation_manifest.json",dict(status="COMPLETE_FIXED_EVALUATION_ARCHIVE",files=files,
        metric_rows=len(rows),event_rows=len(event_rows),source_seal_sha256=source_seal_sha,
        overall_Aff_F1=None,requires_absence_of_failure_marker=True))
    return report
