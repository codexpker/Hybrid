"""Fixed36-bank Stage-A CPU evaluation glue; original metrics, no fitting.

Only a root-accepted complete36 raw seal and separate truth admission permit
execution. Score/threshold math, event/Aff definitions and full-axis confirmation
are delegated to the existing evaluator. Phase summaries never create events.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from . import controlled_wave_evaluation as legacy
from . import controlled_wave as wave
from . import evaluation as backend
from . import scoring,chunked_bank
from .named_evaluate_cli import point_pair,paired_events
from .evaluate_cli import _leaves,PERFORMANCE_DIRECTIONS
from ..engine.events import binary_events


OPERATION="HYBRID_V1_REFERENCE_CENTER_A_FIXED36_EVALUATION"
METHODS=("M_window_matched","M_ref_center","B_ref_fixed_history_mean")
KINDS=("N","LEVEL","NORMAL_DRIFT","REFERENCE_POLLUTED_LEVEL")
PHASES=dict(pre=[1024,1280],transition=[1280,1792],steady=[1792,2560],recovery=[2560,3072])
RESOURCES=dict(total_seconds_including_cleanup=1800,TERM_seconds=1795,cleanup_seconds=5,
    AS_GiB=8,threads=1,GPUh=0,stage_cumulative_CPUh_cap=1,automatic_retry=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    with Path(path).open("x",encoding="utf-8") as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write("\n")


def case_specs():
    return [dict(case_id="run%d_%s"%(run,kind),process_run_id=run,paired_model_seed=seed,kind=kind,
        event=[1280,2560] if kind in ("LEVEL","REFERENCE_POLLUTED_LEVEL") else None)
        for run,seed in ((2,2025),(3,2026),(4,2027)) for kind in KINDS]


def expected_keys():return [(c["case_id"],method) for c in case_specs() for method in METHODS]


def descriptor_contract(banks):
    if [(b.get("case_id"),b.get("method")) for b in banks]!=expected_keys():
        raise PermissionError("Exactly twelve ordered cases times three methods; no aliases")
    if len({str(Path(b["directory"]).resolve()) for b in banks})!=36:
        raise PermissionError("One immutable raw bank cannot masquerade as several methods or seeds")
    cases={c["case_id"]:c for c in case_specs()}
    for desc in banks:
        baseline=desc["method"]==METHODS[2];case=cases[desc["case_id"]]
        if (desc.get("calibration_origin")!="source" or desc.get("active_components")!=(["r_pred"] if baseline else ["r_tf","r_ms","r_pred"])
                or desc.get("source_seed")!=(None if baseline else case["paired_model_seed"])
                or desc.get("inference_seed")!=(None if baseline else case["paired_model_seed"])):
            raise PermissionError("Original own source CAL and fixed process/model pairing required")


def bound_json(asset):
    if set(asset)!={"path","sha256"} or sha(asset["path"])!=asset["sha256"]:
        raise PermissionError("Exact accepted metadata bytes required")
    return json.loads(Path(asset["path"]).read_text(encoding="utf-8"))


def bank_header(desc,manifest,calibration):
    spans=[dict(domain="CONTROLLED_REFERENCE_A",entity="weak_amplitude",run=desc["case_id"],start=1024,end=3072)]
    if (manifest.get("complete") is not True or manifest.get("committed_records")!=2048
            or manifest["contract"]["ranges"]!=spans or manifest["contract"]["identity"]!=desc["identity"]
            or manifest["contract"]["threshold"]!=calibration["threshold"]
            or desc["identity"]["calibration_sha256"]!=desc["calibration"]["sha256"]):
        raise PermissionError("Accepted complete original2048 bank or own threshold differs")
    wave.validate_calibration(calibration,desc["active_components"],scoring)


def collect_scalars(desc,records):
    rows=[]
    for number,row in enumerate(records):
        if (number>=2048 or type(row.get("index")) is not int or row["index"]!=1024+number
                or type(row.get("available_index")) is not int
                or row.get("available_index")!=row["index"]
                or (row["domain"],row["entity"],row["run"])!=("CONTROLLED_REFERENCE_A","weak_amplitude",desc["case_id"])):
            raise ValueError("Ordered complete original arrived axis required before truth")
        rows.append(legacy.scalar_record(row,desc["calibration_values"]))
    if len(rows)!=2048:raise ValueError("Missing raw tail or coverage; no truth may open")
    return dict(index=np.asarray([r[0] for r in rows],np.int64),
        score=np.asarray([r[1] for r in rows],np.float64),valid=np.asarray([r[2] for r in rows],bool),
        available_index=np.asarray([r[3] for r in rows],np.int64),invalid_reasons=[r[4] for r in rows])


def prelabel_banks(cfg):
    """All36 identity/source-score gates finish before the sole truth opener."""
    descriptor_contract(cfg["banks"])
    accepted=bound_json(cfg["whole_raw_seal"])
    if (accepted.get("approved_by")!="root" or not str(accepted.get("status","")).startswith("ACCEPTED")
            or accepted.get("banks")!=cfg["banks"]):
        raise PermissionError("Whole accepted36-bank descriptor seal required")
    arrays={};calibrations={}
    for desc in cfg["banks"]:
        manifest=bound_json(dict(path=str(Path(desc["directory"])/"manifest.json"),sha256=desc["manifest_sha256"]))
        cal=bound_json(desc["calibration"]);bank_header(desc,manifest,cal)
        key=(desc["case_id"],desc["method"])
        arrays[key]=collect_scalars(dict(desc,calibration_values=cal),chunked_bank.read_records(desc["directory"]))
        calibrations[key]=cal
    return arrays,calibrations


def load_truth(asset):
    """Sole truth opener, called only after complete accepted raw/scalar gates."""
    if sha(asset["path"])!=asset["sha256"]:raise PermissionError("Admitted truth bytes differ")
    with np.load(asset["path"],allow_pickle=False) as archive:labels={key:archive[key] for key in archive.files}
    validate_truth(labels)
    return labels


def validate_truth(labels):
    if set(labels)!={c["case_id"] for c in case_specs()}:raise ValueError("Exact twelve original truth arrays")
    for case in case_specs():
        y=labels[case["case_id"]];expected=np.zeros(3072,np.int8)
        if case["event"]:expected[1280:2560]=1
        if y.dtype.kind not in "biu" or y.shape!=(3072,) or not np.array_equal(y,expected):
            raise ValueError("Original event/normal-drift truth and axis cannot be shifted or relabelled")


def steady_diagnostics(result):
    """Predeclared phase counts from already confirmed ORIGINAL-axis alarms.

    Unknown scores stay unknown and split observed miss segments; they are not
    false negatives or normal. Original complete-event metrics remain untouched.
    """
    rows=[r for r in result["point_coverage"] if 1792<=r["index"]<2560]
    output={}
    for stream,field in (("strict_point","strict_point"),("confirmed3_current","confirmed3_current")):
        positive=[r for r in rows if r["label"]==1]
        valid=[r for r in positive if r["valid"]]
        hits=sum(bool(r[field]) for r in valid);unknown=len(positive)-len(valid)
        miss=np.zeros(3072,bool)
        for row in valid:miss[row["index"]]=not bool(row[field])
        lengths=[hi-lo for lo,hi in binary_events(miss)]
        output[stream]=dict(interval=[1792,2560],phase_is_not_an_event=True,
            positive_points=len(positive),valid_positive_points=len(valid),unknown_positive_points=unknown,
            detected_positive_points=hits,missed_valid_positive_points=len(valid)-hits,
            Recall=float(hits/len(valid)) if valid else None,
            longest_observed_consecutive_miss_points=max(lengths,default=0) if positive else None,
            longest_miss_lower_bound_only=bool(unknown),
            complete_steady_all_missed=(hits==0) if positive and not unknown else None,
            NA_reason="NO_POSITIVE_POINTS_IN_PHASE" if not positive else "INCOMPLETE_SCORE_COVERAGE" if unknown else None)
    return output


def summarize_result(result):
    return dict(ranking=result["ranking"],coverage=result["coverage"],
        streams={key:dict(point=value["point"],main_events=dict(metrics=value["main_events"]["metrics"]),
            legacy_complete_only_events=dict(metrics=value["legacy_complete_only_events"]["metrics"]),
            normal=value["normal"],affiliation=value["affiliation"],Range_VUS=value["Range_VUS"])
            for key,value in result["streams"].items()},
        phases=legacy.phase_counts(result,PHASES),steady=steady_diagnostics(result))


def paired_block_summary(per_case):
    """Same descriptive arithmetic as old aggregate: all3/mean/sampleSD/worst.

    B_ref uses three distinct process blocks, not three trained seed replicas.
    No CI, significance test or concatenated timeline/ground-truth sequence.
    """
    output=[];directions=dict(PERFORMANCE_DIRECTIONS)
    for stream in ("strict_point","confirmed3_current"):
        directions["/steady/"+stream+"/Recall"]="minimum"
        directions["/steady/"+stream+"/longest_observed_consecutive_miss_points"]="maximum"
    for kind in KINDS:
        for method in METHODS:
            blocks=[per_case[("run%d_%s"%(run,kind),method)] for run in (2,3,4)]
            leaves=[dict(_leaves(b)) for b in blocks]
            paths=sorted(set(directions).union(*(set(x) for x in leaves)))
            summary={}
            for path in paths:
                values=[x.get(path) for x in leaves];direction=directions.get(path)
                if any(v is None for v in values):
                    summary[path]=dict(all3=values,mean=None,SD=None,worst_paired_block=None,
                        reason="NOT_AVAILABLE_FOR_ALL3_PAIRED_BLOCKS");continue
                worst=int(np.argmin(values) if direction=="minimum" else np.argmax(values)) if direction else None
                summary[path]=dict(all3=values,mean=float(np.mean(values)),SD=float(np.std(values,ddof=1)),
                    minimum=min(values),maximum=max(values),worst_paired_block=(2,3,4)[worst] if worst is not None else None,
                    worst_direction=direction or "NA_DENOMINATOR_OR_NONPERFORMANCE_FIELD")
            output.append(dict(kind=kind,method=method,paired_process_runs=[2,3,4],
                paired_neural_model_seeds=[2025,2026,2027] if method!=METHODS[2] else None,
                deterministic_baseline_CAL_views=1 if method==METHODS[2] else None,
                independent_domains=False,data_seed_model_seed_confounded=method!=METHODS[2],metrics=summary))
    return output


def exploratory_screen(per_case):
    """Only the frozen2pp and normal-cost screen, never automatic research GO."""
    def delta(kind,getter,left=METHODS[0]):
        values=[]
        for run in (2,3,4):
            key="run%d_%s"%(run,kind);a=getter(per_case[(key,left)]);b=getter(per_case[(key,METHODS[1])])
            values.append(None if a is None or b is None else b-a)
        complete=all(v is not None for v in values)
        return dict(all3=values,mean=float(np.mean(values)) if complete else None,
            minimum=min(values) if complete else None,maximum=max(values) if complete else None,
            status="COMPLETE" if complete else "INCOMPLETE")
    recall=lambda row:(None if row["steady"]["confirmed3_current"]["unknown_positive_points"] else
        row["steady"]["confirmed3_current"]["Recall"])
    fpr=lambda row:(row["streams"]["strict_point"]["normal"]["normal_FPR"]
        if row["coverage"]["complete_arrived_raw_coverage"] and not row["coverage"]["invalid_raw_records"] else None)
    shifts={kind:delta(kind,recall) for kind in ("LEVEL","REFERENCE_POLLUTED_LEVEL")}
    for value in shifts.values():
        value["exploratory_2pp_pass"]=(value["mean"]>=.02 and value["minimum"]>=0) if value["status"]=="COMPLETE" else None
    normal={kind:delta(kind,fpr) for kind in ("N","NORMAL_DRIFT")}
    for value in normal.values():
        value["normal_cost_within_frozen_tolerance"]=(value["mean"]<=.002 and value["maximum"]<=.005) if value["status"]=="COMPLETE" else None
    simple={kind:delta(kind,recall,left=METHODS[2]) for kind in ("LEVEL","REFERENCE_POLLUTED_LEVEL")}
    return dict(ref_minus_window_steady_confirmed3=shifts,ref_minus_window_normal_strict_FPR=normal,
        ref_minus_B_ref_steady_confirmed3=simple,absolute_metrics_required=True,
        scope="Exploratory fixed candidate screen only; not sustained-detection success, U robustness or automatic B authorization",
        automatic_research_GO=False,CI=None,significance_tests=False)


def require_dependencies(identity):
    legacy.require_metric_dependencies()
    from ..engine.official_affiliation import load_official_affiliation
    load_official_affiliation(identity)
    if "torch" in sys.modules:raise PermissionError("CPU evaluator must not import Torch")


def source_closure():
    return dict(legacy.required_closure(),**{str(Path(__file__).resolve()):sha(__file__)})


def admit(config,release,trusted):
    cfg=json.loads(Path(config).read_text());signed=json.loads(Path(release).read_text())
    permissions=dict(evaluation=True,labels=True,training=False,target_fit=False,model=False,GPU=False,
                     score_transform=False,threshold_selection=False)
    if (sha(release)!=trusted or cfg.get("status")!="ROOT_BOUND" or signed.get("status")!="APPROVED"
            or signed.get("approved_by")!="root" or cfg.get("operation")!=OPERATION or signed.get("operation")!=OPERATION
            or signed.get("allowed_actions")!=["fixed36_evaluate"] or signed.get("config_sha256")!=sha(config)
            or cfg.get("permissions")!=permissions or signed.get("permissions")!=permissions
            or cfg.get("resources")!=RESOURCES or signed.get("resources")!=RESOURCES
            or cfg.get("source_sha256")!=source_closure() or signed.get("source_sha256")!=source_closure()
            or cfg.get("output")!=signed.get("output") or Path(cfg["output"]).exists()
            or any(signed.get(key)!=cfg.get(key) for key in ("whole_raw_seal","truth","truth_admission","affiliation_identity"))):
        raise PermissionError("Exact root CPU evaluation/truth release and fresh output required")
    descriptor_contract(cfg["banks"])
    # Root accepted whole raw and truth admission JSONs; no truth-body hash/read here.
    seal=bound_json(cfg["whole_raw_seal"]);admission=bound_json(cfg["truth_admission"])
    if (seal.get("approved_by")!="root" or not str(seal.get("status","")).startswith("ACCEPTED")
            or seal.get("banks")!=cfg["banks"] or admission.get("approved_by")!="root"
            or not str(admission.get("status","")).startswith("ACCEPTED") or admission.get("truth")!=cfg["truth"]):
        raise PermissionError("Accepted full36raw and separately admitted truth required")
    bound_json(cfg["affiliation_identity"]);require_dependencies(cfg["affiliation_identity"]["path"])
    return cfg


def run(cfg):
    arrays,calibrations=prelabel_banks(cfg)
    labels=load_truth(cfg["truth"])  # sole truth opener, after ALL36 source scalar gates
    out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=False)
    results={};summaries={};pairs=[];metric_rows=[]
    for desc in cfg["banks"]:
        key=(desc["case_id"],desc["method"]);a=arrays[key];y=labels[key[0]]
        result=backend.evaluate_arrays(**a,full_labels=y,threshold=calibrations[key]["threshold"],
            bank_complete=True,original_length=3072,original_start=0,warmup_points=1024,
            affiliation_identity=cfg["affiliation_identity"]["path"])
        legacy.reject_dependency_na(result)
        backend.save_evaluation(out/key[0]/key[1],result,full_labels=y,
            provenance=dict(descriptor=desc,truth=cfg["truth"],whole_raw_seal=cfg["whole_raw_seal"],
                truth_admission=cfg["truth_admission"],source_sha256=cfg["source_sha256"]),**a)
        results[key]=result;summaries[key]=summarize_result(result)
        metric_rows.append(dict(case_id=key[0],method=key[1],summary=summaries[key]))
    for case in case_specs():
        for left in (METHODS[0],METHODS[2]):
            key=case["case_id"];effect,common=point_pair(arrays[(key,left)],arrays[(key,METHODS[1])],labels[key],0)
            legacy.reject_dependency_na(effect)
            events=paired_events(results[(key,left)],results[(key,METHODS[1])])
            folder=out/"pairs"/key/(left+"__vs__"+METHODS[1]);folder.mkdir(parents=True)
            dump(folder/"paired_events.json",events)
            np.savez_compressed(folder/"common_point_inputs.npz",index=common,full_labels=labels[key],
                reference_score=arrays[(key,left)]["score"][np.searchsorted(arrays[(key,left)]["index"],common)],
                candidate_score=arrays[(key,METHODS[1])]["score"][np.searchsorted(arrays[(key,METHODS[1])]["index"],common)])
            row=dict(case_id=key,reference=left,candidate=METHODS[1],common_point_effect=effect,
                CI=None,independent_domains=False,data_model_seed_confounded=True)
            dump(folder/"summary.json",row);pairs.append(row)
    summary=dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT",methods=metric_rows,comparisons=pairs,
        paired_block_descriptives=paired_block_summary(summaries),exploratory_screen=exploratory_screen(summaries),
        raw_banks=36,case_count=12,methods_count=3,score_transform=False,threshold_fit=False,
        CI=None,significance_tests=False,semantic_truth_aliases=False,independent_domains=False,
        caution="run2/3/4 are paired with model2025/26/27; B_ref is one deterministic method, not3 trained seeds")
    dump(out/"SUMMARY.json",summary);dump(out/"CONFIG.json",cfg)
    if "torch" in sys.modules:raise PermissionError("Unexpected Torch import; evaluation not valid")
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--release");p.add_argument("--trusted-release-sha256");p.add_argument("--execute",action="store_true")
    args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="ROOT_PENDING_PREVIEW",banks_opened=False,truth_opened=False,metrics=False)));return
    if not args.release or not args.trusted_release_sha256:raise PermissionError("Separate root fixed36 truth release required")
    cfg=admit(args.config,args.release,args.trusted_release_sha256)
    print(json.dumps(dict(status=run(cfg)["status"])))


if __name__=="__main__":main()
