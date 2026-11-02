"""Independent CPU label-array evaluation. No training/model/query file loader.

Only already sealed scores are evaluated. Labels remain on their complete
original run axis. Missing scores are never normal points or interpolated.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ..candidate_experiment import confirmed_alarm
from ..engine.events import binary_events, greedy_assign, real_blocks
from ..engine.evaluation import evaluate_point_ranking
from ..engine.official_affiliation import load_official_affiliation, OfficialAffiliationBlocked
from .chunked_bank import read_records

STATISTICS = dict(status="NOT_COMPUTED", block_points=4096, n_replicates=2000,
                  rng_seed=99173, scope="paired_time_block_sensitivity_not_domain_inference")


def _ratio(a, b):
    return float(a/b) if b else None


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _clean(value):
    if isinstance(value, dict):
        return {k:_clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def _coverage(labels, scored, start):
    events=[]
    for number,(lo,hi) in enumerate(binary_events(labels),1):
        count=int(scored[lo:hi].sum()); complete=count==hi-lo
        boundary=lo==0 or hi==len(labels)
        events.append(dict(event_id="E%06d"%number,start=start+lo,end_exclusive=start+hi,
            local_start=lo,local_end=hi,total_points=hi-lo,scored_points=count,
            coverage_ratio=count/(hi-lo),complete=complete,
            touches_original_start=lo==0,touches_original_end=hi==len(labels),
            boundary_censored=boundary,main_eligible=bool(complete and not boundary),
            legacy_complete_only_eligible=bool(complete),
            exclusion_reasons=(["INCOMPLETE_SCORE_COVERAGE"] if not complete else [])+
                              (["ORIGINAL_RUN_BOUNDARY_CENSORED"] if boundary else [])))
    return events


def _event_view(labels, scored, alarm, available, events, start, eligibility):
    original=start+np.arange(len(labels),dtype=np.int64)
    eligible=[e for e in events if e[eligibility]]
    event_valid=scored & (labels==0)
    for e in eligible:
        event_valid[e["local_start"]:e["local_end"]]=True
    blocks=real_blocks(original,event_valid)
    alarms=[]; matched=set(); predicted=0
    block_rows=[]
    for bi,ix in enumerate(blocks):
        lo,hi=int(original[ix[0]]),int(original[ix[-1]])+1
        truth=[(e["event_id"],e["start"],e["end_exclusive"]) for e in eligible
               if e["start"]>=lo and e["end_exclusive"]<=hi]
        spans=[(int(original[ix[a]]),int(original[ix[b-1]])+1) for a,b in binary_events(alarm[ix])]
        assignments=greedy_assign(spans,truth)
        predicted+=len(spans)
        for span,hit in zip(spans,assignments):
            if hit is not None: matched.add(hit)
            alarms.append(dict(block=bi,start=span[0],end_exclusive=span[1],matched_event=hit))
        block_rows.append(dict(block=bi,start=lo,end_exclusive=hi,points=len(ix),
                               true_events=len(truth),predicted_events=len(spans)))
    event_rows=[]; any_hits=0
    for e in events:
        ok=bool(e[eligibility]); lo,hi=e["local_start"],e["local_end"]
        hit_local=np.flatnonzero(alarm[lo:hi])+lo
        first=int(hit_local[0]) if len(hit_local) else None
        hit=bool(len(hit_local)) if ok else None
        if ok: any_hits+=int(hit)
        delay_ok=ok and not e["boundary_censored"]
        event_rows.append(dict(event_id=e["event_id"],eligible=ok,
            any_hit=hit,one_to_one_matched=(e["event_id"] in matched) if ok else None,
            first_observed_alarm_index=(start+first) if first is not None else None,
            first_available_alarm_index=int(available[first]) if first is not None else None,
            delay_points=int(available[first]-e["start"]) if delay_ok and hit else None,
            any_hit_missed=not hit if ok else None,
            matching_missed=e["event_id"] not in matched if ok else None,
            delay_right_censored=bool(delay_ok and not hit),
            censor_after_points=e["total_points"] if delay_ok and not hit else None,
            delay_NA_reason=("INELIGIBLE_EVENT" if not ok else "UNKNOWN_ORIGINAL_EVENT_BOUNDARY"
                            if e["boundary_censored"] else "RIGHT_CENSORED_NO_HIT" if not hit else None)))
    m=len(matched); ntrue=len(eligible)
    metrics=dict(M=m,Npred=predicted,Ntrue=ntrue,
        one_to_one_eventPrecision=_ratio(m,predicted),one_to_one_eventRecall=_ratio(m,ntrue),
        one_to_one_eventF1=_ratio(2*m,predicted+ntrue),
        eventPrecision_NA_reason=None if predicted else "NO_PREDICTED_ALARM_SEGMENTS",
        eventRecall_NA_reason=None if ntrue else "NO_ELIGIBLE_TRUE_EVENTS",
        eventF1_NA_reason=None if predicted+ntrue else "NO_PREDICTED_OR_TRUE_EVENTS",
        auxiliary_any_hit_events=any_hits,auxiliary_any_hit_recall=_ratio(any_hits,ntrue),
        matching_rule="chronological_first_unmatched_overlap_greedy_not_maximum_matching",
        any_hit_is_not_matching_recall=True,
        delay_definition="first_available_alarm_inside_event_any_hit; matching status separately reported",
        delay_units="original_sample_points",delay_eligible_events=sum(e[eligibility] and not e["boundary_censored"] for e in events),
        delay_right_censored_events=sum(e["delay_right_censored"] for e in event_rows),
        point_exposure_in_event_valid_blocks=int(event_valid.sum()))
    return dict(metrics=metrics,events=event_rows,alarms=alarms,blocks=block_rows),blocks


def _normal(labels, scored, alarm, start):
    original=start+np.arange(len(labels),dtype=np.int64)
    blocks=[]; fragments=clipped=hit_blocks=0
    for bi,ix in enumerate(real_blocks(original,scored & (labels==0))):
        spans=binary_events(alarm[ix]); fragments+=len(spans); hit_blocks+=bool(spans)
        for a,b in spans:
            lo,hi=int(ix[a]),int(ix[b-1])+1
            clipped+=bool((lo>0 and labels[lo-1]==1 and alarm[lo-1]) or
                          (hi<len(labels) and labels[hi]==1 and alarm[hi]))
        blocks.append(dict(block=bi,start=int(original[ix[0]]),end_exclusive=int(original[ix[-1]])+1,
                           exposure_points=len(ix),any_alarm=bool(spans),alarm_fragments=len(spans)))
    exposure=int((scored & (labels==0)).sum()); fp=int((alarm & (labels==0)).sum())
    return dict(normal_exposure_points=exposure,normal_false_alarm_points=fp,
        normal_FPR=_ratio(fp,exposure),normal_alarm_fragments=fragments,
        normal_alarm_fragments_per_1000_points=_ratio(1000*fragments,exposure),
        normal_contiguous_blocks=len(blocks),normal_blocks_with_alarm=hit_blocks,
        normal_block_any_alarm_fraction=_ratio(hit_blocks,len(blocks)),
        normal_block_lengths=[b["exposure_points"] for b in blocks],
        normal_rate_NA_reason=None if exposure else "NO_VALID_NORMAL_EXPOSURE",
        anomaly_boundary_clipped_normal_alarm_fragments=clipped,
        fragment_definition="normal-region fragments including boundary clipping; not new alarm onsets",
        physical_time_rate=None,physical_time_NA_reason="NO_VERIFIED_PHYSICAL_CLOCK"),blocks


def _aff_blocks(labels, alarm, blocks, start, callback, unavailable):
    rows=[]
    for bi,ix in enumerate(blocks):
        row=dict(block=bi,start=start+int(ix[0]),end_exclusive=start+int(ix[-1])+1,
                 points=len(ix),Aff_P=None,Aff_R=None,Aff_F1=None)
        gt=binary_events(labels[ix]); pred=binary_events(alarm[ix])
        row.update(true_events=len(gt),predicted_events=len(pred))
        if not gt: row["reason"]="NO_GROUND_TRUTH_EVENTS"
        elif callback is None: row["reason"]=unavailable
        else:
            # Local coordinates within ONE truly contiguous original interval.
            ar=callback(pred,gt,(0,len(ix)))
            p,r=float(ar["precision"]),float(ar["recall"])
            f=2*p*r/(p+r) if np.isfinite(p) and np.isfinite(r) and p+r>0 else None
            row.update(Aff_P=p if np.isfinite(p) else None,Aff_R=r if np.isfinite(r) else None,
                       Aff_F1=f,reason=None if f is not None else "OFFICIAL_UNDEFINED")
        rows.append(row)
    if len(rows)==1:
        global_result={k:rows[0][k] for k in ("Aff_P","Aff_R","Aff_F1","reason")}
    else:
        global_result=dict(Aff_P=None,Aff_R=None,Aff_F1=None,
            reason="MULTIPLE_REAL_BLOCKS_NO_GLOBAL_AGGREGATION" if rows else "NO_EVENT_VALID_BLOCKS")
    return dict(blocks=rows,global_result=global_result)


def evaluate_arrays(index, score, full_labels, threshold, *, bank_complete,
                    original_length, original_start=0, warmup_points=512,
                    valid=None, available_index=None, invalid_reasons=None,
                    affiliation_identity=None):
    """Shared label-array CPU core; actual calls should use evaluate_bank_arrays.

    ``bank_complete`` is mandatory and the bank wrapper checks it from disk.
    ``full_labels`` is a complete original-run array, never labels[index].
    """
    if bank_complete is not True:
        raise ValueError("Only a sealed bank.complete input may be evaluated")
    labels=np.asarray(full_labels); index=np.asarray(index); score=np.asarray(score,dtype=np.float64)
    if (type(original_length) is not int or original_length<1 or labels.shape!=(original_length,)
            or labels.dtype.kind not in "biuf" or not np.isfinite(labels).all() or not np.isin(labels,[0,1]).all()
            or type(original_start) is not int or type(warmup_points) is not int
            or original_start<0 or warmup_points<0 or not np.isfinite(threshold)):
        raise ValueError("Complete binary original labels/length, fixed clock and source threshold required")
    if (index.ndim!=1 or not np.issubdtype(index.dtype,np.integer) or score.shape!=index.shape
            or np.any(np.diff(index)<=0) or np.any(index<original_start)
            or np.any(index>=original_start+original_length)):
        raise ValueError("Ordered unique original score indices within the full label axis required")
    valid=np.isfinite(score) if valid is None else np.asarray(valid)
    available=index.copy() if available_index is None else np.asarray(available_index)
    reasons=[None]*len(index) if invalid_reasons is None else list(invalid_reasons)
    if (valid.dtype!=np.bool_ or valid.shape!=index.shape or available.shape!=index.shape
            or not np.issubdtype(available.dtype,np.integer) or not np.array_equal(available,index)
            or len(reasons)!=len(index) or np.any(valid & ~np.isfinite(score))):
        raise ValueError("Valid flags, finite valid scores and exact frozen arrived availability required")
    for i in np.flatnonzero(~valid):
        if not isinstance(reasons[i],str) or not reasons[i]:
            raise ValueError("Every invalid raw score needs a preserved reason")
    local=index-original_start
    if np.any(local<warmup_points):
        raise ValueError("Score appears inside the declared warmup")
    full_index=original_start+np.arange(original_length,dtype=np.int64)
    scored=np.zeros(original_length,bool); scored[local[valid]]=True
    full_score=np.full(original_length,np.nan); full_score[local]=score
    available_full=full_index.copy(); available_full[local]=available
    coverage_reason=["NOT_IN_DECLARED_BANK_COVERAGE"]*original_length
    coverage_reason[:min(warmup_points,original_length)]=["WARMUP"]*min(warmup_points,original_length)
    for pos in local[valid]:coverage_reason[pos]="SCORED"
    for pos in np.flatnonzero(~valid): coverage_reason[local[pos]]=reasons[pos]
    events=_coverage(labels,scored,original_start)
    y=labels[scored]
    ranking=dict(evaluated_points=int(scored.sum()),positive_points=int((y==1).sum()),
        negative_points=int((y==0).sum()),positive_fraction=float(y.mean()) if len(y) else None,
        unscored_points=int((~scored).sum()),AP=None,trapezoid_PR_AUC=None,ROC_AUC=None)
    if not len(y): ranking["ranking_metric_NA_reason"]="NO_VALID_SCORES"
    elif len(np.unique(y))!=2: ranking["ranking_metric_NA_reason"]="SINGLE_CLASS_SCORED_LABELS"
    else:
        try:
            ranking.update(evaluate_point_ranking(full_index[scored],full_score[scored],labels,
                scored,query_start=original_start)["metrics"])
        except ModuleNotFoundError as exc:
            ranking["ranking_metric_NA_reason"]=("SKLEARN_UNAVAILABLE_NOT_COMPUTED"
                if exc.name and exc.name.startswith("sklearn") else
                "RANKING_DEPENDENCY_UNAVAILABLE_NOT_COMPUTED:"+str(exc.name))
    callback=None; aff_meta=None; aff_unavailable="OFFICIAL_AFFILIATION_IDENTITY_NOT_BOUND"
    if affiliation_identity is not None:
        try: callback,aff_meta=load_official_affiliation(affiliation_identity)
        except OfficialAffiliationBlocked as exc:
            if "dependency unavailable" not in str(exc): raise
            aff_unavailable=str(exc)
    point_alarm=np.zeros(original_length,bool)
    point_alarm[local[valid]]=score[valid]>threshold
    confirmed=np.zeros(original_length,bool)
    if valid.any():
        confirmed[local[valid]]=confirmed_alarm(index[valid],score[valid],threshold)
    streams={}
    for name,alarm in (("strict_point",point_alarm),("confirmed3_current",confirmed)):
        tp=int((alarm & scored & (labels==1)).sum()); fp=int((alarm & scored & (labels==0)).sum())
        fn=int((~alarm & scored & (labels==1)).sum()); tn=int((~alarm & scored & (labels==0)).sum())
        main,blocks=_event_view(labels,scored,alarm,available_full,events,original_start,"main_eligible")
        legacy,_=_event_view(labels,scored,alarm,available_full,events,original_start,"legacy_complete_only_eligible")
        normal,normal_blocks=_normal(labels,scored,alarm,original_start)
        streams[name]=dict(point=dict(TP=tp,FP=fp,TN=tn,FN=fn,Precision=_ratio(tp,tp+fp),
            Recall=_ratio(tp,tp+fn),F1=_ratio(2*tp,2*tp+fp+fn),
            Precision_NA_reason=None if tp+fp else "NO_PREDICTED_POSITIVE_POINTS",
            Recall_NA_reason=None if tp+fn else "NO_VALID_POSITIVE_POINTS",
            F1_NA_reason=None if 2*tp+fp+fn else "NO_PREDICTED_OR_TRUE_POSITIVE_POINTS",
            zero_denominator_policy="NA; no alarm precision is not zero",threshold=float(threshold)),
            main_events=main,legacy_complete_only_events=legacy,normal=normal,normal_blocks=normal_blocks,
            affiliation=_aff_blocks(labels,alarm,blocks,original_start,callback,aff_unavailable),
            Range_VUS=dict(status="NOT_EVALUATED",reason="OFFICIAL_IMPLEMENTATION_EXECUTION_NOT_BOUND",
                slidingWindow=100,Range_window=100,VUS_windowSize=200,thre=250,version="opt_mem",
                global_result=None,global_reason="NO_INVENTED_CROSS_BLOCK_AGGREGATION",
                blocks=[dict(**b,status="NOT_EVALUATED",reason="OFFICIAL_IMPLEMENTATION_EXECUTION_NOT_BOUND") for b in main["blocks"]]))
    result=dict(schema_version=1,ranking=ranking,event_coverage=events,streams=streams,
        coverage=dict(original_length=original_length,original_start=original_start,
            declared_raw_records=len(index),valid_scored_points=int(scored.sum()),
            arrived_candidate_points=max(0,original_length-warmup_points),
            missing_arrived_raw_records=max(0,original_length-warmup_points)-len(index),
            complete_arrived_raw_coverage=len(index)==max(0,original_length-warmup_points),
            final_original_point_has_raw_record=bool(len(index) and index[-1]==original_start+original_length-1),
            final_original_point_valid=bool(scored[-1]),
            invalid_raw_records=int((~valid).sum()),warmup_points=min(warmup_points,original_length),
            unscored_points=int((~scored).sum()),original_events=len(events),
            main_eligible_events=sum(e["main_eligible"] for e in events),
            boundary_censored_events=sum(e["boundary_censored"] for e in events)),
        point_coverage=[dict(index=int(full_index[i]),label=int(labels[i]),valid=bool(scored[i]),
            reason=str(coverage_reason[i]),score=float(full_score[i]) if np.isfinite(full_score[i]) else None,
            strict_point=bool(point_alarm[i]) if scored[i] else None,
            confirmed3_current=bool(confirmed[i]) if scored[i] else None) for i in range(original_length)],
        statistics=dict(STATISTICS),affiliation_source=aff_meta,
        affiliation_requested_binding=(dict(path=str(affiliation_identity),sha256=_sha(affiliation_identity))
                                       if affiliation_identity is not None else None),
        input_identity=dict(label_array_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
            index_sha256=hashlib.sha256(index.tobytes()).hexdigest(),
            score_sha256=hashlib.sha256(score.tobytes()).hexdigest(),
            valid_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
            available_index_sha256=hashlib.sha256(available.tobytes()).hexdigest(),
            invalid_reasons_sha256=hashlib.sha256(json.dumps(reasons,ensure_ascii=False).encode("utf-8")).hexdigest(),
            labels_dtype=labels.dtype.str,index_dtype=index.dtype.str))
    return _clean(result)


def evaluate_bank_arrays(bank_directory, full_labels, threshold, *, expected_manifest_sha256,
                         original_length, original_start=0, warmup_points=512,
                         stream=None, affiliation_identity=None):
    """Independent entry: authenticate completed raw bank, then evaluate labels.

    Does not load a label file. The separately authorized CPU caller supplies the
    original label array and binds its file identity; no model process needs it.
    """
    directory=Path(bank_directory); manifest_path=directory/"manifest.json"
    if _sha(manifest_path)!=expected_manifest_sha256:
        raise ValueError("Sealed bank manifest SHA mismatch")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("bank.complete must be true before any evaluation")
    if manifest["contract"]["threshold"]!=threshold:
        raise ValueError("Evaluation threshold differs from bank source-fixed threshold")
    ranges=manifest["contract"]["ranges"]
    streams={tuple(r[k] for k in ("domain","entity","run")) for r in ranges}
    if stream is None:
        if len(streams)!=1: raise ValueError("One explicit original stream required; never concatenate runs")
        stream=next(iter(streams))
    stream=tuple(stream)
    if stream not in streams: raise ValueError("Unknown original stream")
    expected_all=[tuple(r[k] for k in ("domain","entity","run"))+(i,)
                  for r in ranges for i in range(r["start"],r["end"])]
    if len(expected_all)!=manifest["committed_records"]:
        raise ValueError("Complete bank count contradicts original coverage ranges")
    rows=[]; actual_count=0
    for actual_count,row in enumerate(read_records(directory),1):
        key=tuple(row[k] for k in ("domain","entity","run","index"))
        if actual_count>len(expected_all) or key!=expected_all[actual_count-1]:
            raise ValueError("Raw bank has missing, duplicate or reordered original indices")
        if key[:3]==stream:
            if type(row.get("valid")) is not bool: raise ValueError("Raw score valid flag required")
            rows.append((row["index"],row.get("score"),row["valid"],row["available_index"],row.get("reason")))
    if actual_count!=len(expected_all): raise ValueError("Bank records end before declared complete coverage")
    index=np.array([r[0] for r in rows],dtype=np.int64)
    score=np.array([np.nan if r[1] is None else r[1] for r in rows],dtype=np.float64)
    result=evaluate_arrays(index,score,full_labels,threshold,bank_complete=True,
        original_length=original_length,original_start=original_start,warmup_points=warmup_points,
        valid=np.array([r[2] for r in rows],bool),available_index=np.array([r[3] for r in rows],np.int64),
        invalid_reasons=[r[4] for r in rows],affiliation_identity=affiliation_identity)
    result["bank_identity"]=dict(manifest_sha256=expected_manifest_sha256,contract=manifest["contract"],
        stream=list(stream),complete=True,global_bank_records=actual_count)
    return result


def save_evaluation(directory, result, *, index, score, valid, available_index, full_labels, provenance,
                    invalid_reasons=None):
    """Persist exact calculation inputs/code identity and per-event audit tables.

    Caller passes the exact arrays used for this result; hashes are checked.
    Never overwrites an earlier report. No label/scientific file is opened here.
    """
    arrays=dict(index=np.asarray(index),score=np.asarray(score,dtype=np.float64),
        valid=np.asarray(valid),available_index=np.asarray(available_index),full_labels=np.asarray(full_labels))
    reasons=[None]*len(arrays["index"]) if invalid_reasons is None else list(invalid_reasons)
    if hashlib.sha256(json.dumps(reasons,ensure_ascii=False).encode("utf-8")).hexdigest()!=result["input_identity"]["invalid_reasons_sha256"]:
        raise ValueError("Saved invalid reasons differ from evaluated raw reasons")
    arrays["invalid_reasons"]=np.asarray(["" if r is None else r for r in reasons],dtype=str)
    arrays["invalid_reason_is_none"]=np.asarray([r is None for r in reasons],dtype=bool)
    for name,key in (("index","index_sha256"),("score","score_sha256"),
                     ("valid","valid_sha256"),("full_labels","label_array_sha256"),
                     ("available_index","available_index_sha256")):
        if hashlib.sha256(arrays[name].tobytes()).hexdigest()!=result["input_identity"][key]:
            raise ValueError("Saved computation inputs differ from evaluated arrays")
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(directory/"calculation_inputs.npz",**arrays)
    code_paths=[Path(__file__),Path(__file__).parents[1]/"engine/events.py",
                Path(__file__).parents[1]/"engine/evaluation.py",
                Path(__file__).parents[1]/"engine/official_affiliation.py",
                Path(__file__).parents[1]/"candidate_experiment.py",Path(__file__).parent/"chunked_bank.py"]
    receipt=dict(provenance=provenance,calculation_inputs_sha256=_sha(directory/"calculation_inputs.npz"),
        code=[dict(path=str(p),sha256=_sha(p)) for p in code_paths])
    (directory/"result.json").write_text(json.dumps(_clean(result),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (directory/"provenance.json").write_text(json.dumps(_clean(receipt),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tables={"event_coverage":result["event_coverage"],"point_coverage":result["point_coverage"]}
    for stream,value in result["streams"].items():
        for view in ("main_events","legacy_complete_only_events"):
            for table in ("events","alarms"):
                tables[stream+"_"+view+"_"+table]=value[view][table]
        tables[stream+"_normal_blocks"]=value["normal_blocks"]
        tables[stream+"_aff_blocks"]=value["affiliation"]["blocks"]
    for name,rows in tables.items():
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (directory/(name+".csv")).open("w",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=keys); writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v
                             for k,v in row.items()} for row in rows)
    return receipt
