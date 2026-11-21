"""Descriptive event/quantile/error attribution from already frozen banks only.

No neural model, GPU, forecast, ranking evaluator, fitted statistic or new
detection criterion is used. Derived concentration/interval statistics are
descriptions, never candidates for thresholding or anomaly-score selection.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


BASE_METRICS_SHA = "223dbb8d052cd72868d663feae6568449e2faeac1ae6c5a105a3f4c7519cb5e1"
SEAL_SHA = "49f22b0046b1f535ccec090af452958cdc4680c6f1706618a6bd8c93067ef963"
OLD_METRICS_SHA = "a3c31f2325174a02fb76bd76486013064c71e83e5909ed90368f9b46e3ce9ccb"
OLD_EVENTS_SHA = "6d54b9be99457c42f1c873fe345a30fafbfc09991e7d9983e8df8bf6d96b8789"
NEW_EVENTS_SHA = "427e5337252b00e5bee5af7838992bc562f4cbb25a324c4621db3ac6fd598bec"
PUBLICATION_SHA = "75da2dc12c1d8a288e50741e156d73a2c97f9aa1591cc2eaa83bcdcc6785c774"
MANIFEST_SHA = "420ae8fd09a1847ab523c0b4dfa15b6e6d80f087e7cd4efef27b9d74803189d8"
METHOD = "TimesFM3_frozen"
NATIVE = "TimeRadar_V_A"
PHASES = ("pre_normal", "onset_32", "history_transition_32_512", "steady_after512",
          "recovery_32", "recovery_32_512")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""): digest.update(b)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bank(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def confirmation(index, alarm):
    bits = np.asarray(alarm, bool); out = np.zeros(len(bits), bool)
    out[2:] = (bits[2:] & bits[1:-1] & bits[:-2]
               & (np.diff(index)[1:] == 1) & (np.diff(index)[:-1] == 1))
    return out


def fragments(index, bits):
    """Original positive fragments, splitting both gaps and false positions."""
    p = np.flatnonzero(bits)
    if not len(p): return []
    split = np.flatnonzero((np.diff(p) != 1) | (np.diff(index[p]) != 1))+1
    return list(np.split(p, split))


def phase_bounds(start, end, previous_end, next_start):
    return dict(pre_normal=(max(start-512, previous_end), start),
        onset_32=(start, min(start+32, end)),
        history_transition_32_512=(min(start+32, end), min(start+512, end)),
        steady_after512=(min(start+512, end), end),
        recovery_32=(end, min(end+32, next_start)),
        recovery_32_512=(min(end+32, next_start), min(end+512, next_start)))


def positions(index, bounds, labels=None, normal=False):
    a,b = bounds
    if b < a: raise ValueError("Ordered descriptive phase bounds required")
    p = np.arange(np.searchsorted(index,a), np.searchsorted(index,b))
    if normal:
        if labels is None: raise ValueError("Explicit posthoc annotation required")
        p = p[labels[index[p]] == 0]
    return p


def concentration(errors):
    e = np.asarray(errors, np.float64)
    if e.ndim != 2 or not np.isfinite(e).all() or np.any(e < 0):
        raise ValueError("Finite nonnegative squared errors per point/channel required")
    total = e.sum(axis=1); positive = total > 0
    ordered = np.sort(e, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered,axis=1)
    def ratio(numerator, denominator):
        out = np.full(len(total), np.nan)
        np.divide(numerator, denominator, out=out, where=positive)
        return out
    result = dict(top1_error_share=ratio(ordered[:,0],total),
        top3_error_share=ratio(ordered[:,:min(3,e.shape[1])].sum(axis=1),total),
        effective_error_channels=ratio(total**2,np.einsum("ij,ij->i",e,e)))
    for pct in (50,80):
        v = np.minimum((cumulative < total[:,None]*(pct/100)).sum(axis=1)+1,e.shape[1]).astype(float)
        v[~positive] = np.nan; result[f"channels_for{pct}_error"] = v
    result["one_channel_at_least80_error_fraction"] = np.where(positive,(result["top1_error_share"] >= .8).astype(float),np.nan)
    return result, np.argmax(e,axis=1), positive


def quantile_descriptors(actual, quantiles):
    """Marginal interval descriptions, with ties/zero widths explicit; no score."""
    x,q = np.asarray(actual), np.asarray(quantiles)
    if q.shape != (*x.shape,9) or not np.isfinite(q).all() or not np.isfinite(x).all():
        raise ValueError("Nine finite native quantiles for every actual point/channel")
    if np.any(np.diff(q,axis=-1) < 0): raise ValueError("Keep native sorted quantiles")
    width = q[...,8].astype(np.float64)-q[...,0].astype(np.float64)
    below,above = x < q[...,0], x > q[...,8]
    f = dict(width80_meanC=width.mean(axis=1), width80_medianC=np.median(width,axis=1),
        zero_width80_fraction=(width==0).mean(axis=1),
        below_q10_fraction=below.mean(axis=1),above_q90_fraction=above.mean(axis=1),
        inside_q10_q90_fraction=(~(below|above)).mean(axis=1),
        outside80_channel_count=(below|above).sum(axis=1).astype(float))
    lower_rank = np.zeros(x.shape,np.uint8)
    for j in range(9):
        f[f"actual_lt_q{(j+1)*10:02d}_fraction"] = (x<q[...,j]).mean(axis=1)
        f[f"actual_eq_q{(j+1)*10:02d}_fraction"] = (x==q[...,j]).mean(axis=1)
        lower_rank += (x>q[...,j])
    for j in range(10):f[f"strict_quantiles_below_actual_{j}_fraction"] = (lower_rank==j).mean(axis=1)
    return f


def features(pred, actual, errors, quantiles, native_v, index):
    c,top,positive = concentration(errors)
    f = dict(prediction_mse=errors.mean(axis=1),timeradar_V=np.asarray(native_v,np.float64),
        median_signed_error_meanC=(pred.astype(float)-actual.astype(float)).mean(axis=1),
        median_abs_error_meanC=np.abs(pred.astype(float)-actual.astype(float)).mean(axis=1),
        zero_total_error_fraction=(~positive).astype(float))
    f.update(c); f.update(quantile_descriptors(actual,quantiles))
    for name,v in (("actual",actual),("forecast",pred)):
        for lag in (1,32,512):
            a = np.full(len(index),np.nan)
            valid = index[lag:]-index[:-lag] == lag
            a[lag:] = np.where(valid,np.abs(v[lag:].astype(float)-v[:-lag].astype(float)).mean(axis=1),np.nan)
            f[f"{name}_delta{lag}_abs_meanC"] = a
    return f,top,positive


def describe(f, p):
    row = dict(points=len(p))
    for name,values in f.items():
        x = np.asarray(values)[p]; x = x[np.isfinite(x)]
        row[name+"_valid_points"] = len(x)
        row[name+"_mean"] = float(np.mean(x)) if len(x) else None
        if not name.endswith("fraction"):
            for k,q in (("p10",.1),("median",.5),("p90",.9)):
                row[name+"_"+k] = float(np.quantile(x,q)) if len(x) else None
    return row


def channel_rows(identity, p, errors, actual, pred, quant, names, top, positive):
    rows = []; total = float(errors[p].sum()) if len(p) else 0.
    for j,name in enumerate(names):
        err = errors[p,j]; count = int(np.sum(positive[p] & (top[p]==j)))
        rows.append(dict(identity,channel_index=j,channel_name=name,points=len(p),
            squared_error_sum=float(err.sum()) if len(p) else None,
            pooled_error_share=float(err.sum()/total) if total>0 else None,
            largest_error_point_count=count,positive_total_error_points=int(positive[p].sum()),
            actual_mean=float(actual[p,j].mean(dtype=np.float64)) if len(p) else None,
            forecast_mean=float(pred[p,j].mean(dtype=np.float64)) if len(p) else None,
            q10_mean=float(quant[p,j,0].mean(dtype=np.float64)) if len(p) else None,
            q90_mean=float(quant[p,j,8].mean(dtype=np.float64)) if len(p) else None))
    return rows


def group_name(a,b):
    return {(True,True):"both",(True,False):"TimesFM_only",(False,True):"TimeRadar_only",(False,False):"neither"}[bool(a),bool(b)]


def run(workspace, config, output):
    started = time.perf_counter(); w,output = Path(workspace).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    cfg = read(config)
    if cfg["baseline_metrics_sha256"] != BASE_METRICS_SHA or cfg["model_forward"] or cfg["new_detection_score"]:
        raise ValueError("Fixed cached diagnostic only")
    hashes = {}
    def fixed(rel,expected):
        p = w/rel
        if sha(p) != expected:raise ValueError("Frozen cached identity mismatch: "+rel)
        hashes[rel] = expected; return p
    seal = read(fixed("results/timesfm3_frozen_20261121_v1/GLOBAL_RAW_BANK_SEAL.json",SEAL_SHA))
    main = read(fixed("results/timesfm3_frozen_metrics_20261121_v1/METRICS.json",BASE_METRICS_SHA))
    old = read(fixed("results/native_v_conditioning_metrics_20261119_v1/METRICS.json",OLD_METRICS_SHA))
    old_events = read(fixed("results/native_v_conditioning_metrics_20261119_v1/EVENTS.json",OLD_EVENTS_SHA))
    new_events_path = fixed("results/timesfm3_frozen_metrics_20261121_v1/EVENTS.json",NEW_EVENTS_SHA)
    # Its identity is also bound by the accepted public review's event export.
    published = read(fixed("hybrid_ad/docs/experiments/timesfm3_frozen_20261121/PUBLICATION_BINDING.json",PUBLICATION_SHA))
    events_csv = fixed("hybrid_ad/docs/experiments/timesfm3_frozen_20261121/events.csv",published["public_result_files"]["events.csv"])
    with events_csv.open(newline="",encoding="utf-8") as f:accepted_events = list(csv.DictReader(f))
    events = read(new_events_path);hashes[new_events_path.relative_to(w).as_posix()] = sha(new_events_path)
    if len(accepted_events)!=len(events):raise ValueError("Accepted full event catalog required")
    for accepted,original in zip(accepted_events,events):
        for key in ("domain","method","stream","event_id","start","end_exclusive","main_eligible","any_hit","one_to_one_matched"):
            expected="" if original[key] is None else str(original[key])
            if accepted[key]!=expected:raise ValueError("Published event identity or selection changed")
    manifest = read(fixed("results/tf_conditional_energy_20261119_v1/EXTERNAL_CACHE_MANIFEST_V2.json",MANIFEST_SHA))
    summary,event_rows,channel,normal_rows,normal_fragments_rows,groups = [],[],[],[],[],[]
    validation=[]
    for domain in cfg["domains"]:
        desc = next(b for b in seal["banks"] if b["domain"]==domain and b["role"]=="QUERY")
        tf = bank(fixed(desc["relative"],desc["sha256"]))
        prev = next(r for r in old["rows"] if (r["domain"],r["method"],r["stream"])==(domain,NATIVE,"strict"))
        vv = bank(fixed(f"results/native_v_conditioning_20261119_v1/banks/{domain}/A/QUERY/BANK.npz",prev["bank_QUERY_sha256"]))
        task = read(w/f"results/timesfm3_frozen_20261121_v1/{domain}/TASK.json")
        idx = tf["absolute_time_index"]; pred=tf["prediction_normalized"];actual=tf["actual_normalized"]
        quant=tf["quantiles_normalized"];error=tf["squared_error_per_channel"]
        for k in ("absolute_time_index","available_index","actual_normalized","actual_raw"):
            np.testing.assert_array_equal(tf[k],vv[k])
        np.testing.assert_array_equal(error,(pred.astype(float)-actual.astype(float))**2)
        np.testing.assert_array_equal(tf["canonical_score"],error.mean(axis=1))
        np.testing.assert_array_equal(pred,quant[...,4]);np.testing.assert_array_equal(idx,tf["available_index"])
        np.testing.assert_array_equal(idx-512,tf["history_start"]);np.testing.assert_array_equal(idx,tf["history_end_exclusive"])
        lp=manifest["labels"][domain];path=fixed(lp["path"],lp["sha256"])
        labels=bank(path)[lp["key"]] if path.suffix==".npz" else np.load(path,allow_pickle=False).reshape(-1)
        f,top,positive = features(pred,actual,error,quant,vv["canonical_score"],idx)
        names=task["channel_order"]
        for label_name,mask in (("normal_labeled",labels[idx]==0),("attack_labeled",labels[idx]==1)):
            identity=dict(domain=domain,scope="whole_QUERY",label_group=label_name)
            summary.append(dict(identity,**describe(f,np.flatnonzero(mask))))
            channel.extend(channel_rows(identity,np.flatnonzero(mask),error,actual,pred,quant,names,top,positive))
        for stream in cfg["streams"]:
            rr=next(r for r in main["rows"] if(r["domain"],r["method"],r["stream"])==(domain,METHOD,stream))
            vr=next(r for r in main["rows"] if(r["domain"],r["method"],r["stream"])==(domain,NATIVE,stream))
            a=tf["canonical_score"]>rr["threshold"];b=vv["canonical_score"]>vr["threshold"]
            if stream=="confirmed3":a,b=confirmation(idx,a),confirmation(idx,b)
            ee={e["event_id"]:e for e in events if(e["domain"],e["method"],e["stream"])==(domain,METHOD,stream)}
            ve={e["event_id"]:e for e in old_events if(e["domain"],e["method"],e["stream"])==(domain,NATIVE,stream)}
            original=sorted(ee.values(),key=lambda e:e["start"])
            acc={}
            for i,e in enumerate(original):
                v=ve[e["event_id"]]
                if(e["start"],e["end_exclusive"],e["main_eligible"])!=(v["start"],v["end_exclusive"],v["main_eligible"]):raise ValueError("Event identity mismatch")
                if not e["main_eligible"]:continue
                ep=positions(idx,(e["start"],e["end_exclusive"]))
                if bool(a[ep].any())!=e["any_hit"] or bool(b[ep].any())!=v["any_hit"]:raise ValueError("Original event grouping changed")
                category=group_name(e["any_hit"],v["any_hit"])
                bounds=phase_bounds(e["start"],e["end_exclusive"],original[i-1]["end_exclusive"] if i else 0,
                                    original[i+1]["start"] if i+1<len(original) else len(labels))
                for phase in PHASES:
                    pp=positions(idx,bounds[phase],labels,phase.startswith(("pre_","recovery_")))
                    for subset,p in (("all",pp),("TimesFM_original_alarm",pp[a[pp]])):
                        ident=dict(domain=domain,stream=stream,event_group=category,event_id=e["event_id"],start=e["start"],end_exclusive=e["end_exclusive"],phase=phase,point_subset=subset,
                            phase_start=bounds[phase][0],phase_stop=bounds[phase][1],qualified_event=True,
                            TimesFM_any_hit=e["any_hit"],TimeRadar_any_hit=v["any_hit"],
                            TimesFM_matched=e["one_to_one_matched"],TimeRadar_matched=v["one_to_one_matched"])
                        dr=describe(f,p)
                        dr.update(TimesFM_alarm_points=int(a[p].sum()),TimeRadar_alarm_points=int(b[p].sum()),
                            TimesFM_alarm_fraction=float(a[p].mean()) if len(p) else None,
                            TimeRadar_alarm_fraction=float(b[p].mean()) if len(p) else None)
                        row=dict(ident,**dr);event_rows.append(row)
                        acc.setdefault((category,phase,subset),[]).append((p,row))
                        if subset=="TimesFM_original_alarm":channel.extend(channel_rows(ident,p,error,actual,pred,quant,names,top,positive))
            import itertools
            for category,phase,subset in itertools.product(("both","TimesFM_only","TimeRadar_only","neither"),PHASES,("all","TimesFM_original_alarm")):
                entries=acc.get((category,phase,subset),[])
                p=np.concatenate([e[0]for e in entries]) if entries else np.empty(0,np.int64)
                dr=describe(f,p)
                group=dict(domain=domain,stream=stream,event_group=category,phase=phase,point_subset=subset,
                    event_count=len(entries),nonempty_events=sum(len(e[0])>0 for e in entries),
                    unique_points=len(np.unique(p)),pooled_exposures_allow_overlapping_context=True,**dr)
                for key in dr:
                    if key.endswith("_mean"):
                        values=[e[1][key]for e in entries if e[1][key] is not None]
                        group["event_macro_"+key]=float(np.mean(values)) if values else None
                for name,alarm in (("TimesFM",a),("TimeRadar",b)):
                    values=[float(alarm[e[0]].mean()) for e in entries if len(e[0])]
                    group[name+"_phase_alarm_fraction_pooled"]=float(alarm[p].mean()) if len(p) else None
                    group[name+"_phase_alarm_fraction_macro"]=float(np.mean(values)) if values else None
                groups.append(group)
            normal=labels[idx]==0
            for name,bits,accepted in (("TimesFM",a,rr),("TimeRadar",b,vr)):
                ff=fragments(idx,bits&normal)
                assert len(ff)==accepted["normal_alarm_fragments"]
                assert sum(len(p)for p in ff)==accepted["FP"]
                for number,p in enumerate(ff):
                    ident=dict(domain=domain,stream=stream,model=name,fragment_id=number,start=int(idx[p[0]]),end_exclusive=int(idx[p[-1]])+1,
                        simultaneous_other_model_alarm_points=int((b if name=="TimesFM" else a)[p].sum()))
                    normal_fragments_rows.append(dict(ident,**describe(f,p)))
                ident=dict(domain=domain,stream=stream,normal_group=name+"_all_FP",fragments=len(ff),
                    maximum_fragment_points=max(map(len,ff),default=0),single_point_fragments=sum(len(p)==1 for p in ff))
                pp=np.flatnonzero(bits&normal);normal_rows.append(dict(ident,**describe(f,pp)))
                channel.extend(channel_rows(ident,pp,error,actual,pred,quant,names,top,positive))
            for name,mask in (("both",a&b),("TimesFM_only",a&~b),("TimeRadar_only",~a&b),("neither",~a&~b)):
                pp=np.flatnonzero(mask&normal)
                normal_rows.append(dict(domain=domain,stream=stream,normal_group=name,**describe(f,pp)))
            validation.append(dict(domain=domain,stream=stream,threshold_TimesFM=rr["threshold"],threshold_TimeRadar=vr["threshold"],
                normal_FP_TimesFM=int((a&normal).sum()),normal_FP_TimeRadar=int((b&normal).sum()),all_original_counts_unchanged=True))
        del tf,vv,f,quant,error,pred,actual
    output.mkdir(parents=True)
    for name,rows in (("QUERY_DISTRIBUTIONS",summary),("EVENT_PHASES",event_rows),("EVENT_GROUP_PHASES",groups),
                      ("CHANNEL_CONTRIBUTIONS",channel),("NORMAL_SUMMARY",normal_rows),("NORMAL_FRAGMENTS",normal_fragments_rows)):
        write_csv(output/(name+".csv"),rows)
    for rel,digest in hashes.items():
        if sha(w/rel)!=digest:raise ValueError("Frozen input changed during read-only diagnostic")
    receipt=dict(status="COMPLETED_READ_ONLY",config=cfg,config_sha256=sha(config),input_hashes=hashes,validation=validation,
        tables={"QUERY_DISTRIBUTIONS":len(summary),"EVENT_PHASES":len(event_rows),"EVENT_GROUP_PHASES":len(groups),
            "CHANNEL_CONTRIBUTIONS":len(channel),"NORMAL_SUMMARY":len(normal_rows),"NORMAL_FRAGMENTS":len(normal_fragments_rows)},
        CPU_wall_seconds=time.perf_counter()-started,GPU_hours=0,model_forwards=0,training_updates=0,
        original_scores_changed=False,thresholds_changed=False,ranking_metrics_recomputed=False,
        quantiles_used_for_new_score=False,scope="descriptive cached public-event analysis; no attention/history/channel ablation or causal/physical attribution")
    write(output/"ANALYSIS.json",receipt);return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--config",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();r=run(a.workspace,a.config,a.out);print(json.dumps({k:r[k]for k in ("status","tables","CPU_wall_seconds","GPU_hours")},indent=2))
