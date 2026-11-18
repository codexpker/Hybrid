"""Three paired adaptation seeds; reuse all accepted A and seed2025 B/C metrics."""
import argparse
import concurrent.futures
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from .left_exposure_evaluation import inputs,new_metrics,numerical_keys
from .left_response_preservation_evaluation import METRIC
from .left_swat_natural_u_evaluation import read_json,checked,write_json
from .left_swat_replay_repeats import write_csv
from .left_native_swat_metrics import sha

SEEDS=(2025,2026,2027)
OLD_BC=("results/left_current_u_exposure_metrics_20261118_v1/METRICS.json",
        "3d53bf346edf2bdfafe677a2487b13341b70528d1094eb85c0aa0066a3d7b3ca")
A_INPUTS={
    "SWAT":("results/left_author_swat_source_replay_repeats_metrics_20261118_v1/METRICS.json",
             "6833bee1fc48d74b270a1cfb9fcfe268a43686937a0d79520b4ab874551b0a8e"),
    "PSM":("results/left_author_psm_source_replay_metrics_20261118_v1/METRICS.json",
            "0ec57ff0474f5bcd88c60118a061bbe65cc220a54115e30c8957e4ee6ed10716")}


def seed_metric(item):
    domain,seed,arm,condition,task,scores,labels,metric_path=item
    rows,events,surface=new_metrics((domain,arm,condition,task,scores,labels,metric_path))
    for row in rows:row["seed"]=seed
    for row in events:row["seed"]=seed
    return rows,events,surface


def fixed_summary(rows,keys,group_fields):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k]for k in group_fields)].append(row)
    result=[]
    for group,chosen in groups.items():
        seeds=[r.get("seed")for r in chosen]
        if len(set(seeds))!=len(seeds):raise ValueError("A seed occurs more than once in a metric group")
        for metric in keys(chosen):
            values=[r[metric]for r in chosen];valid=[v for v in values if v is not None and np.isfinite(v)]
            complete=len(valid)==len(chosen)
            mean=float(np.mean(valid))if complete else None
            sd=float(np.std(valid,ddof=1))if complete and len(valid)>1 else None
            result.append(dict(zip(group_fields,group),metric=metric,n=len(chosen),n_valid=len(valid),
                mean=mean,sample_sd=sd,sample_sd_ddof=1,
                note="FIXED_REFERENCE_NO_SEED_SD"if seeds==[None]else
                     "UNDEFINED_IN_AT_LEAST_ONE_FIXED_SEED"if not complete else"paired adaptation seeds"))
    return result


def contrasts_of(rows):
    lookup={(r["domain"],r["seed"],r["method"],r["condition"],r["stream"]):r for r in rows}
    paired=[];interactions=[]
    for domain in ["SWAT","PSM"]:
        for seed in SEEDS:
            for stream in ["strict","confirmed3"]:
                for arm in ["TT","R"]:
                    for left,right in [("B","C"),("B","A"),("C","A")]:
                        a,b=[lookup[(domain,seed,arm,c,stream)]for c in [left,right]]
                        paired.append(dict(domain=domain,seed=seed,method=arm,contrast=f"{left}−{right}",stream=stream,
                            **{k:None if a[k]is None or b[k]is None else a[k]-b[k]for k in numerical_keys([a,b])}))
                rb,rc,tb,tc=[lookup[(domain,seed,a,c,stream)]for a,c in [("R","B"),("R","C"),("TT","B"),("TT","C")]]
                paired.append(dict(domain=domain,seed=seed,method="R_vs_TT",contrast="RB−TTB",stream=stream,
                    **{k:None if rb[k]is None or tb[k]is None else rb[k]-tb[k]for k in numerical_keys([rb,tb])}))
                interactions.append(dict(domain=domain,seed=seed,method="interaction",contrast="(RB−RC)−(TTB−TTC)",stream=stream,
                    **{k:None if any(r[k]is None for r in [rb,rc,tb,tc])else(rb[k]-rc[k])-(tb[k]-tc[k])
                       for k in numerical_keys([rb,rc,tb,tc])}))
    return paired,interactions


def evaluate(workspace,root,output,workers=4):
    started=time.monotonic()
    if output.exists():raise FileExistsError(output)
    pending,labels,identities=inputs(workspace,root,repeat_seeds=[2026,2027])
    old_path=checked(workspace/OLD_BC[0],OLD_BC[1]);old=read_json(old_path)
    rows=[dict(r)for r in old["rows"]if r["condition"]in ["B","C"]]
    assert len(rows)==16 and all(r["seed"]==2025 for r in rows)
    reused={"seed2025_BC_sha256":sha(old_path),"A":{}}
    for domain,(relative,expected)in A_INPUTS.items():
        path=checked(workspace/relative,expected);a=read_json(path)
        for r in a["rows"]:
            assert r["seed"]in SEEDS
            rows.append(dict(r,domain=domain,condition="A"if r["method"]in ["TT","R"]else"reference"))
        no_update=[r for r in a["references"]if r["method"]=="noU"]
        assert len(no_update)==2
        rows.extend(dict(r,domain=domain,condition="reference",seed=None)for r in no_update)
        reused["A"][domain]=dict(relative=relative,sha256=sha(path),rows=len(a["rows"]),noU_fixed_once=True)
    output.mkdir(parents=True,exist_ok=False);events=[];surfaces={};audits={}
    items=[(*item,labels[item[0]],workspace/METRIC)for item in pending]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers)as pool:
        for item,result in zip(items,pool.map(seed_metric,items)):
            rr,ee,surface=result;rows.extend(rr);events.extend(ee)
            domain,seed,arm,condition=item[:4]
            for k,v in zip(["TPR","FPR","precision","windows"],surface):surfaces[f"{domain}_{seed}_{arm}_{condition}_{k}"]=v
    for domain in ["SWAT","PSM"]:
        audits[domain]={2025:old["exposure_audits"][domain]}
        for seed in [2026,2027]:
            path=root/domain/f"seed{seed}/BUNDLE/audit_only/EXPOSURE_AUDIT.json"
            audits[domain][seed]=dict(sha256=sha(path),audit=read_json(path))
    assert len(rows)==88
    if len({(r["domain"],r["seed"],r["method"],r["condition"],r["stream"])for r in rows})!=88:
        raise ValueError("Duplicate or fabricated seed rows")
    paired,interactions=contrasts_of(rows)
    absolute_summary=fixed_summary(rows,numerical_keys,["domain","method","condition","stream"])
    paired_summary=fixed_summary(paired+interactions,numerical_keys,["domain","method","contrast","stream"])
    write_csv(output/"comparison.csv",rows);write_csv(output/"method_summary.csv",absolute_summary)
    write_csv(output/"contrasts.csv",paired);write_csv(output/"interaction.csv",interactions)
    write_csv(output/"paired_summary.csv",paired_summary);write_json(output/"NEW_EVENTS.json",events)
    np.savez_compressed(output/"NEW_VUS_SURFACES.npz",**surfaces)
    result=dict(rows=rows,contrasts=paired,interaction=interactions,method_summary=absolute_summary,
        paired_summary=paired_summary,identities=identities,reused=reused,exposure_audits=audits,
        n_seeds=3,sample_sd_ddof=1,noU_fixed_reference_no_fake_SD=True,query_points_not_concatenated=True,
        old_VUS_recomputed=False,new_scored_runs=16,total_BC_runs=24,
        cpu_wall_seconds=time.monotonic()-started,
        limits=["adaptation/sampling seeds share one pretrained initialization; development data",
                "repeated donor windows are not independent anomaly events or datasets",
                "event-time labels are not physical affected-channel truth or exact gradient contamination",
                "R and TT retain different natural second-half backgrounds; interaction is not stand-alone robustness proof"])
    write_json(output/"METRICS.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    result=evaluate(a.workspace,a.root,a.out,a.workers);print(f"Completed {len(result['rows'])} rows")
