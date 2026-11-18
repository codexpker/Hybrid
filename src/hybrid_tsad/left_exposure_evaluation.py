"""Evaluate fixed current-U B/C replacements, reusing accepted A/reference rows."""
import argparse
import concurrent.futures
import numbers
from pathlib import Path
import time

import numpy as np

from .left_native_swat_metrics import continuous_metrics,load_author_metric,sha
from .left_response_preservation_evaluation import baseline,score_bank,METRIC
from .left_swat_natural_u_evaluation import checked,read_json,write_json
from .left_swat_replay_repeats import write_csv
from .timercd_swat_evaluation import evaluate_score


def inputs(workspace,root,repeat_seeds=None):
    seal=read_json(root/"GLOBAL_RAW_BANK_SEAL.json")
    domains=seal["eligible_domains"]
    if seal["status"]!="COMPLETED"or not seal["no_labels_decoded_by_learner"]:
        raise ValueError("Require a sealed complete label-free-learner stage")
    cases=[(d,s,root/d/f"seed{s}")for d in domains for s in repeat_seeds]if repeat_seeds else[(d,2025,root/d)for d in domains]
    expected={str((folder/f"{a}_{c}/{r}.npz").relative_to(root)).replace("\\","/")
              for d,s,folder in cases for a in ["TT","R"]for c in ["B","C"]for r in ["CAL","QUERY"]}
    ids={b["relative"].replace("\\","/"):b["sha256"]for b in seal["banks"]}
    if set(ids)!=expected or len(seal["jobs"])!=4*len(cases)or any(j["OS_exit_code"]!=0 for j in seal["jobs"]):
        raise ValueError("Incomplete fixed B/C tasks")
    pending=[];identities={"seal_sha256":sha(root/"GLOBAL_RAW_BANK_SEAL.json"),"GPU_hours":seal["actual_GPU_hours"]}
    tasks_by_domain={}
    for domain,seed,case in cases:
        task=read_json(case/"TASK.json");frozen=read_json(case/"FROZEN.json")
        checked(case/"TASK.json",frozen["task_sha256"])
        checked(case/"SAMPLING_PLAN.npz",frozen["plan_sha256"])
        from .left_exposure_run import old_a_plan
        if sha(old_a_plan(workspace,domain,seed))!=frozen["plan_sha256"]:
            raise ValueError("Each condition must reuse its own accepted A sampler, not seed2025 copies")
        if domain in tasks_by_domain:
            previous=tasks_by_domain[domain]
            if any(task[k]!=previous[k]for k in ["source","data","roles","score_axes","labels",
                                                "scoring","CAL","training","sampler"]if k in task or k in previous):
                raise ValueError("Frozen scientific task changed across adaptation seeds")
        tasks_by_domain[domain]=task
        with np.load(case/"BUNDLE/learner/SLOT_MAP.npz")as z:slots=z["replacement_row"].copy()
        if slots.shape!=(2000,128)or np.sum(slots>=0)!=25600:raise ValueError("Replacement count changed")
        for arm in ["TT","R"]:
            for condition in ["B","C"]:
                name=arm+"_"+condition;folder=case/name
                train=read_json(folder/"train/RECEIPT.json");out=read_json(folder/"RECEIPT.json")
                bind=read_json(folder/"EXPOSURE_BINDING.json")
                manifest_path=case/"BUNDLE/learner"/(condition+"_MANIFEST.json")
                manifest=read_json(checked(manifest_path,train["exposure_manifest_sha256"]))
                if (train["updates"]!=2000 or train["batch"]!=256 or train["native_forward_calls"]!=2000
                    or train["seed"]!=seed or train["initial_python_step"]!=task["source"]["python_step"]
                    or train["final_python_step"]!=task["source"]["python_step"]+2000
                    or train["source_checkpoint_sha256"]!=task["source"]["checkpoint"]["sha256"]
                    or train["scaler_sha256"]!=task["source"]["scaler"]["sha256"]
                    or train["plan_sha256"]!=frozen["plan_sha256"]or train["labels_read"]or train["CAL_QUERY_read"]
                    or train["donor_labels_or_identity_read"]or train["replaced_T_slots"]!=25600
                    or out["checkpoint_sha256"]!=train["checkpoint_sha256"]
                    or out["python_step"]!=train["final_python_step"]or out["seed"]!=seed or out["arm"]!=arm
                    or out["source_checkpoint_sha256"]!=task["source"]["checkpoint"]["sha256"]
                    or out["scaler_sha256"]!=task["source"]["scaler"]["sha256"]or not out["source_checkpoint_unchanged"]
                    or not bind["only_first_half_T_modified"]or not bind["numeric_bundle_only"]
                    or bind["manifest_sha256"]!=sha(manifest_path)
                    or bind["slot_map_sha256"]!=sha(case/"BUNDLE/learner/SLOT_MAP.npz")
                    or bind["window_bank_sha256"]!=manifest["window_bank"]["sha256"]
                    or manifest["slot_map"]["sha256"]!=bind["slot_map_sha256"]
                    or manifest["scaler_sha256"]!=task["source"]["scaler"]["sha256"]
                    or manifest["base_sampling_plan_sha256"]!=frozen["plan_sha256"]
                    or manifest.get("sampling_seed",2025)!=seed
                    or not out["readout_state_step_RNG_immutable"]):
                    raise ValueError("Frozen learning/information permissions changed")
                checked(folder/"train/endpoint2000.pth",train["checkpoint_sha256"])
                trace=read_json(folder/"train/TRACE.json")["steps"]
                if sum(x["replaced_T_slots"]for x in trace)!=25600:raise ValueError("Actual replacement exposure mismatch")
                scores={}
                for role,axis in task["score_axes"].items():
                    relative=str((folder/(role+".npz")).relative_to(root)).replace("\\","/")
                    scores[role]=score_bank(checked(root/relative,ids[relative]),axis,
                        task["data"]["target_test"]["shape"][1],task["scoring"]["anchor"])
                pending.append((domain,seed,arm,condition,task,scores)if repeat_seeds else(domain,arm,condition,task,scores))
                identity_key=f"{domain}/{seed}/{name}"if repeat_seeds else f"{domain}/{name}"
                identities[identity_key]=dict(checkpoint_sha256=train["checkpoint_sha256"],
                    plan_sha256=train["plan_sha256"],bundle_manifest_sha256=sha(manifest_path),
                    slot_map_sha256=sha(case/"BUNDLE/learner/SLOT_MAP.npz"),
                    normalized_bank_sha256=bind["window_bank_sha256"],task_sha256=sha(case/"TASK.json"))
    # Labels only enter this evaluation after every new bank, identity and permission check.
    labels={}
    for domain in domains:
        task=tasks_by_domain[domain];d=task["labels"];path=checked(workspace/d["relative"],d["sha256"])
        if path.suffix==".npz":
            with np.load(path)as z:labels[domain]=z["label"].copy()
        else:labels[domain]=np.load(path).reshape(-1)
        if not np.isin(labels[domain],[0,1]).all():raise ValueError("Labels must be binary event-time annotations")
    return pending,labels,identities


def new_metrics(item):
    domain,arm,condition,task,scores,labels,metric_path=item
    metric=load_author_metric(metric_path);axis=task["score_axes"]["QUERY"];index=np.arange(*axis)
    anchor=task["scoring"]["anchor"];available=anchor+(index-anchor)//192*192+191
    calaxis=task["score_axes"]["CAL"];selected=np.linspace(calaxis[0],calaxis[1]-1,512,dtype=np.int64)
    tau=float(np.quantile(scores["CAL"][selected-calaxis[0]],.99,method="higher"))
    rank,surface=continuous_metrics(labels[index],scores["QUERY"],metric)
    rows,events,_,_=evaluate_score(index,scores["QUERY"],labels,tau,available,
        query_start=axis[0],query_stop=axis[1],original_length=len(labels),expected_events=None,ranking=rank)
    for row in rows:row.update(rank,domain=domain,method=arm,condition=condition,seed=2025,
        CAL_exceedances=int(np.sum(scores["CAL"][selected-calaxis[0]]>tau)))
    return rows,[dict(domain=domain,method=arm,condition=condition,**e)for e in events],surface


def numerical_keys(rows):
    shared=set.intersection(*(set(r)for r in rows))
    ignore={"seed","threshold","scored_points","Ntrue","long_event_count","long_steady_total_points",
        "normal_exposure_points","normal_contiguous_blocks","point_exposure_in_event_valid_blocks","delay_eligible_events"}
    real=lambda value:isinstance(value,numbers.Real)and not isinstance(value,bool)
    return sorted(k for k in shared-ignore if any(real(r[k])for r in rows)
                  and all(r[k]is None or real(r[k])for r in rows))


def evaluate(workspace,root,output,workers=4):
    started=time.monotonic()
    if output.exists():raise FileExistsError(output)
    pending,labels,identities=inputs(workspace,root);output.mkdir(parents=True,exist_ok=False)
    rows=[];events=[];contrasts=[];interaction=[];surfaces={};audits={}
    for domain in labels:
        old,_=baseline(workspace,domain)
        rows.extend(dict(r,domain=domain,condition="A"if r["method"]in ["TT","R"]else"reference",seed=2025)for r in old)
        audit_path=root/domain/"BUNDLE/audit_only/EXPOSURE_AUDIT.json"
        audits[domain]=dict(sha256=sha(audit_path),audit=read_json(audit_path))
    items=[(*item,labels[item[0]],workspace/METRIC)for item in pending]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers)as pool:
        for item,result in zip(items,pool.map(new_metrics,items)):
            rr,ee,surface=result;rows.extend(rr);events.extend(ee)
            domain,arm,condition= item[:3]
            for k,v in zip(["TPR","FPR","precision","windows"],surface):surfaces[f"{domain}_{arm}_{condition}_{k}"]=v
    for domain in labels:
        lookup={(r["method"],r["condition"],r["stream"]):r for r in rows if r["domain"]==domain}
        for stream in ["strict","confirmed3"]:
            for arm in ["TT","R"]:
                for left,right in [("B","C"),("B","A"),("C","A")]:
                    a,b=lookup[(arm,left,stream)],lookup[(arm,right,stream)]
                    contrasts.append(dict(domain=domain,method=arm,contrast=f"{left}−{right}",stream=stream,
                        **{k:None if a[k]is None or b[k]is None else a[k]-b[k]for k in numerical_keys([a,b])}))
            rb,rc,tb,tc=[lookup[(a,c,stream)]for a,c in [("R","B"),("R","C"),("TT","B"),("TT","C")]]
            interaction.append(dict(domain=domain,stream=stream,contrast="(RB−RC)−(TTB−TTC)",
                **{k:None if any(r[k]is None for r in [rb,rc,tb,tc])else (rb[k]-rc[k])-(tb[k]-tc[k])
                   for k in numerical_keys([rb,rc,tb,tc])}))
    write_csv(output/"comparison.csv",rows);write_csv(output/"contrasts.csv",contrasts)
    write_csv(output/"interaction.csv",interaction);write_json(output/"EVENTS.json",events)
    np.savez_compressed(output/"VUS_SURFACES.npz",**surfaces)
    result=dict(rows=rows,contrasts=contrasts,interaction=interaction,identities=identities,
        exposure_audits=audits,cpu_wall_seconds=time.monotonic()-started,
        limits=["single-seed controlled development; current-U donors repeat few event runs",
                "event-time labels do not identify physically affected channels",
                "time-label exposure is not an exact anomalous gradient contribution",
                "TT second half remains mixed target; R second half remains author normal history; interaction alone is not robustness proof"])
    write_json(output/"METRICS.json",result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    result=evaluate(a.workspace,a.root,a.out,a.workers);print(f"Completed {len(result['rows'])} rows")
