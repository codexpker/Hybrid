"""Evaluate only four new plain-MLP replay models; reuse accepted external rows."""
import argparse
import concurrent.futures
from pathlib import Path
import time

import numpy as np

from .mlp_replay import mapped_windows
from .candi_native_a import score_starts
from .left_native_swat_metrics import continuous_metrics,load_author_metric,sha
from .left_response_preservation_evaluation import METRIC
from .left_swat_natural_u_evaluation import checked,read_json,write_json
from .left_swat_replay_repeats import write_csv
from .left_exposure_evaluation import numerical_keys
from .timercd_swat_evaluation import evaluate_score

OLD_METRICS="results/candi_native_a_metrics_20261118_v1/METRICS.json"
OLD_SHA="57e3e8957d7ad53406107ce731d0b3ebc33b35c5e45d6911907ae74d9bdcda98"
LEFT_PARAMETERS={"SWAT":3742202,"PSM":4519985}


def validated_inputs(workspace,root):
    seal=read_json(root/"GLOBAL_RAW_BANK_SEAL.json")
    expected={f"{d}/{a}/{r}.npz"for d in ["SWAT","PSM"]for a in ["TT","R"]for r in ["CAL","QUERY"]}
    ids={b["relative"].replace("\\","/"):b["sha256"]for b in seal["banks"]}
    if (seal["status"]!="COMPLETED"or seal["labels_read"]or seal["source_training_repeated"]or set(ids)!=expected
        or len(seal["jobs"])!=4 or any(j["OS_exit_code"]!=0 for j in seal["jobs"])):raise ValueError("Need four complete fixed models and eight raw banks")
    pending=[];tasks={};identities=dict(seal_sha256=sha(root/"GLOBAL_RAW_BANK_SEAL.json"),GPU_hours=seal["actual_GPU_hours"])
    for domain in ["SWAT","PSM"]:
        case=root/domain;frozen=read_json(case/"FROZEN.json");task=read_json(checked(case/"TASK.json",frozen["task_sha256"]));tasks[domain]=task
        plan=checked(case/"SAMPLING_PLAN.npz",frozen["plan_sha256"])
        checked(workspace/task["replay_plan"]["original_relative"],frozen["plan_sha256"])
        checked(case/"MAPPED_ENDPOINT_PLAN.npz",frozen["mapped_plan_sha256"])
        with np.load(plan)as z,np.load(case/"MAPPED_ENDPOINT_PLAN.npz")as mapped:
            for key in ["T_starts","T_prime_starts","S_starts","S_prime_starts"]:
                start,end=mapped_windows(z[key],task["roles"]["U_FIT"if key.startswith("T")else"SOURCE_FIT"])
                if not(np.array_equal(start,mapped[key+"_relative10"])and np.array_equal(end,mapped[key+"_absolute_endpoint"])):
                    raise ValueError("Endpoint-preserving mapping changed")
        for key in ["checkpoint","scaler","config","source_manifest"]:checked(workspace/task["source"][key]["relative"],task["source"][key]["sha256"])
        initials=[]
        for arm in ["TT","R"]:
            folder=case/arm;initial=read_json(folder/"INITIAL_IDENTITY.json");train=read_json(folder/"TRAIN_RECEIPT.json");raw=read_json(folder/"RAW_BANK_SEAL.json")
            if(initial["source_checkpoint_sha256"]!=task["source"]["checkpoint"]["sha256"]or initial["scaler_sha256"]!=task["source"]["scaler"]["sha256"]
                or initial["source_config_sha256"]!=task["source"]["config"]["sha256"]or not initial["native_score_bitexact"]
                or initial["labels_read"]or initial["CAL_QUERY_read"]or initial["parameters"]!=initial["trainable_parameters"]
                or train["updates"]!=2000 or train["batch"]!=256 or train["native_forward_calls"]!=2000 or train["seed"]!=2025
                or train["source_training_repeated"]or train["labels_read"]or train["CAL_QUERY_read"]or train["endpoint_selection"]!="fixed2000"
                or not train["all_parameters_trainable"]or train["parameters"]!=initial["parameters"]
                or raw["labels_read"]or not raw["readout_state_RNG_immutable"]or not raw["readout_optimizer_state_immutable"]
                or raw["optimizer_update_calls_in_readout"]!=0 or not raw["source_assets_unchanged"]
                or raw["endpoint_sha256"]!=train["endpoint_sha256"]or raw["source_checkpoint_sha256"]!=initial["source_checkpoint_sha256"]
                or raw["scaler_sha256"]!=initial["scaler_sha256"]or raw["worker_sha256"]!=frozen["worker_sha256"]
                or raw["plan_sha256"]!=frozen["plan_sha256"]or train["mapped_plan_sha256"]!=frozen["mapped_plan_sha256"]):
                raise ValueError("Fixed plain MLP initialization/learning/score permissions changed")
            checked(folder/"endpoint2000.pth",train["endpoint_sha256"])
            trace=read_json(folder/"TRACE.json")["steps"]
            if [s["update"]for s in trace]!=list(range(1,2001))or not all(np.isfinite(s["loss"])and s["batch"]==256 for s in trace):
                raise ValueError("Exactly2000 valid full-batch updates required")
            expected_roles=["U_FIT"]if arm=="TT"else["U_FIT","SOURCE_FIT"]
            if train["training_roles"]!=expected_roles:raise ValueError("Unexpected training role materialization")
            scores={}
            for role,axis in task["score_axes"].items():
                relative=f"{domain}/{arm}/{role}.npz";path=checked(root/relative,ids[relative])
                with np.load(path,allow_pickle=False)as z:
                    index=np.arange(*axis);starts=score_starts(axis)
                    if(not np.array_equal(z["absolute_time_index"],index)or not np.array_equal(z["available_index"],index)
                        or not np.array_equal(z["window_start"],starts)or not np.array_equal(z["window_end_exclusive"],starts+10)
                        or z["canonical_score"].shape!=index.shape or z["canonical_score"].dtype!=np.float32 or not np.isfinite(z["canonical_score"]).all()):
                        raise ValueError("Keep actual native window endpoints and scores")
                    if starts[0]<task["roles"][role][0]or starts[-1]+10>task["roles"][role][1]:raise ValueError("Window crosses role")
                    scores[role]=z["canonical_score"].copy()
            initials.append(initial);pending.append((domain,arm,task,scores,initial["parameters"]))
            identities[f"{domain}/{arm}"]=dict(initial=initial,train=train,
                setup_peak=read_json(folder/"SETUP_PHASE_PEAK.json"),training_peak=read_json(folder/"TRAINING_PHASE_PEAK.json"),readout_peak=read_json(folder/"READOUT_PHASE_PEAK.json"))
        for key in ["state_sha256","rng","first_T_numeric_sha256","first_T_native_score_sha256","first_T_endpoints","parameters"]:
            if initials[0][key]!=initials[1][key]:raise ValueError("TT/R must share initialization/RNG and every first-batch target slot")
    labels={}
    for domain,task in tasks.items():
        spec=task["labels"];path=checked(workspace/spec["relative"],spec["sha256"])
        if path.suffix==".npz":
            with np.load(path)as z:labels[domain]=z["label"].copy()
        else:labels[domain]=np.load(path).reshape(-1)
        if not np.isin(labels[domain],[0,1]).all():raise ValueError("Original point labels required")
    return pending,labels,identities


def new_metrics(item):
    domain,arm,task,scores,parameters,labels,metric_path=item
    axis=task["score_axes"]["QUERY"];index=np.arange(*axis);cal=task["score_axes"]["CAL"]
    selected=np.linspace(cal[0],cal[1]-1,512,dtype=np.int64);tau=float(np.quantile(scores["CAL"][selected-cal[0]],.99,method="higher"))
    metric=load_author_metric(metric_path);rank,surface=continuous_metrics(labels[index],scores["QUERY"],metric)
    rows,events,_,_=evaluate_score(index,scores["QUERY"],labels,tau,index,query_start=axis[0],query_stop=axis[1],original_length=len(labels),expected_events=None,ranking=rank)
    for row in rows:row.update(rank,domain=domain,method="MLP_"+arm+"_A",seed=2025,native_window=10,native_stride=1,parameters=parameters,
        training="fixed native MLP source; TT/R original192 exposure endpoints, actual10 suffix; fresh Adam defaults2000x256",
        CAL_exceedances=int(np.sum(scores["CAL"][selected-cal[0]]>tau)),CAL_selected_attack_points=int(labels[selected].sum()))
    return rows,[dict(domain=domain,method="MLP_"+arm+"_A",**e)for e in events],surface


def evaluate(workspace,root,output,workers=4):
    started=time.monotonic()
    if output.exists():raise FileExistsError(output)
    pending,labels,identities=validated_inputs(workspace,root)
    old=read_json(checked(workspace/OLD_METRICS,OLD_SHA));rows=[]
    counts={d:p for d,a,t,s,p in pending}
    for r in old["rows"]:
        row=dict(r,reused_from_METRICS_sha256=OLD_SHA,reused_metric_method=r["method"])
        if row["method"]=="CANDI_noU":row.update(method="MLP_noU",parameters=counts[row["domain"]])
        elif row["method"].startswith("LEFT_"):row["parameters"]=LEFT_PARAMETERS[row["domain"]]
        rows.append(row)
    output.mkdir(parents=True,exist_ok=False);events=[];surfaces={}
    items=[(*p,labels[p[0]],workspace/METRIC)for p in pending]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers)as pool:
        for item,result in zip(items,pool.map(new_metrics,items)):
            rr,ee,surface=result;rows.extend(rr);events.extend(ee);domain,arm=item[:2]
            for key,value in zip(["TPR","FPR","precision","windows"],surface):surfaces[f"{domain}_{arm}_{key}"]=value
    assert len(rows)==28 and len({(r["domain"],r["method"],r["stream"])for r in rows})==28
    comparisons=[("MLP_R_A","MLP_TT_A"),("MLP_R_A","MLP_noU"),("MLP_TT_A","MLP_noU"),
        ("LEFT_R_A","LEFT_TT_A"),("LEFT_R_A","LEFT_noU"),("LEFT_TT_A","LEFT_noU"),
        ("CANDI_A","MLP_noU"),("LEFT_R_A","MLP_R_A"),("LEFT_TT_A","MLP_TT_A"),("LEFT_noU","MLP_noU")]
    paired=[]
    for domain in ["SWAT","PSM"]:
        for stream in ["strict","confirmed3"]:
            group={r["method"]:r for r in rows if r["domain"]==domain and r["stream"]==stream}
            for left,right in comparisons:
                a,b=group[left],group[right]
                paired.append(dict(domain=domain,stream=stream,contrast=left+"−"+right,
                    **{k:None if a[k]is None or b[k]is None else a[k]-b[k]for k in numerical_keys([a,b])if k not in ["native_window","native_stride","parameters"]}))
    write_csv(output/"comparison.csv",rows);write_csv(output/"contrasts.csv",paired);write_json(output/"NEW_EVENTS.json",events)
    np.savez_compressed(output/"NEW_VUS_SURFACES.npz",**surfaces)
    result=dict(rows=rows,contrasts=paired,identities=identities,old_METRICS=dict(relative=OLD_METRICS,sha256=OLD_SHA,rows_reused=20),
        old_VUS_recomputed=False,cpu_wall_seconds=time.monotonic()-started,
        limits=["single adaptation seed2025, no FPM/SANA, no new source training",
                "native10-point MLP window and192-point LEFT windows preserve original paired exposure endpoints but not context length",
                "CANDI/MLP SOURCE_FIT-only scaler versus LEFT original full normal-prefix scaler including source SELECT",
                "source learning objectives, source epochs and architecture differ; cross-backbone difference is not isolated time-frequency causality",
                "mixed CAL unchanged, including512/512 attack PSM points; no normal false-positive certification",
                "LEFT and full CANDI historical rows and VUS reused, not rerun"])
    write_json(output/"METRICS.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    result=evaluate(a.workspace,a.root,a.out,a.workers);print(f"Completed {len(result['rows'])} rows")
