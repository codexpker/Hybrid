"""Fixed CANDI endpoint-protocol comparison; existing LEFT rows are reused."""
import argparse
import concurrent.futures
from pathlib import Path
import time

import numpy as np

from .candi_native_a import PIN,score_starts
from .left_native_swat_metrics import continuous_metrics,load_author_metric,sha
from .left_response_preservation_evaluation import baseline,BASELINES,METRIC
from .left_swat_natural_u_evaluation import checked,read_json,write_json
from .left_swat_replay_repeats import write_csv
from .left_exposure_evaluation import numerical_keys
from .timercd_swat_evaluation import evaluate_score


def validated_inputs(workspace,root):
    seal=read_json(root/"GLOBAL_RAW_BANK_SEAL.json")
    if (seal["status"]!="COMPLETED"or seal["labels_read"]or len(seal["jobs"])!=2
        or any(j["OS_exit_code"]!=0 for j in seal["jobs"])):raise ValueError("Require both fixed source/adaptation/readout jobs")
    expected={f"{d}/{m}/{r}.npz"for d in ["SWAT","PSM"]for m in ["noU","CANDI"]for r in ["CAL","QUERY"]}
    ids={b["relative"].replace("\\","/"):b["sha256"]for b in seal["banks"]}
    if set(ids)!=expected:raise ValueError("Fixed eight banks required")
    pending=[];tasks={};identities=dict(global_seal_sha256=sha(root/"GLOBAL_RAW_BANK_SEAL.json"),GPU_hours=seal["actual_GPU_hours"])
    official=read_json(root/"OFFICIAL_SOURCE.json")
    if official["commit"]!=PIN:raise ValueError("Official source identity changed")
    for domain in ["SWAT","PSM"]:
        folder=root/domain;frozen=read_json(folder/"FROZEN.json")
        task=read_json(checked(folder/"TASK.json",frozen["task_sha256"]));tasks[domain]=task
        checked(root/"OFFICIAL_SOURCE.json",frozen["source_manifest_sha256"])
        source=read_json(folder/"SOURCE_TRAIN_RECEIPT.json");adapt=read_json(folder/"ADAPT_RECEIPT.json");checks=read_json(folder/"INCREMENT_CHECK.json")
        raw=read_json(folder/"RAW_BANK_SEAL.json");smoke=read_json(folder/"SOURCE_ROLE_SCORE_SMOKE.json")
        trace=read_json(folder/"SOURCE_TRAIN_TRACE.json");valid=[r for r in trace if"SELECT_MSE"in r]
        selected=min(valid,key=lambda r:r["SELECT_MSE"])
        if (source["source_FIT"]!=task["source"]["normal_FIT"]or source["source_SELECT"]!=task["source"]["normal_SELECT"]
            or source["labels_read"]or not source["source_only"]or source["epochs"]!=30
            or source["selected_epoch_zero_based"]!=selected["epoch"]or source["selected_update"]!=selected["updates"]
            or not smoke["native_whole_window_MSE_bitexact"]or smoke["labels_read"]or smoke["training_target_U_CAL_QUERY_values_read"]
            or adapt["U_FIT"]!=task["roles"]["U_FIT"]or not adapt["chronological_single_pass"]
            or adapt["windows_seen"]!=adapt["expected_windows"]or adapt["labels_read"]or adapt["target_CAL_QUERY_used_for_adaptation"]
            or not adapt["source_backbone_unchanged"]or adapt["FPM_reference"]!="complete normal SOURCE_SELECT"
            or adapt["optimizer_steps"]!=adapt["adaptation_groups"]*task["CANDI"]["steps"]
            or adapt["rejected_windows"]+adapt["hard_selected"]+adapt["moderate_selected"]!=adapt["windows_seen"]
            or not checks["full_reference_retained"]or not checks["formal_model_and_RNG_unchanged"]
            or raw["labels_read"]or raw["source_selected_sha256"]!=source["source_selected_sha256"]
            or raw["adapted_endpoint_sha256"]!=adapt["endpoint_sha256"]or raw["scaler_sha256"]!=source["scaler_sha256"]):
            raise ValueError("CANDI frozen role/selection/native-learning definition changed")
        checked(folder/"SOURCE_SELECTED.pth",source["source_selected_sha256"])
        checked(folder/"ADAPTED_ENDPOINT.pth",adapt["endpoint_sha256"]);checked(folder/"SCALER.npz",source["scaler_sha256"])
        for model in ["noU","CANDI"]:
            receipt=read_json(folder/model/"READOUT_RECEIPT.json")
            recovered=(model=="CANDI"and receipt.get("accepted_complete_endpoint_cache",False)
                and receipt.get("small_replay_state_RNG_immutable",False)
                and receipt.get("model_state_and_optimizer_immutable",False)
                and receipt.get("original_CPU_RNG_guard_failure_retained",False))
            if (receipt["labels_read"]or receipt["adapter_called"]or not(receipt["state_RNG_immutable"]or recovered)
                or receipt["optimizer_update_calls"]!=0 or receipt["source_selected_sha256"]!=source["source_selected_sha256"]
                or receipt["scaler_sha256"]!=source["scaler_sha256"]):raise ValueError("Readout updated or used another source model")
            scores={}
            for role,axis in task["score_axes"].items():
                relative=f"{domain}/{model}/{role}.npz";path=checked(root/relative,ids[relative])
                with np.load(path,allow_pickle=False)as z:
                    index=np.arange(*axis);starts=score_starts(axis)
                    if (not np.array_equal(z["absolute_time_index"],index)or not np.array_equal(z["available_index"],index)
                        or not np.array_equal(z["window_start"],starts)or not np.array_equal(z["window_end_exclusive"],starts+10)
                        or z["canonical_score"].shape!=index.shape or z["canonical_score"].dtype!=np.float32
                        or not np.isfinite(z["canonical_score"]).all()):raise ValueError("Each point must have its actual full native input window")
                    lo,hi=task["roles"][role]
                    if starts[0]<lo or starts[-1]+10>hi:raise ValueError("Window crosses role boundary")
                    scores[role]=z["canonical_score"].copy()
            pending.append((domain,model,task,scores))
        identities[domain]=dict(source_receipt=source,adaptation_receipt=adapt,increment_check=checks,
            source_checkpoint_sha256=source["source_selected_sha256"],endpoint_sha256=adapt["endpoint_sha256"],
            source_scaler_sha256=source["scaler_sha256"],task_sha256=sha(folder/"TASK.json"),worker_sha256=frozen["worker_sha256"])
    # No event labels are opened before all tasks and all eight score banks pass.
    labels={}
    for domain,task in tasks.items():
        d=task["labels"];path=checked(workspace/d["relative"],d["sha256"])
        if path.suffix==".npz":
            with np.load(path)as z:labels[domain]=z["label"].copy()
        else:labels[domain]=np.load(path).reshape(-1)
        if not np.isin(labels[domain],[0,1]).all():raise ValueError("Keep binary original point labels")
    return pending,labels,identities


def new_metrics(item):
    domain,model,task,scores,labels,metric_path=item
    axis=task["score_axes"]["QUERY"];index=np.arange(*axis);cal=task["score_axes"]["CAL"]
    selected=np.linspace(cal[0],cal[1]-1,512,dtype=np.int64)
    tau=float(np.quantile(scores["CAL"][selected-cal[0]],.99,method="higher"))
    metric=load_author_metric(metric_path);rank,surface=continuous_metrics(labels[index],scores["QUERY"],metric)
    rows,events,_,_=evaluate_score(index,scores["QUERY"],labels,tau,index,
        query_start=axis[0],query_stop=axis[1],original_length=len(labels),expected_events=None,ranking=rank)
    for row in rows:row.update(rank,domain=domain,method="CANDI_A"if model=="CANDI"else"CANDI_noU",seed=2025,
        native_window=10,native_stride=1,training="normal SOURCE_FIT MLP / normal SOURCE_SELECT checkpoint+FPM",
        CAL_exceedances=int(np.sum(scores["CAL"][selected-cal[0]]>tau)),CAL_selected_attack_points=int(labels[selected].sum()))
    return rows,[dict(domain=domain,method=model,**e)for e in events],surface


def evaluate(workspace,root,output,workers=4):
    started=time.monotonic()
    if output.exists():raise FileExistsError(output)
    pending,labels,identities=validated_inputs(workspace,root);output.mkdir(parents=True,exist_ok=False)
    rows=[];events=[];surfaces={};reused={}
    for domain in ["SWAT","PSM"]:
        prior,_=baseline(workspace,domain)
        for row in prior:
            if row["method"]in ["noU","TT","R"]:
                name="LEFT_noU"if row["method"]=="noU"else"LEFT_"+row["method"]+"_A"
                rows.append(dict(row,domain=domain,method=name,seed=2025,native_window=192,native_stride=192,
                    training="accepted LEFT author normal source, original full-prefix scaler includes SOURCE_SELECT"))
        relative,expected=BASELINES[domain];checked(workspace/relative,expected);reused[domain]=dict(relative=relative,sha256=expected)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers)as pool:
        items=[(*p,labels[p[0]],workspace/METRIC)for p in pending]
        for item,result in zip(items,pool.map(new_metrics,items)):
            rr,ee,surface=result;rows.extend(rr);events.extend(ee)
            domain,model=item[:2]
            for k,v in zip(["TPR","FPR","precision","windows"],surface):surfaces[f"{domain}_{model}_{k}"]=v
    assert len(rows)==20
    contrasts=[]
    for domain in ["SWAT","PSM"]:
        for stream in ["strict","confirmed3"]:
            group={r["method"]:r for r in rows if r["domain"]==domain and r["stream"]==stream}
            for reference in ["CANDI_noU","LEFT_noU","LEFT_TT_A","LEFT_R_A"]:
                a,b=group["CANDI_A"],group[reference]
                contrasts.append(dict(domain=domain,stream=stream,contrast="CANDI_A−"+reference,
                    **{k:None if a[k]is None or b[k]is None else a[k]-b[k]for k in numerical_keys([a,b])}))
    write_csv(output/"comparison.csv",rows);write_csv(output/"contrasts.csv",contrasts);write_json(output/"EVENTS.json",events)
    np.savez_compressed(output/"VUS_SURFACES.npz",**surfaces)
    result=dict(rows=rows,contrasts=contrasts,identities=identities,reused_LEFT_metrics=reused,
        old_LEFT_VUS_recomputed=False,cpu_wall_seconds=time.monotonic()-started,
        limits=["external CANDI; no change to or contribution credited to LEFT+R",
                "CANDI native whole-window MSE assigned to each actual window endpoint; no original window-any-positive labels",
                "normal SOURCE_FIT-only float32 StandardScaler; LEFT author scaler includes source validation",
                "SWAT author alpha5 script; PSM inherits main config defaults because no released PSM preset",
                "SOURCE_SELECT-only complete FPM references; target CAL/QUERY never adapted",
                "CANDI native ordered one-pass U differs from LEFT fixed sampled update budgets",
                "PSM CAL512 remains entirely attack-labeled; no normal false-positive certification",
                "SWAT initial gating.5 means adaptation-vs-plain-noU includes SANA injection and updates; not isolated FPM causality"])
    write_json(output/"METRICS.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--workers",type=int,default=4);a=p.parse_args()
    d=evaluate(a.workspace,a.root,a.out,a.workers);print(f"Completed {len(d['rows'])} rows")
