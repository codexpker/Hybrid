"""Explicit N-NEG normal-only control and same-pool frozen-Z target CAL.

No query, label loader, model architecture or original F/kernel modification.
Numerical execution requires a new exact root release; GPU also needs a future
owned resource child and its pre-model cap receipt. No launcher is supplied.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
from . import event_adaptation as event
from .data import audit_events, load_manifest, resolve_plan, window

OPERATION = "HYBRID_V1_SMD_MATCHED_NORMAL_CONTROL"
PHASES = {"target_normal_train": {"SUPPORT", "NORMAL_SELECT"},
          "target_normal_calibrate": {"NORMAL_CAL"},
          "source_target_calibrate": {"NORMAL_CAL"}}
SOURCE = "c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812"
RUNTIME_FILES = event.RUNTIME_FILES | {"hybrid_tsad/hybrid_v1/"+n+".py" for n in
                                     ("normal_control", "readout", "chunked_bank")}


def validate_normal_plan(plan, support, *, history, horizon, seed, steps, choice):
    event.no_labels(plan)
    if any(plan.get(k) != v for k,v in dict(history=history,horizon=horizon,seed=seed,choice=choice).items()):
        raise ValueError("Normal plan clock/choice differs")
    K=support["K"];audit_events(support["events"],support["normal"],support["forbidden"],K,history,horizon)
    if steps < 2*K or steps % 2 or len(plan.get("rows",[])) != steps:
        raise ValueError("Even complete attempts and every normal group required")
    for a,row in enumerate(plan["rows"]):
        normal=support["normal"][(a//2)%K]
        if row.get("attempt")!=a or row.get("group")!=normal["matched_event_id"] or (row.get("A") is None)!=(a%2==0):
            raise ValueError("Fixed normal round-robin/odd A changed")
        for name,offset in (("A",0),("B",10000)):
            entry=row.get(name)
            if entry is None:
                if name=="B":raise ValueError("Every attempt needs normal SSL")
                continue
            t=entry.get("index")
            if (type(t) is not int or entry.get("role")!="SUPPORT" or
                any(entry.get(k)!=normal[k] for k in ("domain","entity","run")) or
                not normal["start"]<=t<normal["end"] or entry.get("footprint")!=[t-history,t+horizon] or
                entry.get("mask_seed")!=seed+a+offset):raise ValueError("Only exact matched-normal full footprints/masks")
    return dict(attempts=steps,normal_SSL=steps,normal_negative_BCE=steps//2,
                normal_forward_exposures=steps+steps//2,positive_forward_exposures=0)


def normal_plan_from_f_trace(trace, support, *, history, horizon, seed, choice, trace_sha256):
    """Metadata-only derivation; never read an array or feed positive windows."""
    if trace.get("route")!="F":raise ValueError("Reference must be the F route")
    rows=[]
    for a,old in enumerate(trace["steps"]):
        if old.get("attempt")!=a or old.get("updated") is not True or old.get("label")!=float(a%2==0):
            raise ValueError("Reference F must have complete accepted alternating updates")
        normal=support["normal"][(a//2)%support["K"]]
        if old.get("event_id")!=normal["matched_event_id"]:raise ValueError("F normal group changed")
        def entry(fp,offset):
            t=fp[0]+history
            if fp!=[t-history,t+horizon]:raise ValueError("F footprint is not H/P")
            return dict(**{k:normal[k] for k in ("domain","entity","run")},role="SUPPORT",
                        index=t,footprint=list(fp),mask_seed=seed+a+offset)
        if old.get("actual_mask_seed",seed+a)!=seed+a or old.get("normal_ssl_mask_seed",seed+a+10000)!=seed+a+10000:
            raise ValueError("Reference F masks changed")
        rows.append(dict(attempt=a,group=normal["matched_event_id"],A=entry(old["footprint"],0) if a%2 else None,
                         B=entry(old["normal_ssl_footprint"],10000)))
    plan=dict(history=history,horizon=horizon,seed=seed,choice=choice,source_F_trace_sha256=trace_sha256,rows=rows)
    validate_normal_plan(plan,support,history=history,horizon=horizon,seed=seed,steps=len(rows),choice=choice)
    return plan


def preflight(c, phase):
    """Only metadata/byte identities; no array/Torch construction."""
    if (phase not in PHASES or c.get("operation")!=OPERATION or c.get("scope") not in {"synthetic","public_target"}
        or c.get("permissions")!={"phase":phase,"query":False,"labels":False}):raise PermissionError("Explicit normal-control phase only")
    mode="Z_TCAL_R3" if phase=="source_target_calibrate" else "N_NEG300"
    if c.get("mode")!=mode:raise PermissionError("Normal negative control is not U or F-K0")
    if Path(c["output"]).exists():raise FileExistsError("Fresh phase output required")
    if c["scope"]=="synthetic" and c.get("device")!="cpu":raise PermissionError("Synthetic CPU only")
    if c["scope"]=="public_target":
        fixed=dict(history=512,horizon=32,seed=2025,steps=300,validate_every=50,lr=1e-4,weight_decay=.01,
                   kind="huber",select_count=32,cal_count=512,device="cuda:0",source_checkpoint_sha256=SOURCE)
        if any(c.get(k)!=v for k,v in fixed.items()) or c.get("active_components")!=["r_tf","r_ms","r_pred"]:
            raise PermissionError("Fixed source/N300 recipe changed")
        if any(c.get(k+"_sha256")!=v for k,v in event.ACCEPTED.items()):raise PermissionError("Same accepted SMD normal pool required")
    for k in ("manifest","plans","source_config","source_receipt","source_checkpoint"):event.bound(c,k)
    m,p,sc,sr=[event.read(c[k]) for k in ("manifest","plans","source_config","source_receipt")]
    event.no_labels(m);event.no_labels(p)
    if (sr.get("phase")!="source_train" or sr.get("files",{}).get("source.pt")!=c["source_checkpoint_sha256"]
        or sr["files"].get("config.json")!=c["source_config_sha256"]):raise ValueError("Actual source lineage required")
    for k in ("model","sources","history","horizon","seed","active_components"):
        if sc.get(k)!=c.get(k):raise ValueError("Saved source architecture/recipe differs")
    H,P=c["history"],c["horizon"]
    if set(p)!={"history","horizon","seed","NORMAL_SELECT","NORMAL_CAL"} or any(p[k]!=c[k] for k in ("history","horizon","seed")):
        raise ValueError("Both common SELECT/CAL plans required")
    if len(m.get("runs",[]))!=3 or {r["role"] for r in m["runs"]}!={"SUPPORT","NORMAL_SELECT","NORMAL_CAL"}:
        raise ValueError("Exact independent adaptation roles required")
    ident=tuple(m["runs"][0][k] for k in ("domain","entity","run"))
    for r in m["runs"]:
        if (tuple(r[k] for k in ("domain","entity","run"))!=ident or r.get("original_start")!=0
            or r.get("format")!="npz" or r.get("observed_key")!="observed"):raise ValueError("Native original masked record required")
        if c["scope"]=="public_target" and (r["bounds"]!=event.ROLES[r["role"]] or r["expected_shape"]!=[16585,38]):
            raise ValueError("Fixed prefix role axes differ")
    for role,count in (("NORMAL_SELECT",c["select_count"]),("NORMAL_CAL",c["cal_count"])):
        r=next(r for r in m["runs"] if r["role"]==role);lo,hi=r["bounds"];points=p[role]
        if len(points)!=count or len({x["index"] for x in points})!=count:raise ValueError("Unique fixed normal count required")
        for x in points:
            if (tuple(x[k] for k in ("domain","entity","run"))!=ident or x["role"]!=role
                or x.get("footprint")!=[x["index"]-H,x["index"]+P] or not lo+H<=x["index"]<=hi-P):raise ValueError("Normal full H/P plan differs")
    normal=None;support=None
    if mode=="Z_TCAL_R3" and any(c.get(k) for k in ("adapted_checkpoint","adapted_receipt","normal_plan")):
        raise PermissionError("Frozen Z CAL cannot receive N/F/U adapted assets")
    if mode=="N_NEG300":
        if c.get("choice") not in event.SUPPORT:raise ValueError("Only four fixed normal matched choices")
        if c["scope"]=="public_target" and c.get("support_sha256")!=event.SUPPORT[c["choice"]]:raise PermissionError("Original normal support changed")
        for k in ("support","normal_plan","reference_trace","reference_receipt","reference_acceptance"):event.bound(c,k)
        support,normal,tr,ref=[event.read(c[k]) for k in ("support","normal_plan","reference_trace","reference_receipt")]
        if ref.get("phase")!="target_event_train" or ref.get("status")!="COMPLETED":raise ValueError("Reference is accepted F trace, not N initialization")
        for k in ("source_checkpoint_sha256","manifest_sha256","plans_sha256","support_sha256","choice"):
            if ref.get(k)!=c.get(k):raise ValueError("F normal exposure provenance differs")
        accepted=event.read(c["reference_acceptance"])
        case=next((x for x in accepted.get("cases",[]) if x.get("choice")==c["choice"]),{}) if "cases" in accepted else accepted
        if (accepted.get("approved_by")!="root" or not str(accepted.get("status","")).startswith("ACCEPTED")
            or case.get("adapted_checkpoint_sha256",case.get("checkpoint_sha256"))!=ref.get("adapted_checkpoint_sha256")):
            raise PermissionError("Accepted reference F checkpoint/trace provenance required")
        expected=normal_plan_from_f_trace(tr,support,history=H,horizon=P,seed=c["seed"],choice=c["choice"],trace_sha256=c["reference_trace_sha256"])
        if normal!=expected or len(normal["rows"])!=c["steps"]:raise ValueError("No resampling or changed F normal exposures")
    if phase=="target_normal_calibrate":
        event.bound(c,"adapted_checkpoint");trained=event.read(event.bound(c,"adapted_receipt"))
        if trained.get("operation")!=OPERATION or trained.get("phase")!="target_normal_train" or trained.get("mode")!="N_NEG300" or trained.get("status")!="COMPLETED":
            raise ValueError("Only explicit N training receipt, never F/U")
        for k in ("choice","source_checkpoint_sha256","manifest_sha256","plans_sha256","support_sha256","normal_plan_sha256","reference_trace_sha256","adapted_checkpoint_sha256"):
            if trained.get(k)!=c.get(k):raise ValueError("N CAL lineage differs")
    return m,p,support,normal


def fit_normal_negative(source, fit_runs, select_runs, plan, support, *, history, horizon, seed,
                        lr, weight_decay, kind, validate_every, validation_plan, active_components, artifact_dir):
    """Original normal losses only; no positive forward or original kernel edits."""
    import torch
    from . import training as k
    from .objectives import ssl_objective
    from .readout import _rng, _rng_json
    model=copy.deepcopy(source);out=Path(artifact_dir)
    names=k.update_policy(model,"F");allowed=set(names);initial=copy.deepcopy(model.state_dict())
    before=event.state(model);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=weight_decay)
    rows=plan["rows"];A=resolve_plan(fit_runs,[r["A"] for r in rows if r["A"] is not None],history,horizon)
    validate_normal_plan(plan,support,history=history,horizon=horizon,seed=seed,steps=len(rows),choice=plan["choice"])
    B=resolve_plan(fit_runs,[r["B"] for r in rows],history,horizon);resolve_plan(select_runs,validation_plan,history,horizon)
    for run,t in A+B:
        if not run.observed[t].all() or not (run.observed[t-history+1:t].sum(0)>=2).all():raise ValueError("All-channel fixed normal eligibility required")
    torch.save(initial,out/"initial.pt");event.write(out/"RNG_before.json",_rng_json(_rng()))
    log=[];best=float("inf");chosen=None;selected=None;seen=set();negative=iter(A)
    head=[p for n,p in model.named_parameters() if n.split(".")[0]=="event_head"]
    if not head or set(n.split(".")[0] for n in names)!=event.UPDATE:raise ValueError("Exact five active blocks required")
    def bundle(a):return dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),optimizer_parameter_names=names,RNG=_rng(),final_attempt=a,
                             selected_state_dict=chosen,selected_attempt=selected,normal_plan=plan,resume_implemented=False)
    (out/"training_trace.jsonl").open("x",encoding="utf-8").close()
    for a,row in enumerate(rows):
        model.train();optimizer.zero_grad(set_to_none=True);bce=None;feature_ok=True
        if row["A"] is not None:
            run,t=next(negative);batch=k.tensor_batch(window(run,t,history,horizon),next(model.parameters()).device)
            f,r=k.passes(model,batch,row["A"]["mask_seed"]);obj=ssl_objective(f,r,batch,kind,active_components)
            parts=k.residual_components(f,r,batch,active_components);feature_ok=bool(obj["valid_windows"].all()) and bool(torch.isfinite(parts["event_features"]).all())
            z=model.event_head(parts["event_features"]).flatten();bce=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.zeros_like(z))
        run,t=B[a];batch=k.tensor_batch(window(run,t,history,horizon),next(model.parameters()).device)
        f,r=k.passes(model,batch,row["B"]["mask_seed"]);ssl=ssl_objective(f,r,batch,kind,active_components)
        loss=.1*ssl["total"] if bce is None else bce+.1*ssl["total"]
        if not feature_ok or not bool(ssl["valid_windows"].all()) or not bool(torch.isfinite(loss)):
            event.write(out/"INVALID_ATTEMPT.json",dict(attempt=a,reason="INVALID_FIXED_NORMAL_LOSS_NO_RESAMPLE",updates=len(log)))
            raise ValueError("Invalid fixed normal exposure; retain partial, never skip/retry")
        loss.backward();zero_head=bce is None
        if zero_head:
            if any(p.grad is not None for p in head):raise ValueError("Even-step head unexpectedly received SSL gradient")
            for p in head:p.grad=torch.zeros_like(p)
        head_norm=sum(float(p.grad.detach().abs().sum()) for p in head if p.grad is not None)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if next(model.parameters()).is_cuda:torch.cuda.synchronize()
        if bce is not None:seen.add(row["group"])
        check=(a+1)%validate_every==0 or a==len(rows)-1
        value=k.validation(model,select_runs,history,horizon,kind,seed,active_components=active_components,plan=validation_plan) if check else None
        eligible=seen=={n["matched_event_id"] for n in support["normal"]}
        if eligible and value is not None and value<best:
            best=value;chosen=copy.deepcopy(model.state_dict());selected=a;torch.save(chosen,out/"selected_partial.pt")
        item=dict(attempt=a,updated=True,A=row["A"],B=row["B"],group=row["group"],loss=float(loss.detach()),
                  BCE_negative=float(bce.detach()) if bce is not None else None,normal_SSL=float(ssl["total"].detach()),
                  head_zero_gradient_step=zero_head,head_gradient_L1=head_norm,gradient_norm=float(norm),
                  SELECT_ssl=value,selection_eligible=eligible,seen_normal_groups=sorted(seen))
        log.append(item)
        with (out/"training_trace.jsonl").open("a",encoding="utf-8") as file:file.write(json.dumps(item,allow_nan=False)+"\n")
        if check:torch.save(bundle(a),out/"recovery_latest.pt")
    if chosen is None:raise ValueError("No eligible normal-selected checkpoint")
    clock=[int(optimizer.state[p]["step"]) for p in head]
    if clock!=[len(rows)]*len(head):raise ValueError("Head optimizer clock is not all attempts")
    torch.save(bundle(len(rows)-1),out/"final_training_state.pt");model.load_state_dict(chosen)
    after=event.state(model)
    if any(before[n]!=after[n] for n in before if n not in allowed):raise ValueError("Frozen state/buffers changed")
    event.write(out/"RNG_after.json",_rng_json(_rng()))
    return model,dict(mode="N_NEG300",updates=len(rows),steps=log,updated_names=names,selected_attempt=selected,
                      selected_normal_SELECT=best,normal_SSL_exposures=len(rows),normal_negative_BCE_exposures=len(rows)//2,
                      positive_forward_exposures=0,head_optimizer_steps=clock,
                      changed_names=[n for n in before if before[n]!=after[n]])


def _action(c, phase, metadata):
    from . import cli, training as k
    m,plans,support,normal=metadata;out=Path(c["output"]);out.mkdir(parents=True,exist_ok=False);event.write(out/"config.json",c)
    runs=load_manifest(c["manifest"],roles=PHASES[phase]);role="NORMAL_SELECT" if phase=="target_normal_train" else "NORMAL_CAL"
    selected=[r for r in runs if r.role==role];points=plans[role]
    for run,t in resolve_plan(selected,points,c["history"],c["horizon"]):
        if not run.observed[t].all() or not (run.observed[t-c["history"]+1:t].sum(0)>=2).all():raise ValueError("Normal all-channel eligibility changed")
    path=c["adapted_checkpoint"] if phase=="target_normal_calibrate" else c["source_checkpoint"]
    model=cli.load_model(c,path);cli.save_runtime(out,model);before=event.state(model);flags={n:p.requires_grad for n,p in model.named_parameters()}
    event.write(out/"initial_state_identity.json",before);event.write(out/"initial_parameter_flags.json",flags)
    if phase=="target_normal_train":
        adapted,trace=fit_normal_negative(model,[r for r in runs if r.role=="SUPPORT"],selected,normal,support,
            **{key:c[key] for key in ("history","horizon","seed","lr","weight_decay","kind","validate_every","active_components")},validation_plan=points,artifact_dir=out)
        if event.state(model)!=before or flags!={n:p.requires_grad for n,p in model.named_parameters()}:raise ValueError("Source state/flags changed")
        cli.save_model(out/"adapted.pt",adapted);event.write(out/"training.json",trace);event.write(out/"adapted_state_identity.json",event.state(adapted))
        event.write(out/"source_post_state_identity.json",event.state(model));event.write(out/"state_difference.json",dict(source_unchanged=True,allowed=trace["updated_names"],changed=trace["changed_names"],buffers_frozen=True))
        event.write(out/"source_post_parameter_flags.json",{n:p.requires_grad for n,p in model.named_parameters()})
        event.write(out/"adapted_parameter_flags.json",{n:p.requires_grad for n,p in adapted.named_parameters()})
        extra=dict(updates=trace["updates"],selected_attempt=trace["selected_attempt"],adapted_checkpoint_sha256=event.sha(out/"adapted.pt"),positive_forward_exposures=0)
    else:
        arrived=[dict(x,footprint=[x["index"]-c["history"],x["index"]+1]) for x in points];event.write(out/"CAL_arrived_execution_plan.json",arrived)
        records=list(k.iter_infer(model,selected,history=c["history"],horizon=c["horizon"],seed=c["seed"],event_logits=phase=="target_normal_calibrate",active_components=c["active_components"],index_plan=arrived))
        from .scoring import valid_error
        clock={ (r.domain,r.entity,r.run):r.start for r in selected }
        if len(records)!=len(points) or any(r["index"]!=p["index"] or any(r.get(k)!=p[k] for k in ("domain","entity","run"))
            or r.get("replay",{}).get("input_footprint")!=p["footprint"]
            or r.get("replay",{}).get("actual_mask_seed")!=c["seed"]+r["index"]-clock[tuple(r[k] for k in ("domain","entity","run"))]
            or not all(valid_error(r,z) for z in c["active_components"]) for r,p in zip(records,arrived)):
            raise ValueError("All fixed CAL origins and complete R3 required")
        if phase=="target_normal_calibrate":
            if not all(np.isfinite(r.get("event_logit",np.nan)) for r in records):raise ValueError("Finite continuous N logits required")
            calibration=dict(score_kind="event_logit",q=.99,method="higher",strict_greater=True,threshold=k.logit_threshold(records));name="event_logit_calibration.json"
        else:
            calibration=k.calibrate_route(records,"U",active_components=c["active_components"]);name="target_r3_calibration.json"
            if calibration["source_domain_equal_weight"] is not False:raise ValueError("Z targetCAL must be source=False")
        cli.save_json(out/"normal_CAL_records.json",records);event.write(out/name,calibration)
        if event.state(model)!=before or flags!={n:p.requires_grad for n,p in model.named_parameters()}:raise ValueError("CAL changed state/flags")
        event.write(out/"calibration_parameter_flags.json",{n:p.requires_grad for n,p in model.named_parameters()})
        event.write(out/"calibration_post_state_identity.json",event.state(model));extra=dict(updates=0,calibrated_checkpoint_sha256=event.sha(path),target_calibration_contact=True,calibration_file=name,threshold=calibration["threshold"])
    if event.sha(c["source_checkpoint"])!=c["source_checkpoint_sha256"]:raise ValueError("Source file changed")
    if phase=="target_normal_calibrate" and event.sha(c["adapted_checkpoint"])!=c["adapted_checkpoint_sha256"]:raise ValueError("N checkpoint file changed")
    receipt=dict(status="COMPLETED",operation=OPERATION,phase=phase,mode=c["mode"],choice=c.get("choice"),loaded_roles=sorted(PHASES[phase]),query_or_labels=False,
                 **{key:c.get(key) for key in ("source_checkpoint_sha256","manifest_sha256","plans_sha256","support_sha256","normal_plan_sha256","reference_trace_sha256")},**extra)
    event.write(out/"receipt.json",receipt);return receipt


def run_phase(config, release, trusted, phase):
    if not release or not trusted or event.sha(release)!=trusted:raise PermissionError("New exact root normal phase release required")
    c,r=event.read(config),event.read(release)
    if (r.get("status")!="APPROVED" or r.get("approved_by")!="root" or r.get("operation")!=OPERATION
        or r.get("allowed_actions")!=[phase] or r.get("config_sha256")!=event.sha(config) or r.get("source_sha256")!=c.get("source_sha256")):
        raise PermissionError("Changed or pending normal phase release")
    base=Path(__file__).resolve().parents[2];closure=c["source_sha256"]
    if set(closure)!=RUNTIME_FILES:raise PermissionError("Exact normal-control cold closure required")
    for name,digest in closure.items():
        path=(base/name).resolve()
        if base not in path.parents or event.sha(path)!=digest:raise PermissionError("Changed normal phase source")
    metadata=preflight(c,phase)
    if c["device"]=="cpu":
        if c["scope"]!="synthetic" or os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise PermissionError("Root synthetic CPU-only environment required")
    else:
        cap=event.read(Path(r["lifecycle"])/"GPU_CAP_BEFORE_MODEL.json");limit=(20 if phase=="target_normal_train" else 30)*1024**3
        canonical=lambda v:str(v)[4:] if str(v).startswith("GPU-") else str(v)
        if (os.environ.get("HYBRID_NORMAL_CONTROL_CHILD")!=trusted or os.environ.get("CUDA_VISIBLE_DEVICES")!=str(c.get("physical_gpu")) or cap.get("release_sha256")!=trusted
            or cap.get("config_sha256")!=r["config_sha256"] or cap.get("limit_bytes")!=limit
            or cap.get("set_before_model_construction") is not True or cap.get("physical_gpu")!=c.get("physical_gpu")
            or canonical(cap.get("GPU_uuid"))!=canonical(c.get("expected_gpu_uuid"))):
            raise PermissionError("Future owned resource child/cap-before-model required; no launcher in this package")
    result=_action(c,phase,metadata);event.write(Path(c["output"])/"execution_identity.json",dict(config_sha256=event.sha(config),release_sha256=trusted,source_sha256=closure));return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True);p.add_argument("--phase",required=True,choices=sorted(PHASES));p.add_argument("--execute",action="store_true");p.add_argument("--release");p.add_argument("--trusted-release-sha256");a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,body_or_Torch_opened=False)));return
    print(json.dumps(run_phase(a.config,a.release,a.trusted_release_sha256,a.phase)))


if __name__=="__main__":main()
