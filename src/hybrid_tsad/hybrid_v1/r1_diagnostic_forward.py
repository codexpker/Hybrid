"""Frozen R1 transfer diagnostic; original model/loss remain unchanged.

New synthetic holdout only. No U training, real QUERY, calibration or new loss.
The standard teacher/student squared discrepancy is an observation, not a new
distilled model. Default CLI validates metadata without Torch or array loading.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

OPERATION = "HYBRID_V1_R1_FROZEN_TASK_CHANGE_DIAGNOSTIC"
MODEL_IDS = ("SOURCE_2025", "U_A_2025", "U_B_2025", "U_C_2025")


def jsonable(value):
    if isinstance(value, dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return jsonable(value.tolist())
    if isinstance(value,np.generic):return jsonable(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path, cfg):
    return Path(cfg.get("path_map", {}).get(path, path))


def bound(asset, cfg, *, as_json=True):
    path = resolve(asset["path"], cfg)
    if not asset.get("sha256") or sha(path) != asset["sha256"]:
        raise PermissionError("Changed exact diagnostic asset: "+str(path))
    return json.loads(path.read_text(encoding="utf-8")) if as_json else path


def validate_unit(unit):
    if (unit.get("history") != 512 or unit.get("horizon") != 32
            or unit.get("runtime_role") != "R1_DIAGNOSTIC_HOLDOUT"
            or type(unit.get("target")) is not int or unit["target"] not in (1024,1280,1536,2048,2528,2560)
            or unit.get("footprint") != [unit["target"]-512, unit["target"]+32]
            or type(unit.get("mask_seed")) is not int or not 0 <= unit["mask_seed"] < 2**32
            or unit.get("encoder_asset", {}).get("role") != "DIAGNOSTIC_HOLDOUT"
            or unit["encoder_asset"].get("expected_shape") not in ([3072,1],[3072,3])):
        raise PermissionError("Only the fixed new diagnostic full544 footprint is permitted")
    if any(key in unit for key in ("label", "labels", "phase", "diagnostic_label", "event_id", "weights")):
        raise PermissionError("Truth or sample weights cannot enter numeric forward units")


def metadata(cfg):
    if (cfg.get("operation") != OPERATION or cfg.get("status") not in {"ROOT_BOUND","AUTHORIZED_ROLLING_R1"}
            or cfg.get("model_ids") != list(MODEL_IDS) or cfg.get("mode") != "ALL_EVAL_SAME_MASK"
            or cfg.get("training") is not False or cfg.get("public_forward") is not False
            or cfg.get("kind") != "huber" or cfg.get("active_components") != ["r_tf","r_ms","r_pred"]
            or cfg.get("max_total_GPU_seconds") != 3600 or Path(cfg["output"]).exists()):
        raise PermissionError("Exact read-only R1 contract and fresh output required")
    if cfg["source_sha256"].get(str(Path(__file__).resolve())) != sha(__file__):
        raise PermissionError("Actual R1 caller must be pinned")
    for path, digest in cfg["source_sha256"].items():
        if sha(resolve(path, cfg)) != digest:
            raise PermissionError("Changed diagnostic/original math source")
    plan = bound(cfg["numeric_plan"], cfg)
    models = bound(cfg["model_bindings"], cfg)
    if ([m.get("model_id") for m in models["models"]] != list(MODEL_IDS)
            or plan.get("namespace") != "R1_DIAGNOSTIC_20261105"
            or plan.get("logical_windows") != 96 or plan.get("unique_window_units") != 86
            or len(plan.get("units", [])) != 86 or plan.get("optimization") is not False
            or plan.get("no_public_purge_forward") is not True):
        raise PermissionError("The accepted16view/96logical/86unique plan is required")
    keys = set(); aliases = []
    for unit in plan["units"]:
        validate_unit(unit); key = unit["forward_unit_id"]
        if key in keys:
            raise ValueError("Duplicate forward unit would repeat an already bound computation")
        keys.add(key); aliases.extend(unit["logical_alias_ids"])
        bound(unit["encoder_asset"], cfg, as_json=False)
    if len(aliases) != 96 or len(set(aliases)) != 96:
        raise PermissionError("All96 logical samples must be preserved once")
    for model in models["models"]:
        bound(model["checkpoint"], cfg, as_json=False)
    bound(models["common_model_config"], cfg)
    return plan, models


def equal_task_support(source, student):
    """No tolerance repair of masks, visible support or input-only statistics."""
    checks = [(source["forecast"]["copy_masks"], student["forecast"]["copy_masks"])]
    for key in ("hidden_mask", "visible_mask", "transform_hidden_mask", "observed_mask"):
        checks.append((source["reconstruction"][key], student["reconstruction"][key]))
    for mode in ("forecast", "reconstruction"):
        for key in ("mean", "scale", "channel_valid", "visible_count"):
            checks.append((source[mode]["stats"][key], student[mode]["stats"][key]))
    if not all(np.array_equal(a,b, equal_nan=True) for a,b in checks):
        raise ValueError("Teacher/student masks or input-only statistics differ")


def _masked_mse(error, valid):
    valid = np.broadcast_to(np.asarray(valid, bool), error.shape)
    values = np.asarray(error, np.float64)[valid]
    return float(np.mean(values**2)) if len(values) and np.isfinite(values).all() else None


def discrepancy(source, student):
    """Standard frozen prediction/feature MSE, NOT a difference of GT losses."""
    equal_task_support(source, student)
    sf, uf = source["forecast"], student["forecast"]
    support = sf["valid"] & uf["valid"] & source["batch"]["future_observed"]
    p = (np.asarray(uf["prediction"], np.float64)-sf["prediction"])/sf["stats"]["scale"]
    channel = support.any(1)
    return dict(D_pred_teacher_scale_MSE=_masked_mse(p, support),
        D_T_feature_MSE=_masked_mse(np.asarray(uf["t_features"],np.float64)-sf["t_features"],channel[...,None]),
        D_F_feature_MSE=_masked_mse(np.asarray(uf["f_features"],np.float64)-sf["f_features"],channel[...,None]),
        interpretation="Frozen teacher-student discrepancy only; no new distilled model was trained")


def loss_delta(source, student):
    parts = {}
    for key in ("tf","ms","pred","consistency"):
        a,b = source["loss"]["parts"][key], student["loss"]["parts"][key]
        valid = bool(np.asarray(source["loss"]["valid_counts"][key]).sum() > 0
                     and np.asarray(student["loss"]["valid_counts"][key]).sum() > 0)
        parts[key] = dict(source=a, student=b, student_minus_source=float(b-a) if valid else None,
                          valid=valid, NA_reason=None if valid else "NO_COMMON_VALID_TASK_GT")
    return dict(parts=parts, source_total=source["loss"]["total"],student_total=student["loss"]["total"],
                total_student_minus_source=student["loss"]["total"]-source["loss"]["total"],
                source_raw_residuals=source["raw_residuals"],student_raw_residuals=student["raw_residuals"],
                negative_delta_is_not_automatically_pollution_absorption=True)


def payload(model, batch, seed):
    from .training import passes
    from .objectives import ssl_objective
    from .scoring import residual_components
    from .controlled_mechanism import detached
    f,r = passes(model,batch,seed)
    loss = ssl_objective(f,r,batch,"huber",["r_tf","r_ms","r_pred"])
    raw = residual_components(f,r,batch,["r_tf","r_ms","r_pred"])
    fkeys = ("prediction","prediction_normalized","prediction_by_copy_normalized","valid","stats",
             "t_features","f_features","copy_masks","axes","tf_order")
    rkeys = ("tf_normalized","tf_valid","ms_normalized_by_scale","ms_normalized","ms_valid_by_scale",
             "ms_valid","hidden_mask","visible_mask","observed_mask","stats","transform_hidden_mask",
             "tf_reconstruction","ms_reconstruction_by_scale","ms_reconstruction","axes","order",
             "endpoint_t_features","endpoint_f_features")
    return detached(dict(forecast={k:f[k] for k in fkeys}, reconstruction={k:r[k] for k in rkeys},
        batch=batch,loss=loss,raw_residuals=raw["scalar"],per_channel_residuals=raw["per_channel"],
        one_forecast_and_one_reconstruct=True))


def run(cfg, plan, bindings):
    import torch
    from .cli import load_model,save_json,save_runtime
    from .data import window
    from .training import tensor_batch
    from .stage2_u_matched import evaluation_state
    from .stage2_adaptation import state_identity
    from .controlled_mechanism import save_bundle
    out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=False)
    data_cache, teacher, units = {}, {}, plan["units"]
    # No label/phase/qualification CSV is opened by this process.
    for unit in units:
        asset=unit["encoder_asset"];key=asset["sha256"]
        if key not in data_cache:
            with np.load(bound(asset,cfg,as_json=False),allow_pickle=False) as z:
                if set(z.files)!={"values","observed"}:raise PermissionError("Numeric inputs only")
                x,m=z["values"],z["observed"]
            if (list(x.shape)!=asset["expected_shape"] or x.dtype!=np.float32 or m.dtype!=np.bool_
                    or m.shape!=x.shape or not m.all() or not np.isfinite(x).all()):
                raise ValueError("Changed fixed synthetic numerical input")
            data_cache[key]=(x,m)
    model_config=bound(bindings["common_model_config"],cfg);model_config["device"]=cfg["device"]
    summary=[];started=time.monotonic()
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    for entry in bindings["models"]:
        mid=entry["model_id"];folder=out/mid;folder.mkdir();rawdir=folder/"raw";rawdir.mkdir()
        model=load_model(copy.deepcopy(model_config),str(bound(entry["checkpoint"],cfg,as_json=False)))
        model.eval();state=state_identity(model);save_json(folder/"STATE_BEFORE.json",state);save_runtime(folder,model)
        changes=[];completed=[]
        for unit in units:
            if time.monotonic()-started>=cfg["max_total_GPU_seconds"]:
                raise TimeoutError("R1 total forward cap exhausted; no automatic retry")
            x,m=data_cache[unit["encoder_asset"]["sha256"]]
            run_input=SimpleNamespace(values=x,observed=m,start=0,role="R1_DIAGNOSTIC_HOLDOUT",
                domain="R1_SYNTHETIC_TRANSFER",entity=unit["encoder_asset"]["entity"],run=unit["encoder_asset"]["run"])
            batch=tensor_batch(window(run_input,unit["target"],512,32),cfg["device"])
            with evaluation_state(model,unit["mask_seed"]):
                if any(v.training for v in model.modules()):raise ValueError("Every module must be eval")
                value=payload(model,batch,unit["mask_seed"])
            uid=unit["forward_unit_id"]
            save_bundle(rawdir,uid,value)
            if mid=="SOURCE_2025":
                teacher[uid]=value
                kd=discrepancy(value,value)
                if any(kd[k]!=0 for k in ("D_pred_teacher_scale_MSE","D_T_feature_MSE","D_F_feature_MSE")):
                    raise ValueError("Source-vs-itself discrepancy must be zero")
            else:
                kd=discrepancy(teacher[uid],value)
            changes.append(dict(forward_unit_id=uid,logical_alias_ids=unit["logical_alias_ids"],
                target=unit["target"],mask_seed=unit["mask_seed"],task_change=loss_delta(teacher[uid],value),
                source_discrepancy=kd,same_masks_stats=True))
            completed.append(uid)
            with (folder/"PROGRESS.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(dict(unit=uid,completed=len(completed),model=mid))+"\n");f.flush()
        after=state_identity(model)
        if after!=state:raise ValueError("Frozen model parameter/buffer mutated")
        save_json(folder/"STATE_AFTER.json",after);save_json(folder/"TASK_CHANGE_AND_DISCREPANCY.json",jsonable(changes))
        summary.append(dict(model=mid,unique_units=len(completed),logical_windows=96,state_unchanged=True))
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    save_json(out/"CONFIG.json",cfg)
    result=dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT",models=summary,teacher="SOURCE_2025",
        unique_window_units_per_model=86,logical_windows_per_model=96,total_unique_model_window_units=344,
        total_logical_model_window_units=384,whole_seconds=time.monotonic()-started,training=False,
        peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        peak_reserved_GPU_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
        labels_or_phase_read=False,calibration_or_threshold_fit=False,real_public_QUERY_forward=False,
        transfer_only_not_direct_real_contamination_absorption=True,loss_and_discrepancy_are_different_objects=True)
    save_json(out/"RECEIPT.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--release");p.add_argument("--trusted-release-sha256")
    p.add_argument("--execute",action="store_true");a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding="utf-8"));plan,models=metadata(cfg)
    if not a.execute:
        print(json.dumps(dict(metadata="PASS",Torch_loaded=False,arrays_opened=False)));return
    if cfg.get("status")=="AUTHORIZED_ROLLING_R1":
        if cfg.get("authorization")!={"approved_by":"root","pool":"USER_ROLLING_7D_40_GPUH",
                "stage":"R1","stage_GPU_seconds_cap":3600,"physical_gpu":3,
                "executor_is_sole_SSH_operator":True,"no_training":True}:
            raise PermissionError("Exact root-authorized R1 allocation required")
        print(json.dumps(dict(status=run(cfg,plan,models)["status"])));return
    if not a.release or not a.trusted_release_sha256:raise PermissionError("Exact internal root release required")
    signed=bound(dict(path=a.release,sha256=a.trusted_release_sha256),cfg)
    if (signed.get("approved_by")!="root" or signed.get("status")!="APPROVED"
            or signed.get("operation")!=OPERATION or signed.get("allowed_actions")!=["frozen_synthetic_forward"]
            or signed.get("config_sha256")!=sha(a.config) or signed.get("source_sha256")!=cfg["source_sha256"]
            or signed.get("output")!=cfg["output"] or signed.get("max_total_GPU_seconds")!=3600):
        raise PermissionError("Exact root one-hour synthetic-only release required")
    print(json.dumps(dict(status=run(cfg,plan,models)["status"])))


if __name__=="__main__":main()
