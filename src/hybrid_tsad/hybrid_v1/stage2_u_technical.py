"""Bounded technical L0 for matched-U APIs; never a scientific query alias."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

OPERATION = "HYBRID_V1_MATCHED_U_SYNTHETIC_GPU_L0"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def admit(config, release, trusted):
    c = json.loads(Path(config).read_text()); r = json.loads(Path(release).read_text())
    if (sha(release) != trusted or r.get("approved_by") != "root" or r.get("status") != "APPROVED"
            or r.get("operation") != OPERATION or r.get("allowed_actions") != ["l0"]
            or r.get("config_sha256") != sha(config) or r.get("source_sha256") != c["source_sha256"]):
        raise PermissionError("Exact technical L0 root release required")
    if (c.get("operation") != OPERATION or c.get("phase") != "l0" or c.get("seed") != 2025
            or c.get("history") != 512 or c.get("horizon") != 32 or c.get("channels") != 38
            or c.get("technical_attempts") != 2 or c.get("technical_points") != 65
            or c.get("synthetic_rng_seed") != 91731 or c.get("device") != "cuda:0"
            or c.get("resources") != {"cap_total_seconds":360,"TERM_seconds":355,"cleanup_seconds":5,"allocator_GiB":20}
            or c.get("permissions") != dict(synthetic_only=True,accepted_parent_PT=True,source_CAL_JSON=True,
                formal_role_arrays=False,formal_QUERY=False,labels=False,formal_training=False)
            or Path(c["output_dir"]).exists()):
        raise PermissionError("Fixed fresh synthetic technical scope required")
    for path, expected in c["source_sha256"].items():
        if sha(path) != expected:
            raise PermissionError("Changed explicit frozen runtime")
    if c["source_sha256"].get(str(Path(__file__).resolve())) != sha(__file__):
        raise PermissionError("Executing technical entry must be pinned")
    for path, expected in c["external_source_sha256"].items():
        if sha(path) != expected:
            raise PermissionError("Changed official source")
    if sha(c["installed_frft"]["path"]) != c["installed_frft"]["sha256"]:
        raise PermissionError("Changed installed FRFT")
    for asset in c["assets"].values():
        if sha(asset["path"]) != asset["sha256"]:
            raise PermissionError("Changed accepted parent/CAL/metadata asset")
    parent = c["assets"]["checkpoint"]["sha256"]
    if parent != "c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812":
        raise PermissionError("Only accepted source2025 parent is authorized")
    return c


def run(c, config_sha, release_sha):
    ledger = json.loads(Path(c["budget_ledger"]).read_text()); lease=ledger["leases"][c["lease_id"]]
    if (lease["state"] != "RUNNING" or lease["owner_pid"] != os.getppid()
            or lease["config_sha256"] != config_sha or lease["release_sha256"] != release_sha):
        raise PermissionError("No actual owned live L0 lease before Torch import")
    import torch
    from .cli import load_model, save_json, save_runtime, score_records
    from .stage2_u_matched import train_endpoint, validate, evaluation_state, calibrate, load_roles
    from .stage2_adaptation import state_identity
    from .readout import write_model_bank, _rng, _rng_json
    from .stage2_u_readout import rescore_view
    from .chunked_bank import read_records, read_entries, _pack, ChunkedBank
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one owned visible CUDA device required")
    props=torch.cuda.get_device_properties(0)
    norm=lambda value: str(value).removeprefix("GPU-")
    if norm(props.uuid)!=norm(c["expected_gpu_uuid"]) or "A40" not in props.name:
        raise PermissionError("Actual CUDA UUID/A40 differs from owned NVML identity; no fallback")
    limit=20*1024**3
    torch.cuda.set_per_process_memory_fraction(limit/props.total_memory,0)
    torch.cuda.reset_peak_memory_stats(0)
    out=Path(c["output_dir"]);out.mkdir(parents=True,exist_ok=False)
    save_json(out/"CAP_BEFORE_MODEL.json",dict(allocator_limit_bytes=limit,actual_UUID=str(props.uuid),CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES")))
    source_cfg=json.loads(Path(c["assets"]["source_config"]["path"]).read_text());source_cfg["device"]="cuda:0"
    source=load_model(source_cfg,c["assets"]["checkpoint"]["path"])
    save_runtime(out,source);before=state_identity(source);source_modes=[m.training for m in source.modules()]
    save_json(out/"SOURCE_STATE_BEFORE.json",before)
    rng=np.random.default_rng(91731)
    entries=[]
    def synthetic(name,n,start):
        x=rng.standard_normal((n,38)).astype(np.float32)
        path=out/(name+".SYNTHETIC_ONLY.npz");np.savez_compressed(path,values=x,observed=np.ones(x.shape,bool))
        entries.append(dict(domain="SYNTHETIC_U_L0",entity="C38",run=name,role=name,path=str(path),format="npz",
            key="values",observed_key="observed",sha256=sha(path),expected_shape=[n,38],bounds=[0,n],original_start=start))
    # Distinct run identities are genuinely independently generated technical arrays.
    synthetic("FIT",546,0);synthetic("SELECT",546,10000)
    synthetic("CAL",546,20000);synthetic("QUERY",577,30000)
    manifest=out/"SYNTHETIC_ROLE_MANIFEST.json";save_json(manifest,dict(runs=entries))
    fit,select=load_roles(manifest,{"FIT","SELECT"},512,32)
    def entry(run,t,arrived=False):
        return dict(domain=run.domain,entity=run.entity,run=run.run,role=run.role,index=run.start+t,
                    footprint=[run.start+t-512,run.start+t+(1 if arrived else 32)])
    plan=dict(history=512,horizon=32,seed=2025,FIT=[entry(fit,512),entry(fit,513)],SELECT=[entry(select,512),entry(select,513)])
    save_json(out/"TECHNICAL_PLAN.json",plan)
    adapted,report,opt=train_endpoint(source,[fit],[select],plan,history=512,horizon=32,seed=2025,
        attempts=2,validate_every=1,active_components=["r_tf","r_ms","r_pred"],scope="synthetic",artifact_dir=out)
    assert report["updates"]==2 and report["scientific_training_status"]=="VALID_ALL_ATTEMPTS_UPDATED"
    grad={}
    for prefix in ("fusion","tf_fusion_delta","ms_decoder","forecast_head"):
        values=[float(p.grad.detach().abs().max()) for n,p in adapted.named_parameters() if n.startswith(prefix+".") and p.grad is not None]
        grad[prefix]=max(values,default=0.);assert np.isfinite(grad[prefix]) and grad[prefix]>0
    assert report["state_difference"]["PASS"] and not report["state_difference"]["changed_buffers"]
    assert before==state_identity(source) and source_modes==[m.training for m in source.modules()]
    torch.save(dict(state_dict=adapted.state_dict(),optimizer=opt.state_dict(),discarded=True,
                    not_for_formal_initialization=True),out/"DISCARDED_TECHNICAL_WEIGHTS.pt")
    save_json(out/"training.json",report);save_json(out/"ADAPTED_STATE.json",state_identity(adapted))
    # Actual same-device CUDA RNG and heterogeneous module modes, not a stub.
    torch.cuda.synchronize();r0=_rng_json(_rng());modes=[m.training for m in adapted.modules()]
    validate(adapted,[select],plan["SELECT"],512,32,2025,["r_tf","r_ms","r_pred"])
    torch.cuda.synchronize();r1=_rng_json(_rng())
    assert r0==r1 and modes==[m.training for m in adapted.modules()]
    try:
        with evaluation_state(adapted,99):
            torch.rand(3,device="cuda:0");torch.rand(3);raise RuntimeError("EXPECTED_L0_FINALLY_FIXTURE")
    except RuntimeError as exc:
        assert str(exc)=="EXPECTED_L0_FINALLY_FIXTURE"
    torch.cuda.synchronize();r2=_rng_json(_rng());assert r0==r2 and modes==[m.training for m in adapted.modules()]
    save_json(out/"RNG_BEFORE.json",r0);save_json(out/"RNG_AFTER_SELECT.json",r1);save_json(out/"RNG_AFTER_EXCEPTION.json",r2)
    save_json(out/"MODE_FLAGS.json",dict(before=modes,after=[m.training for m in adapted.modules()],names=[n for n,_ in adapted.named_modules()]))
    calrun=load_roles(manifest,{"CAL"},512,32)[0]
    records,cal=calibrate(adapted,[calrun],[entry(calrun,512,True),entry(calrun,545,True)],history=512,horizon=32,seed=2025,active_components=["r_tf","r_ms","r_pred"])
    save_json(out/"technical_calibration_records.json",records);save_json(out/"technical_calibration.json",cal)
    base=state_identity(adapted)
    identity=dict(input_sha256=sha(out/"QUERY.SYNTHETIC_ONLY.npz"),config_sha256=config_sha,
        checkpoint_sha256=sha(out/"DISCARDED_TECHNICAL_WEIGHTS.pt"),calibration_sha256=sha(out/"technical_calibration.json"),
        source_sha256=hashlib.sha256(json.dumps(c["source_sha256"],sort_keys=True).encode()).hexdigest())
    query=load_roles(manifest,{"QUERY"},512,32)[0]
    result=write_model_bank(adapted,[query],cal,out/"technical_bank",identity=identity,history=512,horizon=32,
                            seed=2025,chunk_size=64,capture_prefix=65,resume=False,active_components=["r_tf","r_ms","r_pred"])
    saved=list(read_records(out/"technical_bank"));assert len(saved)==65 and [x["index"] for x in saved]==list(range(30512,30577))
    expected=score_records(result["prefix"],cal);packed=_pack(saved);p0=_pack(expected)
    assert packed.keys()==p0.keys() and all(packed[k].dtype==p0[k].dtype and packed[k].shape==p0[k].shape and packed[k].tobytes()==p0[k].tobytes() for k in packed)
    for raw in saved:
        assert np.asarray(raw["prediction"]).shape==(32,38) and raw["available_index"]==raw["index"]
        assert raw["replay"]["unarrived_future_observed"] is False
        assert raw["replay"]["actual_mask_seed"]==2025+raw["index"]-query.start
    manifest=json.loads((out/"technical_bank/manifest.json").read_text());assert manifest["complete"] and [x["records"] for x in manifest["chunks"]]==[64,1]
    verifier=object.__new__(ChunkedBank);verifier.contract=manifest["contract"];states={}
    for item in read_entries(out/"technical_bank"):assert verifier._derive(item["raw"],states)==item["derived"]
    sourcecal=json.loads(Path(c["assets"]["source_calibration"]["path"]).read_text())
    view=rescore_view(out/"technical_bank",out/"technical_SECOND_CAL_VIEW",sourcecal,
                      dict(identity,calibration_sha256=c["assets"]["source_calibration"]["sha256"]))
    assert not view["model_forward"] and base==state_identity(adapted) and before==state_identity(source)
    assert sha(c["assets"]["checkpoint"]["path"])==c["assets"]["checkpoint"]["sha256"]
    save_json(out/"SOURCE_STATE_AFTER.json",state_identity(source));save_json(out/"ADAPTED_STATE_AFTER_READOUT.json",state_identity(adapted))
    save_json(out/"point_times.json",result["timings"]);save_json(out/"VIEW_RECEIPT.json",view)
    peak=dict(allocator_limit_bytes=limit,peak_allocated_bytes=torch.cuda.max_memory_allocated(0),peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    assert 0<=peak["peak_allocated_bytes"]<=peak["peak_reserved_bytes"]<=limit
    save_json(out/"GPU_PEAK.json",peak)
    save_json(out/"report.json",dict(status="COMPLETED_SYNTHETIC_MATCHED_U_L0",H=512,P=32,C=38,discarded_updates=2,
        gradient_max_by_block=grad,source_and_frozen_state_exact=True,module_modes_restored=True,actual_CUDA_RNG_restored=True,
        bank_points=65,chunks=[64,1],tail31_retained=True,complete_record_pack_roundtrip_exact=True,role_file_loader_exercised=True,
        original_score_and_confirmation_replay_exact=True,production_CLI_or_QUERY_claimed=False,formal_data_read=False,
        parent_checkpoint_loaded=True,technical_weights_reusable=False))
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    save_json(out/"receipt.json",dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT",phase="l0",config_sha256=config_sha,
        source_sha256=c["source_sha256"],query_labels_opened=False,files=files))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","release","trusted-release-sha256"):p.add_argument("--"+name,required=True)
    p.add_argument("--execute",action="store_true");a=p.parse_args();c=admit(a.config,a.release,a.trusted_release_sha256)
    if not a.execute:
        print(json.dumps(dict(metadata_preflight="PASS",Torch_imported=False,model_started=False)));return
    run(c,sha(a.config),a.trusted_release_sha256)


if __name__=="__main__":main()
