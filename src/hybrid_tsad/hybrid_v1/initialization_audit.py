"""CPU-only parameter initialization comparison; no forward/data/train entry.

The original V4 construct function initializes each model independently. The
reference is read for comparison only, never loaded into a constructed model.
"""
import argparse
import hashlib
import json
from pathlib import Path

OPERATION="HYBRID_V1_A1_CPU_INITIALIZATION_PROOF"
MODES={"full":"left_interaction","none":"none","feature_mean":"feature_mean","concat_linear":"concat_linear"}
PERMISSIONS=dict(parameter_initialization_comparison=True,model_forward=False,data_arrays=False,training=False,labels=False,GPU=False)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def config_equivalence(configs):
    if set(configs)!=set(MODES):raise ValueError("One full and3 named controls required")
    full=configs["full"]
    for name,v in configs.items():
        if v.get("seed")!=2025 or v.get("history")!=512 or v.get("horizon")!=32:
            raise ValueError("Fixed seed2025/H512/P32 required")
        if v["model"].get("fusion")!=MODES[name]:raise ValueError("Wrong named fusion control")
        model=dict(v["model"]);model["fusion"]="left_interaction"
        if model!=full["model"] or v["sources"]!=full["sources"] or v["source_sha256"]!=full["source_sha256"]:
            raise ValueError("Only the fusion model flag may change")


def admit(config_path,release_path,trusted):
    cfg=json.loads(Path(config_path).read_text())
    if not release_path or not trusted:raise PermissionError("Named root CPU proof release required")
    if sha(release_path)!=trusted:raise PermissionError("Release SHA differs")
    signed=json.loads(Path(release_path).read_text())
    if (signed.get("status")!="APPROVED" or signed.get("approved_by")!="root" or signed.get("operation")!=OPERATION
        or signed.get("allowed_actions")!=["initialization_proof"] or signed.get("config_sha256")!=sha(config_path)
        or cfg.get("operation")!=OPERATION or cfg.get("permissions")!=PERMISSIONS):
        raise PermissionError("Pending/wrong CPU-only proof release")
    if Path(cfg["output"]).exists():raise FileExistsError("Fresh proof output required")
    own=Path(__file__).resolve().parent
    if (signed.get("audit_module_sha256")!=sha(__file__) or cfg.get("audit_module_sha256")!=sha(__file__)
        or signed.get("source_sha256")!=cfg["source_sha256"]):raise PermissionError("Exact audit/V4 source closure required")
    for name,digest in cfg["source_sha256"].items():
        if Path(name).name!=name or sha(own/name)!=digest:raise PermissionError("V4 source bytes changed")
    for a in cfg["assets"].values():
        if sha(a["path"])!=a["sha256"]:raise PermissionError("Bound proof asset changed")
    configs={name:json.loads(Path(a["path"]).read_text()) for name,a in cfg["configs"].items()}
    for name,a in cfg["configs"].items():
        if sha(a["path"])!=a["sha256"]:raise PermissionError("Source model config changed")
    config_equivalence(configs)
    if configs["full"]["source_sha256"]!=cfg["source_sha256"]:raise PermissionError("Proof/source config closure differs")
    if cfg.get("resources")!={"wall_seconds":120,"kill_grace_seconds":5,"address_space_GiB":16,"core_dump_bytes":0,"GPUh":0,"automatic_retry":False}:
        raise PermissionError("CPU120+5/AS16GiB only")
    return cfg,configs


def tensor_identity(value):
    import torch
    v=value.detach().cpu().contiguous()
    return dict(dtype=str(v.dtype),shape=list(v.shape),numel=v.numel(),
                sha256=hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                finite=bool(torch.isfinite(v).all()))


def process_memory():
    """Linux process counters only; virtual mappings are not physical RSS."""
    wanted={"VmPeak","VmSize","VmHWM","VmRSS"};result={}
    for line in Path("/proc/self/status").read_text().splitlines():
        key,_,value=line.partition(":")
        if key in wanted:
            fields=value.split();result[key+"_bytes"]=int(fields[0])*1024
    return result


def prove(cfg,configs):
    import gc
    import importlib
    import importlib.metadata
    import os
    import platform
    import resource
    import time
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise PermissionError("CUDA must be invisible before Torch import")
    if resource.getrlimit(resource.RLIMIT_CORE)[0]!=0:raise PermissionError("Outer resource layer must disable core dumps")
    # The AS cap is virtual address space, not a claim of a physical RSS limit.
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    import torch
    from .cli import construct
    from .provenance import OfficialSources
    torch.set_num_threads(1)
    frft=importlib.import_module("torch_frft.frft_module")
    if sha(frft.__file__)!=cfg["assets"]["installed_frft"]["sha256"]:
        raise ValueError("Actually imported installed FRFT source differs")
    output=Path(cfg["output"]);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();reference_asset=cfg["assets"]["reference_initial"]
    reference=torch.load(reference_asset["path"],map_location="cpu",weights_only=True)
    if not isinstance(reference,dict) or not reference or not all(isinstance(v,torch.Tensor) for v in reference.values()):
        raise ValueError("Reference must be a nonempty tensor state_dict")
    reference_ids={k:tensor_identity(v) for k,v in reference.items()}
    if not all(v["finite"] for v in reference_ids.values()):raise ValueError("Nonfinite reference initialization")
    reports={};error=None
    try:
        for name in MODES:
            model=construct(dict(configs[name],device="cpu"))
            state=model.state_dict();rows=[]
            keys_equal=set(state)==set(reference)
            for key in sorted(set(state)|set(reference)):
                actual=tensor_identity(state[key]) if key in state else None;expected=reference_ids.get(key)
                same=bool(actual is not None and expected is not None and actual["dtype"]==expected["dtype"]
                          and actual["shape"]==expected["shape"] and torch.equal(state[key].detach().cpu(),reference[key]))
                rows.append(dict(key=key,actual=actual,reference=expected,torch_equal=same))
            reports[name]=dict(requested_fusion=MODES[name],actual_fusion=model.fusion.mode,
                state_keys_equal=keys_equal,all_tensors_equal=keys_equal and all(r["torch_equal"] for r in rows),
                stored_parameter_count=sum(p.numel() for p in model.parameters()),state_tensors=len(state),per_tensor=rows)
            (output/(name+"_state_comparison.json")).write_text(json.dumps(reports[name],indent=2))
            del state,model;gc.collect()
        if not all(r["all_tensors_equal"] and r["actual_fusion"]==r["requested_fusion"] for r in reports.values()):
            raise ValueError("Independent initialization does not exactly match accepted full initial state")
        if sha(reference_asset["path"])!=reference_asset["sha256"]:raise ValueError("Reference bytes changed during proof")
    except BaseException as e:
        error=repr(e);raise
    finally:
        frft=importlib.import_module("torch_frft.frft_module")
        report=dict(status="PASS_CPU_INITIALIZATION_ONLY" if error is None and len(reports)==4 else "FAILED",
            reports={name:{k:v for k,v in row.items() if k!="per_tensor"} for name,row in reports.items()},
            reference_initial=reference_asset,reference_post_sha256=sha(reference_asset["path"]),
            source_sha256=cfg["source_sha256"],audit_module_sha256=sha(__file__),construct_path=construct.__code__.co_filename,
            external_sources=OfficialSources(**configs["full"]["sources"]).manifest(),
            installed_frft=dict(path=frft.__file__,sha256=sha(frft.__file__)),
            versions=dict(Python=platform.python_version(),Torch=torch.__version__,NumPy=importlib.metadata.version("numpy"),torch_frft=importlib.metadata.version("torch-frft")),
            CPU_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,wall_seconds=time.monotonic()-started,
            process_memory=process_memory(),AS_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0],
            core_dump_limit_bytes=resource.getrlimit(resource.RLIMIT_CORE)[0],
            memory_semantics="AS16GiB bounds virtual mappings, not RSS; actual physical peak is ru_maxrss/VmHWM",
            requested_fresh_constructs=4,forward_calls=0,optimizer_calls=0,data_arrays_opened=False,GPUh=0,
            no_reference_loaded_into_model=True,no_usable_checkpoint_saved=True,error=error)
        (output/"report.json").write_text(json.dumps(report,indent=2))
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True);p.add_argument("--release")
    p.add_argument("--trusted-release-sha256");p.add_argument("--execute",action="store_true");p.add_argument("--preflight-only",action="store_true");a=p.parse_args(argv)
    if not a.execute:print(json.dumps(dict(preview=True,operation=OPERATION,Torch_or_model_or_weight_opened=False)));return
    cfg,configs=admit(a.config,a.release,a.trusted_release_sha256)
    if a.preflight_only:print(json.dumps(dict(metadata="PASS",Torch_or_model_constructed=False)));return
    print(json.dumps(prove(cfg,configs)))


if __name__=="__main__":main()
