"""Multi-entity fixed zero-shot readout, exact shards, sealing and label lock."""
from pathlib import Path
import math
import time

import numpy as np

from ..api import create_predictor, verify_checkpoint_contract
from ..config import OFFICIAL_SHA, read_json, sha256
from .bank import collect_forecasts, inference_metadata, write_bank
from .identity import source_calibration
from .physical_validation import (bind_execution_evidence, physical_validation_spec,
                                  validate_physical_runtime)
from .readout_controls import prepare_catalog, simple_bank, validate_catalog_evidence
from .readout_manifest import (METHODS, MODES, SEEDS, approve_readout_inputs, checked,
                              entity_profile, expected_native, expected_scores, json_write,
                              load_readout_operation, native_key, record, score_key, validate_whitelist)

SCORE_SPACE = "fixed_native_late_canonical_source_controls_v1"
HISTORY_SCALE_RTOL = 2e-6
HISTORY_SCALE_ATOL = 2e-6


def physical_reference(q,profile,*,physical_validation=None):
    """Read-only original values and the unchanged core preprocessing formula.

    Reference reductions use the explicitly approved production backend.
    No score/threshold/label can choose a different normalization version.
    """
    import torch
    from types import SimpleNamespace
    from ..model import HybridPredictor
    spec=physical_validation_spec(physical_validation,profile)
    validate_physical_runtime(spec)
    path=Path(q["array_path"])
    if sha256(path)!=q["array_sha256_from_frozen_parent"]:
        raise ValueError("Original physical query array identity mismatch")
    data=np.load(path,mmap_mode="r",allow_pickle=False)
    if data.shape!=(profile["points"],profile["channels"]) or data.dtype!=np.float32 or not np.isfinite(data).all():
        raise ValueError("Full fixed readout needs finite original float32 query values; no implicit repair")
    origins=q["origins"]
    if any(b-100<q["local_start"] or b+5>q["local_end_exclusive"] for b in origins):
        raise ValueError("Original physical forecast footprint exceeds entity bounds")
    scales=[]
    with torch.no_grad():
        for b in origins:
            history=torch.tensor(data[None,b-100:b].copy(),device=spec["device"])
            _,_,_,scale=HybridPredictor.preprocess(SimpleNamespace(history=100),history)
            scales.append(scale.cpu().numpy()[0])
    index=np.concatenate([np.arange(b,b+5) for b in origins])
    return data,index,np.stack(scales)


def verify_physical_bank(path,q,profile,reference):
    data,index,scales=reference
    with np.load(path,allow_pickle=False) as bank:
        if (bank["time_index"].dtype.kind not in "iu" or bank["forecast_origin"].dtype.kind not in "iu"
                or not np.array_equal(bank["time_index"],index)
                or not np.array_equal(bank["forecast_origin"],q["origins"])):
            raise ValueError("Bank time index/origin differs from the original physical footprint")
        if not np.array_equal(bank["target"],data[index]):
            raise ValueError("Bank physical target differs from actual original query values")
        if (bank["history_scale_by_block"].shape!=scales.shape
                or not np.isfinite(bank["history_scale_by_block"]).all()
                or not np.allclose(bank["history_scale_by_block"],scales,rtol=HISTORY_SCALE_RTOL,atol=HISTORY_SCALE_ATOL)):
            raise ValueError("Bank history scale differs from the unchanged original preprocess formula")


def load_catalog(path, *, require_full=False):
    catalog=read_json(path)
    pairs=[(m["seed"],m["mode"]) for m in catalog["models"]]
    allowed={(s,m) for s in SEEDS for m in MODES}
    if len(pairs)!=len(set(pairs)) or not set(pairs)<=allowed:
        raise ValueError("Duplicate/unknown model catalog entries")
    if require_full and (set(pairs)!=allowed or catalog["scope"]!="full"):
        raise ValueError("Full readout requires all9 frozen source models")
    controls={(x["seed"],x["method"]):x for x in catalog["controls"]}
    if len(controls)!=len(catalog["controls"]):
        raise ValueError("Duplicate source controls")
    if require_full and set(controls)!={(s,m) for s in SEEDS for m in METHODS}:
        raise ValueError("Full readout requires all18 source-calibrated controls")
    for control in controls.values():
        if not math.isfinite(control["scale"]) or control["scale"]<=0 or not math.isfinite(control["threshold"]) or control["threshold"]<0:
            raise ValueError("Invalid frozen control scale/threshold")
    validate_catalog_evidence(catalog,path)
    return catalog,{(x["seed"],x["mode"]):x for x in catalog["models"]},controls


def infer_shard(catalog_path, query_path, whitelist_path, output, *, profile, scope, device="cpu", predictor_factory=None):
    import torch
    catalog,models,controls=load_catalog(catalog_path,require_full=scope=="shard")
    if catalog["profile"]!=profile:
        raise ValueError("Catalog/query profile mismatch")
    entities=entity_profile(read_json(query_path),profile)
    cells=validate_whitelist(read_json(whitelist_path),entities,scope)
    official=checked(catalog["official"],catalog_path)
    if profile["name"]=="TEP100" and sha256(official)!=OFFICIAL_SHA:
        raise ValueError("External official model differs from pinned implementation")
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    archived={}
    torch.set_num_threads(2)
    for cell in cells:
        entity_started=time.perf_counter()
        cuda_device=torch.device(device)
        if cuda_device.type=="cuda":
            torch.cuda.set_device(cuda_device)
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(cuda_device)
        seed,mode,entity=cell["seed"],cell["mode"],cell["entity"]
        spec=models[seed,mode];q=entities[entity]
        checkpoint=checked(spec["checkpoint"],catalog_path)
        metadata=checked(spec["checkpoint_metadata"],catalog_path)
        verify_checkpoint_contract(checkpoint,metadata,history=100,mode=mode)
        meta=read_json(metadata)
        if meta.get("seed")!=seed or meta.get("excluded_target")!="TEP" or set(meta.get("source_domains",[]))!={"PSM","SMD_SUBSET"}:
            raise ValueError("Checkpoint seed/source-exclusion provenance mismatch")
        calibration=source_calibration(checked(spec["calibration"],catalog_path))
        if (calibration.get("history")!=100 or calibration["normal_error_scale"]!=controls[seed,mode]["scale"]
                or calibration["threshold99"]!=controls[seed,mode]["threshold"]
                or calibration["operation_manifest_sha256"]!=meta.get("operation_manifest_sha256")):
            raise ValueError("Native checkpoint/calibration history, operation or score identity mismatch")
        array=Path(q["array_path"])
        if sha256(array)!=q["array_sha256_from_frozen_parent"]:
            raise ValueError("Query array identity mismatch")
        torch.manual_seed(seed)
        factory=predictor_factory or create_predictor
        model=factory(official,mode=mode,history=100,checkpoint=checkpoint,checkpoint_metadata=metadata,device=device)
        data=np.load(array,mmap_mode="r",allow_pickle=False)
        if data.shape!=(profile["points"],profile["channels"]) or data.dtype!=np.float32:
            raise ValueError("Query array shape/dtype mismatch")
        index=np.concatenate([np.arange(b,b+5) for b in cell["origins"]])
        scale=controls[seed,mode]["scale"]
        arrays,rng,times=collect_forecasts(model,data,cell["origins"],scale,mask_seed=84551,expected_index=index)
        key=native_key(seed,mode,entity)
        report=inference_metadata(key,{"seed":seed,"mode":mode,"entity":entity,"source_scale":scale,"history":100},
            points=len(index),blocks=len(cell["origins"]),mask_seed=84551,checkpoint_sha256=sha256(checkpoint),
            query_array_sha256=q["array_sha256_from_frozen_parent"],seconds=time.perf_counter()-entity_started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(cuda_device) if cuda_device.type=="cuda" else None,
            forward_p50_seconds=float(np.median(times)),forward_p95_seconds=float(np.quantile(times,.95)))
        report.update(scope=scope,catalog_sha256=sha256(catalog_path),query_manifest_sha256=sha256(query_path))
        report.update(seconds_scope="entity model/input-I-O/compute before bank output serialization; excludes global catalog checks",
                      forward_timing_samples=len(times),forward_timing_scope="primary forward calls only; replays unmeasured",
                      resource_caveat="two-block L0 quantiles are not a full-group cost forecast" if len(times)<=2 else "per-entity finite samples; not worker elapsed or GPU-hours")
        folder=output/"native"/key
        write_bank(folder,arrays,rng,report)
        whole_seconds=time.perf_counter()-entity_started
        archived[key]={"bank":record(folder/"predictions_and_scores.npz"),"rng":record(folder/"rng_states.npz"),
                       "metadata":record(folder/"inference.json"),"origins":cell["origins"],
                       "whole_entity_seconds":whole_seconds,"whole_entity_seconds_scope":"includes bank NPZ/RNG/metadata output serialization"}
        del model,data
    status={"schema_version":1,"status":"L0_ONLY_LABELS_LOCKED" if scope=="l0" else "SHARD_ONLY_LABELS_LOCKED",
            "scope":scope,"profile":profile,"catalog_sha256":sha256(catalog_path),"query_manifest_sha256":sha256(query_path),
            "whitelist":cells,"banks":archived,"labels_opened":False,"complete_group":False}
    json_write(output/"shard_manifest.json",status)
    return status


def collect_complete_native(shard_list_path,catalog_path,query_path,profile):
    catalog,models,controls=load_catalog(catalog_path,require_full=True)
    entities=entity_profile(read_json(query_path),profile)
    banks={}
    for shard_ref in read_json(shard_list_path):
        path=checked(shard_ref,shard_list_path);shard=read_json(path,max_bytes=64*1024*1024)
        if (shard["scope"]!="shard" or shard["catalog_sha256"]!=sha256(catalog_path)
                or shard["query_manifest_sha256"]!=sha256(query_path) or shard["profile"]!=profile or shard["labels_opened"] is not False):
            raise ValueError("L0/foreign/label-open shard cannot enter full seal")
        cells=validate_whitelist(shard["whitelist"],entities,"shard")
        if set(shard["banks"])!={native_key(c["seed"],c["mode"],c["entity"]) for c in cells}:
            raise ValueError("Shard output keys differ from exact whitelist")
        for key,value in shard["banks"].items():
            if key in banks:
                raise ValueError("Duplicate bank across readout shards")
            paths={k:checked(value[k],path) for k in ("bank","rng","metadata")}
            meta=read_json(paths["metadata"])
            cell=next(c for c in cells if native_key(c["seed"],c["mode"],c["entity"])==key)
            s,m,e=cell["seed"],cell["mode"],cell["entity"]
            q=entities[e]
            expected_meta=dict(seed=s,mode=m,entity=e,history=100,source_scale=controls[s,m]["scale"],
                checkpoint_sha256=models[s,m]["checkpoint"]["sha256"],query_array_sha256=q["array_sha256_from_frozen_parent"],
                points=len(q["origins"])*5,blocks=len(q["origins"]),mask_seed=84551,scope="shard")
            if (meta["bank_sha256"]!=sha256(paths["bank"]) or meta["labels_opened"] is not False
                    or any(meta.get(k)!=v for k,v in expected_meta.items()) or value["origins"]!=q["origins"]):
                raise ValueError("Bank/RNG metadata identity mismatch")
            with np.load(paths["rng"],allow_pickle=False) as rng:
                if rng["before_each_block"].ndim!=2 or len(rng["before_each_block"])!=len(q["origins"]):
                    raise ValueError("Incomplete per-block RNG archive")
            banks[key]={**value,"paths":paths}
    missing=expected_native(entities)-set(banks)
    if missing or set(banks)-expected_native(entities):
        raise ValueError(f"All-bank label lock: missing/extra native banks ({len(missing)} missing)")
    return catalog,entities,banks,controls


def seal_scores(shard_list_path,catalog_path,query_path,output,*,profile,physical_validation=None,execution_evidence_path=None):
    spec=physical_validation_spec(physical_validation,profile)
    catalog,entities,banks,controls=collect_complete_native(shard_list_path,catalog_path,query_path,profile)
    evidence=bind_execution_evidence(execution_evidence_path,shard_list_path,banks,catalog_path,query_path,profile,spec)
    validate_physical_runtime(spec)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    scores={}
    for entity,q in entities.items():
        reference=physical_reference(q,profile,physical_validation=spec)
        data,index,_=reference
        native={}
        for seed in SEEDS:
            for mode in MODES:
                key=native_key(seed,mode,entity);descriptor=banks[key]
                verify_physical_bank(descriptor["paths"]["bank"],q,profile,reference)
                with np.load(descriptor["paths"]["bank"],allow_pickle=False) as bank:
                    if not np.array_equal(bank["time_index"],index):
                        raise ValueError("Native bank lost original entity time index")
                    shape=(len(index),profile["channels"])
                    if any(bank[k].shape!=shape or not np.isfinite(bank[k]).all() for k in ("prediction","target","normalized_channel_residual")):
                        raise ValueError("Incomplete physical prediction/target/channel-residual bank")
                    scales=bank["history_scale_by_block"]
                    if scales.shape!=(len(q["origins"]),1,profile["channels"]) or not np.isfinite(scales).all() or np.any(scales<=0):
                        raise ValueError("Invalid original per-block history scales")
                    expanded=np.repeat(scales,5,axis=1).reshape(shape)
                    residual=np.abs(bank["prediction"]-bank["target"])/expanded
                    if (not np.array_equal(residual,bank["normalized_channel_residual"])
                            or not np.array_equal(bank["raw_score"],residual.mean(-1))
                            or not np.array_equal(bank["valid"],np.ones(len(index),bool))):
                        raise ValueError("Native raw-bank arithmetic/validity is inconsistent")
                    score=bank["standardized_score"].copy()
                    if not np.array_equal(score,bank["raw_score"]/controls[seed,mode]["scale"]):
                        raise ValueError("Native bank score units differ from fixed source scale")
                    native[seed,mode]=score
        simple={m:simple_bank(data,q["origins"],m,1.)["raw_score"] for m in ("history_mean","persistence")}
        for seed in SEEDS:
            values={m:native[seed,m] for m in MODES}
            values["late"]=.5*(native[seed,"temporal"]+native[seed,"tf"])
            values.update({m:simple[m]/controls[seed,m]["scale"] for m in simple})
            for method,value in values.items():
                if value.dtype!=np.float32 or value.shape!=index.shape or not np.isfinite(value).all():
                    raise ValueError("Fixed scores must retain finite canonical float32/time-index shape")
                key=score_key(seed,method,entity);path=output/"scores"/(key+".npz")
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,time_index=index,score=value)
                scores[key]={"file":record(path),"threshold":controls[seed,method]["threshold"],"seed":seed,"method":method,
                             "entity":entity,"fault_id":q["fault_id"],"run_id":q["run_id"]}
    if set(scores)!=expected_scores(entities):
        raise ValueError("Continuous score archive is incomplete")
    seal={"schema_version":1,"status":"ALL_FIXED_SCORES_SEALED_LABELS_LOCKED","complete_group":True,"labels_opened":False,
          "profile":profile,"catalog":record(catalog_path),"query_manifest":record(query_path),
          "native_banks":{k:{n:v[n] for n in ("bank","rng","metadata")} for k,v in banks.items()},"scores":scores,
          "counts":{"native":len(banks),"scores":len(scores)},"overall_Aff_F1":None}
    if evidence is not None:checked(evidence,execution_evidence_path)
    seal.update(physical_validation=spec,execution_evidence=evidence)
    if evidence is not None:seal["source_shards"]=record(shard_list_path)
    json_write(output/"group_seal.json",seal)
    return seal


def verify_label_unlock(seal_path,catalog_path,query_path,profile,*,physical_validation=None):
    spec=physical_validation_spec(physical_validation,profile)
    seal=read_json(seal_path,max_bytes=32*1024*1024)
    if seal.get("physical_validation",physical_validation_spec(None,{"name":"SYNTHETIC"}))!=spec:
        raise ValueError("Seal/evaluate physical validation backend mismatch")
    evidence_path=checked(seal["execution_evidence"],seal_path) if seal.get("execution_evidence") is not None else None
    shards_path=checked(seal["source_shards"],seal_path) if "source_shards" in seal else None
    bind_execution_evidence(evidence_path,shards_path,seal["native_banks"],catalog_path,query_path,profile,spec)
    validate_physical_runtime(spec)
    catalog,models,controls=load_catalog(catalog_path,require_full=True)
    entities=entity_profile(read_json(query_path),profile)
    if (seal["status"]!="ALL_FIXED_SCORES_SEALED_LABELS_LOCKED" or seal["complete_group"] is not True
            or seal["labels_opened"] is not False or seal["profile"]!=profile
            or seal["catalog"]!=record(catalog_path) or seal["query_manifest"]!=record(query_path)
            or set(seal["native_banks"])!=expected_native(entities) or set(seal["scores"])!=expected_scores(entities)):
        raise ValueError("All-bank label lock: seal cardinality/profile/source identities incomplete")
    # Every output is rehashed before returning permission to load any label.
    references={}
    for key,value in seal["native_banks"].items():
        for part in ("bank","rng","metadata"):
            checked(value[part],seal_path)
        metadata=read_json(checked(value["metadata"],seal_path))
        entity=metadata.get("entity")
        if (entity not in entities or key!=native_key(metadata.get("seed"),metadata.get("mode"),entity)
                or metadata.get("checkpoint_sha256")!=models[metadata["seed"],metadata["mode"]]["checkpoint"]["sha256"]):
            raise ValueError("Native bank metadata entity is outside the locked query group")
        if entity not in references:references[entity]=physical_reference(entities[entity],profile,physical_validation=spec)
        verify_physical_bank(checked(value["bank"],seal_path),entities[entity],profile,references[entity])
    for key,value in seal["scores"].items():
        path=checked(value["file"],seal_path)
        if key!=score_key(value["seed"],value["method"],value["entity"]) or value["entity"] not in entities:
            raise ValueError("Score archive descriptor identities differ from the expected group")
        if value["threshold"]!=controls[value["seed"],value["method"]]["threshold"]:
            raise ValueError("Threshold changed after source calibration")
        q=entities[value["entity"]]
        index=np.concatenate([np.arange(b,b+5) for b in q["origins"]])
        with np.load(path,allow_pickle=False) as bank:
            if not np.array_equal(bank["time_index"],index) or bank["score"].shape!=index.shape or not np.isfinite(bank["score"]).all():
                raise ValueError("Score archive lost its original entity index/finite scores")
    return seal,entities


def run_readout(config_path,*,execute=False,allow_labels=False):
    cfg,paths,output,manifest,digest=load_readout_operation(config_path)
    if cfg["score_space"]!=SCORE_SPACE:
        raise ValueError("Unknown fixed multi-method score-space")
    if not execute:
        return {"executed":False,"operation_manifest":manifest,"operation_manifest_sha256":digest,
                "scope":"preview only; no target values or labels opened"}
    approve_readout_inputs(cfg,paths,output,config_path)
    profile=cfg["parameters"]["profile"]
    if cfg["operation"]=="readout-controls":
        return prepare_catalog(paths["definition"],output,profile,cfg["parameters"].get("scope","full"))
    if cfg["operation"]=="readout-infer":
        return infer_shard(paths["catalog"],paths["query_manifest"],paths["whitelist"],output,
                           profile=profile,scope=cfg["parameters"]["scope"],device=cfg["parameters"]["device"])
    if cfg["operation"]=="readout-seal":
        return seal_scores(paths["shards"],paths["catalog"],paths["query_manifest"],output,profile=profile,
                           physical_validation=cfg["parameters"].get("physical_validation"),execution_evidence_path=paths.get("execution_evidence"))
    if not allow_labels:
        raise ValueError("Evaluation requires explicit --allow-labels after all-bank seal")
    from .multi_evaluation import evaluate_group
    return evaluate_group(paths["seal"],paths["catalog"],paths["query_manifest"],paths["affiliation"],output,profile=profile,
                          physical_validation=cfg["parameters"].get("physical_validation"))
