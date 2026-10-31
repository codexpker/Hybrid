"""Frozen canonical simple/late controls; source-only fitting, never query labels."""
from pathlib import Path

import numpy as np

from ..config import read_json, sha256
from ..protocol import score_block
from .source import calibrate_source, read_plan
from .training import source_batch
from .readout_manifest import MODES, SEEDS, checked, json_write, record
from .identity import source_calibration


def simple_forward(x, method):
    """Existing readout_hybrid_controls.forward canonical float32 torch formula."""
    if method not in ("persistence", "history_mean"):
        raise ValueError("Unknown fixed simple method")
    import torch
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Canonical simple history is nonfinite; no implicit filtering/imputation")
    scale = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    mean = x.mean(1, keepdim=True)
    prediction = (x[:, -1:] if method == "persistence" else mean).expand(-1, 5, -1)
    return {"prediction": prediction, "scale": scale, "mean": mean}


def simple_bank(data, origins, method, scale):
    import torch
    from .bank import collect_forecasts
    # A zero-training existing baseline adapter, not a learned model.
    class FixedBaseline:
        training = False
        history = 100
        def parameters(self):
            yield torch.zeros(())
        def __call__(self, history):
            return simple_forward(history, method)
    return collect_forecasts(FixedBaseline(), data, origins, scale, mask_seed=84551)[0]


def _bound_source_calibration(seed,models,assets,definition_path):
    from ..data import ManifestSampler
    paths = {k:checked(v,definition_path) for k,v in assets.items()}
    inputs, rows = read_json(paths["source_inputs"]), read_json(paths["rows"])
    manifest=read_json(paths["source_manifest"])
    if (rows!=manifest["records"] or Path(inputs["manifest"]).resolve()!=paths["source_manifest"]
            or any(r["dataset"] not in ("PSM","SMD_SUBSET") or r["history"]!=100 or r["horizon"]!=5 for r in rows)):
        raise ValueError("Control source rows differ from the frozen source manifest/roles/history")
    required={paths["source_manifest"]}|{Path(r["source"]).resolve() for r in rows}
    if not required<={Path(p).resolve() for p in inputs["hashes"]}:
        raise ValueError("Actual simple source paths are missing frozen input hashes")
    if any(sha256(p)!=h for p,h in inputs["hashes"].items()):
        raise ValueError("Simple source inputs/roles/history mismatch")
    metadata = [read_json(checked(m["checkpoint_metadata"], definition_path)) for m in models if m["seed"]==seed]
    if any(m["shared_plan_sha256"]!=sha256(paths["plan"]) or m["source_manifest_sha256"]!=sha256(paths["source_manifest"]) for m in metadata):
        raise ValueError("Controls do not use each model's source plan/manifest")
    plan, sampler = read_plan(paths["plan"]), ManifestSampler(paths["source_manifest"],exclude_dataset="TEP")
    for suffix,dtype in (("record","uint16"),("origin","uint32"),("channel","uint8")):
        if plan["cal_"+suffix].shape!=(256,) or plan["cal_"+suffix].dtype!=np.dtype(dtype):
            raise ValueError("Control cal trace exposure/dtype changed")
    for ri,origin,channel in zip(plan["cal_record"],plan["cal_origin"],plan["cal_channel"]):
        if int(ri)>=len(rows):raise ValueError("Control cal record is outside source rows")
        row=rows[int(ri)];b,c=int(origin),int(channel)
        if (row["role"]!="cal" or b not in row["forecast_origins"] or not 0<=c<len(row["columns"])
                or b-100<row["local_start"] or b+5>row["local_end_exclusive"]):
            raise ValueError("Control cal trace role/channel/footprint mismatch")
    domains=[rows[int(ri)]["dataset"] for ri in plan["cal_record"]]
    if domains!=[["PSM","SMD_SUBSET"][(n//64)%2] for n in range(256)]:
        raise ValueError("Control source cal domain order changed")
    actual_targets=[]
    for start in range(0,256,64):
        _,target=source_batch(sampler,plan,rows,"cal",start,device="cpu",history=100)
        actual_targets.append(target.numpy())
    actual_targets=np.concatenate(actual_targets)
    actual_valid=np.isfinite(actual_targets)
    for native in (m for m in models if m["seed"]==seed):
        with np.load(checked(native["cal_bank"],definition_path),allow_pickle=False) as bank:
            if (not np.array_equal(bank["target"],actual_targets,equal_nan=True)
                    or not np.array_equal(bank["valid_target"],actual_valid)):
                raise ValueError("Native cal target/mask differs from the actual frozen source plan")
    return paths,plan,rows,sampler


def _simple_source_bank(sampler,plan,rows,method):
    import torch
    parts = {k:[] for k in ("prediction","target","history_scale","valid_target")}
    with torch.no_grad():
        for start in range(0,len(plan["cal_record"]),64):
            x,y = source_batch(sampler,plan,rows,"cal",start,device="cpu",history=100)
            o = simple_forward(x,method)
            for k,v in (("prediction",o["prediction"]),("target",y),("history_scale",o["scale"]),("valid_target",torch.isfinite(y))):
                parts[k].append(v.numpy())
    bank = {k:np.concatenate(v) for k,v in parts.items()}
    return bank


def validate_catalog_evidence(catalog,base):
    """Pure re-derivation from fixed source evidence, before any target read."""
    if "source_definition" not in catalog or "source_assets" not in catalog:
        raise ValueError("Catalog lacks frozen source evidence")
    definition_path=checked(catalog["source_definition"],base)
    definition=read_json(definition_path)
    models={(m["seed"],m["mode"]):m for m in catalog["models"]}
    originals={(m["seed"],m["mode"]):m for m in definition["models"]}
    controls={(c["seed"],c["method"]):c for c in catalog["controls"]}
    if set(models)!=set(originals):raise ValueError("Catalog source model evidence set changed")
    if record(checked(catalog["official"],base))!=record(checked(definition["official"],definition_path)):
        raise ValueError("Catalog official source evidence changed")
    source_scores={}
    for key,model in models.items():
        original=originals[key]
        for field in ("checkpoint","checkpoint_metadata","calibration","cal_bank","cal_scores"):
            if record(checked(model[field],base))!=record(checked(original[field],definition_path)):
                raise ValueError("Catalog native source artifact evidence changed")
        with np.load(checked(model["cal_bank"],base),allow_pickle=False) as bank:
            recomputed,parameters=calibrate_source(bank)
        with np.load(checked(model["cal_scores"],base),allow_pickle=False) as scores:
            if any(not np.array_equal(scores[k],v,equal_nan=True) for k,v in recomputed.items()):
                raise ValueError("Native calibration score array evidence mismatch")
        cal=source_calibration(checked(model["calibration"],base));control=controls[key]
        if (cal.get("source_bank_sha256")!=model["cal_bank"]["sha256"]
                or cal["normal_error_scale"]!=parameters["normal_error_scale"] or cal["threshold99"]!=parameters["threshold99"]
                or control["scale"]!=parameters["normal_error_scale"] or control["threshold"]!=parameters["threshold99"]
                or control["kind"]!="native" or control["provenance"]["source_calibration"]!=model["calibration"]):
            raise ValueError("Native catalog units differ from frozen source calibration evidence")
        source_scores[key]=recomputed
    for seed in sorted({s for s,_ in models}):
        assets=catalog["source_assets"][str(seed)]
        original_assets=definition["source_assets"][str(seed)]
        if set(assets)!=set(original_assets) or any(record(checked(assets[k],base))!=record(checked(original_assets[k],definition_path)) for k in assets):
            raise ValueError("Catalog source rows/plan/input evidence changed")
        paths,plan,rows,sampler=_bound_source_calibration(seed,list(models.values()),assets,base)
        if catalog["scope"]!="full":continue
        t,f=source_scores[seed,"temporal"],source_scores[seed,"tf"]
        if not np.array_equal(t["valid"],f["valid"]):raise ValueError("Late source calibration masks differ")
        combo=.5*(t["standardized_score"]+f["standardized_score"])
        late=controls[seed,"late"]
        if (late["scale"]!=1. or late["threshold"]!=float(np.quantile(combo[t["valid"]],.99))
                or late["kind"]!="fixed_half_native"
                or late["provenance"]!={"temporal_cal":models[seed,"temporal"]["cal_scores"],"tf_cal":models[seed,"tf"]["cal_scores"]}):
            raise ValueError("Late catalog threshold differs from fixed paired native source calibration")
        for method in ("history_mean","persistence"):
            control=controls[seed,method]
            expected=_simple_source_bank(sampler,plan,rows,method)
            evidence=checked(control["provenance"]["source_cal_bank"],base)
            with np.load(evidence,allow_pickle=False) as bank:
                if any(not np.array_equal(bank[k],v,equal_nan=True) for k,v in expected.items()):
                    raise ValueError("Simple cal bank differs from canonical actual source plan predictions/targets")
                _,parameters=calibrate_source(bank)
            if (control["scale"]!=parameters["normal_error_scale"] or control["threshold"]!=parameters["threshold99"]
                    or control["kind"]!="canonical_torch_cpu_float32"
                    or control["provenance"]["source_plan"]!=record(paths["plan"])):
                raise ValueError("Simple catalog units differ from frozen source-cal bank/formula")
    return True


def prepare_catalog(definition_path, output, profile, scope="full"):
    """Assemble verified native artifacts and fit missing fixed source controls."""
    from ..data import ManifestSampler
    definition = read_json(definition_path)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    models, controls = [], []
    expected = {(s, m) for s in (SEEDS if scope == "full" else (0,)) for m in MODES}
    found = set()
    source_scores = {}
    for spec in definition["models"]:
        s, m = spec["seed"], spec["mode"]
        if (s, m) not in expected or (s, m) in found:
            raise ValueError("Unexpected/duplicate source model")
        found.add((s, m))
        paths = {key: checked(spec[key], definition_path) for key in ("checkpoint", "checkpoint_metadata", "calibration", "cal_scores", "cal_bank")}
        meta = read_json(paths["checkpoint_metadata"])
        cal = source_calibration(paths["calibration"])
        if (meta.get("seed") != s or meta.get("mode") != m or meta.get("history") != 100 or meta.get("horizon") != 5
                or meta.get("excluded_target") != "TEP" or set(meta.get("source_domains", [])) != {"PSM", "SMD_SUBSET"}
                or meta.get("checkpoint_sha256") != sha256(paths["checkpoint"]) or meta.get("step") != 10000
                or cal.get("history") != 100 or cal.get("role") != "source_cal"
                or cal.get("source_bank_sha256") != sha256(paths["cal_bank"])):
            raise ValueError("Source checkpoint/calibration identity is not TEP-excluded100 step10000")
        with np.load(paths["cal_scores"], allow_pickle=False) as scores:
            source_scores[s, m] = {key: scores[key].copy() for key in scores.files}
        with np.load(paths["cal_bank"],allow_pickle=False) as bank:
            recomputed,parameters=calibrate_source(bank)
        if (any(not np.array_equal(source_scores[s,m][key],value,equal_nan=True) for key,value in recomputed.items())
                or parameters["normal_error_scale"]!=cal["normal_error_scale"] or parameters["threshold99"]!=cal["threshold99"]):
            raise ValueError("Native source-cal scores/scales do not match their frozen prediction bank")
        models.append({"seed":s, "mode":m, **{k:record(p) for k,p in paths.items()}})
        controls.append(dict(seed=s, method=m, scale=cal["normal_error_scale"], threshold=cal["threshold99"],
                             provenance={"source_calibration":record(paths["calibration"])}, kind="native"))
    if found != expected:
        raise ValueError("Source catalog is incomplete")
    if scope == "full":
        for seed in SEEDS:
            t, f = source_scores[seed,"temporal"], source_scores[seed,"tf"]
            if not np.array_equal(t["valid"], f["valid"]):
                raise ValueError("Late source calibration masks differ")
            combo = .5 * (t["standardized_score"] + f["standardized_score"])
            controls.append(dict(seed=seed, method="late", scale=1., threshold=float(np.quantile(combo[t["valid"]],.99)),
                                 provenance={"temporal_cal":next(m["cal_scores"] for m in models if m["seed"]==seed and m["mode"]=="temporal"),
                                             "tf_cal":next(m["cal_scores"] for m in models if m["seed"]==seed and m["mode"]=="tf")}, kind="fixed_half_native"))
            paths,plan,rows,sampler=_bound_source_calibration(seed,models,definition["source_assets"][str(seed)],definition_path)
            import torch
            for method in ("history_mean","persistence"):
                bank=_simple_source_bank(sampler,plan,rows,method)
                arrays, cal = calibrate_source(bank)
                folder=output/f"s{seed}"/method;folder.mkdir(parents=True)
                np.savez_compressed(folder/"source_cal_bank.npz",**bank)
                np.savez_compressed(folder/"source_cal_scores.npz",**arrays)
                controls.append(dict(seed=seed,method=method,scale=cal["normal_error_scale"],threshold=cal["threshold99"],kind="canonical_torch_cpu_float32",
                                     provenance={"source_cal_bank":record(folder/"source_cal_bank.npz"),"source_plan":record(paths["plan"])}))
    catalog={"schema_version":1,"scope":scope,"profile":profile,"models":models,"controls":controls,
             "official":record(checked(definition["official"],definition_path)),"labels_opened":False,
             "simple_backend":"torch_cpu_float32_canonical_v1","source_definition":record(definition_path),
             "source_assets":{str(seed):{k:record(checked(v,definition_path)) for k,v in definition["source_assets"][str(seed)].items()}
                              for seed in sorted({m["seed"] for m in models})}}
    json_write(output/"catalog.json",catalog)
    return catalog
