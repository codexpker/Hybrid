"""Shared public-manifest Stage1/Z/U/F runner; default preview, explicit execute.

Synthetic smoke uses the same model/objectives/sampler/save-load kernels and is
explicitly non-scientific. It never authorizes real data or GPU execution.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import Run, load_manifest
from .training import fit, adapt, infer, calibrate_route, logit_threshold
from .scoring import apply_calibration


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(clean(value), f, indent=2, ensure_ascii=False, allow_nan=False)


def construct(config):
    from . import HybridV1, HybridV1Config, OfficialSources
    import torch
    torch.manual_seed(config["seed"])
    import os
    settings = HybridV1Config(**config["model"])
    from .scoring import recipe
    active, _ = recipe(config.get("active_components"))
    if (not settings.tf_enabled and "r_tf" in active) or (not settings.ms_enabled and "r_ms" in active):
        raise ValueError("Disabled core branch must be excluded by the fixed active_components recipe")
    if (settings.history, settings.horizon) != (config["history"], config["horizon"]):
        raise ValueError("Core/tool history and horizon must agree")
    sources = {key: os.path.expandvars(value) for key, value in config["sources"].items()}
    if any("$" in value for value in sources.values()):
        raise ValueError("Bind existing official source paths before execution")
    return HybridV1(settings, OfficialSources(**sources)).to(config.get("device", "cpu"))


def synthetic_runs(seed=17, history=128, horizon=2):
    rng = np.random.default_rng(seed)
    runs = []
    # Each role has an explicit independent run; channels remain multivariate.
    for domain, channels in (("A", 1), ("B", 2), ("C", 3)):
        for role in ("FIT", "SELECT", "CAL", "QUERY", "SUPPORT", "NORMAL_SELECT", "NORMAL_CAL"):
            values = rng.normal(size=(history+horizon+8, channels)).astype(np.float32)
            observed = np.ones(values.shape, bool); observed[1, 0] = False
            if role == "SUPPORT":
                runs.append(Run(domain, domain, domain+"_SUPPORT_NORMAL", role, values.copy(), observed.copy()))
                values[history+1:history+3] += 2
            runs.append(Run(domain, domain, domain+"_"+role, role, values, observed))
    return runs


def options(config):
    return {**{k: config[k] for k in ("history", "horizon", "steps", "lr", "kind", "seed")}, "active_components": config.get("active_components")}


def save_model(path, model):
    import torch
    torch.save(model.state_dict(), path)


def load_model(config, path):
    import torch
    model = construct(config)
    model.load_state_dict(torch.load(path, map_location=config.get("device", "cpu"), weights_only=True), strict=True)
    return model


def save_runtime(output, model):
    """Actual imported dependencies and verified upstream bytes, not labels."""
    import importlib.metadata
    import inspect
    import platform
    import torch
    import torch_frft.frft_module as fractional
    location = Path(inspect.getsourcefile(fractional))
    save_json(output/"runtime.json", {"python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "torch_frft": importlib.metadata.version("torch-frft"),
        "installed_frft_source": {"path": str(location), "sha256": hashlib.sha256(location.read_bytes()).hexdigest()},
        "external_source_manifest": model.source_manifest,
        "temporal_class": type(model.temporal.encoder).__module__+"."+type(model.temporal.encoder).__name__,
        "MS_class": type(model.ms_filter.author).__module__+"."+type(model.ms_filter.author).__name__,
        "TF_class": type(model.tf_backbone.author).__module__+"."+type(model.tf_backbone.author).__name__,
        "parameter_counts_flags_not_all_active_paths": model.parameter_counts(),
        "core_config": model.config.to_dict()})


def score_records(records, calibration, event=False):
    output = []
    for record in records:
        row = dict(record)
        if event:
            value = row.get("event_logit", float("nan"))
            row.update(score=value, valid=bool(np.isfinite(value)),
                       alarm=bool(value > calibration["threshold"]) if np.isfinite(value) else None)
        else:
            row.update(apply_calibration(row, calibration))
        output.append(row)
    return output


def execute(config, phase, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save_json(output/"config.json", config)
    kwargs = options(config)
    synthetic = phase == "synthetic_smoke"
    runs = synthetic_runs(config["seed"], config["history"], config["horizon"]) if synthetic else load_manifest(config["manifest"])
    save_json(output/"input_runs.json", [{"domain": r.domain, "entity": r.entity, "run": r.run,
        "role": r.role, "start": r.start, "shape": list(r.values.shape),
        "values_sha256": hashlib.sha256(r.values.tobytes()).hexdigest(),
        "observed_sha256": hashlib.sha256(r.observed.tobytes()).hexdigest()} for r in runs])
    if phase in {"pretrain", "synthetic_smoke"}:
        source = construct(config)
        save_runtime(output, source)
        source, trace = fit(source, [r for r in runs if r.role == "FIT"],
                            [r for r in runs if r.role == "SELECT"], **kwargs)
        save_model(output/"source.pt", source); save_json(output/"source_training.json", trace)
        # Actual save/load path, not only an in-memory fixture branch.
        source = load_model(config, output/"source.pt")
        cal_records = infer(source, [r for r in runs if r.role == "CAL"],
                            history=config["history"], horizon=config["horizon"], seed=config["seed"], active_components=config.get("active_components"))
        source_cal = calibrate_route(cal_records, "STAGE1", active_components=config.get("active_components"))
        save_json(output/"source_calibration_records.json", cal_records)
        save_json(output/"source_calibration.json", source_cal)
        if not synthetic:
            return
    else:
        source = load_model(config, config["checkpoint"])
        save_runtime(output, source)
        source_cal = json.loads(Path(config["source_calibration"]).read_text())
    if phase in {"adapt", "synthetic_smoke"}:
        routes = ("Z", "U", "F") if synthetic else (config["route"],)
        for route in routes:
            pool = [r for r in runs if r.domain == "C"] if synthetic else runs
            support = config.get("support")
            if synthetic:
                h = config["history"]
                support = {"K": 1, "events": [{"event_id": "synthetic_E1", "entity": "C", "run": "C_SUPPORT", "start": h+1, "end": h+3}],
                    "normal": [{"matched_event_id": "synthetic_E1", "entity": "C", "run": "C_SUPPORT_NORMAL", "start": h+1, "end": h+3}],
                    "forbidden": [{"entity": r.entity, "run": r.run, "start": r.start, "end": r.start+len(r.values)}
                                  for r in pool if r.role in {"QUERY", "NORMAL_SELECT", "NORMAL_CAL"}]}
            model, trace = adapt(source, route,
                [r for r in pool if r.role == ("SUPPORT" if route == "F" else "FIT")],
                [r for r in pool if r.role == ("NORMAL_SELECT" if route == "F" else "SELECT")], support=support, **kwargs)
            save_model(output/(route+".pt"), model); save_json(output/(route+"_training.json"), trace)
            actual_route = "Z" if route == "F" and support["K"] == 0 else route
            records = [] if actual_route == "Z" else infer(model,
                            [r for r in pool if r.role == ("NORMAL_CAL" if actual_route == "F" else "CAL")],
                            history=config["history"], horizon=config["horizon"], seed=config["seed"], event_logits=actual_route == "F", active_components=config.get("active_components"))
            calibration = ({"threshold": logit_threshold(records), "normal_support_only": True, "score": "raw_event_logit"}
                           if actual_route == "F" else calibrate_route(records, actual_route, source_cal, config.get("active_components")))
            save_json(output/(route+"_calibration.json"), calibration)
            save_json(output/(route+"_calibration_records.json"), records)
            records = infer(model, [r for r in pool if r.role == "QUERY"], history=config["history"],
                            horizon=config["horizon"], seed=config["seed"], event_logits=actual_route == "F", active_components=config.get("active_components"))
            save_json(output/(route+"_scores.json"), score_records(records, calibration, event=actual_route == "F"))
        save_json(output/"STATUS.json", {"synthetic_non_scientific_results": synthetic,
                  "routes_executed": routes, "not_evidence_of_detection_benefit": True})
    elif phase == "infer":
        calibration = json.loads(Path(config["calibration"]).read_text())
        # A saved F-K0 branch is exactly Z, including later standalone inference.
        event_mode = config["route"] == "F" and calibration.get("score") == "raw_event_logit"
        records = infer(source, [r for r in runs if r.role == "QUERY"], history=config["history"],
                        horizon=config["horizon"], seed=config["seed"], event_logits=event_mode, active_components=config.get("active_components"))
        save_json(output/"scores.json", score_records(records, calibration, event=event_mode))


def evaluate(config, output):
    """Reuse shared evaluator after raw scores exist; label input is exclusive here."""
    from ..candidate_experiment import evaluate_arrays, _clean_json
    records = json.loads(Path(config["scores"]).read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for entry in config["label_runs"]:
        labels = np.load(entry["path"], allow_pickle=False)
        rows = [r for r in records if (r["entity"], r["run"]) == (entry["entity"], entry["run"]) and r["valid"]]
        indices = np.array([r["index"]-entry.get("original_start", 0) for r in rows], dtype=np.int64)
        result = evaluate_arrays(indices, np.array([r["score"] for r in rows]), labels, config["threshold"])
        save_json(output/(entry["entity"]+"_"+entry["run"]+".json"), _clean_json(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("pretrain", "adapt", "infer", "evaluate", "synthetic_smoke"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps({"phase": args.phase, "preview": True, "data_opened": False})); return
    if args.phase == "evaluate":
        evaluate(config, args.output_dir)
    else:
        execute(config, args.phase, args.output_dir)
    output = Path(args.output_dir)
    save_json(output/"receipt.json", {"phase": args.phase,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()},
        "synthetic_non_scientific_results": args.phase == "synthetic_smoke"})


if __name__ == "__main__":
    main()
