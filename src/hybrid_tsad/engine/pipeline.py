"""Unified, opt-in execution over external assets with explicit contract identity."""
import json
import math
from pathlib import Path
import re

from ..config import read_json, resolve_path, sha256
from .identity import SCORE_SPACE, operation_manifest, verify_scoring

REQUIRED_INPUTS = {"train": {"runner"}, "infer": {"official", "checkpoint", "array", "query", "calibration"},
                   "evaluate": {"bank", "bank_metadata", "labels", "coverage", "calibration"}, "calibrate": {"bank"}}
PARAMETERS = {"train": ({"run", "job"}, set()),
              "infer": ({"seed", "mode", "source_scale", "mask_seed", "readout_job", "job"}, {"device"}),
              "evaluate": ({"threshold"}, set()), "calibrate": ({"bank_role"}, set())}


def load_operation(path, operation):
    cfg = read_json(path)
    if set(cfg) != {"engine_schema_version", "operation", "contract", "approval", "inputs", "parameters", "output", "score_space"}:
        raise ValueError("Engine config requires version/operation/contract/approval/inputs/parameters/output")
    if cfg["engine_schema_version"] != 2 or type(cfg["engine_schema_version"]) is not int or cfg["operation"] != operation:
        raise ValueError("Engine schema version or operation mismatch")
    if cfg["score_space"] != (None if operation == "train" else SCORE_SPACE):
        raise ValueError("Unsupported operation score-space; no implicit legacy score mapping")
    if not isinstance(cfg["parameters"], dict) or set(cfg["inputs"]) != REQUIRED_INPUTS[operation]:
        raise ValueError("Wrong operation input names or parameters type")
    required, optional = PARAMETERS[operation]
    if not required <= set(cfg["parameters"]) or set(cfg["parameters"]) - required - optional:
        raise ValueError("Missing or unknown operation parameter")
    records = {"contract": cfg["contract"], "approval": cfg["approval"], **cfg["inputs"]}
    paths = {}
    for name, record in records.items():
        if (not isinstance(record, dict) or set(record) != {"path", "sha256"}
                or not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ValueError(f"{name} requires a path and lowercase SHA256")
        paths[name] = resolve_path(record["path"], path)
    output = resolve_path(cfg["output"], path) if cfg["output"] is not None else None
    if operation != "train" and output is None:
        raise ValueError("This operation requires a new external output directory")
    return cfg, records, paths, output


def _json_safe(value):
    # Undefined legacy metric NaN is represented explicitly by null in JSON.
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def check_operation_inputs(records, paths, cfg, output):
    for name, p in paths.items():
        if not p.is_file() or sha256(p) != records[name]["sha256"]:
            raise ValueError(f"External input identity mismatch: {name}")
    approval = read_json(paths["approval"])
    if approval.get("status") != "APPROVED" or approval.get("contract_sha256") != records["contract"]["sha256"]:
        raise ValueError("Approval must approve the exact external contract SHA256")
    _, digest = operation_manifest(cfg, paths, output)
    if approval.get("operation_manifest_sha256") != digest:
        raise ValueError("Approval does not bind this exact operation manifest SHA256")
    if cfg["operation"] in ("infer", "evaluate"):
        verify_scoring(cfg, paths)


def run_operation(path, operation, *, execute=False, allow_labels=False):
    cfg, records, paths, output = load_operation(path, operation)
    manifest, digest = operation_manifest(cfg, paths, output)
    if not execute:
        return {"executed": False, "operation": operation, "paths": {k: str(v) for k, v in paths.items()},
                "output": str(output) if output else None,
                "operation_manifest": manifest, "operation_manifest_sha256": digest,
                "scope": "schema/path preview only; no data, labels, checkpoint or training opened",
                "implementation": "external frozen training loop bridge" if operation == "train" else "shared package engine"}
    if operation == "evaluate" and not allow_labels:
        raise ValueError("Evaluation execution requires --allow-labels after all fixed scores are archived")
    if output is not None and output.exists():
        raise ValueError("Output must be a new directory; existing artifacts are immutable")
    check_operation_inputs(records, paths, cfg, output)
    parameters = cfg["parameters"]
    if operation == "train":
        from ..legacy import execute as execute_legacy, legacy_command
        run = resolve_path(parameters["run"], path)
        if (paths["contract"] != run / "CONTRACT.md" or paths["approval"] != run / "PRIMARY_APPROVAL.json"
                or output is not None):
            raise ValueError("Legacy training requires its own frozen run approval and no new output override")
        command = legacy_command("source-wave", paths["runner"],
                                 ["train", "--run", str(run), "--job", parameters["job"]])
        code = execute_legacy(command)
        return {"executed": True, "operation": operation, "returncode": code, "valid": code == 0,
                "implementation": "external frozen training loop bridge"}
    import numpy as np
    if operation == "infer":
        import torch
        from ..api import create_predictor
        from .bank import collect_forecasts, inference_metadata, write_bank
        query = read_json(paths["query"])
        if query["history"] != 400 or query["horizon"] != 5:
            raise ValueError("Shared bank engine retains the frozen 400/5 contract")
        torch.set_num_threads(2)
        torch.manual_seed(parameters["seed"])
        model = create_predictor(paths["official"], checkpoint=paths["checkpoint"],
                                 mode=parameters["mode"], device=parameters.get("device", "cpu"))
        data = np.load(paths["array"], mmap_mode="r", allow_pickle=False)
        arrays, rng_states, times = collect_forecasts(
            model, data, query["origins"], parameters["source_scale"], mask_seed=parameters["mask_seed"],
            expected_index=np.asarray(query["expected_time_index"]))
        report = inference_metadata(parameters["readout_job"],
                                    {"job": parameters["job"], "mode": parameters["mode"], "seed": parameters["seed"],
                                     "source_scale": parameters["source_scale"]},
                                    points=len(arrays["time_index"]), blocks=len(query["origins"]),
                                    mask_seed=parameters["mask_seed"],
                                    checkpoint_sha256=records["checkpoint"]["sha256"],
                                    query_array_sha256=records["array"]["sha256"],
                                    forward_p50_seconds=float(np.median(times)),
                                    forward_p95_seconds=float(np.quantile(times, .95)))
        report.update(score_space=cfg["score_space"], score_field="standardized_score",
                      source_calibration_sha256=records["calibration"]["sha256"], operation_manifest_sha256=digest)
        # Input identities must remain stable after collection and before writing.
        if any(sha256(p) != records[k]["sha256"] for k, p in paths.items()):
            raise ValueError("Input changed during inference")
        hashes = write_bank(output, arrays, rng_states, report)
    elif operation == "calibrate":
        from .source import calibrate_source
        if parameters.get("bank_role") != "source_cal":
            raise ValueError("Only explicitly declared source_cal banks may fit calibration")
        with np.load(paths["bank"], allow_pickle=False) as bank:
            arrays, report = calibrate_source(bank)
        report.update(calibration_schema_version=1, role="source_cal", score_space=cfg["score_space"],
                      source_bank_sha256=records["bank"]["sha256"], operation_manifest_sha256=digest)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "cal_scores.npz", **arrays)
        (output / "source_calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        hashes = {p.name: sha256(p) for p in output.iterdir()}
    else:
        from .evaluation import evaluate_fixed_scores
        # All score inputs were hashed above before opening labels. No selection.
        with np.load(paths["bank"], allow_pickle=False) as bank:
            index, score = bank["time_index"].copy(), bank["standardized_score"].copy()
            calibration = verify_scoring(cfg, paths)
            if "raw_score" not in bank or not np.array_equal(score, bank["raw_score"] / calibration["normal_error_scale"]):
                raise ValueError("Bank values do not obey the frozen raw_score/source_scale score-space")
        y = np.load(paths["labels"], mmap_mode="r", allow_pickle=False)[index]
        result = evaluate_fixed_scores(index, y, score, parameters["threshold"], read_json(paths["coverage"]))
        valid = result.pop("complete_event_valid")
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "evaluation_coordinates.npz", time_index=index, label=y,
                            complete_event_valid=valid)
        (output / "evaluation.json").write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
        hashes = {p.name: sha256(p) for p in output.iterdir()}
    return {"executed": True, "operation": operation, "output": str(output), "hashes": hashes,
            "scope": "shared operation; contract approval does not establish scientific validity"}
