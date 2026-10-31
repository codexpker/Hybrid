"""Canonical operation approval and source-calibration/score-space identity."""
import hashlib
import json
import math
import re

from ..config import read_json

SCORE_SPACE = "history_normalized_abs_channelmean_div_source_median_v1"


def operation_manifest(cfg, paths, output):
    """Approval record and its digest are excluded to avoid a self-reference."""
    parameters = dict(cfg["parameters"])
    manifest = {"manifest_schema_version": 1, "engine_schema_version": cfg["engine_schema_version"],
                "operation": cfg["operation"], "score_space": cfg["score_space"],
                "contract": {"path": str(paths["contract"]), "sha256": cfg["contract"]["sha256"]},
                "inputs": {name: {"path": str(paths[name]), "sha256": record["sha256"]}
                           for name, record in cfg["inputs"].items()},
                "parameters": parameters, "output": str(output) if output else None}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return manifest, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_calibration(path):
    value = read_json(path)
    required = {"calibration_schema_version", "role", "score_space", "source_bank_sha256",
                "normal_error_scale", "threshold99", "operation_manifest_sha256"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("Source calibration artifact lacks its identity fields")
    if (type(value["calibration_schema_version"]) is not int or value["calibration_schema_version"] != 1
            or value["role"] != "source_cal" or value["score_space"] != SCORE_SPACE):
        raise ValueError("Unsupported calibration role/version/score-space; no implicit legacy mapping")
    for key in ("source_bank_sha256", "operation_manifest_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError("Calibration source/operation SHA is malformed")
    for key in ("normal_error_scale", "threshold99"):
        if (type(value[key]) not in (float, int) or not math.isfinite(value[key])
                or value[key] < 0 or key == "normal_error_scale" and value[key] == 0):
            raise ValueError("Calibration requires finite scale>0 and threshold>=0")
    return value


def verify_scoring(cfg, paths):
    """Run before inference, loading a score bank, or opening evaluator labels."""
    calibration = source_calibration(paths["calibration"])
    key = "source_scale" if cfg["operation"] == "infer" else "threshold"
    if type(cfg["parameters"][key]) not in (float, int) or not math.isfinite(cfg["parameters"][key]):
        raise ValueError("Explicit score scale/threshold must be finite numeric values")
    if cfg["score_space"] != calibration["score_space"]:
        raise ValueError("Configured score-space differs from source calibration")
    if cfg["operation"] == "infer":
        if cfg["parameters"]["source_scale"] != calibration["normal_error_scale"]:
            raise ValueError("Configured source_scale differs from fixed source calibration")
    else:
        if cfg["parameters"]["threshold"] != calibration["threshold99"]:
            raise ValueError("Configured threshold differs from fixed source calibration")
        metadata = read_json(paths["bank_metadata"])
        if (metadata.get("bank_sha256") != cfg["inputs"]["bank"]["sha256"]
                or metadata.get("source_calibration_sha256") != cfg["inputs"]["calibration"]["sha256"]
                or metadata.get("score_space") != cfg["score_space"]
                or metadata.get("score_field") != "standardized_score"
                or type(metadata.get("source_scale")) not in (float, int)
                or metadata.get("source_scale") != calibration["normal_error_scale"]):
            raise ValueError("Bank score units/calibration identity do not match the approved score-space")
    return calibration
