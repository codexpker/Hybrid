"""Synthetic operation approval fixtures; no real research permissions."""
import json

from hybrid_tsad.config import sha256
from hybrid_tsad.engine.identity import SCORE_SPACE, operation_manifest
from hybrid_tsad.engine.pipeline import load_operation


def calibration_file(path, scale=1.0, threshold=.5):
    path.write_text(json.dumps(dict(calibration_schema_version=1, role="source_cal", score_space=SCORE_SPACE,
                                    source_bank_sha256="a" * 64, normal_error_scale=scale, threshold99=threshold,
                                    operation_manifest_sha256="b" * 64)))
    return path


def record(path):
    return {"path": str(path), "sha256": sha256(path)}


def approve_fixture(cfg, path):
    cfg["engine_schema_version"] = 3 if cfg.get("engine_schema_version") == 3 else 2
    cfg["score_space"] = SCORE_SPACE
    path.write_text(json.dumps(cfg))
    parsed, _, paths, output = load_operation(path, cfg["operation"])
    _, digest = operation_manifest(parsed, paths, output)
    paths["approval"].write_text(json.dumps({"status": "APPROVED", "contract_sha256": cfg["contract"]["sha256"],
                                           "operation_manifest_sha256": digest}))
    cfg["approval"] = record(paths["approval"])
    path.write_text(json.dumps(cfg))
    return cfg
