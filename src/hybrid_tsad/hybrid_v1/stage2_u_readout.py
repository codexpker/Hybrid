"""Thin SMD matched-U readout. No new model, score, calibration or metrics.

Z source/target CAL views share one model bank; the target view replays saved
raw residuals only. This module never authorizes execution by its existence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .stage2_u_matched import sha, no_labels, role_metadata, load_roles, OPERATION as TRAIN_OPERATION

OPERATION = "HYBRID_V1_MATCHED_U_SMD_READOUT"


def bound(asset):
    if set(asset) != {"path", "sha256"} or sha(asset["path"]) != asset["sha256"]:
        raise PermissionError("Changed readout asset")
    return Path(asset["path"])


def metadata(cfg):
    """JSON/byte identity only, no Torch/NPZ body/model loading."""
    no_labels(cfg)
    if (cfg.get("operation") != OPERATION or cfg.get("mode") not in {"ZS_ZT", "U"}
            or cfg.get("seed") not in {2025, 2026, 2027}
            or cfg.get("history") != 512 or cfg.get("horizon") != 32
            or cfg.get("chunk_size") != 64 or cfg.get("resume") is not False
            or cfg.get("condition") not in ({"COMMON"} if cfg["mode"] == "ZS_ZT" else {"A", "B", "C"})):
        raise PermissionError("Fixed mode/seed/condition/H/P/no-resume recipe required")
    if Path(cfg["output"]).exists():
        raise PermissionError("Fresh readout output required")
    for path, h in cfg["source_sha256"].items():
        if sha(path) != h:
            raise PermissionError("Changed explicit readout/runtime closure")
    if cfg["source_sha256"].get(str(Path(__file__).resolve())) != sha(__file__):
        raise PermissionError("Actual reader must be explicitly pinned")
    assets = cfg["assets"]
    expected = {"checkpoint", "parent_source_checkpoint", "model_config", "training_config", "training_receipt",
                "target_calibration", "target_calibration_config", "target_calibration_receipt", "manifest", "plan"}
    if cfg["mode"] == "ZS_ZT":
        expected.add("source_calibration")
    if set(assets) != expected:
        raise PermissionError("Exact own-model/CAL/parent/plan assets required")
    for a in assets.values():
        bound(a)
    seal = json.loads(bound(cfg["asset_seal"]).read_text())
    if (seal.get("approved_by") != "root" or seal.get("status") != "ACCEPTED_MATCHED_U_READOUT_ASSETS"
            or seal.get("mode") != cfg["mode"] or seal.get("condition") != cfg["condition"]
            or seal.get("seed") != cfg["seed"] or seal.get("assets") != assets):
        raise PermissionError("Accepted exact typed asset chain required")
    get = lambda k: json.loads(Path(assets[k]["path"]).read_text())
    model, train, trained = [get(k) for k in ("model_config", "training_config", "training_receipt")]
    cal, calcfg, calreceipt = [get(k) for k in ("target_calibration", "target_calibration_config", "target_calibration_receipt")]
    parent, checkpoint = assets["parent_source_checkpoint"]["sha256"], assets["checkpoint"]["sha256"]
    from .scoring import recipe
    active, weights = recipe(cfg["active_components"])
    if list(active) != ["r_tf", "r_ms", "r_pred"]:
        raise ValueError("Original fixed full score required")
    if any(model.get(k) != cfg[k] for k in ("seed", "history", "horizon", "active_components")):
        raise ValueError("Source model/recipe/seed mismatch")
    if (cal.get("source_domain_equal_weight") is not False or cal.get("weights") != weights
            or cal.get("active_components") != list(active) or set(cal.get("scales", {})) != set(active)
            or not all(np.isfinite(v) and v > 0 for v in cal["scales"].values())
            or not np.isfinite(cal.get("threshold", np.nan))):
        raise ValueError("Own target CAL fixed scalar recipe invalid")
    if (calcfg.get("operation") != TRAIN_OPERATION or calcfg.get("action") != "calibrate"
            or calcfg.get("checkpoint_sha256") != checkpoint
            or calcfg.get("parent_source_checkpoint_sha256") != parent
            or calcfg.get("plan_sha256") != assets["plan"]["sha256"]
            or calcfg.get("manifest_sha256") != assets["manifest"]["sha256"]
            or calcfg.get("seed") != cfg["seed"]
            or calreceipt.get("config_sha256") != assets["target_calibration_config"]["sha256"]
            or calreceipt.get("checkpoint_sha256") != checkpoint
            or calreceipt.get("scientific_training_status") != "ZERO_UPDATE_CAL"
            or calreceipt.get("updates") != 0 or not calreceipt.get("status", "").startswith("COMPLETED")):
        raise PermissionError("Target CAL must belong to the exact checkpoint and role plan")
    if cfg["mode"] == "U":
        if (train.get("operation") != TRAIN_OPERATION or train.get("action") != "train_endpoint"
                or train.get("method") != "plain" or train.get("attempts") != 200
                or train.get("checkpoint_sha256") != parent or train.get("seed") != cfg["seed"]
                or train.get("plan_sha256") != assets["plan"]["sha256"]
                or train.get("manifest_sha256") != assets["manifest"]["sha256"]
                or trained.get("config_sha256") != assets["training_config"]["sha256"]
                or trained.get("checkpoint_sha256") != checkpoint
                or trained.get("parent_source_checkpoint_sha256") != parent
                or trained.get("scientific_training_status") != "VALID_ALL_ATTEMPTS_UPDATED"
                or trained.get("updates") != 200 or not trained.get("status", "").startswith("COMPLETED")):
            raise PermissionError("U200 endpoint/parent/own plan/valid exposure chain mismatch")
    else:
        if (checkpoint != parent or train.get("operation") != "HYBRID_V1_SOURCE_STAGE1"
                or trained.get("phase") != "source_train" or trained.get("files", {}).get("source.pt") != parent
                or trained.get("files", {}).get("config.json") != assets["training_config"]["sha256"]):
            raise PermissionError("Z-S/Z-T must share accepted unchanged source weights")
        sc = get("source_calibration")
        if (sc.get("source_domain_equal_weight") is not True or sc.get("weights") != weights
                or sc.get("active_components") != list(active) or set(sc.get("scales", {})) != set(active)
                or not all(np.isfinite(v) and v > 0 for v in sc["scales"].values())
                or not np.isfinite(sc.get("threshold", np.nan))):
            raise ValueError("Original source CAL recipe invalid")
    manifest = role_metadata(assets["manifest"]["path"], 512, 32)
    query = [r for r in manifest["runs"] if r["role"] == "QUERY"]
    if (len(query) != 1 or query[0]["domain"] != "SMD" or query[0]["entity"] != "machine-3-1"
            or query[0]["run"] != "machine-3-1_test"):
        raise PermissionError("Only the predeclared SMD3-1 original query record is supported")
    ranges = [dict(domain=r["domain"], entity=r["entity"], run=r["run"],
                   start=r["original_start"]+512, end=r["original_start"]+r["expected_shape"][0]) for r in query]
    if cfg["ranges"] != ranges or cfg["scored_points"] != sum(r["end"]-r["start"] for r in ranges):
        raise ValueError("Original arrived axis/complete coverage changed")
    return dict(status="METADATA_READY_NOT_EXECUTED", ranges=ranges, arrays_opened=False,
                Torch_loaded=False, model_forward=False)


def rescore_view(producer, destination, calibration, identity):
    """Offline same-raw score view; no model or original input is opened."""
    from .chunked_bank import ChunkedBank, read_records
    from .cli import score_records
    producer = Path(producer)
    if Path(destination).exists():
        raise PermissionError("Fresh score-view destination required")
    before = sha(producer / "manifest.json")
    m = json.loads((producer / "manifest.json").read_text())
    if not m["complete"]:
        raise PermissionError("A complete producer is required")
    with ChunkedBank(destination, identity=identity, ranges=m["contract"]["ranges"],
                     threshold=calibration["threshold"], chunk_size=64) as bank:
        for raw in read_records(producer):
            bank.append(score_records([raw], calibration)[0])
        bank.finalize()
    if sha(producer / "manifest.json") != before:
        raise ValueError("Producer changed during offline view")
    return dict(producer_manifest_sha256=before, destination_manifest_sha256=sha(Path(destination)/"manifest.json"),
                model_forward=False, calibration_fit=False)


def admit(config, release, trusted):
    cfg = json.loads(Path(config).read_text())
    rel = json.loads(Path(release).read_text())
    if (sha(release) != trusted or rel.get("status") != "APPROVED" or rel.get("approved_by") != "root"
            or rel.get("operation") != OPERATION or rel.get("allowed_actions") != ["model_readout"]
            or rel.get("config_sha256") != sha(config) or rel.get("source_sha256") != cfg["source_sha256"]):
        raise PermissionError("Exact fresh root readout release required")
    return cfg, metadata(cfg)


def run(cfg):
    from .cli import load_model, save_json, save_runtime
    from .readout import write_model_bank, _state, _rng, _rng_json
    metadata(cfg)  # mandatory same-call gate before loading arrays or constructing a model
    a = cfg["assets"]; out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=False)
    runs = load_roles(a["manifest"]["path"], {"QUERY"}, 512, 32)
    model = load_model(json.loads(Path(a["model_config"]["path"]).read_text()), a["checkpoint"]["path"])
    state = _state(model); save_runtime(out, model)
    save_json(out/"MODEL_STATE_BEFORE.json", state); save_json(out/"INITIAL_RNG.json", _rng_json(_rng()))
    calkey = "source_calibration" if cfg["mode"] == "ZS_ZT" else "target_calibration"
    calibration = json.loads(Path(a[calkey]["path"]).read_text())
    identity = dict(input_sha256=a["manifest"]["sha256"], config_sha256=cfg["execution_config_sha256"],
        checkpoint_sha256=a["checkpoint"]["sha256"], calibration_sha256=a[calkey]["sha256"],
        source_sha256=hashlib.sha256(json.dumps(cfg["source_sha256"],sort_keys=True).encode()).hexdigest())
    primary = out/"canonical_bank"
    # Exactly one call to the original writer: Z-S and Z-T cannot repeat inference.
    result = write_model_bank(model, runs, calibration, primary, identity=identity,
        history=512, horizon=32, seed=cfg["seed"], chunk_size=64, resume=False,
        active_components=cfg["active_components"])
    if _state(model) != state:
        raise ValueError("Readout changed model state")
    save_json(out/"MODEL_STATE_AFTER.json", _state(model)); save_json(out/"FINAL_RNG.json", _rng_json(_rng()))
    save_json(out/"point_times.json", result["timings"])
    views = {"U" if cfg["mode"] == "U" else "Z-S": str(primary)}
    if cfg["mode"] == "ZS_ZT":
        target = json.loads(Path(a["target_calibration"]["path"]).read_text())
        targetid = dict(identity, calibration_sha256=a["target_calibration"]["sha256"])
        overlay = rescore_view(primary, out/"Z-T_bank", target, targetid)
        save_json(out/"Z-T_VIEW_RECEIPT.json", overlay); views["Z-T"] = str(out/"Z-T_bank")
    save_json(out/"config.json", cfg)
    save_json(out/"receipt.json", dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT", mode=cfg["mode"],
        seed=cfg["seed"], condition=cfg["condition"], model_forward_bank_count=1, views=views,
        points=cfg["scored_points"], metrics_or_original_labels=False, assets=a))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "release", "trusted-release-sha256"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--execute", action="store_true"); a=p.parse_args()
    c, m = admit(a.config,a.release,a.trusted_release_sha256)
    if not a.execute:
        print(json.dumps(m)); return
    c["execution_config_sha256"] = sha(a.config); run(c)


if __name__ == "__main__":
    main()
