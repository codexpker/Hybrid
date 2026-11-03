"""Fixed H5 six-cell CPU bank evaluation; no model, fit or inference.

The production profile has one original PSM axis and nine preregistered pairs.
Small array-stage unit fixtures never receive a production-profile exemption.
"""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
from . import named_evaluate_cli as named
from .bank_recalibration import bound, sha, bank_metadata, code_gate
from .config import HybridV1Config
from .scoring import recipe

OPERATION = "HYBRID_V1_H5_FIXED_BANK_EVALUATION"
PROFILE = "PSM_H5_SIX_SOURCE_SEED2025"
ACTION = "evaluate_h5_fixed_banks"
CELLS = ("base0_huber", "base0_half_mse", "anomaly10_huber",
         "anomaly10_half_mse", "normal_record10_huber", "normal_record10_half_mse")
ACTIVE = ["r_tf", "r_ms", "r_pred"]
AFF_SHA = "640b05c1351acedca57b46952360fd5a645652e20e87c5cb34231e06aa6340a4"
BASE_MANIFEST = "12dcb30ad1ba3bc2e38d9ca67e8f5bbb0fcd15154c221a03dc2958c2592eae3c"
BASE_CHECKPOINT = "c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812"
BASE_CAL = "da239eac4c5ab1e76718cd651c48f6f89b8d5658b5b9b2a20cb400d88f83d585"
META = ("training_config", "training_receipt", "calibration_config", "calibration_receipt", "calibration")
STATISTICS_NA = named.STATISTICS_NA
scalar_inputs = named.scalar_inputs
point_pair = named.point_pair
paired_events = named.paired_events
load_labels = named.load_labels


def fixed_pairs():
    pairs = []
    for loss in ("huber", "half_mse"):
        pairs.extend(dict(reference=a + "_" + loss, candidate=b + "_" + loss,
                          pairing="same_training_seed") for a, b in
                     (("base0", "anomaly10"), ("base0", "normal_record10"),
                      ("normal_record10", "anomaly10")))
    pairs.extend(dict(reference=p + "_half_mse", candidate=p + "_huber",
                      pairing="same_training_seed") for p in ("base0", "anomaly10", "normal_record10"))
    return pairs


def contract(cfg):
    """Metadata-only production contract. No synthetic-profile or axis bypass."""
    if (cfg.get("profile") != PROFILE or cfg.get("statistics") != STATISTICS_NA
            or any(k in cfg for k in ("model_seeds", "seed_groups", "model_mean_SD", "ensemble", "aggregate"))):
        raise ValueError("Fixed H5 cells are methods, not seeds/ensemble")
    if (cfg.get("history"), cfg.get("original_length"), cfg.get("original_start"), cfg.get("expected_records")) != (512, 87841, 0, 87329):
        raise ValueError("Original H5 PSM clock required")
    if cfg.get("stream") != ["PSM", "anonymous_PSM", "test_original"] or cfg.get("comparisons") != fixed_pairs():
        raise ValueError("Keep the original stream and all nine directed pairs")
    label = cfg["labels"]
    if (label.get("sha256") != named.LABEL_SHA or label.get("shape") != [87841]
            or label.get("format") != "csv" or label.get("drop_columns") != ["timestamp_(min)"]
            or label.get("value_columns") != ["label"] or cfg.get("affiliation_identity_sha256") != AFF_SHA
            or not cfg.get("affiliation_identity")):
        raise PermissionError("Exact original label schema and official Aff identity required")
    if HybridV1Config(**cfg["model"]) != HybridV1Config():
        raise ValueError("H5 retains the complete default V1 architecture")
    entries = cfg["banks"]
    if ([e["id"] for e in entries] != list(CELLS)
            or len({str(Path(e["directory"]).resolve()) for e in entries}) != 6):
        raise ValueError("All six distinct banks in frozen cell order required")
    expected = [dict(zip(("domain", "entity", "run"), cfg["stream"]), start=512, end=87841)]
    for e in entries:
        base = e["id"] == "base0_huber"
        loss = "huber" if e["id"].endswith("_huber") else "half_mse"
        if (e.get("kind") != "source_model" or e.get("calibration_origin") != "source"
                or e.get("adaptation_mode") != "Z" or e.get("training_kind") != loss
                or type(e.get("training_seed")) is not int or e["training_seed"] != 2025
                or type(e.get("inference_seed")) is not int or e["inference_seed"] != 2025
                or e.get("active_components") != ACTIVE or e.get("reused_raw") is not base
                or e.get("control_id") != ("full" if base else "H5_" + e["id"])
                or e["ranges"] != expected or e["identity"].get("input_sha256") != named.INPUT_SHA):
            raise ValueError("Cell source2025 identity/recipe/clock/kind differs")
        if any(not isinstance(e["identity"].get(k), str) or not re.fullmatch(r"[0-9a-f]{64}", e["identity"][k])
               for k in ("input_sha256", "checkpoint_sha256", "calibration_sha256", "config_sha256", "source_sha256")):
            raise ValueError("All five real bank identity SHA256 values required")
        if base and (e.get("manifest_sha256") != BASE_MANIFEST
                     or e["identity"].get("checkpoint_sha256") != BASE_CHECKPOINT
                     or e["identity"].get("calibration_sha256") != BASE_CAL):
            raise ValueError("Huber0 must reuse the original accepted full bank unchanged")


def _source_metadata(e, cfg, manifest):
    values = {k: bound(e[k]) for k in META}
    train, trained, calcfg, calibrated, cal = (values[k] for k in META)
    for actual in (train, calcfg):
        if (actual.get("operation") != "HYBRID_V1_SOURCE_STAGE1" or actual.get("seed") != 2025
                or actual.get("history") != 512 or actual.get("horizon") != 32
                or actual.get("kind") != e["training_kind"] or actual.get("model") != cfg["model"]
                or actual.get("sources") != cfg["sources"] or list(recipe(actual.get("active_components"))[0]) != ACTIVE):
            raise ValueError("Real source train/CAL config differs from this full H5 cell")
    ckpt = e["identity"]["checkpoint_sha256"]
    if (trained.get("phase") != "source_train" or trained.get("synthetic_non_scientific_results") is not False
            or trained.get("files", {}).get("source.pt") != ckpt
            or trained.get("files", {}).get("config.json") != e["training_config"]["sha256"]
            or calibrated.get("phase") != "calibrate_source" or calibrated.get("synthetic_non_scientific_results") is not False
            or calibrated.get("files", {}).get("source_calibration.json") != e["calibration"]["sha256"]
            or calibrated.get("files", {}).get("config.json") != e["calibration_config"]["sha256"]
            or calcfg.get("checkpoint_sha256") != ckpt
            or calcfg.get("train_receipt_sha256") != e["training_receipt"]["sha256"]):
        raise ValueError("Own accepted source checkpoint/config/CAL receipt lineage required")
    if (e["identity"]["calibration_sha256"] != e["calibration"]["sha256"]
            or cal.get("source_domain_equal_weight") is not True or cal.get("domains") != ["SMD", "Sunspot"]
            or cal.get("active_components") != ACTIVE or cal.get("weights") != recipe(ACTIVE)[1]
            or set(cal.get("scales", {})) != set(ACTIVE)
            or any(not np.isfinite(v) or v <= 0 for v in cal["scales"].values())
            or not np.isfinite(cal.get("threshold", np.nan)) or manifest["contract"]["threshold"] != cal["threshold"]):
        raise ValueError("Keep this cell's own fixed source CAL recipe/scales/threshold")
    return cal


def _authenticated_banks(cfg):
    """I/O stage after contract; independently unit-testable with tiny banks."""
    paths = [cfg["raw_seal"]["path"], cfg.get("affiliation_identity")]
    for e in cfg["banks"]:
        paths += [str(Path(e["directory"]) / "manifest.json")] + [e[k]["path"] for k in META]
    for path in paths:
        if path is not None:
            named._not_label(path, cfg)
    seal = bound(cfg["raw_seal"])
    if seal != dict(status="ACCEPTED_H5_FIXED_BANKS", approved_by="root", profile=cfg["profile"], banks=cfg["banks"]):
        raise PermissionError("Whole six-bank root seal required before scalar or label I/O")
    reports = []
    for e in cfg["banks"]:
        mp = Path(e["directory"]) / "manifest.json"
        m = bound(dict(path=str(mp), sha256=e["manifest_sha256"]))
        for chunk in m["chunks"]:
            named._not_label(Path(e["directory"]) / chunk["file"], cfg)
        m = bank_metadata(e)  # Complete/count/identity/ranges and every chunk hash.
        cal = _source_metadata(e, cfg, m)
        reports.append(dict(id=e["id"], threshold=cal["threshold"], manifest_sha256=e["manifest_sha256"],
                            records=m["committed_records"], contract=m["contract"]))
    if cfg.get("affiliation_identity"):
        if sha(cfg["affiliation_identity"]) != cfg["affiliation_identity_sha256"]:
            raise PermissionError("Official Aff metadata changed")
        from ..engine.official_affiliation import load_official_affiliation
        load_official_affiliation(cfg["affiliation_identity"])
    return reports


def bank_gates(cfg):
    contract(cfg)
    return _authenticated_banks(cfg)


def admit(config, release, trusted):
    cfg = json.loads(Path(config).read_text())
    if not release or not trusted:
        raise PermissionError("Exact root H5 CPU release required")
    named._not_label(release, cfg)
    signed = bound(dict(path=release, sha256=trusted))
    if (signed.get("status") != "APPROVED" or signed.get("approved_by") != "root"
            or signed.get("operation") != OPERATION or cfg.get("operation") != OPERATION
            or signed.get("allowed_actions") != [ACTION] or signed.get("allow_labels") is not True
            or signed.get("config_sha256") != sha(config) or not cfg["raw_seal"].get("sha256")
            or signed.get("raw_seal_sha256") != cfg["raw_seal"]["sha256"]
            or cfg.get("permissions") != dict(labels=True, evaluation=True, training=False, target_fit=False, GPU=False)):
        raise PermissionError("Changed/pending H5-only evaluation release")
    contract(cfg)
    for name in cfg["source_sha256"]:
        named._not_label(Path(__file__).resolve().parents[2] / name, cfg)
    code_gate(cfg, signed, {"hybrid_tsad/hybrid_v1/" + n for n in
                          ("h5_evaluate_cli.py", "bank_recalibration.py", "config.py")})
    if Path(cfg["output"]).exists():
        raise FileExistsError("Fresh H5 evaluation output required")
    return cfg


def _save_results(cfg, gates, arrays, labels):
    """Accepted core math on prevalidated arrays; no label-file loader here."""
    from .evaluation import evaluate_arrays, save_evaluation
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for entry, gate in zip(cfg["banks"], gates):
        values = arrays[entry["id"]]
        result = evaluate_arrays(values["index"], values["score"], labels, gate["threshold"], bank_complete=True,
                                 original_length=cfg["original_length"], original_start=cfg["original_start"],
                                 warmup_points=cfg["history"], affiliation_identity=cfg.get("affiliation_identity"),
                                 **{k: values[k] for k in ("valid", "available_index", "invalid_reasons")})
        result["bank_identity"] = dict(manifest_sha256=entry["manifest_sha256"], contract=gate["contract"],
                                       stream=cfg["stream"], complete=True, global_bank_records=gate["records"])
        save_evaluation(output / entry["id"], result, full_labels=labels,
                        provenance=dict(named_descriptor=entry, raw_seal=cfg["raw_seal"], gates=gates,
                                        source_sha256=cfg.get("source_sha256"), labels=cfg["labels"], no_score_ensemble=True), **values)
        results[entry["id"]] = result
    pairs = []
    for pair in cfg["comparisons"]:
        left, right = pair["reference"], pair["candidate"]
        folder = output / "pairs" / (left + "__vs__" + right)
        folder.mkdir(parents=True)
        point, common = point_pair(arrays[left], arrays[right], labels, cfg["original_start"])
        events = paired_events(results[left], results[right])
        with (folder / "paired_events.csv").open("x", newline="", encoding="utf-8") as f:
            keys = list(dict.fromkeys(k for row in events for k in row))
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader(); writer.writerows(events)
        np.savez_compressed(folder / "common_point_inputs.npz", index=common,
                            reference_score=arrays[left]["score"][np.searchsorted(arrays[left]["index"], common)],
                            candidate_score=arrays[right]["score"][np.searchsorted(arrays[right]["index"], common)], full_labels=labels)
        row = dict(pair, common_point_effect=point, statistics=dict(STATISTICS_NA),
                   event_pairing="Original greedy statuses; no rematching or gap-concatenated event/Aff CI")
        (folder / "summary.json").write_text(json.dumps(row, indent=2, allow_nan=False)); pairs.append(row)
    summary = dict(status="COMPLETED_PENDING_INDEPENDENT_REVIEW", profile=cfg["profile"],
                   methods=[dict(descriptor=e, ranking=results[e["id"]]["ranking"], coverage=results[e["id"]]["coverage"],
                                 streams={s: dict(point=v["point"], normal=v["normal"], event_metrics=v["main_events"]["metrics"],
                                                  affiliation=v["affiliation"], Range_VUS=v["Range_VUS"])
                                          for s, v in results[e["id"]]["streams"].items()}) for e in cfg["banks"]],
                   comparisons=pairs, statistics=dict(STATISTICS_NA), model_seed_mean=None, no_score_ensemble=True,
                   Range_VUS="Accepted core NA; no substitute formula", claim_limits=cfg.get("claim_limits", []))
    for name, value in (("SUMMARY.json", summary), ("BANK_GATES.json", gates), ("CONFIG_AND_PROVENANCE.json", cfg)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False))
    (output / "h5_evaluate_cli.py").write_bytes(Path(__file__).read_bytes())
    return summary


def run_evaluation(cfg):
    if Path(cfg["output"]).exists():
        raise FileExistsError("Fresh H5 output required")
    gates = bank_gates(cfg)
    arrays = scalar_inputs(cfg)  # ALL six scalar replays precede any label body.
    if [g["id"] for g in gates] != list(CELLS) or set(arrays) != set(CELLS):
        raise ValueError("All six authenticated scalar inputs required before labels")
    labels = load_labels(cfg["labels"])
    return _save_results(cfg, gates, arrays, labels)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--release"); p.add_argument("--trusted-release-sha256")
    p.add_argument("--execute", action="store_true"); p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(preview=True, labels_or_banks_opened=False, operation=OPERATION))); return
    cfg = admit(a.config, a.release, a.trusted_release_sha256)
    if a.preflight_only:
        bank_gates(cfg); scalar_inputs(cfg)
        print(json.dumps(dict(PASS=True, labels_opened=False))); return
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
