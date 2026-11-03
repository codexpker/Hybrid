"""Fixed processed-MSL bank metadata adapter; accepted metric math is reused.

All five raw banks, source CALs and scalar axes precede the sole label opener.
No model import, fitting, inference, score ensemble or new metric definition.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from . import h5_evaluate_cli as serializer
from . import named_evaluate_cli as named
from .bank_recalibration import bank_metadata, bound, code_gate, sha
from .evaluate_cli import aggregate
from .public_readout import FULL_SOURCE, BASELINE_CAL, MSL_NPZ
from .scoring import recipe

OPERATION = "HYBRID_V1_PUBLIC_MSL_FIXED_BANK_EVALUATION"
PROFILE = "MSL_PROCESSED_COL0_SOURCE_Z_V1"
SEEDS = (2025, 2026, 2027)
MODELS = tuple("full_seed%d" % seed for seed in SEEDS)
CONTROLS = ("persistence", "history_mean512")
METHODS = MODELS + CONTROLS
LABEL_SHA = "32d4e5855b9a1a4906c43f6b3faf0d510114fc739944614790d271fbc55b88d9"
AFF_SHA = "640b05c1351acedca57b46952360fd5a645652e20e87c5cb34231e06aa6340a4"
STREAM = ["MSL_AUTHOR_PROCESSED_COL0", "processed_anonymous_stream", "processed_test_axis"]


def fixed_pairs():
    return [dict(reference=control, candidate=model, pairing="model_vs_deterministic")
            for model in MODELS for control in CONTROLS]


def contract(cfg):
    """Metadata only; never open or hash the query labels here."""
    if (cfg.get("operation") != OPERATION or cfg.get("profile") != PROFILE
            or cfg.get("statistics") != named.STATISTICS_NA
            or cfg.get("comparisons") != fixed_pairs()
            or cfg.get("stream") != STREAM
            or (cfg.get("history"), cfg.get("original_start"), cfg.get("original_length"),
                cfg.get("expected_records")) != (512, 0, 73729, 73217)
            or any(k in cfg for k in ("ensemble", "score_fusion", "selected_seed", "best_seed"))):
        raise ValueError("Fixed MSL axis, all three seeds/two controls/six pairs required")
    label = cfg["labels"]
    if (label.get("format") != "npz" or label.get("key") != "native_test_labels"
            or label.get("sha256") != LABEL_SHA or label.get("shape") != [73729]
            or cfg.get("affiliation_identity_sha256") != AFF_SHA
            or not cfg.get("affiliation_identity")):
        raise PermissionError("Original archived full labels and official Aff identity required")
    banks = cfg["banks"]
    if ([entry["id"] for entry in banks] != list(METHODS)
            or len({str(Path(e["directory"]).resolve()) for e in banks}) != 5):
        raise ValueError("Five distinct fixed banks; no duplicate bank as seeds")
    expected = [dict(zip(("domain", "entity", "run"), STREAM), start=512, end=73729)]
    for e in banks:
        identity = e["identity"]
        if (e.get("calibration_origin") != "source" or e.get("adaptation_mode") != "Z"
                or e.get("ranges") != expected or identity.get("input_sha256") != MSL_NPZ
                or any(not isinstance(identity.get(k), str) or not re.fullmatch(r"[0-9a-f]{64}", identity[k])
                       for k in ("checkpoint_sha256", "calibration_sha256", "config_sha256", "source_sha256"))):
            raise ValueError("Exact source-only bank identity and complete original axis required")
        if e["id"] in MODELS:
            seed = SEEDS[MODELS.index(e["id"])]
            if (e.get("kind") != "source_model" or type(e.get("training_seed")) is not int
                    or e["training_seed"] != seed or e.get("inference_seed") != seed
                    or e.get("control_id") != "FULL_SOURCE_seed%d" % seed
                    or (identity["checkpoint_sha256"], identity["calibration_sha256"]) != FULL_SOURCE[seed]):
                raise ValueError("Retain each original full source seed and own CAL")
        elif (e.get("kind") != "deterministic_baseline" or e.get("control_id") != e["id"]
              or e.get("training_seed") is not None or e.get("inference_seed") is not None
              or identity["calibration_sha256"] != BASELINE_CAL[e["id"]]):
            raise ValueError("Deterministic controls are not model training seeds")


def bank_gates(cfg):
    """Authenticate all five banks before any label-file byte or body access."""
    contract(cfg)
    paths = [cfg["raw_seal"]["path"], cfg["affiliation_identity"]]
    for e in cfg["banks"]:
        paths += [str(Path(e["directory"]) / "manifest.json"), e["calibration"]["path"]]
    for path in paths:
        named._not_label(path, cfg)
    seal = bound(cfg["raw_seal"])
    if seal != dict(status="ACCEPTED_PUBLIC_MSL_FIXED_BANKS", approved_by="root",
                    profile=PROFILE, banks=cfg["banks"]):
        raise PermissionError("Whole-five-bank root seal required before labels")
    reports = []
    for e in cfg["banks"]:
        # Inspect chunk paths before the reused authenticated reader opens them.
        manifest = bound(dict(path=str(Path(e["directory"]) / "manifest.json"), sha256=e["manifest_sha256"]))
        for chunk in manifest["chunks"]:
            named._not_label(Path(e["directory"]) / chunk["file"], cfg)
        manifest = bank_metadata(e)
        cal = bound(e["calibration"])
        active = ["r_tf", "r_ms", "r_pred"] if e["id"] in MODELS else ["r_pred"]
        if (e["calibration"]["sha256"] != e["identity"]["calibration_sha256"]
                or cal.get("active_components") != active or cal.get("weights") != recipe(active)[1]
                or cal.get("domains") != ["SMD", "Sunspot"]
                or cal.get("source_domain_equal_weight") is not True
                or set(cal.get("scales", {})) != set(active)
                or any(not np.isfinite(v) or v <= 0 for v in cal["scales"].values())
                or not np.isfinite(cal.get("threshold", np.nan))
                or manifest["contract"]["threshold"] != cal["threshold"]):
            raise ValueError("This bank must retain its own original source CAL recipe")
        reports.append(dict(id=e["id"], threshold=cal["threshold"], records=73217,
                            manifest_sha256=e["manifest_sha256"], contract=manifest["contract"]))
    if sha(cfg["affiliation_identity"]) != AFF_SHA:
        raise PermissionError("Official Aff identity changed")
    from ..engine.official_affiliation import load_official_affiliation
    load_official_affiliation(cfg["affiliation_identity"])
    return reports


def load_labels(spec):
    """Only label opener. Caller must first authenticate AND replay all banks."""
    if sha(spec["path"]) != LABEL_SHA:
        raise PermissionError("Original label cache changed; no offset repair")
    with np.load(spec["path"], allow_pickle=False) as cached:
        values = cached["native_test_labels"]
    if values.shape not in ((73729,), (73729, 1)):
        raise ValueError("Original complete label axis required")
    values = values.reshape(-1)
    if values.dtype.kind not in "biuf" or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Finite binary original labels required")
    return values.astype(np.int8)


def admit(config, release, trusted):
    cfg = json.loads(Path(config).read_text())
    contract(cfg)
    for path in (config, release):
        if path:
            named._not_label(path, cfg)
    if not release or not trusted:
        raise PermissionError("Exact root CPU label release required")
    signed = bound(dict(path=release, sha256=trusted))
    if (signed.get("status") != "APPROVED" or signed.get("approved_by") != "root"
            or signed.get("operation") != OPERATION or signed.get("allowed_actions") != [OPERATION]
            or signed.get("config_sha256") != sha(config) or signed.get("allow_labels") is not True
            or signed.get("raw_seal_sha256") != cfg["raw_seal"].get("sha256")
            or cfg.get("permissions") != dict(labels=True, evaluation=True, training=False, target_fit=False, GPU=False)):
        raise PermissionError("Exact CPU evaluation-only authority required")
    for name in cfg["source_sha256"]:
        named._not_label(Path(__file__).resolve().parents[2] / name, cfg)
    code_gate(cfg, signed, {"hybrid_tsad/hybrid_v1/" + n for n in
                           ("public_evaluate_cli.py", "public_readout.py", "h5_evaluate_cli.py", "bank_recalibration.py", "config.py")})
    if Path(cfg["output"]).exists():
        raise FileExistsError("Fresh public evaluation output required")
    return cfg


def run_evaluation(cfg):
    if Path(cfg["output"]).exists():
        raise FileExistsError("Fresh public evaluation output required")
    gates = bank_gates(cfg)
    arrays = named.scalar_inputs(cfg)
    if set(arrays) != set(METHODS) or [g["id"] for g in gates] != list(METHODS):
        raise ValueError("All five authenticated original scalar axes precede labels")
    labels = load_labels(cfg["labels"])
    # The existing generic array serializer invokes the accepted evaluator,
    # event pairing and Aff helper. Its historical filename is also archived.
    summary = serializer._save_results(cfg, gates, arrays, labels)
    # Reuse the full saved result schema for the unchanged aggregation helper;
    # the human SUMMARY view intentionally omits legacy event details.
    output = Path(cfg["output"])
    saved = {name: json.loads((output / name / "result.json").read_text()) for name in METHODS}
    results = {("model_seed%d" % seed): saved["full_seed%d" % seed] for seed in SEEDS}
    results.update({name: saved[name] for name in CONTROLS})
    seed_summary = aggregate(results)
    seed_summary["scope"] = "Three training seeds of one processed benchmark; not independent datasets or a score ensemble"
    (output / "MODEL_SEED_SUMMARY.json").write_text(json.dumps(seed_summary, indent=2, allow_nan=False))
    (output / "public_evaluate_cli.py").write_bytes(Path(__file__).read_bytes())
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--release")
    parser.add_argument("--trusted-release-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(preview=True, labels_or_banks_opened=False, operation=OPERATION)))
        return
    cfg = admit(args.config, args.release, args.trusted_release_sha256)
    if args.preflight_only:
        bank_gates(cfg)
        named.scalar_inputs(cfg)
        print(json.dumps(dict(PASS=True, labels_opened=False)))
        return
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
