"""Fixed SMD3-1 matched-U CPU evaluation; no models, fitting or new metric math.

Five primary score views represent FOUR neural banks. Three auxiliary views
only reuse U residuals with both frozen Z-T component scales AND its threshold.
Every raw gate precedes the sole native-label opener. Default CLI is preview.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from . import scoring, chunked_bank
from .bank_recalibration import bank_metadata
from .controlled_wave_evaluation import scalar_record, require_metric_dependencies, reject_dependency_na
from .named_evaluate_cli import point_pair, paired_events
from ..engine.events import binary_events

OPERATION = "HYBRID_V1_MATCHED_U_SMD31_FIXED_EVALUATION"
METHODS = ("Z_S", "Z_T", "U_A", "U_B", "U_C")
PAIRS = (("Z_S", "Z_T"), ("Z_T", "U_A"), ("Z_T", "U_B"),
         ("Z_T", "U_C"), ("U_A", "U_B"), ("U_A", "U_C"), ("U_C", "U_B"))
STREAM = ("SMD", "machine-3-1", "machine-3-1_test")
RANGES = [dict(zip(("domain", "entity", "run"), STREAM), start=10622, end=28700)]
RESOURCES = dict(wall_seconds=1795, cleanup_seconds=5, threads=1, CPU_AS_GiB=8, GPUh=0)
PERMISSIONS = dict(labels=True, evaluation=True, training=False, model=False, GPU=False,
                   calibration_fit=False, threshold_selection=False, cached_ZT_sensitivity=True)
LABEL_SHA = "549348a9c63ed922e85157885bcb21289a8c9f5af94b50147a21cb7b928220ec"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, ensure_ascii=False)
        handle.write("\n")


def bound_json(asset):
    if set(asset) != {"path", "sha256"} or sha(asset["path"]) != asset["sha256"]:
        raise PermissionError("Changed exact root-bound JSON asset")
    return json.loads(Path(asset["path"]).read_text(encoding="utf-8"))


def contract(cfg):
    if (cfg.get("operation") != OPERATION or cfg.get("status") != "ROOT_BOUND"
            or cfg.get("seed") != 2025 or cfg.get("original_length") != 28700
            or cfg.get("original_start") != 0 or cfg.get("warmup_points") != 10622
            or cfg.get("expected_records") != 18078 or cfg.get("ranges") != RANGES
            or cfg.get("comparisons") != [dict(reference=a, candidate=b) for a, b in PAIRS]
            or cfg.get("auxiliary_views") != [m+"_ZT_FIXED" for m in METHODS[2:]]
            or cfg.get("resources") != RESOURCES or cfg.get("permissions") != PERMISSIONS
            or cfg.get("statistics") != {"status": "INCOMPLETE", "CI": None,
                "reason": "ONE_SEED_ONE_DONOR_EVENT_TWO_QUERY_EVENTS_EXPLORATORY_ONLY"}):
        raise PermissionError("Fixed seed/axis/pairs/auxiliary/CPU/exploratory protocol required")
    if cfg.get("truth", {}).get("sha256") != LABEL_SHA:
        raise PermissionError("Original full native label identity required")
    if [b.get("id") for b in cfg.get("banks", [])] != list(METHODS):
        raise PermissionError("Exact ordered five score views required")
    if len({str(Path(b["directory"]).resolve()) for b in cfg["banks"]}) != 5:
        raise PermissionError("Five distinct scalar view banks, not duplicate methods")
    by = {b["id"]: b for b in cfg["banks"]}
    z = by["Z_S"]["identity"]["checkpoint_sha256"]
    for name, desc in by.items():
        if (desc.get("ranges") != RANGES or desc.get("seed") != 2025
                or desc.get("active_components") != ["r_tf", "r_ms", "r_pred"]
                or desc.get("calibration_origin") != ("source" if name == "Z_S" else "target")
                or desc.get("parent_source_checkpoint_sha256") != z):
            raise PermissionError("Common source/seed/input footprint and own CAL required")
        if name == "Z_T" and desc["identity"]["checkpoint_sha256"] != z:
            raise PermissionError("Z-S/Z-T must use identical source weights")
    return by


def reject_truth_aliases(cfg):
    """Metadata-only old evaluator rule: no prelabel asset aliases truth."""
    truth = str(Path(cfg["truth"]["path"]).absolute()).casefold()
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if (key in {"path", "directory", "affiliation_identity"} and isinstance(item, str)
                        and str(Path(item).absolute()).casefold() == truth):
                    raise PermissionError("A prelabel asset aliases the truth path")
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk({key: value for key, value in cfg.items() if key != "truth"})


def source_closure():
    from .controlled_wave_evaluation import required_closure
    base = Path(__file__).resolve().parents[1]
    extra = ("hybrid_v1/bank_recalibration.py", "hybrid_v1/stage2_u_data.py", "hybrid_v1/stage2_u_evaluation.py")
    return dict(required_closure(), **{str((base/name).resolve()): sha(base/name) for name in extra})


def records_to_arrays(records, calibration, *, start, end, stream, auxiliary_calibration=None):
    rows, aux, residuals = [], [], []
    for position, record in enumerate(records):
        if (position >= end-start or type(record.get("index")) is not int
                or type(record.get("available_index")) is not int
                or record["index"] != start+position or record["available_index"] != record["index"]
                or tuple(record.get(k) for k in ("domain", "entity", "run")) != tuple(stream)):
            raise ValueError("Complete ordered original arrived axis required, with tail intact")
        rows.append(scalar_record(record, calibration))
        residuals.append([record.get(k, np.nan) for k in scoring.COMPONENTS])
        if auxiliary_calibration is not None:
            derived = scoring.apply_calibration(record, auxiliary_calibration)
            # Original scorer defines valid/NA; no zero fill and no own-tau reuse.
            aux.append((record["index"], derived["score"], derived["valid"],
                        record["available_index"], derived.get("reason")))
    if len(rows) != end-start:
        raise ValueError("Missing raw point or unpreserved last31 tail")
    def arrays(values):
        return dict(index=np.asarray([r[0] for r in values], np.int64),
                    score=np.asarray([r[1] for r in values], np.float64),
                    valid=np.asarray([r[2] for r in values], bool),
                    available_index=np.asarray([r[3] for r in values], np.int64),
                    invalid_reasons=[r[4] for r in values])
    return arrays(rows), (arrays(aux) if auxiliary_calibration is not None else None), np.asarray(residuals, np.float64)


def prelabel(cfg):
    by = contract(cfg)
    reject_truth_aliases(cfg)
    seal = bound_json(cfg["whole_raw_seal"])
    if (seal.get("status") != "ACCEPTED_MATCHED_U_FIVE_SCORE_VIEWS" or seal.get("approved_by") != "root"
            or seal.get("banks") != cfg["banks"] or seal.get("raw_bank_count") != 4
            or seal.get("score_view_count") != 5):
        raise PermissionError("All four raw banks and five views require one root seal")
    calibrations, reports, common_cal_plan = {}, [], None
    for name, desc in by.items():
        manifest = bank_metadata(desc)
        cal = bound_json(desc["calibration"])
        active, weights = scoring.recipe()
        if (cal.get("active_components") != list(active) or cal.get("weights") != weights
                or set(cal.get("scales", {})) != set(active)
                or not all(np.isfinite(v) and v > 0 for v in cal["scales"].values())
                or not np.isfinite(cal.get("threshold", np.nan))
                or cal.get("source_domain_equal_weight") is not (name == "Z_S")
                or desc["identity"]["calibration_sha256"] != desc["calibration"]["sha256"]
                or manifest["contract"]["threshold"] != cal["threshold"]):
            raise PermissionError("Exact original own-CAL recipe/scale/threshold required")
        receipt = bound_json(desc["readout_receipt"])
        mode = "ZS_ZT" if name in METHODS[:2] else "U"
        condition = "COMMON" if mode == "ZS_ZT" else name[-1]
        view_key = "Z-S" if name == "Z_S" else "Z-T" if name == "Z_T" else "U"
        if (not str(receipt.get("status", "")).startswith("COMPLETED") or receipt.get("mode") != mode
                or receipt.get("condition") != condition or receipt.get("seed") != 2025
                or receipt.get("model_forward_bank_count") != 1 or receipt.get("points") != 18078
                or str(Path(receipt.get("views", {}).get(view_key, "")).resolve()) != str(Path(desc["directory"]).resolve())
                or receipt.get("assets", {}).get("checkpoint", {}).get("sha256") != desc["identity"]["checkpoint_sha256"]):
            raise PermissionError("Actual own readout lineage/condition/one-forward receipt required")
        plan = bound_json(receipt["assets"]["plan"])
        target_cal_plan = plan.get("CAL")
        if (not isinstance(target_cal_plan, list) or len(target_cal_plan) != 512
                or any(row.get("role") != "CAL" for row in target_cal_plan)):
            raise PermissionError("Exact 512 target CAL positions required")
        if common_cal_plan is None:
            common_cal_plan = target_cal_plan
        elif target_cal_plan != common_cal_plan:
            raise PermissionError("Target CAL pool/order must match across all five views")
        calibrations[name] = cal
        reports.append(dict(id=name, manifest_sha256=desc["manifest_sha256"], threshold=cal["threshold"]))
    overlay = bound_json(by["Z_T"]["overlay_receipt"])
    if (overlay.get("producer_manifest_sha256") != by["Z_S"]["manifest_sha256"]
            or overlay.get("destination_manifest_sha256") != by["Z_T"]["manifest_sha256"]
            or overlay.get("model_forward") is not False or overlay.get("calibration_fit") is not False
            or by["Z_T"]["readout_receipt"] != by["Z_S"]["readout_receipt"]):
        raise PermissionError("Z-T must be same accepted raw/source residuals, not second inference")
    arrays, auxiliary, z_residuals = {}, {}, None
    for name, desc in by.items():
        a, aux, residuals = records_to_arrays(chunked_bank.read_records(desc["directory"]), calibrations[name],
            start=10622, end=28700, stream=STREAM,
            auxiliary_calibration=calibrations["Z_T"] if name in METHODS[2:] else None)
        arrays[name] = a
        if aux is not None:
            auxiliary[name+"_ZT_FIXED"] = aux
        if name == "Z_S":
            z_residuals = residuals
        elif name == "Z_T" and not np.array_equal(residuals, z_residuals, equal_nan=True):
            raise ValueError("Z-T raw residuals differ from Z-S: not calibration-only")
    return arrays, auxiliary, calibrations, reports


def long_miss(result):
    """Existing longest-valid-positive-miss definition, per ORIGINAL event.

    Unknown points split observed segments; an ineligible event stays in the
    catalogue and its full-event interpretation is explicitly unavailable.
    """
    output = []
    rows = result["point_coverage"]
    for event in result["event_coverage"]:
        lo, hi = event["local_start"], event["local_end"]
        selected = rows[lo:hi]
        for stream in ("strict_point", "confirmed3_current"):
            valid = [r for r in selected if r["valid"]]
            miss = [r["valid"] and not bool(r[stream]) for r in selected]
            lengths = [b-a for a, b in binary_events(miss)]
            unknown = len(selected)-len(valid)
            output.append(dict(event_id=event["event_id"], start=event["start"], end_exclusive=event["end_exclusive"],
                stream=stream, full_event_eligible=event["main_eligible"], valid_positive_points=len(valid),
                unknown_positive_points=unknown, longest_observed_consecutive_miss_points=max(lengths, default=0) if valid else None,
                longest_miss_lower_bound_only=bool(unknown),
                complete_event_all_missed=(not any(r[stream] for r in valid)) if not unknown else None,
                reason="INCOMPLETE_SCORE_COVERAGE" if unknown else None))
    return output


def load_truth(asset):
    from .stage2_u_data import load_audit_labels
    labels, _ = load_audit_labels(asset["path"], asset["sha256"], 28700)
    if binary_events(labels) != [(1209, 1240), (5399, 5530), (10696, 10717), (27839, 27964)]:
        raise PermissionError("Original admitted event catalogue differs; never shift or clip")
    return labels


def admit(config_path, release_path, trusted):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    signed = bound_json(dict(path=release_path, sha256=trusted))
    contract(cfg)
    reject_truth_aliases(cfg)
    closure = source_closure()
    if (signed.get("status") != "APPROVED" or signed.get("approved_by") != "root"
            or signed.get("operation") != OPERATION or signed.get("allowed_actions") != ["evaluate_fixed_views"]
            or signed.get("config_sha256") != sha(config_path) or signed.get("source_sha256") != closure
            or cfg.get("source_sha256") != closure or signed.get("permissions") != PERMISSIONS
            or signed.get("resources") != RESOURCES or signed.get("output") != cfg["output"]
            or Path(cfg["output"]).exists()
            or any(signed.get(k) != cfg.get(k) for k in ("whole_raw_seal", "truth", "truth_admission", "affiliation_identity"))):
        raise PermissionError("Exact root raw/truth/CPU/source release and fresh output required")
    admission = bound_json(cfg["truth_admission"])
    if (admission.get("approved_by") != "root" or not str(admission.get("status", "")).startswith("ACCEPTED")
            or admission.get("truth") != cfg["truth"]):
        raise PermissionError("Separate post-raw root truth activation required")
    bound_json(cfg["affiliation_identity"])
    require_metric_dependencies()
    from ..engine.official_affiliation import load_official_affiliation
    load_official_affiliation(cfg["affiliation_identity"]["path"])
    if "torch" in sys.modules:
        raise PermissionError("CPU metric process must not import Torch")
    return cfg


def run(cfg):
    arrays, auxiliary, calibrations, gates = prelabel(cfg)
    labels = load_truth(cfg["truth"])  # Sole truth opener AFTER all5 complete scalar gates.
    from .evaluation import evaluate_arrays, save_evaluation
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=False)
    all_arrays = dict(arrays, **auxiliary)
    results = {}
    for name, values in all_arrays.items():
        calibration = calibrations["Z_T"] if name.endswith("_ZT_FIXED") else calibrations[name]
        result = evaluate_arrays(**values, full_labels=labels, threshold=calibration["threshold"], bank_complete=True,
            original_length=28700, original_start=0, warmup_points=10622,
            affiliation_identity=cfg["affiliation_identity"]["path"])
        reject_dependency_na(result)
        save_evaluation(out/name, result, **values, full_labels=labels,
            provenance=dict(primary_score_view=name in METHODS, cached_ZT_scale_AND_threshold=name.endswith("_ZT_FIXED"),
                            whole_raw_seal=cfg["whole_raw_seal"], truth=cfg["truth"], source_sha256=cfg["source_sha256"]))
        dump(out/name/"long_miss.json", long_miss(result)); results[name] = result
    comparisons = []
    for suffix in ("", "_ZT_FIXED"):
        pairs = PAIRS if not suffix else (("U_A", "U_B"), ("U_A", "U_C"), ("U_C", "U_B"))
        for left, right in pairs:
            a, b = left+suffix, right+suffix
            point, common = point_pair(all_arrays[a], all_arrays[b], labels, 0)
            reject_dependency_na(point)
            folder = out/"pairs"/(a+"__vs__"+b); folder.mkdir(parents=True)
            event_rows = paired_events(results[a], results[b])
            dump(folder/"paired_events.json", event_rows)
            np.savez_compressed(folder/"common_inputs.npz", index=common, full_labels=labels,
                                reference_score=all_arrays[a]["score"][np.searchsorted(all_arrays[a]["index"], common)],
                                candidate_score=all_arrays[b]["score"][np.searchsorted(all_arrays[b]["index"], common)])
            row = dict(reference=a, candidate=b, point_effect=point, auxiliary_diagnostic=bool(suffix),
                       statistics=cfg["statistics"])
            dump(folder/"summary.json", row); comparisons.append(row)
    dump(out/"BANK_GATES.json", gates); dump(out/"CONFIG.json", cfg)
    summary = dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT", primary_views=list(METHODS),
        auxiliary_views=list(auxiliary), neural_raw_banks=4, primary_comparisons=7, auxiliary_comparisons=3,
        statistics=cfg["statistics"], original_label_length=28700, expected_scored_points=18078,
        qualification_labels_previously_read=True, development_not_blind=True,
        methods={k:dict(ranking=v["ranking"], coverage=v["coverage"], streams={s:dict(point=q["point"],
            events=q["main_events"]["metrics"], normal=q["normal"], affiliation=q["affiliation"], Range_VUS=q["Range_VUS"])
            for s,q in v["streams"].items()}) for k,v in results.items()}, comparisons=comparisons,
        scientific_decision="ROOT_REQUIRED_NO_AUTOMATIC_ROBUSTNESS_CLAIM_NO_FLOOR_EFFECT_PROOF")
    dump(out/"SUMMARY.json", summary)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "release", "trusted-release-sha256"):
        p.add_argument("--"+key)
    p.add_argument("--execute", action="store_true"); p.add_argument("--preflight-only", action="store_true")
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(preview=True, bank_or_truth_opened=False))); return
    if not all((a.config, a.release, a.trusted_release_sha256)):
        raise PermissionError("Root exact evaluation release required")
    cfg = admit(a.config, a.release, a.trusted_release_sha256)
    if a.preflight_only:
        prelabel(cfg); print(json.dumps(dict(preflight=True, truth_opened=False))); return
    print(json.dumps(dict(status=run(cfg)["status"])))


if __name__ == "__main__":
    main()
