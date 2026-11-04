"""Independent fixed-wave CPU evaluator. Preview never opens banks or truth.

Metric definitions and serialization are delegated to the accepted evaluator.
No neural imports, model calls, calibration fitting or threshold selection.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from . import controlled_wave as wave
from . import scoring, chunked_bank

OPERATION = "HYBRID_V1_CONTROLLED_WAVE1_COMPLETE_FIXED_EVALUATION"


def require_metric_dependencies():
    """Import the real metric submodule before truth; never call its metrics."""
    try:
        metrics = importlib.import_module("sklearn.metrics")
    except ImportError as exc:
        raise PermissionError("Formal sklearn.metrics dependency cannot be imported") from exc
    names = ("average_precision_score", "roc_auc_score", "precision_recall_curve", "auc")
    if not all(callable(getattr(metrics, name, None)) for name in names):
        raise PermissionError("Formal sklearn.metrics requires all four original metric callables")


def reject_dependency_na(value):
    """Do not silently publish backend dependency failures as completed metrics.

    Legitimate single-class/empty-denominator/official undefined NA stays intact.
    """
    if isinstance(value, dict):
        for item in value.values():
            reject_dependency_na(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_dependency_na(item)
    elif isinstance(value, str) and (
            value.startswith("RANKING_DEPENDENCY_UNAVAILABLE")
            or value.startswith("SKLEARN_UNAVAILABLE_NOT_COMPUTED")
            or value.startswith("BLOCKED: official Affiliation dependency unavailable")):
        raise PermissionError("Formal evaluation dependency failure: " + value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar_record(record, calibration):
    """Check typed archived validity/reason as well as source score and alarm."""
    replay = scoring.apply_calibration(record, calibration)
    if type(record.get("valid")) is not bool or record["valid"] != replay["valid"]:
        raise ValueError("Archived validity differs from source scorer")
    if record.get("reason") != replay.get("reason"):
        raise ValueError("Archived reason differs from source scorer")
    if replay["valid"]:
        if (type(record.get("alarm")) is not bool or record["alarm"] != replay["alarm"]
                or not np.isfinite(record.get("score", np.nan))
                or not np.isclose(record["score"], replay["score"], atol=1e-12, rtol=1e-12)):
            raise ValueError("Archived score/alarm differs from source scorer")
    else:
        if (record.get("score") is not None and np.isfinite(record["score"])) or record.get("alarm") is not None:
            raise ValueError("Invalid source score must not become a normal alarm decision")
    return (record["index"], np.nan if record.get("score") is None else record["score"],
            record["valid"], record["available_index"], record.get("reason"))


def prelabel_banks(cfg, protocol):
    """ALL seventy identity/scalar checks complete before any label opener."""
    plan = wave.evaluation_metadata(protocol, cfg["banks"], wave.bound(cfg["raw_seal"]))
    arrays = {}
    for desc in cfg["banks"]:
        case = next(c for c in protocol["cases"] if c["id"] == desc["case"])
        manifest = wave.bound(dict(path=str(Path(desc["directory"])/"manifest.json"),
                                   sha256=desc["manifest_sha256"]))
        spans = [dict(domain=wave.DOMAIN, entity=case["family"], run=case["id"],
                      start=protocol["history"], end=protocol["N"])]
        if (manifest.get("complete") is not True
                or manifest.get("committed_records") != protocol["N"]-protocol["history"]
                or manifest["contract"]["ranges"] != spans
                or manifest["contract"]["identity"] != desc["identity"]):
            raise PermissionError("Incomplete or mismatched fixed bank")
        cal = wave.bound(desc["calibration"])
        wave.validate_calibration(cal, desc["active_components"], scoring)
        if (manifest["contract"]["threshold"] != cal["threshold"]
                or desc["identity"]["calibration_sha256"] != desc["calibration"]["sha256"]):
            raise PermissionError("Own source scale or threshold changed")
        rows = []
        for number, row in enumerate(chunked_bank.read_records(desc["directory"])):
            if (type(row.get("index")) is not int or row["index"] != protocol["history"]+number
                    or type(row.get("available_index")) is not int or row["available_index"] != row["index"]
                    or (row["domain"], row["entity"], row["run"]) != (wave.DOMAIN, case["family"], case["id"])):
                raise ValueError("Original arrived clock or stream mismatch")
            rows.append(scalar_record(row, cal))
        if len(rows) != protocol["N"]-protocol["history"]:
            raise ValueError("Missing final/raw points before labels")
        arrays[(desc["case"], desc["method"])] = dict(
            index=np.asarray([x[0] for x in rows], np.int64),
            score=np.asarray([x[1] for x in rows], np.float64),
            valid=np.asarray([x[2] for x in rows], bool),
            available_index=np.asarray([x[3] for x in rows], np.int64),
            invalid_reasons=[x[4] for x in rows])
    return plan, arrays


def validate_truth(labels, protocol):
    if set(labels) != {c["id"] for c in protocol["cases"]}:
        raise ValueError("Exact full eleven-case truth required")
    for case in protocol["cases"]:
        y = labels[case["id"]]
        expected = np.zeros(protocol["N"], np.int8)
        if case.get("event"):
            lo, hi = case["event"]; expected[lo:hi] = 1
        if y.dtype.kind not in "biu" or not np.array_equal(y, expected):
            raise ValueError("Original full-axis truth or fixed event boundary changed")


def phase_counts(result, spans):
    """Counts from full-axis alarms; never re-run confirmation/events per phase."""
    output = {}
    for name, span in spans.items():
        if span is None or span[0] >= span[1]:
            output[name] = None; continue
        rows = [r for r in result["point_coverage"] if span[0] <= r["index"] < span[1] and r["valid"]]
        output[name] = dict(interval=span, valid_points=len(rows),
            positive_points=sum(r["label"] == 1 for r in rows),
            strict_alarm_points=sum(r["strict_point"] for r in rows),
            confirmed3_alarm_points=sum(r["confirmed3_current"] for r in rows),
            strict_normal_FP=sum(r["strict_point"] and r["label"] == 0 for r in rows),
            confirmed3_normal_FP=sum(r["confirmed3_current"] and r["label"] == 0 for r in rows))
    return output


def required_closure():
    """Explicit package files imported by this adapter and delegated old math."""
    root = Path(__file__).parents[1]
    names = ["__init__.py", "config.py", "candidate_experiment.py", "engine/__init__.py",
             "engine/events.py", "engine/evaluation.py", "engine/statistics.py", "engine/official_affiliation.py",
             "hybrid_v1/__init__.py", "hybrid_v1/config.py", "hybrid_v1/provenance.py",
             "hybrid_v1/evaluation.py", "hybrid_v1/scoring.py",
             "hybrid_v1/chunked_bank.py", "hybrid_v1/statistics.py", "hybrid_v1/evaluate_cli.py",
             "hybrid_v1/named_evaluate_cli.py", "hybrid_v1/controlled_mechanism.py",
             "hybrid_v1/controlled_wave.py", "hybrid_v1/controlled_wave_evaluation.py"]
    return {str((root/name).resolve()): sha(root/name) for name in names}


def admit(config, release, trusted):
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    signed = wave.bound(dict(path=release, sha256=trusted))
    permissions = dict(evaluation=True, labels=True, training=False, GPU=False,
                       model=False, fit=False, resume=False)
    closure = required_closure()
    if (signed.get("status") != "APPROVED" or signed.get("approved_by") != "root"
            or signed.get("operation") != OPERATION or cfg.get("operation") != OPERATION
            or signed.get("allowed_actions") != [OPERATION] or signed.get("config_sha256") != sha(config)
            or cfg.get("permissions") != permissions or signed.get("permissions") != permissions
            or cfg.get("source_sha256") != closure or signed.get("source_sha256") != closure
            or signed.get("resources") != cfg.get("resources")
            or cfg.get("resources") != dict(wall_seconds=900, cleanup_seconds=5, threads=1, GPUh=0)
            or Path(cfg["output"]).exists()):
        raise PermissionError("Exact independent evaluator root release/closure/fresh output required")
    if cfg["protocol"]["sha256"] != wave.PROTOCOL_SHA:
        raise PermissionError("Fixed protocol required")
    # Identity-only label hashing, no NPZ deserialization until all banks pass.
    for key in ("labels", "raw_seal", "affiliation_identity"):
        if signed.get(key) != cfg.get(key) or sha(cfg[key]["path"]) != cfg[key]["sha256"]:
            raise PermissionError("Root-bound truth/raw/official identity required")
    from ..engine.official_affiliation import load_official_affiliation
    load_official_affiliation(cfg["affiliation_identity"]["path"])
    require_metric_dependencies()
    for name in ("sklearn", "scipy"):
        if importlib.util.find_spec(name) is None:
            raise PermissionError("Formal metric dependency unavailable: "+name)
    return cfg, wave.protocol_check(wave.bound(cfg["protocol"]))


def run(cfg, protocol):
    require_metric_dependencies()
    plan, arrays = prelabel_banks(cfg, protocol)
    # This is the sole label opener, after ALL raw evidence gates.
    with np.load(cfg["labels"]["path"], allow_pickle=False) as archive:
        labels = {key: archive[key] for key in archive.files}
    validate_truth(labels, protocol)
    from . import evaluation as backend
    from .named_evaluate_cli import point_pair, paired_events
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=False)
    results = {}; summaries = []
    for desc in cfg["banks"]:
        case, method = desc["case"], desc["method"]
        a = arrays[(case, method)]; y = labels[case]
        threshold = wave.bound(desc["calibration"])["threshold"]
        result = wave.dispatch_evaluation(plan, a, y, threshold, backend,
                                         cfg["affiliation_identity"]["path"])
        reject_dependency_na(result)
        backend.save_evaluation(out/case/method, result, full_labels=y,
            provenance=dict(descriptor=desc, labels=cfg["labels"], raw_seal=cfg["raw_seal"],
                            source_sha256=cfg["source_sha256"]), **a)
        results[(case, method)] = result
        summaries.append(dict(case=case, method=method, ranking=result["ranking"], coverage=result["coverage"],
            streams={k: dict(point=v["point"], events=v["main_events"]["metrics"], normal=v["normal"],
                affiliation=v["affiliation"]) for k,v in result["streams"].items()},
            phases=phase_counts(result, plan["phase_intervals"][case])))
    comparisons = []
    for pair in plan["comparisons"]:
        case, left, right = pair["case"], pair["reference"], pair["candidate"]
        effect, common = point_pair(arrays[(case,left)], arrays[(case,right)], labels[case], 0)
        reject_dependency_na(effect)
        events = paired_events(results[(case,left)], results[(case,right)])
        folder = out/"pairs"/case/(left+"__vs__"+right); folder.mkdir(parents=True)
        wave.dump(folder/"paired_events.json", events)
        np.savez_compressed(folder/"common_inputs.npz", index=common, full_labels=labels[case],
            reference_score=arrays[(case,left)]["score"][np.searchsorted(arrays[(case,left)]["index"],common)],
            candidate_score=arrays[(case,right)]["score"][np.searchsorted(arrays[(case,right)]["index"],common)])
        row = dict(pair, point_effect=effect, CI=None, independent_data_replicates=False)
        wave.dump(folder/"summary.json", row); comparisons.append(row)
    aliases = dict(shams=plan["aliases"], semantic=[], extra_forward=False, independent_replicates=False)
    for alias in plan["semantic_aliases"]:
        case = alias["source"]
        for desc in (d for d in cfg["banks"] if d["case"] == case):
            a = arrays[(case,desc["method"])]; y = np.zeros_like(labels[case])
            result = wave.dispatch_evaluation(plan,a,y,wave.bound(desc["calibration"])["threshold"],backend,
                                              cfg["affiliation_identity"]["path"])
            reject_dependency_na(result)
            backend.save_evaluation(out/"semantic_all_normal"/case/desc["method"],result,full_labels=y,
                provenance=dict(same_bank=desc,alternate_truth="PROTOCOL_ALL_NORMAL",not_independent=True),**a)
            aliases["semantic"].append(dict(source_case=case,method=desc["method"],ranking=result["ranking"],
                no_event_pair_with_original_truth=True))
    wave.dump(out/"ALIASES.json",aliases)
    wave.dump(out/"CONFIG.json",cfg)
    summary = dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT",methods=summaries,comparisons=comparisons,
                   statistics=dict(CI=None,VUS=None,independent_data_replicates=False),plan=plan)
    wave.dump(out/"SUMMARY.json",summary)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",required=True)
    p.add_argument("--release");p.add_argument("--trusted-release-sha256");p.add_argument("--execute",action="store_true")
    p.add_argument("--preflight-only",action="store_true");a=p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(preview=True,banks_opened=False,truth_opened=False)));return
    if not a.release or not a.trusted_release_sha256:raise PermissionError("Separate root truth release required")
    cfg,protocol=admit(a.config,a.release,a.trusted_release_sha256)
    if a.preflight_only:
        prelabel_banks(cfg,protocol);print(json.dumps(dict(preflight=True,truth_opened=False)));return
    print(json.dumps(dict(status=run(cfg,protocol)["status"])))


if __name__ == "__main__":main()
