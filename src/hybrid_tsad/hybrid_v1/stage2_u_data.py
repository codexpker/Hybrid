"""Role-contained Stage2-U data preparation; audit labels never reach learners.

No model, torch, score, or checkpoint dependency. Original coordinates are
half-open throughout. This module does not relax donor eligibility on failure.
"""
import hashlib
import json
from pathlib import Path
import copy

import numpy as np


def identity(path):
    path = Path(path)
    body = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(body)}


def load_native_values(path, expected_sha256, channels=38):
    source = identity(path)
    if source["sha256"] != expected_sha256:
        raise ValueError("Native values byte identity mismatch")
    values = np.loadtxt(path, delimiter=",", ndmin=2)
    if values.ndim != 2 or values.shape[1] != channels:
        raise ValueError("Unexpected native variable geometry")
    observed = np.isfinite(values)
    if not observed.all():
        raise ValueError("Native finite/observed admission failed; no imputation permitted")
    converted = values.astype(np.float32)
    if not np.isfinite(converted).all():
        raise ValueError("Float32 conversion loses finite values")
    return converted, observed, source


def load_audit_labels(path, expected_sha256, length):
    source = identity(path)
    if source["sha256"] != expected_sha256:
        raise ValueError("Native label byte identity mismatch")
    labels = np.loadtxt(path, delimiter=",", ndmin=1)
    if labels.shape != (length,) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Only one binary official label per original point is admitted")
    return labels.astype(np.int8), source


def binary_events(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Binary original-axis labels required")
    change = np.diff(np.r_[0, labels, 0].astype(np.int8))
    return [{"event_id": "machine-3-1_test:%d:%d" % (a, b),
             "start": int(a), "end": int(b), "points": int(b-a)}
            for a, b in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))]


def role_geometry(train_length, test_length, history=512, horizon=32, guard=544):
    if history+horizon != guard:
        raise ValueError("Registered guard must equal H+P")
    donor_end = test_length//3
    result = {"DONOR": [0, donor_end], "TEST_GUARD": [donor_end, donor_end+guard],
              "QUERY": [donor_end+guard, test_length],
              "QUERY_H1_SCORED": [donor_end+guard+history, test_length],
              "history": history, "horizon": horizon, "guard": guard}
    if train_length is not None:
        a, b = 3*train_length//5, 4*train_length//5
        result.update(FIT=[0, a], TRAIN_GUARD_1=[a, a+guard],
                      SELECT=[a+guard, b], TRAIN_GUARD_2=[b, b+guard],
                      CAL=[b+guard, train_length])
        if any(result[role][1]-result[role][0] < history+horizon
               for role in ("FIT", "SELECT", "CAL")):
            raise ValueError("Train roles too short after fixed guards")
    return result


def eligible_targets(observed, lo, hi, history=512, horizon=32, arrived=False):
    observed = np.asarray(observed, dtype=bool)
    end_extra = 1 if arrived else horizon
    targets = np.arange(lo+history, hi-end_extra+1, dtype=np.int64)
    counts = np.concatenate((np.zeros((1, observed.shape[1]), np.int64),
                             np.cumsum(observed, axis=0)), axis=0)
    keep = observed[targets].all(axis=1) & ((counts[targets]-counts[targets-history]) >= 2).all(axis=1)
    return targets[keep]


def donor_admission(labels, observed, donor_bounds, history=512, horizon=32):
    """Report strict and alternative counts separately; do not select a rule."""
    lo, hi = donor_bounds
    ts = eligible_targets(observed, lo, hi, history, horizon)
    starts, ends = ts-history, ts+horizon
    sums = np.r_[0, np.cumsum(labels, dtype=np.int64)]
    positives = sums[ends]-sums[starts]
    events = binary_events(labels)
    complete = [e for e in events if lo <= e["start"] and e["end"] <= hi]
    complete_counts = np.zeros(len(ts), np.int64)
    supervised_counts = np.zeros(len(ts), np.int64)
    details = []
    for event in complete:
        contains = (starts <= event["start"]) & (ends >= event["end"])
        complete_counts += contains
        supervised = (ts >= event["start"]) & (ends <= event["end"])
        supervised_counts += supervised
        details.append(dict(event, containing_full_544_footprints=int(contains.sum()),
                            entire_P32_supervision_inside_same_complete_event=int(supervised.sum())))
    return {"full_footprint_points": history+horizon,
            "eligible_observed_footprints": int(len(ts)),
            "normal_all_544_points": int((positives == 0).sum()),
            "abnormal_all_544_points": int((positives == history+horizon).sum()),
            "any_abnormal_in_544_points": int((positives > 0).sum()),
            "contains_at_least_one_complete_event": int((complete_counts > 0).sum()),
            "primary_abnormal_supervision_same_complete_event": int((supervised_counts > 0).sum()),
            "complete_donor_events": complete, "complete_event_footprint_counts": details,
            "rule_selection": "ROOT_FIXED_BEFORE_COUNTS_P32_IN_SAME_COMPLETE_PREFIX_EVENT_HISTORY_MAY_TRANSITION"}


def query_coverage(labels, scored_indices, query_bounds):
    scored = np.zeros(len(labels), bool)
    scored[np.asarray(scored_indices, dtype=np.int64)] = True
    lo, hi = query_bounds
    events = []
    for event in binary_events(labels):
        if event["start"] >= hi or event["end"] <= lo:
            continue
        covered = int(scored[event["start"]:event["end"]].sum())
        events.append(dict(event, scored_points=covered,
                           coverage_ratio=covered/event["points"],
                           eligible_for_complete_event_metric=covered == event["points"]))
    return {"point_scored_count": int(scored.sum()), "events": events,
            "complete_event_count": sum(e["eligible_for_complete_event_metric"] for e in events),
            "partial_event_count": sum(not e["eligible_for_complete_event_metric"] for e in events),
            "labels_used_for": "qualification_and_full_original_event_coverage_only_not_scores"}


def write_role_npz(destination, values, observed, bounds):
    """Create only numerical role file. No labels, events or weights permitted."""
    destination = Path(destination)
    lo, hi = bounds
    if not (0 <= lo < hi <= len(values)) or observed.shape != values.shape:
        raise ValueError("Invalid original-axis numerical role bounds")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.savez(handle, values=np.asarray(values[lo:hi], np.float32),
                 observed=np.asarray(observed[lo:hi], bool))
    with np.load(destination, allow_pickle=False) as archive:
        if set(archive.files) != {"values", "observed"}:
            raise AssertionError("Learner role archive key mismatch")
        if not (np.array_equal(archive["values"], values[lo:hi]) and
                np.array_equal(archive["observed"], observed[lo:hi])):
            raise AssertionError("Derived role is not the admitted original numerical slice")
    return dict(identity(destination), bounds=[0, hi-lo], expected_shape=[hi-lo, values.shape[1]],
                original_start=lo, format="npz", key="values", observed_key="observed")


def write_json_create_only(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def stream_seed(seed, tag):
    if tag not in {"replace_slots", "anomaly_donor", "normal_donor"}:
        raise ValueError("Only registered independent streams permitted")
    message = "HYBRID_STAGE2_U_V1|"+str(seed)+"|"+tag
    return int.from_bytes(hashlib.sha256(message.encode("utf-8")).digest()[:8], "little")


def donor_target_pools(labels, observed, bounds, history=512, horizon=32):
    ts = eligible_targets(observed, *bounds, history, horizon)
    sums = np.r_[0, np.cumsum(labels, dtype=np.int64)]
    normal = ts[(sums[ts+horizon]-sums[ts-history]) == 0]
    complete = [e for e in binary_events(labels)
                if bounds[0] <= e["start"] and e["end"] <= bounds[1]]
    mask = np.zeros(len(ts), bool)
    for event in complete:
        mask |= (ts >= event["start"]) & (ts+horizon <= event["end"])
    abnormal = ts[mask]
    if not len(normal) or not len(abnormal):
        raise ValueError("BLOCKED: no qualified normal or P32 abnormal donor; never relax")
    return normal, abnormal


def _entry(run, index, history=512, horizon=32):
    return {"domain": run.domain, "entity": run.entity, "run": run.run,
            "role": run.role, "index": int(index),
            "footprint": [int(index)-history, int(index)+horizon]}


def exposure_ledger(entries, labels, native_events, seed, condition, slots):
    """Known donor positivity vs unknown train background, not fictitious 0%."""
    all_test, forecast_test = [], []
    known_full, known_forecast, mapping = 0, 0, []
    for attempt, entry in enumerate(entries):
        if entry["run"] != "machine-3-1_test":
            continue
        lo, hi = entry["footprint"]; t = entry["index"]
        pos = int(labels[lo:hi].sum()); fpos = int(labels[t:hi].sum())
        known_full += pos; known_forecast += fpos
        all_test.extend(range(lo, hi)); forecast_test.extend(range(t, hi))
        mapping.append({"attempt": attempt, "index": t, "footprint": [lo, hi],
                        "known_positive_points_H_plus_P": pos,
                        "known_positive_points_P": fpos,
                        "overlapping_original_events": [e["event_id"] for e in native_events
                            if e["start"] < hi and lo < e["end"]]})
    unique = sorted(set(all_test)); unique_f = sorted(set(forecast_test))
    events = sorted({e["event_id"] for e in native_events
                     if any(e["start"] <= t < e["end"] for t in unique)})
    donors = len(mapping); attempts = len(entries)
    return {"seed": seed, "condition": condition,
            "replacement_slots_in_generated_order": [int(s) for s in slots] if condition != "A" else [],
            "donor_slots": donors, "total_attempts": attempts,
            "slot_fraction": donors/attempts,
            "total_H_plus_P_exposure_points": attempts*544,
            "known_donor_H_plus_P_exposure_points": donors*544,
            "known_positive_H_plus_P_exposure_points": known_full,
            "known_positive_fraction_of_total_H_plus_P_exposure_lower_bound": known_full/(attempts*544),
            "unknown_train_H_plus_P_exposure_points": (attempts-donors)*544,
            "known_donor_P_exposure_points": donors*32,
            "known_positive_P_exposure_points": known_forecast,
            "unknown_train_P_exposure_points": (attempts-donors)*32,
            "unique_donor_H_plus_P_points": len(unique),
            "unique_positive_donor_H_plus_P_points": int(labels[unique].sum()) if unique else 0,
            "unique_donor_P_points": len(unique_f),
            "unique_positive_donor_P_points": int(labels[unique_f].sum()) if unique_f else 0,
            "unique_donor_targets": len({m["index"] for m in mapping}),
            "independent_donor_event_ids": events,
            "independent_donor_event_count": len(events),
            "per_attempt_donor_mapping": mapping,
            "background_true_anomaly_fraction": None,
            "background_status": "Original train nominally normal by dataset use, not independently certified healthy",
            "normal_matching": "Same machine/test-prefix record only; no operating-state metadata available"}


def prepare_machine31(config):
    """Authorized CPU derivation. Output directory must not already exist."""
    from .data import Run, fixed_plan, resolve_plan
    out = Path(config["output"])
    if out.exists():
        raise FileExistsError("Create-only role/plan root already exists")
    train, train_obs, train_id = load_native_values(config["train"], config["train_sha256"])
    test, test_obs, test_id = load_native_values(config["test"], config["test_sha256"])
    labels, labels_id = load_audit_labels(config["audit_labels"], config["audit_labels_sha256"], len(test))
    g = role_geometry(len(train), len(test))
    normal_pool, abnormal_pool = donor_target_pools(labels, test_obs, g["DONOR"])
    query_indices = eligible_targets(test_obs, *g["QUERY"], arrived=True)
    query = query_coverage(labels, query_indices, g["QUERY"])
    events = binary_events(labels)
    out.mkdir(parents=True, exist_ok=False)
    learner, audit = out/"learner", out/"audit_only"
    learner.mkdir(); audit.mkdir()
    train_runs, descriptors = {}, {}
    for role in ("FIT", "SELECT", "CAL"):
        lo, hi = g[role]
        train_runs[role] = Run("SMD", "machine-3-1", "machine-3-1_train", role,
                               train[lo:hi], train_obs[lo:hi], lo)
        descriptors[role] = dict(write_role_npz(learner/(role+".npz"), train, train_obs, [lo, hi]),
                                domain="SMD", entity="machine-3-1", run="machine-3-1_train", role=role)
    for name, role, bounds in (("FIT_pool", "FIT", g["DONOR"]), ("QUERY", "QUERY", g["QUERY"])):
        descriptors[name] = dict(write_role_npz(learner/(name+".npz"), test, test_obs, bounds),
                                 domain="SMD", entity="machine-3-1", run="machine-3-1_test", role=role)
    # Pure learner schema excludes byte counts and source paths/labels.
    allowed = {"domain", "entity", "run", "role", "path", "format", "key", "observed_key",
               "sha256", "expected_shape", "bounds", "original_start"}
    descriptors = {k: {a: b for a, b in v.items() if a in allowed} for k, v in descriptors.items()}
    shared_select = fixed_plan([train_runs["SELECT"]], 512, 32, 32, 2718, "SELECT", replace=False)
    shared_cal = fixed_plan([train_runs["CAL"]], 512, 32, 512, 31415, "CAL", arrived=True, replace=False)
    donor_run = Run("SMD", "machine-3-1", "machine-3-1_test", "FIT",
                    test[:g["DONOR"][1]], test_obs[:g["DONOR"][1]], 0)
    manifest_ids, plan_ids, ledgers = {}, [], []
    for condition in ("A", "B", "C"):
        names = ["FIT", "SELECT", "CAL", "QUERY"]
        if condition != "A":
            names.insert(1, "FIT_pool")
        manifest = {"runs": [descriptors[name] for name in names]}
        p = learner/("MANIFEST_"+condition+".json")
        write_json_create_only(p, manifest); manifest_ids[condition] = identity(p)
    for seed in (2025, 2026, 2027):
        background = fixed_plan([train_runs["FIT"]], 512, 32, 200, seed, "FIT", replace=True)
        slots = np.random.default_rng(stream_seed(seed, "replace_slots")).choice(200, 20, replace=False)
        abnormal = np.random.default_rng(stream_seed(seed, "anomaly_donor")).choice(abnormal_pool, 20, replace=True)
        normal = np.random.default_rng(stream_seed(seed, "normal_donor")).choice(normal_pool, 20, replace=True)
        for condition, replacement in (("A", None), ("B", abnormal), ("C", normal)):
            fit = copy.deepcopy(background)
            if replacement is not None:
                for slot, t in zip(slots, replacement):
                    fit[int(slot)] = _entry(donor_run, int(t))
            runs = [train_runs["FIT"]]+([donor_run] if replacement is not None else [])
            resolve_plan(runs, fit, 512, 32)
            resolve_plan([train_runs["SELECT"]], shared_select, 512, 32)
            resolve_plan([train_runs["CAL"]], shared_cal, 512, 32, arrived=True)
            plan = {"history": 512, "horizon": 32, "seed": seed,
                    "FIT": fit, "SELECT": shared_select, "CAL": shared_cal}
            p = learner/("PLAN_%d_%s.json" % (seed, condition))
            write_json_create_only(p, plan); plan_ids.append(identity(p))
            ledgers.append(exposure_ledger(fit, labels, events, seed, condition, slots))
    # No learner manifest or plan references this audit-only truth/qualification.
    with (audit/"native_test_labels.npz").open("xb") as handle:
        np.savez(handle, labels=labels, original_index=np.arange(len(labels), dtype=np.int64))
    report = {"status": "CPU_DERIVED_PENDING_INDEPENDENT_AUDIT_EXPLORATORY_ONLY",
              "sources": {"train": train_id, "test": test_id, "labels_audit_only": labels_id},
              "geometry": g, "native_train_shape": list(train.shape), "native_test_shape": list(test.shape),
              "all_native_values_finite": True,
              "donor_admission": donor_admission(labels, test_obs, g["DONOR"]),
              "query_coverage": query, "query_h1_scored_points": int(len(query_indices)),
              "learner_manifests": manifest_ids, "learner_plans": plan_ids,
              "exposure_ledgers": ledgers,
              "source_module": identity(__file__),
              "rng_protocol": {"base_fit": "existing data.fixed_plan PCG64 with replacement, seed=modelseed",
                               "SELECT": "existing fixed_plan without replacement seed2718 count32",
                               "CAL": "existing arrived fixed_plan without replacement seed31415 count512",
                               "replacement": "PCG64 SHA256 registered streams, unsorted generated slots, donor with replacement"},
              "permissions": {"training": False, "score_evaluation": False, "GPU": False},
              "limitations": ["One eligible donor event and two query events: statistical confirmation INCOMPLETE",
                              "Historical query exposure: development/exploratory, not blind confirmation",
                              "Native train has no certified health labels: zero additional replacement is not zero true contamination",
                              "Physical clock not authenticated; indices are points, not seconds"]}
    write_json_create_only(audit/"ADMISSION_AND_EXPOSURE.json", report)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    if not args.execute:
        print(json.dumps({"config": identity(args.config), "arrays_opened": False}))
    else:
        result = prepare_machine31(cfg)
        print(json.dumps({"status": result["status"], "geometry": result["geometry"],
                          "plan_count": len(result["learner_plans"]),
                          "query_points": result["query_h1_scored_points"]}, indent=2))
