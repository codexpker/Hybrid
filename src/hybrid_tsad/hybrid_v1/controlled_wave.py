"""Typed controlled-input glue. No model, score or metric math is replaced.

The CPU_L0 sidecar remains immutable. Imports of neural/evaluation kernels are
lazy and restricted to the explicitly hashed frozen package. Default is preview.
"""
import argparse
import importlib
import json
import hashlib
import math
from pathlib import Path
import random

import numpy as np

from . import controlled_mechanism as original

PREFIX = "HYBRID_V1_CONTROLLED_WAVE1_"
DERIVE, MODEL, BASELINE, EVALUATE = [PREFIX + x for x in
                                    ("INPUT_DERIVE", "MODEL_READOUT", "BASELINE_READOUT", "FIXED_EVALUATION")]
TECHNICAL = PREFIX + "INDEPENDENT_TECHNICAL_NORMAL_READOUT"
PROTOCOL_SHA = "e56cc3e9195707acc96e8c0690f82fec85d6cc23384337c725a6b6b36bb856a8"
ORIGINAL_SHA = "004b6c37e18b15c443f981d7a3249205e6113aa86b7844d47660211a4fcd1d10"
MODEL_IDS = ("full2025", "full2026", "full2027", "tf_off2025", "causal_off2025",
             "self_off2025", "cross_off2025")
SCIENCE = {"checkpoint", "calibration", "training_config", "training_receipt",
           "calibration_config", "calibration_receipt"}
DOMAIN = "CONTROLLED_WAVE1"
METHODS = ("persistence", "history_mean512")
BASELINE_CAL = dict(persistence="5513840e1454ef3cd84e373c510a336660d716d60971ad6e14b752e1c946d1cd",
                   history_mean512="16b3a4959a08998ea2dedbf3c2373e7a95e1fae15607f21d065ce42d9825fb9d")


def bound(spec):
    if original.sha(spec["path"]) != spec["sha256"]:
        raise PermissionError("Exact bound file changed: " + str(spec["path"]))
    return json.loads(Path(spec["path"]).read_text(encoding="utf-8"))


def dump(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def phases(case, history, length):
    interval = case.get("event") or case.get("matched_interval") or case.get("allowed_switch_interval")
    s, e = interval
    return {"pre": [history, s], "onset": [s, min(s + history, e)],
            "steady": [s + history, e] if s + history < e else None,
            "recovery": [e, min(e + history, length)],
            "late": [e + history, length] if e + history < length else None}


def anchors(case, history, length):
    points = set()
    for span in phases(case, history, length).values():
        if span and span[0] < span[1]:
            lo, hi = span
            points.update((lo, lo + (hi - lo - 1)//2, hi - 1))
    return sorted(points)


def protocol_check(protocol):
    if ((protocol.get("N"), protocol.get("history"), protocol.get("horizon"),
         protocol.get("burn_in"), protocol.get("data_run_id")) != (2560, 512, 32, 4096, 1)
            or protocol.get("neural_units_total") != 98304
            or protocol.get("loader_role") != "QUERY"
            or protocol.get("chunk_size") != 64
            or len(protocol.get("cases", [])) != 11):
        raise PermissionError("Only the root-reviewed fixed first wave protocol")
    return protocol


def _ar1(family, role, run_id, stream, length):
    seed = original.seed_for(family, role, run_id, stream)
    eps = np.random.Generator(np.random.PCG64(seed)).standard_normal(length)
    x = np.empty(length, np.float64); x[0] = eps[0]
    for t in range(1, length):
        x[t] = .8*x[t-1] + .6*eps[t]
    return x, eps, seed


def build_cases(protocol):
    """Pure builder. CLI calls this only after separate input-derive admission.

    Small tests pass a scaled local geometry, never the formal N2560 protocol.
    The original generator is called once per family/seed role, not per model.
    """
    n, burn = protocol["N"], protocol["burn_in"]
    d0, d1, rid = (protocol[k] for k in ("normal_fixed_delay", "changed_delay", "data_run_id"))
    cases, cache, truth = {}, {}, {}
    for entry in protocol["cases"]:
        family, role = entry["family"], entry["seed_role"]
        key = (family, role)
        interval = entry.get("event") or entry.get("matched_interval") or entry.get("allowed_switch_interval")
        # The base pair always uses the LONG event, even when this view is a pulse.
        long_interval = protocol["long_interval"] if "long_interval" in protocol else [768, 1536]
        if key not in cache:
            cache[key] = original.generate_pair(family, run_id=rid, role=role, n=n,
                burn_in=burn, event=tuple(long_interval), d0=d0, d1=d1)
            truth[family + "/" + role] = cache[key][1]
        pair, latent = cache[key]
        variant = entry["variant"]
        values = pair["A" if variant in ("A", "A_all_normal_ledger", "LEVEL025") else "N"]["values"].copy()
        if variant.startswith("PULSE"):
            s, e = interval
            values[s:e, 0] += entry["amplitude"]
        if variant == "BREAK":
            t = latent["retained_indices"]
            driver, eps, seed = _ar1(family, role, rid, "break_driver", len(latent["latents"]["u"]))
            s, e = interval
            # Fixed t-12, not t-20; shared independent replacement for +/- branches.
            lagged = driver[t[s:e] - d0]
            zp, zm = latent["latents"]["zplus"], latent["latents"]["zminus"]
            values[s:e, 1] = .3*lagged + np.sqrt(.91)*zp[t[s:e]]
            values[s:e, 2] = -.3*lagged + np.sqrt(.91)*zm[t[s:e]]
            truth[family + "/" + role + "/BREAK"] = dict(driver=driver, innovations=eps,
                seed=seed, source_delay=d0, source_sample_indices=t[s:e]-d0)
        if values.shape != (n, entry["C"]) or not np.isfinite(values).all():
            raise ValueError("Declared full-C finite input required")
        label = np.zeros(n, np.int8)
        if entry.get("event"):
            label[entry["event"][0]:entry["event"][1]] = 1
        cases[entry["id"]] = dict(values=values.astype(np.float32), observed=np.ones(values.shape, bool),
                                   index=np.arange(n, dtype=np.int64))
        truth[entry["id"]] = dict(labels=label, phases=phases(entry, protocol["history"], n),
            seed_role=role, loader_role="QUERY", data_run_id=rid, event=entry.get("event"))
    return cases, truth


def derive_inputs(protocol, output):
    """Fresh values-only manifest and separate audit truth. No model or metrics."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases, truth = build_cases(protocol)
    runs = []
    for entry in protocol["cases"]:
        case = entry["id"]; file = output / (case + ".npz")
        with file.open("xb") as handle:
            np.savez_compressed(handle, **cases[case])
        runs.append(dict(domain=DOMAIN, entity=entry["family"], run=case, role="QUERY",
            path=file.name, format="npz", key="values", observed_key="observed",
            bounds=[0, protocol["N"]], original_start=0,
            expected_shape=[protocol["N"], entry["C"]], sha256=original.sha(file)))
    dump(output/"INPUT_MANIFEST.json", dict(runs=runs))
    original.save_bundle(output, "AUDIT_TRUTH_NOT_MODEL_INPUT", truth)
    with (output/"TRUTH_LABELS.npz").open("xb") as handle:
        np.savez_compressed(handle, **{c["id"]: truth[c["id"]]["labels"] for c in protocol["cases"]})
    dump(output/"ALIASES.json", dict(shams=protocol["shams"],
         semantic_counterexamples=protocol["semantic_identical_input_counterexamples"],
         extra_forward=False, independent_replication=False))
    return dict(input_manifest_sha256=original.sha(output/"INPUT_MANIFEST.json"),
                unique_inputs=len(runs), truth_not_in_input_manifest=True)


def validate_active(record, active):
    """Inactive TF NA is preserved. Only the model's own fixed recipe is required."""
    for key in active:
        scalar = record.get(key)
        channels = np.asarray(record["per_channel"][key], dtype=float)
        if scalar is None or not np.isfinite(scalar) or scalar < 0 or not np.isfinite(channels).all() or (channels < 0).any():
            raise ValueError("Invalid required active residual: " + key)


def validate_calibration(cal, active, scoring):
    components, weights = scoring.recipe(active)
    if (cal.get("active_components") != list(components) or cal.get("weights") != weights
            or set(cal.get("scales", {})) != set(components)
            or not all(np.isfinite(v) and v > 0 for v in cal["scales"].values())
            or not np.isfinite(cal.get("threshold", np.nan))
            or cal.get("source_domain_equal_weight") is not True
            or cal.get("domains") != ["SMD", "Sunspot"]):
        raise PermissionError("Own original sourceCAL and fixed active recipe required")


def _numpy_rng():
    return np.random.get_state(), random.getstate()


def _same_numpy_rng(a, b):
    def exact(x, y):
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.array_equal(x, y)
        if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
            return len(x) == len(y) and all(exact(u, v) for u, v in zip(x, y))
        return x == y
    return exact(a, b)


def neural_rng():
    """Separate new adapter codec; old004b CPU codec remains unchanged."""
    import torch
    return dict(numpy_python=_numpy_rng(), cpu=torch.get_rng_state().clone(),
                cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else [])


def same_neural_rng(a, b):
    return (_same_numpy_rng(a["numpy_python"], b["numpy_python"])
            and np.array_equal(original.detached(a["cpu"]), original.detached(b["cpu"]))
            and len(a["cuda"]) == len(b["cuda"])
            and all(np.array_equal(original.detached(x), original.detached(y)) for x, y in zip(a["cuda"], b["cuda"])))


class AnchorModel:
    """Proxy around the actual model, never another model or numerical pass.

    The original iterator calls forecast/reconstruct exactly as before. This
    wrapper observes returned tensors at fixed indices and detaches copies only.
    An injected RNG codec is required (real torch codec after neural release).
    """
    def __init__(self, model, run, wanted, seed, sink, rng, same_rng):
        self.model, self.run = model, run
        self.wanted, self.seed, self.sink = set(wanted), seed, sink
        self.rng, self.same_rng, self.pending = rng, same_rng, None
        self.seen = set(); self.forecasts = self.reconstructions = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def forecast(self, *args, **kwargs):
        actual = kwargs["mask_seed"]
        pre = self.rng() if self.run.start + actual - self.seed in self.wanted else None
        f = self.model.forecast(*args, **kwargs); self.forecasts += 1
        if self.pending is not None:
            raise RuntimeError("Forecast/reconstruction ordering changed")
        self.pending = (actual, f, pre)
        return f

    def reconstruct(self, *args, **kwargs):
        r = self.model.reconstruct(*args, **kwargs); self.reconstructions += 1
        if self.pending is None or self.pending[0] != kwargs["mask_seed"]:
            raise RuntimeError("Reconstruction must pair with this original forecast")
        actual, f, before_pass = self.pending; self.pending = None
        local = actual - self.seed; index = self.run.start + local
        if index not in self.wanted:
            return r
        if index in self.seen:
            raise ValueError("Duplicate anchor; no extra inference is permitted")
        pre = self.rng()
        payload = dict(index=index, actual_seed=actual, RNG_before_forecast=before_pass, RNG_after_reconstruct=pre,
            available_index=index, footprint=[index-self.model.config.history, index+1],
            forecast_history_bounds=[index-self.model.config.history, index],
            reconstruction_bounds=[index-self.model.config.history+1, index+1],
            current_target=self.run.values[local:local+1], current_observed=self.run.observed[local:local+1],
            forecast={k: f[k] for k in ("prediction", "valid", "tf_order", "copy_masks", "stats")},
            reconstruction={k: r[k] for k in ("hidden_mask", "visible_mask", "transform_hidden_mask", "stats", "order")},
            tf_endpoint=r["tf_reconstruction"][:, :, :, -1],
            tf_valid=r["tf_valid"][:, :, :, -1],
            ms_endpoint=r["ms_reconstruction_by_scale"][:, :, :, :, -1],
            ms_valid=r["ms_valid_by_scale"][:, :, :, :, -1],
            group_mask_seeds=[actual+g for g in range(r["hidden_mask"].shape[1])],
            forecast_lag=f["axes"]["lag"], reconstruction_lag=[a["lag"] for a in r["axes"]],
            flags={k: v for k, v in self.model.config.to_dict().items() if isinstance(v, bool)})
        self.sink(original.detached(payload))
        if not self.same_rng(pre, self.rng()):
            raise AssertionError("Anchor export disturbed RNG")
        self.seen.add(index)
        return r

    def finish(self, expected):
        if self.pending is not None or self.forecasts != expected or self.reconstructions != expected or self.seen != self.wanted:
            raise ValueError("Incomplete same-pass anchor/forward coverage")


def consume_records(records, directory, *, identity, ranges, calibration, active, kernels):
    """Thin typed writer: the original iterator, scorer and lossless chunk bank."""
    cli, scoring, bank_module = kernels
    validate_calibration(calibration, active, scoring)
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("Fresh bank required; no automatic resume")
    import time
    timings = []
    with bank_module.ChunkedBank(directory, identity=identity, ranges=ranges,
            threshold=calibration["threshold"], chunk_size=64) as bank:
        iterator = iter(records)
        while True:
            began = time.monotonic()
            try: record = next(iterator)
            except StopIteration: break
            raw_seconds = time.monotonic()-began; began = time.monotonic()
            validate_active(record, active)
            bank.append(cli.score_records([record], calibration)[0])
            timings.append(dict(index=record["index"], forward_raw_anchor_seconds=raw_seconds,
                                score_append_seconds=time.monotonic()-began))
        bank.finalize()
    dump(directory/"point_times.json", timings)
    return dict(path=str(directory), manifest_sha256=original.sha(directory/"manifest.json"),
                point_times_sha256=original.sha(directory/"point_times.json"))


def expected_pairs(protocol):
    pairs = []
    for case in protocol["cases"]:
        for full in ("full2025", "full2026", "full2027"):
            for baseline in METHODS:
                pairs.append(dict(case=case["id"], reference=baseline, candidate=full, scope="same_input_pipeline"))
        for off in (x for x in case["models"] if not x.startswith("full")):
            pairs.append(dict(case=case["id"], reference=off, candidate="full2025", scope="training_SELECT_CAL_pipeline_not_acute"))
    return pairs


def evaluation_metadata(protocol, descriptors, seal):
    """No truth file I/O. Whole-wave exact bank list, aliases not extra methods."""
    expected = [(c["id"], m) for c in protocol["cases"] for m in c["models"] + list(METHODS)]
    if ([(e["case"], e["method"]) for e in descriptors] != expected
            or seal != dict(status="ACCEPTED_CONTROLLED_WAVE1_FIXED_BANKS", approved_by="root", banks=descriptors)):
        raise PermissionError("Whole fixed wave required before truth access")
    return dict(comparisons=expected_pairs(protocol), data_run_id=1,
        independent_data_replicates=False, aliases=protocol["shams"],
        semantic_aliases=protocol["semantic_identical_input_counterexamples"],
        phase_intervals={c["id"]: phases(c, protocol["history"], protocol["N"]) for c in protocol["cases"]},
        post_event_alarms_main_normal_FP=True, excluded_recovery_points=0,
        CI="NOT_COMPUTED", VUS="NOT_COMPUTED", score_ensemble=False,
        original_length=protocol["N"], original_start=0, warmup_points=protocol["history"],
        confirmed3="full original axis once; never reset at phase boundaries",
        events="full truth event catalogue; never split long event into phase events",
        alias_pairing="same-bank different-truth aliases diagnostic only; no paired_events across truth identities",
        all_normal="original evaluator denominator policy; ranking/Recall/Aff NA; Precision/F1 zero iff FP exists, otherwise NA",
        time_units="original samples; false alarm exposure per1000 valid points; no physical clock")


def dispatch_evaluation(metadata, arrays, full_labels, threshold, evaluator, affiliation_identity=None):
    """Array metadata only: defer every metric/NA/confirm calculation to old math.

    Caller must already have passed ALL banks/scalars and a separate root label
    release. Tests inject an explicit fake evaluator, not a metric implementation.
    No phase slicing, reindexing, NA replacement or alias event pairing occurs.
    """
    y = np.asarray(full_labels)
    if (y.shape != (metadata["original_length"],) or not np.isin(y, [0, 1]).all()
            or not np.issubdtype(y.dtype, np.integer)):
        raise ValueError("Full original binary truth axis required, never warmup-trimmed")
    return evaluator.evaluate_arrays(arrays["index"], arrays["score"], y, threshold,
        bank_complete=True, original_length=metadata["original_length"],
        original_start=metadata["original_start"], warmup_points=metadata["warmup_points"],
        valid=arrays["valid"], available_index=arrays["available_index"],
        invalid_reasons=arrays["invalid_reasons"], affiliation_identity=affiliation_identity)


def _source_gate(cfg, signed):
    """Explicit closure, never a glob of concurrently changing working files."""
    sources = cfg["source_sha256"]
    if signed.get("source_sha256") != sources or not sources:
        raise PermissionError("Exact approved source closure required")
    for path, digest in sources.items():
        if original.sha(path) != digest:
            raise PermissionError("Pinned implementation changed")
    if sources.get(str(Path(__file__).resolve())) != original.sha(__file__):
        raise PermissionError("Running adapter must belong to signed closure")
    source = Path(cfg["frozen_source"]["directory"])
    closure = cfg["frozen_source"]["sha256_map"]
    if signed.get("frozen_source") != cfg["frozen_source"] or len(closure) != 23:
        raise PermissionError("Original complete23 frozen kernel map required")
    for name, digest in closure.items():
        if Path(name).name != name or original.sha(source/name) != digest:
            raise PermissionError("Frozen numerical package changed")
    for name in ("scoring.py", "chunked_bank.py"):
        actual = Path(__file__).resolve().parent/name
        if sources.get(str(actual)) != closure[name] or original.sha(actual) != closure[name]:
            raise PermissionError("Metadata math/reader must be pinned to the original frozen SHA")
    if original.sha(original.__file__) != ORIGINAL_SHA:
        raise PermissionError("Original004b must remain byte-identical")


def _input_metadata(cfg, protocol):
    """No NPZ/truth body is opened at admission. Hash only the bound bytes."""
    manifest = bound(cfg["input_manifest"])
    accepted = bound(cfg["input_acceptance"])
    if (accepted.get("approved_by") != "root" or accepted.get("status") != "ACCEPTED_CONTROLLED_WAVE1_INPUTS"
            or accepted.get("protocol_sha256") != PROTOCOL_SHA
            or accepted.get("manifest_sha256") != cfg["input_manifest"]["sha256"]):
        raise PermissionError("Typed approved controlled-input lineage required")
    parent = Path(cfg["input_manifest"]["path"]).resolve().parent
    rows = manifest.get("runs", [])
    if len(rows) != 11:
        raise PermissionError("Exactly the11 fixed input runs required")
    for entry, case in zip(rows, protocol["cases"]):
        expected = dict(domain=DOMAIN, entity=case["family"], run=case["id"], role="QUERY",
            path=case["id"]+".npz", format="npz", key="values", observed_key="observed",
            bounds=[0, 2560], original_start=0, expected_shape=[2560, case["C"]])
        if {k: v for k, v in entry.items() if k != "sha256"} != expected:
            raise PermissionError("QUERY role, original axis or fixed case changed")
        file = parent/entry["path"]
        if file.resolve().parent != parent or original.sha(file) != entry["sha256"]:
            raise PermissionError("Bound case input identity changed")
    return manifest


def _model_metadata(cfg):
    """Six-asset source pipeline, exact own config/recipe/CAL; no PT body read."""
    from .scoring import recipe
    if cfg.get("model_id") not in MODEL_IDS or cfg.get("route") != "SOURCE_CONTROL":
        raise PermissionError("Only the seven already-trained source pipelines")
    expected_seed = int(cfg["model_id"][-4:])
    if cfg.get("source_seed") != expected_seed:
        raise PermissionError("Model ID and source seed must agree")
    assets = cfg["assets"]
    if set(assets) != SCIENCE:
        raise PermissionError("Exact source six-asset schema required")
    seal = bound(cfg["asset_seal"])
    binding = dict(model_id=cfg["model_id"], source_seed=cfg["source_seed"], assets=assets)
    if (seal.get("status") != "ACCEPTED_CONTROLLED_WAVE1_SOURCE_ASSETS" or seal.get("approved_by") != "root"
            or seal.get("models", []).count(binding) != 1):
        raise PermissionError("Root's own named source assets required")
    for value in assets.values():
        if original.sha(value["path"]) != value["sha256"]:
            raise PermissionError("Own source asset changed")
    values = {k: bound(assets[k]) for k in SCIENCE if k != "checkpoint"}
    train, tr, cc, cr, cal = [values[k] for k in
        ("training_config", "training_receipt", "calibration_config", "calibration_receipt", "calibration")]
    for actual in (train, cc):
        if (actual.get("model") != cfg["model"] or actual.get("sources") != cfg["sources"]
                or actual.get("seed") != cfg["source_seed"]
                or actual.get("history") != 512 or actual.get("horizon") != 32
                or recipe(actual.get("active_components"))[0] != recipe(cfg["active_components"])[0]):
            raise PermissionError("Own original training/CAL config and active recipe required")
    if (tr.get("phase") != "source_train" or cr.get("phase") != "calibrate_source"
            or tr.get("synthetic_non_scientific_results") is not False
            or cr.get("synthetic_non_scientific_results") is not False
            or tr.get("files", {}).get("source.pt") != assets["checkpoint"]["sha256"]
            or tr.get("files", {}).get("config.json") != assets["training_config"]["sha256"]
            or cc.get("checkpoint_sha256") != assets["checkpoint"]["sha256"]
            or cc.get("train_receipt_sha256") != assets["training_receipt"]["sha256"]
            or cr.get("files", {}).get("source_calibration.json") != assets["calibration"]["sha256"]
            or cr.get("files", {}).get("config.json") != assets["calibration_config"]["sha256"]):
        raise PermissionError("Source training/CAL receipt chain differs")
    from . import scoring
    validate_calibration(cal, cfg["active_components"], scoring)
    if not cfg["model"].get("tf_enabled", True) and "r_tf" in cfg["active_components"]:
        raise PermissionError("Disabled TF cannot be active")
    if not cfg["model"].get("ms_enabled", True) and "r_ms" in cfg["active_components"]:
        raise PermissionError("Disabled MS cannot be active")
    return cal


def prepare_evaluation(cfg, protocol):
    """Authenticate/replay all scalar banks before returning a truth I/O ticket.

    This is a metadata adapter only: it does not load truth, compute metrics, or
    give a caller permission to do so. Future exact evaluation release is still
    required. Labels may not be opened until every required bank passes here.
    """
    from . import chunked_bank, scoring
    plan = evaluation_metadata(protocol, cfg["banks"], bound(cfg["raw_seal"]))
    for descriptor in cfg["banks"]:
        case = next(c for c in protocol["cases"] if c["id"] == descriptor["case"])
        manifest = bound(dict(path=str(Path(descriptor["directory"])/"manifest.json"),
                              sha256=descriptor["manifest_sha256"]))
        span = [dict(domain=DOMAIN, entity=case["family"], run=case["id"], start=512, end=2560)]
        if (manifest.get("complete") is not True or manifest.get("committed_records") != 2048
                or manifest["contract"]["ranges"] != span
                or manifest["contract"]["identity"] != descriptor["identity"]):
            raise PermissionError("Whole exact bank.complete/coverage/identity gate")
        cal = bound(descriptor["calibration"])
        validate_calibration(cal, descriptor["active_components"], scoring)
        if (cal["threshold"] != manifest["contract"]["threshold"]
                or descriptor["calibration"]["sha256"] != descriptor["identity"]["calibration_sha256"]):
            raise PermissionError("Own sourceCAL/threshold mismatch")
        count = 0
        for record in chunked_bank.read_records(descriptor["directory"]):
            if (record["index"] != 512+count or record["available_index"] != record["index"]
                    or (record["domain"], record["entity"], record["run"]) != (DOMAIN, case["family"], case["id"])):
                raise ValueError("Fixed original arrived scalar axis required")
            validate_active(record, descriptor["active_components"])
            value = scoring.apply_calibration(record, cal)
            if not value["valid"] or value["alarm"] != record["alarm"] or not np.isclose(value["score"], record["score"], rtol=1e-12, atol=1e-12):
                raise ValueError("Original source scalar score/alarm replay mismatch")
            count += 1
        if count != 2048:
            raise ValueError("Full2048 raw records required before truth")
    return dict(status="METADATA_AND_SCALARS_PASSED_TRUTH_NOT_OPENED", plan=plan,
                root_evaluation_release_still_required=True)


def admit(config, release, trusted):
    """No arrays/model/truth until exact action/config/source and fresh checks."""
    if not release or not trusted:
        raise PermissionError("Separate exact scientific root release required")
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    signed = bound(dict(path=release, sha256=trusted))
    op = cfg.get("operation")
    if (op not in (DERIVE, MODEL, BASELINE, EVALUATE, TECHNICAL) or signed.get("status") != "APPROVED"
            or signed.get("approved_by") != "root" or signed.get("operation") != op
            or signed.get("allowed_actions") != [op] or signed.get("config_sha256") != original.sha(config)
            or signed.get("resources") != cfg.get("resources") or signed.get("allow_resume") is not False
            or cfg.get("resume") is not False):
        raise PermissionError("Wrong/unreleased action/config/resource or resume requested")
    permission = dict(generate=op == DERIVE, neural_readout=op in (MODEL, TECHNICAL),
        baseline_readout=op == BASELINE, evaluation=op == EVALUATE, labels=op == EVALUATE,
        training=False, target_fit=False, threshold_fit=False)
    if cfg.get("permissions") != permission or signed.get("permissions") != permission:
        raise PermissionError("Exact action permissions required")
    if op != EVALUATE and any(k in cfg for k in ("truth", "labels", "audit_truth", "label_path")):
        raise PermissionError("Truth/labels are never readout input")
    if Path(cfg["output"]).exists():
        raise FileExistsError("Fresh output required, never resume")
    _source_gate(cfg, signed)
    if cfg["protocol"]["sha256"] != PROTOCOL_SHA:
        raise PermissionError("Fixed root-reviewed protocol required")
    protocol = protocol_check(bound(cfg["protocol"]))
    limits = cfg.get("resources", {})
    if (type(limits.get("wall_seconds")) is not int or limits["wall_seconds"] <= 0
            or limits.get("kill_grace_seconds") != 5 or limits.get("automatic_retry") is not False):
        raise PermissionError("Positive bounded wall+5 and no retry required")
    if op in (MODEL, BASELINE):
        _input_metadata(cfg, protocol)
        case_ids = [c["id"] for c in protocol["cases"] if op == BASELINE or cfg.get("model_id") in c["models"]]
        if cfg.get("case_ids") != case_ids:
            raise PermissionError("All fixed applicable cases; no result-based subset")
    if op == TECHNICAL:
        technical_plan(cfg["technical"])
        needed = ["full2025", "tf_off2025"] if cfg["technical"]["purpose"] == "CPU_PARITY" else ["full2025"]
        if [m["model_id"] for m in cfg.get("models", [])] != needed:
            raise PermissionError("Exact technical models, not a wave/performance matrix")
        for model in cfg["models"]:
            _model_metadata(model)
        expected = (180, 16) if cfg["technical"]["purpose"] == "CPU_PARITY" else (175, 30)
        if limits["wall_seconds"] != expected[0] or limits.get("memory_GiB") != expected[1]:
            raise PermissionError("Technical CPU180+5 AS16 or GPU175+5 eval30 only")
        if cfg["reference_training"]["sha256"] != "40e2247ed3049bf9d2303c380b2da55c1494c874ca38dd24652f741ed33ccc74":
            raise PermissionError("Actual oldV4 training reference required")
        if original.sha(cfg["reference_training"]["path"]) != cfg["reference_training"]["sha256"]:
            raise PermissionError("OldV4 reference changed")
    if op in (MODEL, TECHNICAL):
        model_entries = [cfg] if op == MODEL else cfg["models"]
        if any(m.get("inference_seed") != m.get("source_seed") or m["source_seed"] not in (2025, 2026, 2027) for m in model_entries):
            raise PermissionError("Own source/mask seed required")
    if op == MODEL:
        if cfg.get("inference_seed") != cfg.get("source_seed") or cfg["source_seed"] not in (2025, 2026, 2027):
            raise PermissionError("Own source/mask seed required")
        _model_metadata(cfg)
        if (cfg.get("device") != "cuda:0" or type(cfg.get("physical_gpu")) is not int
                or cfg["physical_gpu"] not in range(4) or not str(cfg.get("expected_gpu_uuid", "")).startswith("GPU-")
                or limits["wall_seconds"] != protocol["GPU_budget_candidate"]["model_TERM_seconds"][cfg["model_id"]]
                or limits.get("evaluation_GiB") != 30):
            raise PermissionError("Fixed named model cap and exact GPU mapping required")
    if op in (MODEL, TECHNICAL):
        external = cfg.get("external_source_sha256", {})
        if len(external) != 8 or signed.get("external_source_sha256") != external:
            raise PermissionError("Actual official8 identity map required before PT loading")
        for path, digest in external.items():
            if original.sha(path) != digest:
                raise PermissionError("Official source bytes changed")
        if signed.get("installed_frft") != cfg.get("installed_frft"):
            raise PermissionError("Installed FRFT identity required")
        if original.sha(cfg["installed_frft"]["path"]) != cfg["installed_frft"]["sha256"]:
            raise PermissionError("Installed FRFT changed")
    if op == BASELINE and cfg.get("methods") != list(METHODS):
        raise PermissionError("Two deterministic baselines once each required")
    if op == BASELINE:
        from . import scoring
        if set(cfg.get("calibrations", {})) != set(METHODS):
            raise PermissionError("Two original source baselineCALs required")
        for method in METHODS:
            if cfg["calibrations"][method]["sha256"] != BASELINE_CAL[method]:
                raise PermissionError("No synthetic or target baselineCAL fitting")
            validate_calibration(bound(cfg["calibrations"][method]), ["r_pred"], scoring)
    if op == EVALUATE:
        prepare_evaluation(cfg, protocol)
    return cfg, protocol


def _frozen_modules(cfg):
    cli, data, training, scoring = original._kernels(Path(cfg["frozen_source"]["directory"]))
    bank = importlib.import_module(training.__package__ + ".chunked_bank")
    readout = importlib.import_module(training.__package__ + ".readout")
    return cli, data, training, scoring, bank, readout


def technical_plan(spec):
    if (spec.get("namespace") != "TECHNICAL_NORMAL_NOT_FORMAL_WAVE"
            or spec.get("length") != 640 or spec.get("data_seed") != 84553
            or spec.get("truth") is not False):
        raise PermissionError("Independent technical normal input, never formal case data")
    if spec.get("purpose") == "CPU_PARITY":
        return [("full2025", 1, [512]), ("tf_off2025", 1, [512]), ("full2025", 3, [512])]
    if spec.get("purpose") == "GPU_THROUGHPUT":
        return [("full2025", c, list(range(512, 576))) for c in (1, 3)]
    raise PermissionError("Only fixed CPU parity or independent GPU throughput")


def _configure_device(cfg):
    import os
    import torch
    if cfg["device"] == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise PermissionError("Explicit CPU CVD empty required")
        torch.set_num_threads(1)
    elif cfg["device"] == "cuda:0":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != str(cfg["physical_gpu"]):
            raise PermissionError("Exact physical/logical mapping required")
        props = torch.cuda.get_device_properties(0)
        if "A40" not in props.name or str(props.uuid).removeprefix("GPU-") != cfg["expected_gpu_uuid"].removeprefix("GPU-"):
            raise PermissionError("Exact A40 UUID required")
        torch.cuda.set_per_process_memory_fraction(30*1024**3/props.total_memory)
    else:
        raise PermissionError("Only CPU or exactly bound cuda0")


def assert_tree_parity(a, b):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b): raise AssertionError("Raw field mismatch")
        for key in a: assert_tree_parity(a[key], b[key])
    elif isinstance(a, (list, tuple, np.ndarray)):
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.dtype.kind in "fc" or bb.dtype.kind in "fc":
            np.testing.assert_allclose(aa, bb, atol=1e-5, rtol=2e-6, equal_nan=True)
        else:
            np.testing.assert_array_equal(aa, bb)
    elif isinstance(a, (float, np.floating)):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=2e-6, equal_nan=True)
    elif a != b:
        raise AssertionError("Original raw metadata changed")


def legacy_singleton(reference, model, runs, *, plan, active):
    """Real old40e API returns a list; never assume the newer iter_infer API."""
    values = reference.infer(model, runs, history=512, horizon=32, seed=2025,
                             active_components=active, index_plan=plan)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError("OldV4 singleton plan must return exactly one list record")
    return original.singleton_record(iter(values), 512)


def persist_evidence(directory, stage, state, cfg, cli, model):
    """Durable actual state/runtime/byte identities, not an assertion substitute."""
    if stage not in ("before", "after"):
        raise ValueError("Explicit evidence stage required")
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    dump(directory/("state_"+stage+".json"), state)
    runtime = directory/("runtime_"+stage); runtime.mkdir()
    cli.save_runtime(runtime, model)
    dump(runtime/"actual_external_bytes.json", dict(
        installed_frft=dict(path=cfg["installed_frft"]["path"], sha256=original.sha(cfg["installed_frft"]["path"])),
        external_source_sha256={path: original.sha(path) for path in cfg["external_source_sha256"]},
        model_flags={k: v for k, v in model.config.to_dict().items() if isinstance(v, bool)}))


def run_technical(cfg):
    """Later exact release only. Never formal inputs, weights training or metrics."""
    import time
    started = time.monotonic()
    import torch
    specs = technical_plan(cfg["technical"])
    cli, data, training, scoring, bank, readout = _frozen_modules(cfg)
    _configure_device(cfg)
    output = Path(cfg["output"]); output.mkdir(parents=True, exist_ok=False)
    # Load the original40e reference in the SAME already hash-verified package.
    name = training.__package__ + ".controlled_reference_v4"
    reference_spec = importlib.util.spec_from_file_location(name, cfg["reference_training"]["path"])
    reference = importlib.util.module_from_spec(reference_spec); reference_spec.loader.exec_module(reference)
    models, states = {}, {}
    for entry in cfg["models"]:
        model = cli.load_model({**cfg, **entry, "seed": 2025, "history": 512, "horizon": 32}, entry["assets"]["checkpoint"]["path"])
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        models[entry["model_id"]] = model; states[entry["model_id"]] = original._state(model)
        persist_evidence(output/"model_evidence"/entry["model_id"], "before", states[entry["model_id"]], cfg, cli, model)
    cold_model_seconds = time.monotonic()-started
    rows = []; began = time.monotonic()
    for number, (model_id, channels, indices) in enumerate(specs):
        entry = next(m for m in cfg["models"] if m["model_id"] == model_id)
        rng = np.random.Generator(np.random.PCG64(84553+channels))
        values = rng.standard_normal((640, channels)).astype(np.float32)
        run = data.Run("TECHNICAL_ONLY", "C"+str(channels), "normal_not_formal", "QUERY", values, np.ones(values.shape, bool))
        directory = output/(str(number)+"_"+model_id+"_C"+str(channels)); directory.mkdir()
        plan = [dict(domain=run.domain, entity=run.entity, run=run.run, role="QUERY", index=i) for i in indices]
        saved_anchors = []
        def sink(payload):
            saved_anchors.append(payload)
            original.save_bundle(directory, "ANCHOR_"+str(payload["index"]), payload)
        model = models[model_id]; proxy = AnchorModel(model, run, indices[:1], 2025, sink, neural_rng, same_neural_rng)
        pre = readout._rng()
        iterator = training.iter_infer(proxy, [run], history=512, horizon=32, seed=2025,
            active_components=entry["active_components"], index_plan=plan)
        records = []
        def observed_records():
            for raw in iterator:
                if cfg["technical"]["purpose"] == "CPU_PARITY": records.append(raw)
                yield raw
        identity = dict(input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
            config_sha256=cfg["execution_config_sha256"], checkpoint_sha256=entry["assets"]["checkpoint"]["sha256"],
            calibration_sha256=entry["assets"]["calibration"]["sha256"], source_sha256=original.sha(__file__))
        if cfg["device"] == "cuda:0": torch.cuda.synchronize()
        timer = time.monotonic()
        product = consume_records(observed_records(), directory/"bank", identity=identity,
            ranges=[dict(domain=run.domain, entity=run.entity, run=run.run, start=indices[0], end=indices[-1]+1)],
            calibration=bound(entry["assets"]["calibration"]), active=entry["active_components"], kernels=(cli, scoring, bank))
        if cfg["device"] == "cuda:0": torch.cuda.synchronize()
        seconds = time.monotonic()-timer; proxy.finish(len(indices)); post = readout._rng()
        parity = "NOT_REQUESTED_GPU_COST"
        if cfg["technical"]["purpose"] == "CPU_PARITY":
            readout._restore(pre)
            try:
                old = legacy_singleton(reference, model, [run], plan=plan, active=entry["active_components"])
                assert_tree_parity(records[0], old)
                if not same_neural_rng(saved_anchors[0]["RNG_after_reconstruct"], neural_rng()):
                    raise AssertionError("Old/new reference post-pass RNG mismatch")
                original.save_bundle(directory, "REFERENCE_RNG_POST", neural_rng())
                original.save_bundle(directory, "REFERENCE_RAW", old)
                parity = "PASS_OLD_V4_RAW_AND_RNG"
            finally:
                readout._restore(post)
        original.save_bundle(directory, "MAIN_RNG_POST", neural_rng())
        if original._state(model) != states[model_id]: raise AssertionError("Technical inference changed source state")
        rows.append(dict(model_id=model_id, channels=channels, records=len(indices), writer_seconds=seconds,
                         product=product, parity=parity))
    for model_id, model in models.items():
        persist_evidence(output/"model_evidence"/model_id, "after", original._state(model), cfg, cli, model)
    return dict(status="COMPLETED_TECHNICAL_ONLY_NOT_SCIENTIFIC_NOT_REUSED", rows=rows,
                cold_model_import_load_seconds=cold_model_seconds, total_wall_seconds=time.monotonic()-started,
                readout_wall_seconds=time.monotonic()-began, metrics=False, formal_data_opened=False,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if cfg["device"] == "cuda:0" else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if cfg["device"] == "cuda:0" else None)


def run_readout(cfg, protocol):
    """Future separately approved readout; tests use only explicit fake kernels."""
    cli, data, training, scoring, bank, readout = _frozen_modules(cfg)
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=False)
    runs = data.load_manifest(cfg["input_manifest"]["path"], roles={"QUERY"})
    run_map = {r.run: r for r in runs}
    for entry in protocol["cases"]:
        run = run_map[entry["id"]]
        if run.values.shape != (2560, entry["C"]) or run.start != 0 or not run.observed.all():
            raise ValueError("Exact full-observed original controlled QUERY run required")
    identity_base = dict(config_sha256=cfg["execution_config_sha256"],
        source_sha256=hashlib.sha256(json.dumps(cfg["frozen_source"], sort_keys=True).encode()).hexdigest())
    products = []
    if cfg["operation"] == MODEL:
        import torch
        from .controlled_mechanism import _state
        if cfg["device"] == "cpu":
            import os
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise PermissionError("Explicit CPU CVD empty required")
        else:
            import os
            if os.environ.get("CUDA_VISIBLE_DEVICES") != str(cfg["physical_gpu"]):
                raise PermissionError("Signed physical GPU mapping required")
            props = torch.cuda.get_device_properties(0)
            if "A40" not in props.name or str(props.uuid).removeprefix("GPU-") != cfg["expected_gpu_uuid"].removeprefix("GPU-"):
                raise PermissionError("Fresh A40 UUID required")
            torch.cuda.set_per_process_memory_fraction(30*1024**3/props.total_memory)
        model = cli.load_model(dict(cfg, seed=cfg["source_seed"], history=512, horizon=32), cfg["assets"]["checkpoint"]["path"])
        model.eval()
        for parameter in model.parameters(): parameter.requires_grad_(False)
        before = _state(model)
        persist_evidence(out, "before", before, cfg, cli, model)
        cal = bound(cfg["assets"]["calibration"])
        for case in protocol["cases"]:
            if case["id"] not in cfg["case_ids"]: continue
            run = run_map[case["id"]]; directory = out/case["id"]
            directory.mkdir()
            selected = anchors(case, 512, 2560)
            def sink(payload):
                original.validate_lag_axes(payload["forecast_lag"], cfg["model"], case["C"])
                for axes in payload["reconstruction_lag"]:
                    original.validate_lag_axes(axes, cfg["model"], case["C"])
                original.save_bundle(directory, "ANCHOR_"+str(payload["index"]), payload)
            original.save_bundle(directory, "INITIAL_RNG", original.detached(neural_rng()))
            proxy = AnchorModel(model, run, selected, cfg["inference_seed"], sink,
                                neural_rng, same_neural_rng)
            iterator = training.iter_infer(proxy, [run], history=512, horizon=32,
                seed=cfg["inference_seed"], active_components=cfg["active_components"])
            input_file = next(x for x in bound(cfg["input_manifest"])["runs"] if x["run"] == case["id"])
            identity = dict(identity_base, input_sha256=input_file["sha256"],
                checkpoint_sha256=cfg["assets"]["checkpoint"]["sha256"],
                calibration_sha256=cfg["assets"]["calibration"]["sha256"])
            span = [dict(domain=run.domain, entity=run.entity, run=run.run, start=512, end=2560)]
            products.append(consume_records(iterator, directory/"bank", identity=identity, ranges=span,
                calibration=cal, active=cfg["active_components"], kernels=(cli, scoring, bank)))
            proxy.finish(2048)
            original.save_bundle(directory, "FINAL_RNG", original.detached(neural_rng()))
        if before != _state(model): raise AssertionError("Frozen model state changed")
        persist_evidence(out, "after", _state(model), cfg, cli, model)
    else:
        for case in protocol["cases"]:
            for method in METHODS:
                cal = bound(cfg["calibrations"][method]); validate_calibration(cal, ["r_pred"], scoring)
                identity = dict(identity_base, input_sha256=next(x for x in bound(cfg["input_manifest"])["runs"]
                    if x["run"] == case["id"])["sha256"],
                    checkpoint_sha256=hashlib.sha256(method.encode()).hexdigest(),
                    calibration_sha256=cfg["calibrations"][method]["sha256"])
                path = out/(case["id"]+"__"+method)
                readout.write_baseline_bank([run_map[case["id"]]], method, cal, path,
                    identity=identity, history=512, chunk_size=64, resume=False)
                products.append(dict(path=str(path), manifest_sha256=original.sha(path/"manifest.json")))
    return dict(status="COMPLETED_CONTROLLED_LABEL_FREE_READOUT", products=products,
                aliases_rerun=False, training=False, labels_opened=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--release")
    parser.add_argument("--trusted-release-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(preview=True, arrays_generated=False, model_loaded=False)))
        return
    cfg, protocol = admit(args.config, args.release, args.trusted_release_sha256)
    if args.preflight_only:
        print(json.dumps(dict(metadata_preflight="PASS", model_loaded=False, labels_opened=False)))
        return
    cfg["execution_config_sha256"] = original.sha(args.config)
    if cfg["operation"] == DERIVE:
        report = derive_inputs(protocol, cfg["output"])
    elif cfg["operation"] in (MODEL, BASELINE):
        report = run_readout(cfg, protocol)
    elif cfg["operation"] == TECHNICAL:
        report = run_technical(cfg)
    else:
        raise PermissionError("Evaluation metadata adapter only; separate exact evaluator binding required")
    dump(Path(cfg["output"])/"report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
