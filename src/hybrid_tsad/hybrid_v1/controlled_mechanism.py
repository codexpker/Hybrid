"""Controlled-input sidecar. Original model/score kernels are never rewritten.

Default CLI is metadata preview. A separate exact CPU-forward release is needed
for execute; the engineering-only release cannot authorize it. Torch is lazy.
"""
import argparse
from contextlib import contextmanager
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

OPERATION = "HYBRID_V1_CONTROLLED_MECHANISM_FROZEN_CPU_L0"
FAMILIES = ("weak_amplitude", "self_lag", "cross_mixed_sign")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(family, role, run_id, stream):
    # Variant/output suffixes deliberately do not participate in latent seeds.
    if family not in FAMILIES or type(run_id) is not int or run_id < 0:
        raise ValueError("Fixed family and integer latent run ID required")
    text = "MECH1|{}|{}|{}|{}".format(family, role, run_id, stream)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def generate_pair(family, *, run_id, role, n, burn_in, event, d0=12, d1=20):
    """Small fixtures or separately approved L0 only; truth is a separate return.

    A pair shares exact innovations/latent paths. Self interventions keep the
    recurrence state continuous, so recovery is not claimed identical to N.
    """
    s, e = event
    if (type(n) is not int or type(burn_in) is not int or not 0 <= s < e <= n
            or not 0 < d0 < d1 or burn_in < d1):
        raise ValueError("Legal noncircular positive-lag geometry required")
    length = burn_in + d1 + n
    seeds, innovations, latents = {}, {}, {}

    def innovation(name):
        seeds[name] = seed_for(family, role, run_id, name)
        rng = np.random.Generator(np.random.PCG64(seeds[name]))
        innovations[name] = rng.standard_normal(length)
        return innovations[name]

    def ar1(name):
        eps = innovation(name)
        q = np.empty(length, np.float64)
        q[0] = eps[0]
        for t in range(1, length):
            q[t] = .8 * q[t - 1] + .6 * eps[t]
        latents[name] = q
        return q

    index = burn_in + d1 + np.arange(n)
    normal_lag = np.full(n, d0, np.int64)
    changed_lag = normal_lag.copy()
    changed_lag[s:e] = d1
    if family == "weak_amplitude":
        normal = ar1("y")[index, None]
        anomaly = normal.copy()
        anomaly[s:e, 0] += .25
    elif family == "self_lag":
        eps = innovation("self")
        initial = innovation("initial")
        qn = np.empty(length, np.float64)
        qa = np.empty(length, np.float64)
        qn[:d1] = initial[:d1]
        qa[:d1] = initial[:d1]
        start, end = int(index[s]), int(index[e - 1]) + 1
        for t in range(d1, length):
            qn[t] = .8 * qn[t - d0] + .6 * eps[t]
            delay = d1 if start <= t < end else d0
            qa[t] = .8 * qa[t - delay] + .6 * eps[t]
        latents.update(normal_self=qn, changed_self=qa)
        normal, anomaly = qn[index, None], qa[index, None]
    elif family == "cross_mixed_sign":
        u, zp, zm = ar1("u"), ar1("zplus"), ar1("zminus")
        def observed(lag):
            return np.column_stack((u[index], .3 * u[index - lag] + np.sqrt(.91) * zp[index],
                                    -.3 * u[index - lag] + np.sqrt(.91) * zm[index]))
        normal, anomaly = observed(normal_lag), observed(changed_lag)
    else:
        raise ValueError("Unknown fixed mechanism family")
    labels = np.zeros(n, np.int8)
    labels[s:e] = 1
    observations = {name: {"values": x.astype(np.float32), "observed": np.ones(x.shape, bool)}
                    for name, x in (("N", normal), ("A", anomaly))}
    if not all(np.isfinite(x["values"]).all() for x in observations.values()):
        raise ValueError("Nonfinite synthetic values")
    truth = {"seeds": seeds, "innovations": innovations, "latents": latents,
             "retained_indices": index, "normal_lag": normal_lag, "changed_lag": changed_lag,
             "labels_A": labels, "event": [s, e], "role": role, "latent_run_id": run_id}
    return observations, truth


def detached(value):
    """Copy only: no random draws, extra forward, rounding or scientific math."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [detached(v) for v in value]
    return value


def save_bundle(directory, name, tree):
    """Small L0 NPZ + shape-preserving JSON; not a future full-query bank."""
    directory = Path(directory)
    npz, schema = directory / (name + ".npz"), directory / (name + ".json")
    if npz.exists() or schema.exists():
        raise FileExistsError("Immutable bundle already exists")
    arrays = {}
    def pack(x):
        if isinstance(x, (np.ndarray, np.generic)):
            key = "array{:05d}".format(len(arrays)); arrays[key] = np.asarray(x)
            return {"npz_key": key, "dtype": str(arrays[key].dtype), "shape": list(arrays[key].shape)}
        if isinstance(x, dict):
            return {k: pack(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [pack(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x):
            return {"nonfinite_scalar": str(x)}
        return x
    metadata = pack(tree)
    with npz.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with schema.open("x", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)


def singleton_record(iterator, expected_index):
    first = next(iterator)
    if first["index"] != expected_index:
        raise ValueError("Wrong singleton arrival index")
    try:
        next(iterator)
    except StopIteration:
        return first
    raise ValueError("Reference yielded more than the single approved point")


@contextmanager
def count_calls(model):
    """Observation-only call counters; restore exact instance lookup in finally."""
    counts = {"forecast": 0, "reconstruct": 0, "_encode": 0}
    saved = {}
    for name in counts:
        saved[name] = (name in model.__dict__, model.__dict__.get(name), getattr(model, name))
        def counted(*args, _name=name, **kwargs):
            counts[_name] += 1
            return saved[_name][2](*args, **kwargs)
        setattr(model, name, counted)
    try:
        yield counts
    finally:
        for name, (had, original, _) in saved.items():
            if had:
                setattr(model, name, original)
            else:
                delattr(model, name)


def _path(root, name):
    p = Path(name)
    return p if p.is_absolute() else root / p


def admit(config_path, release_path, trusted, workspace):
    """Metadata-only gate before array generation, PT identity read or Torch."""
    if not release_path or not trusted:
        raise PermissionError("Separate exact root CPU-forward release required")
    raw = Path(release_path).read_bytes()
    release = json.loads(raw)
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if (hashlib.sha256(raw).hexdigest() != trusted or release.get("status") != "APPROVED"
            or release.get("approved_by") != "root" or release.get("operation") != OPERATION
            or release.get("allowed_actions") != ["cpu_l0"]
            or release.get("config_sha256") != sha(config_path)
            or release.get("sidecar_sha256") != sha(__file__)
            or release.get("source_sha256") != cfg["model_source"]["sha256_map"]):
        raise PermissionError("Unapproved or mismatched root CPU-forward binding")
    permission = cfg.get("permissions", {})
    if (cfg.get("status") != "ROOT_BOUND" or cfg.get("operation_proposed") != OPERATION
            or permission.get("prepare_only") is not False
            or permission.get("generate_arrays_now") is not True
            or permission.get("forward_now") is not True
            or any(permission.get(k) is not False for k in
                   ("training", "optimizer", "threshold_fit", "real_query_or_labels", "GPU"))):
        raise PermissionError("Only the fixed synthetic frozen CPU L0 is permitted")
    if (cfg.get("history"), cfg.get("horizon"), cfg.get("source_seed"), cfg.get("inference_seed"),
            cfg.get("L0_max_model_evaluation_units")) != (512, 32, 2025, 2025, 7):
        raise ValueError("Original L0 dimensions/seed/count required")
    expected = [{"family": f, "variant": v, "index": 2048} for f in FAMILIES for v in ("N", "A")]
    plan = [{"domain": "CONTROLLED_MECHANISM", "entity": "cross_mixed_sign", "run": "run0_N",
             "role": "QUERY", "index": 2048}]
    gen = cfg["generator"]
    if (cfg["L0_endpoint_plan"] != expected or cfg["L0_parity_index_plan"] != plan
            or cfg["L0_parity_Run_start"] != 0 or cfg["L0_parity_actual_seed"] != 4073
            or cfg["L0_parity_group_seeds"] != [4073, 4074] or cfg["L0_parity_groups"] != 2
            or (gen["N"], gen["burn_in"], gen["run_id"], gen["event_half_open"]) != (4096, 4096, 0, [1536, 2560])):
        raise ValueError("Exactly six fixed units and one singleton reference required")
    if cfg["resources_candidate"] != release.get("resources"):
        raise PermissionError("Exact independently approved CPU limits required")
    out = _path(Path(workspace), cfg["output_proposed"])
    if out.exists():
        raise FileExistsError("Output must be fresh; no automatic retry")
    source = _path(Path(workspace), cfg["model_source"]["path"]) / "hybrid_tsad" / "hybrid_v1"
    for name, expected_sha in cfg["model_source"]["sha256_map"].items():
        if sha(source / name) != expected_sha:
            raise PermissionError("Frozen source mismatch: " + name)
    return cfg, out, source


def _kernels(source):
    """Actual original package in a separate namespace, never working-tree math."""
    source = source.resolve()
    prefix = "_controlled_frozen_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    if prefix not in sys.modules:
        spec = importlib.util.spec_from_file_location(prefix, str(source / "__init__.py"),
                                                      submodule_search_locations=[str(source)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[prefix] = module
        spec.loader.exec_module(module)
    elif Path(sys.modules[prefix].__file__).resolve() != source / "__init__.py":
        raise RuntimeError("Frozen namespace location mismatch")
    modules = tuple(importlib.import_module(prefix + "." + name)
                    for name in ("cli", "data", "training", "scoring"))
    for module in modules:
        if Path(module.__file__).resolve().parent != source:
            raise RuntimeError("Non-frozen numerical kernel imported")
    return modules


def _rng(torch):
    return {"torch": torch.random.get_rng_state().clone(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def _restore_rng(torch, state):
    torch.random.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _rng_equal(a, b):
    return (np.array_equal(detached(a["torch"]), detached(b["torch"]))
            and a["numpy"][0] == b["numpy"][0]
            and np.array_equal(a["numpy"][1], b["numpy"][1])
            and a["numpy"][2:] == b["numpy"][2:] and a["python"] == b["python"])


def _state(model):
    return {name: {"shape": list(x.shape), "dtype": str(x.dtype),
                   "sha256": hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name, x in model.state_dict().items()}


def _unpack(x):
    out = np.unpackbits(np.frombuffer(bytes.fromhex(x["packbits_hex"]), np.uint8), bitorder="little")
    a = out[:int(np.prod(x["shape"]))].astype(bool).reshape(x["shape"])
    if hashlib.sha256(a.tobytes()).hexdigest() != x["unpacked_sha256"]:
        raise ValueError("Reference packed mask identity mismatch")
    return a


def validate_lag_axes(meta, cfg, channels):
    """Legal/finite VALID entries only; keep invalid NaN/-Inf raw, never fill."""
    maximum, history = cfg["max_lag"], cfg["history"]
    expected_shape = (1, channels, channels, maximum + 1)
    score, overlap = meta["correlation"], meta["overlap_count"]
    if score.shape != expected_shape or overlap.shape != expected_shape:
        raise AssertionError("Wrong explicit B/i/j/lag axes")
    if (meta["target_axis"], meta["source_axis"], meta["lag_unit"], meta["circular"]) != ("i", "j", "original_sample", False):
        raise AssertionError("Wrong directional/noncircular axis metadata")
    np.testing.assert_array_equal(meta["source_sample_offset"], history - 1 - np.arange(maximum + 1))
    eye = np.eye(channels, dtype=bool)[None, :, :, None]
    lags = np.arange(maximum + 1)[None, None, None]
    for branch in ("self", "cross"):
        valid = meta[branch + "_candidate_valid"]
        if valid.dtype != np.bool_ or valid.shape != expected_shape:
            raise AssertionError("Candidate mask must be bool with i/j/lag axes")
        legal = (eye & (lags > 0)) if branch == "self" else np.broadcast_to(~eye, expected_shape)
        if branch == "cross" and cfg.get("cross_synchronous_only", False):
            legal = legal & (lags == 0)
        if np.any(valid & ~legal):
            raise AssertionError("Illegal self/cross candidate")
        if (not np.isfinite(score[valid]).all() or not np.isfinite(overlap[valid]).all()
                or np.any(overlap[valid] < cfg["lag_min_overlap"])
                or np.any(overlap[valid] > np.broadcast_to(cfg["lag_window"] - lags, expected_shape)[valid])):
            raise AssertionError("Invalid finite/overlap support on valid candidates")
        route = meta[branch]
        source, delay, selected, weights = [route[k] for k in ("source", "lag", "selected_valid", "weights")]
        if (source.shape != delay.shape or selected.shape != source.shape or weights.shape != source.shape
                or source.shape[:2] != (1, channels) or source.dtype.kind not in "iu"
                or delay.dtype.kind not in "iu" or selected.dtype != np.bool_):
            raise AssertionError("Wrong selected-route shape/type")
        if np.any((source < 0) | (source >= channels) | (delay < 0) | (delay > maximum)):
            raise AssertionError("Selected source/lag index outside original axes")
        expected_valid = np.take_along_axis(valid.reshape(1, channels, -1), source * (maximum + 1) + delay, -1)
        np.testing.assert_array_equal(selected, expected_valid)
        np.testing.assert_array_equal(route["valid"], selected.any(-1))
        if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights[~selected] != 0):
            raise AssertionError("Invalid route weights or nonzero invalid-route weight")
        np.testing.assert_allclose(weights.sum(-1), selected.any(-1).astype(float), atol=1e-6, rtol=1e-6)
        if channels == 1 and branch == "cross" and valid.any():
            raise AssertionError("C1 cannot have a valid Cross candidate")
    return {"status": "PASS_VALID_DIRECTIONAL_AXES", "channels": channels,
            "self_enabled": cfg.get("self_lag_enabled", True),
            "cross_enabled": cfg.get("cross_lag_enabled", True),
            "cross_synchronous_only": cfg.get("cross_synchronous_only", False),
            "invalid_scores_preserved": int((~np.isfinite(score)).sum())}


def validate_components(parts):
    for name in ("r_tf", "r_ms", "r_pred"):
        if (not np.isfinite(parts["scalar"][name]).all() or np.any(parts["scalar"][name] < 0)
                or not np.isfinite(parts["per_channel"][name]).all()
                or np.any(parts["per_channel"][name] < 0)):
            raise AssertionError("Fixed fully observed L0 required component invalid: " + name)


def compare_reference(direct, reference):
    """Serialization parity only; same original kernel and fixed tolerances."""
    def close(a, b):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=2e-6, equal_nan=True)
    f, r, parts, batch = (direct[k] for k in ("forecast", "reconstruction", "parts", "batch"))
    if reference["index"] != batch["index"] or reference["replay"]["actual_mask_seed"] != direct["actual_seed"]:
        raise AssertionError("Reference index/seed mismatch")
    if reference["available_index"] != batch["available_index"] or reference["forecast_issued_index"] != batch["index"] - 1:
        raise AssertionError("Arrival/forecast issue index mismatch")
    close(f["prediction"][0], reference["prediction"])
    for name, value in parts["scalar"].items():
        close(value[0], reference[name])
    for name, value in parts["per_channel"].items():
        close(value[0], reference["per_channel"][name])
    replay = reference["replay"]
    history, index = batch["history"].shape[1], batch["index"]
    if (replay["input_footprint"] != batch["footprint"]
            or replay["forecast_history_bounds"] != [index-history, index]
            or replay["reconstruction_bounds"] != [index-history+1, index+1]
            or replay["group_mask_seeds"] != [direct["actual_seed"] + g for g in range(r["hidden_mask"].shape[1])]
            or replay["base_seed"] != direct["actual_seed"] - index
            or replay["unarrived_future_observed"] is not False
            or replay["input_identity"] != direct["input_identity"]):
        raise AssertionError("Original footprint/input/group-seed identity mismatch")
    for key, value in {"hidden_mask": r["hidden_mask"], "visible_mask": r["visible_mask"],
                       "copy_masks": r["transform_hidden_mask"], "forecast_copy_masks": f["copy_masks"],
                       "current_observed": batch["current_observed"], "forecast_valid": f["valid"],
                       "tf_endpoint_valid": r["tf_valid"][:, :, :, -1],
                       "ms_endpoint_valid_by_scale": r["ms_valid_by_scale"][:, :, :, :, -1]}.items():
        np.testing.assert_array_equal(value, _unpack(replay[key]))
    for key, value in {"order": r["order"], "forecast_order": f["tf_order"],
                       "current_target": batch["current_target"],
                       "tf_endpoint_by_group_copy": r["tf_reconstruction"][:, :, :, -1],
                       "ms_endpoint_by_group_copy_scale": r["ms_reconstruction_by_scale"][:, :, :, :, -1]}.items():
        close(value, replay[key])
    for group, values in (("stats", r["stats"]), ("forecast_stats", f["stats"])):
        for name, value in values.items():
            if value.dtype == np.bool_:
                np.testing.assert_array_equal(value, replay[group][name])
            else:
                close(value, replay[group][name])
    return {"status": "PASS_ORIGINAL_SINGLETON_NUMERICAL_INTERFACE", "atol": 1e-5, "rtol": 2e-6}


def run_l0(cfg, output, source, workspace):
    """Future separately released CPU action; engineering tests never call it."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU action requires explicit CUDA_VISIBLE_DEVICES empty before Torch")
    checkpoint, calibration_path = cfg["checkpoint"], cfg["calibration"]
    for asset in (checkpoint, calibration_path):
        if sha(_path(Path(workspace), asset["path"])) != asset["sha256"]:
            raise PermissionError("Accepted frozen asset identity mismatch")
    cli, data, training, scoring = _kernels(source)
    import torch
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic(); success = False
    kernel = {"model": cfg["model"], "sources": cfg["sources"], "history": 512, "horizon": 32,
              "seed": 2025, "device": "cpu", "active_components": ["r_tf", "r_ms", "r_pred"]}
    model = cli.load_model(kernel, str(_path(Path(workspace), checkpoint["path"])))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before_state = _state(model)
    cal = json.loads(_path(Path(workspace), calibration_path["path"]).read_text())
    cli.save_runtime(output, model)
    runs, truths = {}, {}
    for family in FAMILIES:
        observations, truth = generate_pair(family, run_id=0, role=cfg["generator"]["role"],
            n=4096, burn_in=4096, event=(1536, 2560))
        truths[family] = truth
        for variant, observation in observations.items():
            runs[(family, variant)] = data.Run("CONTROLLED_MECHANISM", family, "run0_" + variant,
                                               "QUERY", observation["values"], observation["observed"])
    save_bundle(output, "AUDIT_TRUTH_NOT_MODEL_INPUT", truths)
    save_bundle(output, "OBSERVATIONS_ONLY", {f + "/" + v: {"values": run.values, "observed": run.observed}
                                             for (f, v), run in runs.items()})
    raw_units, rng_units = [], []
    reference_direct = reference_pre = reference_post = None
    counts = {}
    try:
        with count_calls(model) as counts, torch.no_grad():
            for entry in cfg["L0_endpoint_plan"]:
                run = runs[(entry["family"], entry["variant"])]; t = entry["index"]
                batch = training.tensor_batch(data.arrived_window(run, t, 512, 32), "cpu")
                pre = _rng(torch)
                forecast, reconstruction = training.passes(model, batch, 2025 + t)
                parts = scoring.residual_components(forecast, reconstruction, batch)
                post_compute = _rng(torch)
                raw = detached({"batch": batch, "forecast": forecast, "reconstruction": reconstruction,
                                "parts": parts, "actual_seed": 2025 + t, "identity": entry})
                raw["model_flags"] = {k: v for k, v in model.config.to_dict().items() if isinstance(v, bool)}
                raw["input_identity"] = {"values_float32_sha256": hashlib.sha256(run.values.tobytes()).hexdigest(),
                    "observed_sha256": hashlib.sha256(run.observed.tobytes()).hexdigest(),
                    "start": run.start, "shape": list(run.values.shape)}
                if not _rng_equal(post_compute, _rng(torch)):
                    raise AssertionError("Diagnostic export changed RNG")
                raw["lag_validation"] = {"forecast": validate_lag_axes(raw["forecast"]["axes"]["lag"], cfg["model"], run.values.shape[1]),
                    "reconstruction_groups": [validate_lag_axes(a["lag"], cfg["model"], run.values.shape[1])
                                               for a in raw["reconstruction"]["axes"]]}
                if len(raw["lag_validation"]["reconstruction_groups"]) != min(2, run.values.shape[1]):
                    raise AssertionError("Wrong hidden group count")
                validate_components(raw["parts"])
                raw["source_score"] = scoring.apply_calibration({k: float(v[0]) for k, v in raw["parts"]["scalar"].items()}, cal)
                raw_units.append(raw); rng_units.append(detached({"pre": pre, "post": post_compute}))
                save_bundle(output, "DIRECT_UNIT_{}".format(len(raw_units)), raw)
                save_bundle(output, "DIRECT_RNG_{}".format(len(raw_units)), rng_units[-1])
                if (entry["family"], entry["variant"]) == ("cross_mixed_sign", "N"):
                    reference_direct, reference_pre, reference_post = raw, pre, post_compute
            main_final = _rng(torch)
            try:
                _restore_rng(torch, reference_pre)
                run = runs[("cross_mixed_sign", "N")]
                plan = cfg["L0_parity_index_plan"]
                resolved = data.resolve_plan([run], plan, 512, 32, arrived=True)
                if len(resolved) != 1 or resolved[0][1] != 2048:
                    raise ValueError("Reference must resolve exactly one local point2048")
                record = singleton_record(training.iter_infer(model, [run], history=512, horizon=32,
                    seed=2025, index_plan=plan), 2048)
                save_bundle(output, "RAW_SINGLETON_REFERENCE", record)
                if not _rng_equal(reference_post, _rng(torch)):
                    raise AssertionError("Original iterator and direct post-call RNG differ")
                parity = compare_reference(reference_direct, record)
                if scoring.apply_calibration(record, cal) != reference_direct["source_score"]:
                    raise AssertionError("Original singleton source score/alarm mismatch")
            finally:
                _restore_rng(torch, main_final)
            if counts != {"forecast": 7, "reconstruct": 7, "_encode": 17}:
                raise AssertionError("Actual public/internal forward count exceeded fixed plan: " + repr(counts))
        if _state(model) != before_state or sha(_path(Path(workspace), checkpoint["path"])) != checkpoint["sha256"]:
            raise AssertionError("Frozen source state/checkpoint changed")
        save_bundle(output, "RAW_SIX_UNITS", raw_units)
        save_bundle(output, "RNG_REPLAY", {"units": rng_units, "main_final": detached(main_final)})
        report = {"status": "COMPLETED_LIMITED_TECHNICAL_L0_NOT_MECHANISM_EVIDENCE", "parity": parity,
                  "counts": counts, "source_state_unchanged": True, "source_checkpoint_unchanged": True,
                  "seconds": time.monotonic() - began, "GPUh": 0, "metrics_computed": False,
                  "torch": str(torch.__version__), "numpy": np.__version__}
        success = True
        return report
    finally:
        save_bundle(output, "STATUS_AND_STATE", {"success": success, "counts": counts,
            "state_before": before_state, "state_after": _state(model),
            "checkpoint": checkpoint, "calibration": cfg["calibration"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--release")
    parser.add_argument("--trusted-release-sha256")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"preview": True, "model_loaded": False, "arrays_generated": False,
                          "CPU_forward_requires_separate_root_release": True}))
        return
    workspace = Path(os.environ["HYBRID_V1_WORKSPACE"])
    cfg, output, source = admit(args.config, args.release, args.trusted_release_sha256, workspace)
    if args.preflight_only:
        print(json.dumps({"metadata_preflight": "PASS", "PT_opened": False, "Torch_imported": False}))
        return
    report = run_l0(cfg, output, source, workspace)
    with (output / "report.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
