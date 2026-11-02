"""Phase-separated U training / target-only calibration; never query or labels.

This module only assembles existing kernels. A root SHA-pinned operation release
is required even for synthetic execution; a pending template is not permission.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .cli import load_model, save_json, save_model, save_runtime
from .data import load_manifest, resolve_plan, BalancedSampler
from .training import adapt, iter_infer, calibrate_route, UPDATE_U

PHASES = {"target_train": {"FIT", "SELECT"}, "target_calibrate": {"CAL"}}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound(config, key):
    path = Path(config[key])
    if sha256(path) != config[key+"_sha256"]:
        raise ValueError("Changed bound asset: "+key)
    return path


def _permissions(phase):
    return {"target_FIT_SELECT": phase == "target_train", "target_CAL": phase == "target_calibrate",
            "source_training": False, "query_or_labels": False, "F": False}


def _no_labels(value):
    if isinstance(value, dict):
        if any("label" in k.lower() for k in value):
            raise PermissionError("Label fields are forbidden in target metadata")
        for child in value.values():
            _no_labels(child)
    elif isinstance(value, list):
        for child in value:
            _no_labels(child)


def metadata_preflight(config, phase, output):
    """Read metadata/checkpoint bytes only, not target array bodies or Torch."""
    if phase not in PHASES or config.get("operation") != "HYBRID_V1_STAGE2_U":
        raise PermissionError("Only explicit U phases are supported; no F/query")
    if config.get("permissions") != _permissions(phase):
        raise PermissionError("Exact phase-only permissions required")
    if config.get("mode") not in {"U", "Z_TCAL"} or (phase == "target_train" and config["mode"] != "U"):
        raise ValueError("Z_TCAL is calibration-only")
    if config.get("scope") not in {"synthetic", "public_target"}:
        raise ValueError("Explicit synthetic/public_target scope required")
    if Path(output).exists() or Path(output).resolve() != Path(config["output"]).resolve():
        raise PermissionError("Exact fresh output required")
    if config["scope"] == "synthetic" and config.get("device") != "cpu":
        raise PermissionError("Synthetic proof is CPU-only")
    if config["scope"] == "public_target" and (
            config["history"] != 512 or config["horizon"] != 32 or config["seed"] != 2025
            or config["steps"] != 200 or config["validate_every"] != 50
            or config["select_count"] != 32 or config["cal_count"] != 512
            or config["lr"] != 1e-4 or config["weight_decay"] != .01 or config["kind"] != "huber"
            or config.get("active_components") != ["r_tf", "r_ms", "r_pred"]):
        raise ValueError("Fixed U200 public-target recipe changed")
    for key in ("manifest", "plans", "source_config", "source_checkpoint", "source_receipt"):
        _bound(config, key)
    source_config = json.loads(Path(config["source_config"]).read_text(encoding="utf-8"))
    for key in ("model", "sources", "history", "horizon", "active_components", "seed"):
        if source_config.get(key) != config.get(key):
            raise ValueError("Accepted source architecture/recipe/seed changed: "+key)
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    _no_labels(manifest)
    entries = manifest["runs"]
    identities = set()
    for entry in entries:
        if entry["role"] not in {"FIT", "SELECT", "CAL"} or any(
                "label" in k.lower() for k in entry):
            raise PermissionError("Target manifest cannot expose query/support/labels")
        key = tuple(entry[k] for k in ("domain", "entity", "run", "role"))
        if key in identities:
            raise ValueError("Unique run/role identities required")
        identities.add(key)
        if not entry.get("sha256") or not entry.get("expected_shape"):
            raise ValueError("Target bytes and full shape must be bound")
        # Explicit original observation masks, never inferred from LOCF values.
        if entry.get("format") != "npz" or not entry.get("observed_key") or not entry.get("key"):
            raise ValueError("Bound NPZ values + explicit boolean observation mask required")
        if type(entry.get("original_start", 0)) is not int:
            raise ValueError("Integer original axis required")
    for i, a in enumerate(entries):
        for b in entries[i+1:]:
            if tuple(a[k] for k in ("domain", "entity", "run")) == tuple(b[k] for k in ("domain", "entity", "run")):
                alo, ahi = [v+a.get("original_start", 0) for v in a["bounds"]]
                blo, bhi = [v+b.get("original_start", 0) for v in b["bounds"]]
                gap = max(blo-ahi, alo-bhi)
                if gap < config["history"]+config["horizon"]:
                    raise ValueError("Original role ranges require H+P guard")
    roles = PHASES[phase]
    if not roles.issubset({e["role"] for e in entries}):
        raise ValueError("Missing current phase target role")
    plans = json.loads(Path(config["plans"]).read_text(encoding="utf-8"))
    _no_labels(plans)
    if set(plans) != {"history", "horizon", "seed", "SELECT", "CAL"}:
        raise ValueError("Only SELECT/CAL target plans are supported")
    if any(plans[k] != config[k] for k in ("history", "horizon", "seed")):
        raise ValueError("Plan H/P/seed mismatch")
    role = "SELECT" if phase == "target_train" else "CAL"
    points = plans[role]
    if len(points) != config["select_count" if role == "SELECT" else "cal_count"] or any(
            e["role"] != role or type(e["index"]) is not int for e in points):
        raise ValueError("Wrong fixed observation-only plan")
    keys = [tuple(e[k] for k in ("domain", "entity", "run", "role", "index")) for e in points]
    if len(set(keys)) != len(keys):
        raise ValueError("Fixed target plans must be unique")
    if phase == "target_calibrate" and config["mode"] == "U":
        _bound(config, "adapted_checkpoint"); receipt = _bound(config, "adapted_receipt")
        trained = json.loads(receipt.read_text(encoding="utf-8"))
        if (trained.get("status") != "COMPLETED" or trained.get("phase") != "target_train"
                or trained.get("source_checkpoint_sha256") != config["source_checkpoint_sha256"]
                or trained.get("manifest_sha256") != config["manifest_sha256"]
                or trained.get("plans_sha256") != config["plans_sha256"]
                or trained.get("adapted_checkpoint_sha256") != config["adapted_checkpoint_sha256"]):
            raise ValueError("Adapted checkpoint provenance mismatch")
    return {"metadata_preflight": "PASS", "phase": phase, "loaded_target_roles": sorted(roles),
            "target_arrays_opened": False, "query_or_labels": False, "training_executed": False}


def approval(config_path, release_path, trusted_sha, phase, output):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not release_path or not trusted_sha:
        raise PermissionError("Exact root phase release required")
    raw = Path(release_path).read_bytes(); release = json.loads(raw)
    own = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    if (hashlib.sha256(raw).hexdigest() != trusted_sha or release.get("approved_by") != "root"
            or release.get("status") != "APPROVED" or release.get("allowed_actions") != [phase]
            or release.get("config_sha256") != sha256(config_path) or release.get("source_sha256") != own):
        raise PermissionError("Pending/changed phase/source/config release")
    return config, metadata_preflight(config, phase, output)


def state_identity(model):
    """Exact bytes for every parameter and buffer (including negative zero)."""
    return {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                   "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name, value in model.state_dict().items()}


def state_difference(before, after, parameter_names, train=False):
    if before.keys() != after.keys():
        raise ValueError("State keys changed")
    changed = [name for name in before if before[name] != after[name]]
    allowed = {n for n in parameter_names if n.split(".")[0] in UPDATE_U} if train else set()
    forbidden = sorted(set(changed)-allowed)
    report = {"changed_parameters": [n for n in changed if n in parameter_names],
              "changed_buffers": [n for n in changed if n not in parameter_names],
              "forbidden_changes": forbidden, "allowed_parameter_names": sorted(allowed),
              "before": before, "after": after, "PASS": not forbidden}
    return report


def sampler_trace(runs, config):
    """Independent replay of unchanged local-RNG U sampler, including skips."""
    sampler = BalancedSampler(runs, config["history"], config["horizon"], config["seed"], role="FIT")
    rows = []
    for attempt in range(config["steps"]):
        batch = sampler.sample()
        rows.append({"attempt": attempt, **{k: batch[k] for k in
            ("domain", "entity", "run", "role", "index", "footprint")},
            "actual_mask_seed": config["seed"]+attempt})
    return rows


def _phase_action(config, phase, output):
    """Private action; public run_phase always verifies the exact release first."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save_json(output/"config.json", config)
    save_json(output/"bound_inputs.json", {key: {"path": config[key], "sha256": config[key+"_sha256"]}
        for key in ("manifest", "plans", "source_config", "source_checkpoint", "source_receipt")})
    roles = PHASES[phase]
    runs = load_manifest(config["manifest"], roles=roles)
    if {r.role for r in runs} != roles:
        raise ValueError("Only current phase roles may be loaded")
    save_json(output/"input_runs.json", [{"domain": r.domain, "entity": r.entity, "run": r.run,
        "role": r.role, "start": r.start, "shape": list(r.values.shape),
        "values_sha256": hashlib.sha256(r.values.tobytes()).hexdigest(),
        "observed_sha256": hashlib.sha256(r.observed.tobytes()).hexdigest()} for r in runs])
    plan = json.loads(Path(config["plans"]).read_text(encoding="utf-8"))
    plan_role = "SELECT" if phase == "target_train" else "CAL"
    resolved = resolve_plan([r for r in runs if r.role == plan_role], plan[plan_role],
                            config["history"], config["horizon"], arrived=plan_role == "CAL")
    for run, t in resolved:
        if not run.observed[t].all() or not (run.observed[t-config["history"]:t].sum(0) >= 2).all():
            raise ValueError("Fixed plan no longer observation-eligible; never redraw")
    path = config["source_checkpoint"] if phase == "target_train" or config["mode"] == "Z_TCAL" else config["adapted_checkpoint"]
    model = load_model(config, path)
    save_runtime(output, model)
    before = state_identity(model); parameter_names = set(dict(model.named_parameters()))
    source_flags = {n: p.requires_grad for n, p in model.named_parameters()}
    if phase == "target_train":
        # adapt deep-copies source; fit's original seeded BalancedSampler is used.
        replay = sampler_trace([r for r in runs if r.role == "FIT"], config)
        save_json(output/"FIT_sampler_replay.json", replay)
        adapted, trace = adapt(model, "U", [r for r in runs if r.role == "FIT"],
            [r for r in runs if r.role == "SELECT"], history=config["history"], horizon=config["horizon"],
            steps=config["steps"], lr=config["lr"], kind=config["kind"], seed=config["seed"],
            active_components=config.get("active_components"), validate_every=config["validate_every"],
            weight_decay=config["weight_decay"], validation_plan=plan["SELECT"], artifact_dir=output)
        for row, expected in zip(trace["steps"], replay):
            if row["updated"] and any(row[k] != expected[k] for k in ("domain", "run", "footprint", "actual_mask_seed")):
                raise ValueError("Actual fit trace disagrees with original sampler replay")
        if len(trace["steps"]) != len(replay):
            raise ValueError("Sampler attempt trace is incomplete")
        save_json(output/"training.json", trace)
        source_diff = state_difference(before, state_identity(model), parameter_names)
        diff = state_difference(before, state_identity(adapted), parameter_names, train=True)
        source_diff["requires_grad_unchanged"] = source_flags == {n: p.requires_grad for n, p in model.named_parameters()}
        expected_trainable = sorted(n for n in parameter_names if n.split(".")[0] in UPDATE_U)
        if sorted(trace["updated_names"]) != expected_trainable or not source_diff["requires_grad_unchanged"]:
            raise ValueError("U update policy/source flags changed")
        save_json(output/"source_state_difference.json", source_diff)
        save_json(output/"adapted_state_difference.json", diff)
        if not source_diff["PASS"] or not diff["PASS"]:
            raise ValueError("Source or frozen parameter/buffer changed")
        save_model(output/"adapted.pt", adapted)
        extra = {"adapted_checkpoint_sha256": sha256(output/"adapted.pt"), "updates": trace["updates"],
                 "selected_attempt": trace["selected_attempt"], "FIT_sampler": "original BalancedSampler(seed); no fixed sampling_plan",
                 "target_fit": True}
    else:
        records = list(iter_infer(model, runs, history=config["history"], horizon=config["horizon"],
            seed=config["seed"], active_components=config.get("active_components"), index_plan=plan["CAL"]))
        # U means target source=False calibration, also for frozen-model Z_TCAL.
        calibration = calibrate_route(records, "U", active_components=config.get("active_components"))
        diff = state_difference(before, state_identity(model), parameter_names)
        save_json(output/"calibration_state_difference.json", diff)
        if not diff["PASS"]:
            raise ValueError("Calibration changed model state")
        save_json(output/"target_calibration_records.json", records)
        save_json(output/"target_calibration.json", calibration)
        extra = {"updates": 0, "target_fit": False, "target_calibration_contact": True,
                 "pure_source_only_Z": False, "calibration_route_kernel": "U/source=False",
                 "normal_1percent_FPR_claim": False, "calibrated_checkpoint_sha256": sha256(path)}
    if sha256(config["source_checkpoint"]) != config["source_checkpoint_sha256"]:
        raise ValueError("Original source checkpoint bytes changed")
    receipt = {"status": "COMPLETED", "phase": phase, "scope": config["scope"], "mode": config["mode"],
        "source_checkpoint_sha256": config["source_checkpoint_sha256"], "manifest_sha256": config["manifest_sha256"],
        "plans_sha256": config["plans_sha256"], "query_or_labels": False, "loaded_roles": sorted(roles), **extra}
    save_json(output/"receipt.json", receipt)
    return receipt


def run_phase(config_path, release_path, trusted_sha, phase, output):
    config, _ = approval(config_path, release_path, trusted_sha, phase, output)
    result = _phase_action(config, phase, output)
    save_json(Path(output)/"execution_identity.json", {"config_sha256": sha256(config_path),
        "release_sha256": trusted_sha, "release": json.loads(Path(release_path).read_text(encoding="utf-8")),
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--phase", choices=sorted(PHASES), required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--release"); parser.add_argument("--trusted-release-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        result = run_phase(args.config, args.release, args.trusted_release_sha256, args.phase, args.output)
    else:
        result = metadata_preflight(json.loads(Path(args.config).read_text(encoding="utf-8")), args.phase, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
