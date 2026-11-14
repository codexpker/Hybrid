"""Independent matched-U interface. Legacy training/model/scoring stay unchanged.

Only plain SSL is implemented. A future robust objective must retain this
role/plan/mode/endpoint interface and receive its own versioned release.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import numpy as np

OPERATION = "HYBRID_V1_MATCHED_U_ENDPOINT"
ALLOWED = ("fusion", "tf_fusion_delta", "ms_decoder", "forecast_head")


def exposure_status(updates, attempts):
    if not 0 <= updates <= attempts or attempts < 1:
        raise ValueError("Invalid update accounting")
    return ("NO_VALID_UPDATES" if updates == 0 else
            "PARTIAL_UPDATE_EXPOSURE" if updates < attempts else "VALID_ALL_ATTEMPTS_UPDATED")


def append_trace(path, row):
    """Append-only forensic evidence; caller failures never produce a receipt."""
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
        f.flush()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def no_labels(value):
    if isinstance(value, dict):
        if any(any(s in k.lower() for s in ("label", "event_id", "anomaly_weight")) for k in value):
            raise PermissionError("Labels/event annotations cannot enter the U interface")
        for v in value.values():
            no_labels(v)
    elif isinstance(value, list):
        for v in value:
            no_labels(v)


def role_metadata(manifest, history=512, horizon=32):
    """Metadata only: every role is a distinct physical NPZ, never shared keys."""
    p = Path(manifest)
    d = json.loads(p.read_text())
    no_labels(d)
    paths, identities, physical = set(), set(), []
    for e in d["runs"]:
        if set(e) != {"domain", "entity", "run", "role", "path", "format", "key", "observed_key",
                      "sha256", "expected_shape", "bounds", "original_start"}:
            raise PermissionError("Only explicit role-file metadata fields are allowed")
        if e["role"] not in {"FIT", "SELECT", "CAL", "QUERY"}:
            raise PermissionError("Unsupported U role")
        path = Path(e["path"])
        path = path if path.is_absolute() else p.parent / path
        key = str(path.resolve())
        ident = tuple(e[k] for k in ("domain", "entity", "run", "role"))
        if key in paths or ident in identities:
            raise PermissionError("Role files and identities must be physically distinct")
        paths.add(key); identities.add(ident)
        for previous in physical:
            if path.exists() and previous.exists() and path.samefile(previous):
                raise PermissionError("Hardlink aliases cannot bypass physical role separation")
        physical.append(path)
        if (e.get("format") != "npz" or e.get("key") != "values"
                or e.get("observed_key") != "observed" or not e.get("sha256")
                or len(e["expected_shape"]) != 2 or min(e["expected_shape"]) < 1
                or e["bounds"] != [0, e["expected_shape"][0]]
                or type(e.get("original_start")) is not int):
            raise ValueError("A complete standalone role array and explicit original axis are required")
    for i, a in enumerate(d["runs"]):
        for b in d["runs"][i+1:]:
            if a["role"] != b["role"] and tuple(a[k] for k in ("domain", "entity", "run")) == tuple(b[k] for k in ("domain", "entity", "run")):
                alo, ahi = a["original_start"], a["original_start"] + a["expected_shape"][0]
                blo, bhi = b["original_start"], b["original_start"] + b["expected_shape"][0]
                if max(blo-ahi, alo-bhi) < history + horizon:
                    raise ValueError("Original same-record roles require H+P guard")
    return d


def load_roles(manifest, roles, history=512, horizon=32):
    """Open/hash/decode only allowed role files; never decode other role members."""
    from .data import Run
    d = role_metadata(manifest, history, horizon); out = []; p = Path(manifest)
    for e in d["runs"]:
        if e["role"] not in roles:
            continue
        path = Path(e["path"])
        path = path if path.is_absolute() else p.parent / path
        if sha(path) != e["sha256"]:
            raise ValueError("Changed role bytes")
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"values", "observed"}:
                raise PermissionError("Role NPZ must contain only its values and observed arrays")
            x, m = z["values"], z["observed"]
        if (x.dtype != np.float32 or m.dtype != np.bool_ or list(x.shape) != e["expected_shape"]
                or m.shape != x.shape):
            raise ValueError("Role dtype/shape mismatch")
        out.append(Run(e["domain"], e["entity"], e["run"], e["role"], x, m, e["original_start"]))
    if {r.role for r in out} != set(roles):
        raise ValueError("Missing requested roles")
    return out


def resolve_u_plan(runs, entries, history, horizon, role, *, arrived=False):
    from .data import resolve_plan
    no_labels(entries)
    if any(set(e) - {"domain", "entity", "run", "role", "index", "footprint"} for e in entries):
        raise PermissionError("Plans contain only axes, never training labels or weights")
    if not entries or any(e["role"] != role for e in entries) or any(r.role != role for r in runs):
        raise PermissionError("Explicit matching plan/run role required")
    # Repetition is legal for FIT exposure; SELECT/CAL must remain unique.
    if role != "FIT":
        keys = [tuple(e[k] for k in ("domain", "entity", "run", "role", "index")) for e in entries]
        if len(set(keys)) != len(keys):
            raise ValueError("SELECT/CAL plans must be unique")
    return resolve_plan(runs, entries, history, horizon, arrived=arrived)


def adaptation_mode(model):
    """Reapply after every model.train(): root masks train; frozen children eval."""
    model.train()
    for name, child in model.named_children():
        child.train(name in ALLOWED)


def rng_devices(model):
    devices = {p.device.index for p in model.parameters() if p.device.type == "cuda"}
    return sorted(devices)


@contextmanager
def evaluation_state(model, seed):
    import torch
    states = [(m, m.training) for m in model.modules()]
    devices = rng_devices(model)
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            # Seed only CPU and actually used CUDA devices, not all visible GPUs.
            torch.random.default_generator.manual_seed(int(seed))
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            yield
    finally:
        # train(old) would destroy heterogeneous child modes; restore every flag.
        for module, flag in states:
            module.training = flag


def validate(model, runs, plan, history, horizon, seed, active_components=None):
    from .training import tensor_batch, passes
    from .data import window
    from .objectives import ssl_objective
    resolved = resolve_u_plan(runs, plan, history, horizon, "SELECT")
    losses = []
    with evaluation_state(model, seed):
        for run, t in resolved:
            batch = tensor_batch(window(run, t, history, horizon), next(model.parameters()).device)
            f, r = passes(model, batch, seed + t)
            obj = ssl_objective(f, r, batch, "huber", active_components)
            if bool(obj["valid_windows"].all()):
                losses.append(float(obj["total"]))
    if not losses or not np.isfinite(losses).all():
        raise ValueError("No finite SELECT diagnostic")
    return float(np.mean(losses))


def train_endpoint(source, fit_runs, select_runs, plan, *, history, horizon,
                   seed, attempts=200, validate_every=50, lr=1e-4,
                   weight_decay=.01, active_components=None, scope="public_target", method="plain", artifact_dir=None):
    """Consume every fixed attempt once; SELECT diagnoses, never chooses weights."""
    if method != "plain":
        raise NotImplementedError("No robust formula has been implemented or approved")
    if (scope not in {"synthetic", "public_target"} or type(attempts) is not int or attempts < 1
            or type(validate_every) is not int or validate_every < 1):
        raise ValueError("Explicit finite U recipe required")
    if scope == "public_target" and (attempts != 200 or history != 512 or horizon != 32
                                     or validate_every != 50 or len(plan["SELECT"]) != 32):
        raise ValueError("Public U is the fixed 200-attempt endpoint protocol")
    if len(plan["FIT"]) != attempts:
        raise ValueError("FIT plan must cover all attempts, including skips")
    no_labels(plan)
    if any(plan.get(k) != v for k, v in (("history", history), ("horizon", horizon), ("seed", seed))):
        raise ValueError("Common plan H/P/seed must match the caller exactly")
    resolved = resolve_u_plan(fit_runs, plan["FIT"], history, horizon, "FIT")
    resolve_u_plan(select_runs, plan["SELECT"], history, horizon, "SELECT")
    import torch
    from .training import update_policy, tensor_batch, passes
    from .data import window
    from .objectives import ssl_objective
    from .stage2_adaptation import state_identity, state_difference
    model = copy.deepcopy(source)
    initial = state_identity(source)
    names = update_policy(model, "U")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    trace = []
    trace_path = None
    if artifact_dir is not None:
        trace_path = Path(artifact_dir) / "training_trace.jsonl"
        trace_path.open("x", encoding="utf-8").close()
    for attempt, (run, t) in enumerate(resolved):
        adaptation_mode(model)
        optimizer.zero_grad(set_to_none=True)
        mask_seed = seed + attempt
        batch = tensor_batch(window(run, t, history, horizon), next(model.parameters()).device)
        if trace_path is not None:
            append_trace(trace_path, dict(attempt=attempt, status="ATTEMPT_STARTED", index=run.start+t,
                                         actual_mask_seed=mask_seed, footprint=batch["footprint"]))
        f, r = passes(model, batch, mask_seed)
        obj = ssl_objective(f, r, batch, "huber", active_components)
        row = dict(attempt=attempt, actual_mask_seed=mask_seed, domain=run.domain,
                   entity=run.entity, run=run.run, index=run.start + t, footprint=batch["footprint"], updated=False)
        if bool(obj["valid_windows"].all()):
            if not bool(torch.isfinite(obj["total"])):
                raise ValueError("Nonfinite U objective")
            obj["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            row.update(updated=True, loss=float(obj["total"].detach()), gradient_norm=float(norm))
        else:
            row["reason"] = "NO_VALID_GT_NO_REDRAW"
        # Fixed attempt clock also runs after a skipped attempt, without redraw.
        if (attempt + 1) % validate_every == 0 or attempt == attempts - 1:
            row["SELECT_diagnostic"] = validate(model, select_runs, plan["SELECT"], history, horizon, seed, active_components)
        trace.append(row)
        if trace_path is not None:
            append_trace(trace_path, dict(row, status="ATTEMPT_COMPLETED"))
    parameters = set(dict(model.named_parameters()))
    delta = state_difference(initial, state_identity(model), parameters, train=True)
    if not delta["PASS"] or state_identity(source) != initial:
        raise ValueError("Source/frozen parameter or buffer mutated")
    updates = sum(x["updated"] for x in trace)
    return model, dict(method=method, attempts=attempts, updates=updates,
                       scientific_training_status=exposure_status(updates, attempts),
                       selection="FIXED_ATTEMPT_ENDPOINT_NO_BEST_CHECKPOINT", endpoint_attempt=attempts-1,
                       updated_names=names, trace=trace, state_difference=delta,
                       source_unchanged=True, mode_policy="root_train_frozen_children_eval_allowed_children_train"), optimizer


def calibrate(model, runs, plan, *, history, horizon, seed, active_components=None):
    from .training import iter_infer, calibrate_route
    from .stage2_adaptation import state_identity
    resolve_u_plan(runs, plan, history, horizon, "CAL", arrived=True)
    before = state_identity(model)
    with evaluation_state(model, seed):
        records = list(iter_infer(model, runs, history=history, horizon=horizon,
                                 seed=seed, index_plan=plan, active_components=active_components))
    if before != state_identity(model):
        raise ValueError("CAL changed parameters or buffers")
    return records, calibrate_route(records, "U", active_components=active_components)


def admit(config_path, release_path, trusted):
    c = json.loads(Path(config_path).read_text()); no_labels(c)
    if sha(release_path) != trusted:
        raise PermissionError("Changed root release")
    rel = json.loads(Path(release_path).read_text())
    if (rel.get("approved_by") != "root" or rel.get("status") != "APPROVED"
            or rel.get("operation") != OPERATION or rel.get("allowed_actions") != [c["action"]]
            or rel.get("config_sha256") != sha(config_path) or rel.get("source_sha256") != c["source_sha256"]
            or c["action"] not in {"train_endpoint", "calibrate"}):
        raise PermissionError("Exact root operation/config/action required")
    if Path(c["output"]).exists():
        raise PermissionError("Fresh output required")
    if (c.get("operation") != OPERATION or c.get("scope") != "public_target"
            or c.get("method") != "plain" or c.get("attempts") != 200
            or c.get("history") != 512 or c.get("horizon") != 32):
        raise PermissionError("Only explicitly bound public endpoint-U200 is a CLI action")
    if c["source_sha256"].get(str(Path(__file__).resolve())) != sha(__file__):
        raise PermissionError("Executing adapter itself must be explicitly SHA-bound")
    for p, h in c["source_sha256"].items():
        if sha(p) != h:
            raise PermissionError("Changed frozen source")
    for key in ("manifest", "plan", "model_config", "checkpoint"):
        if sha(c[key]) != c[key + "_sha256"]:
            raise PermissionError("Changed bound input")
    original = json.loads(Path(c["model_config"]).read_text())
    if any(original.get(k) != c[k] for k in ("history", "horizon", "seed", "active_components")):
        raise ValueError("Source model recipe and matched run seed must agree")
    role_metadata(c["manifest"], c["history"], c["horizon"])
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "release", "trusted-release-sha256"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--execute", action="store_true"); a = p.parse_args()
    c = admit(a.config, a.release, a.trusted_release_sha256)
    if not a.execute:
        print(json.dumps(dict(metadata="PASS", Torch_loaded=False, arrays_opened=False))); return
    from .cli import load_model, save_json, save_runtime
    import torch
    roles = {"FIT", "SELECT"} if c["action"] == "train_endpoint" else {"CAL"}
    runs = load_roles(c["manifest"], roles, c["history"], c["horizon"])
    plan = json.loads(Path(c["plan"]).read_text()); no_labels(plan)
    model = load_model(json.loads(Path(c["model_config"]).read_text()), c["checkpoint"])
    out = Path(c["output"]); out.mkdir(parents=True, exist_ok=False)
    save_runtime(out, model); save_json(out / "config.json", c)
    kwargs = {k: c[k] for k in ("history", "horizon", "seed", "active_components")}
    if c["action"] == "train_endpoint":
        torch.save(model.state_dict(), out / "initial.pt")
        try:
            adapted, report, opt = train_endpoint(model, [r for r in runs if r.role == "FIT"],
                [r for r in runs if r.role == "SELECT"], plan, **kwargs,
                attempts=c["attempts"], scope=c["scope"], method=c["method"], artifact_dir=out)
        except BaseException as error:
            save_json(out/"FAILURE.json", dict(status="FAILED_NO_RETRY", exception_type=type(error).__name__,
                error=str(error), complete_receipt_written=False))
            raise
        torch.save(adapted.state_dict(), out / "adapted.pt")
        import random
        torch.save(dict(state_dict=adapted.state_dict(), optimizer=opt.state_dict(),
                        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                        python_rng=random.getstate(),
                        cuda_rng={d: torch.cuda.get_rng_state(d) for d in rng_devices(adapted)},
                        resume_implemented=False), out / "final_training_state.pt")
        save_json(out / "training.json", report)
    else:
        records, calibration = calibrate(model, runs, plan["CAL"], **kwargs)
        save_json(out / "calibration_records.json", records); save_json(out / "calibration.json", calibration)
    save_json(out / "receipt.json", dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT", action=c["action"],
        config_sha256=sha(a.config), release_sha256=a.trusted_release_sha256, query_or_labels=False,
        scientific_training_status=report["scientific_training_status"] if c["action"] == "train_endpoint" else "ZERO_UPDATE_CAL",
        updates=report["updates"] if c["action"] == "train_endpoint" else 0,
        checkpoint_sha256=sha(out / "adapted.pt") if c["action"] == "train_endpoint" else c["checkpoint_sha256"],
        parent_source_checkpoint_sha256=c.get("parent_source_checkpoint_sha256", c["checkpoint_sha256"]),
        plan_sha256=c["plan_sha256"], manifest_sha256=c["manifest_sha256"], seed=c["seed"]))


if __name__ == "__main__":
    main()
