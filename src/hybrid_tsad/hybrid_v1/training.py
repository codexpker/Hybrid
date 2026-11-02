"""Shared Stage1/Z/U/F kernels. All branches start from independent state copies."""
import copy
import numpy as np
from .data import BalancedSampler, window, targets, arrived_window, arrived_targets, audit_events
from .objectives import ssl_objective
from .scoring import residual_components, fit_calibration, apply_calibration, COMPONENTS

UPDATE_U = ("fusion", "tf_fusion_delta", "ms_decoder", "forecast_head")
UPDATE_F = UPDATE_U + ("event_head",)


def tensor_batch(batch, device="cpu"):
    import torch
    return {k: torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v for k, v in batch.items()}


def passes(model, batch, seed):
    prediction = model.forecast(batch["history"], batch["observed"], mask_seed=seed)
    reconstruction = model.reconstruct(batch["reconstruction_context"], batch["reconstruction_observed"], mask_seed=seed)
    return prediction, reconstruction


def update_policy(model, route):
    if route not in {"STAGE1", "Z", "U", "F"}:
        raise ValueError("Unknown adaptation route")
    allow = UPDATE_F if route == "F" else UPDATE_U
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(route == "STAGE1" or (route != "Z" and name.split(".")[0] in allow))
    return [name for name, p in model.named_parameters() if p.requires_grad]


def validation(model, runs, history, horizon, kind, seed, limit=4, active_components=None):
    import torch
    old = model.training; model.eval()
    values = []
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        for run in runs:
            for target in targets(run, history, horizon)[:limit]:
                batch = tensor_batch(window(run, int(target), history, horizon), next(model.parameters()).device)
                f, r = passes(model, batch, seed+int(target))
                loss = ssl_objective(f, r, batch, kind, active_components)
                if bool(loss["valid_windows"].all()):
                    values.append(float(loss["total"]))
    model.train(old)
    if not values:
        raise ValueError("No legal SELECT supervision")
    return float(np.mean(values))


def fit(model, fit_runs, select_runs, *, history, horizon, steps=2, lr=1e-3,
        kind="half_mse", seed=17, route="STAGE1", support=None, active_components=None):
    import torch
    names = update_policy(model, route)
    if route == "Z":
        return model, {"route": "Z", "updates": 0, "target_fit": False, "updated_names": [], "steps": []}
    if not names or not select_runs:
        raise ValueError("Trainable policy and independent SELECT required")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sampler = BalancedSampler(fit_runs, history, horizon, seed, role="SUPPORT" if route == "F" else "FIT")
    rng = np.random.default_rng(seed)
    event_audit = None
    if route == "F":
        event_audit = audit_events(support["events"], support["normal"], support["forbidden"], support["K"], history, horizon)
        if support["K"] == 0:
            raise ValueError("K0 must route to Z before fitting")
    device = next(model.parameters()).device
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    log, best, chosen = [], float("inf"), None
    seen_events, seen_normal = set(), set()
    selected_attempt, selected_seen_events, selected_seen_normal = None, [], []
    initial = copy.deepcopy(model.state_dict())
    for attempt in range(steps):
        model.train(); optimizer.zero_grad(set_to_none=True)
        label = None; event_id = None
        if route == "F":
            if steps < 2*support["K"]:
                raise ValueError("F needs exposure to each budgeted event and matched normal")
            positive = attempt % 2 == 0; i = (attempt//2) % support["K"]
            entry = support["events"][i] if positive else support["normal"][i]
            run = next(x for x in fit_runs if (x.entity, x.run) == (entry["entity"], entry["run"]))
            target = int(rng.integers(entry["start"], entry["end"]))-run.start
            batch = tensor_batch(window(run, target, history, horizon), device)
            for blocked in support["forbidden"]:
                if ((blocked["entity"], blocked["run"]) == (run.entity, run.run)
                        and batch["footprint"][0] < blocked["end"] and blocked["start"] < batch["footprint"][1]):
                    raise ValueError("Full support footprint overlaps held-out role")
            label, event_id = float(positive), support["events"][i]["event_id"]
        else:
            batch = tensor_batch(sampler.sample(), device)
        f, r = passes(model, batch, seed+attempt)
        objective = ssl_objective(f, r, batch, kind, active_components)
        if not bool(objective["valid_windows"].all()):
            log.append({"attempt": attempt, "updated": False, "reason": "NO_VALID_GT"}); continue
        loss = objective["total"]
        bce = None
        normal_ssl = None
        normal_footprint = None
        if route == "F":
            components = residual_components(f, r, batch, active_components)
            features = components["event_features"]
            if not bool(torch.isfinite(features).all()):
                log.append({"attempt": attempt, "updated": False, "reason": "INVALID_EVENT_FEATURES"}); continue
            logit = model.event_head(features).flatten()
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.full_like(logit, label))
            normal = support["normal"][i]
            nr = next(x for x in fit_runs if (x.entity, x.run) == (normal["entity"], normal["run"]))
            nb = tensor_batch(window(nr, int(rng.integers(normal["start"], normal["end"]))-nr.start, history, horizon), device)
            normal_footprint = nb["footprint"]
            for blocked in support["forbidden"]:
                if ((blocked["entity"], blocked["run"]) == (nr.entity, nr.run)
                        and nb["footprint"][0] < blocked["end"] and blocked["start"] < nb["footprint"][1]):
                    raise ValueError("Additional normal SSL full footprint overlaps held-out role")
            nf, nrec = passes(model, nb, seed+attempt+10000)
            normal_ssl = ssl_objective(nf, nrec, nb, kind, active_components)
            if not bool(normal_ssl["valid_windows"].all()):
                continue
            loss = bce+.1*normal_ssl["total"]
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite supervised objective")
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        if route == "F" and label == 1.:
            seen_events.add(event_id)
        if route == "F" and label == 0.:
            seen_normal.add(event_id)
        value = validation(model, select_runs, history, horizon, kind, seed, active_components=active_components)
        required = {e["event_id"] for e in support["events"]} if route == "F" else set()
        eligible = route != "F" or (seen_events == required and seen_normal == required)
        if eligible and value < best:
            best, chosen = value, copy.deepcopy(model.state_dict())
            selected_attempt, selected_seen_events = attempt, sorted(seen_events)
            selected_seen_normal = sorted(seen_normal)
        log.append({"attempt": attempt, "updated": True, "loss": float(loss.detach()), "SELECT_ssl": value,
                    "gradient_norm": float(norm), "parts": {k: float(v.detach()) for k, v in objective["parts"].items()},
                    "valid_counts": {k: v.tolist() for k, v in objective["valid_counts"].items()},
                    "natural_missing": int(objective["natural_missing_count"]),
                    "artificial_hidden_observed": int(objective["artificial_hidden_observed_count"]),
                    "domain": batch["domain"], "run": batch["run"], "footprint": batch["footprint"],
                    "label": label, "event_id": event_id, "BCE": float(bce.detach()) if bce is not None else None,
                    "selection_eligible": eligible, "seen_event_ids": sorted(seen_events),
                    "seen_matched_normal_ids": sorted(seen_normal), "actual_mask_seed": seed+attempt,
                    "normal_ssl_mask_seed": seed+attempt+10000 if normal_ssl is not None else None,
                    "normal_ssl_total": float(normal_ssl["total"].detach()) if normal_ssl is not None else None,
                    "normal_ssl_parts": {k: float(v.detach()) for k, v in normal_ssl["parts"].items()} if normal_ssl is not None else None,
                    "normal_ssl_valid_counts": {k: v.tolist() for k, v in normal_ssl["valid_counts"].items()} if normal_ssl is not None else None,
                    "normal_ssl_footprint": normal_footprint})
    if chosen is None:
        model.load_state_dict(initial); raise ValueError("No valid optimizer updates")
    if route == "F" and {x["event_id"] for x in log if x.get("updated") and x.get("label") == 1.} != {e["event_id"] for e in support["events"]}:
        raise ValueError("Not every budgeted event received a valid supervised update")
    model.load_state_dict(chosen)
    changes = {n: float((p.detach()-before[n]).abs().max()) for n, p in model.named_parameters()}
    return model, {"route": route, "updates": sum(x["updated"] for x in log), "steps": log,
                   "updated_names": names, "parameter_changes": changes, "event_budget_audit": event_audit,
                   "selected_attempt": selected_attempt, "selected_seen_event_ids": selected_seen_events,
                   "selected_seen_matched_normal_ids": selected_seen_normal,
                   "attempt_indexing": "zero_based", "selected_normal_SELECT": best,
                   "target_fit": route in {"U", "F"}}


def adapt(source, route, fit_runs=(), select_runs=(), *, support=None, **kwargs):
    model = copy.deepcopy(source)
    if route == "F" and support["K"] == 0:
        route = "Z"
    return fit(model, fit_runs, select_runs, route=route, support=support, **kwargs)


def infer(model, runs, *, history, horizon, seed=17, event_logits=False, active_components=None):
    import torch
    import hashlib
    model.eval(); records = []
    def raw(value):
        return value.detach().cpu().tolist()
    def packed(value):
        array = value.detach().cpu().numpy().astype(bool)
        return {"shape": list(array.shape), "dtype": "bool", "bitorder": "little",
                "packbits_hex": np.packbits(array.reshape(-1), bitorder="little").tobytes().hex(),
                "unpacked_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    with torch.no_grad():
        for run in runs:
            run_identity = {"values_float32_sha256": hashlib.sha256(run.values.tobytes()).hexdigest(),
                            "observed_sha256": hashlib.sha256(run.observed.tobytes()).hexdigest(),
                            "start": run.start, "shape": list(run.values.shape)}
            for t in arrived_targets(run, history):
                batch = tensor_batch(arrived_window(run, int(t), history, horizon), next(model.parameters()).device)
                actual_seed = seed+int(t)
                f, r = passes(model, batch, actual_seed)
                parts = residual_components(f, r, batch, active_components)
                record = {key: float(value[0]) for key, value in parts["scalar"].items()}
                record.update(domain=run.domain, entity=run.entity, run=run.run, index=batch["index"],
                    available_index=batch["available_index"], forecast_issued_index=batch["index"]-1,
                    prediction=f["prediction"][0].cpu().tolist(),
                    per_channel={k: v[0].cpu().tolist() for k, v in parts["per_channel"].items()})
                # Compact original endpoint bank plus deterministic whole-pass
                # reconstruction identity. No future/label read to obtain it.
                record["replay"] = {"base_seed": seed, "actual_mask_seed": actual_seed,
                    "group_mask_seeds": [actual_seed+g for g in range(r["hidden_mask"].shape[1])],
                    "input_identity": run_identity, "input_footprint": batch["footprint"],
                    "forecast_history_bounds": [batch["index"]-history, batch["index"]],
                    "reconstruction_bounds": [batch["index"]-history+1, batch["index"]+1],
                    "current_target": raw(batch["current_target"]), "current_observed": packed(batch["current_observed"]),
                    "hidden_mask": packed(r["hidden_mask"]), "visible_mask": packed(r["visible_mask"]),
                    "copy_masks": packed(r["transform_hidden_mask"]), "forecast_copy_masks": packed(f["copy_masks"]),
                    "order": raw(r["order"]), "forecast_order": raw(f["tf_order"]),
                    "stats": {k: raw(v) for k, v in r["stats"].items()},
                    "forecast_stats": {k: raw(v) for k, v in f["stats"].items()},
                    "tf_endpoint_by_group_copy": raw(r["tf_reconstruction"][:, :, :, -1]),
                    "ms_endpoint_by_group_copy_scale": raw(r["ms_reconstruction_by_scale"][:, :, :, :, -1]),
                    "tf_endpoint_valid": packed(r["tf_valid"][:, :, :, -1]),
                    "ms_endpoint_valid_by_scale": packed(r["ms_valid_by_scale"][:, :, :, :, -1]),
                    "forecast_valid": packed(f["valid"]),
                    "full_reconstruction": "deterministic from pinned config/source/checkpoint, input identity/footprint and recorded masks; raw endpoint values above",
                    "unarrived_future_observed": False}
                if event_logits:
                    record["event_logit"] = float(model.event_head(parts["event_features"])[0, 0])
                    record["event_features"] = raw(parts["event_features"])
                records.append(record)
    return records


def calibrate_route(records, route, source_calibration=None, active_components=None):
    if route == "Z":
        if source_calibration is None:
            raise ValueError("Z requires existing source calibration, no target fit")
        from .scoring import recipe
        if list(recipe(active_components)[0]) != source_calibration["active_components"]:
            raise ValueError("Source calibration uses a different fixed component recipe")
        return copy.deepcopy(source_calibration)
    return fit_calibration(records, source=route == "STAGE1", active_components=active_components)


def logit_threshold(records):
    values = [r["event_logit"] for r in records if np.isfinite(r["event_logit"])]
    if not values:
        raise ValueError("No valid independent normal support CAL logits")
    return float(np.quantile(values, .99, method="higher"))
