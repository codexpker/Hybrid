"""A-stage fixed causal reference center; original neural kernels stay intact.

No Torch import on metadata paths. All references are role-contained, separately
observed prefixes. Model instances are single-owner: overlapping/reentrant use
is rejected rather than changing process-global normalizers.
"""
from contextlib import contextmanager
import hashlib
import inspect
import threading
import types

import numpy as np

REFERENCE_LENGTH = 512
ARMS = ("window", "reference")


def function_clone(function, globals_dict):
    """Reuse the exact Python code object with a private globals dictionary."""
    clone = types.FunctionType(function.__code__, globals_dict, function.__name__,
                               function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    return clone


def reference_window(run, target, history, horizon, *, arrived=False, data_module=None):
    from . import data as original_data
    data = original_data if data_module is None else data_module
    if run.role not in {"FIT", "SELECT", "CAL", "QUERY"}:
        raise ValueError("Only explicit source/controlled roles are supported")
    if type(target) is not int or target < REFERENCE_LENGTH + history:
        raise ValueError("Reference must precede and be disjoint from the full history")
    batch = (data.arrived_window if arrived else data.window)(run, target, history, horizon)
    reference = run.values[:REFERENCE_LENGTH]
    observed = run.observed[:REFERENCE_LENGTH]
    if reference.shape[0] != REFERENCE_LENGTH or not (observed.sum(0) >= 2).all():
        raise ValueError("Every fixed-reference channel requires two real observations")
    batch.update(reference_values=reference[None].copy(), reference_observed=observed[None].copy(),
                 reference_bounds=[run.start, run.start+REFERENCE_LENGTH],
                 reference_role=run.role, reference_identity=(run.domain, run.entity, run.run, run.role),
                 run_role=run.role, run_start=run.start,
                 forecast_history_bounds=[run.start+target-history, run.start+target],
                 reconstruction_bounds=[run.start+target-history+1, run.start+target+1])
    return batch


def validate_reference_batch(batch, history, horizon):
    """Actual engine gate at EVERY batch, before either encoder is entered."""
    role = batch["run_role"]
    if role not in {"FIT", "SELECT", "CAL", "QUERY"} or batch["reference_role"] != role:
        raise ValueError("Reference role mismatch")
    identity = (batch["domain"], batch["entity"], batch["run"], role)
    if tuple(batch["reference_identity"]) != identity:
        raise ValueError("Reference run/entity/domain identity mismatch")
    start, t = batch["run_start"], batch["index"]
    if batch["reference_bounds"] != [start, start+REFERENCE_LENGTH]:
        raise ValueError("Reference is not the declared role-local fixed prefix")
    if batch["forecast_history_bounds"] != [t-history, t] or batch["reconstruction_bounds"] != [t-history+1, t+1]:
        raise ValueError("Forecast/reconstruction axis mismatch")
    footprint = batch["footprint"]
    if footprint not in ([t-history, t+horizon], [t-history, t+1]):
        raise ValueError("Only original full-training or arrived-only footprint is allowed")
    if batch["reference_bounds"][1] > footprint[0] or t < start+REFERENCE_LENGTH+history:
        raise ValueError("Reference intersects input/hidden/current/future supervision")
    if batch["available_index"] != t:
        raise ValueError("Wrong current availability time")
    for key, length in (("history", history), ("reconstruction_context", history),
                        ("reference_values", REFERENCE_LENGTH), ("reference_observed", REFERENCE_LENGTH)):
        if len(batch[key].shape) != 3 or batch[key].shape[1] != length:
            raise ValueError("Reference/current window shape mismatch")
    if batch["history"].shape[0::2] != batch["reference_values"].shape[0::2]:
        raise ValueError("Reference batch/channel dimensions differ")


def visible_reference_statistics(values, observed):
    import torch
    if values.dtype != torch.float32 or observed.dtype != torch.bool or values.shape != observed.shape or values.device != observed.device:
        raise ValueError("Float32 reference values and matching boolean observed required")
    if values.ndim != 3 or values.shape[1] != REFERENCE_LENGTH:
        raise ValueError("Fixed 512-point [B,R,C] reference required")
    if bool((observed & ~torch.isfinite(values)).any()):
        raise ValueError("Nonfinite observed reference is not valid teacher information")
    count = observed.sum(1, keepdim=True)
    if not bool((count >= 2).all()):
        raise ValueError("Every reference channel needs at least two observations")
    mean = (torch.where(observed, values, torch.zeros_like(values)).sum(1, keepdim=True)/count).detach()
    return {"mean": mean, "count": count, "valid": count >= 2}


def reference_normalize(values, visible, reference_stats, original_normalize):
    import torch
    _, local = original_normalize(values, visible)
    center = reference_stats["mean"]
    if center.shape != local["mean"].shape or center.device != values.device:
        raise ValueError("Bound reference channel/batch/device differs")
    safe = torch.where(visible, values, torch.zeros_like(values))
    valid = local["channel_valid"] & reference_stats["valid"]
    student = torch.where(visible & valid, (safe-center)/local["scale"], torch.zeros_like(safe))
    stats = dict(local, mean=center, channel_valid=valid, local_mean=local["mean"],
                 reference_mean=center, reference_count=reference_stats["count"],
                 reference_valid=reference_stats["valid"])
    return student, stats


def model_class(core_module):
    """Subclass the pinned HybridV1, reusing its exact _encode code object."""
    class ReferenceCenterHybridV1(core_module.HybridV1):
        def __init__(self, config, sources, *, center_mode="window"):
            if center_mode not in ARMS:
                raise ValueError("Only the two registered center arms")
            super().__init__(config, sources)
            self.center_mode = center_mode
            self._reference_bound = None
            self._reference_owner = None
            # Plain Python lock, not a parameter/buffer/module; deepcopy is not
            # used in A. Stage1 original fit only deepcopies state_dict tensors.
            self._reference_lock = threading.Lock()

        @contextmanager
        def reference_context(self, batch):
            if not self._reference_lock.acquire(blocking=False):
                raise RuntimeError("One model cannot serve overlapping/reentrant reference contexts")
            try:
                if self._reference_bound is not None:
                    raise RuntimeError("Reference state leaked across a previous pass")
                validate_reference_batch(batch, self.config.history, self.config.horizon)
                self._reference_bound = visible_reference_statistics(batch["reference_values"], batch["reference_observed"])
                self._reference_owner = threading.get_ident()
                yield
            finally:
                self._reference_bound = None
                self._reference_owner = None
                self._reference_lock.release()

        def _encode(self, values, observed, hidden, *, mask_seed, copy_masks=None):
            if self.center_mode == "window":
                return super()._encode(values, observed, hidden, mask_seed=mask_seed, copy_masks=copy_masks)
            if self._reference_bound is None or self._reference_owner != threading.get_ident():
                raise RuntimeError("Reference arm requires an admitted per-batch reference context")
            namespace = dict(core_module.HybridV1._encode.__globals__)
            bound = self._reference_bound
            namespace["visible_normalize"] = lambda x, mask: reference_normalize(x, mask, bound, core_module.visible_normalize)
            encode = function_clone(core_module.HybridV1._encode, namespace)
            return encode(self, values, observed, hidden, mask_seed=mask_seed, copy_masks=copy_masks)

        def reference_provenance(self):
            path = inspect.getsourcefile(core_module.HybridV1._encode)
            with open(path, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
            return {"center_mode": self.center_mode, "base_model_source": path, "base_model_sha256": digest,
                    "reused_code_object": "HybridV1._encode", "process_global_normalizer_modified": False,
                    "new_parameters_or_buffers": False, "single_owner_nonreentrant_context": True,
                    "variance": "original visible local mean-centered population variance +1e-5",
                    "reference_length": REFERENCE_LENGTH}

    ReferenceCenterHybridV1.__module__ = __name__
    return ReferenceCenterHybridV1


def training_namespace(original_training):
    """Private cloned training globals; original fit/SELECT/iter math unchanged."""
    from . import data
    namespace = dict(original_training.__dict__)
    for name, value in original_training.__dict__.items():
        if isinstance(value, types.FunctionType) and value.__globals__ is original_training.__dict__:
            namespace[name] = function_clone(value, namespace)
    namespace["window"] = lambda run, t, h, p: reference_window(run, int(t), h, p, data_module=data)
    namespace["arrived_window"] = lambda run, t, h, p: reference_window(run, int(t), h, p, arrived=True, data_module=data)

    def passes(model, batch, seed):
        with model.reference_context(batch):
            return original_training.passes(model, batch, seed)

    namespace["passes"] = passes
    original_iter = namespace["iter_infer"]

    def iter_infer(model, runs, *, history, horizon, seed=17, event_logits=False, active_components=None, index_plan=None):
        if event_logits:
            raise ValueError("A center experiment does not train/use an event head")
        # Original iterator default begins at H; A requires R+H. A supplied CAL
        # plan is already checked by the same per-batch gate.
        if index_plan is None:
            index_plan = [{"domain": r.domain, "entity": r.entity, "run": r.run, "role": r.role,
                           "index": r.start+t, "footprint": [r.start+t-history, r.start+t+1]}
                          for r in runs for t in range(REFERENCE_LENGTH+history, len(r.values))]
        lookup = {(r.domain, r.entity, r.run): r for r in runs}
        for record in original_iter(model, runs, history=history, horizon=horizon, seed=seed,
                                    event_logits=False, active_components=active_components, index_plan=index_plan):
            r = lookup[(record["domain"], record["entity"], record["run"])]
            record["replay"]["reference"] = {"bounds": [r.start, r.start+REFERENCE_LENGTH], "role": r.role,
                                            "center_mode": model.center_mode, "prefix_from_same_input_identity": True}
            yield record

    namespace["iter_infer"] = iter_infer
    original_fit = namespace["fit"]
    original_update_policy = namespace["update_policy"]

    def update_policy(model, route):
        if route not in {"STAGE1", "Z"}:
            raise NotImplementedError("New reference-center policy supports Stage1/Z only")
        return original_update_policy(model, route)

    def fit(*args, **kwargs):
        if kwargs.get("route", "STAGE1") not in {"STAGE1", "Z"}:
            raise NotImplementedError("Reference-center A implements Stage1/Z only; legacy U/F remains separate")
        return original_fit(*args, **kwargs)

    def adapt(*args, **kwargs):
        raise NotImplementedError("No reference-center target adaptation is authorized or implemented in A")

    namespace["fit"], namespace["adapt"], namespace["update_policy"] = fit, adapt, update_policy
    return types.SimpleNamespace(**namespace)


def fixed_reference_record(run, t, *, history=512):
    """The sole no-training simple control, same reference/history eligibility."""
    batch = reference_window(run, int(t), history, 32, arrived=True)
    reference = batch["reference_values"][0]
    observed_ref = batch["reference_observed"][0]
    past, observed = batch["history"][0], batch["observed"][0]
    count = observed.sum(0)
    denominator = np.maximum(count, 1).astype(np.float32)
    local_mean = np.where(observed, past, np.float32(0)).sum(0, dtype=np.float32)/denominator
    centered = np.where(observed, past-local_mean, np.float32(0))
    scale = np.sqrt((centered**2).sum(0, dtype=np.float32)/denominator+np.float32(1e-5))
    rcount = observed_ref.sum(0)
    center = np.where(observed_ref, reference, np.float32(0)).sum(0, dtype=np.float32)/rcount.astype(np.float32)
    valid = (rcount >= 2) & (count >= 2) & batch["current_observed"][0]
    residual = ((batch["current_target"][0]-center)/scale)**2
    scalar = float(residual[valid].mean()) if valid.any() else float("nan")
    return {"domain": run.domain, "entity": run.entity, "run": run.run, "index": run.start+t,
            "r_pred": scalar, "reference_bounds": batch["reference_bounds"],
            "prediction": center.tolist(), "reference_count": rcount.tolist(),
            "history_visible_count": count.tolist(),
            "r_pred_by_channel": [float(residual[i]) if valid[i] else None for i in range(len(valid))],
            "current_target": batch["current_target"][0].tolist(), "current_observed": batch["current_observed"][0].tolist(),
            "local_mean": local_mean.tolist(), "scale": scale.tolist(), "channel_valid": valid.tolist(),
            "method": "B_ref_fixed_history_mean", "trained": False}
