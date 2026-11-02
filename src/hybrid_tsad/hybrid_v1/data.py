"""Run-contained multivariate windows. Learning records never contain labels."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Run:
    domain: str
    entity: str
    run: str
    role: str
    values: np.ndarray
    observed: np.ndarray
    start: int = 0

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        self.observed = np.asarray(self.observed, dtype=bool)
        if (self.values.ndim != 2 or self.observed.shape != self.values.shape
                or np.isinf(self.values).any() or not np.isfinite(self.values[self.observed]).all()):
            raise ValueError("Finite observed 2D multivariate run required")
        if self.role not in {"FIT", "SELECT", "CAL", "QUERY", "SUPPORT", "NORMAL_SELECT", "NORMAL_CAL"}:
            raise ValueError("Explicit legal role required")


def targets(run, history, horizon):
    return np.arange(history, len(run.values)-horizon+1, dtype=np.int64)


def window(run, target, history, horizon):
    if not isinstance(target, (int, np.integer)) or target < history or target+horizon > len(run.values):
        raise ValueError("Full H+P footprint must stay inside one role/run")
    # Zero is a student placeholder, never imputed GT; observed masks stay separate.
    local = slice(target-history, target+horizon)
    values = np.where(run.observed[local], run.values[local], 0).astype(np.float32)
    return {"history": values[None, :history],
            "observed": run.observed[None, target-history:target],
            "reconstruction_context": values[None, 1:history+1],
            "reconstruction_observed": run.observed[None, target-history+1:target+1],
            "future_targets": values[None, history:],
            "future_observed": run.observed[None, target:target+horizon],
            "current_target": values[None, history], "current_observed": run.observed[None, target],
            "domain": run.domain, "entity": run.entity, "run": run.run, "role": run.role,
            "index": run.start+target, "available_index": run.start+target,
            "footprint": [run.start+target-history, run.start+target+horizon]}


def arrived_targets(run, history):
    """Inference needs H past points plus CURRENT, not unarrived P-1 values."""
    return np.arange(history, len(run.values), dtype=np.int64)


def arrived_window(run, target, history, horizon):
    if (not isinstance(target, (int, np.integer)) or target < history
            or target >= len(run.values) or horizon < 1):
        raise ValueError("Arrived H+1 inference footprint must stay in one role/run")
    observed = run.observed[target-history:target+1]
    values = np.where(observed, run.values[target-history:target+1], 0).astype(np.float32)
    future = np.zeros((1, horizon, values.shape[1]), np.float32)
    future_observed = np.zeros(future.shape, bool)
    future[:, 0] = values[-1]
    future_observed[:, 0] = observed[-1]
    return {"history": values[None, :history], "observed": observed[None, :history],
            "reconstruction_context": values[None, 1:], "reconstruction_observed": observed[None, 1:],
            "future_targets": future, "future_observed": future_observed,
            "current_target": values[None, -1], "current_observed": observed[None, -1],
            "domain": run.domain, "entity": run.entity, "run": run.run, "role": run.role,
            "index": run.start+target, "available_index": run.start+target,
            "footprint": [run.start+target-history, run.start+target+1],
            "unarrived_future_is_padding": True}


class BalancedSampler:
    """Choose domain, then run uniformly, then a legal complete window."""
    def __init__(self, runs, history, horizon, seed, role="FIT"):
        self.runs = [r for r in runs if r.role == role and len(r.values) >= history+horizon]
        self.domains = sorted({r.domain for r in self.runs})
        if not self.domains:
            raise ValueError("No role-contained windows")
        self.history, self.horizon = history, horizon
        self.rng = np.random.default_rng(seed)

    def sample(self):
        domain = self.domains[int(self.rng.integers(len(self.domains)))]
        pool = [r for r in self.runs if r.domain == domain]
        run = pool[int(self.rng.integers(len(pool)))]
        target = int(self.rng.integers(self.history, len(run.values)-self.horizon+1))
        return window(run, target, self.history, self.horizon)


def audit_events(events, normal, forbidden, k, history=0, horizon=1):
    """Intervals are original half-open coordinates; all labeled anomalies count."""
    if type(k) is not int or k < 0 or len(events) != k:
        raise ValueError("K counts all independent labeled anomaly events")
    ids = [e["event_id"] for e in events]
    if len(set(ids)) != k or any(not x for x in ids):
        raise ValueError("Unique complete event IDs required")
    def overlap(a, b):
        return (a["entity"], a["run"]) == (b["entity"], b["run"]) and a["start"] < b["end"] and b["start"] < a["end"]
    for entry in events+normal:
        if entry["end"] <= entry["start"] or any(overlap(entry, x) for x in forbidden):
            raise ValueError("Support overlaps held-out footprint")
    if len(normal) != k or any(n.get("matched_event_id") != e["event_id"] or
                             n["end"]-n["start"] != e["end"]-e["start"] for e, n in zip(events, normal)):
        raise ValueError("Matched normal support length and event mapping required")
    for i, entry in enumerate(events+normal):
        if any(overlap(entry, other) for other in (events+normal)[i+1:]):
            raise ValueError("Support events/normal intervals overlap")
    for normal_entry in normal:
        footprint = dict(normal_entry, start=normal_entry["start"]-history,
                         end=normal_entry["end"]+horizon-1)
        if any(overlap(footprint, blocked) for blocked in forbidden):
            raise ValueError("Normal SSL full footprint overlaps held-out role")
        if any(overlap(footprint, event) for event in events):
            raise ValueError("Normal SSL footprint includes a labeled anomaly event")
    for entry in events:
        footprint = dict(entry, start=entry["start"]-history, end=entry["end"]+horizon-1)
        if any(overlap(footprint, blocked) for blocked in forbidden):
            raise ValueError("Event support full footprint overlaps held-out role")
    return {"K": k, "events": events, "matched_normal": normal,
            "normal_points": sum(n["end"]-n["start"] for n in normal), "forbidden": forbidden,
            "history": history, "horizon": horizon}


def load_manifest(path, roles=None):
    """Explicit public-run manifest; label files are not a supported input field."""
    import json
    from pathlib import Path
    path = Path(path)
    entries = json.loads(path.read_text(encoding="utf-8"))["runs"]
    runs = []
    cache = {}
    for entry in entries:
        if "labels" in entry or "label_path" in entry:
            raise ValueError("SSL manifests must not expose anomaly labels")
        if roles is not None and entry["role"] not in roles:
            continue
        p = Path(entry["path"]); p = p if p.is_absolute() else path.parent/p
        identity = (str(p.resolve()), entry.get("format", "npy"), entry.get("key"),
                    entry.get("observed_key"), tuple(entry.get("columns", [])), entry.get("has_header", True),
                    entry.get("sha256"), tuple(entry.get("expected_shape", [])))
        if identity not in cache:
            if "sha256" in entry:
                import hashlib
                if hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError("Source bytes SHA256 mismatch")
            explicit_observed = None
            if entry.get("format", "npy") == "csv":
                import csv
                has_header = entry.get("has_header", True)
                if type(has_header) is not bool:
                    raise ValueError("has_header must be explicit boolean")
                if has_header:
                    with p.open(encoding="utf-8-sig", newline="") as f:
                        header = next(csv.reader(f))
                    cols = [header.index(name) for name in entry["columns"]]
                else:
                    cols = entry["columns"]
                    if any(type(c) is not int or c < 0 for c in cols):
                        raise ValueError("Headerless columns must be nonnegative integer indices")
                array = np.genfromtxt(p, delimiter=",", skip_header=int(has_header), usecols=cols, ndmin=2)
            else:
                loaded = np.load(p, allow_pickle=False)
                if isinstance(loaded, np.lib.npyio.NpzFile):
                    try:
                        array = loaded[entry["key"]].copy()
                        if "observed_key" in entry:
                            explicit_observed = loaded[entry["observed_key"]].copy()
                            if explicit_observed.dtype != np.bool_ or explicit_observed.shape != array.shape:
                                raise ValueError("Explicit observed mask must be boolean and match values")
                    finally:
                        loaded.close()
                else:
                    if "observed_key" in entry:
                        raise ValueError("Explicit observed key requires NPZ")
                    array = loaded
            if array.ndim != 2 or ("expected_shape" in entry and list(array.shape) != entry["expected_shape"]):
                raise ValueError("Source shape mismatch")
            cache[identity] = (array, np.isfinite(array) if explicit_observed is None else explicit_observed)
        array, observed = cache[identity]
        lo, hi = entry["bounds"]
        if not 0 <= lo < hi <= len(array):
            raise ValueError("Invalid run/role bounds")
        array = array[lo:hi]
        runs.append(Run(entry["domain"], entry["entity"], entry["run"], entry["role"], array,
                        observed[lo:hi], start=entry.get("original_start", 0)+lo))
    for i, a in enumerate(runs):
        for b in runs[i+1:]:
            if (a.entity, a.run) == (b.entity, b.run) and a.start < b.start+len(b.values) and b.start < a.start+len(a.values):
                raise ValueError("Role ranges overlap inside original run")
    return runs


def resolve_plan(runs, entries, history, horizon, arrived=False):
    """Validate predeclared original indices; never choose on loss or labels."""
    result = []
    lookup = {(r.domain, r.entity, r.run, r.role): r for r in runs}
    if len(lookup) != len(runs):
        raise ValueError("Unique domain/entity/run/role required for a fixed plan")
    for entry in entries:
        key = tuple(entry[k] for k in ("domain", "entity", "run", "role"))
        if key not in lookup or type(entry["index"]) is not int:
            raise ValueError("Fixed plan contains an unknown run/role or noninteger index")
        run = lookup[key]
        local = entry["index"]-run.start
        batch = (arrived_window if arrived else window)(run, local, history, horizon)
        if "footprint" in entry and entry["footprint"] != batch["footprint"]:
            raise ValueError("Fixed plan footprint changed")
        result.append((run, local))
    if not result:
        raise ValueError("Nonempty fixed plan required")
    return result


def fixed_plan(runs, history, horizon, count, seed, role, arrived=False, replace=True):
    """Observation-only eligible grid, domain-uniform then run-uniform sampling.

    All current channels and at least two past observations per channel are
    required; no imputed LOCF point is teacher GT. No values/scores/labels select
    these positions. Sampling is with replacement, and duplicates stay recorded.
    """
    if type(count) is not int or count < 1:
        raise ValueError("Positive fixed count required")
    eligible = {}
    for r in runs:
        if r.role != role:
            continue
        ts = arrived_targets(r, history) if arrived else targets(r, history, horizon)
        cs = np.concatenate((np.zeros((1, r.observed.shape[1]), np.int64),
                             np.cumsum(r.observed, axis=0)), axis=0)
        valid = r.observed[ts].all(axis=1) & ((cs[ts]-cs[ts-history]) >= 2).all(axis=1)
        ts = ts[valid]
        if len(ts):
            eligible.setdefault(r.domain, []).append((r, ts))
    expected = {r.domain for r in runs if r.role == role}
    if not expected or set(eligible) != expected:
        raise ValueError("Every source domain needs observation-eligible positions")
    domains = sorted(eligible); rng = np.random.default_rng(seed); output = []
    if not replace and (len(domains) != 1 or sum(len(ts) for _, ts in eligible[domains[0]]) < count):
        raise ValueError("Unique fixed plan needs one domain and enough eligible positions")
    for _ in range(count):
        domain = domains[int(rng.integers(len(domains)))]
        pool = eligible[domain]; r, ts = pool[int(rng.integers(len(pool)))]
        t = int(ts[int(rng.integers(len(ts)))])
        output.append(dict(domain=r.domain, entity=r.entity, run=r.run, role=r.role,
                           index=r.start+t, footprint=[r.start+t-history, r.start+t+(1 if arrived else horizon)]))
        if not replace:
            remaining = ts[ts != t]
            pool[:] = [(a, remaining if a is r else indices) for a, indices in pool if a is not r or len(remaining)]
    return output
