"""Lossless NumPy-only raw-record bank; no model/scoring changes or label I/O.

One OS-locked writer commits immutable NPZ chunks then an atomic small manifest.
Unreferenced chunks/temporary files after interruption remain forensic orphans.
Resume verifies the committed prefix, never silently accepts changed raw records.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np


IDENTITY_KEYS = ("input_sha256", "config_sha256", "checkpoint_sha256",
                 "calibration_sha256", "source_sha256")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pack(records):
    """Typed flat value pools plus a UTF8 container tree; no pickle/object arrays.

    Python float64 (including NaN/Inf/-0), int64, bool and NumPy dtypes survive.
    Ragged/optional/different-C records are legal; dictionary order is retained.
    """
    pools, sizes = {}, {}

    def array(value, kind):
        a = np.asarray(value)
        if a.dtype.hasobject or a.dtype.kind not in "biufcSU":
            raise TypeError("Only non-object numeric/string NumPy arrays are supported")
        dtype = a.dtype.str
        name = "p" + str(list(pools).index(dtype)) if dtype in pools else "p" + str(len(pools))
        start = sizes.get(dtype, 0)
        pools.setdefault(dtype, []).append(a.reshape(-1))
        sizes[dtype] = start + a.size
        return [kind, name, dtype, start, int(a.size), list(a.shape)]

    def encode(value):
        if value is None:
            return ["null"]
        if isinstance(value, np.ndarray):
            return array(value, "ndarray")
        if isinstance(value, np.generic):
            return array(value, "numpy_scalar")
        if type(value) in (bool, int, float):
            dtype = {bool: np.bool_, int: np.int64, float: np.float64}[type(value)]
            return array(np.array(value, dtype=dtype), type(value).__name__)
        if isinstance(value, str):
            return ["str", value]
        if isinstance(value, dict):
            if any(not isinstance(k, str) for k in value):
                raise TypeError("Record dictionary keys must be strings")
            return ["dict", [[k, encode(v)] for k, v in value.items()]]
        if isinstance(value, (list, tuple)):
            # Compact homogeneous rectangular Python numeric lists, preserving types.
            def types(v):
                if isinstance(v, list):
                    return set().union(*(types(x) for x in v)) if v else set()
                return {type(v)}
            ts = types(value) if isinstance(value, list) else set()
            if len(ts) == 1 and next(iter(ts)) in (bool, int, float):
                try:
                    a = np.asarray(value)
                    if not a.dtype.hasobject:
                        return array(a, "numeric_list")
                except ValueError:
                    pass
            return ["tuple" if isinstance(value, tuple) else "list", [encode(v) for v in value]]
        raise TypeError("Unsupported raw record type: " + type(value).__name__)

    trees = [encode(record) for record in records]
    arrays = {"p" + str(i): np.concatenate(parts) for i, parts in enumerate(pools.values())}
    arrays["tree_utf8"] = np.frombuffer(_json(trees).encode("utf-8"), dtype=np.uint8)
    return arrays


def _unpack(arrays):
    trees = json.loads(arrays["tree_utf8"].tobytes().decode("utf-8"))

    def decode(node):
        kind = node[0]
        if kind == "null":
            return None
        if kind == "str":
            return node[1]
        if kind == "dict":
            return {key: decode(value) for key, value in node[1]}
        if kind in ("list", "tuple"):
            values = [decode(v) for v in node[1]]
            return tuple(values) if kind == "tuple" else values
        _, name, dtype, start, length, shape = node
        a = arrays[name][start:start + length]
        if a.dtype.str != dtype:
            raise ValueError("Typed payload/tree mismatch")
        a = a.reshape(shape).copy()
        if kind == "ndarray":
            return a
        if kind == "numpy_scalar":
            return a[()]
        if kind == "numeric_list":
            return a.tolist()
        return {"int": int, "bool": bool, "float": float}[kind](a.item())

    return [decode(tree) for tree in trees]


def _fingerprint(record):
    digest = hashlib.sha256()
    for name, a in _pack([record]).items():
        digest.update(name.encode("ascii"))
        digest.update(a.dtype.str.encode("ascii"))
        digest.update(a.tobytes())
    return digest.hexdigest()


def _key(record):
    if any(not isinstance(record.get(k), str) for k in ("domain", "entity", "run")):
        raise ValueError("Raw records need domain/entity/run strings")
    if type(record.get("index")) is not int:
        raise ValueError("Raw record index must be a Python original integer")
    return tuple(record[k] for k in ("domain", "entity", "run", "index"))


class _Lock:
    def __init__(self, directory):
        self.handle = (directory / "writer.lock").open("a+b")
        self.handle.seek(0, 2)
        if self.handle.tell() == 0:
            self.handle.write(b"0"); self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Bank already has a live writer") from None

    def close(self):
        self.handle.close()  # OS lock also releases on process termination.


def _atomic_manifest(directory, manifest, fault):
    temporary = directory / ("manifest.pending-" + uuid.uuid4().hex + ".json")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_json(manifest) + "\n"); handle.flush(); os.fsync(handle.fileno())
    fault("before_manifest_replace")
    os.replace(temporary, directory / "manifest.json")
    fault("after_manifest_replace")


class ChunkedBank:
    """Strict declared original-index plan and durable prefix; bounded RAM.

    ranges: [{domain, entity, run, start, end}] ordered half-open original indices.
    Explicit separate ranges represent gaps; infer horizons never truncate tails.
    identity includes five hashes and optional ``input_runs`` with exact replay
    input_identity per domain/entity/run. Bind all before any numerical query.
    """
    def __init__(self, directory, *, identity, ranges, threshold, chunk_size=128,
                 confirmation_points=3, fault=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = _Lock(self.directory)
        self.closed, self.poisoned, self.buffer = False, False, []
        self.fault = fault or (lambda stage: None)
        try:
            for key in IDENTITY_KEYS:
                value = identity.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("Missing/invalid bound identity: " + key)
            if type(chunk_size) is not int or chunk_size < 1 or type(confirmation_points) is not int or confirmation_points < 1:
                raise ValueError("Positive integer chunk/confirmation sizes required")
            if not np.isfinite(threshold):
                raise ValueError("Finite source-fixed threshold required")
            previous = {}
            for r in ranges:
                stream = tuple(r[k] for k in ("domain", "entity", "run"))
                if (any(not isinstance(s, str) for s in stream) or type(r["start"]) is not int
                        or type(r["end"]) is not int or r["start"] < 0 or r["end"] <= r["start"]
                        or r["start"] < previous.get(stream, -1)):
                    raise ValueError("Ordered nonoverlapping original-index ranges required")
                previous[stream] = r["end"]
            if not ranges:
                raise ValueError("Nonempty declared coverage plan required")
            self.contract = json.loads(_json(dict(identity=identity, ranges=ranges, threshold=float(threshold),
                chunk_size=chunk_size, confirmation_points=confirmation_points)))
            path = self.directory / "manifest.json"
            if path.exists():
                self.manifest = json.loads(path.read_text(encoding="utf-8"))
                if self.manifest["contract"] != self.contract or self.manifest["schema_version"] != 1:
                    raise ValueError("Bank input/config/checkpoint/calibration/source/coverage contract changed")
                self._validate_existing()
            else:
                self.manifest = dict(schema_version=1, contract=self.contract, committed_records=0,
                                     chunks=[], state=[], complete=False)
                _atomic_manifest(self.directory, self.manifest, self.fault)
            self.states = {tuple(s["stream"]): dict(s) for s in self.manifest["state"]}
        except BaseException:
            self.lock.close(); self.closed = True
            raise

    @property
    def total_records(self):
        return sum(r["end"] - r["start"] for r in self.contract["ranges"])

    def expected_key(self, position):
        for r in self.contract["ranges"]:
            size = r["end"] - r["start"]
            if position < size:
                return tuple(r[k] for k in ("domain", "entity", "run")) + (r["start"] + position,)
            position -= size
        return None

    def remaining_plan(self):
        """Only durable committed prefix is skipped, no caller-selected indices."""
        for position in range(self.manifest["committed_records"], self.total_records):
            key = self.expected_key(position)
            yield dict(zip(("domain", "entity", "run", "index"), key))

    def _validate_existing(self):
        count, states = 0, {}
        for chunk in self.manifest["chunks"]:
            path = self.directory / chunk["file"]
            if path.parent != self.directory or path.name != chunk["file"] or _sha(path) != chunk["sha256"]:
                raise ValueError("Committed chunk identity/path mismatch")
            with np.load(path, allow_pickle=False) as arrays:
                entries = _unpack(arrays)
            if len(entries) != chunk["records"] or chunk["start_position"] != count:
                raise ValueError("Committed chunk count/order mismatch")
            if len(entries) > self.contract["chunk_size"]:
                raise ValueError("Oversize committed chunk")
            if len(entries) < self.contract["chunk_size"] and count + len(entries) != self.total_records:
                raise ValueError("Only the final coverage chunk may be short")
            for entry in entries:
                if _key(entry["raw"]) != self.expected_key(count):
                    raise ValueError("Committed original-index coverage mismatch")
                derived = self._derive(entry["raw"], states)
                if derived != entry["derived"]:
                    raise ValueError("Committed confirmation state differs from replay")
                count += 1
        if count != self.manifest["committed_records"] or list(states.values()) != self.manifest["state"]:
            raise ValueError("Manifest count/confirmation state mismatch")
        if self.manifest["complete"] and count != self.total_records:
            raise ValueError("Premature complete marker")

    def _derive(self, record, states):
        stream = _key(record)[:3]
        index = record["index"]
        state = states.get(stream)
        value = record.get("score")
        finite = isinstance(value, (float, int, np.floating)) and math.isfinite(value)
        valid = record.get("valid", finite)
        if type(valid) is not bool:
            raise ValueError("valid must be an explicit Python bool when provided")
        strict = bool(valid and finite and value > self.contract["threshold"])
        streak = state["streak"] if state and index == state["last_index"] + 1 else 0
        streak = streak + 1 if strict else 0
        states[stream] = dict(stream=list(stream), last_index=index, streak=streak)
        return dict(score_valid=bool(valid and finite), strict_alarm=strict,
                    confirmed3_current=streak >= self.contract["confirmation_points"])

    def append(self, record):
        if self.closed or self.poisoned or self.manifest["complete"]:
            raise RuntimeError("Bank is closed, poisoned, or already complete")
        key = _key(record)
        if key != self.expected_key(self.manifest["committed_records"] + len(self.buffer)):
            raise ValueError("Duplicate, missing, out-of-order or undeclared original index")
        for bound in self.contract["identity"].get("input_runs", []):
            if tuple(bound[k] for k in ("domain", "entity", "run")) == key[:3]:
                if record.get("replay", {}).get("input_identity") != bound["input_identity"]:
                    raise ValueError("Record input array/observed identity mismatch")
                break
        else:
            if self.contract["identity"].get("input_runs"):
                raise ValueError("Record stream has no bound input identity")
        # Copy through the lossless codec: later caller mutation cannot alter a bank.
        copied = _unpack(_pack([record]))[0]
        derived = self._derive(copied, self.states)
        self.buffer.append(dict(raw=copied, derived=derived))
        if len(self.buffer) == self.contract["chunk_size"]:
            self.flush()

    def flush(self, *, final=False):
        if self.closed or self.poisoned:
            raise RuntimeError("Closed/poisoned bank")
        if not self.buffer:
            return
        if len(self.buffer) != self.contract["chunk_size"] and not (
                final and self.manifest["committed_records"] + len(self.buffer) == self.total_records):
            raise ValueError("Only a complete final coverage chunk may be short")
        try:
            name = "chunk-%06d-%s.npz" % (len(self.manifest["chunks"]), uuid.uuid4().hex)
            temporary = self.directory / (name + ".pending")
            with temporary.open("xb") as handle:
                np.savez_compressed(handle, **_pack(self.buffer))
                handle.flush(); os.fsync(handle.fileno())
            self.fault("before_chunk_replace")
            os.replace(temporary, self.directory / name)
            self.fault("after_chunk_replace")
            updated = {**self.manifest, "chunks": self.manifest["chunks"] + [dict(file=name,
                sha256=_sha(self.directory / name), records=len(self.buffer),
                bytes=(self.directory / name).stat().st_size,
                start_position=self.manifest["committed_records"])],
                "committed_records": self.manifest["committed_records"] + len(self.buffer),
                "state": list(self.states.values())}
            _atomic_manifest(self.directory, updated, self.fault)
            self.manifest = updated
            self.buffer = []
        except BaseException:
            self.poisoned = True
            raise

    def write(self, records, *, verify_committed_prefix=False):
        """Resume either remaining_plan input, or replay a full verified prefix."""
        records = iter(records)
        if verify_committed_prefix:
            if self.buffer:
                raise RuntimeError("Prefix replay requires an empty buffer")
            for saved in read_records(self.directory):
                try:
                    incoming = next(records)
                except StopIteration:
                    raise ValueError("Iterator ends before durable prefix") from None
                if _fingerprint(saved) != _fingerprint(incoming):
                    raise ValueError("Replayed durable prefix differs; identity/raw drift")
        for record in records:
            self.append(record)

    def finalize(self):
        if self.manifest["committed_records"] + len(self.buffer) != self.total_records:
            raise ValueError("Incomplete declared original coverage; cannot finalize")
        self.flush(final=True)
        updated = {**self.manifest, "complete": True}
        try:
            _atomic_manifest(self.directory, updated, self.fault)
        except BaseException:
            self.poisoned = True
            raise
        self.manifest = updated

    def close(self):
        """Unflushed records are deliberately uncommitted; never implicit success."""
        if not self.closed:
            self.lock.close(); self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_entries(directory):
    """Read only SHA-verified manifested immutable chunks, ignoring crash orphans."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for chunk in manifest["chunks"]:
        path = directory / chunk["file"]
        if path.parent != directory or path.name != chunk["file"] or _sha(path) != chunk["sha256"]:
            raise ValueError("Chunk identity/path mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            entries = _unpack(arrays)
        if len(entries) != chunk["records"]:
            raise ValueError("Chunk record count mismatch")
        yield from entries


def read_records(directory):
    for entry in read_entries(directory):
        yield entry["raw"]


def iter_pointwise_infer(model, runs, *, history, horizon, seed=17,
                         event_logits=False, active_components=None, infer_fn=None,
                         index_plan=None):
    """Consume shared iter_infer directly; no per-point full-run hashing.

    The generator owns every pass, mask seed, raw value and NaN; this wrapper
    does not call a random-number API or apply scores. ``infer_fn`` is an optional
    injectable record iterable factory with the same generator signature.
    ``index_plan`` may come from remaining_plan(); role comes from the bound run.
    Torch imports only when the real infer is actually requested.
    """
    if infer_fn is None:
        from .training import iter_infer as infer_fn
    runs = list(runs)
    lookup = {(r.domain, r.entity, r.run): r for r in runs}
    if len(lookup) != len(runs):
        raise ValueError("Unique original run identity required")
    plan = None
    if index_plan is not None:
        plan = [{**entry, "role": lookup[tuple(entry[k] for k in
                    ("domain", "entity", "run"))].role} for entry in index_plan]
        if not plan:
            return
    expected = (iter(tuple(e[k] for k in ("domain", "entity", "run", "index")) for e in plan)
                if plan is not None else
                iter((r.domain, r.entity, r.run, r.start+t)
                     for r in runs for t in range(history, len(r.values))))
    rows = infer_fn(model, runs, history=history, horizon=horizon, seed=seed,
                    event_logits=event_logits, active_components=active_components,
                    index_plan=plan)
    for row in rows:
        if _key(row) != next(expected, None):
            raise ValueError("iter_infer original point/order differs from the declared plan")
        yield row
    if next(expected, None) is not None:
        raise ValueError("iter_infer ended before declared coverage")
