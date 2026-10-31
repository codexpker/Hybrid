"""Exact multi-entity readout identities. No label file is opened here."""
import hashlib
import json
from pathlib import Path

from ..config import read_json, resolve_path, sha256
from .identity import operation_manifest

MODES = ("temporal", "tf", "hybrid")
METHODS = MODES + ("late", "history_mean", "persistence")
SEEDS = (0, 1, 2)


def record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def checked(value, base):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("Expected one frozen path/SHA record")
    path = resolve_path(value["path"], base)
    if not path.is_file() or sha256(path) != value["sha256"]:
        raise ValueError(f"Readout input identity mismatch: {path.name}")
    return path


def entity_profile(manifest, profile):
    if profile["name"] == "TEP100":
        expected = dict(faults=list(range(1, 21)), runs=list(range(31, 51)), points=960,
                        channels=52, event_start=160, sample_interval_minutes=3)
        if any(profile.get(k) != v for k, v in expected.items()) or manifest.get("dataset") != "TEP":
            raise ValueError("TEP100 entity/run/fault contract changed")
    elif profile["name"] == "SYNTHETIC":
        if manifest.get("dataset") != "SYNTHETIC":
            raise ValueError("Synthetic fixtures cannot declare real TEP records")
    else:
        raise ValueError("Unknown multi-entity readout profile")
    records = manifest["records"]
    expected_pairs = {(f, r) for f in profile["faults"] for r in profile["runs"]}
    pairs, entities = set(), set()
    for row in records:
        pair = (row["fault_id"], row["run_id"])
        if pair in pairs or row["entity"] in entities or pair not in expected_pairs:
            raise ValueError("Duplicate/unapproved entity/fault/run")
        pairs.add(pair); entities.add(row["entity"])
        if row["entity"] != f"fault{pair[0]:02}_run{pair[1]:03}":
            raise ValueError("Entity name does not match its frozen fault/run identity")
        if (row["role"] != "query" or row["history"] != 100 or row["horizon"] != 5
                or row["local_start"] != 0 or row["local_end_exclusive"] != profile["points"]
                or row["channels"] != profile["channels"] or row["guard"] != 0
                or row["origins"] != list(range(100, profile["points"] - 4, 5))):
            raise ValueError("Entity history/footprint/origin contract mismatch")
    if pairs != expected_pairs:
        raise ValueError("Incomplete entity Cartesian product")
    return {row["entity"]: row for row in records}


def native_key(seed, mode, entity):
    return f"s{seed}/{mode}/{entity}"


def score_key(seed, method, entity):
    return f"s{seed}/{method}/{entity}"


def expected_native(entities):
    return {native_key(s, m, e) for s in SEEDS for m in MODES for e in entities}


def expected_scores(entities):
    return {score_key(s, m, e) for s in SEEDS for m in METHODS for e in entities}


def validate_whitelist(whitelist, entities, scope):
    cells, seen = [], set()
    for row in whitelist:
        if set(row) != {"seed", "mode", "entity", "origins"}:
            raise ValueError("Whitelist must explicitly name seed/mode/entity/origins")
        s, m, e = row["seed"], row["mode"], row["entity"]
        key = native_key(s, m, e)
        if s not in SEEDS or m not in MODES or e not in entities or key in seen:
            raise ValueError("Duplicate or unknown whitelist cell")
        allowed = entities[e]["origins"]
        if scope == "l0":
            first = min(entities, key=lambda name: (entities[name]["fault_id"], entities[name]["run_id"]))
            if s != 0 or e != first or row["origins"] != allowed[:2]:
                raise ValueError("L0 is seed0/first entity/first2 blocks only")
        elif scope == "shard":
            if row["origins"] != allowed:
                raise ValueError("A full-readout shard must contain complete entity banks")
        else:
            raise ValueError("Unknown inference scope")
        seen.add(key); cells.append(row)
    if not cells:
        raise ValueError("Empty inference whitelist")
    return cells


def load_readout_operation(path):
    cfg = read_json(path)
    required = {"engine_schema_version", "operation", "contract", "approval", "inputs", "parameters", "output", "score_space"}
    if set(cfg) != required or cfg["engine_schema_version"] != 4 or cfg["operation"] not in (
            "readout-controls", "readout-infer", "readout-seal", "readout-evaluate"):
        raise ValueError("Invalid multi-readout operation schema")
    allowed_inputs={"readout-controls":{"definition"},"readout-infer":{"catalog","query_manifest","whitelist"},
                    "readout-seal":{"catalog","query_manifest","shards"},"readout-evaluate":{"catalog","query_manifest","seal","affiliation"}}
    allowed_parameters={"readout-controls":{"profile","scope"},"readout-infer":{"profile","scope","device"},
                        "readout-seal":{"profile"},"readout-evaluate":{"profile"}}
    if set(cfg["inputs"])!=allowed_inputs[cfg["operation"]] or set(cfg["parameters"])!=allowed_parameters[cfg["operation"]]:
        raise ValueError("Unexpected/missing readout inputs or parameters; no label inputs before unlock")
    paths = {name: resolve_path(value["path"], path) for name, value in {"contract":cfg["contract"], "approval":cfg["approval"], **cfg["inputs"]}.items()}
    output = resolve_path(cfg["output"], path)
    manifest, digest = operation_manifest(cfg, paths, output)
    return cfg, paths, output, manifest, digest


def approve_readout_inputs(cfg, paths, output, base):
    for name, value in {"contract":cfg["contract"], "approval":cfg["approval"], **cfg["inputs"]}.items():
        checked(value, base)
    _, digest = operation_manifest(cfg, paths, output)
    approval = read_json(paths["approval"])
    if (approval.get("status") != "APPROVED" or approval.get("contract_sha256") != cfg["contract"]["sha256"]
            or approval.get("operation_manifest_sha256") != digest):
        raise ValueError("Multi-readout release does not approve this exact operation")
    if output.exists():
        raise ValueError("Readout output must be a new immutable directory")
    return digest


def json_write(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
