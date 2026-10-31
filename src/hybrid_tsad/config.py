"""Portable inventory configuration, not a training launcher or approval gate."""
import hashlib
import json
import os
import re
from pathlib import Path

OFFICIAL_SHA = "881e8912e4c5b298b202e5b7ac2fe58107c7dcf1b56acf733e28c99f2ffc0317"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, max_bytes=2 * 1024 * 1024):
    """Read bounded JSON, rejecting duplicate keys and nonstandard NaN values."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    with Path(path).open("rb") as stream:
        content = stream.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError(f"JSON exceeds {max_bytes} bytes: {path}")
    return json.loads(content.decode("utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=invalid)


def _keys(value, expected, location):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{location} requires exactly: {', '.join(expected)}")


def _integer(value, minimum, location):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{location} must be an integer >= {minimum}")


def resolve_path(value, config_file):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Paths must be non-empty strings")
    # Explicit ${NAME} works identically on Windows and Unix; unset is an error.
    def substitute(match):
        name = match.group(1)
        if not os.environ.get(name):
            raise ValueError(f"Missing environment variable: {name}")
        return os.environ[name]
    expanded = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", substitute, value)
    if "${" in expanded:
        raise ValueError(f"Unresolved path expression: {value}")
    path = Path(expanded).expanduser()
    return (Path(config_file).resolve().parent / path).resolve()


def validate_config(path, check_paths=False):
    cfg = read_json(path)
    _keys(cfg, ("schema_version", "model", "research", "paths", "upstream"), "config")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    model = cfg["model"]
    _keys(model, ("mode", "history", "copies", "temporal_depth"), "model")
    if model["mode"] not in ("hybrid", "temporal", "tf"):
        raise ValueError("model.mode must be hybrid, temporal or tf")
    _integer(model["history"], 100, "model.history")
    _integer(model["copies"], 2, "model.copies")
    _integer(model["temporal_depth"], 1, "model.temporal_depth")
    if model["history"] % 5 or model["copies"] % 2:
        raise ValueError("history must divide by 5 and copies must be even")
    research = cfg["research"]
    _keys(research, ("target", "sources", "seed", "contract"), "research")
    _integer(research["seed"], 0, "research.seed")
    allowed = {"PSM", "SMD", "SMD_SUBSET", "TEP", "MSL", "SWaT"}
    sources = research["sources"]
    if (not isinstance(research["target"], str) or research["target"] not in allowed
            or not isinstance(sources, list) or not sources
            or any(not isinstance(s, str) or s not in allowed for s in sources)):
        raise ValueError("Unknown or empty research target/sources")
    canonical = lambda s: "SMD" if s == "SMD_SUBSET" else s
    normalized = [canonical(s) for s in sources]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate or aliased source domains")
    if canonical(research["target"]) in normalized:
        raise ValueError("Target domain must be excluded from source domains")
    _keys(cfg["paths"], ("source_manifest", "query_manifest", "run_root"), "paths")
    _keys(cfg["upstream"], ("official_file", "sha256"), "upstream")
    if cfg["upstream"]["sha256"] != OFFICIAL_SHA:
        raise ValueError("Official SHA must match the immutable core baseline")
    resolved = {key: resolve_path(value, path) for key, value in cfg["paths"].items()}
    resolved["official_file"] = resolve_path(cfg["upstream"]["official_file"], path)
    resolved["contract"] = resolve_path(research["contract"], path)
    if check_paths:
        for key in ("source_manifest", "query_manifest", "official_file", "contract"):
            if not resolved[key].is_file():
                raise ValueError(f"Missing file for {key}: {resolved[key]}")
        if sha256(resolved["official_file"]) != OFFICIAL_SHA:
            raise ValueError("External official file SHA256 mismatch")
        if resolved["run_root"].exists() and not resolved["run_root"].is_dir():
            raise ValueError("run_root exists and is not a directory")
    return {"valid": True, "paths_checked": check_paths,
            "scope": "inventory schema only; not experiment approval or manifest leakage audit",
            "config": cfg, "resolved_paths": {k: str(v) for k, v in resolved.items()}}
