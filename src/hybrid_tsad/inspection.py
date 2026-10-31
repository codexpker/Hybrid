"""Bounded read-only artifact inspection; no arrays, labels or checkpoints loaded."""
from pathlib import Path

from .config import read_json, sha256


def inspect_run(path):
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Run directory does not exist: {root}")
    entries = []
    for item in sorted(root.iterdir(), key=lambda p: p.name):
        entries.append({"name": item.name, "kind": "symlink" if item.is_symlink() else
                        "directory" if item.is_dir() else "file",
                        "bytes": item.stat().st_size if item.is_file() else None})
    summaries = {}
    # Metadata allowlist only. No arbitrary recursive JSON/label scans.
    for name in ("config.json", "jobs.json", "FINAL_MANIFEST.json", "inputs.json",
                 "PRIMARY_APPROVAL.json", "executor_check.json"):
        item = root / name
        if item.is_symlink() or not item.is_file():
            continue
        try:
            value = read_json(item)
            summary = {"sha256": sha256(item), "type": type(value).__name__}
            if isinstance(value, dict):
                summary["keys"] = sorted(value)
                for key in ("status", "target", "seed", "history", "steps", "batch"):
                    if key in value and isinstance(value[key], (str, int, float, bool, type(None))):
                        summary[key] = value[key]
                if isinstance(value.get("jobs"), dict):
                    summary["job_names"] = sorted(value["jobs"])
                files = value.get("files", value if name == "FINAL_MANIFEST.json" else {})
                if isinstance(files, dict):
                    summary["manifest_entries"] = sum(
                        isinstance(v, str) and len(v) == 64 for v in files.values())
            summaries[name] = summary
        except (ValueError, OSError) as exc:
            summaries[name] = {"error": str(exc)}
    return {"run": str(root), "entries": entries, "metadata": summaries,
            "scope": "metadata inventory only; recorded status is not audit acceptance",
            "array_checkpoint_label_payloads_opened": False}


def verify_core(source_root=None):
    package = Path(__file__).parent
    manifest = read_json(package / "core_provenance.json")
    checks = []
    for record in manifest["files"]:
        destination = package / record["package_file"]
        result = {"file": record["package_file"], "expected_sha256": record["sha256"],
                  "actual_sha256": sha256(destination)}
        result["package_matches"] = result["actual_sha256"] == record["sha256"]
        if source_root is not None:
            original = Path(source_root) / record["original_file"]
            result["original_matches"] = original.is_file() and sha256(original) == record["sha256"]
        checks.append(result)
    return {"valid": all(c["package_matches"] and c.get("original_matches", True) for c in checks),
            "checks": checks, "scope": "byte identity, not numerical end-to-end equivalence"}
