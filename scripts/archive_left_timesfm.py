"""Package small, sealed experiment evidence; leave all original assets intact."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def archive(workspace):
    root = workspace / "results/left_timesfm_hidden_20261126_v1"
    assert (root / "U_READOUT_COMPLETE.json").exists()
    review = json.loads((root / "summary/cache_recheck.json").read_text())
    assert review["status"] == "PASS"
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        # Scores and model/hidden tensors remain in the original run directory.
        # The lightweight package contains their sealed identities and receipts.
        if path.suffix.lower() not in {".json", ".csv", ".md", ".png", ".pdf", ".log", ".py", ".patch"}:
            continue
        if path.name in {"evidence_manifest.json"}:
            continue
        files.append((path, relative))
    manifest = {
        "experiment": root.name,
        "remote_root": str(root),
        "scope": "Receipts, metrics, plots, executed-source snapshots and logs; original model and score banks retained remotely",
        "files": [{"path": str(r), "bytes": p.stat().st_size, "sha256": digest(p)} for p, r in files],
    }
    manifest_path = root / "evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output = workspace / "tmp/left_timesfm_hidden_evidence.tar.gz"
    with tarfile.open(output, "w:gz") as package:
        for path, relative in files:
            package.add(path, arcname=str(relative), recursive=False)
        package.add(manifest_path, arcname=manifest_path.name, recursive=False)
    print(json.dumps({"archive": str(output), "bytes": output.stat().st_size,
                      "sha256": digest(output), "files": len(files) + 1}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    archive(parser.parse_args().workspace)
