"""Package the small completed-run artifacts, with hashes of every member."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

RUN = Path("results/left_holdout_history_20261127_v1")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def archive(workspace):
    root = workspace / RUN
    if not (root / "summary/PUBLIC.md").is_file():
        raise RuntimeError("The full fixed report has not been sealed")
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and p.suffix in {".json", ".csv", ".md", ".log"}
        and p.name != "evidence_manifest.json"
    ]
    manifest = root / "evidence_manifest.json"
    entries = [
        dict(
            path=p.relative_to(root).as_posix(),
            bytes=p.stat().st_size,
            sha256=digest(p),
        )
        for p in files
    ]
    manifest.write_text(
        json.dumps(dict(run=RUN.as_posix(), files=entries), indent=2) + "\n",
        encoding="utf-8",
    )
    archive = workspace / "tmp/left_holdout_history_20261127_evidence.tar.gz"
    partial = archive.with_suffix(".part")
    with tarfile.open(partial, "w:gz") as handle:
        for p in [*files, manifest]:
            handle.add(p, p.relative_to(root).as_posix(), recursive=False)
    partial.replace(archive)
    print(
        json.dumps(dict(archive=str(archive), files=len(files), sha256=digest(archive)))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    archive(parser.parse_args().workspace)
