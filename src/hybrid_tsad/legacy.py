"""Explicit external frozen-runner bridge; never rewrites archived run files."""
import subprocess
import sys
from pathlib import Path

from .config import sha256

RUNNERS = {
    "source-wave": {"filename": "hybrid_10k_wave.py",
                    "sha256": "00ed75a4c1da76600d573be67dead4179b455131610607d7b1cea3bb0dc8e394"},
    "readout": {"filename": "run_hybrid_10k_psm_inference.py",
                "sha256": "ce37f1f025c5ad29e3ffe537d582f703fed39f5d7650bdd1b33141e5c67b3866"},
}


def legacy_command(kind, script, arguments):
    path = Path(script).resolve()
    record = RUNNERS[kind]
    if path.name != record["filename"] or sha256(path) != record["sha256"]:
        raise ValueError("External runner does not match the pinned legacy version")
    if not arguments:
        raise ValueError("Explicit legacy runner arguments are required")
    return [sys.executable, str(path), *arguments]


def execute(command):
    # A list avoids shell parsing; inherits the caller's environment and cwd.
    return subprocess.run(command, check=False).returncode
