"""Non-training commands intentionally require only the Python standard library."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys

from . import __version__
from .config import validate_config
from .inspection import inspect_run, verify_core


def check_env(require_model=False):
    dependencies = {}
    for name, module in (("numpy", "numpy"), ("pandas", "pandas"),
                         ("torch", "torch"), ("torch-frft", "torch_frft")):
        available = importlib.util.find_spec(module) is not None
        try:
            version = importlib.metadata.version(name) if available else None
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        dependencies[name] = {"discoverable": available, "version": version}
    ready = all(d["discoverable"] for d in dependencies.values())
    return {"valid": sys.version_info >= (3, 10) and (ready or not require_model),
            "python": sys.version.split()[0], "executable": sys.executable,
            "platform": platform.platform(), "dependencies": dependencies,
            "model_dependencies_discoverable": ready, "cuda_probed": False,
            "scope": "metadata discovery only; torch/CUDA/upstream execution not tested"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Hybrid TSAD research core: read-only engineering tools")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    env = sub.add_parser("check-env", help="Inspect Python and dependency metadata without importing torch")
    env.add_argument("--require-model", action="store_true", help="Fail if core model dependencies are missing")
    config = sub.add_parser("config", help="Portable inventory configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate = config_sub.add_parser("validate")
    validate.add_argument("path")
    validate.add_argument("--check-paths", action="store_true", help="Also require input files and verify external official SHA")
    inspect = sub.add_parser("inspect-run", help="List an external run and summarize selected metadata")
    inspect.add_argument("path")
    core = sub.add_parser("verify-core", help="Check byte identity against frozen provenance")
    core.add_argument("--source-root", help="Optional research_workspace path containing original src/hybrid_tsad")
    legacy = sub.add_parser("legacy", help="Preview or explicitly execute an external hash-pinned runner")
    legacy.add_argument("kind", choices=("source-wave", "readout"))
    legacy.add_argument("--script", required=True)
    legacy.add_argument("--execute", action="store_true", help="Execute; requires existing experiment approval")
    # Parse passthrough after -- separately so --script remains an ordinary option.
    raw = list(sys.argv[1:] if argv is None else argv)
    passthrough = []
    if raw and raw[0] == "legacy" and "--" in raw:
        separator = raw.index("--")
        raw, passthrough = raw[:separator], raw[separator + 1:]
    args = parser.parse_args(raw)
    try:
        if args.command == "check-env":
            result = check_env(args.require_model)
        elif args.command == "config":
            result = validate_config(args.path, args.check_paths)
        elif args.command == "inspect-run":
            result = inspect_run(args.path)
        elif args.command == "legacy":
            from .legacy import execute, legacy_command
            command = legacy_command(args.kind, args.script, passthrough)
            if args.execute:
                return execute(command)
            result = {"command": command, "executed": False,
                      "scope": "external frozen runner bridge; no package-native training migration"}
        else:
            result = verify_core(args.source_root)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("valid", True) else 1
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
