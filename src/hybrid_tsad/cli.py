"""Non-training commands intentionally require only the Python standard library."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys

from . import __version__
from .config import read_json, validate_config
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
    for name in ("train", "infer", "evaluate", "calibrate"):
        operation = sub.add_parser(name, help="Unified engine config; default preview, explicit execution only")
        operation.add_argument("--config", required=True)
        operation.add_argument("--execute", action="store_true")
        if name == "evaluate":
            operation.add_argument("--allow-labels", action="store_true")
    readout = sub.add_parser("readout", help="Multi-entity fixed readout; default preview only")
    readout.add_argument("--config", required=True)
    readout.add_argument("--execute", action="store_true")
    readout.add_argument("--allow-labels", action="store_true")
    stage2 = sub.add_parser("stage2", help="Head-only U stage5 config; preview unless explicitly approved")
    stage2_inputs=stage2.add_mutually_exclusive_group(required=True)
    stage2_inputs.add_argument("--config")
    stage2_inputs.add_argument("--runtime-identity",action="store_true",help="Print current source SHA identity without torch/data")
    stage2.add_argument("--execute", action="store_true")
    stage2_readout = sub.add_parser("stage2-readout", help="Independent schema6 readout; default preview, no labels")
    readout6_inputs=stage2_readout.add_mutually_exclusive_group(required=True)
    readout6_inputs.add_argument("--config")
    readout6_inputs.add_argument("--runtime-identity",action="store_true")
    stage2_readout.add_argument("--execute", action="store_true")
    stage2_evaluate = sub.add_parser("stage2-evaluate", help="Independent evaluation runtime; default preview, separate label release")
    evaluation_inputs=stage2_evaluate.add_mutually_exclusive_group(required=True)
    evaluation_inputs.add_argument("--config")
    evaluation_inputs.add_argument("--runtime-identity",action="store_true")
    stage2_evaluate.add_argument("--execute",action="store_true")
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
            raw_config = read_json(args.path)
            if isinstance(raw_config, dict) and "engine_schema_version" in raw_config:
                from .engine.pipeline import check_operation_inputs, load_operation, run_operation
                operation = raw_config.get("operation")
                if operation not in ("train", "infer", "evaluate", "calibrate"):
                    raise ValueError("Unknown engine operation")
                result = run_operation(args.path, operation)
                if args.check_paths:
                    cfg, records, paths, output = load_operation(args.path, operation)
                    check_operation_inputs(records, paths, cfg, output)
                result.update(valid=True, paths_checked=args.check_paths)
            else:
                result = validate_config(args.path, args.check_paths)
        elif args.command == "inspect-run":
            result = inspect_run(args.path)
        elif args.command in ("train", "infer", "evaluate", "calibrate"):
            from .engine.pipeline import run_operation
            result = run_operation(args.config, args.command, execute=args.execute,
                                   allow_labels=getattr(args, "allow_labels", False))
        elif args.command == "readout":
            from .engine.multi_readout import run_readout
            result = run_readout(args.config, execute=args.execute, allow_labels=args.allow_labels)
        elif args.command == "stage2":
            if args.runtime_identity:
                if args.execute:raise ValueError("Runtime identity is read-only")
                from .engine.stage2_identity import runtime_identity
                result=runtime_identity()
            else:
                from .engine.stage2 import run_stage2
                result = run_stage2(args.config, execute=args.execute)
        elif args.command == "stage2-readout":
            if args.runtime_identity:
                if args.execute:raise ValueError("Runtime identity is read-only")
                from .engine.stage2_readout_identity import runtime_identity
                result=runtime_identity()
            else:
                from .engine.stage2_readout import run_readout
                result = run_readout(args.config, execute=args.execute)
        elif args.command == "stage2-evaluate":
            if args.runtime_identity:
                if args.execute:raise ValueError("Runtime identity is read-only")
                from .engine.stage2_evaluation import runtime_identity
                result=runtime_identity()
            else:
                from .engine.stage2_evaluation import run_evaluation
                result=run_evaluation(args.config,execute=args.execute)
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
    except (OSError, ValueError, TypeError, ImportError, KeyError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
