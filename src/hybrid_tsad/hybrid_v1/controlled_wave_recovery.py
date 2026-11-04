"""Typed evidence-only recovery of failed technical parity. No automatic retry.

One saved new bank is reused, never forwarded again. A separately signed real
Torch RNG-only fixture can test serialization before any model/PT is loaded.
"""
import argparse
import datetime
from contextlib import ExitStack
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np

from . import controlled_wave as wave

RECOVERY = wave.PREFIX + "TECHNICAL_CPU_PARITY_RECOVERY"
RNG_ONLY = wave.PREFIX + "TORCH_RNG_SERIALIZATION_FIXTURE"
FAILED_SHA = "cfac6026ff384843a7ae00b8806ae055c8e8789407ab574888796ce629154b27"
PARENT_CONFIG = "dfe7eb7142e9a46b454cbd0b21841dc4f48b652dd7b02b96e959af466076c9d0"
PARENT_RELEASE = "72b1c4b7fcd568668c41b95dc1583e0b7104ccd979d25e9aed302bb4bd3b0f2e"
PARTIAL_ACCEPTANCE = "118dd594391d3f489034f3d855fdff9d36eca63190bd57874778e9b1514c730d"


def save_rng(directory, name, state):
    # Tensor conversion belongs here/new adapter, never a change to old004b.
    wave.original.save_bundle(directory, name, wave.original.detached(state))


def rng_equal(a, b):
    """RNG-only canonical containers; exact leaf types/dtype/shape/bytes.

    Old JSON schemas turn tuples into lists. Normalize ONLY these state trees,
    never ordinary observation/bundle lists or numerical data. No tolerances.
    """
    def canonical(value):
        if hasattr(value, "detach"): value = wave.original.detached(value)
        if isinstance(value, np.generic): value = np.asarray(value)
        if isinstance(value, np.ndarray):
            return ("array", value.dtype.str, tuple(value.shape), value.tobytes())
        if isinstance(value, (tuple, list)):
            return ("state_sequence", tuple(canonical(v) for v in value))
        if isinstance(value, dict):
            return ("state_mapping", tuple((k, canonical(value[k])) for k in sorted(value)))
        if type(value) is float:
            return ("python_float64", np.asarray(value, dtype=np.float64).tobytes())
        if value is None or type(value) in (int, bool, str):
            return (type(value).__name__, value)
        raise TypeError("Unsupported RNG state leaf: " + type(value).__name__)
    return canonical(a) == canonical(b)


def load_bundle(schema_path, npz_path):
    """Exact old004b typed schema; caller first pins both file byte identities."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    used = set()
    with np.load(npz_path, allow_pickle=False) as arrays:
        def unpack(node):
            if isinstance(node, dict) and "npz_key" in node:
                if set(node) != {"npz_key", "dtype", "shape"}:
                    raise ValueError("Unexpected typed bundle node")
                array = arrays[node["npz_key"]]
                if (array.dtype.hasobject or str(array.dtype) != node["dtype"]
                        or list(array.shape) != node["shape"]):
                    raise ValueError("Bundle dtype/shape identity differs")
                used.add(node["npz_key"])
                return array.copy()
            if isinstance(node, dict) and set(node) == {"nonfinite_scalar"}:
                if node["nonfinite_scalar"] not in ("nan", "inf", "-inf"):
                    raise ValueError("Unknown nonfinite marker")
                return float(node["nonfinite_scalar"])
            if isinstance(node, dict): return {k: unpack(v) for k, v in node.items()}
            if isinstance(node, list): return [unpack(v) for v in node]
            return node
        value = unpack(schema)
        if used != set(arrays.files): raise ValueError("Unreferenced bundle array pool")
    return value


def restore_snapshot(snapshot):
    """Restore the detached CPU/CUDA/NumPy/Python state without random draws."""
    import torch
    def tuples(value):
        return tuple(tuples(x) for x in value) if isinstance(value, list) else value
    def tensor(value):
        array = np.asarray(value)
        if array.dtype != np.uint8 or array.ndim != 1:
            raise ValueError("Byte vector RNG state required")
        return torch.from_numpy(array.copy())
    np.random.set_state(tuple(snapshot["numpy_python"][0]))
    random.setstate(tuples(snapshot["numpy_python"][1]))
    torch.set_rng_state(tensor(snapshot["cpu"]))
    if snapshot["cuda"]:
        if not torch.cuda.is_initialized(): raise ValueError("No implicit GPU initialization")
        torch.cuda.set_rng_state_all([tensor(x) for x in snapshot["cuda"]])


def compare_saved_new(saved, reference):
    extra = set(saved) - set(reference)
    if not extra <= {"score", "valid", "alarm", "reason"} or not set(reference) <= set(saved):
        raise AssertionError("Saved original raw schema changed beyond scorer outputs")
    wave.assert_tree_parity({k: saved[k] for k in reference}, reference)


def parent_metadata(cfg):
    """Validate forensic metadata before any raw/anchor numerical body read."""
    if cfg["parent_failed_archive"]["sha256"] != FAILED_SHA:
        raise PermissionError("Exact failed attempt archive required")
    archive = wave.bound(cfg["parent_failed_archive"])
    if (archive.get("status") != "FAILED_OS1_PARTIAL_ORIGINALS_PRESERVED_NO_RETRY"
            or archive["OS"]["OS_exit_code"] != 1 or archive["report_exists"] is not False):
        raise PermissionError("Only the actual incomplete failed attempt")
    if cfg["partial_acceptance"]["sha256"] != PARTIAL_ACCEPTANCE:
        raise PermissionError("Exact root partial-new-bank acceptance required")
    accepted = wave.bound(cfg["partial_acceptance"])
    if (accepted.get("status") != "ACCEPTED_PARTIAL_C1_NEW_BANK_ONLY" or accepted.get("approved_by") != "root"
            or accepted.get("parent_failed_manifest_sha256") != FAILED_SHA
            or accepted.get("original_producer_sha256") != "b19c9c3e0842284e43b2e36783bc55e987cad5ff90315491b0c5908d5443103c"
            or accepted.get("index") != 512 or accepted.get("input_seed") != 84554
            or accepted.get("input_shape") != [640, 1]):
        raise PermissionError("Root accepted only the original first new query")
    for key, name in (("saved_bank_manifest", "bank/manifest.json"),
                      ("saved_anchor_schema", "ANCHOR_512.json"), ("saved_anchor_npz", "ANCHOR_512.npz")):
        if cfg[key] != accepted["files"][name]:
            raise PermissionError("Root partial acceptance and reused artifact differ")
    root = Path(cfg["parent_workspace"])
    by_path = {str((root/e["path"]).resolve()): e for e in archive["files"]}
    for e in archive["files"]:
        file = root/e["path"]
        if wave.original.sha(file) != e["sha256"] or file.stat().st_size != e["bytes"]:
            raise PermissionError("Original failed attempt artifact changed")
    for key in ("parent_config", "parent_release", "saved_bank_manifest", "saved_anchor_schema", "saved_anchor_npz"):
        asset = cfg[key]; e = by_path.get(str(Path(asset["path"]).resolve()))
        if e is None or asset["sha256"] != e["sha256"]:
            raise PermissionError("Recovery input must be in exact failed archive: " + key)
    if cfg["parent_config"]["sha256"] != PARENT_CONFIG or cfg["parent_release"]["sha256"] != PARENT_RELEASE:
        raise PermissionError("Original scientific config/release required")
    old_cfg = wave.bound(cfg["parent_config"])
    new_out, old_out = Path(cfg["output"]).resolve(), Path(old_cfg["output"]).resolve()
    if new_out == old_out or old_out in new_out.parents or new_out in old_out.parents:
        raise PermissionError("Recovery cannot overwrite old scientific output")
    saved = wave.bound(cfg["saved_bank_manifest"])
    full = cfg["models"][0]
    identity = saved["contract"]["identity"]
    if (saved.get("complete") is not True or saved.get("committed_records") != 1
            or identity["config_sha256"] != PARENT_CONFIG
            or identity["checkpoint_sha256"] != full["assets"]["checkpoint"]["sha256"]
            or identity["calibration_sha256"] != full["assets"]["calibration"]["sha256"]
            or identity["source_sha256"] != "b19c9c3e0842284e43b2e36783bc55e987cad5ff90315491b0c5908d5443103c"
            or saved["contract"]["ranges"] != [dict(domain="TECHNICAL_ONLY", entity="C1", run="normal_not_formal", start=512, end=513)]):
        raise PermissionError("Only completed first fullC1 new bank can be reused")
    return archive


def recovery_count_totals(counters):
    total = {key: sum(count[key] for count in counters.values()) for key in ("forecast", "reconstruct", "_encode")}
    if total != dict(forecast=5, reconstruct=5, _encode=12):
        raise AssertionError("Recovery exceeded exact5/5/12 model calls: " + repr(total))
    return total


def admit(config, release, trusted, phase):
    if phase not in ("recover", "rng_roundtrip"): raise PermissionError("Unknown typed action")
    if not release or not trusted: raise PermissionError("Exact new root approval required")
    cfg = json.loads(Path(config).read_text())
    signed = wave.bound(dict(path=release, sha256=trusted))
    op = RECOVERY if phase == "recover" else RNG_ONLY
    if (cfg.get("operation") != op or signed.get("operation") != op
            or signed.get("status") != "APPROVED" or signed.get("approved_by") != "root"
            or signed.get("allowed_actions") != [op] or signed.get("config_sha256") != wave.original.sha(config)
            or signed.get("resources") != cfg.get("resources") or cfg.get("resume") is not False
            or signed.get("allow_resume") is not False):
        raise PermissionError("Distinct scientific recovery/RNG fixture authority required")
    if Path(cfg["output"]).exists(): raise FileExistsError("Fresh separate recovery output required")
    wave._source_gate(cfg, signed)
    if cfg["source_sha256"].get(str(Path(__file__).resolve())) != wave.original.sha(__file__):
        raise PermissionError("Running recovery adapter must be pinned")
    expected = dict(wall_seconds=180 if phase == "recover" else 25,
                    kill_grace_seconds=5, memory_GiB=16, automatic_retry=False)
    if cfg.get("resources") != expected or cfg.get("device") != "cpu":
        raise PermissionError("Exact bounded CPU-only action")
    permissions = dict(recovery=phase == "recover", torch_rng_only=True,
                       labels=False, training=False, GPU=False, formal_inputs=False)
    if cfg.get("permissions") != permissions or signed.get("permissions") != permissions:
        raise PermissionError("Exact limited permissions required")
    if phase == "recover":
        if cfg.get("rng_preflight") != dict(required=True, hard_seconds=30, before_model_PT_load=True):
            raise PermissionError("Merged real Torch RNG gate must precede model/PT loading")
        if (cfg.get("technical") != dict(namespace="TECHNICAL_NORMAL_NOT_FORMAL_WAVE", purpose="CPU_PARITY",
                length=640, data_seed=84553, truth=False)
                or [m["model_id"] for m in cfg["models"]] != ["full2025", "tf_off2025"]):
            raise PermissionError("Same original independent technical inputs and two models")
        for model in cfg["models"]: wave._model_metadata(model)
        if signed.get("external_source_sha256") != cfg["external_source_sha256"] or len(cfg["external_source_sha256"]) != 8:
            raise PermissionError("Exact original external8 required")
        for path, digest in cfg["external_source_sha256"].items():
            if wave.original.sha(path) != digest: raise PermissionError("External code changed")
        if signed.get("installed_frft") != cfg["installed_frft"] or wave.original.sha(cfg["installed_frft"]["path"]) != cfg["installed_frft"]["sha256"]:
            raise PermissionError("Actual installed FRFT required")
        if cfg["reference_training"]["sha256"] != "40e2247ed3049bf9d2303c380b2da55c1494c874ca38dd24652f741ed33ccc74":
            raise PermissionError("Original infer reference required")
        if wave.original.sha(cfg["reference_training"]["path"]) != cfg["reference_training"]["sha256"]:
            raise PermissionError("Original reference changed")
        parent_metadata(cfg)
    return cfg


def rng_roundtrip(cfg):
    """Future root-signed real Torch fixture. No model/PT/series/label/GPU."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise PermissionError("CVD empty required")
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_initialized(): raise PermissionError("No CUDA context permitted")
    out = Path(cfg["output"]); out.mkdir(parents=True, exist_ok=False)
    before = wave.neural_rng()
    try:
        save_rng(out, "REAL_TORCH_RNG", before)
        if not rng_equal(before, wave.neural_rng()): raise AssertionError("Serialization advanced RNG")
        decoded = load_bundle(out/"REAL_TORCH_RNG.json", out/"REAL_TORCH_RNG.npz")
        if not rng_equal(before, decoded): raise AssertionError("Serialized RNG differs")
        torch.rand(1); np.random.random(); random.random()
        restore_snapshot(decoded)
        if not rng_equal(before, wave.neural_rng()): raise AssertionError("RNG restore differs")
        return dict(status="PASS_REAL_TORCH_RNG_ONLY_ROUNDTRIP", torch=str(torch.__version__),
                    saved_exact=True, saving_advanced_RNG=False, restore_exact=True,
                    GPU_initialized=torch.cuda.is_initialized(), model_PT_or_series_loaded=False)
    finally:
        restore_snapshot(wave.original.detached(before))


def run_rng_gate(cfg, fixture=None):
    """Merged fixture: hard30s alarm on the CPU main thread, before any model."""
    import signal
    from time import monotonic
    output = Path(cfg["output"])
    if fixture is None: fixture = rng_roundtrip
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise RuntimeError("Do not replace an existing process alarm")
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame): raise TimeoutError("Torch RNG-only gate hard30seconds")
    began = monotonic(); success = False; result = None; error = None
    signal.signal(signal.SIGALRM, expired); signal.setitimer(signal.ITIMER_REAL, 30.)
    try:
        result = fixture(dict(cfg, output=str(output/"TORCH_RNG_ONLY_PREFLIGHT")))
        if result.get("status") != "PASS_REAL_TORCH_RNG_ONLY_ROUNDTRIP":
            raise AssertionError("Actual Torch serialization/restore must pass before model loading")
        if monotonic()-began > 30: raise TimeoutError("Torch RNG-only gate over30seconds")
        success = True
        return result
    except BaseException as exc:
        error = repr(exc); raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.); signal.signal(signal.SIGALRM, old_handler)
        wave.dump(output/"RNG_PREFLIGHT_STATUS.json", dict(success=success, seconds=monotonic()-began,
            hard_seconds=30, before_model_PT_load=True, report=result, error=error))


def run_recovery(cfg):
    """Five paired units: old-fullC1 only + remaining2 original new/old pairs."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": raise PermissionError("CPU only")
    from time import monotonic
    start = monotonic()
    output = Path(cfg["output"]); output.mkdir(parents=True, exist_ok=False)
    rng_preflight = run_rng_gate(cfg)
    import torch
    cli, data, training, scoring, bank, readout = wave._frozen_modules(cfg)
    wave._configure_device(cfg)
    full = cfg["models"][0]
    model = cli.load_model({**cfg, **full, "seed": 2025, "history": 512, "horizon": 32}, full["assets"]["checkpoint"]["path"])
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    before_state = wave.original._state(model)
    wave.persist_evidence(output/"recovered_first_reference_model", "before", before_state, cfg, cli, model)
    source_directory = Path(cfg["saved_bank_manifest"]["path"]).parent
    saved = wave.original.singleton_record(bank.read_records(source_directory), 512)
    anchor = load_bundle(cfg["saved_anchor_schema"]["path"], cfg["saved_anchor_npz"]["path"])
    values = np.random.Generator(np.random.PCG64(84554)).standard_normal((640, 1)).astype(np.float32)
    run = data.Run("TECHNICAL_ONLY", "C1", "normal_not_formal", "QUERY", values, np.ones(values.shape, bool))
    if (saved["replay"]["input_identity"]["values_float32_sha256"] != hashlib.sha256(values.tobytes()).hexdigest()
            or anchor["index"] != 512 or anchor["actual_seed"] != 2537):
        raise AssertionError("Original technical sample/anchor identity differs")
    name = training.__package__ + ".recovered_reference_v4"
    spec = importlib.util.spec_from_file_location(name, cfg["reference_training"]["path"])
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    counters = {}
    with ExitStack() as stack:
        counters["recovered_first_old_reference"] = stack.enter_context(wave.original.count_calls(model))
        save_rng(output, "RECOVERY_LOCAL_RNG_BEFORE_PARENT_RESTORE", wave.neural_rng())
        restore_snapshot(anchor["RNG_before_forecast"])
        if not rng_equal(anchor["RNG_before_forecast"], wave.neural_rng()):
            raise AssertionError("Exact archived pre-forecast RNG restoration required")
        save_rng(output, "RECOVERED_ANCHOR_PRE_FORECAST_RNG", wave.neural_rng())
        old = wave.legacy_singleton(reference, model, [run], plan=[dict(domain=run.domain, entity=run.entity,
            run=run.run, role="QUERY", index=512)], active=full["active_components"])
        compare_saved_new(saved, old)
        wave.assert_tree_parity(saved, cli.score_records([old], wave.bound(full["assets"]["calibration"]))[0])
        if not rng_equal(anchor["RNG_after_reconstruct"], wave.neural_rng()):
            raise AssertionError("Recovered old reference must match original new post RNG")
        wave.original.save_bundle(output, "RECOVERED_FIRST_REFERENCE_RAW", old)
        recovered_reference_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_rng(output, "RECOVERED_FIRST_REFERENCE_RNG_POST", wave.neural_rng())
        restore_snapshot(anchor["RNG_after_reconstruct"])
        save_rng(output, "RECOVERED_CODE_DERIVED_MAIN_TARGET_RNG", wave.neural_rng())
        wave.dump(output/"RECOVERY_EVIDENCE_LINEAGE.json", dict(
            recovered_reference_UTC=recovered_reference_utc, parent_failed_archive=cfg["parent_failed_archive"],
            original_standalone_MAIN_RNG_POST_existed=False, original_reference_raw_was_not_saved=True,
            recovery_main_target="archived anchor after_reconstruct; code-derived equivalent after old non-RNG score/pack/write",
            not_an_original_main_final_snapshot=True, first_new_bank_forwarded_again=False))
        if wave.original._state(model) != before_state: raise AssertionError("Reference changed source state")
        wave.persist_evidence(output/"recovered_first_reference_model", "after", wave.original._state(model), cfg, cli, model)
        def hook(model_id, remaining_model):
            counters["remaining_"+model_id] = stack.enter_context(wave.original.count_calls(remaining_model))
        remaining_cfg = dict(cfg, output=str(output/"remaining_two_pairs"))
        remaining = wave.run_technical(remaining_cfg,
            recovery_units=wave.technical_plan(cfg["technical"])[1:], counter_hook=hook)
        total = recovery_count_totals(counters)
    return dict(status="COMPLETED_FIVE_UNIT_TECHNICAL_RECOVERY_NOT_MECHANISM_EVIDENCE",
        rng_preflight=rng_preflight,
        recovery_actual_counts=total, counters=counters, first_new_bank_forwarded_again=False,
        parent_archive=cfg["parent_failed_archive"], reused_bank=cfg["saved_bank_manifest"],
        reused_anchor=[cfg["saved_anchor_schema"], cfg["saved_anchor_npz"]],
        recovered_first_reference_new_attempt=True, remaining=remaining,
        original_standalone_main_final_snapshot_missing=True,
        main_target_is_code_derived_anchor_equivalence_not_old_saved_main_final=True,
        historical_first_attempt=dict(known_public_forecast=2, known_public_reconstruct=2,
            structural_encodes=4, actual_counter_was_not_archived=True),
        cumulative_known_public_forecast=7, cumulative_known_public_reconstruct=7,
        cumulative_structural_encodes=16, wall_seconds=monotonic()-start, GPUh=0, metrics=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("rng_roundtrip", "recover"), required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--release"); p.add_argument("--trusted-release-sha256")
    p.add_argument("--execute", action="store_true"); p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(preview=True, Torch_loaded=False, model_loaded=False))); return
    cfg = admit(args.config, args.release, args.trusted_release_sha256, args.phase)
    if args.preflight_only:
        print(json.dumps(dict(metadata_preflight="PASS", Torch_loaded=False, model_loaded=False))); return
    cfg["execution_config_sha256"] = wave.original.sha(args.config)
    report = rng_roundtrip(cfg) if args.phase == "rng_roundtrip" else run_recovery(cfg)
    wave.dump(Path(cfg["output"])/"report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
