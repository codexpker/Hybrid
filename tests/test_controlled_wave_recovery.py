"""Evidence/metadata fakes only. Real Torch RNG and neural recovery NOT run."""
import ast
import copy
import importlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/"src/hybrid_tsad/hybrid_v1"
PACKAGE = "_controlled_rng_recovery_engineering"
pkg = types.ModuleType(PACKAGE); pkg.__path__ = [str(SOURCE)]; sys.modules[PACKAGE] = pkg
recovery = importlib.import_module(PACKAGE+".controlled_wave_recovery")
wave = recovery.wave


class EvidenceTensorFake:
    """Explicit fake only, reproducing detach/cpu/numpy evidence protocol."""
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def fake_rng():
    return dict(cpu=EvidenceTensorFake(np.arange(8, dtype=np.uint8)), cuda=[],
                numpy_python=wave._numpy_rng())


class RecoveryEvidenceEngineeringTests(unittest.TestCase):
    def test_negative_raw_tensor_fixture_reproduces_serialization_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError): wave.original.save_bundle(tmp, "RAW_UNDETACHED_FAKE", fake_rng())

    def test_detached_cpu_rng_roundtrip_dtype_bytes_and_numpy_python_state(self):
        before = wave._numpy_rng()
        with tempfile.TemporaryDirectory() as tmp:
            state = fake_rng(); recovery.save_rng(tmp, "DETACHED", state)
            saved = recovery.load_bundle(Path(tmp)/"DETACHED.json", Path(tmp)/"DETACHED.npz")
            self.assertTrue(wave.same_neural_rng(wave.original.detached(state), saved))
            self.assertEqual(saved["cpu"].dtype, np.uint8)
        self.assertTrue(wave._same_numpy_rng(before, wave._numpy_rng()))

    def test_rng_live_tuple_vs_saved_list_exact_dtype_shape_bytes_and_python_ints(self):
        with tempfile.TemporaryDirectory() as tmp:
            live = fake_rng(); recovery.save_rng(tmp, "LIVE_TO_DISK", live)
            loaded = recovery.load_bundle(Path(tmp)/"LIVE_TO_DISK.json", Path(tmp)/"LIVE_TO_DISK.npz")
            self.assertIsInstance(live["numpy_python"], tuple)
            self.assertIsInstance(loaded["numpy_python"], list)
            self.assertTrue(recovery.rng_equal(live, loaded))
            wrong = copy.deepcopy(loaded); wrong["cpu"] = wrong["cpu"].astype(np.uint16)
            self.assertFalse(recovery.rng_equal(live, wrong))
            wrong = copy.deepcopy(loaded); wrong["cpu"] = wrong["cpu"].reshape(2, 4)
            self.assertFalse(recovery.rng_equal(live, wrong))
            wrong = copy.deepcopy(loaded); wrong["cpu"][0] ^= 1
            self.assertFalse(recovery.rng_equal(live, wrong))
            wrong = copy.deepcopy(loaded); wrong["numpy_python"][0][1][0] ^= np.uint32(1)
            self.assertFalse(recovery.rng_equal(live, wrong))
            wrong = copy.deepcopy(loaded); wrong["numpy_python"][1][1][0] += 1
            self.assertFalse(recovery.rng_equal(live, wrong))

    def test_nested_cuda_tensor_fake_converted_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = fake_rng(); state["cuda"] = [EvidenceTensorFake(np.array([2, 4, 7], np.uint8))]
            recovery.save_rng(tmp, "FAKE_CUDA_NOT_GPU", state)
            saved = recovery.load_bundle(Path(tmp)/"FAKE_CUDA_NOT_GPU.json", Path(tmp)/"FAKE_CUDA_NOT_GPU.npz")
            np.testing.assert_array_equal(saved["cuda"][0], [2, 4, 7])

    def test_schema_dtype_and_extra_array_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            recovery.save_rng(tmp, "BAD", fake_rng())
            schema = Path(tmp)/"BAD.json"; value = json.loads(schema.read_text())
            value["cpu"]["dtype"] = "float64"; schema.write_text(json.dumps(value))
            with self.assertRaises(ValueError): recovery.load_bundle(schema, Path(tmp)/"BAD.npz")

    def test_restore_fake_codec_preserves_global_rng_and_no_cuda_init(self):
        snapshot = wave.original.detached(fake_rng()); saved = []
        fake = types.SimpleNamespace(from_numpy=lambda a: a.copy(), set_rng_state=lambda a: saved.append(a),
            cuda=types.SimpleNamespace(is_initialized=lambda: False))
        before = wave._numpy_rng()
        with mock.patch.dict(sys.modules, {"torch": fake}): recovery.restore_snapshot(snapshot)
        self.assertTrue(wave._same_numpy_rng(before, wave._numpy_rng()))
        np.testing.assert_array_equal(saved[0], snapshot["cpu"])
        self.assertNotIn("torch", sys.modules)

    def test_existing_new_record_compared_without_reforward_and_integer_exact(self):
        old = dict(index=512, prediction=[[.125]], replay=dict(actual_mask_seed=2537))
        saved = dict(old, score=.3, valid=True, alarm=False)
        recovery.compare_saved_new(saved, old)
        with self.assertRaises(AssertionError): recovery.compare_saved_new(dict(saved, added_unknown_field=1), old)
        with self.assertRaises(AssertionError): recovery.compare_saved_new(dict(saved, index=513), old)

    def test_five_unit_count_and_historical_structure_not_original6(self):
        counts = dict(first=dict(forecast=1, reconstruct=1, _encode=2),
            tf=dict(forecast=2, reconstruct=2, _encode=4), full3=dict(forecast=2, reconstruct=2, _encode=6))
        self.assertEqual(recovery.recovery_count_totals(counts), dict(forecast=5, reconstruct=5, _encode=12))
        bad = copy.deepcopy(counts); bad["first"]["forecast"] += 1
        with self.assertRaises(AssertionError): recovery.recovery_count_totals(bad)

    def test_count_context_restores_instance_methods_even_on_failure(self):
        class ExplicitFakeModel:
            def _encode(self): return None
            def forecast(self): self._encode()
            def reconstruct(self): self._encode()
        model = ExplicitFakeModel(); original_dict = dict(model.__dict__)
        with self.assertRaises(RuntimeError):
            with wave.original.count_calls(model) as actual:
                model.forecast(); model.reconstruct(); raise RuntimeError("fixture")
        self.assertEqual(model.__dict__, original_dict)
        self.assertEqual(actual, dict(forecast=1, reconstruct=1, _encode=2))

    def test_adapter_all_rng_bundle_calls_are_detached_in_ast(self):
        tree = ast.parse((SOURCE/"controlled_wave.py").read_text())
        observed = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "save_bundle"
                    and len(node.args) >= 3 and isinstance(node.args[1], ast.Constant)
                    and "RNG" in str(node.args[1].value)):
                observed.append(node.args[1].value)
                self.assertIsInstance(node.args[2], ast.Call)
                self.assertIsInstance(node.args[2].func, ast.Attribute)
                self.assertEqual(node.args[2].func.attr, "detached")
        self.assertEqual(set(observed), {"REFERENCE_RNG_POST", "MAIN_RNG_POST", "INITIAL_RNG", "FINAL_RNG"})

    def test_unapproved_recovery_or_rng_fixture_refused_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/"config.json"; config.write_text('{}')
            for phase in ("recover", "rng_roundtrip"):
                with self.assertRaises(PermissionError): recovery.admit(config, None, None, phase)
        self.assertNotIn("torch", sys.modules)

    def test_merged_gate_success_and_failure_metadata_only_fake_fixture(self):
        calls = []
        fake_signal = types.SimpleNamespace(ITIMER_REAL=0, SIGALRM=14,
            getitimer=lambda kind: (0., 0.), getsignal=lambda kind: "old",
            signal=lambda kind, value: calls.append(("handler", value)),
            setitimer=lambda kind, value: calls.append(("timer", value)))
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            good = base/"good"; good.mkdir()
            with mock.patch.dict(sys.modules, {"signal": fake_signal}):
                recovery.run_rng_gate(dict(output=str(good)), fixture=lambda cfg: dict(status="PASS_REAL_TORCH_RNG_ONLY_ROUNDTRIP", fixture_only=True))
            status = json.loads((good/"RNG_PREFLIGHT_STATUS.json").read_text())
            self.assertTrue(status["success"]); self.assertTrue(status["before_model_PT_load"])
            bad = base/"bad"; bad.mkdir()
            with mock.patch.dict(sys.modules, {"signal": fake_signal}):
                with self.assertRaises(AssertionError): recovery.run_rng_gate(dict(output=str(bad)), fixture=lambda cfg: dict(status="FAILED_FAKE"))
            self.assertFalse(json.loads((bad/"RNG_PREFLIGHT_STATUS.json").read_text())["success"])
        self.assertIn(("timer", 30.), calls); self.assertIn(("timer", 0.), calls)
        self.assertNotIn("torch", sys.modules)

    def _parent_fixture(self, root):
        files = {}
        def save(name, value):
            path = root/name; path.write_text(json.dumps(value))
            asset = dict(path=str(path), sha256=wave.original.sha(path))
            files[name] = dict(path=name, sha256=asset["sha256"], bytes=path.stat().st_size)
            return asset
        parent_config = save("parent_config.json", dict(output=str(root/"old_output")))
        parent_release = save("parent_release.json", dict(status="APPROVED"))
        ckpt, cal = "a"*64, "b"*64
        bank = save("bank.json", dict(complete=True, committed_records=1, contract=dict(
            identity=dict(config_sha256=parent_config["sha256"], checkpoint_sha256=ckpt, calibration_sha256=cal,
                source_sha256="b19c9c3e0842284e43b2e36783bc55e987cad5ff90315491b0c5908d5443103c"),
            ranges=[dict(domain="TECHNICAL_ONLY", entity="C1", run="normal_not_formal", start=512, end=513)])))
        anchor_schema = save("anchor_schema.json", {})
        anchor_pool = save("anchor_pool_metadata_only.json", {})
        archive = save("failed_manifest.json", dict(status="FAILED_OS1_PARTIAL_ORIGINALS_PRESERVED_NO_RETRY",
            OS=dict(OS_exit_code=1), report_exists=False, files=list(files.values())))
        cfg = dict(parent_workspace=str(root), parent_failed_archive=archive, parent_config=parent_config,
            parent_release=parent_release, saved_bank_manifest=bank, saved_anchor_schema=anchor_schema,
            saved_anchor_npz=anchor_pool, output=str(root/"fresh_recovery"),
            models=[dict(assets=dict(checkpoint=dict(sha256=ckpt), calibration=dict(sha256=cal)))])
        accepted = save("partial_acceptance.json", dict(status="ACCEPTED_PARTIAL_C1_NEW_BANK_ONLY", approved_by="root",
            parent_failed_manifest_sha256=archive["sha256"],
            original_producer_sha256="b19c9c3e0842284e43b2e36783bc55e987cad5ff90315491b0c5908d5443103c",
            index=512, input_seed=84554, input_shape=[640, 1], files={"bank/manifest.json":bank,
                "ANCHOR_512.json":anchor_schema, "ANCHOR_512.npz":anchor_pool}))
        cfg["partial_acceptance"] = accepted
        return cfg

    def test_parent_archive_hash_and_old_output_protected_metadata_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._parent_fixture(Path(tmp))
            with mock.patch.object(recovery, "FAILED_SHA", cfg["parent_failed_archive"]["sha256"]), \
                 mock.patch.object(recovery, "PARENT_CONFIG", cfg["parent_config"]["sha256"]), \
                 mock.patch.object(recovery, "PARENT_RELEASE", cfg["parent_release"]["sha256"]), \
                 mock.patch.object(recovery, "PARTIAL_ACCEPTANCE", cfg["partial_acceptance"]["sha256"]):
                recovery.parent_metadata(cfg)
                cfg["output"] = str(Path(tmp)/"old_output/inside")
                with self.assertRaises(PermissionError): recovery.parent_metadata(cfg)

    def test_parent_artifact_change_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._parent_fixture(Path(tmp))
            with mock.patch.object(recovery, "FAILED_SHA", cfg["parent_failed_archive"]["sha256"]), \
                 mock.patch.object(recovery, "PARTIAL_ACCEPTANCE", cfg["partial_acceptance"]["sha256"]):
                Path(cfg["saved_anchor_schema"]["path"]).write_text('{"modified":true}')
                with self.assertRaises(PermissionError): recovery.parent_metadata(cfg)


if __name__ == "__main__":
    unittest.main()
