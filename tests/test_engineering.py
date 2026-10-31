import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hybrid_tsad.cli import main
from hybrid_tsad.config import OFFICIAL_SHA, read_json, resolve_path, validate_config
from hybrid_tsad.inspection import inspect_run, verify_core
from hybrid_tsad.legacy import RUNNERS, legacy_command

ROOT = Path(__file__).resolve().parents[1]


class EngineeringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.cfg = read_json(ROOT / "configs/core.example.json")
        self.path = self.directory / "config.json"

    def validate(self, cfg=None, check_paths=False):
        self.path.write_text(json.dumps(self.cfg if cfg is None else cfg), encoding="utf-8")
        return validate_config(self.path, check_paths)

    def test_example_and_relative_paths(self):
        result = self.validate()
        self.assertTrue(result["valid"])
        self.assertFalse(result["paths_checked"])
        self.assertEqual(Path(result["resolved_paths"]["run_root"]),
                         (self.path.parent / self.cfg["paths"]["run_root"]).resolve())

    def test_model_contract_rejects_invalid_values(self):
        for key, value in [("copies", 3), ("copies", True), ("history", 399),
                           ("temporal_depth", 0), ("mode", "unknown")]:
            with self.subTest(key=key, value=value):
                cfg = copy.deepcopy(self.cfg)
                cfg["model"][key] = value
                with self.assertRaises(ValueError):
                    self.validate(cfg)

    def test_target_exclusion_and_alias_duplicate(self):
        for target, sources in [("SMD", ["SMD_SUBSET"]), ("PSM", ["PSM"]),
                                ("PSM", ["SMD", "SMD_SUBSET"])]:
            cfg = copy.deepcopy(self.cfg)
            cfg["research"].update(target=target, sources=sources)
            with self.assertRaises(ValueError):
                self.validate(cfg)

    def test_unknown_keys_and_sha_rejected(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["upstream"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.validate(cfg)
        cfg = copy.deepcopy(self.cfg)
        cfg["model"]["histroy"] = 400
        with self.assertRaises(ValueError):
            self.validate(cfg)

    def test_environment_paths(self):
        with patch.dict(os.environ, {"HYBRID_TEST_ROOT": str(self.directory)}):
            self.assertEqual(resolve_path("${HYBRID_TEST_ROOT}/x", self.path), self.directory / "x")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                resolve_path("${HYBRID_TEST_ROOT}/x", self.path)

    def test_path_check_and_wrong_official_identity(self):
        for key in self.cfg["paths"]:
            self.cfg["paths"][key] = str(self.directory / key)
        self.cfg["research"]["contract"] = str(self.directory / "contract.md")
        self.cfg["upstream"]["official_file"] = str(self.directory / "official.py")
        with self.assertRaises(ValueError):
            self.validate(check_paths=True)
        for name in ("source_manifest", "query_manifest", "contract.md", "official.py"):
            (self.directory / name).write_text("{}")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.validate(check_paths=True)

    def test_json_duplicate_and_nonfinite_and_size(self):
        for content in ('{"x":1,"x":2}', '{"x":NaN}'):
            self.path.write_text(content)
            with self.assertRaises(ValueError):
                read_json(self.path)
        self.path.write_text(" " * 32)
        with self.assertRaises(ValueError):
            read_json(self.path, max_bytes=10)

    def test_inspect_does_not_open_arrays_or_follow_manifest(self):
        self.path.write_text('{"status":"Running","steps":10000,"jobs":{"x":{}}}')
        (self.directory / "labels.npy").write_bytes(b"not an array")
        (self.directory / "FINAL_MANIFEST.json").write_text(json.dumps({"files": {"missing.npy": "a" * 64}}))
        result = inspect_run(self.directory)
        self.assertFalse(result["array_checkpoint_label_payloads_opened"])
        self.assertEqual(result["metadata"]["config.json"]["status"], "Running")
        self.assertEqual(result["metadata"]["FINAL_MANIFEST.json"]["manifest_entries"], 1)
        self.assertFalse((self.directory / "missing.npy").exists())

    def test_core_byte_identity_and_original_mapping(self):
        self.assertTrue(verify_core()["valid"])
        if (ROOT.parent / "src/hybrid_tsad/model.py").exists():
            self.assertTrue(verify_core(ROOT.parent)["valid"])
        self.assertFalse(verify_core(self.directory)["valid"])

    def test_cli_failure_exit_and_torch_not_imported(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["config", "validate", str(self.directory / "absent.json")]), 2)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["check-env"]), 0)
        self.assertFalse(json.loads(output.getvalue())["cuda_probed"])

    def test_legacy_preview_and_reject_changed_script(self):
        invalid = self.directory / "hybrid_10k_wave.py"
        invalid.write_text("raise SystemExit(0)")
        with self.assertRaises(ValueError):
            legacy_command("source-wave", invalid, ["--help"])
        # No external legacy scripts are required for standalone clone tests.
        with patch("hybrid_tsad.legacy.sha256", return_value=RUNNERS["source-wave"]["sha256"]):
            command = legacy_command("source-wave", invalid, ["--help"])
            self.assertEqual(command[-1], "--help")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["legacy", "source-wave", "--script", str(invalid),
                                       "--", "--help"]), 0)
        self.assertFalse(json.loads(output.getvalue())["executed"])


if __name__ == "__main__":
    unittest.main()
