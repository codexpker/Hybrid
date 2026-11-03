"""Only synthetic public-table fixtures; no real dataset/model/label access."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from hybrid_tsad.hybrid_v1 import processed_query as pq
from hybrid_tsad.hybrid_v1.data import load_manifest


class ProcessedQueryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixture(self, value="1.125"):
        source = self.root / "input.csv"
        rows = [["date", "data", "cols"]]
        for feature in range(3):
            for index in range(5):
                token = "TRAIN_NOT_NUMERIC" if index < 2 else value
                if feature:
                    token = "NONPRIMARY_NOT_NUMERIC"
                rows.append([str(index + 1), token, "col_" + str(feature)])
        with source.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)
            # Invalid trailing CSV is deliberately outside the feature iterator.
            stream.write('"LABEL_NEVER_REQUESTED\n')
        config = dict(operation=pq.OPERATION, dataset="SYNTHETIC_PROCESSED",
                      input=dict(path=str(source), bytes=source.stat().st_size,
                                 sha256=pq.sha256(source)),
                      schema=dict(header=rows[0], block_rows=5, split=2,
                                  feature_names=["col_0", "col_1", "col_2"],
                                  primary_feature_indices=[0]),
                      output_dir=str(self.root / "derived"), permissions=pq.PERMISSIONS.copy())
        return source, rows, config

    def mutate(self, source, rows, config):
        with source.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows(rows)
        config["input"].update(bytes=source.stat().st_size, sha256=pq.sha256(source))

    def test_sentinels_manifest_and_original_unchanged(self):
        source, _, config = self.fixture()
        before = source.read_bytes()
        report = pq.convert_query(config)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(report["numeric_query_tokens"], 3)
        self.assertEqual(report["label_records_requested_by_csv_iterator"], 0)
        run, = load_manifest(Path(config["output_dir"]) / "INPUT_MANIFEST.json")
        self.assertEqual(run.values.shape, (3, 1))
        self.assertEqual(run.start, 0)
        self.assertEqual(run.role, "QUERY")
        np.testing.assert_array_equal(run.values, np.full((3, 1), 1.125, dtype=np.float32))
        with np.load(Path(config["output_dir"]) / "query_inputs.npz") as bank:
            np.testing.assert_array_equal(bank["processed_index"], [2, 3, 4])
            np.testing.assert_array_equal(bank["test_index"], [0, 1, 2])
        archive = json.loads((Path(config["output_dir"]) / "ARTIFACT_MANIFEST.json").read_text())
        for item in archive["files"]:
            self.assertEqual(pq.sha256(Path(config["output_dir"]) / item["path"]), item["sha256"])

    def test_iterator_never_requests_label_record(self):
        _, rows, config = self.fixture()
        lines = []
        for row in rows:
            buffer = io.StringIO()
            csv.writer(buffer).writerow(row)
            lines.append(buffer.getvalue())

        def limited():
            yield from lines
            raise AssertionError("Attempted to request label record")

        values, observed = pq._parse_features(limited(), config["schema"])
        self.assertEqual(values.shape, (3, 1))
        self.assertTrue(observed.all())

    def test_missing_dtype_and_rounding(self):
        source, rows, config = self.fixture(value="1.234567890123")
        rows[4][1] = "NaN"
        self.mutate(source, rows, config)
        pq.convert_query(config)
        with np.load(Path(config["output_dir"]) / "query_inputs.npz") as bank:
            self.assertEqual(bank["values"].dtype, np.dtype("float32"))
            self.assertEqual(bank["observed"].dtype, np.dtype("bool"))
            self.assertEqual(bank["test_index"].dtype, np.dtype("int64"))
            self.assertTrue(np.isnan(bank["values"][1, 0]))
            self.assertFalse(bank["observed"][1, 0])
            self.assertEqual(bank["values"][0, 0], np.float32(1.234567890123))

    def test_inf_and_overflow_fail_closed(self):
        for value in ["inf", "-inf", "3.5e38", "1e999", "NOT_A_NUMBER"]:
            with self.subTest(value=value):
                source, _, config = self.fixture(value)
                config["output_dir"] += value.replace("/", "_")
                with self.assertRaises(ValueError):
                    pq.convert_query(config)
                self.assertTrue((Path(config["output_dir"]) / "CONVERSION_FAILURE.json").exists())

    def test_hash_mismatch_before_output(self):
        _, _, config = self.fixture()
        config["input"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            pq.convert_query(config)
        self.assertFalse(Path(config["output_dir"]).exists())

    def test_byte_mismatch_before_output(self):
        _, _, config = self.fixture()
        config["input"]["bytes"] += 1
        with self.assertRaisesRegex(ValueError, "byte size"):
            pq.convert_query(config)
        self.assertFalse(Path(config["output_dir"]).exists())

    def test_fresh_conflict_before_input_access(self):
        _, _, config = self.fixture()
        Path(config["output_dir"]).mkdir()
        config["input"]["path"] = str(self.root / "missing")
        with mock.patch.object(pq, "sha256", side_effect=AssertionError("hash called")):
            with self.assertRaises(FileExistsError):
                pq.convert_query(config)

    def test_header_mismatch(self):
        source, rows, config = self.fixture()
        rows[0] = ["data", "date", "cols"]
        self.mutate(source, rows, config)
        with self.assertRaisesRegex(ValueError, "header mismatch"):
            pq.convert_query(config)

    def test_date_and_block_mismatch(self):
        source, rows, config = self.fixture()
        for position, column, value in [(4, 0, "999"), (6, 2, "col_0")]:
            with self.subTest(position=position):
                changed = [row[:] for row in rows]
                changed[position][column] = value
                self.mutate(source, changed, config)
                config["output_dir"] = str(self.root / ("bad" + str(position)))
                with self.assertRaisesRegex(ValueError, "structure mismatch"):
                    pq.convert_query(config)

    def test_truncated_feature_blocks(self):
        source, rows, config = self.fixture()
        self.mutate(source, rows[:-1], config)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            pq.convert_query(config)

    def test_preview_does_not_open_input(self):
        _, _, config = self.fixture()
        config["input"]["path"] = str(self.root / "does_not_exist")
        path = self.root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            pq.main(["--config", str(path)])
        self.assertFalse(json.loads(stream.getvalue())["input_opened"])
        self.assertFalse(Path(config["output_dir"]).exists())

    def test_execute_without_authority_never_reads_input(self):
        _, _, config = self.fixture()
        config["input"]["path"] = str(self.root / "does_not_exist")
        path = self.root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "explicit release"):
            pq.main(["--config", str(path), "--execute"])

    def test_exact_root_binding_positive_and_wrong_config(self):
        _, _, config = self.fixture()
        package = Path(pq.__file__).resolve().parents[1]
        config["source_sha256"] = {name: pq.sha256(package / name) for name in [
            "__init__.py", "hybrid_v1/__init__.py", "hybrid_v1/config.py",
            "hybrid_v1/provenance.py", "hybrid_v1/processed_query.py"]}
        path = self.root / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        release = dict(status=pq.RELEASE_STATUS, approved_by="root", operation=pq.OPERATION,
                       config_sha256=pq.sha256(path), permissions=pq.PERMISSIONS,
                       source_sha256=config["source_sha256"])
        authority = self.root / "release.json"
        authority.write_text(json.dumps(release), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            pq.main(["--config", str(path), "--execute", "--release", str(authority),
                     "--trusted-release-sha256", pq.sha256(authority)])
        config["output_dir"] += "other"
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "binding required"):
            pq.admit(path, authority, pq.sha256(authority))
        self.assertFalse(Path(config["output_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
