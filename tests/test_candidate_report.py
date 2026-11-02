"""Tiny synthetic JSON only; no Torch, dataset, labels, score bank or GPU."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hybrid_tsad import candidate_report as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    files = {}
    write(root/"prepare/receipt.json", {"stage": "prepare", "protocol_sha256": "fixed-protocol", "from_scratch": True})
    metric = {k: None for k in report.METRICS}
    metric.update(point_Precision=.5, point_Recall=.25, point_F1=1/3, normal_FPR=.1, FP=2,
                  normal_exposure_points=20, normal_false_alarm_points=2, normal_alarm_fragments=1,
                  complete_event_recall_any_hit_aux=.5, one_to_one_event_precision=.5,
                  complete_event_hits=1, complete_events=2, original_events=2, predicted_events=2,
                  raw_threshold=3., delay_units="original sample points; no reliable physical clock",
                  Aff_F1_reason="NOT_EVALUATED_EXTERNAL_CALLBACK_REQUIRED",
                  affiliation_status="NOT_EVALUATED_EXTERNAL_CALLBACK_REQUIRED")
    events = [dict(event_id="E1", complete=True, any_hit=True, delay_sample_points=2,
                   delay_censored=False, censor_after_points=None),
              dict(event_id="E2", complete=True, any_hit=False, delay_sample_points=None,
                   delay_censored=True, censor_after_points=10)]
    for mode in report.MODES:
        value = {"ranking": {"AP": .42, "trapezoid_PR_AUC": .4, "ROC_AUC": .6,
                             "ranking_metric_NA_reason": None},
                 "streams": {s: {"metrics": copy.deepcopy(metric), "events": copy.deepcopy(events),
                                  "affinity": [], "alarms": []} for s in report.STREAMS},
                 "Range_VUS": {"status": "NOT_EVALUATED_OFFICIAL_DEPENDENCY_AND_FIXED_BUFFER_REQUIRED"},
                 "affiliation_provenance": None}
        files["evaluate/"+mode+".json"] = write(root/"evaluate"/(mode+".json"), value)
        write(root/("train_"+mode)/"receipt.json", {"stage": "train", "mode": mode, "updates": 200,
              "selected_step": 150, "normal_SELECT_MSE": .7, "protocol_sha256": "fixed-protocol"})
    write(root/"evaluate/receipt.json", {"stage": "evaluate", "labels_opened": True,
          "protocol_sha256": "fixed-protocol", "files": files})


class ReportTests(unittest.TestCase):
    def test_export_preserves_values_na_censoring_and_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            root, output = Path(d)/"run", Path(d)/"report"
            fixture(root)
            original = Path.read_bytes
            permitted = {"receipt.json", "tf.json", "lag.json", "concat.json", "gated.json", "candidate_report.py"}
            def guarded(path):
                self.assertIn(path.name, permitted, "Unexpected raw data/labels/weights read")
                return original(path)
            with patch.object(Path, "read_bytes", guarded):
                # Output hashing also reads the two newly written report artifacts.
                permitted.update({"metrics.csv", "SUMMARY.md"})
                manifest = report.export_report(root, output)
            with (output/"metrics.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0]["raw_AP"], "0.42")
            self.assertEqual(rows[0]["Aff_F1"], "NA")
            self.assertEqual(rows[0]["Aff_F1_reason"], "NOT_EVALUATED_EXTERNAL_CALLBACK_REQUIRED")
            self.assertEqual(rows[0]["point_F1"], str(1/3))
            self.assertTrue(json.loads(rows[0]["event_delays_and_censoring_json"])[1]["delay_censored"])
            self.assertEqual(json.loads(rows[0]["Range_VUS_json"])["status"],
                             "NOT_EVALUATED_OFFICIAL_DEPENDENCY_AND_FIXED_BUFFER_REQUIRED")
            self.assertEqual(len(manifest["input_sha256"]), 10)
            self.assertFalse(manifest["metrics_recomputed"])
            self.assertIn("from_scratch", (output/"SUMMARY.md").read_text(encoding="utf-8"))
            for name, expected in manifest["output_sha256"].items():
                self.assertEqual(report.digest((output/name).read_bytes()), expected)
            with self.assertRaises(FileExistsError):
                report.export_report(root, output)

    def test_missing_mode_and_changed_hash_rejected_without_output(self):
        for change in ("missing", "tamper", "receipt"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root, output = Path(d)/"run", Path(d)/"report"
                fixture(root)
                if change == "missing":
                    (root/"evaluate/tf.json").unlink()
                elif change == "tamper":
                    (root/"evaluate/lag.json").write_text("{}")
                else:
                    (root/"evaluate/receipt.json").unlink()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    report.export_report(root, output)
                self.assertFalse(output.exists())

    def test_incomplete_receipt_or_training_protocol_rejected(self):
        for change in ("receipt", "protocol", "schema"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root, output = Path(d)/"run", Path(d)/"report"
                fixture(root)
                if change == "receipt":
                    p = root/"evaluate/receipt.json"; obj = json.loads(p.read_text())
                    obj["files"].pop("evaluate/tf.json"); write(p, obj)
                elif change == "protocol":
                    p = root/"train_tf/receipt.json"; obj = json.loads(p.read_text())
                    obj["protocol_sha256"] = "different"; write(p, obj)
                else:
                    p = root/"evaluate/tf.json"; obj = json.loads(p.read_text())
                    obj["streams"]["strict_point"]["metrics"].pop("FP")
                    h = write(p, obj); p = root/"evaluate/receipt.json"
                    obj = json.loads(p.read_text()); obj["files"]["evaluate/tf.json"] = h; write(p, obj)
                with self.assertRaises(ValueError):
                    report.export_report(root, output)
                self.assertFalse(output.exists())

    def test_single_class_na_stays_na(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"run"; fixture(root)
            p = root/"evaluate/tf.json"; obj = json.loads(p.read_text())
            obj["ranking"].update(AP=None, trapezoid_PR_AUC=None, ROC_AUC=None,
                                  ranking_metric_NA_reason="SINGLE_CLASS_SCORED_LABELS")
            obj["streams"] = {s: {"status": "NA_SINGLE_CLASS"} for s in report.STREAMS}
            h = write(p, obj); p = root/"evaluate/receipt.json"
            obj = json.loads(p.read_text()); obj["files"]["evaluate/tf.json"] = h; write(p, obj)
            report.export_report(root, Path(d)/"report")
            with (Path(d)/"report/metrics.csv").open(encoding="utf-8-sig", newline="") as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["raw_AP"], "NA")
            self.assertEqual(row["point_F1"], "NA")
            self.assertEqual(row["stream_status"], "NA_SINGLE_CLASS")
            self.assertEqual(row["raw_ranking_metric_NA_reason"], "SINGLE_CLASS_SCORED_LABELS")

    def test_scope_uses_actual_updates_and_initialization(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/"run"; fixture(root)
            write(root/"prepare/receipt.json", {"stage": "prepare", "protocol_sha256": "fixed-protocol", "from_scratch": False})
            for mode, updates in zip(report.MODES, (300, 400, 500, 600)):
                p = root/("train_"+mode)/"receipt.json"; value = json.loads(p.read_text())
                value["updates"] = updates; write(p, value)
            p = root/"evaluate/receipt.json"; value = json.loads(p.read_text())
            value["scope"] = "Original recorded scope, unchanged."; write(p, value)
            output = Path(d)/"report"; report.export_report(root, output)
            text = (output/"SUMMARY.md").read_text(encoding="utf-8")
            self.assertIn("False", text)
            self.assertIn("tf=300; lag=400; concat=500; gated=600", text)
            self.assertIn(value["scope"], text)
            self.assertNotIn("200-update", text)
            self.assertNotIn("单seed", text)
            write(root/"prepare/receipt.json", {"stage": "prepare", "protocol_sha256": "fixed-protocol"})
            output = Path(d)/"unverified"; report.export_report(root, output)
            self.assertIn("not verified", (output/"SUMMARY.md").read_text(encoding="utf-8"))
            write(root/"prepare/receipt.json", {"stage": "prepare", "protocol_sha256": "different"})
            with self.assertRaises(ValueError):
                report.export_report(root, Path(d)/"bad")


if __name__ == "__main__":
    unittest.main()
