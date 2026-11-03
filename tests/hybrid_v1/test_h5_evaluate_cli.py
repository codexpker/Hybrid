"""Synthetic H5 stages; production contract never accepts a toy profile/axis."""
import contextlib
import copy
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import h5_evaluate_cli as h
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank, IDENTITY_KEYS
from hybrid_tsad.hybrid_v1.scoring import apply_calibration, recipe


def production_metadata(root):
    cfg = dict(profile=h.PROFILE, history=512, original_length=87841, original_start=0, expected_records=87329,
               stream=["PSM", "anonymous_PSM", "test_original"], model=json.loads(json.dumps(h.HybridV1Config().to_dict())),
               sources={"synthetic_source_metadata": "never imported"}, comparisons=h.fixed_pairs(), statistics=dict(h.STATISTICS_NA),
               output=str(root / "out"), raw_seal=dict(path=str(root / "seal.json"), sha256=None),
               affiliation_identity=str(root / "aff_identity.json"), affiliation_identity_sha256=h.AFF_SHA,
               labels=dict(path=str(root / "labels.csv"), sha256=h.named.LABEL_SHA, shape=[87841], format="csv",
                           drop_columns=["timestamp_(min)"], value_columns=["label"]), banks=[])
    for cell in h.CELLS:
        base = cell == "base0_huber"
        identity = {k: "a" * 64 for k in IDENTITY_KEYS}
        identity["input_sha256"] = h.named.INPUT_SHA
        if base:
            identity.update(checkpoint_sha256=h.BASE_CHECKPOINT, calibration_sha256=h.BASE_CAL)
        entry = dict(id=cell, kind="source_model", control_id="full" if base else "H5_" + cell,
                     reused_raw=base, training_kind="huber" if cell.endswith("_huber") else "half_mse",
                     calibration_origin="source", adaptation_mode="Z", training_seed=2025, inference_seed=2025,
                     active_components=list(h.ACTIVE), directory=str(root / cell), identity=identity,
                     ranges=[dict(zip(("domain", "entity", "run"), cfg["stream"]), start=512, end=87841)],
                     manifest_sha256=h.BASE_MANIFEST if base else "b" * 64)
        entry.update({k: dict(path=str(root / (cell + "_" + k + ".json")), sha256="c" * 64) for k in h.META})
        cfg["banks"].append(entry)
    return cfg


def write_json(path, value):
    path = Path(path); path.write_text(json.dumps(value))
    return dict(path=str(path), sha256=h.sha(path))


def seal(cfg):
    cfg["raw_seal"] = write_json(cfg["raw_seal"]["path"],
                                dict(status="ACCEPTED_H5_FIXED_BANKS", approved_by="root", profile=cfg["profile"], banks=cfg["banks"]))


def tiny_stages(root, bad_last_clock=False):
    """Real tiny bank/metadata stages, intentionally inadmissible at CLI contract."""
    cfg = production_metadata(root)
    cfg.update(history=2, original_length=18, expected_records=16, stream=["SYN", "E", "R"], affiliation_identity=None)
    labels = np.zeros(18, np.int8)
    for a, b in ((0, 3), (5, 7), (8, 11), (12, 14), (16, 18)):
        labels[a:b] = 1
    cfg["labels"] = write_json(root / "synthetic_labels.json", labels.tolist())
    ranges = [dict(domain="SYN", entity="E", run="R", start=a, end=b) for a, b in ((2, 8), (10, 18))]
    for number, e in enumerate(cfg["banks"]):
        cal = dict(scales={k: 1. for k in h.ACTIVE}, threshold=.5 + number * .01,
                   source_domain_equal_weight=True, domains=["SMD", "Sunspot"],
                   active_components=list(h.ACTIVE), weights=recipe(h.ACTIVE)[1])
        e["calibration"] = write_json(e["calibration"]["path"], cal)
        e["identity"]["calibration_sha256"] = e["calibration"]["sha256"]
        e["ranges"] = copy.deepcopy(ranges)
        train = dict(operation="HYBRID_V1_SOURCE_STAGE1", seed=2025, history=512, horizon=32,
                     kind=e["training_kind"], model=cfg["model"], sources=cfg["sources"], active_components=list(h.ACTIVE))
        e["training_config"] = write_json(e["training_config"]["path"], train)
        e["training_receipt"] = write_json(e["training_receipt"]["path"], dict(phase="source_train", synthetic_non_scientific_results=False,
            files={"source.pt": e["identity"]["checkpoint_sha256"], "config.json": e["training_config"]["sha256"]}))
        calcfg = dict(train, checkpoint_sha256=e["identity"]["checkpoint_sha256"], train_receipt_sha256=e["training_receipt"]["sha256"])
        e["calibration_config"] = write_json(e["calibration_config"]["path"], calcfg)
        e["calibration_receipt"] = write_json(e["calibration_receipt"]["path"], dict(phase="calibrate_source", synthetic_non_scientific_results=False,
            files={"source_calibration.json": e["calibration"]["sha256"], "config.json": e["calibration_config"]["sha256"]}))
        with ChunkedBank(e["directory"], identity=e["identity"], ranges=ranges, threshold=cal["threshold"], chunk_size=64) as bank:
            for span in ranges:
                for i in range(span["start"], span["end"]):
                    raw = dict(domain="SYN", entity="E", run="R", index=i, available_index=i,
                               **{k: (9. if i in (5, 6, 14) else .1) for k in h.ACTIVE})
                    if i == 12:
                        raw["r_pred"] = float("nan")
                    raw.update(apply_calibration(raw, cal)); raw["reason"] = raw.get("reason")
                    if bad_last_clock and number == 5 and i == 17:
                        raw["available_index"] = 18
                    bank.append(raw)
            bank.finalize()
        e["manifest_sha256"] = h.sha(Path(e["directory"]) / "manifest.json")
    seal(cfg)
    return cfg, labels


class H5Evaluation(unittest.TestCase):
    def test_fixed_six_nine_production_contract_no_body(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = production_metadata(Path(d))
            with patch("pathlib.Path.open", side_effect=AssertionError("metadata contract must not read")):
                h.contract(cfg)
            self.assertEqual(len(h.fixed_pairs()), 9)
            self.assertEqual(h.fixed_pairs()[-1]["candidate"], "normal_record10_huber")

    def test_missing_cell_pair_fake_seed_target_and_other_profile_rejected(self):
        for change in ("cell", "pair", "seed", "target", "kind", "profile", "mean", "axis", "base_reuse", "label_schema", "aff"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                cfg = production_metadata(Path(d))
                if change == "cell": cfg["banks"].pop()
                elif change == "pair": cfg["comparisons"].pop()
                elif change == "seed": cfg["banks"][1]["training_seed"] = 2026
                elif change == "target": cfg["banks"][1]["calibration_origin"] = "target"
                elif change == "kind": cfg["banks"][1]["training_kind"] = "huber"
                elif change == "profile": cfg["profile"] = "SYNTHETIC_H5"
                elif change == "mean": cfg["model_mean_SD"] = True
                elif change == "axis": cfg["banks"][1]["ranges"][0]["start"] = 513
                elif change == "base_reuse": cfg["banks"][0]["manifest_sha256"] = "0" * 64
                elif change == "label_schema": cfg["labels"]["shape"] = [87329]
                else: cfg["affiliation_identity_sha256"] = "0" * 64
                with patch("pathlib.Path.open", side_effect=AssertionError("no file I/O")):
                    with self.assertRaises((ValueError, PermissionError)): h.contract(cfg)

    def test_pending_changed_seal_rejected_before_labels(self):
        for change in ("pending", "changed"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                cfg = production_metadata(Path(d)); seal(cfg)
                if change == "pending": cfg["raw_seal"] = write_json(cfg["raw_seal"]["path"], dict(status="PENDING"))
                else: Path(cfg["raw_seal"]["path"]).write_text("{}")
                with patch.object(h, "load_labels", side_effect=AssertionError("no label read")) as loader:
                    with self.assertRaises(PermissionError): h.run_evaluation(cfg)
                    loader.assert_not_called()

    def test_all_label_aliases_rejected_before_hash_or_body(self):
        for key in ("raw_seal", "calibration", "training_config", "calibration_receipt", "manifest", "affiliation"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
                cfg = production_metadata(Path(d)); target = cfg["labels"]["path"]
                if key == "raw_seal": cfg["raw_seal"]["path"] = target
                elif key == "manifest": cfg["labels"]["path"] = str(Path(cfg["banks"][5]["directory"]) / "manifest.json")
                elif key == "affiliation": cfg["affiliation_identity"] = target
                else: cfg["banks"][5][key]["path"] = target
                with patch.object(h, "bound", side_effect=AssertionError("alias must fail before hashing")):
                    with self.assertRaises(PermissionError): h.bank_gates(cfg)

    def test_real_metadata_lineage_ownCAL_and_chunk_failure(self):
        for change in ("none", "target_phase", "parent", "CAL_weights", "chunk_alias"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                cfg, _ = tiny_stages(Path(d)); e = cfg["banks"][5]
                if change == "target_phase":
                    receipt = json.loads(Path(e["calibration_receipt"]["path"]).read_text()); receipt["phase"] = "target_calibrate"
                    e["calibration_receipt"] = write_json(e["calibration_receipt"]["path"], receipt)
                elif change == "parent":
                    c = json.loads(Path(e["calibration_config"]["path"]).read_text()); c["checkpoint_sha256"] = "0" * 64
                    e["calibration_config"] = write_json(e["calibration_config"]["path"], c)
                elif change == "CAL_weights":
                    c = json.loads(Path(e["calibration"]["path"]).read_text()); c["weights"]["r_pred"] = .6
                    e["calibration"] = write_json(e["calibration"]["path"], c)
                elif change == "chunk_alias":
                    mp = Path(e["directory"]) / "manifest.json"; m = json.loads(mp.read_text()); m["chunks"][0]["file"] = cfg["labels"]["path"]
                    write_json(mp, m); e["manifest_sha256"] = h.sha(mp)
                seal(cfg)
                if change == "none":
                    self.assertEqual(len(h._authenticated_banks(cfg)), 6)
                    with self.assertRaises(ValueError): h.contract(cfg)  # Toy cannot enter production.
                else:
                    with self.assertRaises((ValueError, PermissionError)): h._authenticated_banks(cfg)

    def test_six_method_nine_pair_math_parity_and_censor_counterexamples(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, labels = tiny_stages(Path(d)); gates = h._authenticated_banks(cfg); arrays = h.scalar_inputs(cfg)
            result = h._save_results(cfg, gates, arrays, labels)
            from hybrid_tsad.hybrid_v1.evaluation import evaluate_bank_arrays
            for e in cfg["banks"]:
                threshold = next(g["threshold"] for g in gates if g["id"] == e["id"])
                direct = evaluate_bank_arrays(e["directory"], labels, threshold, expected_manifest_sha256=e["manifest_sha256"],
                                              original_length=18, warmup_points=2, stream=cfg["stream"])
                saved = json.loads((Path(cfg["output"]) / e["id"] / "result.json").read_text())
                self.assertEqual(saved, direct)
                self.assertIsNotNone(saved["ranking"]["AP"], "Real sklearn AP must execute in CPU acceptance")
                self.assertTrue(np.isfinite(saved["ranking"]["AP"]))
                self.assertEqual(saved["coverage"]["missing_arrived_raw_records"], 2)
                self.assertEqual(saved["coverage"]["invalid_raw_records"], 1)
                self.assertEqual(saved["coverage"]["main_eligible_events"], 1)
                self.assertEqual(saved["event_coverage"][0]["scored_points"], 1)
                self.assertTrue(saved["event_coverage"][-1]["boundary_censored"])
                self.assertIsNone(saved["point_coverage"][8]["strict_point"])
                self.assertIsNone(saved["point_coverage"][12]["strict_point"])
            self.assertEqual(len(result["methods"]), 6); self.assertEqual(len(result["comparisons"]), 9)
            self.assertEqual(len(list((Path(cfg["output"]) / "pairs").glob("*/paired_events.csv"))), 9)
            self.assertTrue(all(p["common_point_effect"]["common_valid_points"] == 13 for p in result["comparisons"]))
            self.assertIsNone(result["model_seed_mean"])

    def test_all_scalar_replays_finish_before_label_loader(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, _ = tiny_stages(Path(d), bad_last_clock=True); gates = h._authenticated_banks(cfg)
            # Instrument this new orchestration boundary, never an old profile guard.
            with patch.object(h, "bank_gates", return_value=gates), patch.object(h, "load_labels", side_effect=AssertionError("label read too early")) as loader:
                with self.assertRaises(ValueError): h.run_evaluation(cfg)
                loader.assert_not_called()

    def test_preview_and_cold_import_no_torch(self):
        with patch("pathlib.Path.open", side_effect=AssertionError("preview no I/O")), contextlib.redirect_stdout(io.StringIO()):
            h.main(["--config", "DOES_NOT_EXIST"])
        src = str(Path(h.__file__).resolve().parents[2])
        code = "import sys;sys.path.insert(0," + repr(src) + ");import hybrid_tsad.hybrid_v1.h5_evaluate_cli;assert 'torch' not in sys.modules;print('H5_COLD')"
        r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("H5_COLD", r.stdout)

    def test_signed_metadata_admit_and_changed_code_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cfg = production_metadata(root); seal(cfg); base = Path(h.__file__).resolve().parents[2]
            required = h.named.REQUIRED_SOURCE | {"hybrid_tsad/hybrid_v1/" + n for n in
                ("h5_evaluate_cli.py", "bank_recalibration.py", "named_evaluate_cli.py", "scoring.py", "config.py")}
            cfg.update(operation=h.OPERATION, permissions=dict(labels=True, evaluation=True, training=False, target_fit=False, GPU=False),
                       source_sha256={k: h.sha(base / k) for k in required})
            config = root / "config.json"; release = root / "release.json"
            def signed():
                write_json(config, cfg)
                return write_json(release, dict(status="APPROVED", approved_by="root", operation=h.OPERATION,
                    allowed_actions=[h.ACTION], allow_labels=True, config_sha256=h.sha(config),
                    raw_seal_sha256=cfg["raw_seal"]["sha256"], source_sha256=cfg["source_sha256"]))["sha256"]
            self.assertEqual(h.admit(config, release, signed())["profile"], h.PROFILE)
            cfg["source_sha256"]["hybrid_tsad/hybrid_v1/h5_evaluate_cli.py"] = "0" * 64
            with self.assertRaises(PermissionError): h.admit(config, release, signed())


if __name__ == "__main__":
    unittest.main()
