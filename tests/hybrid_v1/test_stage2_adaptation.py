"""Synthetic-only phase isolation and real existing-kernel CPU checks.

Run only in a root-released CPU proof job. No real arrays/checkpoints are used.
"""
import copy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import stage2_adaptation as stage2


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return str(path)


def fixture(root):
    rng = np.random.default_rng(13)
    entries = []; points = {}
    for role in ("FIT", "SELECT", "CAL"):
        path = root/(role+".npz")
        values = rng.normal(size=(166, 2)).astype(np.float32)
        observed = np.ones(values.shape, bool); observed[2, 0] = False; values[2, 0] = np.nan
        np.savez(path, values=values, observed=observed)
        entries.append(dict(domain="synthetic", entity="fixture", run=role, role=role,
            path=str(path), format="npz", key="values", observed_key="observed",
            sha256=stage2.sha256(path), expected_shape=[166, 2], bounds=[0, 166]))
        if role != "FIT":
            points[role] = [dict(domain="synthetic", entity="fixture", run=role, role=role,
                index=t, footprint=[t-128, t+(1 if role == "CAL" else 32)])
                for t in ([128, 130] if role == "SELECT" else [128, 164, 165])]
    cfg = dict(operation="HYBRID_V1_STAGE2_U", scope="synthetic", mode="U", device="cpu",
        seed=2025, history=128, horizon=32, steps=2, lr=1e-4, kind="huber", weight_decay=.01,
        validate_every=1, select_count=2, cal_count=3, model={}, sources={},
        active_components=["r_tf", "r_ms", "r_pred"], output=str(root/"train"),
        permissions=stage2._permissions("target_train"))
    write(root/"manifest.json", {"runs": entries})
    write(root/"plans.json", dict(history=128, horizon=32, seed=2025, **points))
    write(root/"source_config.json", cfg)
    (root/"source.pt").write_bytes(b"metadata-fixture-not-a-model")
    write(root/"source_receipt.json", {"fixture": True})
    for key, name in (("manifest", "manifest.json"), ("plans", "plans.json"),
            ("source_config", "source_config.json"), ("source_checkpoint", "source.pt"),
            ("source_receipt", "source_receipt.json")):
        cfg[key] = str(root/name); cfg[key+"_sha256"] = stage2.sha256(root/name)
    return cfg


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.cfg = fixture(self.root)

    def check(self, cfg=None, phase="target_train"):
        cfg = self.cfg if cfg is None else cfg
        return stage2.metadata_preflight(cfg, phase, cfg["output"])

    def test_metadata_never_opens_npz_or_torch(self):
        with patch.object(np, "load", side_effect=AssertionError("array opened")):
            self.assertFalse(self.check()["target_arrays_opened"])

    def test_permission_route_source_hash_and_fresh_output(self):
        for key, value in (("mode", "F"), ("permissions", stage2._permissions("target_calibrate")),
                           ("source_checkpoint_sha256", "0"*64)):
            cfg = copy.deepcopy(self.cfg); cfg[key] = value
            with self.subTest(key=key), self.assertRaises((ValueError, PermissionError)):
                self.check(cfg)
        Path(self.cfg["output"]).mkdir()
        with self.assertRaises(PermissionError): self.check()

    def test_query_labels_missing_observed_and_role_overlap_rejected(self):
        original = json.loads(Path(self.cfg["manifest"]).read_text())
        for change in (lambda e: e.update(role="QUERY"), lambda e: e.update(label_path="NEVER_OPEN"),
                       lambda e: e.pop("observed_key"), lambda e: e.update(run="SELECT")):
            manifest = copy.deepcopy(original); change(manifest["runs"][0])
            write(Path(self.cfg["manifest"]), manifest)
            cfg = dict(self.cfg, manifest_sha256=stage2.sha256(self.cfg["manifest"]))
            with self.assertRaises((PermissionError, ValueError)): self.check(cfg)

    def test_plan_duplicates_wrong_axis_and_source_config_mismatch(self):
        plans = json.loads(Path(self.cfg["plans"]).read_text())
        plans["SELECT"][1] = plans["SELECT"][0]
        write(Path(self.cfg["plans"]), plans)
        cfg = dict(self.cfg, plans_sha256=stage2.sha256(self.cfg["plans"]))
        with self.assertRaises(ValueError): self.check(cfg)
        cfg = dict(self.cfg, history=512)
        with self.assertRaises(ValueError): self.check(cfg)

    def test_unreleased_action_rejected_before_target_load(self):
        path = write(self.root/"config.json", self.cfg)
        with patch.object(stage2, "load_manifest", side_effect=AssertionError("loaded")):
            with self.assertRaises(PermissionError):
                stage2.run_phase(path, None, None, "target_train", self.cfg["output"])
        self.assertFalse(Path(self.cfg["output"]).exists())

    def test_bound_release_metadata_and_changed_source_rejection(self):
        path = write(self.root/"config.json", self.cfg)
        own = {p.name: stage2.sha256(p) for p in Path(stage2.__file__).parent.glob("*.py")}
        release = dict(approved_by="root", status="APPROVED", allowed_actions=["target_train"],
                       config_sha256=stage2.sha256(path), source_sha256=own)
        location = write(self.root/"fixture_release.json", release)
        _, result = stage2.approval(path, location, stage2.sha256(location), "target_train", self.cfg["output"])
        self.assertFalse(result["training_executed"])
        release["source_sha256"]["stage2_adaptation.py"] = "0"*64
        write(Path(location), release)
        with self.assertRaises(PermissionError):
            stage2.approval(path, location, stage2.sha256(location), "target_train", self.cfg["output"])

    def test_wrong_original_plan_index_fails_before_model_load(self):
        plans = json.loads(Path(self.cfg["plans"]).read_text())
        plans["SELECT"][0]["index"] = 3
        write(Path(self.cfg["plans"]), plans)
        cfg = dict(self.cfg, plans_sha256=stage2.sha256(self.cfg["plans"]))
        self.check(cfg)
        with patch.object(stage2, "load_model", side_effect=AssertionError("model opened")):
            with self.assertRaises(ValueError): stage2._phase_action(cfg, "target_train", cfg["output"])

    def test_frozen_buffer_and_parameter_changes_fail(self):
        before = {n: {"sha256": "a"} for n in ("fusion.weight", "temporal.weight", "fusion.running_mean")}
        after = copy.deepcopy(before); after["fusion.weight"] = {"sha256": "b"}
        names = {"fusion.weight", "temporal.weight"}
        self.assertTrue(stage2.state_difference(before, after, names, train=True)["PASS"])
        self.assertFalse(stage2.state_difference(before, after, names)["PASS"])
        after["fusion.running_mean"] = {"sha256": "b"}
        self.assertEqual(stage2.state_difference(before, after, names, train=True)["forbidden_changes"], ["fusion.running_mean"])
        after["temporal.weight"] = {"sha256": "b"}
        self.assertIn("temporal.weight", stage2.state_difference(before, after, names, train=True)["forbidden_changes"])

    def test_sampler_replay_matches_original_and_preserves_global_rng(self):
        from hybrid_tsad.hybrid_v1.data import load_manifest, BalancedSampler
        runs = load_manifest(self.cfg["manifest"], roles={"FIT"})
        before = copy.deepcopy(np.random.get_state())
        trace = stage2.sampler_trace(runs, self.cfg)
        reference = BalancedSampler(runs, 128, 32, 2025)
        self.assertEqual([r["footprint"] for r in trace], [reference.sample()["footprint"] for _ in trace])
        self.assertEqual(trace, stage2.sampler_trace(runs, self.cfg))
        after = np.random.get_state()
        self.assertEqual(before[0], after[0]); np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])


@unittest.skipUnless(importlib.util.find_spec("torch"), "TORCH_UNAVAILABLE_REAL_KERNELS_NOT_TESTED")
class ActualKernelCPU(unittest.TestCase):
    def test_train_calibrate_and_calibration_only_exact_existing_kernels(self):
        import torch
        from hybrid_tsad.hybrid_v1 import HybridV1Config, OfficialSources
        from hybrid_tsad.hybrid_v1.cli import construct, load_model
        from hybrid_tsad.hybrid_v1.data import load_manifest
        from hybrid_tsad.hybrid_v1.training import adapt, iter_infer, calibrate_route
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = fixture(root)
            cfg["model"] = asdict(HybridV1Config(history=128, horizon=32, local_window=20, width=16,
                temporal_patch=16, temporal_layers=1, heads=2, tf_hidden=8, tf_repr=16,
                tf_depth=1, tf_patch=5, lag_window=20, max_lag=8, lag_width=8,
                lag_top_k=2, history_mask_probability=0., dropout=0.))
            # JSON canonicalizes tuple fields to arrays exactly as production configs.
            cfg["model"] = json.loads(json.dumps(cfg["model"]))
            cfg["sources"] = asdict(OfficialSources.from_workspace(os.environ["HYBRID_V1_WORKSPACE"]))
            source = construct(cfg); torch.save(source.state_dict(), cfg["source_checkpoint"])
            cfg["source_checkpoint_sha256"] = stage2.sha256(cfg["source_checkpoint"])
            write(Path(cfg["source_config"]), cfg)
            cfg["source_config_sha256"] = stage2.sha256(cfg["source_config"])
            source_bytes = Path(cfg["source_checkpoint"]).read_bytes()
            reference = load_model(cfg, cfg["source_checkpoint"])
            runs = load_manifest(cfg["manifest"], roles={"FIT", "SELECT"})
            plan = json.loads(Path(cfg["plans"]).read_text())
            reference, reference_trace = adapt(reference, "U", [r for r in runs if r.role == "FIT"],
                [r for r in runs if r.role == "SELECT"], history=128, horizon=32, steps=2,
                lr=1e-4, kind="huber", seed=2025, active_components=cfg["active_components"],
                validate_every=1, weight_decay=.01, validation_plan=plan["SELECT"])
            original_loader = stage2.load_manifest
            def restricted_loader(path, roles):
                self.assertEqual(roles, {"FIT", "SELECT"})
                original_np_load = np.load
                def allowed(p, **kwargs):
                    self.assertIn(Path(p).name, {"FIT.npz", "SELECT.npz"})
                    return original_np_load(p, **kwargs)
                with patch.object(np, "load", side_effect=allowed):
                    return original_loader(path, roles=roles)
            stage2.metadata_preflight(cfg, "target_train", cfg["output"])
            with patch.object(stage2, "load_manifest", side_effect=restricted_loader):
                receipt = stage2._phase_action(cfg, "target_train", cfg["output"])
            adapted = load_model(cfg, root/"train/adapted.pt")
            self.assertEqual(stage2.state_identity(reference), stage2.state_identity(adapted))
            actual_trace = json.loads((root/"train/training.json").read_text())
            for actual, expected in zip(actual_trace["steps"], reference_trace["steps"]):
                for key in ("loss", "SELECT_ssl", "gradient_norm", "parts", "footprint", "actual_mask_seed"):
                    self.assertEqual(actual[key], expected[key], key)
            self.assertEqual(receipt["updates"], 2)
            self.assertTrue(json.loads((root/"train/source_state_difference.json").read_text())["PASS"])
            self.assertTrue(json.loads((root/"train/adapted_state_difference.json").read_text())["PASS"])
            for mode in ("U", "Z_TCAL"):
                cal_cfg = copy.deepcopy(cfg); cal_cfg.update(mode=mode, output=str(root/(mode+"_cal")),
                    permissions=stage2._permissions("target_calibrate"),
                    adapted_checkpoint=str(root/"train/adapted.pt"), adapted_checkpoint_sha256=receipt["adapted_checkpoint_sha256"],
                    adapted_receipt=str(root/"train/receipt.json"), adapted_receipt_sha256=stage2.sha256(root/"train/receipt.json"))
                stage2.metadata_preflight(cal_cfg, "target_calibrate", cal_cfg["output"])
                def cal_loader(path, roles):
                    self.assertEqual(roles, {"CAL"})
                    original_np_load = np.load
                    def allowed(p, **kwargs):
                        self.assertEqual(Path(p).name, "CAL.npz")
                        return original_np_load(p, **kwargs)
                    with patch.object(np, "load", side_effect=allowed):
                        return original_loader(path, roles=roles)
                with patch.object(stage2, "load_manifest", side_effect=cal_loader), patch.object(stage2, "adapt", side_effect=AssertionError("CAL attempted update")):
                    result = stage2._phase_action(cal_cfg, "target_calibrate", cal_cfg["output"])
                self.assertEqual(result["updates"], 0); self.assertTrue(result["target_calibration_contact"])
                model = load_model(cfg, root/"train/adapted.pt" if mode == "U" else cfg["source_checkpoint"])
                raw = list(iter_infer(model, load_manifest(cfg["manifest"], roles={"CAL"}), history=128,
                    horizon=32, seed=2025, active_components=cfg["active_components"], index_plan=plan["CAL"]))
                expected = calibrate_route(raw, "U", active_components=cfg["active_components"])
                self.assertEqual(json.loads((Path(cal_cfg["output"])/"target_calibration.json").read_text()), expected)
                records = json.loads((Path(cal_cfg["output"])/"target_calibration_records.json").read_text())
                self.assertEqual([r["index"] for r in records], [128, 164, 165])  # includes P31 tail
            self.assertEqual(source_bytes, Path(cfg["source_checkpoint"]).read_bytes())


if __name__ == "__main__":
    unittest.main()
