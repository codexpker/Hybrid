"""Engineering fixtures only: small NumPy arrays/explicit fake objects, no PT/Torch."""
import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np

# Import only the sidecar under test. Do not import a model package or Torch.
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("controlled_engineering_fixture",
    ROOT / "src/hybrid_tsad/hybrid_v1/controlled_mechanism.py")
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)


class ControlledMechanismEngineeringTests(unittest.TestCase):
    def pair(self, family, **kwargs):
        return cm.generate_pair(family, run_id=3, role="UNIT_FIXTURE", n=160, burn_in=80,
                                event=(64, 96), d0=2, d1=4, **kwargs)

    def test_pair_innovations_not_output_suffix_seed(self):
        a, ta = self.pair("cross_mixed_sign")
        b, tb = self.pair("cross_mixed_sign")
        renamed_outputs = {"renamed_A": a["A"], "renamed_N": a["N"]}
        np.testing.assert_array_equal(renamed_outputs["renamed_A"]["values"], b["A"]["values"])
        for key in ta["innovations"]:
            np.testing.assert_array_equal(ta["innovations"][key], tb["innovations"][key])
        self.assertEqual(ta["seeds"], tb["seeds"])
        self.assertNotEqual(cm.seed_for("self_lag", "UNIT_FIXTURE", 3, "self"),
                            cm.seed_for("self_lag", "OTHER_ROLE", 3, "self"))

    def test_truth_separate_values_observed_only_and_dtype(self):
        for family in cm.FAMILIES:
            observations, truth = self.pair(family)
            self.assertIn("labels_A", truth)
            for item in observations.values():
                self.assertEqual(set(item), {"values", "observed"})
                self.assertEqual(item["values"].dtype, np.float32)
                self.assertEqual(item["observed"].dtype, np.bool_)
                self.assertTrue(item["observed"].all())

    def test_weak_delta_and_outside_exact_sham(self):
        x, truth = self.pair("weak_amplitude")
        delta = x["A"]["values"] - x["N"]["values"]
        np.testing.assert_allclose(delta[64:96], .25, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(delta[:64], 0)
        np.testing.assert_array_equal(delta[96:], 0)
        n = x["N"]["values"]
        np.testing.assert_array_equal(np.concatenate((n[:64], n[64:96], n[96:])), n)

    def test_self_continuous_recurrence_same_innovations_no_reset(self):
        x, t = self.pair("self_lag")
        qn, qa, eps = t["latents"]["normal_self"], t["latents"]["changed_self"], t["innovations"]["self"]
        indices = t["retained_indices"]
        start, end = indices[64], indices[95] + 1
        for k in (start - 1, start, start + 1, end - 1, end, end + 1):
            delay = 4 if start <= k < end else 2
            self.assertAlmostEqual(qa[k], .8 * qa[k - delay] + .6 * eps[k], places=14)
            self.assertAlmostEqual(qn[k], .8 * qn[k - 2] + .6 * eps[k], places=14)
        np.testing.assert_array_equal(x["N"]["values"][:64], x["A"]["values"][:64])
        self.assertTrue(np.any(x["N"]["values"][96:100] != x["A"]["values"][96:100]))

    def test_cross_positive_negative_and_legal_nonwrap_exact_equations(self):
        x, t = self.pair("cross_mixed_sign")
        u, zp, zm = [t["latents"][k] for k in ("u", "zplus", "zminus")]
        ix = t["retained_indices"]
        for variant, lag in (("N", t["normal_lag"]), ("A", t["changed_lag"])):
            self.assertGreaterEqual(int((ix - lag).min()), 0)
            expected = np.column_stack((u[ix], .3 * u[ix - lag] + np.sqrt(.91) * zp[ix],
                                        -.3 * u[ix - lag] + np.sqrt(.91) * zm[ix])).astype(np.float32)
            np.testing.assert_array_equal(x[variant]["values"], expected)
        np.testing.assert_array_equal(x["A"]["values"][:, 0], x["N"]["values"][:, 0])
        np.testing.assert_array_equal(x["A"]["values"][:64], x["N"]["values"][:64])
        np.testing.assert_array_equal(x["A"]["values"][96:], x["N"]["values"][96:])

    def test_generation_preserves_global_rng(self):
        before = np.random.get_state()
        for family in cm.FAMILIES:
            self.pair(family)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_generation_geometry_refused(self):
        with self.assertRaises(ValueError):
            cm.generate_pair("self_lag", run_id=0, role="UNIT", n=20, burn_in=2,
                             event=(8, 16), d0=2, d1=4)
        with self.assertRaises(ValueError):
            cm.seed_for("not_a_family", "UNIT", 0, "u")

    def test_singleton_next_then_stop_no_scan(self):
        seen = []
        def original_iterator_fixture():
            seen.append(2048)
            yield {"index": 2048}
        self.assertEqual(cm.singleton_record(original_iterator_fixture(), 2048)["index"], 2048)
        self.assertEqual(seen, [2048])

    def test_singleton_wrong_extra_or_missing_refused(self):
        with self.assertRaises(ValueError):
            cm.singleton_record(iter([{"index": 2047}]), 2048)
        with self.assertRaises(ValueError):
            cm.singleton_record(iter([{"index": 2048}, {"index": 2049}]), 2048)
        with self.assertRaises(StopIteration):
            cm.singleton_record(iter([]), 2048)

    def test_fake_call_counts_7_7_17_and_instance_restored(self):
        # Explicit engineering fake, not a replacement for real forward parity.
        class Fake:
            def _encode(self): return "unchanged"
            def forecast(self, c): return self._encode()
            def reconstruct(self, c): return [self._encode() for _ in range(min(2, c))]
        model = Fake(); before = dict(model.__dict__)
        with cm.count_calls(model) as counts:
            for channels in [1, 1, 1, 1, 3, 3, 3]:
                self.assertEqual(model.forecast(channels), "unchanged")
                model.reconstruct(channels)
        self.assertEqual(counts, {"forecast": 7, "reconstruct": 7, "_encode": 17})
        self.assertEqual(model.__dict__, before)
        with self.assertRaises(RuntimeError):
            with cm.count_calls(model): raise RuntimeError("fixture")
        self.assertEqual(model.__dict__, before)

    def test_detach_copy_no_extra_computation(self):
        class ExplicitFakeTensor:
            def __init__(self): self.calls = []; self.value = np.array([1., 2.], np.float32)
            def detach(self): self.calls.append("detach"); return self
            def cpu(self): self.calls.append("cpu"); return self
            def numpy(self): self.calls.append("numpy"); return self.value
        tensor = ExplicitFakeTensor(); before = np.random.get_state()
        result = cm.detached({"route": tensor})
        self.assertEqual(tensor.calls, ["detach", "cpu", "numpy"])
        result["route"][0] = 99
        self.assertEqual(tensor.value[0], 1.)
        np.testing.assert_array_equal(before[1], np.random.get_state()[1])

    def test_bundle_raw_dtype_nan_and_fresh_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            a = np.array([1, np.nan], np.float32); mask = np.array([True, False])
            cm.save_bundle(folder, "raw", {"a": a, "mask": mask})
            meta = json.loads((pathlib.Path(folder) / "raw.json").read_text())
            with np.load(pathlib.Path(folder) / "raw.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z[meta["mask"]["npz_key"]], mask)
                self.assertEqual(z[meta["a"]["npz_key"]].dtype, np.float32)
                self.assertTrue(np.isnan(z[meta["a"]["npz_key"]][1]))
            with self.assertRaises(FileExistsError): cm.save_bundle(folder, "raw", {})

    def axes_fixture(self, channels):
        cfg = {"history": 16, "max_lag": 4, "lag_window": 8, "lag_min_overlap": 2}
        shape = (1, channels, channels, 5)
        eye = np.eye(channels, dtype=bool)[None, :, :, None]
        self_ok = np.broadcast_to(eye & (np.arange(5)[None, None, None] > 0), shape).copy()
        cross_ok = np.broadcast_to(~eye, shape).copy()
        union = self_ok | cross_ok
        score = np.where(union, 0., -np.inf)
        count = np.broadcast_to(8 - np.arange(5)[None, None, None], shape).astype(float).copy()
        meta = {"correlation": score, "overlap_count": count, "self_candidate_valid": self_ok,
                "cross_candidate_valid": cross_ok, "source_sample_offset": 15-np.arange(5),
                "target_axis": "i", "source_axis": "j", "lag_unit": "original_sample", "circular": False}
        for branch in ("self", "cross"):
            source = (np.arange(channels) if branch == "self" else (np.arange(channels)+1) % channels)[None, :, None]
            delay = np.full((1, channels, 1), 1, np.int64)
            valid = np.ones(delay.shape, bool) if branch == "self" or channels > 1 else np.zeros(delay.shape, bool)
            meta[branch] = {"source": source, "lag": delay, "selected_valid": valid,
                            "weights": valid.astype(float), "valid": valid.any(-1)}
        return meta, cfg

    def test_axes_valid_c1_c3_invalid_inf_preserved_flags(self):
        for channels in (1, 3):
            meta, cfg = self.axes_fixture(channels)
            before = meta["correlation"].copy()
            report = cm.validate_lag_axes(meta, cfg, channels)
            self.assertTrue(report["self_enabled"])
            np.testing.assert_array_equal(meta["correlation"], before)
            self.assertGreater(report["invalid_scores_preserved"], 0)

    def test_axes_illegal_self_zero_cross_self_offset_and_weights_rejected(self):
        for issue in ("selfzero", "crossself", "offset", "weights", "valid_nan"):
            meta, cfg = self.axes_fixture(3)
            if issue == "selfzero": meta["self_candidate_valid"][0, 0, 0, 0] = True
            if issue == "crossself": meta["cross_candidate_valid"][0, 0, 0, 1] = True
            if issue == "offset": meta["source_sample_offset"][0] += 1
            if issue == "weights": meta["self"]["weights"][0, 0, 0] = -.1
            if issue == "valid_nan": meta["correlation"][0, 0, 0, 1] = np.nan
            with self.subTest(issue=issue), self.assertRaises(AssertionError):
                cm.validate_lag_axes(meta, cfg, 3)

    def test_required_components_finite_auxiliary_nan_not_filled(self):
        parts = {"scalar": {}, "per_channel": {}}
        for key in ("r_tf", "r_ms", "r_pred", "r_var"):
            parts["scalar"][key] = np.array([np.nan if key == "r_var" else 1.])
            parts["per_channel"][key] = np.array([[np.nan if key == "r_var" else 1.]])
        cm.validate_components(parts)
        self.assertTrue(np.isnan(parts["scalar"]["r_var"][0]))
        parts["scalar"]["r_pred"][0] = np.nan
        with self.assertRaises(AssertionError): cm.validate_components(parts)

    def parity_fixture(self):
        def pack(a):
            return {"shape": list(a.shape), "packbits_hex": np.packbits(a.reshape(-1), bitorder="little").tobytes().hex(),
                    "unpacked_sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        stats = {"mean": np.zeros((1, 1, 1)), "scale": np.ones((1, 1, 1)),
                 "channel_valid": np.ones((1, 1, 1), bool)}
        f = {"prediction": np.zeros((1, 2, 1)), "stats": stats, "valid": np.ones((1, 2, 1), bool),
             "copy_masks": np.zeros((1, 2, 1, 1), bool), "tf_order": np.array([.5])}
        r = {"hidden_mask": np.ones((1, 1, 4, 1), bool), "visible_mask": np.zeros((1, 1, 4, 1), bool),
             "transform_hidden_mask": np.zeros((1, 1, 2, 1, 1), bool), "order": np.array([[.5]]),
             "tf_reconstruction": np.zeros((1, 1, 2, 4, 1)), "tf_valid": np.ones((1, 1, 2, 4, 1), bool),
             "ms_reconstruction_by_scale": np.zeros((1, 1, 2, 2, 4, 1)),
             "ms_valid_by_scale": np.ones((1, 1, 2, 2, 4, 1), bool), "stats": stats}
        b = {"index": 2048, "available_index": 2048, "history": np.zeros((1, 4, 1)),
             "footprint": [2044, 2049], "current_target": np.ones((1, 1)), "current_observed": np.ones((1, 1), bool)}
        parts = {"scalar": {"r_tf": np.array([0.])}, "per_channel": {"r_tf": np.zeros((1, 1))}}
        direct = {"forecast": f, "reconstruction": r, "batch": b, "parts": parts,
                  "actual_seed": 4073, "input_identity": {"fixture": True}}
        masks = {"hidden_mask": r["hidden_mask"], "visible_mask": r["visible_mask"],
                 "copy_masks": r["transform_hidden_mask"], "forecast_copy_masks": f["copy_masks"],
                 "current_observed": b["current_observed"], "forecast_valid": f["valid"],
                 "tf_endpoint_valid": r["tf_valid"][:, :, :, -1],
                 "ms_endpoint_valid_by_scale": r["ms_valid_by_scale"][:, :, :, :, -1]}
        replay = {k: pack(v) for k, v in masks.items()}
        replay.update(actual_mask_seed=4073, input_footprint=[2044, 2049], forecast_history_bounds=[2044, 2048],
                      reconstruction_bounds=[2045, 2049], group_mask_seeds=[4073], base_seed=2025,
                      unarrived_future_observed=False, input_identity={"fixture": True}, order=r["order"],
                      forecast_order=f["tf_order"], current_target=b["current_target"],
                      tf_endpoint_by_group_copy=r["tf_reconstruction"][:, :, :, -1],
                      ms_endpoint_by_group_copy_scale=r["ms_reconstruction_by_scale"][:, :, :, :, -1],
                      stats=stats, forecast_stats=stats)
        ref = {"index": 2048, "available_index": 2048, "forecast_issued_index": 2047,
               "prediction": f["prediction"][0], "r_tf": 0., "per_channel": {"r_tf": np.zeros(1)}, "replay": replay}
        return direct, ref

    def test_reference_exact_masks_footprint_available_and_group_seeds(self):
        direct, ref = self.parity_fixture()
        cm.compare_reference(direct, ref)
        for issue in ("available", "footprint", "groupseed", "mask"):
            bad = copy.deepcopy(ref)
            if issue == "available": bad["available_index"] += 1
            if issue == "footprint": bad["replay"]["input_footprint"][0] += 1
            if issue == "groupseed": bad["replay"]["group_mask_seeds"] = [999]
            if issue == "mask": bad["replay"]["forecast_valid"]["packbits_hex"] = "00"
            with self.subTest(issue=issue), self.assertRaises((AssertionError, ValueError)):
                cm.compare_reference(direct, bad)

    def gate_fixture(self, root):
        cfg = json.loads((ROOT / "configs/hybrid_v1/controlled-mechanism-cpu-l0.example.json").read_text(encoding="utf-8"))
        cfg["status"] = "ROOT_BOUND"
        cfg["permissions"].update(prepare_only=False, generate_arrays_now=True, forward_now=True)
        cfg["output_proposed"] = "fresh_fixture_output"
        source = root / "frozen" / "hybrid_tsad" / "hybrid_v1"; source.mkdir(parents=True)
        f = source / "__init__.py"; f.write_text("# metadata-only engineering fixture, never imported\n")
        cfg["model_source"] = {"path": "frozen", "sha256_map": {"__init__.py": cm.sha(f)}}
        cp, rp = root / "config.json", root / "release.json"
        cp.write_text(json.dumps(cfg))
        release = {"status": "APPROVED", "approved_by": "root", "operation": cm.OPERATION,
                   "allowed_actions": ["cpu_l0"], "config_sha256": cm.sha(cp),
                   "sidecar_sha256": cm.sha(cm.__file__), "source_sha256": cfg["model_source"]["sha256_map"],
                   "resources": cfg["resources_candidate"]}
        rp.write_text(json.dumps(release))
        return cfg, release, cp, rp, source

    def test_metadata_gate_valid_no_assets_or_torch_opened(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder); cfg, release, cp, rp, source = self.gate_fixture(root)
            result, output, src = cm.admit(cp, rp, cm.sha(rp), root)
            self.assertEqual(src, source)
            self.assertFalse(output.exists())
            self.assertFalse((root / "BIND_ACCEPTED_SOURCE_PT").exists())
            self.assertNotIn("torch", sys.modules)

    def test_missing_root_engineering_only_and_wrong_signature_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder); cfg, release, cp, rp, source = self.gate_fixture(root)
            with self.assertRaises(PermissionError): cm.admit(cp, None, None, root)
            with self.assertRaises(PermissionError): cm.admit(cp, rp, "0" * 64, root)
            release["status"] = "ROOT_APPROVED_LIMITED_SIDECAR_ENGINEERING_ONLY"
            rp.write_text(json.dumps(release))
            with self.assertRaises(PermissionError): cm.admit(cp, rp, cm.sha(rp), root)

    def test_config_source_fresh_and_multiple_reference_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder); cfg, release, cp, rp, source = self.gate_fixture(root)
            cfg["source_seed"] = 2026; cp.write_text(json.dumps(cfg))
            with self.assertRaises(PermissionError): cm.admit(cp, rp, cm.sha(rp), root)
            cfg["source_seed"] = 2025; cp.write_text(json.dumps(cfg)); release["config_sha256"] = cm.sha(cp)
            rp.write_text(json.dumps(release)); (source / "__init__.py").write_text("changed")
            with self.assertRaises(PermissionError): cm.admit(cp, rp, cm.sha(rp), root)
            (source / "__init__.py").write_text("# metadata-only engineering fixture, never imported\n")
            (root / "fresh_fixture_output").mkdir()
            with self.assertRaises(FileExistsError): cm.admit(cp, rp, cm.sha(rp), root)
            (root / "fresh_fixture_output").rmdir()
            cfg["L0_parity_index_plan"].append(copy.deepcopy(cfg["L0_parity_index_plan"][0]))
            cp.write_text(json.dumps(cfg)); release["config_sha256"] = cm.sha(cp); rp.write_text(json.dumps(release))
            with self.assertRaises(ValueError): cm.admit(cp, rp, cm.sha(rp), root)

    def test_no_torch_import_and_no_formal_matrix_in_engineering(self):
        self.assertNotIn("torch", sys.modules)
        observations, truth = self.pair("weak_amplitude")
        self.assertEqual(observations["N"]["values"].shape, (160, 1))
        self.assertNotEqual(len(observations["N"]["values"]), 4096)


if __name__ == "__main__":
    unittest.main()
