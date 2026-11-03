"""Small NumPy/stdlib engineering fixtures. No Torch import, PT or formal wave."""
import copy
import ast
import hashlib
import importlib
import importlib.util
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
PACKAGE = "_controlled_wave_engineering"
pkg = types.ModuleType(PACKAGE); pkg.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = pkg
wave = importlib.import_module(PACKAGE+".controlled_wave")
data = importlib.import_module(PACKAGE+".data")
scoring = importlib.import_module(PACKAGE+".scoring")
bank = importlib.import_module(PACKAGE+".chunked_bank")


def fixture_protocol():
    # Never generate the formal N2560/run1. This scaled data is disposable.
    cases = []
    def add(case, family, variant, channels, event=None, **kw):
        cases.append(dict(id=case, family=family, variant=variant, C=channels,
            event=event, matched_interval=[24, 64], seed_role="ENGINEERING_FIXTURE",
            loader_role="QUERY", models=["full2025", "full2026", "full2027"], **kw))
    add("WEAK_N", "weak_amplitude", "N", 1)
    add("WEAK_LEVEL025", "weak_amplitude", "LEVEL025", 1, [24, 64], amplitude=.25)
    add("WEAK_PULSE025", "weak_amplitude", "PULSE025", 1, [30, 34], amplitude=.25)
    add("STRONG_PULSE2", "weak_amplitude", "PULSE2", 1, [30, 34], amplitude=2.)
    add("SELF_N", "self_lag", "N", 1)
    add("SELF_A", "self_lag", "A", 1, [24, 64])
    add("SELF_ALLOWED", "self_lag", "A_all_normal_ledger", 1)
    cases[-1].update(seed_role="ENGINEERING_ALLOWED", allowed_switch_interval=[24, 64])
    add("CROSS_N", "cross_mixed_sign", "N", 3)
    add("CROSS_A", "cross_mixed_sign", "A", 3, [24, 64])
    add("CROSS_BREAK", "cross_mixed_sign", "BREAK", 3, [24, 64])
    add("CROSS_ALLOWED", "cross_mixed_sign", "A_all_normal_ledger", 3)
    cases[-1].update(seed_role="ENGINEERING_ALLOWED", allowed_switch_interval=[24, 64])
    return dict(N=96, history=16, horizon=4, burn_in=80, data_run_id=7,
        normal_fixed_delay=2, changed_delay=4, long_interval=[24, 64], cases=cases,
        shams=[dict(id="S0", alias_input_and_bank="WEAK_N", extra_forward=False)],
        semantic_identical_input_counterexamples=[dict(source="SELF_A", alternate_all_normal_truth=True)])


def calibration(active=("r_tf", "r_ms", "r_pred")):
    return dict(active_components=list(active), weights=scoring.recipe(active)[1],
        scales={k: 1. for k in active}, threshold=.5, domains=["SMD", "Sunspot"], source_domain_equal_weight=True)


def row(t, tf=float("nan")):
    return dict(domain=wave.DOMAIN, entity="fixture", run="small", index=t,
        available_index=t, r_tf=tf, r_ms=.2, r_pred=.3,
        per_channel=dict(r_tf=[tf], r_ms=[.2], r_pred=[.3]))


class FakeConfig:
    history = 16
    def to_dict(self): return dict(tf_enabled=True, self_lag_enabled=True, cross_lag_enabled=True)


class FakeModel:
    """Engineering fake only: this is NOT an official neural forward test."""
    config = FakeConfig()
    def forecast(self, x, mask_seed):
        np.random.random()  # Explicit fake compute advances RNG, not observation.
        return dict(prediction=np.zeros((1, 1, 1)), valid=np.ones((1, 1, 1), bool),
            tf_order=np.ones(1), copy_masks=np.zeros((1, 1, 16, 1), bool),
            stats=dict(scale=np.ones((1, 1, 1))), axes=dict(lag={"sentinel": mask_seed}))
    def reconstruct(self, x, mask_seed):
        np.random.random()
        return dict(hidden_mask=np.zeros((1, 1, 16, 1), bool), visible_mask=np.ones((1, 1, 16, 1), bool),
            transform_hidden_mask=np.zeros((1, 1, 2, 16, 1), bool), stats=dict(scale=np.ones((1, 1, 1))),
            order=np.ones(1), tf_reconstruction=np.zeros((1, 1, 2, 16, 1)),
            tf_valid=np.ones((1, 1, 2, 16, 1), bool),
            ms_reconstruction_by_scale=np.zeros((1, 1, 2, 2, 16, 1)),
            ms_valid_by_scale=np.ones((1, 1, 2, 2, 16, 1), bool), axes=[dict(lag={"sentinel": mask_seed})])


class ControlledWaveEngineeringTests(unittest.TestCase):
    def test_builder_unique_11_not_model_or_alias_repeats(self):
        with mock.patch.object(wave.original, "generate_pair", wraps=wave.original.generate_pair) as generate:
            values, truth = wave.build_cases(fixture_protocol())
        self.assertEqual(len(values), 11)
        self.assertEqual(generate.call_count, 5)  # three main families +2 independent-normal roles
        self.assertEqual(truth["WEAK_N"]["loader_role"], "QUERY")

    def test_global_rng_unchanged_by_builder(self):
        before = wave._numpy_rng(); wave.build_cases(fixture_protocol())
        self.assertTrue(wave._same_numpy_rng(before, wave._numpy_rng()))

    def test_level_matches_original_pair_float32_exactly(self):
        p = fixture_protocol(); values, _ = wave.build_cases(p)
        old, _ = wave.original.generate_pair("weak_amplitude", run_id=7, role="ENGINEERING_FIXTURE",
            n=96, burn_in=80, event=(24, 64), d0=2, d1=4)
        np.testing.assert_array_equal(values["WEAK_LEVEL025"]["values"], old["A"]["values"])

    def test_pulse_width_amplitude_and_unchanged_outside(self):
        values, _ = wave.build_cases(fixture_protocol()); base = values["WEAK_N"]["values"]
        for key, amplitude in (("WEAK_PULSE025", .25), ("STRONG_PULSE2", 2.)):
            expected = base.copy(); expected[30:34] += amplitude
            np.testing.assert_array_equal(values[key]["values"], expected)

    def test_break_shared_independent_driver_at_d0(self):
        p = fixture_protocol(); values, truth = wave.build_cases(p)
        family = "cross_mixed_sign/ENGINEERING_FIXTURE"
        t = truth[family]["retained_indices"]
        latent = truth[family]["latents"]; driver = truth[family+"/BREAK"]["driver"]
        expected = values["CROSS_N"]["values"].copy()
        expected[24:64, 1] = .3*driver[t[24:64]-2]+np.sqrt(.91)*latent["zplus"][t[24:64]]
        expected[24:64, 2] = -.3*driver[t[24:64]-2]+np.sqrt(.91)*latent["zminus"][t[24:64]]
        np.testing.assert_array_equal(values["CROSS_BREAK"]["values"], expected)
        self.assertEqual(truth[family+"/BREAK"]["source_delay"], 2)
        self.assertFalse(np.array_equal(driver, latent["u"]))

    def test_allowed_normal_independent_role_and_all_zero_ledger(self):
        values, truth = wave.build_cases(fixture_protocol())
        self.assertFalse(np.array_equal(values["SELF_ALLOWED"]["values"], values["SELF_A"]["values"]))
        self.assertEqual(int(truth["SELF_ALLOWED"]["labels"].sum()), 0)
        self.assertEqual(int(truth["CROSS_ALLOWED"]["labels"].sum()), 0)

    def test_self_labels_do_not_expand_after_recovery(self):
        values, truth = wave.build_cases(fixture_protocol())
        np.testing.assert_array_equal(truth["SELF_A"]["labels"][64:], 0)
        self.assertFalse(np.array_equal(values["SELF_N"]["values"][64:], values["SELF_A"]["values"][64:]))

    def test_phases_pulse_na_and_anchor_no_extra_indices(self):
        case = dict(event=[998, 1050])
        self.assertIsNone(wave.phases(case, 512, 2560)["steady"])
        selected = wave.anchors(case, 512, 2560)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(all(512 <= x < 2560 for x in selected))

    def test_derive_fresh_values_only_manifest_no_truth_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"new"
            wave.derive_inputs(fixture_protocol(), output)
            text = (output/"INPUT_MANIFEST.json").read_text()
            self.assertNotIn("labels", text); self.assertNotIn("truth", text.lower())
            rows = json.loads(text)["runs"]
            self.assertEqual(len(rows), 11)
            self.assertTrue(all(x["role"] == "QUERY" for x in rows))
            with self.assertRaises(FileExistsError): wave.derive_inputs(fixture_protocol(), output)

    def test_original_loader_accepts_exported_small_npz(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/"new"; wave.derive_inputs(fixture_protocol(), output)
            runs = data.load_manifest(output/"INPUT_MANIFEST.json", roles={"QUERY"})
            self.assertEqual(len(runs), 11)
            self.assertTrue(all(r.start == 0 and r.values.dtype == np.float32 and r.observed.all() for r in runs))

    def test_tf_inactive_nan_preserved_active_only(self):
        item = row(16)
        wave.validate_active(item, ["r_ms", "r_pred"])
        self.assertTrue(np.isnan(item["r_tf"]))
        with self.assertRaises(ValueError): wave.validate_active(item, ["r_tf", "r_ms", "r_pred"])

    def test_calibration_own_recipe_not_runtime_reweight(self):
        cal = calibration(("r_ms", "r_pred"))
        wave.validate_calibration(cal, ["r_ms", "r_pred"], scoring)
        with self.assertRaises(PermissionError): wave.validate_calibration(cal, ["r_tf", "r_ms", "r_pred"], scoring)
        cal["weights"]["r_ms"] = .25
        with self.assertRaises(PermissionError): wave.validate_calibration(cal, ["r_ms", "r_pred"], scoring)

    def _write(self, output, count=70, stop=False):
        def score_records(records, cal):
            return [dict(r, **scoring.apply_calibration(r, cal)) for r in records]
        def records():
            for i in range(count):
                if stop and i == 65: raise RuntimeError("engineering interrupted")
                yield row(16+i)
        identity = {k: "a"*64 for k in bank.IDENTITY_KEYS}
        return wave.consume_records(records(), output, identity=identity,
            ranges=[dict(domain=wave.DOMAIN, entity="fixture", run="small", start=16, end=86)],
            calibration=calibration(("r_ms", "r_pred")), active=["r_ms", "r_pred"],
            kernels=(types.SimpleNamespace(score_records=score_records), scoring, bank))

    def test_original_chunk_bank_full_tail_and_nan_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"bank"; self._write(out)
            manifest = json.loads((out/"manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertEqual([x["records"] for x in manifest["chunks"]], [64, 6])
            rows = list(bank.read_records(out))
            self.assertEqual(rows[-1]["index"], 85)
            self.assertTrue(np.isnan(rows[0]["r_tf"]))
            with self.assertRaises(FileExistsError): self._write(out)

    def test_interrupted_bank_incomplete_no_automatic_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"bank"
            with self.assertRaises(RuntimeError): self._write(out, stop=True)
            manifest = json.loads((out/"manifest.json").read_text())
            self.assertFalse(manifest["complete"]); self.assertEqual(manifest["committed_records"], 64)
            with self.assertRaises(FileExistsError): self._write(out)

    def test_anchor_same_objects_exact_passes_and_rng(self):
        values = np.zeros((40, 1), np.float32)
        run = data.Run("fixture", "entity", "r", "QUERY", values, np.ones(values.shape, bool))
        sink = []; before = wave._numpy_rng()
        model = FakeModel(); proxy = wave.AnchorModel(model, run, [18, 20], 2025, sink.append, wave._numpy_rng, wave._same_numpy_rng)
        for index in range(16, 24):
            f = proxy.forecast(None, mask_seed=2025+index)
            r = proxy.reconstruct(None, mask_seed=2025+index)
            self.assertEqual(f["prediction"].shape, (1, 1, 1))
            self.assertEqual(r["hidden_mask"].shape, (1, 1, 16, 1))
        after = wave._numpy_rng(); proxy.finish(8)
        self.assertEqual([x["index"] for x in sink], [18, 20])
        for x in sink:
            self.assertEqual(x["group_mask_seeds"], [2025+x["index"]])
            self.assertEqual(x["footprint"], [x["index"]-16, x["index"]+1])
            self.assertIn("RNG_before_forecast", x); self.assertIn("RNG_after_reconstruct", x)
        np.random.set_state(before[0]); random.setstate(before[1])
        for _ in range(8): model.forecast(None, mask_seed=0); model.reconstruct(None, mask_seed=0)
        self.assertTrue(wave._same_numpy_rng(after, wave._numpy_rng()))

    def test_bad_sink_rng_or_duplicate_anchor_fails(self):
        x = np.zeros((40, 1), np.float32); run = data.Run("f", "e", "r", "QUERY", x, np.ones(x.shape, bool))
        proxy = wave.AnchorModel(FakeModel(), run, [16], 2025, lambda p: np.random.random(), wave._numpy_rng, wave._same_numpy_rng)
        proxy.forecast(None, mask_seed=2041)
        with self.assertRaises(AssertionError): proxy.reconstruct(None, mask_seed=2041)

    def test_missing_anchor_or_mismatched_pass_seed_fails(self):
        x = np.zeros((40, 1), np.float32); run = data.Run("f", "e", "r", "QUERY", x, np.ones(x.shape, bool))
        proxy = wave.AnchorModel(FakeModel(), run, [16], 2025, lambda p: None, wave._numpy_rng, wave._same_numpy_rng)
        proxy.forecast(None, mask_seed=2041)
        with self.assertRaises(RuntimeError): proxy.reconstruct(None, mask_seed=2042)
        with self.assertRaises(ValueError): proxy.finish(1)

    def test_evaluation_whole_gate_before_any_truth(self):
        p = fixture_protocol()
        entries = [dict(case=c["id"], method=m) for c in p["cases"] for m in c["models"]+list(wave.METHODS)]
        seal = dict(status="ACCEPTED_CONTROLLED_WAVE1_FIXED_BANKS", approved_by="root", banks=entries)
        plan = wave.evaluation_metadata(p, entries, seal)
        self.assertFalse(plan["independent_data_replicates"])
        self.assertEqual(plan["excluded_recovery_points"], 0)
        with self.assertRaises(PermissionError): wave.evaluation_metadata(p, entries[:-1], seal)

    def test_evaluation_adapter_full_axis_and_unchanged_NA_no_phase_reset(self):
        metadata = dict(original_length=96, original_start=0, warmup_points=16)
        labels = np.zeros(96, np.int8)
        arrays = dict(index=np.arange(16, 96, dtype=np.int64), score=np.zeros(80),
            valid=np.ones(80, bool), available_index=np.arange(16, 96, dtype=np.int64), invalid_reasons=[None]*80)
        expected = {"ranking": {"AP": None}, "point": {"Precision": 0., "F1": 0., "Recall": None}}
        fake = types.SimpleNamespace(evaluate_arrays=mock.Mock(return_value=expected))
        actual = wave.dispatch_evaluation(metadata, arrays, labels, .5, fake)
        self.assertIs(actual, expected)
        args, kwargs = fake.evaluate_arrays.call_args
        self.assertIs(args[2], labels)
        self.assertEqual(kwargs["original_length"], 96)
        self.assertEqual(kwargs["warmup_points"], 16)
        self.assertEqual(fake.evaluate_arrays.call_count, 1)
        with self.assertRaises(ValueError): wave.dispatch_evaluation(metadata, arrays, labels[16:], .5, fake)

    def test_unapproved_and_wrong_action_rejected_before_body(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp)/"config.json"; config.write_text('{}')
            with self.assertRaises(PermissionError): wave.admit(config, None, None)
            release = Path(temp)/"release.json"; release.write_text('{"status":"ROOT_PENDING"}')
            with self.assertRaises(PermissionError): wave.admit(config, release, wave.original.sha(release))

    def test_truth_key_refused_and_fresh_output_before_kernels(self):
        with tempfile.TemporaryDirectory() as temp:
            t = Path(temp); cfg = dict(operation=wave.MODEL, resume=False, output=str(t/"out"), resources={})
            perm = dict(generate=False, neural_readout=True, baseline_readout=False, evaluation=False,
                        labels=False, training=False, target_fit=False, threshold_fit=False)
            cfg.update(permissions=perm, truth="MUST_NOT_OPEN")
            c = t/"c.json"; c.write_text(json.dumps(cfg))
            signed = dict(status="APPROVED", approved_by="root", operation=wave.MODEL,
                allowed_actions=[wave.MODEL], config_sha256=wave.original.sha(c), resources={}, allow_resume=False, permissions=perm)
            s = t/"s.json"; s.write_text(json.dumps(signed))
            with self.assertRaises(PermissionError): wave.admit(c, s, wave.original.sha(s))
            del cfg["truth"]; c.write_text(json.dumps(cfg)); signed["config_sha256"] = wave.original.sha(c); s.write_text(json.dumps(signed))
            (t/"out").mkdir()
            with self.assertRaises(FileExistsError): wave.admit(c, s, wave.original.sha(s))

    def test_changed_source_refused_before_kernels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"fake.py"; path.write_text('original')
            sources = {str(path): wave.original.sha(path)}; path.write_text('changed')
            with self.assertRaises(PermissionError): wave._source_gate(dict(source_sha256=sources), dict(source_sha256=sources))

    def test_formal_metadata_geometry_not_tiny_exemption(self):
        with self.assertRaises(PermissionError): wave.protocol_check(fixture_protocol())
        self.assertNotIn("torch", sys.modules)

    def test_technical_input_scope_not_formal_matrix(self):
        spec = dict(namespace="TECHNICAL_NORMAL_NOT_FORMAL_WAVE", length=640, data_seed=84553,
                    truth=False, purpose="CPU_PARITY")
        self.assertEqual(wave.technical_plan(spec), [("full2025", 1, [512]), ("tf_off2025", 1, [512]), ("full2025", 3, [512])])
        spec["purpose"] = "GPU_THROUGHPUT"
        plan = wave.technical_plan(spec)
        self.assertEqual(sum(len(x[2]) for x in plan), 128)
        spec["length"] = 2560
        with self.assertRaises(PermissionError): wave.technical_plan(spec)

    def test_raw_parity_exact_integer_masks_nan_float_policy(self):
        wave.assert_tree_parity(dict(seed=[2537], mask=[True], error=float("nan")),
                               dict(seed=[2537], mask=[True], error=float("nan")))
        with self.assertRaises(AssertionError): wave.assert_tree_parity(dict(seed=[2537]), dict(seed=[2538]))

    def test_real_v4_reference_has_infer_list_API_not_iter_infer(self):
        path = Path(os.environ["CONTROLLED_WAVE_REFERENCE_TRAINING"])
        self.assertEqual(wave.original.sha(path), "40e2247ed3049bf9d2303c380b2da55c1494c874ca38dd24652f741ed33ccc74")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("infer", functions); self.assertNotIn("iter_infer", functions)
        self.assertIn("index_plan", [arg.arg for arg in functions["infer"].args.kwonlyargs])

    def test_legacy_singleton_uses_actual_infer_once_and_no_extra_points(self):
        fake = types.SimpleNamespace(infer=mock.Mock(return_value=[dict(index=512)]))
        plan = [dict(index=512, role="QUERY")]
        self.assertEqual(wave.legacy_singleton(fake, object(), [], plan=plan, active=["r_pred"]), dict(index=512))
        self.assertEqual(fake.infer.call_count, 1)
        self.assertIs(fake.infer.call_args.kwargs["index_plan"], plan)
        fake.infer.return_value = [dict(index=512), dict(index=513)]
        with self.assertRaises(ValueError): wave.legacy_singleton(fake, object(), [], plan=plan, active=["r_pred"])

    def test_state_runtime_and_actual_bytes_evidence_interface(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp); source = path/"official.py"; source.write_text("synthetic file identity")
            cfg = dict(installed_frft=dict(path=str(source)), external_source_sha256={str(source): "not_reused_as_actual"})
            fake = types.SimpleNamespace(save_runtime=lambda directory, model: (directory/"runtime.json").write_text('{"fixture":"no Torch"}'))
            fingerprint = {"weight": dict(shape=[1], dtype="float32", sha256="a"*64)}
            for stage in ("before", "after"):
                wave.persist_evidence(path/"evidence", stage, fingerprint, cfg, fake, FakeModel())
                self.assertEqual(json.loads((path/"evidence"/("state_"+stage+".json")).read_text()), fingerprint)
                extra = json.loads((path/"evidence"/("runtime_"+stage)/"actual_external_bytes.json").read_text())
                self.assertEqual(extra["installed_frft"]["sha256"], wave.original.sha(source))
                self.assertTrue(extra["model_flags"]["tf_enabled"])
            with self.assertRaises(FileExistsError): wave.persist_evidence(path/"evidence", "before", fingerprint, cfg, fake, FakeModel())


if __name__ == "__main__":
    unittest.main()
