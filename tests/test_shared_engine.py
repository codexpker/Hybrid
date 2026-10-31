import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from hybrid_tsad.config import sha256
from hybrid_tsad.engine.pipeline import run_operation
from hybrid_tsad.engine.identity import SCORE_SPACE
from engine_fixture_support import approve_fixture, calibration_file

ROOT = Path(__file__).resolve().parents[1]
HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def reference(name):
    record = json.loads((ROOT / "tests/fixtures/legacy_engine_reference.json").read_text())[name]
    source = record["source"]
    if hashlib.sha256(source.encode()).hexdigest() != record["excerpt_sha256"]:
        raise AssertionError("Frozen reference excerpt changed")
    return source


class PipelineContractTests(unittest.TestCase):
    def test_dry_run_does_not_open_missing_inputs_or_import_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "engine.json"
            record = {"path": "absent", "sha256": "0" * 64}
            cfg = {"engine_schema_version": 2, "score_space": SCORE_SPACE, "operation": "infer", "contract": record,
                   "approval": record, "inputs": {k: record for k in ["official", "checkpoint", "array", "query", "calibration"]},
                   "parameters": dict(seed=0, mode="hybrid", source_scale=1, mask_seed=84551,
                                      readout_job="synthetic", job="train_job"), "output": "new_output"}
            p.write_text(json.dumps(cfg))
            result = run_operation(p, "infer")
            self.assertFalse(result["executed"])
            self.assertFalse((Path(temp) / "new_output").exists())
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                run_operation(p, "infer", execute=True)
            cfg["operation"] = "evaluate"
            cfg["inputs"] = {k: record for k in ["bank", "bank_metadata", "labels", "coverage", "calibration"]}
            cfg["parameters"] = {"threshold": 1.0}
            p.write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError, "allow-labels"):
                run_operation(p, "evaluate", execute=True)


@unittest.skipUnless(HAS_NUMPY, "existing NumPy required for shared array-engine checks")
class SharedEngineTests(unittest.TestCase):
    def test_source_plan_rng_dtype_and_rejections_match_original_excerpt(self):
        import numpy as np
        from hybrid_tsad.engine.source import draw_source_plan
        data = np.arange(450 * 2, dtype="float32").reshape(450, 2)
        data[0] = np.nan
        data[201, 0] = np.nan  # legal interior missingness
        rows = [dict(dataset=d, role=r, forecast_origins=[400, 405, 410], columns=[0, 1])
                for d in ("SMD_SUBSET", "TEP") for r in ("fit", "select", "cal")]
        sampler = SimpleNamespace(array=lambda row: data)
        roles = [("fit", 130, 0), ("select", 17, 50000), ("cal", 19, 60000)]
        namespace = dict(np=np, sampler=sampler, rows=rows, sources=["SMD_SUBSET", "TEP"],
                         sample_seed=67031, arrays={}, quality={}, planname="synthetic", test_roles=roles)
        source = reference("plan").replace("[('fit',640000,0),('select',256,50000),('cal',256,60000)]", "test_roles")
        exec(compile(source, "frozen_source_plan_excerpt", "exec"), namespace)
        actual, quality = draw_source_plan(sampler, rows, namespace["sources"], 67031,
                                           role_sizes={r: n for r, n, _ in roles})
        for key, expected in namespace["arrays"].items():
            self.assertTrue(np.array_equal(actual[key], expected), key)
            self.assertEqual(actual[key].dtype, expected.dtype)
        self.assertEqual(quality, namespace["quality"])
        self.assertGreater(sum(v["rejected"] for v in quality.values()), 0)

    def test_source_calibration_matches_original_statements(self):
        import numpy as np
        from hybrid_tsad.engine.source import calibrate_source
        prediction = np.arange(40, dtype="float32").reshape(4, 5, 2) / 7
        target = prediction + np.linspace(0, 1, 40, dtype="float32").reshape(4, 5, 2)
        target[0, 0, 0] = np.nan
        bank = dict(prediction=prediction, target=target, history_scale=np.ones((4, 1, 2), "float32") * .7,
                    valid_target=np.isfinite(target))
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            np.savez_compressed(out / "cal_primary.npz", **bank)
            namespace = dict(np=np, out=out)
            exec(compile(reference("calibration"), "frozen_calibration_excerpt", "exec"), namespace)
            arrays, meta = calibrate_source(bank)
            with np.load(out / "cal_scores.npz") as old:
                for key in arrays:
                    self.assertTrue(np.array_equal(arrays[key], old[key], equal_nan=True), key)
            self.assertEqual(meta["normal_error_scale"], namespace["scale"])
            self.assertEqual(meta["threshold99"], float(np.quantile(namespace["scores"][namespace["pointvalid"]], .99)))
            namespace["cal"].close()

    def test_event_and_ap_functions_have_identical_source_asts(self):
        import hybrid_tsad.engine.events as events
        import hybrid_tsad.engine.statistics as statistics
        manifest = json.loads((ROOT / "docs/provenance/engine_extractions.json").read_text())
        for module in (events, statistics):
            nodes = {n.name: n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
            for record in manifest:
                if record["function"] in nodes:
                    actual = hashlib.sha256(ast.dump(nodes[record["function"]], include_attributes=False).encode()).hexdigest()
                    self.assertEqual(actual, record["function_ast_sha256"])

    def test_calibration_pipeline_executes_on_synthetic_bank_and_preserves_inputs(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = root / "contract.md"
            contract.write_text("synthetic engineering fixture; no research experiment")
            approval = root / "approval.json"
            approval.write_text(json.dumps({"status": "APPROVED", "contract_sha256": sha256(contract)}))
            bank = root / "source_cal.npz"
            np.savez_compressed(bank, prediction=np.zeros((2, 5, 1), "float32"),
                                target=np.ones((2, 5, 1), "float32"),
                                history_scale=np.ones((2, 1, 1), "float32"), valid_target=np.ones((2, 5, 1), bool))
            record = lambda p: {"path": str(p), "sha256": sha256(p)}
            cfg = dict(engine_schema_version=1, operation="calibrate", contract=record(contract), approval=record(approval),
                       inputs={"bank": record(bank)}, parameters={"bank_role": "source_cal"}, output=str(root / "out"))
            path = root / "operation.json"
            approve_fixture(cfg, path)
            before = sha256(bank)
            result = run_operation(path, "calibrate", execute=True)
            self.assertTrue(result["executed"])
            self.assertEqual(before, sha256(bank))
            self.assertEqual(json.loads((root / "out/source_calibration.json").read_text())["normal_error_scale"], 1.0)
            with self.assertRaisesRegex(ValueError, "immutable"):
                run_operation(path, "calibrate", execute=True)

    def test_metadata_v2_recovery_and_write_once(self):
        import numpy as np
        from hybrid_tsad.engine.bank import inference_metadata, recovered_metadata, write_bank
        kwargs = dict(points=10, blocks=2, mask_seed=84551, checkpoint_sha256="a" * 64, query_array_sha256="b" * 64)
        meta = inference_metadata("readout_job", {"job": "training_job"}, **kwargs)
        self.assertEqual(meta["readout_job"], "readout_job")
        self.assertEqual(meta["job"], "training_job")
        recovered = recovered_metadata("readout_job", {"job": "training_job"}, bank_hashes={"bank": "c" * 64}, **kwargs)
        self.assertEqual(recovered["new_forward_passes"], 0)
        self.assertIsNone(recovered["seconds"])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "new_bank"
            hashes = write_bank(output, {"time_index": np.arange(10)}, np.zeros((2, 4), "uint8"), meta)
            self.assertEqual(len(hashes), 3)
            with self.assertRaises(FileExistsError):
                write_bank(output, {}, np.zeros((2, 4)), meta)

    def test_shared_events_split_at_gaps(self):
        import numpy as np
        from hybrid_tsad.engine.events import binary_events, greedy_assign, real_blocks
        index = np.array([1, 2, 10, 11])
        self.assertEqual([x.tolist() for x in real_blocks(index, np.ones(4, bool))], [[0, 1], [2, 3]])
        self.assertEqual(binary_events([0, 1, 1, 0]), [(1, 3)])
        self.assertEqual(greedy_assign([(0, 10)], [("a", 1, 2), ("b", 3, 4)]), ["a"])

    @unittest.skipUnless(HAS_SKLEARN, "existing scikit-learn required")
    def test_fixed_evaluation_and_paired_bootstrap(self):
        import numpy as np
        from sklearn.metrics import average_precision_score
        from hybrid_tsad.engine.evaluation import evaluate_fixed_scores, paired_block_differences
        index = np.arange(20)
        y = np.tile([0, 1, 1, 0, 0], 4)
        s = np.linspace(0, 1, 20)
        coverage = [dict(event_id=str(i), start=i * 5 + 1, end_exclusive=i * 5 + 3,
                         eligible_complete=True, scored_points=2, total_points=2) for i in range(4)]
        result = evaluate_fixed_scores(index, y, s, .5, coverage)
        self.assertEqual(result["metrics"]["AP"], average_precision_score(y, s))
        self.assertEqual(result["metrics"]["complete_events"], 4)
        self.assertEqual(result["metrics"]["complete_event_hits"], 2)
        scores = {f"{name}_s{i}": (s if name == "temporal" else y + s * .1) + i * .01
                  for name in ("hybrid", "temporal") for i in range(3)}
        bootstrap = paired_block_differences(y, scores, index // 5, methods=("temporal",), n_replicates=8)
        for j, draw in enumerate(bootstrap["draws"]):
            resampled = np.concatenate([np.flatnonzero(index // 5 == block) for block in draw])
            expected = np.mean([average_precision_score(y[resampled], scores[f"hybrid_s{i}"][resampled]) -
                                average_precision_score(y[resampled], scores[f"temporal_s{i}"][resampled]) for i in range(3)])
            self.assertAlmostEqual(bootstrap["AP_difference"][j, 0], expected, places=12)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = root / "contract.md"
            contract.write_text("synthetic fixed-score evaluation; no research labels")
            approval = root / "approval.json"
            approval.write_text(json.dumps({"status": "APPROVED", "contract_sha256": sha256(contract)}))
            bank = root / "bank.npz"
            np.savez_compressed(bank, time_index=index, standardized_score=s, raw_score=s)
            labels = root / "synthetic_labels.npy"
            np.save(labels, y)
            events = root / "coverage.json"
            events.write_text(json.dumps(coverage))
            calibration = calibration_file(root / "calibration.json", scale=1.0, threshold=.5)
            metadata = root / "bank_metadata.json"
            metadata.write_text(json.dumps(dict(bank_sha256=sha256(bank), source_calibration_sha256=sha256(calibration),
                                                score_space=SCORE_SPACE, score_field="standardized_score", source_scale=1.0)))
            record = lambda p: {"path": str(p), "sha256": sha256(p)}
            cfg = dict(engine_schema_version=1, operation="evaluate", contract=record(contract), approval=record(approval),
                       inputs=dict(bank=record(bank), bank_metadata=record(metadata), calibration=record(calibration),
                                   labels=record(labels), coverage=record(events)),
                       parameters={"threshold": .5}, output=str(root / "evaluation"))
            config = root / "operation.json"
            approve_fixture(cfg, config)
            result = run_operation(config, "evaluate", execute=True, allow_labels=True)
            self.assertTrue(result["executed"])
            saved = json.loads((root / "evaluation/evaluation.json").read_text())
            self.assertEqual(saved["metrics"]["AP"], average_precision_score(y, s))
            self.assertIsNone(saved["metrics"]["Aff_F1"])

    @unittest.skipUnless(HAS_TORCH, "existing torch required for CPU forecast/RNG check")
    def test_forecast_bank_matches_original_loop_on_cpu(self):
        import numpy as np
        import torch
        from hybrid_tsad.engine.bank import collect_forecasts
        from hybrid_tsad.protocol import score_block
        class Fixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.zeros(()))
                self.history = 400
            def forward(self, x):
                return {"prediction": x[:, -1:] + torch.rand((1, 5, 1)), "scale": torch.ones((1, 1, 1))}
        model = Fixture().eval()
        data = np.linspace(0, 1, 420, dtype="float32")[:, None]
        q = {"origins": [400, 405, 410]}
        values = {k: [] for k in ["prediction", "target", "history_scale_by_block", "normalized_channel_residual", "raw_score", "standardized_score"]}
        source = reference("forecast")
        for old, new in [("device='cuda'", "device='cpu'"), ("torch.cuda.get_rng_state()", "torch.get_rng_state()"),
                         ("torch.cuda.set_rng_state(", "torch.set_rng_state("),
                         ("torch.cuda.synchronize()", "None"), ("torch.cuda.max_memory_allocated()", "0")]:
            source = source.replace(old, new)
        import time
        namespace = dict(np=np, torch=torch, model=model, data=data, q=q, values=values,
                         states=[], indices=[], times=[], spec={"source_scale": .7}, score_block=score_block,
                         a=SimpleNamespace(job="synthetic"), time=time)
        torch.manual_seed(245)
        exec(compile(source, "frozen_forecast_loop_cpu_adaptation", "exec"), namespace)
        arrays, states, _ = collect_forecasts(model, data, q["origins"], .7, mask_seed=245,
                                               expected_index=np.arange(400, 415))
        for key, value in values.items():
            expected = np.stack(value) if key == "history_scale_by_block" else np.concatenate(value)
            self.assertTrue(np.array_equal(arrays[key], expected), key)
        self.assertTrue(np.array_equal(states, np.stack(namespace["states"])))


if __name__ == "__main__":
    unittest.main()
