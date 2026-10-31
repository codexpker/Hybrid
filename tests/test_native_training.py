"""Frozen-loop comparison on synthetic CPU data; never scientific training."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hybrid_tsad.api import verify_checkpoint_contract
from hybrid_tsad.config import sha256
from engine_fixture_support import approve_fixture, calibration_file, record
from hybrid_tsad.engine.identity import SCORE_SPACE
from hybrid_tsad.engine.pipeline import run_operation, load_operation, check_operation_inputs

ROOT = Path(__file__).resolve().parents[1]
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DATA = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas"))
OFFICIAL = os.environ.get("HYBRID_OFFICIAL_FILE")


class CheckpointHistoryTests(unittest.TestCase):
    def test_400_checkpoint_cannot_claim_100_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"opaque; this guard must precede torch.load")
            metadata = root / "checkpoint_metadata.json"
            metadata.write_text(json.dumps(dict(checkpoint_sha256=sha256(checkpoint), history=400, horizon=5,
                                                  mode="hybrid", copies=2, temporal_depth=4)))
            with self.assertRaisesRegex(ValueError, "history/architecture"):
                verify_checkpoint_contract(checkpoint, metadata, history=100, mode="hybrid")
            with self.assertRaisesRegex(ValueError, "explicit"):
                verify_checkpoint_contract(checkpoint, None, history=100, mode="hybrid")
            verify_checkpoint_contract(checkpoint, metadata, history=400, mode="hybrid")

    def test_schema3_source_checkpoint_query_history_must_agree_before_forward(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract, approval = root/"contract.md", root/"approval.json"
            contract.write_text("synthetic metadata guard only")
            approval.write_text("{}")
            checkpoint = root/"weights.pt"
            checkpoint.write_bytes(b"opaque; no torch.load in this guard")
            query = root/"query.json"
            query.write_text(json.dumps(dict(history=100,horizon=5,origins=[100])))
            metadata = root/"checkpoint.json"
            value = dict(checkpoint_sha256=sha256(checkpoint),history=400,horizon=5,mode="hybrid",copies=2,temporal_depth=4)
            metadata.write_text(json.dumps(value))
            calibration = calibration_file(root/"cal.json", scale=.7, threshold=1.)
            cal = json.loads(calibration.read_text())
            cal.update(calibration_schema_version=2,history=100)
            calibration.write_text(json.dumps(cal))
            cfg = dict(engine_schema_version=3,operation="infer",contract=record(contract),approval=record(approval),
                       inputs=dict(official=record(checkpoint),checkpoint=record(checkpoint),array=record(checkpoint),
                                   query=record(query),checkpoint_metadata=record(metadata),calibration=record(calibration)),
                       parameters=dict(mode="hybrid",history=100,seed=0,source_scale=.7,mask_seed=53,job="fit",readout_job="readout"),
                       score_space=SCORE_SPACE,output=str(root/"out"))
            path=root/"operation.json"
            approve_fixture(cfg,path)
            with self.assertRaisesRegex(ValueError,"history/architecture"):
                run_operation(path,"infer",execute=True)
            value["history"]=100
            metadata.write_text(json.dumps(value));cfg["inputs"]["checkpoint_metadata"]=record(metadata)
            approve_fixture(cfg,path)
            parsed,records,paths,out=load_operation(path,"infer")
            check_operation_inputs(records,paths,parsed,out)
            query.write_text(json.dumps(dict(history=400,horizon=5,origins=[400])))
            cfg["inputs"]["query"]=record(query);approve_fixture(cfg,path)
            with self.assertRaisesRegex(ValueError,"query history"):
                run_operation(path,"infer",execute=True)


@unittest.skipUnless(HAS_DATA, "existing NumPy/Pandas required for source-asset guards")
class NativeAssetGuardTests(unittest.TestCase):
    def test_frozen_budget_trace_history_and_source_hash_rejections(self):
        import numpy as np
        from hybrid_tsad.engine.training import validate_training_assets
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {key:root/(key+suffix) for key,suffix in dict(wave_config=".json",source_inputs=".json",
                     source_manifest=".json",plan=".npz",rows=".json",official=".py",entry_gate=".json").items()}
            paths["official"].write_text("synthetic guard fixture only")
            source = root/"source.csv";source.write_text("0\n1\n")
            rows = [dict(dataset="SMD_SUBSET",role=role,source=str(source),entity="fixture",local_start=0,
                         local_end_exclusive=405,history=400,horizon=5,columns=[0],forecast_origins=[400])
                    for role in ("fit","select","cal")]
            paths["rows"].write_text(json.dumps(rows))
            paths["source_manifest"].write_text(json.dumps({"records":rows}))
            paths["entry_gate"].write_text('{"status":"PASS"}')
            cfg = dict(steps=10000,batch=64,lr=.0003,weight_decay=.01,warmup=500,validate_every=500,
                       query_evaluation=False,target="PSM",history=400,mask_seed=84551,
                       jobs={"job":dict(mode="hybrid",sources=["SMD_SUBSET"],plan="plan")})
            paths["wave_config"].write_text(json.dumps(cfg))
            inputs = dict(manifest=str(paths["source_manifest"]),official=str(paths["official"]),
                          hashes={str(p):sha256(p) for p in (source,paths["source_manifest"],paths["official"])})
            paths["source_inputs"].write_text(json.dumps(inputs))
            plan = {}
            for ri,(role,n) in enumerate((("fit",640000),("select",256),("cal",256))):
                plan[role+"_record"]=np.full(n,ri,"uint16")
                plan[role+"_origin"]=np.full(n,400,"uint32")
                plan[role+"_channel"]=np.zeros(n,"uint8")
            np.savez_compressed(paths["plan"],**plan)
            with patch("hybrid_tsad.engine.training.OFFICIAL_SHA",sha256(paths["official"])):
                validate_training_assets(paths,"job")
                for key,value in (("steps",9999),("history",100),("target","SMD")):
                    modified={**cfg,key:value};paths["wave_config"].write_text(json.dumps(modified))
                    with self.subTest(key=key),self.assertRaises(ValueError):
                        validate_training_assets(paths,"job")
                paths["wave_config"].write_text(json.dumps(cfg))
                plan["fit_origin"][0]=401
                np.savez_compressed(paths["plan"],**plan)
                with self.assertRaisesRegex(ValueError,"footprint/channel"):
                    validate_training_assets(paths,"job")
                plan["fit_origin"][0]=400
                plan["fit_origin"]=plan["fit_origin"][:-1]
                np.savez_compressed(paths["plan"],**plan)
                with self.assertRaisesRegex(ValueError,"exposure/dtype"):
                    validate_training_assets(paths,"job")
                source.write_text("changed synthetic source")
                with self.assertRaisesRegex(ValueError,"source input hash"):
                    validate_training_assets(paths,"job")


@unittest.skipUnless(HAS_TORCH, "existing torch required; CPU only")
class NativeTrainingEquivalenceTests(unittest.TestCase):
    def assert_state_equal(self, left, right):
        import torch
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(list(left), list(right))
            for key in left:
                self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    @staticmethod
    def toy_class():
        import torch
        class SyntheticPredictor(torch.nn.Module):
            def __init__(self, official=None, mode="hybrid", history=400):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(()))
                self.inactive = torch.nn.Parameter(torch.rand(()), requires_grad=False)
                self.history = history
            def cuda(self):
                return self
            def parameter_counts(self):
                return dict(stored=2, active=1, inactive=1)
            def forward(self, history):
                mean = history.mean(1, keepdim=True)
                scale = (history.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
                pred = self.weight + torch.rand((len(history), 5, 1)) * .1
                return dict(prediction_normalized=pred, prediction=pred*scale+mean, mean=mean, scale=scale,
                            tf_reconstruction=None)
        return SyntheticPredictor

    def test_real_old_and_native_loop_500_synthetic_steps_match(self):
        self._compare_real_loops()

    def test_seeded_jobs_actual_loop_summary_and_metadata(self):
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                self._compare_real_loops(job_seed=seed)

    def test_conflicting_report_seed_rejected_by_actual_loop(self):
        self._compare_real_loops(job_seed=2, conflicting_report_seed=True)

    def _compare_real_loops(self, job_seed=None, conflicting_report_seed=False):
        import math
        import time
        import numpy as np
        import torch
        import hybrid_tsad.engine.training as native
        from hybrid_tsad.data import canonical_domain
        from hybrid_tsad.model import objective
        frozen = json.loads((ROOT / "tests/fixtures/native_train_reference.json").read_text())
        data = np.sin(np.arange(430, dtype="float32") / 13)[:, None]
        rows = [dict(dataset="SMD_SUBSET", role=role, local_start=0, local_end_exclusive=430,
                     history=400, horizon=5, columns=[0], forecast_origins=list(range(400, 426)))
                for role in ("fit", "select", "cal")]
        plan = {}
        for i, (role, n) in enumerate((("fit", 500*64), ("select", 256), ("cal", 256))):
            plan[role+"_record"] = np.full(n, i, "uint16")
            plan[role+"_origin"] = np.resize(np.arange(400, 426, dtype="uint32"), n)
            plan[role+"_channel"] = np.zeros(n, "uint8")
        sampler = SimpleNamespace(array=lambda row: data)
        Toy = self.toy_class()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old_run = root / "old"
            (old_run / "plans/synthetic").mkdir(parents=True)
            plan_path = old_run / "plans/synthetic/plan.npz"
            np.savez_compressed(plan_path, **plan)
            (plan_path.parent / "rows.json").write_text(json.dumps(rows))
            (old_run / "entry_gate.json").write_text('{"status":"PASS"}')
            official = root / "official.py"
            official.write_text("synthetic fixture, not upstream source")
            cfg = dict(target="PSM", history=400, seed=3, mask_seed=84551,
                       jobs={"synthetic": dict(mode="hybrid", plan="synthetic", sources=["SMD_SUBSET"])})
            if job_seed is not None:
                cfg["jobs"]["synthetic"]["seed"] = job_seed
            wave_config = root / "wave_config.json"
            wave_config.write_text(json.dumps(cfg))
            source_manifest = root / "source_manifest.json"
            source_manifest.write_text(json.dumps({"records": rows}))
            inputs = dict(manifest="fixture", official=str(official), hashes={str(official): sha256(official)})
            namespace = dict(np=np, torch=torch, json=json, math=math, time=time, sha=sha256,
                             read_plan=native.read_plan,
                             load=lambda a: (cfg, inputs, Toy, objective, lambda *args, **kwargs: sampler, canonical_domain))
            for name in ("batch", "evaluate", "train"):
                entry = frozen["functions"][name]
                self.assertEqual(hashlib.sha256(entry["source"].encode()).hexdigest(), entry["source_sha256"])
                # Only shorten the old fixed loop for the engineering fixture.
                source = entry["source"].replace("range(1,10001)", "range(1,501)")
                exec(compile(source, "frozen_original_"+name, "exec"), namespace)
            real_tensor, real_fork = torch.tensor, torch.random.fork_rng
            def cpu_tensor(*args, **kwargs):
                if kwargs.get("device") == "cuda":
                    kwargs["device"] = "cpu"
                return real_tensor(*args, **kwargs)
            with patch.object(torch, "tensor", cpu_tensor), patch.object(torch.random, "fork_rng", lambda **kw: real_fork(devices=[])), \
                    patch.object(torch.cuda, "reset_peak_memory_stats", lambda *a: None), \
                    patch.object(torch.cuda, "max_memory_allocated", lambda *a: 0):
                for start in (0, 64, 499*64):
                    old_x, old_y = namespace["batch"](sampler, plan, rows, "fit", start)
                    new_x, new_y = native.source_batch(sampler, plan, rows, "fit", start, device="cpu")
                    self.assertTrue(torch.equal(old_x, new_x))
                    self.assertTrue(torch.equal(old_y, new_y))
                namespace["train"](SimpleNamespace(run=old_run, job="synthetic"))
            old_rng, old_numpy = torch.get_rng_state(), np.random.get_state()
            native_source = inspect.getsource(native.run_native_training).replace("range(1, 10001)", "range(1, 501)")
            native_namespace = dict(native.__dict__)
            native_namespace["validate_training_assets"] = lambda paths, job: (cfg, inputs, cfg["jobs"][job], rows, plan)
            if conflicting_report_seed:
                original_calibrate = native.calibrate_source
                def conflict_after_calibration(bank):
                    result = original_calibrate(bank)
                    cfg["jobs"]["synthetic"]["seed"] = (job_seed if job_seed is not None else cfg["seed"]) + 1
                    return result
                native_namespace["calibrate_source"] = conflict_after_calibration
            exec(compile(native_source, "native_loop_synthetic_500_step_adaptation", "exec"), native_namespace)
            output = root / "native"
            with patch("hybrid_tsad.model.HybridPredictor", Toy), patch("hybrid_tsad.data.ManifestSampler", lambda *a, **kw: sampler):
                paths = {"official": official, "plan": plan_path, "wave_config": wave_config, "source_manifest": source_manifest}
                if conflicting_report_seed:
                    with self.assertRaisesRegex(ValueError, "actual training seed"):
                        native_namespace["run_native_training"](paths, output, job="synthetic", device="cpu", operation_digest="a"*64)
                    self.assertFalse((output / "summary.json").exists())
                    self.assertFalse((output / "checkpoint_metadata.json").exists())
                    return
                native_namespace["run_native_training"](paths, output, job="synthetic", device="cpu", operation_digest="a"*64)
            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            self.assertTrue(np.array_equal(old_numpy[1], np.random.get_state()[1]))
            for name in ("step10000.pt", "best.pt", "optimizer_step10000.pt"):
                self.assert_state_equal(torch.load(old_run / "synthetic" / name, weights_only=True),
                                        torch.load(output / name, weights_only=True))
            for name in ("select_step500.npz", "select_primary.npz", "cal_primary.npz", "cal_scores.npz"):
                with np.load(old_run / "synthetic" / name) as a, np.load(output / name) as b:
                    self.assertEqual(a.files, b.files)
                    for key in a.files:
                        self.assertTrue(np.array_equal(a[key], b[key], equal_nan=True), (name, key))
            old_curve = json.loads((old_run / "synthetic/curve.json").read_text())
            new_curve = json.loads((output / "curve.json").read_text())
            for row in old_curve + new_curve:
                row.pop("seconds")
            self.assertEqual(old_curve, new_curve)
            summary = json.loads((output / "summary.json").read_text())
            metadata = json.loads((output / "checkpoint_metadata.json").read_text())
            expected_seed = cfg["seed"] if job_seed is None else job_seed
            self.assertEqual(summary["seed"], expected_seed)
            self.assertEqual(metadata["seed"], expected_seed)
            self.assertEqual(metadata["source_domains"], cfg["jobs"]["synthetic"]["sources"])
            self.assertEqual(metadata["excluded_target"], cfg["target"])
            self.assertEqual(metadata["weight_sha256"], sha256(output / "step10000.pt"))
            self.assertEqual(metadata["checkpoint_sha256"], metadata["weight_sha256"])
            self.assertEqual(metadata["source_manifest_sha256"], sha256(source_manifest))
            self.assertEqual(metadata["shared_plan_sha256"], sha256(plan_path))
            self.assertEqual(metadata["training_config_sha256"], sha256(wave_config))
            self.assertEqual(metadata["operation_manifest_sha256"], "a"*64)

    def test_lr_boundaries_and_checkpoint_rng_restore(self):
        import math
        import numpy as np
        import torch
        from hybrid_tsad.engine.training import learning_rate, optimizer_step, capture_step_state, restore_step_state
        from hybrid_tsad.model import objective
        for step in (1, 499, 500, 501, 10000):
            expected = .0003*min(step/500,1)*(1 if step<=500 else .5*(1+math.cos(math.pi*(step-500)/9500)))
            self.assertEqual(learning_rate(step), expected)
        torch.manual_seed(23)
        model = self.toy_class()()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0003, weight_decay=.01)
        x, y = torch.linspace(0,1,400).reshape(1,400,1), torch.ones((1,5,1))
        optimizer_step(model, opt, x, y, objective, 1)
        state = capture_step_state(model, opt, 1, device="cpu")
        optimizer_step(model, opt, x, y, objective, 2)
        expected = capture_step_state(model, opt, 2, device="cpu")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "engineering_state.pt"
            torch.save(state, path)
            recovered = torch.load(path, weights_only=True)
        self.assertEqual(restore_step_state(model, opt, recovered, device="cpu"), 1)
        optimizer_step(model, opt, x, y, objective, 2)
        actual = capture_step_state(model, opt, 2, device="cpu")
        self.assert_state_equal(expected, actual)

    @unittest.skipUnless(OFFICIAL, "existing external official source required")
    def test_history100_core_footprint_future_perturbation_and_small_updates(self):
        import numpy as np
        import torch
        from hybrid_tsad.api import create_predictor
        from hybrid_tsad.engine.bank import collect_forecasts
        from hybrid_tsad.engine.source import draw_source_plan
        from hybrid_tsad.engine.training import source_batch, optimizer_step, capture_step_state, restore_step_state
        from hybrid_tsad.model import objective
        data = np.linspace(0, 1, 115, dtype="float32")[:, None]
        rows = [dict(dataset="PSM", role=role, local_start=0, local_end_exclusive=115, history=100,
                     horizon=5, columns=[0], forecast_origins=[100,105,110]) for role in ("fit","select","cal")]
        sampler = SimpleNamespace(array=lambda row: data)
        plan, _ = draw_source_plan(sampler, rows, ["PSM"], 31, role_sizes={"fit":64,"select":64,"cal":64}, history=100)
        x, y = source_batch(sampler, plan, rows, "fit", 0, device="cpu", history=100, n=1)
        origin = int(plan["fit_origin"][0])
        self.assertEqual(tuple(x.shape), (1,100,1))
        self.assertTrue(np.array_equal(x.numpy()[0], data[origin-100:origin]))
        old_history = x.clone()
        data[origin:origin+5] += 100
        perturbed_x, perturbed_y = source_batch(sampler, plan, rows, "fit", 0, device="cpu", history=100, n=1)
        self.assertTrue(torch.equal(old_history, perturbed_x))
        self.assertFalse(torch.equal(y, perturbed_y))
        data[origin:origin+5] -= 100
        torch.manual_seed(41)
        model = create_predictor(OFFICIAL, history=100, device="cpu")
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0003, weight_decay=.01)
        optimizer_step(model, opt, x, y, objective, 1)
        state = capture_step_state(model, opt, 1, device="cpu")
        optimizer_step(model, opt, x, y, objective, 2)
        expected = capture_step_state(model, opt, 2, device="cpu")
        restore_step_state(model, opt, state, device="cpu")
        optimizer_step(model, opt, x, y, objective, 2)
        self.assert_state_equal(expected, capture_step_state(model, opt, 2, device="cpu"))
        model.eval()
        arrays, states, _ = collect_forecasts(model, data, [100,105,110], .7, mask_seed=53,
                                              expected_index=np.arange(100,115))
        self.assertEqual(arrays["prediction"].shape, (15,1))
        self.assertEqual(len(states), 3)
