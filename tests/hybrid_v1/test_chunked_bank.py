"""Real NumPy storage/clock/recovery tests. No Torch, labels, GPU or real data."""
import copy
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from hybrid_tsad.hybrid_v1.chunked_bank import (
    ChunkedBank, IDENTITY_KEYS, _fingerprint, _pack, _unpack,
    iter_pointwise_infer, read_entries, read_records,
)


def identity():
    return {key: hashlib.sha256(key.encode()).hexdigest() for key in IDENTITY_KEYS}


def ranges(start=512, end=522, run="A"):
    return [dict(domain="synthetic", entity="fixture", run=run, start=start, end=end)]


def record(index, score=2., run="A", channels=3):
    rng = np.random.default_rng(index)
    return dict(domain="synthetic", entity="fixture", run=run, index=index,
        available_index=index, forecast_issued_index=index-1, score=float(score),
        valid=bool(np.isfinite(score)), r_tf=.1, r_ms=.2, r_pred=.3,
        r_var=float("nan"), prediction=rng.normal(size=(32, channels)).tolist(),
        per_channel={"r_tf": rng.normal(size=channels).tolist()},
        replay=dict(actual_mask_seed=2025+index, future_observed=False,
                    packed_bool="0fff", input_footprint=[index-512, index+1]),
        extra=dict(array=np.array([0., -0., np.nan, np.inf, -np.inf], dtype=np.float32),
                   flag=np.bool_(True), small=np.int16(-9), empty=np.empty((0, channels)),
                   names=np.array(["a", "中文"], dtype="U2"), optional=None,
                   mixed=[True, 1, 1.25, "ok", None], pair=(1, False)))


class Storage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bank"
        self.kwargs = dict(identity=identity(), ranges=ranges(), threshold=1., chunk_size=3)

    def assertRawEqual(self, left, right):
        self.assertEqual(_fingerprint(left), _fingerprint(right))

    def test_lossless_typed_values_nan_negative_zero_bool_ragged_C(self):
        original = [record(512, channels=1), record(513, channels=25), record(514)]
        unpacked = _unpack(_pack(original))
        for a, b in zip(original, unpacked):
            self.assertRawEqual(a, b)
            self.assertEqual(b["extra"]["array"].dtype, np.float32)
            self.assertTrue(np.signbit(b["extra"]["array"][1]))
            self.assertIs(type(b["extra"]["flag"]), np.bool_)
            self.assertIs(type(b["extra"]["small"]), np.int16)
            self.assertIs(type(b["extra"]["mixed"][0]), bool)
            self.assertIs(type(b["extra"]["mixed"][1]), int)

    def test_chunks_complete_tail_and_raw_identity(self):
        rows = [record(i) for i in range(512, 552)]
        kw = {**self.kwargs, "ranges": ranges(end=552), "chunk_size": 7}
        with ChunkedBank(self.path, **kw) as bank:
            bank.write(iter(rows)); bank.finalize()
            self.assertEqual([c["records"] for c in bank.manifest["chunks"]], [7]*5+[5])
            self.assertTrue(bank.manifest["complete"])
        actual = list(read_records(self.path))
        self.assertEqual([r["index"] for r in actual], list(range(512, 552)))
        self.assertEqual([r["index"] for r in actual[-31:]], list(range(521, 552)))
        for a, b in zip(rows, actual):
            self.assertRawEqual(a, b)

    def test_confirmation_cross_chunks_strict_threshold_nan_gap_and_run(self):
        plan = ranges(512, 518) + ranges(520, 524) + ranges(512, 516, "B")
        rows = [record(i, score=(1. if i == 515 else np.nan if i == 516 else 2.))
                for i in range(512, 518)]
        rows += [record(i) for i in range(520, 524)]
        rows += [record(i, run="B") for i in range(512, 516)]
        with ChunkedBank(self.path, **{**self.kwargs, "ranges": plan, "chunk_size": 2}) as bank:
            bank.write(rows); bank.finalize()
        entries = list(read_entries(self.path))
        self.assertEqual([e["derived"]["confirmed3_current"] for e in entries],
                         [False, False, True, False, False, False,
                          False, False, True, True, False, False, True, True])
        self.assertFalse(entries[3]["derived"]["strict_alarm"])
        self.assertFalse(entries[4]["derived"]["score_valid"])

    def test_recovery_at_all_four_atomic_boundaries_without_duplicates(self):
        for stage in ("before_chunk_replace", "after_chunk_replace",
                      "before_manifest_replace", "after_manifest_replace"):
            with self.subTest(stage=stage):
                target = self.path / stage
                armed = [False]
                def fault(s):
                    if armed[0] and s == stage:
                        raise OSError("simulated process interruption")
                bank = ChunkedBank(target, **self.kwargs, fault=fault)
                bank.write(record(i) for i in range(512, 515))
                armed[0] = True
                with self.assertRaises(OSError):
                    bank.write(record(i) for i in range(515, 518))
                with self.assertRaises(RuntimeError):
                    bank.flush()
                bank.close()
                with ChunkedBank(target, **self.kwargs) as resumed:
                    committed = resumed.manifest["committed_records"]
                    self.assertEqual(committed, 6 if stage == "after_manifest_replace" else 3)
                    self.assertEqual(next(resumed.remaining_plan())["index"], 512+committed)
                    resumed.write((record(i) for i in range(512, 522)), verify_committed_prefix=True)
                    resumed.finalize()
                self.assertEqual([r["index"] for r in read_records(target)], list(range(512, 522)))
                self.assertEqual(sum(c["records"] for c in json.loads((target/"manifest.json").read_text())["chunks"]), 10)

    def test_resume_rejects_raw_drift_and_short_replayed_prefix(self):
        with ChunkedBank(self.path, **self.kwargs) as bank:
            bank.write(record(i) for i in range(512, 515))
        with ChunkedBank(self.path, **self.kwargs) as bank:
            changed = [record(i) for i in range(512, 522)]
            changed[0]["prediction"][0][0] += 1
            with self.assertRaisesRegex(ValueError, "prefix differs"):
                bank.write(changed, verify_committed_prefix=True)
            with self.assertRaisesRegex(ValueError, "ends before"):
                bank.write([], verify_committed_prefix=True)
            self.assertEqual(bank.manifest["committed_records"], 3)

    def test_actual_process_exit_releases_OS_lock_and_ignores_orphan(self):
        source = str(Path(__file__).resolve().parents[2]/"src")
        code = """import hashlib,os,sys
sys.path.insert(0,sys.argv[1])
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,IDENTITY_KEYS
def crash(stage):
    if stage=='after_chunk_replace': os._exit(23)
identity={k:hashlib.sha256(k.encode()).hexdigest() for k in IDENTITY_KEYS}
ranges=[dict(domain='synthetic',entity='fixture',run='A',start=512,end=522)]
b=ChunkedBank(sys.argv[2],identity=identity,ranges=ranges,threshold=1.,chunk_size=3,fault=crash)
for i in range(512,515): b.append(dict(domain='synthetic',entity='fixture',run='A',index=i,score=2.,valid=True))
"""
        result = subprocess.run([sys.executable, "-c", code, source, str(self.path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(len(list(self.path.glob("chunk-*.npz"))), 1)
        with ChunkedBank(self.path, **self.kwargs) as bank:
            self.assertEqual(bank.manifest["committed_records"], 0)
            bank.write(record(i) for i in range(512, 522)); bank.finalize()
        self.assertEqual([r["index"] for r in read_records(self.path)], list(range(512, 522)))

    def test_all_five_bound_identities_and_contract_changes_rejected(self):
        with ChunkedBank(self.path, **self.kwargs):
            pass
        for key in IDENTITY_KEYS:
            bad = identity(); bad[key] = "a"*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                ChunkedBank(self.path, **{**self.kwargs, "identity": bad})
        for change in ({"threshold": 2.}, {"chunk_size": 2},
                       {"confirmation_points": 2}, {"ranges": ranges(end=523)}):
            with self.assertRaises(ValueError):
                ChunkedBank(self.path, **{**self.kwargs, **change})

    def test_raw_stream_input_identity_enforced(self):
        bound = {**identity(), "input_runs": [dict(domain="synthetic", entity="fixture", run="A",
                  input_identity=dict(values_float32_sha256="a"*64, observed_sha256="b"*64))]}
        with ChunkedBank(self.path, **{**self.kwargs, "identity": bound}) as bank:
            with self.assertRaisesRegex(ValueError, "input array/observed"):
                bank.append(record(512))
            row = record(512); row["replay"]["input_identity"] = bound["input_runs"][0]["input_identity"]
            bank.append(row)

    def test_duplicate_out_of_order_missing_and_incomplete_refused(self):
        with ChunkedBank(self.path, **self.kwargs) as bank:
            with self.assertRaises(ValueError):
                bank.append(record(513))
            bank.append(record(512))
            with self.assertRaises(ValueError):
                bank.append(record(512))
            with self.assertRaises(ValueError):
                bank.finalize()

    def test_uncommitted_buffer_never_visible_and_record_mutation_isolated(self):
        row = record(512)
        original = copy.deepcopy(row)
        with ChunkedBank(self.path, **{**self.kwargs, "ranges": ranges(512, 513)}) as bank:
            bank.append(row); row["prediction"][0][0] = 888
            self.assertEqual(list(read_records(self.path)), [])
            with self.assertRaisesRegex(ValueError, "final coverage chunk"):
                bank.flush()
            bank.finalize()
        self.assertRawEqual(next(read_records(self.path)), original)

    def test_tamper_and_concurrent_writer_refused(self):
        with ChunkedBank(self.path, **self.kwargs) as bank:
            with self.assertRaises(RuntimeError):
                ChunkedBank(self.path, **self.kwargs)
            bank.write(record(i) for i in range(512, 515))
            chunk = self.path/bank.manifest["chunks"][0]["file"]
        with chunk.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "identity/path"):
            list(read_records(self.path))
        with self.assertRaisesRegex(ValueError, "identity/path"):
            ChunkedBank(self.path, **self.kwargs)

    def test_codec_rejects_object_payload_and_bank_import_does_not_load_torch(self):
        with self.assertRaises(TypeError):
            _pack([np.array([object()], dtype=object)])
        # Fresh interpreter also works when a broader suite already imported Torch.
        source = str(Path(__file__).resolve().parents[2]/"src")
        code = "import sys; sys.path.insert(0,sys.argv[1]); import hybrid_tsad.hybrid_v1.chunked_bank; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code, source], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class PointwiseAdapter(unittest.TestCase):
    def test_injected_iterator_raw_seed_clock_order_and_rng_equivalence(self):
        runs = [SimpleNamespace(domain="synthetic", entity="fixture", run=r, role="QUERY",
                                start=1000, values=np.zeros((552, c), np.float32))
                for r, c in (("A", 1), ("B", 25))]
        calls = []
        def fake_infer(model, runs, *, history, horizon, seed, event_logits,
                       active_components, index_plan=None):
            calls.append(index_plan)
            rows = []
            for run in runs:
                indices = (range(history, len(run.values)) if index_plan is None else
                           [x["index"]-run.start for x in index_plan if x["run"] == run.run])
                for local in indices:
                    row = record(run.start+local, run=run.run, channels=run.values.shape[1])
                    row["replay"]["actual_mask_seed"] = seed+local
                    rows.append(row)
            return rows
        py_rng, np_rng = random.getstate(), np.random.get_state()
        reference = fake_infer(None, runs, history=512, horizon=32, seed=2025,
                               event_logits=False, active_components=None)
        calls.clear()
        actual = list(iter_pointwise_infer(None, runs, history=512, horizon=32, seed=2025,
                                           infer_fn=fake_infer))
        self.assertEqual([_fingerprint(r) for r in reference], [_fingerprint(r) for r in actual])
        self.assertEqual(len(calls), 1)
        self.assertEqual(actual[0]["replay"]["actual_mask_seed"], 2025+512)
        self.assertEqual(actual[-1]["index"], 1551)
        self.assertEqual(py_rng, random.getstate())
        after = np.random.get_state()
        self.assertEqual(np_rng[0], after[0]); np.testing.assert_array_equal(np_rng[1], after[1])
        self.assertEqual(np_rng[2:], after[2:])

    def test_remaining_plan_single_generator_call_and_original_order(self):
        run = SimpleNamespace(domain="synthetic", entity="fixture", run="A", role="QUERY",
                              start=0, values=np.zeros((522, 1)))
        with tempfile.TemporaryDirectory() as directory:
            with ChunkedBank(directory, identity=identity(), ranges=ranges(), threshold=1., chunk_size=3) as bank:
                bank.write(record(i) for i in range(512, 515))
                calls = []
                def injected(model, runs, **kwargs):
                    calls.append(kwargs["index_plan"])
                    for entry in kwargs["index_plan"]:
                        yield record(entry["index"])
                bank.write(iter_pointwise_infer(None, [run], history=512, horizon=32,
                    index_plan=bank.remaining_plan(), infer_fn=injected))
                bank.finalize()
                self.assertEqual(len(calls), 1)
                self.assertEqual([e["index"] for e in calls[0]], list(range(515,522)))
            self.assertEqual([r["index"] for r in read_records(directory)], list(range(512,522)))

    def test_only_inference_section_changed_and_per_point_AST_identical(self):
        root = Path(__file__).resolve().parents[3]
        frozen = root/"preparations/hybrid_v1_stage1_source_build_20261102_v1/snapshot_v4/src/hybrid_tsad/hybrid_v1/training.py"
        current = root/"hybrid_ad/src/hybrid_tsad/hybrid_v1/training.py"
        old, new = frozen.read_text(), current.read_text()
        self.assertEqual(old[:old.index("def infer(")], new[:new.index("def iter_infer(")])
        self.assertEqual(old[old.index("def calibrate_route("):], new[new.index("def calibrate_route("):])
        tree_old, tree_new = ast.parse(old), ast.parse(new)
        old_fn = next(n for n in tree_old.body if isinstance(n,ast.FunctionDef) and n.name=="infer")
        new_fn = next(n for n in tree_new.body if isinstance(n,ast.FunctionDef) and n.name=="iter_infer")
        old_loop = next(n for n in ast.walk(old_fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="t")
        new_loop = next(n for n in ast.walk(new_fn) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=="t")
        old_compute = old_loop.body[:-1]  # only records.append excluded
        new_compute = new_loop.body[0].body  # identical compute, per-point no_grad
        self.assertEqual([ast.dump(n) for n in old_compute], [ast.dump(n) for n in new_compute])

    def test_fake_tensor_control_flow_equivalence_and_run_hash_once(self):
        # Explicit fixture, not a Torch/model acceptance. Executes both actual
        # function bodies with NumPy tensor stand-ins and identical pass outputs.
        from hybrid_tsad.hybrid_v1 import training
        root = Path(__file__).resolve().parents[3]
        frozen = root/"preparations/hybrid_v1_stage1_source_build_20261102_v1/snapshot_v4/src/hybrid_tsad/hybrid_v1/training.py"
        spec = importlib.util.spec_from_file_location("hybrid_tsad.hybrid_v1._bank_fixture_frozen", frozen)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        class Tensor(np.ndarray):
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.asarray(self)
        def tensor(x): return np.asarray(x).view(Tensor)
        def passes(model, batch, seed):
            c = batch["history"].shape[-1]
            rng = np.random.default_rng(seed)
            f = dict(prediction=tensor(rng.normal(size=(1,32,c))), valid=tensor(np.ones((1,32,c),bool)),
                     stats={"scale":tensor(np.ones((1,1,c)))},copy_masks=tensor(np.ones((1,2,c,4),bool)),tf_order=tensor([.5]))
            r = dict(hidden_mask=tensor(np.ones((1,1,128,c),bool)), visible_mask=tensor(np.zeros((1,1,128,c),bool)),
                     transform_hidden_mask=tensor(np.ones((1,1,2,c,4),bool)),order=tensor([.5]),stats=f["stats"],
                     tf_reconstruction=tensor(rng.normal(size=(1,1,2,128,c))),
                     ms_reconstruction_by_scale=tensor(rng.normal(size=(1,1,2,2,128,c))),
                     tf_valid=tensor(np.ones((1,1,2,128,c),bool)),ms_valid_by_scale=tensor(np.ones((1,1,2,2,128,c),bool)))
            return f,r
        def residual(f,r,b,active):
            return dict(scalar={"r_tf":tensor([np.nan]),"r_ms":tensor([.25]),"r_pred":tensor([.5])},
                        per_channel={"r_tf":tensor([[np.nan]])})
        from hybrid_tsad.hybrid_v1.data import Run
        run=Run("synthetic","fixture","A","QUERY",np.arange(164,dtype=np.float32)[:,None],np.ones((164,1),bool))
        plan=[dict(domain=run.domain,entity=run.entity,run=run.run,role=run.role,index=i) for i in range(128,164)]
        model=SimpleNamespace(eval=lambda: None,parameters=lambda: iter([SimpleNamespace(device="cpu")]))
        targets={run.values.tobytes():0,run.observed.tobytes():0}
        real_sha=hashlib.sha256
        def counted(data=b""):
            if data in targets: targets[data]+=1
            return real_sha(data)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,{"torch":SimpleNamespace(no_grad=contextlib.nullcontext)}))
            for module in (training,old):
                stack.enter_context(patch.object(module,"tensor_batch",lambda b,d: {k:tensor(v) if isinstance(v,np.ndarray) else v for k,v in b.items()}))
                stack.enter_context(patch.object(module,"passes",passes))
                stack.enter_context(patch.object(module,"residual_components",residual))
            reference=old.infer(model,[run],history=128,horizon=32,seed=2025,index_plan=plan)
            with patch.object(hashlib,"sha256",counted):
                actual=list(training.iter_infer(model,[run],history=128,horizon=32,seed=2025,index_plan=plan))
            self.assertEqual(list(targets.values()),[1,1])
            self.assertEqual([_fingerprint(r) for r in reference],[_fingerprint(r) for r in actual])
            self.assertEqual([_fingerprint(r) for r in reference],[_fingerprint(r) for r in training.infer(model,[run],history=128,horizon=32,seed=2025,index_plan=plan)])


if __name__ == "__main__":
    unittest.main()
