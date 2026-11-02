"""CPU-only real external-source tiny inference parity; never loads real data.

Requires existing Torch + official dependencies and HYBRID_V1_WORKSPACE.
Local no-Torch skip is explicit, NOT a passed model-equivalence acceptance.
"""
import importlib.util
import hashlib
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hybrid_tsad.hybrid_v1.chunked_bank import (
    ChunkedBank, IDENTITY_KEYS, _fingerprint, iter_pointwise_infer, read_records,
)


@unittest.skipUnless(importlib.util.find_spec("torch"), "TORCH_UNAVAILABLE_MODEL_EQUIVALENCE_NOT_TESTED")
class RealSourceCPUParity(unittest.TestCase):
    def test_real_tiny_multirun_infer_raw_rng_and_chunk_bank_exact(self):
        import torch
        from hybrid_tsad.hybrid_v1 import HybridV1, HybridV1Config, OfficialSources
        from hybrid_tsad.hybrid_v1.data import Run
        from hybrid_tsad.hybrid_v1.training import infer, iter_infer
        from hybrid_tsad.hybrid_v1.scoring import apply_calibration, fit_calibration
        workspace = os.environ["HYBRID_V1_WORKSPACE"]
        frozen = Path(workspace)/"preparations/hybrid_v1_stage1_source_build_20261102_v1/snapshot_v4/src/hybrid_tsad/hybrid_v1/training.py"
        self.assertEqual(hashlib.sha256(frozen.read_bytes()).hexdigest(),
                         "40e2247ed3049bf9d2303c380b2da55c1494c874ca38dd24652f741ed33ccc74")
        spec = importlib.util.spec_from_file_location("hybrid_tsad.hybrid_v1._bank_real_frozen_v4", frozen)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        torch.set_num_threads(1); torch.manual_seed(937)
        random.seed(937); np.random.seed(937)
        cfg = HybridV1Config(history=128, horizon=32, local_window=20, width=16,
            temporal_patch=16, temporal_layers=1, heads=2, tf_hidden=8, tf_repr=16,
            tf_depth=1, tf_patch=5, lag_window=20, max_lag=8, lag_width=8,
            lag_top_k=2, history_mask_probability=0., dropout=0.)
        model = HybridV1(cfg, OfficialSources.from_workspace(workspace)).cpu().eval()
        self.assertEqual(next(model.parameters()).device.type, "cpu")
        rng = np.random.default_rng(71); runs = []
        for channels in (1, 3):
            values = rng.normal(size=(164, channels)).astype(np.float32)
            observed = np.ones(values.shape, bool)
            observed[132] = False
            values[132] = np.nan
            runs.append(Run("synthetic", "C%d" % channels, "r", "QUERY", values, observed, start=1000))
        before_state = {name: value.clone() for name, value in model.state_dict().items()}
        py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        reference = old.infer(model, runs, history=128, horizon=32, seed=2025)
        py_after, np_after, torch_after = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        plan = [dict(domain=r.domain,entity=r.entity,run=r.run,role=r.role,index=r.start+t)
                for r in runs for t in range(128,len(r.values))]
        variants = [
            ("new_generator",lambda: iter_infer(model,runs,history=128,horizon=32,seed=2025)),
            ("new_list_compat",lambda: infer(model,runs,history=128,horizon=32,seed=2025)),
            ("new_bank_outer_explicit_plan",lambda: iter_pointwise_infer(model,runs,history=128,horizon=32,seed=2025,index_plan=iter(plan))),
        ]
        original_sha = hashlib.sha256
        grad_enabled = torch.is_grad_enabled()
        for name, factory in variants:
            with self.subTest(variant=name):
                random.setstate(py_before); np.random.set_state(np_before); torch.set_rng_state(torch_before)
                counts={r.values.tobytes():0 for r in runs}
                counts.update({r.observed.tobytes():0 for r in runs})
                def counted(data=b""):
                    if data in counts: counts[data]+=1
                    return original_sha(data)
                actual=[]
                with patch.object(hashlib,"sha256",counted):
                    for row in factory():
                        self.assertEqual(torch.is_grad_enabled(),grad_enabled,"Generator leaked no_grad into caller")
                        actual.append(row)
                self.assertEqual(list(counts.values()),[1,1,1,1],"Each run values/observed must hash once per invocation")
                self.assertEqual([_fingerprint(r) for r in reference], [_fingerprint(r) for r in actual])
                self.assertEqual(py_after, random.getstate())
                np_now = np.random.get_state()
                self.assertEqual(np_after[0], np_now[0]); np.testing.assert_array_equal(np_after[1], np_now[1])
                self.assertEqual(np_after[2:], np_now[2:])
                self.assertTrue(torch.equal(torch_after, torch.get_rng_state()))
        self.assertTrue(any(np.isnan(r["r_pred"]) for r in reference))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(before_state[name], value), name)
        for r in runs:
            subset = [row for row in actual if row["entity"] == r.entity]
            self.assertEqual([row["index"] for row in subset], list(range(1128, 1164)))
            self.assertEqual([row["index"] for row in subset[-31:]], list(range(1133, 1164)))
            self.assertEqual(subset[0]["replay"]["actual_mask_seed"], 2025+128)
            self.assertEqual(subset[-1]["available_index"], 1163)
        calibration = fit_calibration(reference, source=True)
        for row in actual:
            row.update(apply_calibration(row, calibration))
        with tempfile.TemporaryDirectory() as directory:
            contract = {key: "a"*64 for key in IDENTITY_KEYS}  # Clearly synthetic identity, no checkpoint loaded.
            plan = [dict(domain=r.domain, entity=r.entity, run=r.run, start=1128, end=1164) for r in runs]
            with ChunkedBank(directory, identity=contract, ranges=plan,
                             threshold=calibration["threshold"], chunk_size=7) as bank:
                bank.write(actual); bank.finalize()
            self.assertEqual([_fingerprint(r) for r in actual],
                             [_fingerprint(r) for r in read_records(directory)])


if __name__ == "__main__":
    unittest.main()
