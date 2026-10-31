"""Opt-in CPU-only check; no data, training, GPU or checkpoint output."""
import importlib.util
import json
import os
from pathlib import Path
import unittest
import tempfile

from hybrid_tsad.config import OFFICIAL_SHA, sha256
from engine_fixture_support import approve_fixture, calibration_file

OFFICIAL = os.environ.get("HYBRID_OFFICIAL_FILE")
HAS_TORCH = importlib.util.find_spec("torch") is not None
LEGACY = Path(os.environ.get("HYBRID_LEGACY_MODEL",
              str(Path(__file__).resolve().parents[2] / "src/hybrid_tsad/model.py")))


@unittest.skipUnless(HAS_TORCH and OFFICIAL and LEGACY.is_file(),
                     "requires torch, HYBRID_OFFICIAL_FILE and existing HYBRID_LEGACY_MODEL/reference source")
class ModelCompatibilityTests(unittest.TestCase):
    def test_inference_pipeline_executes_on_synthetic_inputs_cpu(self):
        import numpy as np
        import torch
        from hybrid_tsad.api import create_predictor
        from hybrid_tsad.engine.pipeline import run_operation
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            contract = root / "contract.md"
            contract.write_text("synthetic engineering fixture; no training")
            approval = root / "approval.json"
            approval.write_text(json.dumps({"status": "APPROVED", "contract_sha256": sha256(contract)}))
            torch.manual_seed(41)
            model = create_predictor(OFFICIAL, device="cpu")
            checkpoint = root / "random_fixture.pt"
            torch.save(model.state_dict(), checkpoint)
            data = root / "synthetic.npy"
            np.save(data, np.linspace(0, 1, 410, dtype="float32")[:, None])
            query = root / "query.json"
            query.write_text(json.dumps({"history": 400, "horizon": 5, "origins": [400, 405],
                                         "expected_time_index": list(range(400, 410))}))
            record = lambda p: {"path": str(p), "sha256": sha256(p)}
            calibration = calibration_file(root / "source_calibration.json", scale=.7, threshold=1.0)
            cfg = dict(engine_schema_version=1, operation="infer", contract=record(contract), approval=record(approval),
                       inputs=dict(official=record(Path(OFFICIAL)), checkpoint=record(checkpoint), array=record(data),
                                   query=record(query), calibration=record(calibration)),
                       parameters=dict(mode="hybrid", device="cpu", seed=41, source_scale=.7, mask_seed=53,
                                       readout_job="synthetic", job="fixture_not_training"), output=str(root / "bank"))
            path = root / "operation.json"
            approve_fixture(cfg, path)
            result = run_operation(path, "infer", execute=True)
            self.assertTrue(result["executed"])
            with np.load(root / "bank/predictions_and_scores.npz") as bank:
                self.assertTrue(np.array_equal(bank["time_index"], np.arange(400, 410)))
                self.assertEqual(bank["prediction"].shape, (10, 1))
            report = json.loads((root / "bank/inference.json").read_text())
            self.assertFalse(report["labels_opened"])
            self.assertFalse(report["training"])

    def test_legacy_state_dict_and_prediction_replay(self):
        import torch
        from hybrid_tsad.api import create_predictor, predict

        self.assertEqual(sha256(OFFICIAL), OFFICIAL_SHA)
        original = LEGACY
        spec = importlib.util.spec_from_file_location("hybrid_legacy_reference", original)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        records = []
        # CPU only and controlled RNG. Check all three modes and strict keys/shapes.
        with torch.random.fork_rng(devices=[]):
            for mode in ("temporal", "tf", "hybrid"):
                torch.manual_seed(41)
                old = legacy.HybridPredictor(OFFICIAL, mode=mode).cpu().eval()
                torch.manual_seed(41)
                new = create_predictor(OFFICIAL, mode=mode, device="cpu")
                self.assertEqual(list(old.state_dict()), list(new.state_dict()))
                new.load_state_dict(old.state_dict(), strict=True)
                self.assertTrue(all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items()))
                x = torch.linspace(0, 1, 400).reshape(1, 400, 1)
                torch.manual_seed(53)
                with torch.inference_mode():
                    expected = old(x)
                torch.manual_seed(53)
                actual = predict(new, x)
                self.assertEqual(actual["prediction"].shape, (1, 5, 1))
                self.assertTrue(torch.equal(actual["prediction"], expected["prediction"]))
                records.append({"mode": mode, "state_dict_keys": len(old.state_dict()),
                                "strict_load": True, "prediction_bit_exact": True,
                                "shape": list(actual["prediction"].shape)})
                with tempfile.TemporaryDirectory() as tmp:
                    checkpoint = Path(tmp) / "state.pt"
                    torch.save(old.state_dict(), checkpoint)
                    loaded = create_predictor(OFFICIAL, mode=mode, checkpoint=checkpoint, device="cpu")
                    self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k, v in old.state_dict().items()))
        print(json.dumps({"torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                          "torch_threads": torch.get_num_threads(), "legacy_sha256": sha256(original),
                          "official_sha256": OFFICIAL_SHA, "checks": records,
                          "training": False, "device": "cpu"}, indent=2))
