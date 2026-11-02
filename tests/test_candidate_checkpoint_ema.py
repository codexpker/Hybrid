"""Synthetic checkpoint fixtures only; official real weights are never opened."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from hybrid_tsad import candidate_tf_lag as candidate
    from hybrid_tsad import candidate_autoformer_lag as lag_module

    class FixtureOfficial(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.decoder = nn.Linear(2, 2)

    class FixtureLag(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.provenance = {"repository": "fixture", "commit": "fixture", "class": "fixture", "source_sha256": "fixture"}


@unittest.skipUnless(HAS_TORCH, "Torch unavailable locally; pending existing remote CPU verification")
class CheckpointEMA(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.upstream = patch.object(candidate, "load_official", side_effect=lambda *a: FixtureOfficial())
        self.upstream.start()
        self.model = candidate.TFLagCandidate(candidate.CandidateConfig(mode="tf"), "fixture.py")
        self.before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def tearDown(self):
        self.upstream.stop(); self.temp.cleanup()

    def payload(self):
        return {k: torch.full_like(v, .25) for k, v in self.model.tf.state_dict().items()}

    def save(self, state):
        path = self.root/"synthetic.pt"; torch.save(state, path)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def ema(self):
        return {"tau_n": torch.tensor(.125), "tau_a": torch.tensor(.75, dtype=torch.float64),
                "tau_inited": torch.tensor(True)}

    def unchanged(self):
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(v, self.before[k], rtol=0, atol=0)

    def test_default_exact_model_and_state_keys(self):
        path, sha = self.save(self.payload())
        with patch.object(self.model.tf, "load_state_dict", wraps=self.model.tf.load_state_dict) as load:
            receipt = self.model.load_tf_checkpoint(path, expected_sha256=sha)
        self.assertIs(load.call_args.kwargs["strict"], True)
        self.assertEqual(set(self.model.state_dict()), set(self.before))
        self.assertFalse(receipt["allow_official_ema_buffers"])
        self.assertEqual(receipt["stripped_official_ema_buffers"], {})
        torch.testing.assert_close(self.model.head.weight, self.before["head.weight"], rtol=0, atol=0)

    def test_explicit_complete_triple_preserves_metadata(self):
        state = dict(self.payload(), **self.ema()); path, sha = self.save(state)
        with patch.object(self.model.tf, "load_state_dict", wraps=self.model.tf.load_state_dict) as load:
            receipt = self.model.load_tf_checkpoint(path, expected_sha256=sha, allow_official_ema_buffers=True)
        self.assertIs(load.call_args.kwargs["strict"], True)
        self.assertEqual(set(load.call_args.args[0]), set(self.model.tf.state_dict()))
        self.assertEqual(set(self.model.state_dict()), set(self.before))
        for key, value in self.ema().items():
            record = receipt["stripped_official_ema_buffers"][key]
            self.assertEqual(record["value"], value.item())
            self.assertEqual(record["shape"], [])
            self.assertEqual(record["dtype"], str(value.dtype))
            self.assertEqual(record["source_checkpoint_sha256"], sha)
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            self.assertEqual(record["raw_tensor_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(self.model.tf_checkpoint_provenance, receipt)
        self.assertIn("_build_model", receipt["ema_metadata_source"]["function"])

    def test_default_rejects_triple_before_load(self):
        path, sha = self.save(dict(self.payload(), **self.ema()))
        with patch.object(self.model.tf, "load_state_dict", side_effect=AssertionError("load called")):
            with self.assertRaises(ValueError):
                self.model.load_tf_checkpoint(path, expected_sha256=sha)
        self.unchanged()

    def test_partial_and_unknown_extras_rejected_before_load(self):
        keys = tuple(self.ema())
        additions = [{k: self.ema()[k] for k in subset}
                     for size in (1, 2) for subset in itertools.combinations(keys, size)]
        additions += [dict(self.ema(), unknown=torch.tensor(1.)), {"module.tau_n": torch.tensor(1.)}]
        for extra in additions:
            path, sha = self.save(dict(self.payload(), **extra))
            with patch.object(self.model.tf, "load_state_dict", side_effect=AssertionError("load called")):
                with self.assertRaises(ValueError):
                    self.model.load_tf_checkpoint(path, expected_sha256=sha, allow_official_ema_buffers=True)
            self.unchanged()

    def test_metadata_dtype_shape_nonfinite_rejected_before_load(self):
        cases = [("tau_n", torch.tensor(1)), ("tau_a", torch.tensor(True)),
                 ("tau_n", torch.tensor(1+2j)), ("tau_a", torch.tensor(float("inf"))),
                 ("tau_n", torch.tensor(float("nan"))), ("tau_n", torch.tensor([1.])),
                 ("tau_inited", torch.tensor(1.)), ("tau_inited", torch.tensor([True]))]
        for key, invalid in cases:
            state = dict(self.payload(), **self.ema()); state[key] = invalid
            path, sha = self.save(state)
            with patch.object(self.model.tf, "load_state_dict", side_effect=AssertionError("load called")):
                with self.assertRaises(ValueError):
                    self.model.load_tf_checkpoint(path, expected_sha256=sha, allow_official_ema_buffers=True)
            self.unchanged()

    def test_model_key_mismatch_still_rejected_before_load(self):
        for change in ("missing", "shape", "dtype", "nan"):
            state = dict(self.payload(), **self.ema()); key = next(iter(self.payload()))
            if change == "missing":
                state.pop(key)
            elif change == "shape":
                state[key] = torch.ones(7)
            elif change == "dtype":
                state[key] = state[key].double()
            else:
                state[key] = torch.full_like(state[key], float("nan"))
            path, sha = self.save(state)
            with patch.object(self.model.tf, "load_state_dict", side_effect=AssertionError("load called")):
                with self.assertRaises(ValueError):
                    self.model.load_tf_checkpoint(path, expected_sha256=sha, allow_official_ema_buffers=True)
            self.unchanged()

    def test_sha_and_non_boolean_optin_rejected(self):
        path, sha = self.save(self.payload())
        with patch.object(torch, "load", side_effect=AssertionError("checkpoint parsed")):
            with self.assertRaises(ValueError):
                self.model.load_tf_checkpoint(path, expected_sha256="0"*64, allow_official_ema_buffers=True)
            with self.assertRaises(ValueError):
                self.model.load_tf_checkpoint(path, expected_sha256=sha, allow_official_ema_buffers="true")
        self.unchanged()

    def config(self, checkpoint):
        cfg = {"model": {"mode": "gated"}, "official_file": "fixture.py",
               "official_sha256": candidate.OFFICIAL_SHA256, "lag_official_file": "fixture_lag.py",
               "lag_official_sha256": "fixture", "lag_official_commit": "fixture", "lag": {},
               "tf_checkpoint": checkpoint}
        path = self.root/"fixture.json"; path.write_text(json.dumps(cfg)); return path

    def test_lag_config_checkpoint_is_na_without_read_or_resolve(self):
        path = self.config({"path": "${UNRESOLVED_NEVER_OPEN}", "sha256": "unread", "allow_official_ema_buffers": True})
        with patch.object(lag_module, "AutoformerLagEndpoint", FixtureLag), \
                patch.object(candidate.TFLagCandidate, "load_tf_checkpoint", side_effect=AssertionError("TF loader called")), \
                patch.object(torch, "load", side_effect=AssertionError("weights opened")):
            model = candidate.candidate_from_config(path, mode="lag")
        self.assertEqual(model.tf_checkpoint_provenance["status"], "NOT_APPLICABLE_LAG_ONLY")
        self.assertFalse(model.tf_checkpoint_provenance["checkpoint_opened"])

    def test_factory_three_tf_modes_explicit_optin_and_provenance(self):
        state = dict(self.payload(), **self.ema()); weights, sha = self.save(state)
        path = self.config({"path": str(weights), "sha256": sha, "allow_official_ema_buffers": True})
        with patch.object(lag_module, "AutoformerLagEndpoint", FixtureLag):
            for mode in ("tf", "concat", "gated"):
                model = candidate.candidate_from_config(path, mode=mode)
                self.assertTrue(model.tf_checkpoint_provenance["allow_official_ema_buffers"])
                self.assertEqual(set(model.tf_checkpoint_provenance["stripped_official_ema_buffers"]), set(self.ema()))


if __name__ == "__main__":
    unittest.main()
