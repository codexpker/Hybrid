import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import hashlib
import json
import tempfile
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1 import stage2_u_data as d
from hybrid_tsad.hybrid_v1.stage2_u_matched import no_labels, role_metadata, load_roles, resolve_u_plan


class Stage2UPlanTests(unittest.TestCase):
    def test_registered_independent_streams(self):
        wanted = int.from_bytes(hashlib.sha256(b"HYBRID_STAGE2_U_V1|2025|replace_slots").digest()[:8], "little")
        self.assertEqual(d.stream_seed(2025, "replace_slots"), wanted)
        self.assertNotEqual(wanted, d.stream_seed(2025, "normal_donor"))
        with self.assertRaises(ValueError):
            d.stream_seed(2025, "best_loss")

    def test_unknown_background_is_not_silently_clean(self):
        entry = dict(run="machine-3-1_train", index=600, footprint=[88, 632])
        ledger = d.exposure_ledger([entry], np.zeros(1000, np.int8), [], 2025, "A", [])
        self.assertIsNone(ledger["background_true_anomaly_fraction"])
        self.assertEqual(ledger["unknown_train_H_plus_P_exposure_points"], 544)
        self.assertEqual(ledger["known_positive_fraction_of_total_H_plus_P_exposure_lower_bound"], 0)

    def test_synthetic_end_to_end_physical_roles_and_frozen_plans(self):
        # Synthetic only. This does not open any real machine labels or arrays.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.ones((8000, 38), np.float32); z = np.ones((6000, 38), np.float32)
            y = np.zeros(6000, np.int8); y[800:900] = 1; y[3000:3050] = 1
            np.savetxt(root/"train.txt", x, delimiter=",")
            np.savetxt(root/"test.txt", z, delimiter=",")
            np.savetxt(root/"labels.txt", y, fmt="%d")
            cfg = {"output": str(root/"derived")}
            for field, filename in (("train", "train.txt"), ("test", "test.txt"), ("audit_labels", "labels.txt")):
                cfg[field] = str(root/filename); cfg[field+"_sha256"] = d.identity(root/filename)["sha256"]
            report = d.prepare_machine31(cfg)
            self.assertEqual(len(report["learner_plans"]), 9)
            learner = root/"derived"/"learner"
            shared_select, shared_cal = None, None
            for seed in (2025, 2026, 2027):
                plans = {}
                for condition in ("A", "B", "C"):
                    manifest = learner/("MANIFEST_"+condition+".json")
                    metadata = role_metadata(manifest); no_labels(metadata)
                    self.assertFalse(any("audit_only" in e["path"] for e in metadata["runs"]))
                    roles = load_roles(manifest, {"FIT", "SELECT", "CAL"})
                    plan = json.loads((learner/("PLAN_%d_%s.json" % (seed, condition))).read_text())
                    plans[condition] = plan; no_labels(plan)
                    for role in ("FIT", "SELECT", "CAL"):
                        resolve_u_plan([r for r in roles if r.role == role], plan[role], 512, 32,
                                       role, arrived=role == "CAL")
                    self.assertEqual((len(plan["FIT"]), len(plan["SELECT"]), len(plan["CAL"])), (200, 32, 512))
                    if shared_select is None:
                        shared_select, shared_cal = plan["SELECT"], plan["CAL"]
                    self.assertEqual(plan["SELECT"], shared_select)
                    self.assertEqual(plan["CAL"], shared_cal)
                b_slots = [i for i, e in enumerate(plans["B"]["FIT"]) if e["run"] == "machine-3-1_test"]
                c_slots = [i for i, e in enumerate(plans["C"]["FIT"]) if e["run"] == "machine-3-1_test"]
                self.assertEqual(b_slots, c_slots); self.assertEqual(len(b_slots), 20)
                for i in set(range(200))-set(b_slots):
                    self.assertEqual(plans["A"]["FIT"][i], plans["B"]["FIT"][i])
                    self.assertEqual(plans["A"]["FIT"][i], plans["C"]["FIT"][i])
            self.assertTrue((root/"derived"/"audit_only"/"native_test_labels.npz").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
