import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import tempfile
import unittest
import numpy as np
from hybrid_tsad.hybrid_v1 import stage2_u_data as d


class Stage2UDataTests(unittest.TestCase):
    def test_original_event_boundaries_never_merge(self):
        self.assertEqual([(e["start"], e["end"]) for e in d.binary_events([1, 1, 0, 1])], [(0, 2), (3, 4)])

    def test_absolute_guards_and_query_h1_tail(self):
        g = d.role_geometry(28700, 28700)
        self.assertEqual(g["FIT"], [0, 17220])
        self.assertEqual(g["SELECT"], [17764, 22960])
        self.assertEqual(g["CAL"], [23504, 28700])
        self.assertEqual(g["QUERY_H1_SCORED"], [10622, 28700])
        ts = d.eligible_targets(np.ones((28700, 2), bool), *g["QUERY"], arrived=True)
        self.assertEqual((len(ts), ts[-1]), (18078, 28699))

    def test_complete_p32_is_required_not_all544(self):
        y = np.zeros(1500, np.int8); y[600:631] = 1; y[1000:1032] = 1
        report = d.donor_admission(y, np.ones((1500, 1), bool), [0, 1500])
        self.assertEqual(report["primary_abnormal_supervision_same_complete_event"], 1)
        self.assertEqual(report["abnormal_all_544_points"], 0)

    def test_donor_event_crossing_prefix_is_not_clipped(self):
        y = np.zeros(1500, np.int8); y[900:1100] = 1
        report = d.donor_admission(y, np.ones((1500, 1), bool), [0, 1000])
        self.assertEqual(report["complete_donor_events"], [])
        self.assertEqual(report["primary_abnormal_supervision_same_complete_event"], 0)

    def test_partial_query_event_is_ineligible_not_relabelled(self):
        y = np.zeros(15, np.int8); y[3:8] = 1; y[10:13] = 1
        r = d.query_coverage(y, np.arange(6, 15), [5, 15])
        self.assertEqual((r["complete_event_count"], r["partial_event_count"]), (1, 1))
        self.assertEqual(r["events"][0]["start"], 3)
        self.assertEqual(r["events"][0]["scored_points"], 2)

    def test_unobserved_current_and_history_not_imputed(self):
        m = np.ones((15, 1), bool); m[5] = False
        ts = d.eligible_targets(m, 0, 15, history=4, horizon=2)
        self.assertNotIn(5, ts)
        self.assertTrue(all(t-4 >= 0 and t+2 <= 15 for t in ts))

    def test_role_archive_has_no_audit_labels_and_is_exact(self):
        x = np.arange(40, dtype=np.float32).reshape(20, 2)
        m = np.ones(x.shape, bool); m[6, 0] = False
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"FIT.npz"
            meta = d.write_role_npz(p, x, m, [5, 12])
            self.assertEqual((meta["bounds"], meta["original_start"]), ([0, 7], 5))
            with np.load(p, allow_pickle=False) as z:
                self.assertEqual(set(z.files), {"values", "observed"})
                np.testing.assert_array_equal(z["values"], x[5:12])
                np.testing.assert_array_equal(z["observed"], m[5:12])

    def test_create_only_preserves_existing_artifact(self):
        x = np.ones((10, 1), np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"FIT.npz"
            before = d.write_role_npz(p, x, np.ones(x.shape, bool), [0, 10])["sha256"]
            with self.assertRaises(FileExistsError):
                d.write_role_npz(p, x*2, np.ones(x.shape, bool), [0, 10])
            self.assertEqual(d.identity(p)["sha256"], before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
