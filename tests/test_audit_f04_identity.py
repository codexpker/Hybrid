"""F04 regressions: original-record identity, role permissions, no data training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from hybrid_tsad.hybrid_v1.data import (
    Run, audit_events, load_manifest, record_identity, resolve_plan, support_run,
)


def interval(domain="A", **kw):
    return dict(domain=domain, entity="same", run="same", start=10, end=15, **kw)


class AuditIdentity(unittest.TestCase):
    def test_overlap_other_domain_is_legal(self):
        event = interval(event_id="E")
        normal = dict(interval(matched_event_id="E"), start=30, end=35)
        audit_events([event], [normal], [interval("B", role="QUERY")], 1)

    def test_overlap_same_domain_different_role_is_rejected(self):
        event = interval(event_id="E", role="SUPPORT")
        normal = dict(interval(matched_event_id="E"), start=30, end=35)
        with self.assertRaisesRegex(ValueError, "held-out"):
            audit_events([event], [normal], [interval(role="QUERY")], 1)

    def test_full_normal_footprint_is_domain_aware(self):
        event = dict(interval(event_id="E"), start=40, end=45)
        normal = dict(interval(matched_event_id="E"), start=20, end=25)
        forbidden = dict(interval("B"), start=14, end=18)
        audit_events([event], [normal], [forbidden], 1, history=8)
        with self.assertRaisesRegex(ValueError, "Normal SSL full footprint"):
            audit_events([event], [normal], [dict(forbidden, domain="A")], 1, history=8)

    def test_missing_domain_is_not_inferred(self):
        entry = interval()
        del entry["domain"]
        with self.assertRaisesRegex(ValueError, "domain/entity/run"):
            record_identity(entry)

    def manifest(self, root, runs):
        np.save(root/"data.npy", np.ones((50, 2), np.float32))
        path = root/"manifest.json"
        path.write_text(json.dumps(dict(runs=[dict(path="data.npy", **x) for x in runs])))
        return path

    def test_manifest_cross_domain_same_coordinates_are_legal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = [dict(domain=d, entity="same", run="same", role="FIT", bounds=[0, 20]) for d in ("A", "B")]
            runs = load_manifest(self.manifest(root, entries))
            self.assertEqual([record_identity(x) for x in runs], [("A", "same", "same"), ("B", "same", "same")])

    def test_manifest_cross_role_overlap_rejected_even_with_filter(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entries = [dict(domain="A", entity="same", run="same", role=r, bounds=b)
                       for r, b in [("FIT", [0, 20]), ("QUERY", [19, 40])]]
            path = self.manifest(root, entries)
            for roles in (None, ["FIT"]):
                with self.subTest(roles=roles), self.assertRaisesRegex(ValueError, "overlap"):
                    load_manifest(path, roles=roles)

    def test_adjacent_roles_legal_and_filtered_data_not_opened(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.manifest(root, [dict(domain="A", entity="same", run="same", role=r, bounds=b)
                                        for r, b in [("FIT", [0, 20]), ("QUERY", [20, 40])]])
            raw = json.loads(path.read_text())
            raw["runs"][1]["path"] = "MISSING_QUERY_MUST_NOT_OPEN.npy"
            path.write_text(json.dumps(raw))
            self.assertEqual(len(load_manifest(path, roles=["FIT"])), 1)

    def test_support_lookup_uses_domain_and_role(self):
        x = np.ones((50, 2), np.float32)
        runs = [Run(d, "same", "same", role, x, np.ones_like(x, bool))
                for d, role in [("B", "SUPPORT"), ("A", "FIT"), ("A", "SUPPORT")]]
        self.assertIs(support_run(runs, interval()), runs[2])
        with self.assertRaisesRegex(ValueError, "SUPPORT role"):
            support_run(runs, interval(role="FIT"))
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            support_run(runs[:2], interval())

    def test_disjoint_support_segments_resolve_by_original_interval(self):
        x = np.ones((20, 1), np.float32)
        runs = [Run("A", "same", "same", "SUPPORT", x, np.ones_like(x, bool), start=s)
                for s in (0, 30)]
        self.assertIs(support_run(runs, dict(interval(), start=31, end=33)), runs[1])
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            support_run(runs + [runs[1]], dict(interval(), start=31, end=33))

    def test_plan_still_requires_role_but_distinguishes_domain(self):
        x = np.ones((20, 1), np.float32)
        runs = [Run(d, "same", "same", "FIT", x, np.ones_like(x, bool)) for d in ("A", "B")]
        entry = dict(domain="B", entity="same", run="same", role="FIT", index=8)
        self.assertIs(resolve_plan(runs, [entry], 4, 2)[0][0], runs[1])
        with self.assertRaisesRegex(ValueError, "unknown run/role"):
            resolve_plan(runs, [dict(entry, role="SELECT")], 4, 2)

    def test_legacy_evaluate_keeps_domains_separate_and_does_not_overwrite(self):
        from hybrid_tsad.hybrid_v1.cli import evaluate
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scores = root/"scores.json"
            scores.write_text(json.dumps([
                dict(domain=d, entity="same", run="same", index=1, score=s, valid=True)
                for d, s in [("A", 2.), ("B", 9.)]]))
            labels = root/"labels.npy"
            np.save(labels, np.array([0, 1], np.int64))
            cfg = dict(scores=str(scores), threshold=1., label_runs=[
                dict(domain=d, entity="same", run="same", path=str(labels)) for d in ("A", "B")])
            captured = []
            def fake_metrics(indices, values, truth, threshold):
                captured.append(values.tolist())
                return dict(scores=values.tolist())
            with patch("hybrid_tsad.candidate_experiment.evaluate_arrays", side_effect=fake_metrics):
                evaluate(cfg, root/"out")
            self.assertEqual(captured, [[2.], [9.]])
            self.assertEqual(len(list((root/"out").glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
