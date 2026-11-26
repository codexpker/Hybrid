"""Independent CPU readback of sealed banks, rank ties, CAL counts and optimizer traces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from hybrid_tsad.trajectory_evaluation import labels, PSM_CAL, SW_CAL
from hybrid_tsad.left_source_constraints import ARMS, OUT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(w):
    report = {}
    for d in ("PSM", "SWAT"):
        root = w / OUT / d
        results = pd.read_csv(root / "mixed.csv").set_index("arm")
        op = pd.read_csv(root / "operations.csv")
        truth = labels(w, d)
        cal_points = PSM_CAL if d == "PSM" else SW_CAL
        for a in ARMS:
            row = results.loc[a]
            train = root / a
            receipt = json.loads((train / "receipt.json").read_text())
            assert digest(train / "steps.csv") == receipt["step_log_sha256"]
            assert digest(train / "selected.pt") == receipt["sha256"]
            cal = np.load(root / "banks" / (a + "_CAL.npz"))
            query = np.load(root / "banks" / (a + "_QUERY.npz"))
            axis = np.arange(int(row.start), int(row.stop))
            positions = np.searchsorted(query["index"], axis)
            np.testing.assert_array_equal(axis, query["index"][positions])
            s = query["score"][positions]
            v = query["available"][positions]
            assert np.all((v >= axis) & (v - axis < 192))
            y = truth[axis].astype(bool)
            order = np.argsort(-s, kind="stable")
            ends = np.r_[np.flatnonzero(np.diff(s[order]) != 0), len(s) - 1]
            tp = y[order].cumsum()[ends]
            ap = np.dot(np.diff(np.r_[0, tp]), tp / (ends + 1)) / y.sum()
            roc = (rankdata(s)[y].sum() - y.sum() * (y.sum() + 1) / 2) / (
                y.sum() * (~y).sum()
            )
            ci = np.searchsorted(cal["index"], cal_points)
            np.testing.assert_array_equal(cal["index"][ci], cal_points)
            tau = sorted(cal["score"][ci])[int(np.ceil(0.99 * (len(ci) - 1)))]
            np.testing.assert_allclose(
                [ap, roc], [row.AP, row.ROC_AUC], atol=1e-12, rtol=0
            )
            alarm = s > tau
            triples = np.zeros_like(alarm)
            triples[2:] = alarm[:-2] & alarm[1:-1] & alarm[2:]
            for stream, pred in (("strict", alarm), ("confirmed3", triples)):
                rec = op.query("arm==@a and stream==@stream").iloc[0]
                np.testing.assert_array_equal(
                    [sum(pred & ~y), sum(pred & y)], [rec.FP, rec.TP]
                )
                np.testing.assert_allclose(rec.threshold, tau, atol=1e-14, rtol=0)
            audit = dict(
                points=len(s),
                AP=float(ap),
                ROC=float(roc),
                threshold=float(tau),
                CAL_attack=int(truth[cal_points].sum()),
            )
            if a != "source_only":
                trace = pd.read_csv(train / "steps.csv").set_index("step")
                assert len(trace) == 2000
                norm = (
                    (trace.tep_actual + trace.smd_actual) / 2
                    if a == "mean"
                    else np.maximum(trace.tep_actual, trace.smd_actual)
                )
                audit["positive_enforced_raw_dots"] = int((norm > 0).sum())
                np.testing.assert_array_equal(
                    trace.loc[trace.trigger == 0, "correction"], 0
                )
                for f, h in receipt["vectors"].items():
                    assert digest(train / f) == h
                    ix = int(Path(f).stem.split("_")[-1])
                    sample = np.load(train / f)
                    p, b = (sample[k].astype(float) for k in ("proposed", "actual"))
                    np.testing.assert_allclose(
                        [np.linalg.norm(p), np.linalg.norm(b), np.linalg.norm(b - p)],
                        trace.loc[ix, ["proposed", "actual", "correction"]],
                        rtol=1e-10,
                        atol=1e-12,
                    )
                audit["sampled_vectors"] = sorted(receipt["vectors"])
            report[d + "/" + a] = audit
    (w / OUT / "review.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    check(p.parse_args().workspace)
