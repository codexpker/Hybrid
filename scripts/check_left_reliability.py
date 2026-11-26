"""CPU check of exposure ranks, matched mass and sealed monitoring metrics."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from hybrid_tsad import left_multisource_eval as ev
from hybrid_tsad import left_multisource as base
from hybrid_tsad.left_reliability import OUT, ARMS, plan, weights_file


def check(w):
    review = {}
    for d in base.WIDTH:
        root = w / OUT / d
        with np.load(root / "weights/teacher.npz") as b:
            starts, scores = b["starts"], b["scores"]
        ranks = pd.read_csv(root / "mixed.csv").set_index("arm")
        ops = pd.read_csv(root / "operations.csv").set_index(["arm", "stream"])
        labels = ev.labels(w, d)
        audit = {}
        for k in "ABC":
            p = plan(w, d, k)
            vals = scores[np.searchsorted(starts, p)]
            flat = np.sort(vals.ravel())
            ww = weights_file(w, d, k)
            # Independently count strict-less and ties for four complete batches.
            ix = [0, 639, 1279, 1999]
            z = vals[ix]
            raw = 1.5 - (
                np.searchsorted(flat, z, "left") + np.searchsorted(flat, z, "right")
            ) / (2 * flat.size)
            expected = raw / raw.mean(axis=(1, 2), keepdims=True)
            np.testing.assert_allclose(ww[ix], expected, rtol=0, atol=1.2e-7)
            mean = ww.mean(axis=2, dtype=np.float32)
            budget = float(np.max(np.abs(mean.mean(axis=1, dtype=np.float64) - 1)))
            assert budget < 2e-7
            assert np.max(np.abs(ww.mean(axis=2, dtype=np.float64) - mean)) < 3e-7
            audit[k] = dict(
                max_target_budget_error=budget, reference_checked_batches=ix
            )
        cal = ev.PSM_CAL if d == "PSM" else ev.SW_CAL
        for arm in ARMS:
            receipt = base.read(root / arm / "receipt.json")
            assert base.sha(root / arm / "selected.pt") == receipt["sha256"]
            q = np.load(root / "banks" / f"{arm}_QUERY.npz")
            c = np.load(root / "banks" / f"{arm}_CAL.npz")
            row = ranks.loc[arm]
            axis = np.arange(int(row.start), int(row.stop))
            at = np.searchsorted(q["index"], axis)
            np.testing.assert_array_equal(q["index"][at], axis)
            assert np.all(
                (q["available"][at] >= axis) & (q["available"][at] - axis < 192)
            )
            truth = labels[axis].astype(bool)
            s = q["score"][at]
            ap = average_precision_score(truth, s)
            roc = roc_auc_score(truth, s)
            np.testing.assert_allclose(
                [ap, roc], [row.AP, row.ROC_AUC], rtol=0, atol=1e-12
            )
            ci = np.searchsorted(c["index"], cal)
            np.testing.assert_array_equal(c["index"][ci], cal)
            tau = np.sort(c["score"][ci])[int(np.ceil(0.99 * (len(ci) - 1)))]
            y = s > tau
            confirmed = np.r_[False, False, y[:-2] & y[1:-1] & y[2:]]
            for stream, alarm in (("strict", y), ("confirmed3", confirmed)):
                np.testing.assert_array_equal(
                    [np.sum(alarm & truth), np.sum(alarm & ~truth)],
                    ops.loc[(arm, stream), ["TP", "FP"]].values,
                )
            audit[arm] = dict(AP=float(ap), ROC_AUC=float(roc), tau=float(tau))
        review[d] = audit
    base.dump(w / OUT / "review.json", review)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    check(p.parse_args().workspace)
