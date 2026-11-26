"""Read back numeric exposure contracts, endpoint identities and q99 operating counts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from types import SimpleNamespace
from sklearn.metrics import average_precision_score, roc_auc_score

from hybrid_tsad.left_memory_update import ARMS, OUT, selector
from hybrid_tsad.left_multisource import dump, read, sha
from hybrid_tsad.left_multisource_eval import labels, PSM_CAL, SW_CAL, ROLES


def check(w):
    # Arithmetic check independent of whether native confidence admits real windows.
    cfg = SimpleNamespace(
        source_writes=True, update_q_err=0.3, update_q_ad=0.3, conf_threshold=0.85
    )
    err = torch.cat([torch.zeros(16), torch.arange(16)]).float()
    confidence = torch.ones(32)
    frequency = confidence.clone()
    frequency[16] = 0.8
    native = (err <= torch.quantile(err, 0.3)) & (frequency >= 0.85)
    q = torch.quantile(err, 0.3)
    result = selector(cfg, native, err, err, confidence, frequency, q, q)
    assert result.nonzero().ravel().tolist() == [17, 18, 19, 20]
    report = {"arithmetic_check_source_slots": [17, 18, 19, 20]}
    for d in ("PSM", "SWAT"):
        root = w / OUT / d
        y = labels(w, d)
        lo, hi = ROLES[d]["U"]
        rng, r2, _ = [
            np.random.default_rng(v) for v in np.random.SeedSequence(2025).spawn(3)
        ]
        first = rng.integers(hi - lo - 191, size=(2000, 16))
        second = r2.integers(hi - lo - 191, size=(2000, 16))
        selected = np.zeros(first.size, bool)
        slots = np.load(root / "plan/audit_slots.npy")
        assert len(np.unique(slots)) == 3200
        selected[slots] = True
        for k in "ABC":
            z = np.load(root / "plan" / (k + ".npz"))
            np.testing.assert_array_equal(z["second"], second)
            np.testing.assert_array_equal(
                z["first"].flat[~selected], first.flat[~selected]
            )
            if k == "A":
                np.testing.assert_array_equal(z["first"], first)
            else:
                starts = z["first"].flat[selected]
                assert (y[lo + starts[:, None] + np.arange(192)] == (k == "B")).all()
        rank = pd.read_csv(root / "mixed.csv").set_index("arm")
        ops = pd.read_csv(root / "operations.csv").set_index(["arm", "stream"])
        cal = PSM_CAL if d == "PSM" else SW_CAL
        report[d] = dict()
        for a in ARMS:
            receipt = read(root / a / "receipt.json")
            assert sha(root / a / "selected.pt") == receipt["sha256"]
            assert sha(root / a / "trace.npz") == receipt["trace_sha256"]
            assert sha(root / "plan" / (a[-1] + ".npz")) == receipt["numeric_plan"]
            q = np.load(root / "banks" / (a + "_QUERY.npz"))
            c = np.load(root / "banks" / (a + "_CAL.npz"))
            row = rank.loc[a]
            idx = np.arange(int(row.start), int(row.stop))
            pos = np.searchsorted(q["index"], idx)
            np.testing.assert_array_equal(q["index"][pos], idx)
            v = q["available"][pos]
            assert np.all((0 <= v - idx) & (v - idx < 192))
            truth = y[idx].astype(bool)
            score = q["score"][pos]
            ap = average_precision_score(truth, score)
            roc = roc_auc_score(truth, score)
            np.testing.assert_allclose(
                [ap, roc], [row.AP, row.ROC_AUC], rtol=0, atol=1e-12
            )
            ci = np.searchsorted(c["index"], cal)
            np.testing.assert_array_equal(c["index"][ci], cal)
            tau = np.sort(c["score"][ci])[int(np.ceil(0.99 * (len(ci) - 1)))]
            strict = score > tau
            three = np.r_[False, False, strict[2:] & strict[1:-1] & strict[:-2]]
            for stream, alarm in (("strict", strict), ("confirmed3", three)):
                np.testing.assert_array_equal(
                    [np.sum(alarm & truth), np.sum(alarm & ~truth)],
                    ops.loc[(a, stream), ["TP", "FP"]].values,
                )
            trace = np.load(root / a / "trace.npz")
            written = trace["written"]
            if a.startswith("restricted"):
                assert not written[:, :16].any()
            else:
                np.testing.assert_array_equal(written, trace["native"])
            report[d][a] = dict(
                AP=float(ap),
                ROC=float(roc),
                threshold=float(tau),
                ema_steps=int(written.any(1).sum()),
                ema_presentations=int(written.sum()),
            )
    dump(w / OUT / "review.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    check(p.parse_args().workspace)
