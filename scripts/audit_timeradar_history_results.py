"""Check sealed selection, actual update scope, AP/ROC and sparse raw score readback."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import rankdata

from hybrid_tsad.timeradar_history_experiment import model
from hybrid_tsad.timeradar_history_evaluation import bank, bounds
from hybrid_tsad.timeradar_history_context import record_window
from hybrid_tsad.native_v_conditioning import load_native, sha
from hybrid_tsad.left_multisource import target_inputs
from hybrid_tsad.trajectory_experiment import target, ROLES, dump
from hybrid_tsad.trajectory_evaluation import labels


def independent_ranking(y, values):
    positive = y.astype(bool)
    p, n = int(positive.sum()), int((~positive).sum())
    roc = (rankdata(values)[positive].sum() - p * (p + 1) / 2) / (p * n)
    order = np.argsort(-values, kind="stable")
    stops = np.r_[np.flatnonzero(np.diff(values[order]) != 0), len(values) - 1]
    tp = positive[order].cumsum()[stops]
    ap = np.sum(np.diff(np.r_[0, tp]) * tp / (stops + 1)) / p
    return float(ap), float(roc)


def run(w, out):
    torch.set_num_threads(1)
    started = time.monotonic()
    official = load_native(w, "cpu").state_dict()
    selection = []
    for arm in ("native", "history"):
        folder = out / "source" / arm
        risk = json.loads((folder / "select.json").read_text())
        assert [r["step"] for r in risk] == list(range(0, 241, 40))
        chosen = min(risk, key=lambda r: (r["copy"], r["step"]))
        receipt = json.loads((folder / "receipt.json").read_text())
        assert chosen["step"] == receipt["selected_step"]
        selection.append(dict(arm=arm, selected_step=chosen["step"], copy=chosen["copy"]))
    samples, rank_checks, updates = [], [], []
    with (out / "metrics.csv").open(newline="") as f:
        metrics = list(csv.DictReader(f))
    for domain in ("PSM", "SWAT"):
        truth = labels(w, domain)
        _, _, scaler, _ = target_inputs(w, domain)
        values = torch.from_numpy(scaler.transform(target(w, domain, "TEST")).astype(np.float32)).cuda()
        origin = ROLES[domain]["TEST"][0]
        for arm in ("native", "history"):
            source = torch.load(out / "source" / arm / "selected.pt", map_location="cpu", weights_only=True)["model"]
            for stage in ("Z", "U"):
                root = out / ("source" if stage == "Z" else domain) / arm
                path = root / "selected.pt"
                receipt = json.loads((root / "receipt.json").read_text())
                assert sha(path) == receipt["sha256"]
                state = torch.load(path, map_location="cpu", weights_only=True)["model"]
                assert all(torch.equal(state["native." + key], v) for key, v in official.items()
                           if not key.startswith("decoder."))
                if stage == "U":
                    curve = json.loads((root / "curve.json").read_text())
                    assert [r["step"] for r in curve] == list(range(1, 2001))
                    def norm(predicate):
                        return sum((state[k] - source[k]).abs().square().sum().item()
                                   for k in state if predicate(k)) ** .5
                    decoder = norm(lambda k: k.startswith("native.decoder."))
                    history = norm(lambda k: not k.startswith("native."))
                    assert decoder > 0
                    assert history > 0 if arm == "history" else history == 0
                    updates.append(dict(domain=domain, arm=arm, steps=2000,
                                        decoder_displacement=decoder, history_displacement=history))
                scores = bank(out, domain, arm, stage)
                panels = ("public", "QUERY") if stage == "Z" else ("QUERY",)
                for panel in panels:
                    lo, hi = bounds(domain, "U" if panel == "QUERY" else "Z")
                    hit = (scores["index"] >= lo) & (scores["index"] < hi)
                    ap, roc = independent_ranking(truth[scores["index"][hit]], scores["score"][hit])
                    row = next(r for r in metrics if (r["domain"], r["method"], r["stage"], r["panel"]) ==
                               (domain, arm, stage, panel))
                    np.testing.assert_allclose([ap, roc], [float(row["AP"]), float(row["ROC_AUC"])],
                                               atol=1e-11, rtol=0)
                    rank_checks.append(dict(domain=domain, arm=arm, stage=stage, panel=panel, AP=ap, ROC_AUC=roc))
                m = model(w, arm)
                m.load_state_dict(state)
                lo, hi = bounds(domain, stage)
                for t in (lo, (lo + hi) // 2, hi - 1):
                    # Independent slicing through the checked record-window API.
                    h, x, meta = record_window(values, t - origin - 99, (0, len(values)), 512)
                    with torch.no_grad():
                        y = m(h, x, **meta, bypass=arm == "native")[1]
                        actual = y.var(0, correction=1).mean(-1)[0, -1].item()
                    stored = scores["score"][np.searchsorted(scores["index"], t)]
                    assert actual == stored
                    samples.append(dict(domain=domain, arm=arm, stage=stage, index=t, score=actual,
                                        exact_forward_readback=True, history_start=t-611, current_start=t-99))
                assert sha(path) == receipt["sha256"]
    dump(out / "independent_review.json", dict(source_selection=selection, update_checks=updates,
        AP_ROC_checks=rank_checks, raw_score_samples=samples, seconds=time.monotonic()-started,
        VUS_scope="same pinned implementation in main evaluator; no independent VUS reimplementation",
        source_scope="saved native nondecoder tensors compared to official; selected source risk rechecked",
        query_scope="no optimization or selection; fixed three endpoint samples per bank after sealing",
        reviewer="separate executable and ranking calculation, not an independent human or agent review"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workspace", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.workspace, a.output)
