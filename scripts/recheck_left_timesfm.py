"""A separate, CPU-only check of sealed outputs and paired summaries."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from hybrid_tsad.native_full_window_metrics import truth
from hybrid_tsad.trajectory_evaluation import labels


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def review(workspace):
    root = workspace / "results/left_timesfm_hidden_20261126_v1"
    summary = root / "summary"
    frame = pd.read_csv(summary / "entity_metrics.csv")
    learned = frame[frame.method.isin(["A", "B", "C", "D", "E", "A_U", "C_U"])]
    assert len(learned) == 420
    truths, banks, maximum = {}, {}, 0.0
    for row in learned.itertuples():
        key = row.domain, row.entity
        if key not in truths:
            truths[key] = truth(workspace, *key) if row.domain == "SMD" else labels(workspace, row.domain)
        bank_key = *key, row.method, row.role
        if bank_key not in banks:
            path = root / "public" / row.domain / row.entity / f"{row.method}_{row.role}.npz"
            meta = read(path.with_suffix(".json"))
            assert sha(path) == meta["sha256"]
            with np.load(path) as data:
                i, s = data["index"].copy(), data["score"].copy()
                assert np.all(np.diff(i) == 1) and np.isfinite(s).all()
                np.testing.assert_array_equal(data["available"], data["current_start"] + 191)
                np.testing.assert_array_equal(data["history_end_exclusive"], data["history_start"] + 512)
                assert np.all(data["history_end_exclusive"] <= data["current_start"])
                assert data["history_start"].min() >= meta["role_start"]
                assert data["available"].max() < meta["role_stop"]
                if row.method != "D":
                    np.testing.assert_array_equal(data["history_end_exclusive"], data["current_start"])
                banks[bank_key] = i, s
        i, s = banks[bank_key]
        take = (i >= row.start) & (i < row.stop)
        np.testing.assert_array_equal(i[take], np.arange(row.start, row.stop))
        y, score = truths[key][i[take]], s[take]
        if len(np.unique(y)) < 2:
            assert pd.isna(row.AP) and pd.isna(row.ROC_AUC)
        else:
            delta = max(abs(average_precision_score(y, score) - row.AP),
                        abs(roc_auc_score(y, score) - row.ROC_AUC))
            maximum = max(maximum, delta)
            assert delta < 1e-10
    paired = pd.read_csv(summary / "paired_summary.csv")
    for row in paired.itertuples():
        subset = frame[(frame.domain == row.domain) & (frame.panel == row.panel) & (frame.role == row.role)]
        first, second = row.contrast.split("-")
        a = subset[subset.method == first].set_index("entity")
        b = subset[subset.method == second].set_index("entity").loc[a.index]
        np.testing.assert_array_equal(a[["start", "stop"]], b[["start", "stop"]])
        values = (a[row.metric] - b[row.metric]).dropna().to_numpy()
        assert len(values) == row.valid_entities
        assert abs(values.mean() - row.mean) < 1e-10
        assert abs(np.median(values) - row.median) < 1e-10
        assert (values > 1e-12).sum() == row.wins
        assert (np.abs(values) <= 1e-12).sum() == row.ties
        assert (values < -1e-12).sum() == row.losses
    checkpoints = []
    for domain in ("PSM", "SWAT", "SMD"):
        records = [read(root / domain / a / "receipt.json") for a in "ABCDE"]
        parent_root = (
            workspace / "results/left_smd_reliability_20261126_v1/source"
            if domain == "SMD" else
            workspace / "results/left_multisource_20261126_v1" / domain / "source"
        )
        parent = read(parent_root / "receipt.json")
        assert all(r["parent"] == parent["sha256"] and r["endpoint"] == parent["selected"] + 2000
                   for r in records)
        for field in ("parent", "plan_sha256", "native_rng_start", "native_rng_end", "endpoint"):
            assert len({r[field] for r in records}) == 1
        for arm, r in zip("ABCDE", records):
            assert r["updates"] == 2000 and r["current_windows"] == 64000
            path = root / domain / arm / "selected.pt"
            assert sha(path) == r["sha256"]
            checkpoints.append(str(path.relative_to(root)))
        for entity_folder in (root / domain / "U").iterdir():
            rs = [read(entity_folder / a / "receipt.json") for a in ("A", "C")]
            for field in ("source_plan", "target_plan", "initial_step", "endpoint"):
                assert rs[0][field] == rs[1][field]
            for arm, r in zip(("A", "C"), rs):
                assert r["updates"] == 2000 and r["endpoint"] == r["initial_step"] + 2000
                assert r["source_exposure"] == r["target_exposure"] == 32000
                assert r["labels_read"] is False
                assert r["parent"] == read(root / domain / arm / "receipt.json")["sha256"]
                path = entity_folder / arm / "selected.pt"
                assert sha(path) == r["sha256"]
                checkpoints.append(str(path.relative_to(root)))
    assert len(checkpoints) == 75
    result = dict(status="PASS", learned_metric_rows=len(learned), unique_score_banks=len(banks),
                  max_AP_ROC_abs_error=maximum, paired_rows=len(paired), checkpoints=checkpoints,
                  score_review="Separate CPU script: raw banks to sklearn AP/ROC, exact footprints and checkpoint hashes",
                  statistics_review="All paired means, medians, signs and legal entity denominators",
                  VUS_review="Aggregation only; no independent VUS implementation or rerun",
                  reviewer="Root agent using a separate calculation path; not a fresh external reviewer",
                  script_sha256=sha(Path(__file__)))
    (summary / "cache_recheck.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    review(parser.parse_args().workspace)
