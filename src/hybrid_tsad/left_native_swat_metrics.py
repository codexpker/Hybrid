"""Score an already frozen LEFT–SWaT author-protocol readout."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import sklearn
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from threadpoolctl import threadpool_limits


START = 495000
TEST_POINTS = 449919
WINDOW = 192
SCORED = (TEST_POINTS // WINDOW) * WINDOW
AUTHOR_METRIC_SHA = "808a07460a84d31c87edce1d803e1df231111ec53ffa702ee32eb54a79897dfc"
LABEL_SHA = "3b36bcfb5c519e40b2d4c51d56aff36a015718b826436e2d91657d0864be14bc"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(raw, label, author_metric, out):
    if out.exists():
        raise FileExistsError(out)
    before = time.perf_counter()
    receipt_path = raw.with_name("RAW_SCORE_RECEIPT.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw_sha = sha(raw)
    if (receipt["status"] != "COMPLETED" or receipt["bank_sha256"] != raw_sha
            or receipt["scored"] != [START, START + SCORED] or receipt["labels_read"]):
        raise ValueError("Expected the sealed, label-free native score bank")
    if sha(author_metric) != AUTHOR_METRIC_SHA or sha(label) != LABEL_SHA:
        raise ValueError("Expected frozen author metric source and SWaT label asset")
    with np.load(raw, allow_pickle=False) as bank:
        index = bank["absolute_time_index"]
        score = bank["canonical_score"]
        per_channel = bank["per_channel"]
    if not np.array_equal(index, np.arange(START, START + SCORED)):
        raise ValueError("Native complete-window scoring axis changed")
    if score.shape != (SCORED,) or score.dtype != np.float32 or not np.isfinite(score).all():
        raise ValueError("Expected the finite native float32 channel-mean score")
    if per_channel.shape != (SCORED, 31) or not np.array_equal(score, per_channel.mean(axis=-1)):
        raise ValueError("Canonical score must reproduce the author's channel mean")

    with np.load(label, allow_pickle=False) as bank:
        all_labels = bank["label"]
    if all_labels.shape != (START + TEST_POINTS,) or not np.isin(all_labels, (0, 1)).all():
        raise ValueError("Original SWaT binary label axis changed")
    truth = all_labels[index]
    spec = importlib.util.spec_from_file_location("pinned_left_auc_vus", author_metric)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with threadpool_limits(limits=1):
        ap = average_precision_score(truth, score)
        roc = roc_auc_score(truth, score)
        precision, recall, _ = precision_recall_curve(truth, score)
        trapezoid = auc(recall, precision)
        tpr, fpr, vus_precision, windows, vus_roc, vus_pr = (
            module.metricor().RangeAUC_volume_opt(truth, score, windowSize=100, thre=250)
        )

    result = {
        "status": "COMPLETED",
        "scored_points": SCORED,
        "score_axis": [START, START + SCORED],
        "test_points": TEST_POINTS,
        "unscored_tail_points": TEST_POINTS - SCORED,
        "unscored_tail_positive_points": int(all_labels[START + SCORED:].sum()),
        "scored_positive_points": int(truth.sum()),
        "context": "Offline whole-window scores. Effective availability is each 192-point window end.",
        "available_index_used_in_metrics": False,
        "AP": float(ap),
        "ROC_AUC": float(roc),
        "AUC_PR_trapezoidal": float(trapezoid),
        "VUS_ROC": float(vus_roc),
        "VUS_PR": float(vus_pr),
        "VUS": {"implementation": "LEFT metricor.RangeAUC_volume_opt",
                "windowSize": 100, "thre": 250, "author_metric_sha256": AUTHOR_METRIC_SHA},
        "threshold_metrics": None,
        "raw_sha256": raw_sha,
        "raw_score_receipt_sha256": sha(receipt_path),
        "label_sha256": LABEL_SHA,
        "metrics_source_sha256": sha(Path(__file__)),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cpu_wall_seconds": time.perf_counter() - before,
    }
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / "vus_surfaces.npz", tpr=tpr, fpr=fpr,
                        precision=vus_precision, windows=windows)
    with (out / "METRICS.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("raw", "label", "author-metric", "out"):
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.raw, args.label, args.author_metric, args.out), indent=2))


if __name__ == "__main__":
    main()
