"""Compute full native-window SMD ranks after all 28 offline score banks are sealed."""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_smd28_author import OLD, RUN, bindings
from .left_swat_author import read, sha

METRIC = Path("preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py")
RANK = ("AP","ROC_AUC","VUS_PR","VUS_ROC")


def entity_metrics(item):
    record, target = item
    e = record["id"]
    out = RUN/e
    train = read(out/"TRAIN.json")
    scored = read(out/"SCORES.json")
    bank = out/"scores.npz"
    path = Path(target["path"])
    assert sha(path) == target["sha256"] and sha(bank) == scored["bank_sha256"]
    with np.load(path) as z:
        labels = z["label"].reshape(-1)
        np.testing.assert_array_equal(z["index"],np.arange(len(labels)))
    with np.load(bank) as z:
        ix, score = z["index"], z["score"]
        np.testing.assert_array_equal(ix,np.arange(len(labels)//192*192))
        np.testing.assert_array_equal(score,z["per_channel"].mean(axis=1))
        np.testing.assert_array_equal(z["available_index"],np.repeat(ix[191::192],192))
    metrics,_ = continuous_metrics(labels[ix],score,load_author_metric(METRIC))
    return dict(entity=e, channels=record["channels"], axis_start=0,axis_stop=len(ix),
                test_points=len(labels), positives=int(labels[ix].sum()), omitted_tail=len(labels)-len(ix),
                omitted_tail_positive=int(labels[len(ix):].sum()),**metrics, train_points=record["train_points"],
                fit_end=train["fit"][1], updates=train["updates"], epochs=train["epochs"],
                selected_epoch=train["selected"]["epoch"],selected_step=train["selected"]["step"],
                selected_loss=train["selected"]["val"], final_model_step=train["final_step"],
                train_seconds=train["train_seconds"],score_seconds=scored["score_seconds"],
                device=train["device"],peak_reserved_bytes=max(train["train_peak_reserved"],scored["score_peak_reserved"]),
                checkpoint_sha256=train["checkpoint_sha256"],score_sha256=scored["bank_sha256"],
                scaler_sha256=train["scaler_sha256"],test_sha256=scored["input_sha256"],label_sha256=target["sha256"])


def save(path,rows):
    with path.open("x",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    records=bindings(Path.cwd())
    labels={r["id"]:r for r in read(OLD/"evaluation_manifest_full28.json")["records"]}
    assert set(records)==set(labels)
    assert all((RUN/e/"SCORES.json").is_file() for e in records)
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(entity_metrics,[(records[e],labels[e]) for e in sorted(records)]))
    save(RUN/"entity_native.csv",rows)
    summary=dict(method="LEFT",mode="offline complete nonoverlapping windows",entities=len(rows),
                 total_scored=sum(r["axis_stop"] for r in rows),
                 total_test=sum(r["test_points"] for r in rows),
                 train_hours=sum(r["train_seconds"] for r in rows)/3600,
                 score_hours=sum(r["score_seconds"] for r in rows)/3600,
                 operating="NA: published normal validation and scaler include project CAL")
    for k in RANK:
        vec=np.array([r[k] for r in rows])
        summary[k]=vec.mean()
        summary[k+"_median"]=np.median(vec)
    save(RUN/"native_summary.csv",[summary])


if __name__=="__main__":
    main()
