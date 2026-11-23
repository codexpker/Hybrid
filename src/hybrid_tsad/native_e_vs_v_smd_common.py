"""Compute only new readouts on the fixed 703,212-point SMD LEFT/TimeRadar intersection."""
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .native_e_vs_v_smd import FRESH_COLS, KEYS, OUT, old
from .native_event_smd28_metrics import LABELS, LABELS_SHA
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .native_v_conditioning_runner import read, sha

ARCHIVE = Path("results/native_event_smd28_common_20261123_v1/common_entity.csv")
ARCHIVE_SHA = "e245de36890e6aec5795820e65ca383ed796c62f409fe50c005fb541baea76f7"
RANKS = ["AP","ROC_AUC","VUS_PR","VUS_ROC"]
NAMES = {"frozen_native_V":"official_V","2025_copy":"copy_V","2025_point_weighted":"Vtrain_V","LEFT":"LEFT"}


def evaluate(args):
    w, record, prior = args
    entity = record["id"]
    bank = w/OUT/f"{entity}.npz"
    assert sha(bank)==read(bank.with_suffix(".json"))["sha256"] and sha(w/record["path"])==record["sha256"]
    with np.load(w/record["path"]) as f:
        labels=f["label"].reshape(-1)
    axis=np.arange(99,len(labels)//192*192)
    assert (prior["first"],prior["stop"],prior["points"],prior["positives"]) == (axis[0],axis[-1]+1,len(axis),labels[axis].sum())
    metric=load_author_metric(w/"preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py")
    rows=[]
    with np.load(bank) as f:
        np.testing.assert_array_equal(f["keys"],KEYS)
        np.testing.assert_array_equal(f["query_index"][:len(axis)],axis)
        values=f["query"][:len(axis)]
        for j in FRESH_COLS:
            ranks,_=continuous_metrics(labels[axis],values[:,j],metric)
            rows.append(dict(entity=entity,model=KEYS[j],first=prior["first"],stop=prior["stop"],
                             points=len(axis),positives=int(labels[axis].sum()),**{k:ranks[k] for k in RANKS}))
    return pd.DataFrame(rows)


def run(w):
    assert sha(w/ARCHIVE)==ARCHIVE_SHA and sha(w/LABELS)==LABELS_SHA
    prior=pd.read_csv(w/ARCHIVE)
    prior=prior[prior.model.isin(NAMES)].copy()
    prior["model"]=prior.model.map(NAMES)
    ref=prior[prior.model=="LEFT"].set_index("entity").to_dict("index")
    remaining={r["id"]:r for r in read(w/LABELS)["records"]}
    assert set(ref)==set(remaining)==set(old.ENTITIES)
    out=w/OUT
    parts=out/"entry_common"
    parts.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        while remaining:
            ready=[k for k in remaining if (out/f"{k}.json").exists()]
            new=[k for k in ready if not (parts/f"{k}.csv").exists()]
            for key,df in zip(new,pool.map(evaluate,[(w,remaining[k],ref[k]) for k in new])):
                tmp=parts/f"{key}.tmp"
                df.to_csv(tmp,index=False)
                tmp.replace(tmp.with_suffix(".csv"))
            for k in ready:
                remaining.pop(k)
            if remaining:
                time.sleep(20)
    full=pd.concat([prior,*[pd.read_csv(parts/f"{k}.csv") for k in sorted(ref)]],ignore_index=True)
    assert not full.duplicated(["entity","model"]).any() and len(full)==28*9
    assert full.groupby("model").points.sum().eq(703212).all()
    full.to_csv(out/"left_common_entity.csv", index=False)
    summary=full.groupby("model")[RANKS].mean().rename_axis("method")
    summary["dataset"],summary["entities"],summary["points"]="SMD",28,703212
    summary.to_csv(out/"left_common_macro.csv")


if __name__=="__main__":
    run(Path.cwd())
