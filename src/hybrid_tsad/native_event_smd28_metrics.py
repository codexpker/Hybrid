"""Pointwise causal SMD ranking, operating costs, and paired source-seed reporting."""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_swat_replay_repeats import write_csv
from .native_event_smd28 import ENTITIES, OUT
from .native_event_space import operating_metrics
from .native_event_space_summary import group_operating, RANKS
from .native_v_conditioning_runner import read, sha

LABELS = Path("preparations/timeradar_smd_full_20261116_v1/evaluation_manifest_full28.json")
LABELS_SHA = "aee4c628d138cc2355535fb5e0bf24fa103d88fd44a50953b8ed2d4e2a05132b"


def evaluate(args):
    w, record = args
    entity = record["id"]
    root = w / OUT
    bank = root / f"{entity}.npz"
    receipt = read(bank.with_suffix(".json"))
    assert sha(w / record["path"]) == record["sha256"] and sha(bank) == receipt["sha256"]
    metric = load_author_metric(w / "preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py")
    methods, events = [], []
    with np.load(w / record["path"]) as z:
        truth = z["label"].reshape(-1).astype(int)
        np.testing.assert_array_equal(z["index"],np.arange(len(truth)))
    with np.load(bank) as z:
        np.testing.assert_array_equal(z["query_index"],np.arange(99,len(truth)))
        idx = z["query_index"]
        for j,key in enumerate(z["keys"].tolist()):
            result = z["query"][:,j].astype(float)
            cal = z["cal"][:,j].astype(float)
            assert len(cal) == 512
            tau = np.quantile(cal,.99,method="higher")
            rank,_ = continuous_metrics(truth[idx],result,metric)
            rows, details = operating_metrics(idx,result,truth,tau,rank)
            binding = dict(entity=entity,key=key,query_points=len(idx),cal_points=len(cal),cal_exceedances=int((cal>tau).sum()))
            methods.extend({**row,**binding,**rank} for row in rows)
            events.extend({**event,**binding} for event in details)
    return methods,events


def main(w):
    assert sha(w / LABELS) == LABELS_SHA
    labels = read(w / LABELS)["records"]
    root = w / OUT
    assert sorted(r["id"] for r in labels) == sorted(ENTITIES) and all((root / f"{e}.json").exists() for e in ENTITIES)
    methods, events = [], []
    with ProcessPoolExecutor(max_workers=6) as executor:
        for rows, outputs in executor.map(evaluate,[(w,r) for r in labels]):
            methods.extend(rows)
            events.extend(outputs)
    write_csv(root/"entity_methods.csv", methods)
    write_csv(root/"events.csv", events)
    entity = pd.DataFrame(methods).query("stream == 'strict'").set_index(["entity","key"])
    rank = entity[list(RANKS)].groupby("key").mean()
    rank.to_csv(root/"ranking_macro.csv")
    deltas,oper = [],[]
    with (root/"entity_methods.csv").open() as f:
        rows=list(csv.DictReader(f))
    with (root/"events.csv").open() as f:
        events=list(csv.DictReader(f))
    for key in rank.index:
        for stream in ("strict","confirmed3"):
            rr=[r for r in rows if r["key"]==key and r["stream"]==stream]
            ee=[r for r in events if r["key"]==key and r["stream"]==stream]
            assert len(rr)==28
            oper.append(dict(key=key,stream=stream,**group_operating(rr,ee)))
    write_csv(root/"operating.csv",oper)
    for seed in (2025,2026,2027):
        a=entity.xs(f"{seed}_point_weighted",level="key")[list(RANKS)]
        for reference,key in (("frozen_native_V","frozen_native_V"),("copy",f"{seed}_copy")):
            delta=a-entity.xs(key,level="key")[list(RANKS)]
            assert len(delta)==28 and np.isfinite(delta.to_numpy()).all()
            for metric in RANKS:
                v=delta[metric]
                deltas.append(dict(seed=seed,reference=reference,metric=metric,macro_delta=v.mean(),median_delta=v.median(),
                                   wins=int((v>0).sum()),ties=int((v==0).sum()),losses=int((v<0).sum())))
    write_csv(root/"pairs.csv",deltas)
    pd.DataFrame(deltas).groupby(["reference","metric"]).macro_delta.agg(["mean","std","count"]).to_csv(root/"pairs_summary.csv")
    write_csv(root/"banks.csv",[read(root/f"{e}.json") for e in ENTITIES])


if __name__ == "__main__":
    main(Path.cwd())
