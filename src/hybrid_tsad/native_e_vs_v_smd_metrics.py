"""Evaluate new SMD readouts on existing SMD28 full causal axes; reuse archived V metrics."""
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .native_e_vs_v_smd import FRESH_COLS, KEYS, OUT, old
from .native_event_smd28_metrics import LABELS, LABELS_SHA
from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_swat_replay_repeats import write_csv
from .native_event_space import operating_metrics
from .native_event_space_summary import group_operating, RANKS
from .native_v_conditioning_runner import read, sha


def evaluate(args):
    w,r = args
    bank = w/OUT/f"{r['id']}.npz"
    seal = read(bank.with_suffix(".json"))
    assert sha(bank) == seal["sha256"] and sha(w/r["path"]) == r["sha256"]
    metric = load_author_metric(w/"preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py")
    with np.load(w/r["path"]) as f:
        y = f["label"].reshape(-1)
        np.testing.assert_array_equal(f["index"],np.arange(len(y)))
    rows,events = [],[]
    with np.load(bank) as f:
        q = f["query_index"]
        np.testing.assert_array_equal(f["keys"],KEYS)
        np.testing.assert_array_equal(q,np.arange(99,len(y)))
        np.testing.assert_array_equal(f["query_window_start"],q-99)
        np.testing.assert_array_equal(f["query_available_index"],q)
        for j in FRESH_COLS:
            cal, test = f["cal"][:,j].astype(float), f["query"][:,j].astype(float)
            assert len(cal) == 512
            threshold = np.quantile(cal,.99,method="higher")
            rank,_ = continuous_metrics(y[q],test,metric)
            rr,ee = operating_metrics(q,test,y,threshold,rank)
            identity = dict(entity=r["id"],method=KEYS[j],dataset="SMD",query_points=len(q),
                            cal_points=len(cal),cal_exceedances=int((cal>threshold).sum()))
            rows.extend({**row,**identity,**rank} for row in rr)
            events.extend({**ev,**identity} for ev in ee)
    return rows,events


def archive(w,name,digest):
    names = {"frozen_native_V":"official_V","2025_copy":"copy_V","2025_point_weighted":"Vtrain_V"}
    path = w/old.OUT/name
    assert sha(path)==digest
    with path.open() as file:
        for row in csv.DictReader(file):
            key = row.pop("key")
            if key in names:
                yield dict(row,dataset="SMD",method=names[key])


def run(w):
    assert sha(w/LABELS)==LABELS_SHA
    records = read(w/LABELS)["records"]
    assert sorted(r["id"] for r in records)==sorted(old.ENTITIES)
    root = w/OUT
    assert all((root/f"{r['id']}.json").exists() for r in records)
    methods = list(archive(w,"entity_methods.csv","8489e3a2b15b7e555736e1e6e82f866b63260ccf31dec6a6ba67f41c30cfecc0"))
    events = list(archive(w,"events.csv","147fa5fc3b3fc0469a901d1126b7a2d673f4f4e7ccc9f569689f47854e5fb08e"))
    with ProcessPoolExecutor(max_workers=6) as pool:
        for mr,er in pool.map(evaluate,[(w,r) for r in records]):
            methods.extend(mr)
            events.extend(er)
    write_csv(root/"entity_methods.csv",methods)
    write_csv(root/"events.csv",events)
    entity = pd.read_csv(root/"entity_methods.csv").query("stream=='strict'").set_index(["entity","method"])
    assert entity.index.is_unique and len(entity)==28*len(KEYS)
    summary = entity[list(RANKS)].groupby("method").mean()
    summary["entities"],summary["dataset"]=28,"SMD"
    summary.to_csv(root/"ranking_macro.csv")
    with (root/"entity_methods.csv").open() as f:
        mr = list(csv.DictReader(f))
    with (root/"events.csv").open() as f:
        er = list(csv.DictReader(f))
    ops=[]
    for key in KEYS:
        for stream in ("strict","confirmed3"):
            rr=[r for r in mr if r["method"]==key and r["stream"]==stream]
            ee=[e for e in er if e["method"]==key and e["stream"]==stream]
            assert len(rr)==28
            ops.append(dict(dataset="SMD",method=key,stream=stream,**group_operating(rr,ee)))
    write_csv(root/"operating.csv",ops)
    paired=[]
    for key,ref in (("Vtrain_V","Etrain_E"),("Vtrain_V","official_V"),("Etrain_E","official_E"),
                    ("Etrain_V","official_V"),("Vtrain_V","Etrain_V"),("Vtrain_E","official_E")):
        a,b=[entity.xs(k,level="method")[list(RANKS)] for k in (key,ref)]
        for metric,delta in (a-b).items():
            paired.append(dict(dataset="SMD",method=key,reference=ref,metric=metric,macro_delta=delta.mean(),
                               median_delta=delta.median(),wins=int((delta>0).sum()),ties=int((delta==0).sum()),
                               losses=int((delta<0).sum())))
    write_csv(root/"paired.csv",paired)
    write_csv(root/"banks.csv",[read(root/f"{r['id']}.json") for r in records])


if __name__=="__main__":
    run(Path.cwd())
