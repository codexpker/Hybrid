"""Common-axis ranking, original calibrated operating protocols and within-entity contrasts."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import fcntl
import gzip
import json
import sys
import time
import numpy as np
import pandas as pd
from .native_tep_support import OUT, ARMS
from .native_tep_readout import AXIS_SHA
from .external_frozen_bank import OUT as EXTERNAL
from . import external_endpoint_metrics as ext
from .native_full_window_metrics import METRICS, AUTHOR, truth
from .left_native_swat_metrics import load_author_metric, continuous_metrics
from .left_context_unmodulated import ranking_check
from .native_event_space import operating_metrics
from .native_event_space_summary import group_operating
from .native_e_vs_v_psm import SUFFIX, evaluate_score, causal_timing
from .native_v_conditioning_runner import read, dump, sha


def entity(args):
    w,r = args
    d,e = r["domain"],r["entity"]
    root = w/OUT
    bank = root/"scores"/d/(e+".npz")
    info = read(bank.with_suffix(".json"))
    assert sha(bank)==info["sha256"] and info["native_parity_points"]==7
    y = truth(w,d,e)
    metric = load_author_metric(w/AUTHOR)
    ranks,ops,events = [],[],[]
    max_error = 0.
    with np.load(bank) as b:
        ci,qi = b["cal_index"],b["query_index"]
        assert b["keys"].tolist()==list(ARMS)
        np.testing.assert_array_equal(qi,np.arange(*r["causal_axis"]))
        for j,arm in enumerate(ARMS):
            score = b["query"][:,j].astype(float)
            tau = np.quantile(b["cal"][:,j],.99,method="higher")
            panels = {"causal":r["causal_axis"],"common_offline":r["offline_axis"]}
            if "history_axis" in r:
                panels["causal_history"]=r["history_axis"]
            for panel,(lo,hi) in panels.items():
                ix = np.arange(lo,hi)
                s = score[ix-99]
                vals,_ = continuous_metrics(y[ix],s,metric)
                if panel=="causal":
                    causal_rank=vals
                ap,roc = ranking_check(y[ix],s)
                max_error=max(max_error,abs(ap-vals["AP"]),abs(roc-vals["ROC_AUC"]))
                ranks.append(dict(domain=d,entity=e,method=arm,seed=2025,panel=panel,start=lo,stop=hi,
                    points=len(ix),positives=int(y[ix].sum()),timing="causal W100 endpoint",
                    bank_sha256=info["sha256"],**{k:vals[k] for k in METRICS}))
            if d=="PSM":
                assert y[ci].sum()==len(ci)==512
                rr,ee,_,_ = evaluate_score(SUFFIX,score[SUFFIX-99],y,tau,SUFFIX,query_start=23016,
                    query_stop=87744,original_length=len(y),expected_events=(48,5),ranking=None)
                causal_timing(rr,ee)
            else:
                rr,ee = operating_metrics(qi,score,y,tau,causal_rank)
            ident = dict(domain=d,entity=e,method=arm)
            ops.extend(dict(k,**ident,cal_n=len(ci),cal_threshold=tau) for k in rr)
            events.extend(dict(k,**ident) for k in ee)
    assert max_error<1e-12
    return ranks,ops,events,max_error,info


def evaluate(w):
    root = w/OUT
    prior = w/EXTERNAL
    manifest = prior/"AXES_AND_INPUTS.json"
    assert sha(manifest)==AXIS_SHA
    rows = read(manifest)["records"]
    ranks,ops,events,costs=[],[],[],[]
    err=0.
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=12) as pool:
        for rs,os,es,er,info in pool.map(cached_entity,((w,r) for r in rows)):
            ranks.extend(rs)
            ops.extend(os)
            events.extend(es)
            costs.append(info)
            err=max(err,er)
    assert len(costs)==109
    frame=pd.concat([pd.read_csv(prior/"entity_metrics.csv"),pd.DataFrame(ranks)],ignore_index=True)
    fields=["domain","panel","method","seed","timing"]
    grouped=frame.groupby(fields,sort=False)
    macro=grouped[list(METRICS)].mean()
    macro["entities"]=grouped.entity.size()
    macro["points"]=grouped.points.sum()
    macro.reset_index().to_csv(root/"methods.csv",index=False)
    pd.DataFrame(ranks).to_csv(root/"entity_metrics.csv",index=False)
    pairs=[]
    for (d,panel),group in frame.groupby(["domain","panel"]):
        a=group[group.method=="expanded_V"].set_index("entity").sort_index()
        if a.empty:
            continue
        for (m,seed),ref in group.groupby(["method","seed"]):
            if m=="expanded_V":
                continue
            b=ref.set_index("entity").sort_index()
            assert a[["start","stop","points"]].equals(b[["start","stop","points"]])
            for entity_id,row in (a[list(METRICS)]-b[list(METRICS)]).iterrows():
                pairs.append(dict(domain=d,panel=panel,entity=entity_id,reference=m,ref_seed=seed,**row.to_dict()))
    pf=pd.DataFrame(pairs)
    pf.to_csv(root/"entity_pairs.csv",index=False)
    tidy=pf.melt(id_vars=["domain","panel","reference","ref_seed","entity"],var_name="metric",value_name="delta")
    dist=tidy.groupby(["domain","panel","reference","ref_seed","metric"]).delta.agg(
        mean="mean",median="median",wins=lambda s:(s>0).sum(),ties=lambda s:(s==0).sum(),losses=lambda s:(s<0).sum())
    dist.reset_index().to_csv(root/"paired_distribution.csv",index=False)
    pd.DataFrame(ops).to_csv(root/"operating_entities.csv",index=False)
    ef=pd.DataFrame(events)
    ef.to_csv(root/"events.csv",index=False)
    oper=[]
    def str_row(row):
        return {k:"" if v is None or pd.isna(v) else str(v) for k,v in row.items()}
    for (d,m,stream),part in pd.DataFrame(ops).groupby(["domain","method","stream"]):
        detail=[str_row(ev) for ev in events if (ev["domain"],ev["method"],ev["stream"])==(d,m,stream)]
        oper.append(dict(domain=d,method=m,stream=stream,**group_operating(part.to_dict("records"),detail)))
    pd.DataFrame(oper).to_csv(root/"operating.csv",index=False)
    pd.DataFrame(costs).to_csv(root/"inference_cost.csv",index=False)
    dump(root/"EVALUATION.json",dict(source_selection_before_scoring=True,axis_sha256=AXIS_SHA,
        aggregation_wall_seconds=time.perf_counter()-start,entities=len(rows),AP_ROC_max_error=err,
        metric_worker_seconds=sum(r["metric_worker_seconds"] for r in costs),
        rank_rows=len(ranks),VUS_window=100,VUS_thresholds=250,
        train_seconds=sum(read(root/a/"SELECTION.json")["seconds"] for a in ARMS),
        inference_seconds=sum(c["seconds"] for c in costs),source_sha256=sha(__file__)))


def cached_entity(args):
    w,r=args
    folder=w/OUT/"metric_parts"/r["domain"]
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/(r["entity"]+".json.gz")
    with path.with_suffix(".lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if not path.exists():
            tick=time.perf_counter()
            result=entity(args)
            result[-1]["metric_worker_seconds"]=time.perf_counter()-tick
            temp=path.with_suffix(".part")
            with gzip.open(temp,"wt",encoding="utf8") as out:
                json.dump(result,out,default=lambda x:x.item(),allow_nan=False)
            temp.replace(path)
        with gzip.open(path,"rt",encoding="utf8") as inp:
            return json.load(inp)


def collect(w):
    manifest=w/EXTERNAL/"AXES_AND_INPUTS.json"
    assert sha(manifest)==AXIS_SHA
    pending=read(manifest)["records"]
    with ProcessPoolExecutor(max_workers=8) as pool:
        while pending:
            ready=[r for r in pending if (w/OUT/"scores"/r["domain"]/(r["entity"]+".json")).exists()]
            for _ in pool.map(cached_entity,((w,r) for r in ready)):
                pass
            pending=[r for r in pending if r not in ready]
            if pending:
                time.sleep(20)


if __name__=="__main__":
    (collect if sys.argv[1:]==["collect"] else evaluate)(Path.cwd())
