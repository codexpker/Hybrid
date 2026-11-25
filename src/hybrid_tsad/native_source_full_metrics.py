"""Original source contract, full-parameter models versus archived decoder controls."""
import sys
import time
import fcntl
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np
from . import native_tep_metrics as metrics
from .native_source_full import OUT,ARMS
from .external_frozen_bank import OUT as EXTERNAL
from .native_tep_readout import AXIS_SHA
from .native_full_window_metrics import METRICS,causal
from .native_v_conditioning_runner import read,dump,sha

metrics.OUT=OUT
metrics.ARMS=ARMS
NAMES={"copy":"Full normal reconstruction / V","point_weighted":"Full V supervision / V",
       "point_weighted_e":"Full E supervision / E"}
MATCHED={"copy":"Source reconstruction continuation","point_weighted":"Source-supervised endpoint V",
         "point_weighted_e":"Decoder E supervision / E"}


def entity(args):
    ranks,ops,events,error,info=metrics.entity(args)
    w,r=args
    # E/E existed on the very same full causal axes, but was omitted from the external-only table.
    ix,scores,binding=causal(w,r["domain"],r["entity"])
    from .native_full_window_metrics import truth,AUTHOR
    from .left_native_swat_metrics import continuous_metrics,load_author_metric
    from .left_context_unmodulated import ranking_check
    y=truth(w,r["domain"],r["entity"]);scorer=load_author_metric(w/AUTHOR)
    for panel in dict.fromkeys(row["panel"] for row in ranks):
        template=next(row for row in ranks if row["panel"]==panel)
        axis=np.arange(template["start"],template["stop"])
        s=scores["Etrain_E"][axis-99]
        values,_=continuous_metrics(y[axis],s,scorer)
        ap,roc=ranking_check(y[axis],s)
        error=max(error,abs(ap-values["AP"]),abs(roc-values["ROC_AUC"]))
        ranks.append(dict(template,method="Decoder E supervision / E",bank_sha256=binding[0]["sha256"],
                          **{k:values[k] for k in METRICS}))
    for row in ranks+ops+events:
        row["method"]=NAMES.get(row["method"],row["method"])
    return ranks,ops,events,error,info


def cached(args):
    w,r=args
    folder=w/OUT/"metric_parts"/r["domain"];folder.mkdir(parents=True,exist_ok=True)
    p=folder/(r["entity"]+".json.gz")
    with p.with_suffix(".lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if not p.exists():
            tick=time.perf_counter();data=entity(args);data[-1]["metric_worker_seconds"]=time.perf_counter()-tick
            tmp=p.with_suffix(".part")
            with gzip.open(tmp,"wt") as out:json.dump(data,out,default=lambda a:a.item(),allow_nan=False)
            tmp.replace(p)
        with gzip.open(p,"rt") as inp:return json.load(inp)


def run(w):
    manifest=w/EXTERNAL/"AXES_AND_INPUTS.json"
    assert sha(manifest)==AXIS_SHA
    todo=read(manifest)["records"]
    ranks=[];ops=[];events=[];infos=[];error=0.
    with ProcessPoolExecutor(max_workers=8) as pool:
        while todo:
            ready=[r for r in todo if (w/OUT/"scores"/r["domain"]/(r["entity"]+".json")).exists()]
            for rr,oo,ee,er,info in pool.map(cached,((w,r) for r in ready)):
                ranks.extend(rr);ops.extend(oo);events.extend(ee);infos.append(info);error=max(error,er)
            todo=[r for r in todo if r not in ready]
            if todo:time.sleep(20)
    root=w/OUT
    frame=pd.DataFrame(ranks)
    prior=pd.read_csv(w/EXTERNAL/"entity_metrics.csv")
    combined=pd.concat([prior,frame],ignore_index=True)
    grouped=combined.groupby(["domain","panel","method","seed","timing"],sort=False)
    macro=grouped[list(METRICS)].mean()
    macro["entities"]=grouped.entity.size();macro["points"]=grouped.points.sum()
    macro.reset_index().to_csv(root/"methods.csv",index=False)
    frame.to_csv(root/"entity_metrics.csv",index=False)
    pairs=[]
    contrasts=[(NAMES[a],MATCHED[a]) for a in ARMS]+[(NAMES["point_weighted"],m) for m in
        (NAMES["copy"],NAMES["point_weighted_e"],"official_V","Complete LEFT","DADA official V",
         "TimeRCD official weights, FIT-only adapter")]
    for (d,p),group in combined.groupby(["domain","panel"]):
        for arm,ref in contrasts:
            a=group[(group.method==arm)&(group.seed==2025)].set_index("entity").sort_index()
            b=group[(group.method==ref)&(group.seed.isin([0,2025]))].set_index("entity").sort_index()
            if a.empty or b.empty:continue
            pd.testing.assert_frame_equal(a[["start","stop"]],b[["start","stop"]])
            for e,row in (a[list(METRICS)]-b[list(METRICS)]).iterrows():
                pairs.append(dict(domain=d,panel=p,entity=e,method=arm,reference=ref,**row.to_dict()))
    pf=pd.DataFrame(pairs);pf.to_csv(root/"entity_pairs.csv",index=False)
    tidy=pf.melt(id_vars=["domain","panel","entity","method","reference"],var_name="metric",value_name="delta")
    tidy.groupby(["domain","panel","method","reference","metric"]).delta.agg(
        mean="mean",median="median",wins=lambda s:(s>0).sum(),ties=lambda s:(s==0).sum(),
        losses=lambda s:(s<0).sum()).reset_index().to_csv(root/"paired_distribution.csv",index=False)
    pd.DataFrame(ops).to_csv(root/"operating_entities.csv",index=False)
    pd.DataFrame(events).to_csv(root/"events.csv",index=False)
    oper=[]
    for (d,m,s),group in pd.DataFrame(ops).groupby(["domain","method","stream"]):
        detail=[{k:"" if v is None else str(v) for k,v in ev.items()} for ev in events
                if (ev["domain"],ev["method"],ev["stream"])==(d,m,s)]
        oper.append(dict(domain=d,method=m,stream=s,**metrics.group_operating(group.to_dict("records"),detail)))
    pd.DataFrame(oper).to_csv(root/"operating.csv",index=False)
    pd.DataFrame(infos).to_csv(root/"inference_cost.csv",index=False)
    dump(root/"EVALUATION.json",dict(entities=len(infos),AP_ROC_max_error=error,rank_rows=len(frame),
        VUS_window=100,VUS_thresholds=250,inference_seconds=sum(r["seconds"] for r in infos),
        train_seconds=sum(read(root/a/"SELECTION.json")["seconds"] for a in ARMS),
        metric_worker_seconds=sum(r["metric_worker_seconds"] for r in infos),axis_sha256=AXIS_SHA))


if __name__=="__main__":
    run(Path.cwd())
