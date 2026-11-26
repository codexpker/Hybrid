"""Post-seal evaluation on the established LEFT-comparable offline axes."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .trajectory_evaluation import (labels, save_rows, independent_check, load_author_metric,
    continuous_metrics, evaluate_score, PSM_CAL, SW_CAL, PSM_AXES, SW_AXIS)
from .trajectory_experiment import sha, dump, ROLES

ROOT = Path("results/intervention_reference_20261126_v1")
PUBLIC = {"PSM":(0,87744), "SWAT":(495000,944856)}
MIXED = {"PSM":PSM_AXES["QUERY"], "SWAT":SW_AXIS}


def read_bank(w,d,role,arm):
    path=w/ROOT/d/"banks"/(role+".npz")
    receipt=json.loads(path.with_suffix(".json").read_text())
    assert sha(path)==receipt["sha256"]
    with np.load(path) as data:
        i,s,a=data["index"],data[arm],data["available"]
    lo,hi=ROLES[d][role]
    assert np.array_equal(i,np.arange(((lo+99)//100)*100,hi//100*100))
    assert np.array_equal(a,np.repeat(i[99::100],100)) and np.isfinite(s).all()
    return i,s,a


def evaluate(args):
    w,metric,role,d,arm=args
    y=labels(w,d)
    i,s,a=read_bank(w,d,role,arm)
    lo,hi=(PUBLIC if role=="TEST" else MIXED)[d]
    use=(i>=lo)&(i<hi)
    i,s,a=i[use],s[use],a[use]
    assert np.array_equal(i,np.arange(lo,hi))
    rank,_=continuous_metrics(y[i],s,load_author_metric(metric))
    ref=independent_check(y[i],s)
    np.testing.assert_allclose(ref,[rank["AP"],rank["ROC_AUC"]],atol=1e-11,rtol=0)
    row=dict(domain=d,role=role,arm=arm,start=lo,stop=hi,**rank)
    op,events=[],[]
    if role=="QUERY":
        ci,cs,_=read_bank(w,d,"CAL",arm)
        p=PSM_CAL if d=="PSM" else SW_CAL
        cp=np.searchsorted(ci,p)
        assert np.array_equal(ci[cp],p)
        tau=float(np.quantile(cs[cp],.99,method="higher"))
        op,events,_,_=evaluate_score(i,s,y,tau,a,query_start=lo,query_stop=hi,
            original_length=len(y),expected_events=None,ranking=rank)
        base=read_bank(w,d,role,"pair")[1][use].astype(float)
        delta=s-base
        row.update(delta_from_Z_mean=float(delta.mean()),delta_from_Z_sd=float(delta.std()),
                   rank_correlation_Z=float(spearmanr(base,s).statistic))
        for item in [*op,*events]:
            item.update(domain=d,arm=arm,cal_attack=int(y[p].sum()),cal_count=len(p))
    return row,op,events


def main():
    p=argparse.ArgumentParser()
    p.add_argument("workspace",type=Path)
    p.add_argument("metric",type=Path)
    p.add_argument("role",choices=("TEST","QUERY"))
    args=p.parse_args()
    start=time.monotonic()
    arms=("official","reconstruction","pair") if args.role=="TEST" else ("pair","replay","absolute","relative")
    out=args.workspace/ROOT/"summary"
    out.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(evaluate,[(args.workspace,args.metric,args.role,d,arm)
            for d in PUBLIC for arm in arms]))
    save_rows(out/(args.role+".csv"),[r for r,_,_ in result])
    if args.role=="QUERY":
        save_rows(out/"operating.csv",[x for _,op,_ in result for x in op])
        save_rows(out/"events.csv",[x for _,_,ev in result for x in ev])
    dump(out/(args.role+"_review.json"),dict(seconds=time.monotonic()-start,
         metric_sha256=sha(args.metric),independent="every AP and ROC recomputed from sorted raw scores; VUS called once per row"))
if __name__=="__main__":
    main()
