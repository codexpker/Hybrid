"""Read sealed banks, join public labels once and evaluate fixed window footprints."""
import argparse
import csv
import json
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from . import trajectory_subspace as ts
from .trajectory_experiment import OUT, NORMAL, MIXED, ROLES, sha, dump, target, blocks
from .left_native_swat_metrics import continuous_metrics, load_author_metric, LABEL_SHA
from .left_swat_natural_u_evaluation import CAL_POINTS as SW_CAL, QUERY_AXIS as SW_AXIS, LABEL_PATH
from .left_psm_replay_evaluation import CAL_POINTS as PSM_CAL, AXES as PSM_AXES
from .timercd_swat_evaluation import evaluate_score


def save_rows(path,rows):
    with path.open("w",newline="") as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def labels(w,d):
    if d=="PSM":
        path=w/"data/common_psm_20261028_v1/test_label.npy"
        assert sha(path)=="309449a288d65de43ff5e58ca2bc06b9913556e46024f5aa0b04153a40b30a69"
        return np.load(path).ravel().astype(int)
    assert sha(w/LABEL_PATH)==LABEL_SHA
    with np.load(w/LABEL_PATH) as z:
        return z["label"].astype(int)


def bank(out,d,a,r):
    dest=out/d/a
    path=dest/(r+".npz")
    assert sha(path)==json.loads((dest/(r+".json")).read_text())["sha256"]
    with np.load(path) as z:
        i,s,v=z["index"],z["score"],z["available"]
        lo,hi=ROLES[d][r]
        assert np.array_equal(i,np.arange(lo,lo+(hi-lo)//96*96))
        assert np.array_equal(v,np.repeat(i[95::96],96)) and np.isfinite(s).all()
        return i,s,v


def independent_check(y,s):
    npos=int(y.sum())
    nneg=len(y)-npos
    auc=(rankdata(s)[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)
    order=np.argsort(-s,kind="stable")
    ranked=y[order]
    ends=np.r_[np.flatnonzero(np.diff(s[order])!=0),len(y)-1]
    hits=ranked.cumsum()[ends]
    ap=np.sum(np.diff(np.r_[0,hits])*hits/(ends+1))/npos
    return ap,auc


def evaluate_one(args):
    w,d,a,r,metric_path=args
    out=w/OUT
    y=labels(w,d)
    i,s,v=bank(out,d,a,r)
    full=(int(i[0]),int(i[-1])+1)
    if r=="TEST":
        common=(0,87744) if d=="PSM" else (495000,944856)
    else:
        common=PSM_AXES["QUERY"] if d=="PSM" else SW_AXIS
    assert full[0]<=common[0]<common[1]<=full[1]
    rows,oper,events=[],[],[]
    cached={}
    for axis,bounds in (("W96",full),("common",common)):
        take=slice(bounds[0]-full[0],bounds[1]-full[0])
        idx,score,available=i[take],s[take],v[take]
        if bounds not in cached:
            start=time.monotonic()
            rank,_=continuous_metrics(y[idx],score,load_author_metric(metric_path))
            ap,roc=independent_check(y[idx],score)
            np.testing.assert_allclose([ap,roc],[rank["AP"],rank["ROC_AUC"]],rtol=0,atol=1e-11)
            cached[bounds]=(rank,time.monotonic()-start)
        rank,seconds=cached[bounds]
        rows.append(dict(domain=d,method=a,role=r,axis=axis,start=bounds[0],stop=bounds[1],
                         metric_seconds=seconds if axis=="W96" or common!=full else 0.,**rank))
        if axis=="common" and r=="QUERY":
            ci,cs,_=bank(out,d,a,"CAL")
            points=PSM_CAL if d=="PSM" else SW_CAL
            where=np.searchsorted(ci,points)
            assert np.array_equal(ci[where],points)
            tau=float(np.quantile(cs[where],.99,method="higher"))
            streams,detail,_,_=evaluate_score(idx,score,y,tau,available,query_start=bounds[0],
                      query_stop=bounds[1],original_length=len(y),expected_events=None,ranking=rank)
            for line in streams:
                oper.append(dict(domain=d,arm=a,cal_attack=int(y[points].sum()),cal_points=len(points),**line))
            for line in detail:
                events.append(dict(domain=d,arm=a,**line))
    return rows,oper,events


def examples(w,d,report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=w/OUT
    dest=out/d/"source_robust"
    i,s,_=bank(out,d,"source_robust","QUERY")
    with np.load(dest/"scaler.npz") as sc:
        x=blocks((target(w,d,"QUERY")-sc["mean"])/sc["scale"])
    y=np.load(dest/"QUERY_normal.npy",mmap_mode="r")
    labels_all=labels(w,d)
    lo,hi=PSM_AXES["QUERY"] if d=="PSM" else SW_AXIS
    index=i.reshape(-1,96)
    complete=(index[:,0]>=lo)&(index[:,-1]<hi)
    fig,axs=plt.subplots(2,2,figsize=(9,5.7),layout="constrained")
    meta=[]
    for col,label in enumerate((0,1)):
        hits=np.flatnonzero(complete & np.all(labels_all[index]==label,axis=1))
        if not len(hits):
            raise ValueError("No complete example of requested class")
        n=hits[0]
        raw,clean=x[n],y[n]
        residual=raw-clean
        o=ts.soft(residual)
        rms=lambda a:np.sqrt(np.mean(a*a,axis=1))
        np.testing.assert_allclose(abs(residual).mean(1),s.reshape(-1,96)[n])
        ax=axs[0,col]; ax.plot(index[n],rms(raw),label="Observed X",color="#333333")
        ax.plot(index[n],rms(clean),label="Estimated Y",color="#007A78",linestyle="--")
        ax.set_title(("Label-normal" if label==0 else "Labeled event")+f"  [{index[n,0]}, {index[n,-1]+1})")
        ax.set_ylabel("Channel RMS (FIT units)"); ax.legend()
        ax=axs[1,col];ax.plot(index[n],rms(o),label="Sparse deviation O",color="#B05126")
        ax.plot(index[n],rms(residual-o),label="Remaining residual",color="#666699",linestyle=":")
        ax.set_xlabel("Record index");ax.set_ylabel("Channel RMS (FIT units)");ax.legend()
        meta.append(dict(domain=d,label=label,start=int(index[n,0]),stop=int(index[n,-1]+1),
                         active_fraction=float(np.mean(o!=0)),mean_score=float(abs(residual).mean())))
    fig.suptitle(d+"  |  First complete blocks of each label on the shared U axis")
    fig.savefig(report/(d+"_decomposition.png"),dpi=150)
    plt.close(fig)
    return meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace",type=Path)
    parser.add_argument("metric",type=Path)
    parser.add_argument("--workers",type=int,default=12)
    args=parser.parse_args()
    w=args.workspace
    out=w/OUT
    tasks=[(w,d,a,r,args.metric) for d in ROLES for a in dict.fromkeys(NORMAL+MIXED)
           for r in (("TEST",) if a in NORMAL else ())+(("QUERY",) if a in MIXED else ())]
    for _,d,a,r,_ in tasks:
        for role in (("CAL",r) if r=="QUERY" else (r,)):
            bank(out,d,a,role)
    report=out/"summary"
    report.mkdir(exist_ok=True)
    with threadpool_limits(1), ProcessPoolExecutor(args.workers) as pool:
        result=list(pool.map(evaluate_one,tasks))
    for col,name in enumerate(("ranking","operating","events")):
        save_rows(report/(name+".csv"),[row for group in result for row in group[col]])
    save_rows(report/"examples.csv",[row for d in ROLES for row in examples(w,d,report)])
    diagnostics=[]
    seals=[]
    for d in ROLES:
        for a in dict.fromkeys(NORMAL+MIXED):
            folder=out/d/a
            rec=json.loads((folder/"train.json").read_text())
            seals.append(dict(domain=d,arm=a,model=rec["model_sha256"],scaler=rec["scaler_sha256"],
                              fit_wall=rec["seconds"],reference_seconds=rec["reference_covariance_seconds"]))
            for log in rec["sweeps"]:
                n=log["sweep"]; diag=np.load(folder/f"sweep{n}_diagnostics.npy")
                diagnostics.append(dict(domain=d,arm=a,role="train",**log,mean_iterations=diag[:,0].mean(),
                                        max_residual=diag[:,1].max()))
            for role in ("TEST","CAL","QUERY"):
                path=folder/(role+".json")
                if path.is_file():
                    info=json.loads(path.read_text())
                    with np.load(folder/(role+".npz")) as z:
                        diag=z["diagnostics"]
                    diagnostics.append(dict(domain=d,arm=a,role=role,**info,mean_iterations=diag[:,0].mean(),
                                            max_residual=diag[:,1].max()))
    save_rows(report/"diagnostics.csv",diagnostics)
    save_rows(report/"models.csv",seals)
    dump(report/"numeric_check.json",dict(comparisons=len(tasks),AP_ROC="all full and common banks, custom tie-group AP and rank-sum ROC, absolute tolerance 1e-11",
         VUS="pinned implementation only; no second VUS implementation",metric_sha256=sha(args.metric),
         python=platform.python_version(),numpy=np.__version__))


if __name__=="__main__":
    main()
