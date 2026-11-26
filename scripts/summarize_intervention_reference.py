"""Render fixed score tables from archived summaries without running a model."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

METRICS = ("AP","ROC_AUC","VUS_PR","VUS_ROC")
DOMAINS = ("PSM","SWAT")
NAMES = dict(official="Official TimeRadar", reconstruction="Source reconstruction",
             pair="Source intervention pairs (Z)", replay="Ordinary reconstruction replay",
             absolute="Absolute intervention reference", relative="Local intervention reference")


def read(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def heading(cols):
    return ["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]+["---:"]*(len(cols)-1))+" |"]


def table(rows, columns, labels):
    data=defaultdict(list)
    for row in rows:
        for metric in METRICS:
            data[row["arm"],row["domain"],metric].append(float(row[metric]))
    lines=heading([columns]+[d+" "+m.replace("_","-") for d in DOMAINS for m in METRICS])
    for arm,label in labels.items():
        cells=[label]
        for d in DOMAINS:
            for m in METRICS:
                sample=data[arm,d,m]
                cells.append("NA" if not sample else f"{mean(sample):.4f}"+
                             (f" ± {stdev(sample):.4f}" if len(sample)>1 else ""))
        lines.append("| "+" | ".join(cells)+" |")
    return "\n".join(lines)


def render(repo):
    root=repo/"docs/experiments/intervention_reference_20261126"
    ext=read(repo/"docs/experiments/left_decoder_context_20261124/public_extension/methods.csv")
    public=read(root/"TEST.csv")
    mixed=read(root/"QUERY.csv")
    public_names={k:NAMES[k] for k in ("official","reconstruction","pair")}
    for row in ext:
        if row["panel"]!="archived_full":
            continue
        name=row["method"]
        if name.startswith("CATCH"):
            name="CATCH W192"
        if name not in ("Native LEFT","CATCH W192","PCA residual","Pointwise level deviation"):
            continue
        public.append(dict(row,arm=name))
        public_names[name]=name
    parts=["## Complete public test",table(public,"Detector",public_names),
           "## Independent mixed-target comparison",table(mixed,"Adaptation",{k:NAMES[k] for k in ("pair","replay","absolute","relative")})]
    differences=[]
    for rows, contrasts in [(public,[("pair","official"),("pair","reconstruction")]),
                           (mixed,[("replay","pair"),("absolute","pair"),("relative","pair"),
                                   ("relative","absolute"),("relative","replay")])]:
        for hi,lo in contrasts:
            for d in DOMAINS:
                a=next(r for r in rows if r["domain"]==d and r["arm"]==hi)
                b=next(r for r in rows if r["domain"]==d and r["arm"]==lo)
                differences.append(dict(arm=hi+" − "+lo,domain=d,**{m:float(a[m])-float(b[m]) for m in METRICS}))
    parts.extend(["## Within-axis differences",table(differences,"Difference",{r["arm"]:r["arm"] for r in differences})])
    cols=("normal_FPR","Recall","normal_alarm_fragments","M","Ntrue","auxiliary_any_hit_events","one_to_one_eventF1",
          "event_point_coverage","long_steady_macro_coverage","long_steady_point_coverage","max_event_miss_points")
    names=("FPR","Recall","Normal fragments","Matched","Events","Any hit","Event F1","Coverage","Long macro","Long pooled","Longest miss")
    lines=heading(["Task","Method","Stream",*names])
    for r in read(root/"operating.csv"):
        values=[f"{float(r[c]):.4f}" if "." in r[c] or "e-" in r[c] else r[c] or "NA" for c in cols]
        lines.append("| "+" | ".join([r["domain"],NAMES[r["arm"]],r["stream"],*values])+" |")
    parts.extend(["## CAL operating points","\n".join(lines)])
    return "\n\n".join(parts)+"\n"

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("repo",type=Path)
    print(render(p.parse_args().repo))
