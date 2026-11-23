"""Collect the fixed single-seed E/V comparisons from per-domain metric receipts."""
import csv
from pathlib import Path


METHODS = ("official_V", "official_E", "copy_V", "Vtrain_V", "Etrain_E",
           "official_V_plus_E", "Vtrain_E", "Etrain_V")
RANKS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")
OPERATING = ("TP","FP","TN","FN","normal_FPR","Recall","F1","normal_alarm_fragments",
             "M","Npred","Ntrue","event_precision","event_recall","event_F1",
             "auxiliary_any_hit_events","any_hit_recall","complete_event_macro_coverage",
             "complete_event_pooled_coverage","long_events","long_steady_macro_coverage",
             "long_steady_pooled_coverage","longest_event_miss","hit_only_median_delay")
PAIRS = (("Vtrain_V","Etrain_E"),("Vtrain_V","official_V"),("Etrain_E","official_E"),
         ("Etrain_V","official_V"),("Vtrain_V","Etrain_V"),("Vtrain_E","official_E"),
         ("Etrain_E","Vtrain_E"),("Vtrain_V","copy_V"))
DOMAINS = dict(PSM=("psm",1), MSL=("space",27), SMAP=("space",53), SMD=("smd",28))


def read_rows(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(root):
    ranks, pairs, operating = [], [], []
    for dataset, (subdir, count) in DOMAINS.items():
        by_name = {r["method"]:r for r in read_rows(root/subdir/"ranking_macro.csv") if r["dataset"]==dataset}
        assert set(by_name)==set(METHODS)
        ops = {(r["method"],r["stream"]):r for r in read_rows(root/subdir/"operating.csv") if r["dataset"]==dataset}
        assert len(ops)==2*len(METHODS)
        for method in METHODS:
            ranks.append(dict(dataset=dataset,method=method,entities=count,
                              **{metric:by_name[method][metric] for metric in RANKS}))
            for stream in ("strict","confirmed3"):
                op = ops[method,stream]
                operating.append(dict(dataset=dataset,method=method,stream=stream,
                                      **{metric:op[metric] for metric in OPERATING}))
        for arm,control in PAIRS:
            pairs.append(dict(dataset=dataset,method=arm,reference=control,
                         **{metric:float(by_name[arm][metric])-float(by_name[control][metric]) for metric in RANKS}))
    save(root/"methods.csv",ranks)
    save(root/"paired.csv",pairs)
    save(root/"operating.csv",operating)
    write_tables(root,ranks,pairs,operating)


def write_tables(root, ranks, pairs, operating):
    names = ("Official / V","Official / E","Reconstruction continuation / V",
             "V supervision / V","E supervision / E","Official / V+E",
             "V supervision / E (aux.)","E supervision / V (aux.)")
    titles=dict(zip(METHODS,names))
    op={(r["dataset"],r["method"],r["stream"]):r for r in operating}
    out=["# Matched source-event supervision of V and E, seed 2025",
         "",
         "## Complete method comparison","",
         "AP and ROC-AUC use the full causal endpoint axis. VUS uses the existing LEFT implementation (windowSize 100, 250 thresholds). Each machine or spacecraft entity contributes equally to domain ranking. Threshold columns use each readout’s own fixed CAL q99/higher and strict exceedance. Strict and three-confirmation operating results pool counts across entities. Three-confirmation starts at the third consecutive crossing. Matched and any-hit counts refer to complete eligible events. Long steady coverage begins 512 points after event onset.",
         "",
         "| Domain | Weights / endpoint score | AP | ROC-AUC | VUS-PR | VUS-ROC | Strict FPR / recall | Three-confirmation FPR / recall | Three-confirmation matched / any-hit / eligible | False fragments | Event F1 | Long macro / pooled | Longest miss |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def val(x):
        return f"{float(x):.4f}" if x not in (None,"") else "NA"
    for r in ranks:
        d,m=r["dataset"],r["method"]
        a,b=[op[d,m,s] for s in ("strict","confirmed3")]
        out.append("| "+" | ".join([d,titles[m],*[val(r[k]) for k in RANKS],
             " / ".join(val(a[k]) for k in ("normal_FPR","Recall")),
             " / ".join(val(b[k]) for k in ("normal_FPR","Recall")),
             " / ".join(str(b[k]) for k in ("M","auxiliary_any_hit_events","Ntrue")),
             str(b["normal_alarm_fragments"]),val(b["event_F1"]),
             " / ".join(val(b[k]) for k in ("long_steady_macro_coverage","long_steady_pooled_coverage")),
             str(b["longest_event_miss"])])+" |")
    out.extend(["","[Full precision ranking](methods.csv) · [Strict and three-confirmation counts, fragments, matching, coverage and delay](operating.csv).",
                "","## Fixed training and readout contrasts","",
                "Each cell gives the within-domain difference in AP / ROC-AUC / VUS-PR / VUS-ROC. The two cross-readouts were specified before evaluation.",
                "","| Training / readout contrast | PSM | MSL27 | SMAP53 | SMD28 |","|---|---:|---:|---:|---:|"])
    for arm,ref in PAIRS:
        cells=[]
        for d in DOMAINS:
            r=next(row for row in pairs if (row["dataset"],row["method"],row["reference"])==(d,arm,ref))
            cells.append(" / ".join(f"{r[k]:+.4f}" for k in RANKS))
        out.append("| "+" | ".join([titles[arm]+" minus "+titles[ref]]+cells)+" |")
    out.extend(["","[Full precision paired differences](paired.csv)."])
    (root/"TABLES.md").write_text("\n".join(out)+"\n",encoding="utf-8")

if __name__=="__main__":
    build(Path("docs/experiments/native_e_vs_v_20261123"))
