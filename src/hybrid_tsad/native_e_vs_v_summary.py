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


if __name__=="__main__":
    build(Path("docs/experiments/native_e_vs_v_20261123"))
