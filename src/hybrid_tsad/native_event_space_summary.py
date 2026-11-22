"""Aggregate the sealed frozen NASA fold without concatenating entity time series."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


RANKS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")
ARMS = ("frozen_native_V", "copy", "point_weighted")


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fraction(n, d):
    return n / d if d else None


def group_operating(rows, events):
    totals = {key: sum(int(row[key]) for row in rows)
              for key in ("TP", "TN", "FP", "FN", "M", "Npred", "Ntrue",
                          "normal_alarm_fragments", "auxiliary_any_hit_events")}
    eligible = [e for e in events if e["main_eligible"] == "True"]
    assert len(eligible) == totals["Ntrue"]
    steady = [e for e in eligible if int(e["total_points"]) > 512]
    delays = [int(e["delay_points"]) for e in eligible if e["delay_points"]]
    steady_count = sum(int(e["total_points"]) - 512 for e in steady)
    steady_hits = sum(round(float(e["steady_after512_coverage"]) * (int(e["total_points"]) - 512)) for e in steady)
    tp, tn, fp, fn = (totals[k] for k in ("TP", "TN", "FP", "FN"))
    nt, npred, m = (totals[k] for k in ("Ntrue", "Npred", "M"))
    lengths = sum(int(e["total_points"]) for e in eligible)
    return dict(**totals, scored_points=tp+tn+fp+fn, normal_FPR=fraction(fp, tn+fp),
                Recall=fraction(tp, tp+fn), F1=fraction(2*tp, 2*tp+fp+fn),
                normal_alarm_fragments_per_1000=fraction(1000*totals["normal_alarm_fragments"], tn+fp),
                event_precision=fraction(m, npred), event_recall=fraction(m, nt),
                event_F1=fraction(2*m, npred+nt), any_hit_recall=fraction(totals["auxiliary_any_hit_events"],nt),
                ineligible_events=len(events)-nt, long_events=len(steady), long_steady_points=steady_count,
                long_steady_macro_coverage=np.mean([float(e["steady_after512_coverage"]) for e in steady]) if steady else None,
                long_steady_pooled_coverage=fraction(steady_hits, steady_count),
                complete_event_macro_coverage=np.mean([float(e["coverage"]) for e in eligible]) if eligible else None,
                complete_event_pooled_coverage=fraction(sum(int(e["observed_alarm_points"]) for e in eligible), lengths),
                longest_event_miss=max((int(e["longest_miss_points"]) for e in eligible),default=None),
                hit_only_median_delay=np.median(delays) if delays else None)


def summarize(root):
    rank, paired, operating, ledger = [], [], [], []
    for domain, expected in (("MSL",27),("SMAP",53)):
        base = root / domain
        methods = read_csv(base/"methods.csv")
        events = read_csv(base/"events.csv")
        lookup = {(r["entity"], r["method"]): r for r in methods if r["stream"]=="strict"}
        entities = sorted({key[0] for key in lookup})
        assert len(entities) == expected and len(methods) == expected * 6
        for entity in entities:
            info = json.loads((base/f"{entity}.json").read_text(encoding="utf-8"))
            with np.load(base/f"{entity}.npz") as bank:
                anchors = bank["cal_index"]
                first, last = int(anchors.min()), int(anchors.max())
            ledger.append(dict(dataset=domain, entity=entity, sensor=0, fit_stop=info["fit_count"],
                               scaler_mean=info["mean"],scaler_scale=info["scale"],
                               cal_start=info["cal_origin"],cal_stop_exclusive=info["cal_stop"],
                               cal_first_end=first,cal_last_end=last,cal_points=info["cal_points"],
                               query_start=99,query_stop_exclusive=info["query_size"],query_points=info["query_points"],
                               distinct_float32_windows=info["hash_identical_windows"],
                               device_slot=info.get("device_slot",entities.index(entity)%3+1),
                               device_name=info.get("device_name","NVIDIA A40"),
                               elapsed_seconds=info["seconds"],bank_sha256=info["sha256"]))
        for arm in ARMS:
            selected = [lookup[(entity,arm)] for entity in entities]
            means = {k: float(np.mean([float(r[k]) for r in selected])) for k in RANKS}
            medians = {k+"_median": float(np.median([float(r[k]) for r in selected])) for k in RANKS}
            rank.append(dict(dataset=domain, method=arm, entities=expected, **means, **medians))
            for stream in ("strict","confirmed3"):
                by_method = [r for r in methods if r["stream"]==stream and r["method"]==arm]
                event_rows = [e for e in events if e["stream"]==stream and e["method"]==arm]
                operating.append(dict(dataset=domain,method=arm,stream=stream,**group_operating(by_method,event_rows)))
        for i, arm in enumerate(ARMS):
            for ref in ARMS[:i]:
                for key in RANKS:
                    delta = np.array([float(lookup[e,arm][key])-float(lookup[e,ref][key]) for e in entities])
                    assert np.isfinite(delta).all()
                    paired.append(dict(dataset=domain,method=arm,reference=ref,metric=key,entities=len(delta),
                                       average=delta.mean(),median=np.median(delta),wins=int(sum(delta>0)),
                                       ties=int(sum(delta==0)),losses=int(sum(delta<0))))
    for name, rows in (("ranking_macro",rank),("ranking_paired",paired),("operating",operating),("banks",ledger)):
        write_csv(root/f"{name}.csv",rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("result_root", type=Path)
    summarize(p.parse_args().result_root)
