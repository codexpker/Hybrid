"""Tabulate the three prespecified within-target contrasts on potential Stage 2 benefits."""
import argparse
from pathlib import Path

import pandas as pd

DOMAINS = ["PSM", "machine-3-2", "machine-3-3"]
ARMS = ["noU", "replay_copy", "replay_pair", "source_pair"]
NAMES = dict(noU="Frozen point-weighted checkpoint", replay_copy="Replay reconstruction",
             replay_pair="Pair-preserving mixed adaptation", source_pair="Source-only continuation")
RANK = ["AP", "ROC_AUC", "VUS_PR", "VUS_ROC"]
OPS = ["normal_FPR", "Recall", "normal_alarm_fragments", "M", "Ntrue", "event_F1",
       "auxiliary_any_hit_events", "long_steady_macro_coverage", "long_steady_pooled_coverage",
       "longest_event_miss", "hit_only_median_delay"]


def table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def fmt(v, sign=False):
    if pd.isna(v):
        return "NA"
    return f"{v:+.5f}" if sign else f"{v:.5f}"


def make_tables(root):
    data = pd.read_csv(root / "methods.csv")
    indexed = data.set_index(["domain", "arm", "stream"])
    assert len(indexed) == len(DOMAINS)*len(ARMS)*2 and indexed.index.is_unique
    rows, paired, difference_rows, operating_rows = [], [], [], []
    for domain in DOMAINS:
        for a in ARMS:
            s = indexed.loc[(domain, a, "strict")]
            t = indexed.loc[(domain, a, "confirmed3")]
            assert s[RANK].notna().all() and s.scored_points == t.scored_points
            rows.append([domain, NAMES[a], *[fmt(s[k]) for k in RANK]])
            for stream, row in [("strict", s), ("confirmed3", t)]:
                operating_rows.append([domain, NAMES[a], stream, fmt(row.normal_FPR), int(row.normal_alarm_fragments),
                                       fmt(row.Recall), f"{int(row.M)}/{int(row.Ntrue)}", int(row.auxiliary_any_hit_events),
                                       fmt(row.event_F1), fmt(row.long_steady_macro_coverage),
                                       fmt(row.long_steady_pooled_coverage), fmt(row.longest_event_miss),
                                       fmt(row.hit_only_median_delay)])
        for ref in ["replay_copy", "noU", "source_pair"]:
            for stream in ("strict", "confirmed3"):
                a, b = [indexed.loc[(domain, arm, stream)] for arm in ("replay_pair", ref)]
                assert a.scored_points == b.scored_points and a.Ntrue == b.Ntrue
                delta = {k: a[k]-b[k] for k in RANK + OPS}
                paired.append(dict(domain=domain, reference=ref, stream=stream, **delta))
                if stream == "strict":
                    difference_rows.append([domain, NAMES[ref], *[fmt(delta[k], True) for k in RANK]])
    pd.DataFrame(paired).to_csv(root / "paired.csv", index=False)
    headings = ["Task", "Method", "AP", "ROC-AUC", "VUS-PR", "VUS-ROC"]
    text = ["# Stage 2 mixed-unlabeled adaptation of the point-weighted native variance detector",
            "## Four methods on the fixed adaptation suffixes", table(headings,rows),
            "## Paired differences", "Each difference is the pair-preserving mixed model minus the named reference on identical target indices. Each target has one seed-2025 observation per arm.",
            table(["Task","Reference",*headings[2:]],difference_rows),
            "## Fixed CAL operating points",
            table(["Task","Method","Stream","Normal FPR","False fragments","Recall","Matched/eligible",
                   "Any-hit","Event F1","Long macro","Long pooled","Longest miss","Hit-only median delay"],operating_rows)]
    (root / "RESULTS.md").write_text("\n\n".join(text)+"\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    make_tables(parser.parse_args().result_directory)
