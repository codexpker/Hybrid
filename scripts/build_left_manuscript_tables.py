"""Fill the active manuscript's numeric tables with formatted archived values."""
import argparse
from pathlib import Path
import re

from summarize_left_exposure_repeats import FREEZE, METRICS, REPEAT, read_rows

DOMAIN = {"PSM":"PSM","SWAT":"SWaT","SMD28":"SMD28"}
DRAFT = Path("manuscript/next_generation_20261115/draft.md")
METRIC_NAMES = ("AP", "ROC-AUC", "VUS-PR", "VUS-ROC")
CONTRASTS = (
    ("W_minus_R_A","W−R in A"),
    ("W_minus_R_B","W−R in B"),
    ("W_minus_R_C","W−R in C"),
    ("R_B_minus_C","R B−C"),
    ("W_B_minus_C","W B−C"),
    ("interaction","Interaction"),
)


def table(header, rows):
    return "\n".join("| " + " | ".join(row) + " |"
                     for row in (header, ["---"]*len(header), *rows))


def value(raw, precision, signed=False):
    if raw=="NA":
        return raw
    sign = "+" if signed else ""
    return format(float(raw),f"{sign}.{precision}f")


def benchmark(rows, panel):
    group = [r for r in rows if r["panel"]==panel]
    mixed = panel=="mixed_suffix"
    precision = 6 if mixed else 5
    method_order = ("A","B","N","C","D","E","F","I","J","CATCH","PCA residual","Pointwise level deviation")
    keys = {k:i for i,k in enumerate(method_order)}
    group.sort(key=lambda r:(list(DOMAIN).index(r["domain"]),keys.get(r["method_id"],len(keys)),r["method"]))
    header = ["Target","Method"] + ([] if mixed else ["Target fitting"]) + list(METRIC_NAMES)
    cells = []
    for row in group:
        item = [DOMAIN[row["domain"]],row["method"]]
        if not mixed:
            item.append(row["target_parameter_fit"])
        for m in METRICS:
            shown = value(row[m],precision)
            if row[m+"_sample_sd"]!="NA":
                shown += " ± "+value(row[m+"_sample_sd"],precision)
            item.append(shown)
        cells.append(item)
    return table(header,cells)


def repeat_tables(paired, summary):
    ap = [r for r in paired if r["metric"]=="AP"]
    ordered_conditions = ("RA","WA","RB","WB","RC","WC")
    headings = ["Target","Seed"]
    endpoint = table(headings+[" ".join(k) for k in ordered_conditions],
                     [[DOMAIN[r["domain"]],r["seed"]]+[value(r[c],6) for c in ordered_conditions] for r in ap])
    differences = table(headings+[label for _,label in CONTRASTS],
                        [[DOMAIN[r["domain"]],r["seed"]]+[value(r[k],6,True) for k,_ in CONTRASTS] for r in ap])
    summary_index = {(r["domain"],r["metric"],r["contrast"]):r for r in summary}
    stats, sensitivity = [], []
    for d in ("PSM","SWAT"):
        for key,label in CONTRASTS:
            r = summary_index[d,"AP",key]
            stats.append([DOMAIN[d],label,value(r["mean"],6,True)+" ± "+value(r["sample_sd"],6),
                          "["+value(r["minimum"],6,True)+", "+value(r["maximum"],6,True)+"]",
                          "/".join(r[k] for k in ("positive","zero","negative"))])
        for key,label in (CONTRASTS[0],CONTRASTS[1],CONTRASTS[-1]):
            sensitivity.append([DOMAIN[d],label]+
                [value(summary_index[d,m,key]["mean"],6,True) for m in METRICS])
    return dict(repeat_endpoints=endpoint, repeat_pairs=differences,
                repeat_stats=table(["Target","Contrast","Mean ± sample SD","Min, max","+/0/−"],stats),
                repeat_metrics=table(["Target","Contrast",*METRIC_NAMES],sensitivity))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository",type=Path)
    args = parser.parse_args()
    repo = args.repository
    draft = repo/DRAFT
    archival = read_rows(repo/FREEZE/"main_metrics.csv")
    repeat = repo/REPEAT
    tables = repeat_tables(read_rows(repeat/"paired.csv"),read_rows(repeat/"paired_summary.csv"))
    for panel in ("public_native_grid","public_external_intersection","mixed_suffix"):
        tables[panel] = benchmark(archival,panel)
    text = draft.read_text(encoding="utf-8")
    for name, rows in tables.items():
        pattern = rf"(<!-- table:{name} -->)\n.*?\n(<!-- end:{name} -->)"
        text, count = re.subn(pattern,lambda m: m[1]+"\n\n"+rows+"\n\n"+m[2],text,flags=re.S)
        assert count==1,name
    draft.write_text(text,encoding="utf-8",newline="\n")


if __name__ == "__main__":
    main()
