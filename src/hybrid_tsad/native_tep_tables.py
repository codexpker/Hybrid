"""Render the two source-support comparison tables from sealed entity aggregates."""
from pathlib import Path
import pandas as pd

METRICS = ("AP","ROC_AUC","VUS_PR","VUS_ROC")
DOMAINS = ("PSM","SMD","MSL","SMAP")
NAMES = {
    "single_V":"SWaT event supervision, matched V",
    "expanded_V":"SWaT + TEP event supervision, V",
    "expanded_E":"SWaT + TEP event supervision, E",
    "official_V":"Official TimeRadar V",
    "official_E":"Official TimeRadar E",
    "official_V_plus_E":"Official weights, fixed V + E",
}
HEADER = "| Method | PSM | SMD28 | MSL27 | SMAP53 |"
RULE = "|---|---:|---:|---:|---:|"


def build(root):
    all_methods=pd.read_csv(root/"methods.csv")
    frame=all_methods[all_methods.panel=="common_offline"]
    methods=frame[["method","seed"]].drop_duplicates()
    lines=["# Fixed source-support comparison","",
           "Each cell gives AP / ROC-AUC / VUS-PR / VUS-ROC. Results for multi-entity datasets are equal-entity means. The three new methods use project seed 2025.","",
           "## Complete detectors on the common external axis","",HEADER,RULE]
    for name,seed in methods.itertuples(index=False,name=None):
        records=frame[(frame.method==name)&(frame.seed==seed)].set_index("domain")
        label=NAMES.get(name,name)
        if seed and name not in NAMES:
            label+=f" (seed {seed})"
        cells=[" / ".join(f"{records.loc[d,k]:.4f}" for k in METRICS) if d in records.index else "NA" for d in DOMAINS]
        lines.append("| "+" | ".join([label,*cells])+" |")
    lines.extend(["","All new scores become available at the window endpoint. Complete LEFT and DADA use full windows. TimeRCD FIT-only uses the whole input block for context. Normalization permissions, input hashes and timestamp intersections are fixed by the preceding external benchmark. PSM PCA uses the current observation. Its CATCH comparisons have target FIT training. The auxiliary historical-context and whole-QUERY normalization panels are preserved separately in methods.csv.","",
        "## Support and deployment statistic on the full causal axis","",
        "| Paired difference | PSM | SMD28 | MSL27 | SMAP53 |",RULE])
    dist=pd.read_csv(root/"paired_distribution.csv")
    for ref in ("single_V","expanded_E"):
        subset=dist[(dist.panel=="causal")&(dist.reference==ref)].set_index(["domain","metric"])
        cells=[" / ".join(f"{subset.loc[(d,k),'mean']:+.4f}" for k in METRICS) for d in DOMAINS]
        lines.append("| "+" | ".join(["Expanded V minus "+NAMES[ref],*cells])+" |")
    lines.extend(["","Each difference was computed within entity on identical points before aggregation. The two V models differ in the source of positive and paired-normal FIT windows. Both use the same combined source-validation panel. The V/E contrast compares training, score-specific selection and deployment together.","",
        "[All panels and full precision](methods.csv) · [Entity scores](entity_metrics.csv) · [Within-entity differences](entity_pairs.csv) · [Median differences and win/tie/loss counts](paired_distribution.csv) · [Operating costs](operating.csv)."])
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    import sys
    build(Path(sys.argv[1]))
