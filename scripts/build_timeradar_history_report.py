"""Build development tables/figures from sealed, full-precision run artifacts.

This command never reads raw target arrays or invokes a model.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METRICS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")
NAMES = {
    "history": "TimeRadar + trainable history context",
    "native": "Matched native decoder continuation",
    "official": "Official TimeRadar, matched project readout",
    "multisource_LEFT": "Frozen multi-source LEFT (offline W192)",
}


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def build(run, dest):
    dest.mkdir(parents=True, exist_ok=True)
    data = rows(run / "metrics.csv")
    for row in data:
        for key in METRICS:
            row[key] = float(row[key])
    lookup = {(r["domain"], r["method"], r["stage"], r["panel"]): r for r in data}
    differences = []
    for domain in ("PSM", "SWAT"):
        for stage, panel in (("Z", "public"), ("Z", "QUERY"), ("U", "QUERY")):
            ref = lookup[domain, "native", stage, panel]
            candidate = lookup[domain, "history", stage, panel]
            differences.append(dict(domain=domain, comparison="history_minus_native", stage=stage,
                                    panel=panel, start=ref["start"], stop=ref["stop"],
                                    **{m: candidate[m] - ref[m] for m in METRICS}))
        for method in ("native", "history"):
            before = lookup[domain, method, "Z", "QUERY"]
            after = lookup[domain, method, "U", "QUERY"]
            differences.append(dict(domain=domain, comparison=method + "_U_minus_Z", stage="U-Z",
                                    panel="QUERY", start=before["start"], stop=before["stop"],
                                    **{m: after[m] - before[m] for m in METRICS}))
    write_csv(dest / "paired_differences.csv", differences)
    text = ["# Paired source and full-U development results", "",
            "Single project training seed 2025. Values below retain separate public and QUERY panels. "
            "All ranking differences are computed on identical ticks within a panel. "
            "LEFT retains its offline context permission. No calibration threshold is introduced.", ""]
    for stage, panel in (("Z", "public"), ("Z", "QUERY"), ("U", "QUERY")):
        text.extend([f"## {stage}: {panel}", "", "| Method | Dataset | AP | ROC-AUC | VUS-PR | VUS-ROC | Axis |",
                     "| --- | --- | ---: | ---: | ---: | ---: | --- |"])
        for method in NAMES:
            for domain in ("PSM", "SWAT"):
                row = lookup.get((domain, method, stage, panel))
                if row:
                    values = " | ".join(f"{row[m]:.6f}" for m in METRICS)
                    text.append(f"| {NAMES[method]} | {domain} | {values} | [{row['start']},{row['stop']}) |")
        text.append("")
    text.extend(["## Paired differences", "", "| Comparison | Dataset | Stage/panel | AP | ROC-AUC | VUS-PR | VUS-ROC |",
                 "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in differences:
        values = " | ".join(f"{row[m]:+.6f}" for m in METRICS)
        text.append(f"| {row['comparison']} | {row['domain']} | {row['stage']}/{row['panel']} | {values} |")
    text.append("")
    (dest / "TABLES.md").write_text("\n".join(text), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for arm, label in (("native", "Native continuation"), ("history", "History context")):
        curve = json.loads((run / "source" / arm / "select.json").read_text())
        for ax, key, title in zip(axes, ("copy", "E_full", "V_full"), ("Copy MSE", "Mean-output error E", "Sample variance V")):
            ax.plot([r["step"] for r in curve], [r[key] for r in curve], "o-", label=label)
            ax.set(xlabel="Source optimizer updates", ylabel=title)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(dest / "source_select.svg")
    fig.savefig(dest / "source_select.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for ax, domain in zip(axes, ("PSM", "SWAT")):
        for arm, label in (("native", "Native continuation"), ("history", "History context")):
            ap = [lookup[domain, arm, stage, "QUERY"]["AP"] for stage in ("Z", "U")]
            ax.plot((0, 1), ap, "o-", label=label)
        ax.set(xticks=(0, 1), xticklabels=("Z", "U (2000 updates)"), ylabel="AP", title=domain)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(dest / "query_Z_U.svg")
    fig.savefig(dest / "query_Z_U.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.run, args.destination)
