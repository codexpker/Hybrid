"""Draw the fixed comparisons from sealed tables, without model execution."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(workspace):
    root = workspace / "results/left_timesfm_hidden_20261126_v1"
    output = root / "summary"
    controls = pd.read_csv(output / "controlled_core.csv")
    comparisons = [("weak", "weak_control"), ("lag", "lag_amplitude"), ("lag", "synchronous")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for row, domain in enumerate(("PSM", "SWAT")):
        for col, group in enumerate(("coupled_AR", "normal_SELECT")):
            ax = axes[row, col]
            selected = controls[(controls.domain == domain) & (controls.group == group)]
            for k, arm in enumerate("ABCDE"):
                values = []
                for target, reference in comparisons:
                    item = selected[(selected.arm == arm) & (selected.target == target) & (selected.reference == reference)]
                    assert len(item) == 1
                    values.append(float(item.iloc[0].ROC_AUC))
                ax.bar(np.arange(3) + (k - 2) * .15, values, .14, label=arm)
            ax.axhline(.5, color="black", linestyle="--", linewidth=.8)
            ax.set(ylim=(0, 1), xticks=range(3),
                   xticklabels=["Weak / smooth", "Lag / amplitude", "Lag / synchronous"],
                   ylabel="Core discrimination ROC-AUC", title=f"{domain}: {group}")
            ax.legend(ncol=5, fontsize=8)
    fig.suptitle("Fixed controlled contrasts; D retains aligned history on the one-history synthetic records", fontsize=11)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"controlled_response.{ext}", dpi=170)
    plt.close(fig)

    paired = pd.read_csv(output / "paired_entities.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    for ax, role, panel, contrast in zip(axes, ("TEST", "QUERY"), ("complete", "mixed"), ("C-A", "C_U-A_U")):
        rows = paired[(paired.domain == "SMD") & (paired.role == role) &
                      (paired.panel == panel) & (paired.contrast == contrast)].sort_values("entity")
        assert len(rows) == 28
        ax.bar(np.arange(28), rows.AP * 100)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(ylabel="AP difference (percentage points)", title=f"{role}: {contrast}; one project training seed")
        ax.set_xticks(np.arange(28), [n.removeprefix("machine-") for n in rows.entity], rotation=60)
    axes[-1].set_xlabel("SMD machine; both panels retain all 28 entities on their own legal axes")
    for ext in ("png", "pdf"):
        fig.savefig(output / f"smd_paired_distribution.{ext}", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    plot(parser.parse_args().workspace)
