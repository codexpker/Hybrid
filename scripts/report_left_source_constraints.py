"""Build the LEFT mixed-target tables and update statistics from sealed receipts."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("docs/experiments/left_multisource_20261126")
DOC = Path("docs/experiments/left_source_constraints_20261126")
NAMES = {
    "source": "Frozen source LEFT",
    "random_u": "Random-init target U",
    "ordinary": "Pretrained target U",
    "replay": "Ordinary replay",
    "source_only": "Source continuation",
    "mean": "Mean-source constraint",
    "separate": "Separate source constraints",
}
RANK = ["AP", "ROC_AUC", "VUS_PR", "VUS_ROC"]


def read(path):
    return json.loads(path.read_text())


def table(frame, places=4):
    rows = [frame.columns.tolist(), ["---"] * len(frame.columns)]
    for row in frame.itertuples(index=False, name=None):
        rows.append(
            [
                f"{x:.{places}f}" if isinstance(x, (float, np.floating)) else str(x)
                for x in row
            ]
        )
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)


def main(root):
    DOC.mkdir(parents=True, exist_ok=True)
    (DOC / "traces").mkdir(exist_ok=True)
    (DOC / "review.json").write_bytes((root / "review.json").read_bytes())
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey="row", layout="constrained")
    method, operation, event, records, costs, manifests = [], [], [], [], [], {}
    for j, domain in enumerate(("PSM", "SWAT")):
        folder = root / domain
        for suffix, container in (
            ("mixed", method),
            ("operations", operation),
            ("events", event),
        ):
            container.extend(
                pd.read_csv(p / domain / (suffix + ".csv")) for p in (BASE, root)
            )
        manifests[domain] = {"gate": read(folder / "check/check.json"), "arms": {}}
        elapsed, user, system, rss = (folder / "eval-time.txt").read_text().split()
        manifests[domain]["evaluation"] = dict(
            wall_seconds=float(elapsed),
            user_cpu_seconds=float(user),
            system_cpu_seconds=float(system),
            peak_RSS_KiB=int(rss),
        )
        for arm in ("source_only", "mean", "separate"):
            receipt = read(folder / arm / "receipt.json")
            trace = (folder / arm / "steps.csv").read_bytes()
            (DOC / "traces" / f"{domain}_{arm}.csv.gz").write_bytes(
                gzip.compress(trace, mtime=0)
            )
            manifests[domain]["arms"][arm] = receipt
            banks = [
                read(folder / "banks" / f"{arm}_{role}.json")
                for role in ("CAL", "QUERY")
            ]
            manifests[domain]["arms"][arm]["banks"] = banks
            costs.append(
                dict(
                    domain=domain,
                    arm=arm,
                    train_s=receipt["seconds"],
                    bank_s=sum(e["seconds"] for e in banks),
                    peak_MiB=receipt["peak_bytes"] / 2**20,
                    windows_target=receipt["target_windows"],
                    windows_each_source=receipt["windows_per_source"],
                )
            )
            if arm == "source_only":
                continue
            data = pd.read_csv(folder / arm / "steps.csv")
            np.testing.assert_array_equal(data.step, np.arange(1, 2001))
            ratio = data.actual / data.proposed
            rec = dict(
                domain=domain,
                arm=arm,
                trigger=data.trigger.mean(),
                ratio_mean=ratio.mean(),
                ratio_min=ratio.min(),
                ratio_q05=ratio.quantile(0.05),
                near_zero=(ratio <= 0.01).sum(),
                correction_relative=(data.correction / data.proposed).mean(),
                correction_mean=data.correction.mean(),
                proposed_mean=data.proposed.mean(),
                actual_mean=data.actual.mean(),
                max_enforced_residual=data.enforced_residual.max(),
                positive_enforced=(data.enforced_residual > 0).sum(),
                nudged=(data.storage_nudges > 0).sum(),
                max_storage_error=data.storage_error.max(),
            )
            for s in ("tep", "smd"):
                rec[s + "_raw_positive"] = (data[s + "_raw"] > 0).mean()
                rec[s + "_actual_positive"] = (data[s + "_actual"] > 0).mean()
                rec[s + "_finite_rise"] = (data[s + "_change"] > 0).mean()
                rec[s + "_mean_change"] = data[s + "_change"].mean()
            records.append(rec)
            blocks = (data.step - 1) // 100
            color = "#087f83" if arm == "separate" else "#ae5818"
            for ax, series in (
                (axes[0, j], data.trigger),
                (axes[1, j], 100 * (1 - ratio)),
            ):
                points = series.groupby(blocks).mean()
                ax.plot(
                    (points.index + 1) * 100,
                    points,
                    marker=".",
                    color=color,
                    label=NAMES[arm],
                )
                ax.grid(alpha=0.22)
            axes[0, j].set_title(domain)
            axes[1, j].set_xlabel("Adam update (100-step averages)")
        axes[0, j].set_ylim(bottom=0)
        axes[1, j].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Fraction projected")
    axes[1, 0].set_ylabel("Reduction in step norm (%)")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(DOC / "updates.png", dpi=165)
    plt.close(fig)
    rank = pd.concat(method, ignore_index=True)
    ops = pd.concat(operation, ignore_index=True)
    events = pd.concat(event, ignore_index=True)
    stats = pd.DataFrame(records)
    costs = pd.DataFrame(costs)
    deltas = []
    for d, rows in rank.groupby("domain"):
        scores = rows.set_index("arm")[RANK]
        for a, b in (
            ("separate", "source"),
            ("separate", "replay"),
            ("separate", "mean"),
            ("separate", "source_only"),
            ("mean", "replay"),
            ("source_only", "source"),
        ):
            deltas.append(
                dict(
                    domain=d,
                    comparison=NAMES[a] + " − " + NAMES[b],
                    **(scores.loc[a] - scores.loc[b]),
                )
            )
    diff = pd.DataFrame(deltas)
    for name, frame in (
        ("methods", rank),
        ("contrasts", diff),
        ("operations", ops),
        ("events", events),
        ("mechanism", stats),
        ("costs", costs),
    ):
        frame.to_csv(DOC / (name + ".csv"), index=False)
    (DOC / "receipts.json").write_text(json.dumps(manifests, indent=2) + "\n")
    wide = rank.pivot(index="arm", columns="domain", values=RANK).swaplevel(axis=1)
    wide = wide.loc[list(NAMES), [(d, m) for d in ("PSM", "SWAT") for m in RANK]]
    wide.columns = [" ".join(pair).replace("_", "-") for pair in wide.columns]
    display = wide.rename(index=NAMES).rename_axis("Method").reset_index()
    opfields = [
        "domain",
        "arm",
        "stream",
        "normal_FPR",
        "Recall",
        "normal_alarm_fragments",
        "M",
        "Ntrue",
        "auxiliary_any_hit_events",
        "one_to_one_eventF1",
        "event_point_coverage",
        "long_steady_macro_coverage",
        "long_steady_point_coverage",
        "max_event_miss_points",
    ]
    op = ops[opfields].replace({"arm": NAMES})
    text = [
        "# Source-constrained mixed adaptation of LEFT",
        "",
        "## Mixed-target ranking",
        "",
        "One development seed (2025), native retrospective full-window scoring. PSM uses 65,088 points [22656,87744), SWaT 336,321 points [608535,944856). The four previously scored arms on each target are reused. Source checkpoints, scaler, score reduction, window grid and target/CAL roles follow the [base experiment](../left_multisource_20261126/RESULTS.md).",
        "",
        table(display),
        "",
        "## Matched and source-only contrasts",
        "",
        table(diff, places=6),
        "",
        "PSM showed a net AP gain over frozen transfer and over source continuation. Ordinary replay captured nearly all of that gain. Separate constraints added 0.000319 AP over replay on PSM and changed AP by −0.000221 on SWaT. On SWaT the small VUS-PR gain came with more normal alarm fragments. The two separate source inequalities removed first-order conflicts that the average allowed, with no material or consistent advantage over average-constrained or ordinary replay. The single seed and lack of a separate length-matched arm limit attribution of the small differences.",
        "",
        "## What the optimizer changed",
        "",
        table(
            stats[
                [
                    "domain",
                    "arm",
                    "trigger",
                    "ratio_mean",
                    "ratio_min",
                    "near_zero",
                    "tep_actual_positive",
                    "smd_actual_positive",
                    "tep_finite_rise",
                    "smd_finite_rise",
                ]
            ]
        ),
        "",
        "Fractions are over 2,000 endpoint updates per arm. Near-zero counts use an applied/proposed norm ratio at most 0.01. Positive dot products use the applied float32 parameter displacement; finite rises count strictly positive changes in the same source batch loss at fixed post-EMA state. See [all step summaries](mechanism.csv).",
        "",
        "![Projection frequency and retained step length](updates.png)",
        "",
        "## CAL and event costs",
        "",
        "Thresholds use each model’s own 512 mixed-CAL scores (q99, higher, strict exceedance). PSM has 512 annotated attack points in CAL, SWaT has 38. The empirical quantile gives no certified normal false-alarm budget. Confirmed alarms start on the third consecutive exceedance. M is the number of one-to-one matched events. Any-hit counts a different event statistic. Long-event columns give equal-event and pooled-point steady coverage. All times reflect retrospective localization. [Per-event results](events.csv).",
        "",
        table(op),
        "",
        "## Incremental cost",
        "",
        table(costs),
        "",
        "Independent CPU readback verified the six checkpoint identities, both legal axes, retrospective availability, AP with grouped ties, rank-sum ROC-AUC, CAL quantiles and strict/three-confirmed confusion counts. It checked displacement norms against 12 archived vector pairs and zero correction on every non-triggered step. VUS and event matching use the original metric implementations; they were not independently reimplemented. [Readback record](review.json).",
        "",
        "Train seconds are elapsed times on one A40 per job and include periodic vector snapshots. Bank seconds include checkpoint and scaler loading plus CAL/QUERY scoring and serialization. Full parameter displacement vectors were archived at steps 1, 1000 and 2000; the CSV trace records both norms, the repair, both signed source dot products and both finite loss changes at every constrained step. [Asset identities](receipts.json).",
        "",
        "Run the training and evaluation shell entry points in sequence on the existing LEFT workspace. Both use the same original source checkpoint and FIT assets as commit `c69955a3cbeb575455fc0279c8c2838ae4a46134`. The six compressed traces are in [traces](traces/). The complete checkpoints, CAL/QUERY banks, step traces and vector samples are in the experiment workspace under `results/left_source_constraints_20261126_v1`.",
        "",
    ]
    (DOC / "RESULTS.md").write_text("\n".join(text), encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
