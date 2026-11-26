"""Produce tables and representative U-FIT windows from sealed adaptation outputs."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from report_left_memory_update import METRICS, OPS, markdown, load_json

DOC = Path("docs/experiments/left_local_reliability_20261126")
OLD = Path("docs/experiments/left_memory_exposure_20261126")
DOMAINS = ("PSM", "SWAT")
METHODS = ("ordinary", "replay", "window", "local")
NAMES = {
    "source": "Frozen multisource LEFT",
    "source_only": "Source continuation",
    "random_u": "Random-init target U",
    "ordinary": "Target U",
    "replay": "Source replay",
    "window": "Window-weighted replay",
    "local": "Local-weighted replay",
}


def assemble(root, new_file, old_file):
    old = pd.read_csv(OLD / old_file)
    old = old[old.method.isin(NAMES)]
    parts = [old]
    for d in DOMAINS:
        df = pd.read_csv(root / d / new_file)
        df[["method", "condition"]] = df.arm.str.split("_", expand=True)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def contrasts(df, cols):
    rows = []
    for d, part in df.groupby("domain"):
        a = part.set_index(["method", "condition"])[cols]

        def record(name, v):
            rows.append(dict(domain=d, comparison=name, **v.to_dict()))

        for m in METHODS:
            record(f"{m} A - Z", a.loc[m, "A"] - a.loc["source", "ref"])
            for x, y in (("B", "C"), ("B", "A"), ("C", "A")):
                record(f"{m} {x}-{y}", a.loc[m, x] - a.loc[m, y])
        for m, n in (("window", "replay"), ("local", "replay"), ("local", "window")):
            for k in "ABC":
                record(f"{m}-{n} {k}", a.loc[m, k] - a.loc[n, k])
            record(
                f"{m}-{n} interaction",
                (a.loc[m, "B"] - a.loc[m, "C"]) - (a.loc[n, "B"] - a.loc[n, "C"]),
            )
    return pd.DataFrame(rows)


def figure(root):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), layout="constrained", sharex=True)
    for ax, d in zip(axes, DOMAINS):
        with np.load(root / d / "example.npz") as ex:
            x = np.arange(192)
            ax.plot(
                x,
                ex["point_weight"],
                lw=1.5,
                color="#087f83",
                label="Local coefficient",
            )
            ax.axhline(
                float(ex["window_weight"]),
                ls="--",
                color="#b05e24",
                label="Window coefficient",
            )
            ax.axhline(1, color="#333333", lw=0.6, ls=":")
            shade = ex["label"].astype(bool)
            edges = np.diff(np.r_[False, shade, False].astype(int))
            for j, (lo, hi) in enumerate(
                zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            ):
                ax.axvspan(
                    lo - 0.5,
                    hi - 0.5,
                    color="#b85068",
                    alpha=0.14,
                    label="Labeled attack" if j == 0 else None,
                )
            ax.set_title(
                f"{d}   |   U-FIT start {ex['index'][0]}   |   {shade.sum()}/192 attack-labeled points"
            )
            ax.set_ylabel("Loss coefficient")
            ax.set_xlim(-0.5, 191.5)
            ax.legend(loc="best", ncol=3, fontsize=8)
            ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Position in the target window")
    fig.savefig(DOC / "example_weights.png", dpi=165)
    plt.close(fig)


def report(root):
    DOC.mkdir(parents=True, exist_ok=True)
    ranks = assemble(root, "mixed.csv", "methods.csv")
    ops = assemble(root, "operations.csv", "operations.csv")
    events = assemble(root, "events.csv", "events.csv")
    delta = contrasts(ranks, METRICS)
    opdelta = pd.concat(
        [contrasts(x, OPS).assign(stream=k) for k, x in ops.groupby("stream")]
    )
    mechanism = pd.concat(
        [pd.read_csv(root / d / "mechanism.csv") for d in DOMAINS], ignore_index=True
    )
    for name, df in (
        ("methods", ranks),
        ("operations", ops),
        ("events", events),
        ("contrasts", delta),
        ("operating_contrasts", opdelta),
        ("weights_by_label", mechanism),
    ):
        df.to_csv(DOC / (name + ".csv"), index=False)
    wide = ranks.pivot(
        index=["method", "condition"], columns="domain", values=METRICS
    ).swaplevel(axis=1)
    order = [("source", "ref"), ("random_u", "ref"), ("source_only", "ref")] + [
        (m, k) for k in "ABC" for m in METHODS
    ]
    wide = wide.loc[order, [(d, m) for d in DOMAINS for m in METRICS]]
    wide.columns = [d + " " + m.replace("_", "-") for d, m in wide.columns]
    wide = wide.reset_index().replace({"method": NAMES})
    costs = []
    receipts = {}
    for d in DOMAINS:
        folder = root / d
        teacher = load_json(folder / "weights/receipt.json")
        receipts[d] = dict(
            teacher=teacher, check=load_json(folder / "prepare/check.json"), arms={}
        )
        for mode in ("bank", "metrics"):
            receipts[d][mode + "_elapsed_user_sys_rss"] = (
                (folder / ("time_" + mode + ".txt")).read_text().strip()
            )
        costs.append(
            dict(
                domain=d,
                stage="Frozen scoring and exposure ranks",
                seconds=teacher["seconds"],
            )
        )
        for m in ("window", "local"):
            for k in "ABC":
                arm = f"{m}_{k}"
                payload = load_json(folder / arm / "receipt.json")
                banks = [
                    load_json(folder / "banks" / f"{arm}_{r}.json")
                    for r in ("CAL", "QUERY")
                ]
                payload["banks"] = banks
                receipts[d]["arms"][arm] = payload
                costs.extend(
                    [
                        dict(
                            domain=d,
                            stage=arm + " training",
                            seconds=payload["seconds"],
                        ),
                        dict(
                            domain=d,
                            stage=arm + " readout",
                            seconds=sum(b["seconds"] for b in banks),
                        ),
                    ]
                )
    pd.DataFrame(costs).to_csv(DOC / "costs.csv", index=False)
    (DOC / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    shutil.copyfile(root / "review.json", DOC / "review.json")
    figure(root)
    core = delta[
        delta.comparison.str.endswith(
            (
                "B-C",
                "interaction",
                "-window A",
                "-replay A",
                "-window B",
                "-replay B",
                "local A - Z",
            )
        )
    ]
    op = ops[
        (ops.stream == "confirmed3") & ops.method.isin(("replay", "window", "local"))
    ]
    classes = (
        mechanism.query("category!='all' and position=='all'")
        .pivot(
            index=["domain", "condition"], columns="category", values="window_weight"
        )
        .reset_index()
    )
    wcols = [
        "domain",
        "condition",
        "position",
        "window_exposures",
        "positions",
        "window_weight",
        "local_weight",
        "local_minus_window",
        "window_fraction_local_above_uniform",
    ]
    info = [
        "# Local versus window-level reliability in multisource LEFT replay",
        "",
        "## Matched mixed-target detection",
        "",
        "One development seed, 2,000 Adam steps, fixed endpoint and unchanged native full-window scores. PSM uses [22656,87744) and SWaT [608535,944856). Baseline banks and the [public detection table](../left_multisource_20261126/RESULTS.md) are reused. U-FIT weights use 6,144,000 time-position presentations in each condition, including repeated windows; the estimator receives no target annotations, CAL or QUERY scores.",
        "",
        markdown(wide),
        "",
        "At fixed target mass, local weighting improved all four ranking measures over window weighting on naturally mixed PSM. At the abnormal-replacement endpoint it gave back PR performance relative to window weighting, while improving ROC-AUC and VUS-ROC. SWaT's natural AP difference was 0.000005, and neither weighted arm recovered frozen-source AP. Both domains assigned more weight to the nominal part of mixed windows. The detection comparisons support retaining the allocation observation and the PSM result, without promoting the local scheme as a demonstrated improvement over window weighting across conditions.",
        "",
        "## Exposure responses",
        "",
        "B and C replace the same 3,200 shared target slots with fully anomalous or fully nominal donor windows from U-FIT. This is 5% of all training-window presentations. The interaction subtracts the comparator's B−C response from the named method's response. Positive ranking differences favor the first method. [All A/B/C and frozen-source contrasts](contrasts.csv).",
        "",
        markdown(core, 6),
        "",
        "## Allocation of the target weight budget",
        "",
        "The two allocations have the same mean coefficient in each window. The local arm varies coefficients only on the original 192-point grid. Coarse scales and the spectral cycle receive the window mean in both arms. Coefficients are not probabilities of health or measurements of gradient contribution.",
        "",
        "Average target coefficients by window class (shared by both arms). A coefficient of one is the target minibatch average.",
        "",
        markdown(classes),
        "",
        "Position-level coefficients pooled over all target exposures, grouped by the public point label.",
        "",
        markdown(mechanism.query("category=='all' and position!='all'")[wcols[:-1]]),
        "",
        "Within partially anomalous target windows:",
        "",
        markdown(mechanism.query("category=='partial' and position!='all'")[wcols]),
        "",
        "The last column is the fraction of partially anomalous window presentations whose positions of that label have a higher mean local coefficient than the uniform window coefficient (numerical tolerance 1e-7). Repeated windows count. [Full-window classes and pooled normal/abnormal positions](weights_by_label.csv).",
        "",
        "![Local and window weights on two partially anomalous U-FIT windows](example_weights.png)",
        "",
        "Each panel uses natural A and the first presented partial window closest to its domain's median number of anomalous positions among partial exposures. Public labels enter this illustration and the allocation analysis after sealing all detection banks.",
        "",
        "## Operating costs at historical mixed CAL",
        "",
        markdown(
            op[
                [
                    "domain",
                    "method",
                    "condition",
                    "normal_FPR",
                    "Recall",
                    "normal_alarm_fragments",
                    "M",
                    "Ntrue",
                    "auxiliary_any_hit_events",
                    "one_to_one_eventF1",
                    "long_steady_macro_coverage",
                    "long_steady_point_coverage",
                    "max_event_miss_points",
                ]
            ]
        ),
        "",
        "Each method uses its own strict-exceedance threshold from the same 512 CAL points (higher q99). All 512 PSM CAL labels and 38 SWaT CAL labels are attacks. These retrospective operating points do not certify a normal false-positive budget. [All strict and confirmed results](operations.csv), [operating contrasts](operating_contrasts.csv), [event details](events.csv).",
        "",
        "## Verification and compute",
        "",
        "Checks cover native all-one loss and parameter gradients, nonuniform local versus window gradients, per-window and batch budgets, tied score ranks checked by independent strict/inclusive counts, checkpoint identities, AP/ROC and CAL confusion counts. VUS and event matching use the established implementations. Per-stage elapsed seconds and model fingerprints are archived. [Costs](costs.csv), [training and score receipts](receipts.json), [CPU readback](review.json).",
        "",
    ]
    (DOC / "RESULTS.md").write_text("\n".join(info), encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    report(p.parse_args().root)
