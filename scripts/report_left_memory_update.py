"""Assemble paired contamination results from distributions of sealed per-point scores."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DOC = Path("docs/experiments/left_memory_exposure_20261126")
PREVIOUS = Path("docs/experiments/left_source_constraints_20261126")
METRICS = ["AP", "ROC_AUC", "VUS_PR", "VUS_ROC"]
METHODS = ("ordinary", "replay", "restricted")
NAMES = {
    "source": "Frozen source LEFT",
    "random_u": "Random-init target U",
    "source_only": "Source continuation",
    "ordinary": "Target U",
    "replay": "Source replay",
    "restricted": "Replay with source-only EMA",
}
OPS = [
    "normal_FPR",
    "Recall",
    "normal_alarm_fragments",
    "M",
    "auxiliary_any_hit_events",
    "one_to_one_eventF1",
    "event_macro_coverage",
    "event_point_coverage",
    "long_steady_macro_coverage",
    "long_steady_point_coverage",
    "max_event_miss_points",
]


def markdown(df, digits=4):
    rows = [df.columns, ["---"] * len(df.columns)]
    for row in df.itertuples(index=False, name=None):
        rows.append(
            [
                "NA"
                if pd.isna(x)
                else (
                    f"{x:.{digits}f}" if isinstance(x, (float, np.floating)) else str(x)
                )
                for x in row
            ]
        )
    return "\n".join("| " + " | ".join(str(x) for x in row) + " |" for row in rows)


def load_json(p):
    return json.loads(p.read_text())


def group_names(arm):
    if arm in ("ordinary", "replay"):
        return arm, "A"
    if arm in NAMES:
        return arm, "ref"
    return arm.rsplit("_", 1)


def joined(root, file, oldfile):
    old = pd.read_csv(PREVIOUS / oldfile)
    old = old[old.arm.isin(["ordinary", "replay", "source", "random_u", "source_only"])]
    df = pd.concat(
        [old] + [pd.read_csv(root / d / file) for d in ("PSM", "SWAT")],
        ignore_index=True,
    )
    df[["method", "condition"]] = [list(group_names(a)) for a in df.arm]
    return df


def comparisons(df, columns):
    result = []
    for d, t in df.groupby("domain"):
        a = t.set_index(["method", "condition"])[columns]

        def record(name, x):
            result.append(dict(domain=d, comparison=name, **x.to_dict()))

        for m in METHODS:
            for x, y in (("B", "C"), ("B", "A"), ("C", "A")):
                record(f"{m} {x}-{y}", a.loc[m, x] - a.loc[m, y])
        for m, n in (("restricted", "replay"), ("replay", "ordinary")):
            for k in "ABC":
                record(f"{m}-{n} {k}", a.loc[m, k] - a.loc[n, k])
            record(
                f"{m}-{n} interaction",
                (a.loc[m, "B"] - a.loc[m, "C"]) - (a.loc[n, "B"] - a.loc[n, "C"]),
            )
    return pd.DataFrame(result)


def figure(mech):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), layout="constrained")
    for ix, d in enumerate(("PSM", "SWAT")):
        a = mech[mech.domain == d].sort_values(["condition", "method"])
        names = [
            {"ordinary": "U", "replay": "R", "restricted": "S"}[m] + "-" + k
            for m, k in zip(a.method, a.condition)
        ]
        x = np.arange(len(a))
        top, bottom = axes[:, ix]
        top.bar(
            x - 0.18,
            a.native_target + a.native_source,
            0.35,
            label="Native eligibility",
            color="#526c89",
        )
        top.bar(
            x + 0.18,
            a.written_target + a.written_source,
            0.35,
            label="Admitted for EMA",
            color="#da8836",
        )
        top.set_ylabel("Window presentations")
        if (
            not a[
                ["native_target", "native_source", "written_target", "written_source"]
            ]
            .to_numpy()
            .any()
        ):
            top.text(
                0.5,
                0.55,
                "0 eligible windows in all seven runs",
                ha="center",
                transform=top.transAxes,
            )
            top.set_ylim(0, 1)
        top.set_title(d)
        bottom.bar(
            x - 0.18,
            a.t_ema_total + a.f_ema_total,
            0.35,
            label="Explicit EMA",
            color="#da8836",
        )
        bottom.bar(
            x + 0.18,
            a.t_adam_total + a.f_adam_total,
            0.35,
            label="Adam",
            color="#237e79",
        )
        bottom.set_ylabel("Cumulative prototype movement")
        for ax in (top, bottom):
            ax.set_xticks(x, names)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.16)
    fig.supxlabel("U = target U     R = replay     S = replay with source-only EMA")
    fig.savefig(DOC / "prototype_updates.png", dpi=170)
    plt.close(fig)


def main(root):
    DOC.mkdir(parents=True, exist_ok=True)
    traces = DOC / "traces"
    traces.mkdir(exist_ok=True)
    shutil.copyfile(root / "review.json", DOC / "review.json")
    ranks = joined(root, "mixed.csv", "methods.csv")
    ops = joined(root, "operations.csv", "operations.csv")
    events = joined(root, "events.csv", "events.csv")
    ranks.to_csv(DOC / "methods.csv", index=False)
    ops.to_csv(DOC / "operations.csv", index=False)
    events.to_csv(DOC / "events.csv", index=False)
    diffs = comparisons(ranks, METRICS)
    diffs.to_csv(DOC / "contamination.csv", index=False)
    opdiff = pd.concat(
        [comparisons(o, OPS).assign(stream=s) for s, o in ops.groupby("stream")]
    )
    opdiff.to_csv(DOC / "operating_contrasts.csv", index=False)
    mech = pd.concat([pd.read_csv(root / d / "mechanism.csv") for d in ("PSM", "SWAT")])
    comp = pd.concat(
        [pd.read_csv(root / d / "composition.csv") for d in ("PSM", "SWAT")]
    )
    mech.to_csv(DOC / "mechanism.csv", index=False)
    comp.to_csv(DOC / "selection_composition.csv", index=False)
    figure(mech)
    receipts = {}
    costs = []
    for d in ("PSM", "SWAT"):
        folder = root / d
        receipts[d] = dict(
            qualification=load_json(folder / "plan/audit.json"),
            check=load_json(folder / "check/check.json"),
            arms={},
        )
        for mode in ("bank", "metrics"):
            receipts[d][mode + "_time_elapsed_user_sys_maxrss"] = (
                (folder / ("time_" + mode + ".txt")).read_text().strip()
            )
        for r in mech[mech.domain == d].itertuples():
            receipt = load_json(folder / r.arm / "receipt.json")
            trace = folder / r.arm / "trace.npz"
            assert (
                hashlib.sha256(trace.read_bytes()).hexdigest()
                == receipt["trace_sha256"]
            )
            shutil.copyfile(trace, traces / f"{d}_{r.arm}.npz")
            banks = [
                load_json(folder / "banks" / f"{r.arm}_{role}.json")
                for role in ("CAL", "QUERY")
            ]
            receipt["banks"] = banks
            receipts[d]["arms"][r.arm] = receipt
            costs.append(
                dict(
                    domain=d,
                    arm=r.arm,
                    train_s=receipt["seconds"],
                    bank_s=sum(b["seconds"] for b in banks),
                    peak_MiB=receipt["peak_bytes"] / 2**20,
                )
            )
    (DOC / "receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    costs = pd.DataFrame(costs)
    costs.to_csv(DOC / "costs.csv", index=False)
    order = [("source", "ref"), ("random_u", "ref"), ("source_only", "ref")] + [
        (m, k) for k in "ABC" for m in METHODS
    ]
    wide = ranks.pivot(
        index=["method", "condition"], columns="domain", values=METRICS
    ).swaplevel(axis=1)
    wide = wide.loc[order, [(d, m) for d in ("PSM", "SWAT") for m in METRICS]]
    wide.columns = [d + " " + m.replace("_", "-") for d, m in wide.columns]
    wide = wide.reset_index().replace({"method": NAMES})
    panel = diffs[
        diffs.comparison.str.endswith(
            ("B-C", "interaction", "-replay A", "-replay B", "-ordinary B")
        )
    ]
    s = ops[ops.stream == "confirmed3"]
    exposure = comp.query("stage == 'presented' and group == 'all_target'").copy()
    exposure["attack_point_fraction_all_slots"] = (
        exposure.attack_point_presentations / (64000 * 192)
    )
    text = [
        "# Does restricting explicit prototype writes protect mixed-target adaptation?",
        "",
        "## Absolute detection on the fixed target suffix",
        "",
        "All rows use native LEFT retrospective W192 scores and the same points within each task. PSM covers [22656,87744), SWaT [608535,944856). The official target normal prefix supplies the scaler. Fourteen new seed-2025 fits start from the archived width-specific source models. Target U and ordinary replay under A, and the three reference rows, reuse earlier banks. [Complete public benchmark](../left_multisource_20261126/RESULTS.md).",
        "",
        markdown(wide),
        "",
        "## Controlled exposure",
        "",
        "A uses natural U-FIT. B and C replace the same 3,200 of 32,000 shared target slots, respectively with wholly attack-labeled or wholly normal-labeled W192 windows from U-FIT. This replaces 5% of all 64,000 window presentations. Source and second target streams are untouched. Data construction used public labels; learners consumed numeric starts and values. Eligibility does not match operating conditions between donors.",
        "",
        markdown(panel, 6),
        "",
        "On PSM, ordinary replay reduced the AP loss under B versus C from 0.05103 to 0.02062. Its B AP also exceeded target-only U by 0.04891, although both ROC measures fell under B and showed no analogous protection. On SWaT the absolute B AP difference still favored replay by 0.00219, but replay suffered a larger B−C AP decline than target-only U. Restricting EMA writes added no demonstrable protection in either domain.",
        "",
        "PSM has 649 eligible abnormal starts across five events; 3,200 draws used 645 distinct starts. SWaT has 3,468 eligible starts across nine events; the same number of draws used 2,094 distinct starts. The C draws used 2,826 and 3,126 distinct nominal starts, respectively. A repeated slot or donor is a presentation, not another event. The point fractions below use all 64,000 windows, counting source slots under their normal training role.",
        "",
        markdown(
            exposure[
                [
                    "domain",
                    "arm",
                    "presentations",
                    "fully_anomalous",
                    "partially_anomalous",
                    "overlapping_U_events",
                    "attack_point_fraction_all_slots",
                ]
            ]
        ),
        "",
        "Interaction is the difference between the two B−C responses in the order shown. A positive interaction can coexist with a lower absolute B score. [All direct contrasts, B−A and C−A](contamination.csv).",
        "",
        "## Source and target windows admitted for explicit writes",
        "",
        "Across the 28,000 newly observed updates, neither the ordinary full-batch filter nor the restricted write rule admitted a window. The explicit EMA displacement was zero in both prototype banks. The banks moved through Adam in every run. Thus the B/C comparison varies the data reaching the gradient update, but supplies no instance of a harmful native write that the restriction prevented. Small numerical ranking differences between the two replay implementations cannot establish a benefit of changing write permissions.",
        "",
        "The native filter requires low reconstruction error and low time/frequency assignment JS divergence (within the native 0.30 quantiles) and both native confidence tests (at least 0.85). Restricted replay computes the two effective cutoffs on slots 16–31 and admits only qualified source slots. Native-filter columns describe the full-batch rule on the same forward, including its shadow value for restricted replay. Adam still sees all samples. EMA momentum is 0.99. Each active step calls one hard update per bank. Archived ordinary A and replay A have no per-window selection traces; their counts are unavailable.",
        "",
        markdown(
            mech[
                [
                    "domain",
                    "arm",
                    "native_target",
                    "native_source",
                    "written_target",
                    "written_source",
                    "ema_write_steps",
                    "no_write_steps",
                    "t_ema_total",
                    "f_ema_total",
                    "t_adam_total",
                    "f_adam_total",
                ]
            ]
        ),
        "",
        "![Native eligibility, explicit writes and prototype movement](prototype_updates.png)",
        "",
        "Movement sums the Frobenius norm of consecutive substep displacements, separately for each bank. The [full table](mechanism.csv) also gives start-to-end drift and actual labeled point exposure. [Window and event composition](selection_composition.csv) distinguishes full, partial and zero overlap, distinct starts and distinct observed event intervals. Summaries were joined only after sealing all 28 CAL/QUERY banks.",
        "",
        "## Three-consecutive-point operating results",
        "",
        markdown(
            s[
                [
                    "domain",
                    "arm",
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
        "The model-specific thresholds use the same 512 CAL points, higher q99 and strict exceedance. PSM CAL contains 512 labeled attack points and SWaT CAL 38. These historical calibration sets do not certify a normal false-alarm budget. [Strict and confirmed results](operations.csv), [operating interactions](operating_contrasts.csv), [all events](events.csv).",
        "",
        "## Cost and replication",
        "",
        markdown(costs),
        "",
        "Post-seal CPU readback checked slot counts, the untouched target stream, all donor footprints, checkpoint and trace identities, raw-bank AP/ROC, fixed CAL thresholds and strict/three-confirmed confusion counts. VUS and event matching use the previously fixed implementations. [Readback values](review.json).",
        "",
        "Train and bank seconds are elapsed on one A40 per job. The two-domain run uses one development seed. Archive paths under `results/left_memory_exposure_20261126_v1` contain the numeric plans, 14 endpoints and all score banks. [Full-bank identities and donor qualification](receipts.json), [per-step eligibility and prototype displacement](traces/).",
        "",
    ]
    (DOC / "RESULTS.md").write_text("\n".join(text), encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
