"""Verified entity-level score archives to method panels, paired contrasts and cost totals."""

import argparse
import hashlib
import json
from math import fsum
from pathlib import Path

import numpy as np
import pandas as pd

M = ["AP", "ROC_AUC", "VUS_PR", "VUS_ROC"]
D = {"PSM": 1, "SWAT": 1, "SMD": 28}
N = {
    "frozen": "Frozen multisource LEFT",
    "continue_native": "Matched source continuation LEFT",
    "continue_simple": "Last historical block, lag persistence",
    "joint_joint": "LEFT with learned historical lag references",
    "continue_native_simple": "Matched LEFT with lag persistence",
    "joint_native": "Joint checkpoint, native score only",
    "joint_native_simple": "Joint checkpoint, native + persistence",
    "joint_relation": "Learned relation error only",
    "joint_self": "Self-lag error only",
    "joint_cross": "Ordered cross-lag error only",
    "joint_native_self": "Joint checkpoint, native + self-lag",
    "joint_native_cross": "Joint checkpoint, native + cross-lag",
    "scratch": "Target-normal LEFT from scratch",
    "target_normal": "Multisource LEFT with target-normal training",
    "level": "Level deviation (SmoothL1)",
    "timeradar_endpoint": "TimeRadar, official endpoint variance",
    "timeradar_block": "TimeRadar, archived official block readout",
    "pca": "PCA residual",
    "catch": "CATCH",
    "dada": "DADA, official frozen readout",
    "timercd": "TimeRCD, official weights, FIT-only adapter",
    "history512": "History Mean512",
}
CORE = [
    "frozen",
    "continue_native",
    "continue_simple",
    "continue_native_simple",
    "joint_joint",
]
EXT = CORE + [
    "scratch",
    "target_normal",
    "level",
    "timeradar_block",
    "timeradar_endpoint",
    "dada",
    "timercd",
    "pca",
    "catch",
]
COMP = [
    "joint_native",
    "joint_native_simple",
    "joint_native_self",
    "joint_native_cross",
    "joint_relation",
    "joint_self",
    "joint_cross",
]
PAIRS = [
    ("joint_joint", b)
    for b in (
        "frozen",
        "continue_native",
        "continue_native_simple",
        "continue_simple",
        "joint_native",
        "joint_native_self",
        "joint_native_cross",
        "scratch",
        "target_normal",
        "level",
    )
] + [
    ("joint_native", "continue_native"),
    ("joint_relation", "continue_simple"),
    ("joint_native_simple", "continue_native_simple"),
    ("joint_native_self", "joint_native"),
    ("joint_native_cross", "joint_native"),
]


def parse(folder):
    rows, costs = [], []
    for p in sorted(folder.glob("*/*/*.json")):
        record = json.loads(p.read_text())
        if p.stem.startswith("metrics_"):
            rows.extend(record["rows"])
        elif p.stem not in ("continue", "joint"):
            continue
        else:
            assert (
                not record["labels_read"]
                and record["history"] == 256
                and record["window"] == 192
            )
            digest = hashlib.sha256(p.with_suffix(".npz").read_bytes()).hexdigest()
            assert digest == record["sha256"]
            selected = folder.parent / record["domain"] / record["arm"] / "selected.pt"
            assert (
                hashlib.sha256(selected.read_bytes()).hexdigest()
                == record["checkpoint"]
            )
        costs.append(
            dict(
                domain=p.parent.parent.name,
                entity=p.parent.name,
                operation=p.stem,
                seconds=record["seconds"],
            )
        )
    return pd.DataFrame(rows), pd.DataFrame(costs)


def build(folder, dest):
    dest.mkdir(parents=True, exist_ok=True)
    frame, costs = parse(folder)
    key = ["panel", "domain", "entity", "method"]
    assert not frame.duplicated(key).any() and frame.check_ap_roc_maxerr.max() < 1e-10
    frame.sort_values(key).to_csv(dest / "public_entity_metrics.csv", index=False)
    costs.to_csv(dest / "public_costs.csv", index=False)
    axis = ["panel", "domain", "entity", "start", "stop", "points", "positives"]
    axes = frame[axis].drop_duplicates()
    assert not axes.duplicated(axis[:3]).any()
    axes.to_csv(dest / "public_axes.csv", index=False)
    macro = frame.groupby(["panel", "domain", "method"])[M].mean().reset_index()
    checks = (
        frame.groupby(["panel", "domain", "method"])[M]
        .agg(lambda values: fsum(values) / len(values))
        .reset_index()
    )
    np.testing.assert_allclose(macro[M], checks[M], rtol=0, atol=3e-15)
    macro.to_csv(dest / "public_macro.csv", index=False)
    pairs = []
    for d, n in D.items():
        sub = frame[(frame.panel == "complete") & (frame.domain == d)]
        for method in CORE + COMP + ["scratch", "target_normal", "level"]:
            assert len(sub[sub.method == method]) == n, (method, d)
        refs = sub.set_index(["entity", "method"])
        for e in sub.entity.unique():
            np.testing.assert_array_equal(
                refs.loc[(e, "continue_simple"), M].to_numpy(dtype=float),
                refs.loc[(e, "joint_simple"), M].to_numpy(dtype=float),
            )
            for a, b in PAIRS:
                delta = refs.loc[(e, a), M] - refs.loc[(e, b), M]
                pairs.append(dict(domain=d, entity=e, left=a, right=b, **dict(delta)))
    diffs = pd.DataFrame(pairs)
    diffs.to_csv(dest / "public_entity_differences.csv", index=False)
    summary = []
    for (d, a, b), group in diffs.groupby(["domain", "left", "right"], sort=False):
        for m in M:
            v = group[m]
            summary.append(
                dict(
                    domain=d,
                    left=a,
                    right=b,
                    metric=m,
                    mean=v.mean(),
                    median=v.median(),
                    wins=(v > 0).sum(),
                    ties=(v == 0).sum(),
                    losses=(v < 0).sum(),
                    entities=len(v),
                )
            )
    sm = pd.DataFrame(summary)
    sm.to_csv(dest / "public_paired_summary.csv", index=False)

    def row(values):
        return "| " + " | ".join(map(str, values)) + " |"

    def value(panel, d, k, m):
        grp = macro[(macro.panel == panel) & (macro.domain == d)]
        if k == "catch":
            x = grp[grp.method.isin(("catch_0", "catch_1", "catch_2"))][m]
            return f"{x.mean():.5f} ± {x.std(ddof=1):.5f}" if len(x) == 3 else "NA"
        x = grp[grp.method == k][m].to_list()
        return f"{x[0]:.5f}" if x else "NA"

    lines = [
        "# Public detection with historical lag references",
        "",
        "Scores cover every eligible current W192 window on its original grid. The first current window begins 384 points after each record origin, with history [128,384). A window's 192 scores become available at its end. Lag scores share one value across each of its six 32-point blocks. SMD averages entity metrics with equal weight over all 28 machines. New source training uses seed 2025.",
        "",
        "## External comparison",
        "",
        "The intersection begins at PSM 384, SWaT 495384, and SMD relative 384. Exclusive ends are PSM 87744, SWaT 944856, and SMD min(192 floor(N/192),100 floor(N/100)). [Per-entity axes and label counts](public_axes.csv)",
        "",
    ]
    for m in M:
        lines += [
            "### " + m.replace("_", "-"),
            "",
            row(["Detector", "PSM", "SWaT", "SMD28"]),
            "|---|---:|---:|---:|",
        ]
        lines += [row([N[k], *[value("external", d, k, m) for d in D]]) for k in EXT]
        lines += [""]
    lines += [
        "## Full eligible window grid",
        "",
        row(["Method", "Domain", *M]),
        "|---|---|---:|---:|---:|---:|",
    ]
    for k in CORE:
        for d in D:
            lines.append(row([N[k], d, *[value("complete", d, k, m) for m in M]]))
    lines.append("")
    lines += [
        "PSM/SWaT source checkpoints use normal TEP and SMD; the SMD fold uses TEP and HAI. Source-transfer rows share target-normal prefix statistics. Target-normal LEFT also uses these records for normal training and selection. Level deviation uses normal statistics and the current observation. Native LEFT readouts use full W192. The persistence statistic compares fixed lag coordinates with the last observed history block.",
        "",
        "TimeRadar endpoint uses past/current W100 and official frozen weights. The archived block0 readout retains its own grouped-window context. DADA uses frozen official weights and complete W100. TimeRCD uses FIT-only statistics and full windows up to 5000. CATCH uses W192 and target-normal training. Its three archived seeds give the mean and sample SD. PCA is a target-normal pointwise residual. NA indicates unavailable compatible coverage.",
        "",
        "### Literature panel",
        "",
        row(["Method and reported metric", "PSM", "SWaT", "SMD"]),
        "|---|---:|---:|---:|",
        "| CrossAD VUS-PR | .5596 | .4767 | .2344 |",
        "| CrossAD VUS-ROC | .7302 | .7865 | .8580 |",
        "| ModernTCN VUS-PR | .4668 | .0941 | .1596 |",
        "| ModernTCN VUS-ROC | .6480 | .2735 | .7707 |",
        "",
        "Author-reported values retain their published data, metrics and coverage. [CrossAD Tables 2–3](https://arxiv.org/html/2510.12489) · [ModernTCN in LEFT Table 2](https://arxiv.org/html/2602.08638v2#S4.T2)",
        "",
        "## Source update and readout comparisons",
        "",
        "Differences below use every eligible W192 point through 192 floor(N/192) on each record. The learned method and continued LEFT share source windows and optimizer budget. Same-checkpoint readout contrasts evaluate self-lag, cross-lag, native, learned and persistence components without target fitting.",
        "",
        row(["Signed difference", "Domain", *M]),
        "|---|---|---:|---:|---:|---:|",
    ]
    for a, b in PAIRS:
        for d in D:
            s = sm[(sm.left == a) & (sm.right == b) & (sm.domain == d)].set_index(
                "metric"
            )
            lines += [
                row(
                    [
                        N[a] + " minus " + N[b],
                        d,
                        *[f"{s.loc[m, 'mean']:+.5f}" for m in M],
                    ]
                )
            ]
    lines += [
        "",
        "[All absolute readouts](public_macro.csv) · [Raw-score bindings and entity metrics](public_entity_metrics.csv) · [Individual paired differences](public_entity_differences.csv) · [Medians, wins, ties and losses](public_paired_summary.csv)",
        "",
        "## History Mean512",
        "",
        "The shared axis starts at relative 512 for this comparison. The archived SWaT history cache covers only a suffix, so the comparisons below cover PSM and SMD.",
        "",
        row(["Method", "Domain", *M]),
        "|---|---|---:|---:|---:|---:|",
    ]
    for d in ("PSM", "SMD"):
        for k in ("history512", "continue_native", "joint_joint"):
            assert (
                len(
                    frame[
                        (frame.panel == "history512")
                        & (frame.domain == d)
                        & (frame.method == k)
                    ]
                )
                == D[d]
            )
            lines += [row([N[k], d, *[value("history512", d, k, m) for m in M]])]
    lines += [
        "",
        "AP is average precision. VUS uses the archived LEFT software, windowSize 100 and 250 thresholds. An independent tie-aware ranking calculation checked every AP/ROC against the archived score arrays. VUS uses one implementation with independently checked aggregates. [Device score times and CPU evaluator worker-seconds](public_costs.csv)",
    ]
    (dest / "PUBLIC_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    audit = dict(
        full_records=D,
        entity_metric_rows=len(frame),
        paired_record_contrasts=len(diffs),
        mirror_score_banks_sha256_verified=int(
            costs.operation.isin(["continue", "joint"]).sum()
        ),
        macro_max_abs=float(np.abs(macro[M].to_numpy() - checks[M].to_numpy()).max()),
        independent_AP_ROC_max_abs=float(frame.check_ap_roc_maxerr.max()),
        score_seconds=float(
            costs.loc[costs.operation.isin(["continue", "joint"]), "seconds"].sum()
        ),
        metric_worker_seconds=float(
            costs.loc[costs.operation.str.startswith("metrics_"), "seconds"].sum()
        ),
    )
    (dest / "public_review.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("banks", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    build(a.banks, a.dest)
