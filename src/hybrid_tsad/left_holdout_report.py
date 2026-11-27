"""Describe the fixed development panels and gate expansion to all of SMD28."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

from .left_holdout_training import OUT
from .left_timesfm_evaluation import METRICS
from .left_swat_author import dump, read
from .trajectory_evaluation import save_rows

DEVELOPMENT = ("PSM", "SWAT")
MIN_AP = 0.005  # absolute AP, 0.5 percentage points
METHODS = {
    "A": "Holdout LEFT",
    "B": "Holdout LEFT + frozen pretrained TimesFM",
    "C": "Holdout LEFT + frozen random TimesFM",
    "native": "Matched-budget native LEFT normal training",
    "e59_native_Z": "Native LEFT matched source continuation",
    "e59_history_Z": "Unmasked LEFT + frozen pretrained TimesFM",
    "frozen": "Earlier multisource LEFT",
    "scratch": "Native LEFT trained on target normal data",
    "target_normal": "Earlier multisource LEFT + target normal training",
    "level": "Normal-level deviation",
    "catch": "CATCH (three archived seeds)",
    "pca": "PCA residual",
    "timeradar_block": "TimeRadar (archived block score)",
    "timeradar_endpoint": "TimeRadar endpoint V",
    "dada": "DADA",
    "timercd": "TimeRCD (FIT-only statistics)",
}


def paired(rows):
    by_key = {
        (r["domain"], r["entity"], r["stage"], r["panel"], r["method"]): r for r in rows
    }
    output = []
    contrasts = (
        ("B", "A", "complete"),
        ("B", "C", "complete"),
        ("B", "native", "complete"),
        ("B", "e59_native_Z", "external"),
        ("A", "e59_native_Z", "external"),
        ("B", "e59_history_Z", "external"),
        ("B", "B_nonoverlap", "nonoverlap"),
    )
    for (domain, entity, stage, panel, method), row in by_key.items():
        for first, other, use in contrasts:
            if (method, panel) != (first, use):
                continue
            ref = by_key.get((domain, entity, stage, panel, other))
            if ref is None:
                continue
            assert (row["start"], row["stop"]) == (ref["start"], ref["stop"])
            for metric in METRICS:
                a, b = row[metric], ref[metric]
                output.append(
                    dict(
                        domain=domain,
                        entity=entity,
                        stage=stage,
                        panel=panel,
                        contrast=f"{first}-{other}",
                        metric=metric,
                        delta=None if None in (a, b) else a - b,
                    )
                )
    return output


def gate(root):
    output = root / OUT
    rows = [
        row
        for d in DEVELOPMENT
        for row in read(output / "public" / d / d / "METRICS.json")
    ]
    comparisons = {
        (r["domain"], r["stage"], r["contrast"]): r["delta"]
        for r in paired(rows)
        if r["metric"] == "AP"
        and r["panel"] == "complete"
        and r["contrast"] in ("B-A", "B-C")
    }
    assert len(comparisons) == 8
    support = []
    for d in DEVELOPMENT:
        floor = {
            s: min(comparisons[d, s, c] for c in ("B-A", "B-C")) for s in ("Z", "N")
        }
        if min(floor.values()) > 0 and max(floor.values()) >= MIN_AP:
            support.append(d)
    record = dict(
        expand_to_all_SMD28=bool(support),
        supporting_development_domains=support,
        rule="Within at least one development domain, B AP exceeds both A and C on Z and N; the smaller of the two gains reaches 0.005 AP in at least one stage.",
        interpretation="A development trigger for a full-domain check; not an uncertainty estimate or a mechanism claim.",
        contrasts=[
            dict(domain=d, stage=s, contrast=c, AP=v)
            for (d, s, c), v in comparisons.items()
        ],
    )
    dump(output / "EXPANSION.json", record)
    return record


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[k] for k in keys)].append(row)
    return result


def summarize(rows):
    values = []
    for (domain, stage, panel, method), records in grouped(
        rows, ("domain", "stage", "panel", "method")
    ).items():
        result = dict(
            domain=domain,
            stage=stage,
            panel=panel,
            method=method,
            entities=len(records),
        )
        for metric in METRICS:
            s = [r[metric] for r in records if r[metric] is not None]
            result.update({metric: mean(s) if s else None, metric + "_valid": len(s)})
        values.append(result)
    for (domain, stage, panel), records in grouped(
        values.copy(), ("domain", "stage", "panel")
    ).items():
        seeds = [r for r in records if r["method"].startswith("catch_")]
        if not seeds:
            continue
        assert len(seeds) == 3
        new = dict(seeds[0], method="catch", archived_seeds=3)
        for m in METRICS:
            new[m] = mean(r[m] for r in seeds)
            new[m + "_seed_sd"] = stdev(r[m] for r in seeds)
        values.append(new)
    return values


def format_metric(row, metric):
    if row is None or row[metric] is None:
        return "NA"
    display = f"{row[metric]:.6f}"
    if metric + "_seed_sd" in row:
        display += f" ± {row[metric + '_seed_sd']:.6f}"
    return display


def learning_and_cost(out, domains, files):
    learn, cost = [], []
    for domain in domains:
        folder = out / domain / "Z"
        z = read(folder / "COMPLETE.json")
        feature = z["feature_intervals"]
        cost.append(
            dict(
                domain=domain,
                object="all",
                stage="joint source train and SELECT",
                seconds=z["seconds"],
                native_forwards=z["native_forwards"],
                history_forwards=sum(r["calls"] for r in feature),
                history_seconds=sum(r["seconds"] for r in feature),
                peak_bytes=z["peak_bytes"],
            )
        )
        check = read(folder / "CHECKS.json")
        for arm in ("A", "B", "C"):
            source = read(folder / arm / "receipt.json")
            learn.append(
                dict(
                    domain=domain,
                    entity="all",
                    stage="Z",
                    arm=arm,
                    selected=source["selected"],
                    updates=source["updates"],
                    normal_select_objective=source["normal_select_objective"],
                    checkpoint=source["sha256"],
                )
            )
        cost.append(
            dict(
                domain=domain,
                object="fixed normal-source panel",
                stage="isolation and archived C bypass",
                seconds=check["seconds"],
                native_forwards=sum(
                    r["native_forwards"] for r in check["arms"].values()
                )
                + check["prior_C"]["native_forwards"],
            )
        )
    for file in files:
        domain, entity = file.parent.parent.name, file.parent.name
        folder = out / domain / "N" / entity
        n = read(folder / "COMPLETE.json")
        feature = n["history_feature_cost"]
        if isinstance(feature, dict):
            feature = [feature]
        cost.append(
            dict(
                domain=domain,
                object=entity,
                stage="joint normal train and SELECT",
                seconds=n["seconds"],
                native_forwards=n["native_forwards"],
                history_forwards=sum(r["calls"] for r in feature),
                history_seconds=sum(r["seconds"] for r in feature),
                peak_bytes=n["peak_bytes"],
            )
        )
        for arm in ("A", "B", "C", "native"):
            selected = read(folder / arm / "receipt.json")
            learn.append(
                dict(
                    domain=domain,
                    entity=entity,
                    stage="N",
                    arm=arm,
                    selected=selected["selected"],
                    updates=selected["selected_updates"],
                    normal_select_objective=selected["normal_select_objective"],
                    checkpoint=selected["sha256"],
                )
            )
        inference = read(file.parent / "INPUTS.json")["historical_features"]
        cost.append(
            dict(
                domain=domain,
                object=entity,
                stage="TEST historical feature extraction",
                seconds=sum(r["seconds"] for r in inference.values()),
                history_forwards=sum(r["calls"] for r in inference.values()),
                archival_cache_reuse=all(r.get("path") for r in inference.values()),
            )
        )
        for path in file.parent.glob("?_*.json"):
            row = read(path)
            cost.append(
                dict(
                    domain=domain,
                    object=path.stem,
                    stage="TEST frozen scoring",
                    seconds=row["seconds"],
                    native_forwards=row["native_forwards"],
                    window_forwards=row["window_forwards"],
                )
            )
    for path in sorted((out / "controlled").glob("*/*_*.json")):
        row = read(path)
        cost.append(
            dict(
                domain=path.parent.name,
                object=path.stem,
                stage="controlled scoring",
                seconds=row["seconds"],
                native_forwards=row["native_forwards"],
                window_forwards=row["window_forwards"],
            )
        )
    return learn, cost


def report(root):
    out = root / OUT
    decision = gate(root)
    files = [out / "public" / d / d / "METRICS.json" for d in DEVELOPMENT]
    if decision["expand_to_all_SMD28"]:
        from .left_history_public import members

        files.extend(
            out / "public" / "SMD" / e / "METRICS.json" for e in members(root, "SMD")
        )
    rows = [row for file in files for row in read(file)]
    summary = summarize(rows)
    pairs = paired(rows)
    paired_summary = []
    for (d, s, p, c, m), rs in grouped(
        pairs, ("domain", "stage", "panel", "contrast", "metric")
    ).items():
        ds = [x["delta"] for x in rs if x["delta"] is not None]
        paired_summary.append(
            dict(
                domain=d,
                stage=s,
                panel=p,
                contrast=c,
                metric=m,
                n=len(ds),
                mean=mean(ds) if ds else None,
                median=median(ds) if ds else None,
                wins=sum(v > 0 for v in ds),
                ties=sum(v == 0 for v in ds),
                losses=sum(v < 0 for v in ds),
            )
        )
    dest = out / "summary"
    dest.mkdir(exist_ok=True)
    domains = tuple(dict.fromkeys(r["domain"] for r in rows))
    learning, cost = learning_and_cost(out, domains, files)
    for name, data in (
        ("metrics", rows),
        ("macro", summary),
        ("paired_entities", pairs),
        ("paired_summary", paired_summary),
        ("learning", learning),
        ("costs", cost),
    ):
        columns = list(dict.fromkeys(k for r in data for k in r))
        save_rows(dest / f"{name}.csv", [{k: r.get(k) for k in columns} for r in data])
    text = [
        "# Holdout LEFT public evaluation",
        "",
        "All scores are available at the end of the current W192 window. "
        "The Z and N stages use the same legal TEST timestamps within each panel. "
        "External axes can trim tails. SMD, when eligible, averages entity metrics equally. "
        "The project uses one training seed. Only CATCH uses three archived seeds.",
        "",
        "Operation points are NA. This full-score comparison does not replace the old CAL panel.",
        "",
    ]
    table = {(r["domain"], r["stage"], r["panel"], r["method"]): r for r in summary}
    labels = {"Z": "Frozen source models (Z)", "N": "Target normal training (N)"}
    for stage in ("Z", "N"):
        for panel, title in (
            ("complete", "complete internal axis"),
            ("external", "external common axis"),
        ):
            text += [f"## {labels[stage]}, {title}", ""]
            for m in METRICS:
                text += [
                    f"### {m}",
                    "",
                    "| Method | " + " | ".join(domains) + " |",
                    "|---|" + "---:|" * len(domains),
                ]
                for key, method in METHODS.items():
                    rr = [table.get((d, stage, panel, key)) for d in domains]
                    if any(r is not None for r in rr):
                        text.append(
                            "| "
                            + method
                            + " | "
                            + " | ".join(format_metric(r, m) for r in rr)
                            + " |"
                        )
                text.append("")
    text += [
        "## Matched differences",
        "",
        "| Domain | Stage | Axis | Contrast | AP | ROC-AUC | VUS-PR | VUS-ROC |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for (d, s, p, c), group in grouped(
        paired_summary, ("domain", "stage", "panel", "contrast")
    ).items():
        mapping = {r["metric"]: r["mean"] for r in group}
        text.append(
            f"| {d} | {s} | {p} | {c} | "
            + " | ".join(
                "NA" if mapping[m] is None else f"{mapping[m]:+.6f}" for m in METRICS
            )
            + " |"
        )
    text += [
        "",
        "## Controlled core ROC-AUC",
        "",
        "| Domain | Background | Stage | Contrast | Holdout | Pretrained history | Random history |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    with (out / "controlled" / "comparisons.csv").open(newline="") as handle:
        control = [
            r
            for r in csv.DictReader(handle)
            if r["region"] == "core"
            and (r["edit"], r["reference"])
            in (
                ("weak", "weak_control"),
                ("lag", "lag_amplitude"),
                ("lag", "synchronous"),
            )
        ]
    for (d, g, s, e, r), records in grouped(
        control, ("domain", "group", "stage", "edit", "reference")
    ).items():
        values = {v["arm"]: float(v["ROC_AUC"]) for v in records}
        text.append(
            f"| {d} | {g} | {s} | {e} / {r} | "
            + " | ".join(f"{values[a]:.6f}" for a in ("A", "B", "C"))
            + " |"
        )
    text += [
        "",
        "## Measured stage cost",
        "",
        "| Stage | Summed wall seconds | Native calls | History calls |",
        "|---|---:|---:|---:|",
    ]
    for (stage,), records in grouped(cost, ("stage",)).items():
        seconds = sum(r["seconds"] for r in records)
        forwards = sum(r.get("native_forwards", 0) for r in records)
        history = sum(r.get("history_forwards", 0) for r in records)
        text.append(f"| {stage} | {seconds:.3f} | {forwards} | {history} |")
    (dest / "PUBLIC.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(
        "REPORT_COMPLETE",
        len(rows),
        len(pairs),
        decision["expand_to_all_SMD28"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("action", choices=("report", "gate"))
    args = parser.parse_args()
    if args.action == "gate":
        raise SystemExit(0 if gate(args.workspace)["expand_to_all_SMD28"] else 42)
    report(args.workspace)
