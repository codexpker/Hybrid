"""Aggregate sealed TimesFM-conditioned LEFT results without changing a score."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .left_swat_author import read, dump, sha
from .left_timesfm_evaluation import OUT, METRICS


NAMES = {
    "A": "Matched source continuation LEFT",
    "B": "LEFT with history mean/variance",
    "C": "LEFT with pretrained TimesFM hidden history",
    "D": "LEFT with causally reassigned TimesFM history",
    "E": "LEFT with frozen random TimesFM history",
    "A_U": "Matched LEFT + ordinary source replay U",
    "C_U": "TimesFM hidden LEFT + ordinary source replay U",
    "frozen": "Earlier multi-source LEFT",
    "scratch": "LEFT target-normal training from scratch",
    "target_normal": "Multi-source LEFT + target-normal training",
    "level": "Normal-level deviation",
    "timeradar_block": "Official TimeRadar fixed block readout",
    "timeradar_endpoint": "Official TimeRadar endpoint V",
    "dada": "DADA official pretrained",
    "timercd": "TimeRCD official, FIT-only statistics",
    "pca": "PCA residual",
    "catch_2025": "CATCH seed2025",
    "catch_2026": "CATCH seed2026",
    "catch_2027": "CATCH seed2027",
    "catch": "CATCH (archived seeds, mean ± sample SD)",
    "old_source": "Earlier multi-source LEFT frozen",
    "old_random_u": "Random-initialized LEFT mixed U",
    "old_ordinary": "Earlier LEFT ordinary mixed U",
    "old_replay": "Earlier LEFT ordinary source replay U",
    "old_window": "Earlier LEFT window-reliability replay U",
    "old_source_only": "Earlier LEFT source-only continuation",
    "timesfm_forecast": "Official pretrained TimesFM median forecast residual",
}


def aggregate(frame):
    rows = []
    for keys, part in frame.groupby(["domain", "method", "panel", "role"], sort=False):
        row = dict(zip(("domain", "method", "panel", "role"), keys))
        assert not part.entity.duplicated().any()
        row.update(entities=len(part), points=int(part.points.sum()))
        for metric in METRICS:
            row[metric] = float(part[metric].mean())
            row[metric + "_valid_entities"] = int(part[metric].notna().sum())
        rows.append(row)
    frame = pd.DataFrame(rows)
    catches = frame[frame.method.str.startswith("catch_")]
    if len(catches):
        combined = []
        for (domain, panel, role), part in catches.groupby(["domain", "panel", "role"]):
            r = dict(
                domain=domain,
                panel=panel,
                role=role,
                method="catch",
                archived_seeds=len(part),
                entities=int(part.entities.iloc[0]),
                points=int(part.points.iloc[0]),
            )
            for metric in METRICS:
                r[metric] = float(part[metric].mean())
                r[metric + "_seed_sd"] = (
                    float(part[metric].std(ddof=1)) if len(part) > 1 else np.nan
                )
                r[metric + "_valid_entities"] = int(
                    part[metric + "_valid_entities"].min()
                )
            combined.append(r)
        frame = pd.concat(
            [frame[~frame.method.str.startswith("catch_")], pd.DataFrame(combined)],
            ignore_index=True,
        )
    return frame


def paired(frame):
    rows, details = [], []
    for (domain, panel, role), part in frame.groupby(["domain", "panel", "role"]):
        pairs = [("C", "A"), ("C", "B"), ("C", "D"), ("C", "E"), ("A", "frozen")]
        if role == "QUERY":
            pairs = [("C_U", "A_U"), ("C_U", "C"), ("A_U", "A"), ("C", "A")]
        for first, second in pairs:
            a = part[part.method == first].set_index("entity")
            b = part[part.method == second].set_index("entity")
            if not len(a) or not len(b):
                continue
            assert set(a.index) == set(b.index), (domain, first, second)
            b = b.loc[a.index]
            np.testing.assert_array_equal(a[["start", "stop"]], b[["start", "stop"]])
            delta = a[list(METRICS)] - b[list(METRICS)]
            for entity, value in delta.iterrows():
                details.append(
                    dict(
                        domain=domain,
                        entity=entity,
                        panel=panel,
                        role=role,
                        contrast=first + "-" + second,
                        **value.to_dict(),
                    )
                )
            for metric in METRICS:
                v = delta[metric].dropna()
                rows.append(
                    dict(
                        domain=domain,
                        panel=panel,
                        role=role,
                        contrast=first + "-" + second,
                        metric=metric,
                        valid_entities=len(v),
                        mean=float(v.mean()),
                        median=float(v.median()),
                        wins=int((v > 1e-12).sum()),
                        ties=int((v.abs() <= 1e-12).sum()),
                        losses=int((v < -1e-12).sum()),
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(details)


def number(value):
    return "NA" if pd.isna(value) else f"{value:.6f}"


def panel_markdown(frame, role, panel):
    part = frame[(frame.role == role) & (frame.panel == panel)]
    if part.empty:
        return "Not yet evaluated.\n"
    lines = []
    for metric in METRICS:
        lines += [
            f"#### {metric}",
            "",
            "| Method | PSM | SWaT | SMD28 macro |",
            "|---|---:|---:|---:|",
        ]
        for name in dict.fromkeys(part.method):
            source = part[part.method == name].set_index("domain")
            values = []
            for domain in ("PSM", "SWAT", "SMD"):
                if domain not in source.index:
                    values.append("NA")
                    continue
                value = number(source.loc[domain, metric])
                if name == "catch" and not pd.isna(
                    source.loc[domain, metric + "_seed_sd"]
                ):
                    value += " ± " + number(source.loc[domain, metric + "_seed_sd"])
                values.append(value)
            lines.append("| " + " | ".join([NAMES.get(name, name), *values]) + " |")
        lines.append("")
    return "\n".join(lines)


def costs(w):
    rows = []
    for folder in (w / OUT).iterdir():
        if folder.name not in ("PSM", "SWAT", "SMD"):
            continue
        receipt = folder / "A" / "receipt.json"
        if receipt.exists():
            r = read(receipt)
            rows.append(
                dict(
                    domain=folder.name,
                    stage="source A-E jointly",
                    object=folder.name,
                    seconds=r["shared_fold_elapsed_seconds"],
                    peak_bytes=r["peak_bytes"],
                    units=5,
                    updates_each=r["updates"],
                    current_windows_each=r["current_windows"],
                )
            )
        for path in folder.glob("U/*/A/receipt.json"):
            r = read(path)
            rows.append(
                dict(
                    domain=folder.name,
                    stage="U A/C jointly",
                    object=r["entity"],
                    seconds=r["shared_pair_seconds"],
                    peak_bytes=r["peak_bytes"],
                    units=2,
                    updates_each=r["updates"],
                    current_windows_each=r["target_exposure"] + r["source_exposure"],
                )
            )
        source_cache = folder / "U_SOURCE_FEATURES" / "receipt.json"
        if source_cache.exists():
            r = read(source_cache)
            rows.append(
                dict(
                    domain=folder.name,
                    stage="shared U source feature encoding",
                    object=folder.name,
                    seconds=r["seconds"],
                    units=1,
                    windows=r["rows"],
                    bytes=sum(f["bytes"] for f in r["files"].values()),
                )
            )
    for path in (w / OUT / "features").glob("*/*/*/*.json"):
        r = read(path)
        rows.append(
            dict(
                domain=r["inputs"]["domain"],
                stage="frozen feature extraction",
                object=str(path.relative_to(w / OUT)),
                seconds=r["seconds"],
                units=1,
                windows=r["shape"][0],
                bytes=0 if "parent_file" in r else int(np.prod(r["shape"])) * 4,
            )
        )
    for path in (w / OUT / "public").glob("*/*/*.json"):
        r = read(path)
        rows.append(
            dict(
                domain=r["domain"],
                stage="LEFT score",
                object=str(path.relative_to(w / OUT)),
                seconds=r["seconds"],
                units=1,
                points=r["points"],
            )
        )
    for path in (w / OUT / "controlled").glob("*/*.json"):
        r = read(path)
        if "seconds" in r:
            rows.append(
                dict(
                    domain=path.parent.name,
                    stage="controlled score or features",
                    object=str(path.relative_to(w / OUT)),
                    seconds=r["seconds"],
                    units=1,
                    timing_scope="native LEFT readout, excluding model loading and bank serialization"
                    if "checkpoint" in r
                    else "frozen encoding, output serialization and cache hash; excluding encoder loading",
                )
            )
    for path in (w / OUT / "metrics").glob("*/*/*.json"):
        r = read(path)
        rows.append(
            dict(
                domain=path.parent.parent.name,
                stage="CPU sealed-score metrics",
                object=str(path.relative_to(w / OUT)),
                seconds=r["seconds"],
                units=1,
                timing_scope="ranking/operating evaluation after input banks, labels and metric implementation loaded; excludes input verification and output serialization",
            )
        )
    scopes = {
        "source A-E jointly": "trainer wall time including initialization, feature encoding, all five updates, SELECT and state hashing; excludes final checkpoint serialization and prerequisite engineering tests",
        "U A/C jointly": "trainer wall time including initialization, target feature caching and paired updates; excludes reusable source feature precompute, prerequisite engineering tests and final checkpoint serialization",
        "shared U source feature encoding": "shared cache construction including encoder loading, encoding, flush and hashes; excludes source input loading before timer",
        "frozen feature extraction": "encode loop through cache flush; excludes encoder loading and pre/post model/cache SHA verification; zero for exact parent-row cache reuse",
        "LEFT score": "native score batches only, with synchronization; excludes feature encoding/loading, model loading, state hashes and bank serialization",
    }
    for row in rows:
        row.setdefault("timing_scope", scopes.get(row["stage"], "see receipt"))
        row["measurement"] = "stage wall-clock seconds; not GPU kernel time"
    return pd.DataFrame(rows)


def curves(w, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), constrained_layout=True)
    for ax, domain in zip(axes, ("PSM", "SWAT", "SMD")):
        path = w / OUT / domain / "CURVES.json"
        if not path.exists():
            continue
        r = read(path)
        for arm in ("A", "B", "C", "D", "E"):
            ax.plot([x["step"] for x in r], [x["loss"][arm] for x in r], label=arm)
        ax.set(
            xlabel="Source update", ylabel="Native complete training loss", title=domain
        )
        ax.legend(ncol=3, fontsize=8)
    fig.savefig(destination / "source_learning_curves.pdf")
    fig.savefig(destination / "source_learning_curves.png", dpi=160)
    plt.close(fig)


def source_history_contrast(w, destination):
    import torch

    records = []
    for domain in ("PSM", "SWAT", "SMD"):
        path = w / OUT / domain / "SELECT_FEATURE_FIXTURE.pt"
        if not path.exists():
            continue
        digest = sha(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        current, valid = data["features"]["C"]
        donor, donor_valid = data["features"]["D"]
        mask = valid & donor_valid
        for i in range(len(current)):
            a = current[i][mask[i]].double().reshape(-1)
            b = donor[i][mask[i]].double().reshape(-1)
            rms = float((a - b).square().mean().sqrt())
            reference = float(a.square().mean().sqrt())
            cosine = float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-30))
            row = data["rows"][i]
            shuffled = data["donors"][i]
            records.append(
                dict(
                    domain=domain,
                    row=i,
                    source_domain=int(row[0]),
                    record=int(row[1]),
                    current_start=int(row[2]),
                    donor_history_end=int(shuffled[2]),
                    history_overlap=max(
                        0, int(data["lengths"][i]) - int(row[2]) + int(shuffled[2])
                    )
                    / int(data["lengths"][i]),
                    feature_RMS_difference=rms,
                    relative_RMS=rms / max(reference, 1e-30),
                    feature_cosine=cosine,
                    fixture_sha256=digest,
                )
            )
    pd.DataFrame(records).to_csv(
        destination / "source_SELECT_history_contrast.csv", index=False
    )


def summarize(w, allow_partial=False):
    output = w / OUT / "summary"
    output.mkdir(exist_ok=True, parents=True)
    files = sorted((w / OUT / "metrics").glob("*/*/*.json"))
    records, operations, events = [], [], []
    identities = []
    for path in files:
        data = read(path)
        records.extend(data["rows"])
        operations.extend(data["operations"])
        events.extend(data["events"])
        identities.append(dict(path=str(path.relative_to(w)), sha256=sha(path)))
    assert records, "No sealed metric records available"
    detail = pd.DataFrame(records)
    assert not detail.duplicated(["domain", "entity", "method", "panel", "role"]).any()
    if not allow_partial:
        for domain, count in (("PSM", 1), ("SWAT", 1), ("SMD", 28)):
            for role, panel, arms in (
                ("TEST", "complete", ("A", "B", "C", "D", "E")),
                ("QUERY", "mixed", ("A", "C", "A_U", "C_U")),
            ):
                for arm in arms:
                    selected = detail[
                        (detail.domain == domain)
                        & (detail.role == role)
                        & (detail.panel == panel)
                        & (detail.method == arm)
                    ]
                    assert len(selected) == count, (
                        "incomplete",
                        domain,
                        role,
                        arm,
                        len(selected),
                        count,
                    )
        assert (w / OUT / "controlled/comparisons.csv").exists()
    detail.to_csv(output / "entity_metrics.csv", index=False)
    mean = aggregate(detail)
    mean.to_csv(output / "macro_metrics.csv", index=False)
    comparison, entity_comparison = paired(detail)
    comparison.to_csv(output / "paired_summary.csv", index=False)
    entity_comparison.to_csv(output / "paired_entities.csv", index=False)
    pd.DataFrame(operations).to_csv(output / "operations.csv", index=False)
    pd.DataFrame(events).to_csv(output / "events.csv", index=False)
    budget = costs(w)
    budget.to_csv(output / "actual_costs.csv", index=False)
    curves(w, output)
    source_history_contrast(w, output)
    controlled = w / OUT / "controlled/comparisons.csv"
    if controlled.exists():
        control = pd.read_csv(controlled)
        control[control.region == "core"].to_csv(
            output / "controlled_core.csv", index=False
        )
        contrasts = []
        for (domain, group, region, target, reference), part in control.groupby(
            ["domain", "group", "region", "target", "reference"]
        ):
            indexed = part.set_index("arm")
            for comparator in ("A", "B", "D", "E"):
                if "C" in indexed.index and comparator in indexed.index:
                    contrasts.append(
                        dict(
                            domain=domain,
                            group=group,
                            region=region,
                            target=target,
                            reference=reference,
                            contrast="C-" + comparator,
                            AP=float(
                                indexed.loc["C", "AP"] - indexed.loc[comparator, "AP"]
                            ),
                            ROC_AUC=float(
                                indexed.loc["C", "ROC_AUC"]
                                - indexed.loc[comparator, "ROC_AUC"]
                            ),
                        )
                    )
        pd.DataFrame(contrasts).to_csv(output / "controlled_contrasts.csv", index=False)
    axes = read(w / OUT / "TARGET_INPUTS.json")
    cal = [
        dict(
            domain=r["domain"],
            entity=r["entity"],
            legal_points=r["cal_legal_count"],
            fixed_points=512,
            status=r["operations"],
        )
        for r in axes["records"]
        if r["role"] == "CAL"
    ]
    pd.DataFrame(cal).to_csv(output / "cal_eligibility.csv", index=False)
    checks = dict(
        metric_files=identities,
        independent_AP_ROC_max_error=float(detail.independent_AP_ROC_error.max()),
        aggregation="per-entity then equal-weight domain macro; one development seed",
        VUS_review="fixed author implementation, aggregation only; no independent VUS algorithm",
        axes_sha256=sha(w / OUT / "TARGET_INPUTS.json"),
        donors_and_archives_sha256=sha(w / OUT / "DONORS_AND_ARCHIVE_AXES.json"),
    )
    dump(output / "summary_receipt.json", checks)
    text = "# Frozen TimesFM hidden-conditioned LEFT: sealed results\n\n"
    text += "All learned rows use seed2025. SMD reports all28 entities with equal weights; entities are not training repetitions. All current-window scores are native LEFT, available only at the192-point current-window end. Preceding512 observations are restricted to the same record and role. The frozen TimesFM residual forecast baseline appears only on the compatible mixed-role intersection; it has no complete-test cache here.\n\n"
    text += "Source continuation uses a fixed2000-update endpoint in each fold; normal SOURCE SELECT is diagnostic only and never selects a checkpoint. D assigns an earlier legal same-record history with isolated randomness; it is not a frequency-preserving permutation. Short TEP runs can make the reassigned histories overlap substantially. The saved SOURCE SELECT fixture quantifies feature differences and overlap without another forward pass.\n\n"
    text += "actual_costs.csv reports measured stage wall-clock times with a timing_scope field. Encoding timers exclude model initialization and state hashes, and score timers exclude encoding. These phases are not exact GPU kernel times or a complete end-to-end billing ledger. Waiting for source checkpoints is not charged as model computation. CPU metric reevaluation is listed separately.\n\n"
    text += "## Complete native common axis\n\n" + panel_markdown(
        mean, "TEST", "complete"
    )
    text += "## External intersection axis\n\n" + panel_markdown(
        mean, "TEST", "external"
    )
    text += "The previous full LEFT source, scratch and target-normal rows retain their separate learning permissions. TimeRadar endpoint, DADA and TimeRCD rows retain their original preprocessing and availability; common points do not make information permissions identical. CATCH summarizes archived seeds with sample standard deviation; individual values remain in entity_metrics.csv. No external method is retrained.\n\n"
    text += "## Independent mixed-role axis\n\n" + panel_markdown(
        mean, "QUERY", "mixed"
    )
    text += "The original512 CAL points remain fixed. When any lacks a full legal512 history, threshold operations are NA; no replacement CAL, zero padding across roles, or threshold selection is performed. Historical PSM all-attack CAL and prior SMD nominal-normal CAL results retain their old protocol.\n\n"
    text += "## Paired evidence\n\nAll four metrics and per-entity signs/medians are in paired_summary.csv and paired_entities.csv. Differences are computed only after confirming identical current start/stop axes. The controlled panel reuses the archived edits with256 actual historical observations and left padding to512. Synthetic records each offer only one legal history, so D retains identity there; D cannot test history reassignment in that controlled group.\n"
    (output / "RESULTS.md").write_text(text, encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    print(summarize(args.workspace, args.allow_partial))
