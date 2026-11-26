"""Assemble complete entity-level and macro-average tables from sealed CPU results."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path("docs/experiments")
NEW = ROOT / "left_smd_reliability_20261126"
MULTI = ROOT / "left_multisource_20261126"
EXT = ROOT / "native_endpoint_external_20261125"
HIST = ROOT / "left_decoder_context_20261124/public_extension"
METRICS = ["AP", "ROC_AUC", "VUS_PR", "VUS_ROC"]
NAMES = {
    "source": "Frozen multisource LEFT",
    "N": "Multisource LEFT + normal training",
    "scratch_N": "Target-normal LEFT from scratch",
    "random_u": "Random-init target U",
    "ordinary": "Multisource target U",
    "replay": "Multisource source replay",
    "window": "Window-weighted source replay",
    "source_only": "Source continuation",
}


def load_tables(folder):
    blobs = [json.loads(p.read_text()) for p in sorted(folder.glob("machine-*.json"))]
    assert len(blobs) == 28 * 9
    return tuple(
        pd.DataFrame(row for blob in blobs for row in blob[key])
        for key in ("rankings", "operations", "events")
    )


def pairs(metrics):
    differences = []
    summaries = []
    for panel, target, comp in [
        ("public", "N", "source"),
        ("public", "N", "scratch_N"),
        ("mixed", "window", "replay"),
        ("mixed", "window", "source"),
        ("mixed", "replay", "ordinary"),
        ("mixed", "window", "source_only"),
        ("mixed", "ordinary", "random_u"),
    ]:
        g = metrics[metrics.panel.eq(panel)]
        lhs = g[g.arm.eq(target)].set_index("entity")
        rhs = g[g.arm.eq(comp)].set_index("entity")
        assert len(lhs) == len(rhs) == 28 and lhs.index.equals(rhs.index)
        for key in ["start", "stop", "points", "positives"]:
            assert lhs[key].equals(rhs[key])
        delta = lhs[METRICS] - rhs[METRICS]
        differences.append(
            delta.assign(panel=panel, target=target, reference=comp).reset_index()
        )
        for key, series in delta.items():
            s = series.dropna()
            summaries.append(
                dict(
                    panel=panel,
                    target=target,
                    reference=comp,
                    metric=key,
                    n=len(s),
                    mean=s.mean(),
                    median=s.median(),
                    win=(s > 0).sum(),
                    tie=(s == 0).sum(),
                    loss=(s < 0).sum(),
                )
            )
    return pd.concat(differences), pd.DataFrame(summaries)


def public_rows(metrics, psm_ext):
    old = pd.read_csv(HIST / "methods.csv")
    external = pd.read_csv(EXT / "entity_metrics.csv")
    external = external[
        (external.panel == "common_offline") & external.domain.isin(["PSM", "SMD"])
    ]
    names = {
        "official_V": "Pretrained TimeRadar V (causal endpoint)",
        "DADA official V": "Pretrained DADA",
        "TimeRCD official weights, FIT-only adapter": "Pretrained TimeRCD (FIT-only)",
        "PCA residual": "PCA residual",
    }
    direct = (
        metrics[metrics.panel.isin(["public", "external_common"])]
        .assign(domain="SMD")
        .copy()
    )
    rows = []
    for (panel, arm), g in direct.groupby(["panel", "arm"]):
        rows.append(
            dict(
                panel=panel,
                domain="SMD",
                method=NAMES[arm],
                **g[METRICS].mean().to_dict(),
            )
        )
    for d in ("PSM", "SWAT"):
        for row in pd.read_csv(MULTI / d / "public.csv").to_dict("records"):
            rows.append(
                dict(
                    panel="public",
                    domain=d,
                    method=NAMES[row["arm"]],
                    **{k: row[k] for k in METRICS},
                )
            )
    for r in pd.read_csv(psm_ext).to_dict("records"):
        rows.append(
            dict(
                panel="external_common",
                domain="PSM",
                method=NAMES[r["arm"]],
                **{k: r[k] for k in METRICS},
            )
        )
    for (domain, method), g in external.groupby(["domain", "method"]):
        if domain == "SMD":
            match = direct[
                (direct.panel == "external_common") & (direct.arm == "N")
            ].set_index("entity")
            matched = g.drop_duplicates("entity").set_index("entity")
            assert len(match) == len(matched) == 28
            assert (
                (
                    match[["start", "stop", "positives"]].sort_index()
                    == matched[["start", "stop", "positives"]].sort_index()
                )
                .all()
                .all()
            )
        if method not in names and not method.startswith("CATCH "):
            continue
        rows.append(
            dict(
                panel="external_common",
                domain=domain,
                method=names.get(method, "CATCH"),
                **g[METRICS].mean().to_dict(),
            )
        )
    for r in old.to_dict("records"):
        panel = r["panel"]
        d = r["domain"]
        m = r["method"]
        if not (
            (d in ("PSM", "SWAT") and panel == "archived_full")
            or (d == "SMD" and panel == "internal")
        ):
            continue
        if m.startswith("TimeRadar official block0"):
            m = "Pretrained TimeRadar (native block grid)"
        elif m.startswith("CATCH "):
            m = "CATCH"
        elif m not in ("PCA residual", "Pointwise level deviation"):
            continue
        rows.append(
            dict(panel="public", domain=d, method=m, **{k: r[k] for k in METRICS})
        )
    return (
        pd.DataFrame(rows)
        .groupby(["panel", "domain", "method"])[METRICS]
        .mean()
        .reset_index()
    )


def report(folder, psm_ext):
    NEW.mkdir(parents=True, exist_ok=True)
    rank, ops, events = load_tables(folder)
    assert not rank.duplicated(["entity", "arm", "panel"]).any()
    assert ops.groupby(["arm", "stream"]).size().eq(28).all()
    rank.to_csv(NEW / "smd_entity_metrics.csv", index=False)
    ops.to_csv(NEW / "smd_entity_operations.csv", index=False)
    events.to_csv(NEW / "smd_events.csv", index=False)
    dif, stats = pairs(rank)
    dif.to_csv(NEW / "smd_entity_deltas.csv", index=False)
    stats.to_csv(NEW / "smd_paired_distribution.csv", index=False)
    scratch = (
        rank[(rank.arm == "scratch_N") & rank.panel.eq("public")]
        .set_index("entity")
        .sort_index()
    )
    cached = pd.read_csv(HIST / "entity_metrics.csv")
    cached = (
        cached[
            (cached.domain == "SMD")
            & (cached.method == "Native LEFT")
            & (cached.panel == "internal")
        ]
        .set_index("entity")
        .sort_index()
    )
    assert len(cached) == 28 and (scratch.bank_sha256 == cached.cache_sha256).all()
    cached_error = (scratch[METRICS] - cached[METRICS]).abs().max().to_dict()
    assert max(cached_error.values()) < 1e-10
    catch_summaries = []
    old = pd.read_csv(HIST / "methods.csv")
    external = pd.read_csv(EXT / "entity_metrics.csv")
    for panel, archive, label in (
        ("public", old, "archived_full"),
        ("external_common", external, "common_offline"),
    ):
        catch = archive[
            (archive.domain == "PSM")
            & (archive.panel == label)
            & archive.method.str.startswith("CATCH W192 seed")
        ]
        assert len(catch) == catch.method.nunique() == 3
        for key in METRICS:
            catch_summaries.append(
                dict(
                    panel=panel,
                    metric=key,
                    n=3,
                    mean=catch[key].mean(),
                    sample_sd=catch[key].std(ddof=1),
                )
            )
    pd.DataFrame(catch_summaries).to_csv(NEW / "psm_catch_seeds.csv", index=False)
    public = public_rows(rank, psm_ext)
    public.to_csv(NEW / "public_methods.csv", index=False)
    op_metrics = [
        "normal_FPR",
        "Recall",
        "F1",
        "one_to_one_eventF1",
        "auxiliary_any_hit_recall",
        "normal_alarm_fragments",
        "event_macro_coverage",
        "event_point_coverage",
        "long_steady_macro_coverage",
        "long_steady_point_coverage",
        "max_event_miss_points",
        "offline_availability_offset_median",
    ]
    aggregate = ops.groupby(["stream", "arm"])[op_metrics].agg(["mean", "count"])
    aggregate.to_csv(NEW / "smd_operation_macro.csv")
    totals = []
    for (stream, arm), g in ops.groupby(["stream", "arm"]):
        ge = events[
            (events.stream == stream) & (events.arm == arm) & events.main_eligible
        ]
        long = ge[ge.total_points > 512]
        hits = int(
            (long.steady_after512_coverage * (long.total_points - 512)).round().sum()
        )
        steady = int((long.total_points - 512).sum())
        totals.append(
            dict(
                stream=stream,
                arm=arm,
                qualified_events=len(ge),
                qualified_long_events=len(long),
                matched=int(g.M.sum()),
                any_hit=int(g.auxiliary_any_hit_events.sum()),
                false_alarm_fragments=int(g.normal_alarm_fragments.sum()),
                long_event_macro_coverage=long.steady_after512_coverage.mean(),
                long_steady_points=steady,
                long_steady_alarm_points=hits,
                long_point_pooled_coverage=hits / steady if steady else None,
            )
        )
    pd.DataFrame(totals).to_csv(NEW / "smd_event_totals.csv", index=False)
    denominators = rank.groupby(["panel", "arm"])[METRICS].count()
    denominators.to_csv(NEW / "smd_metric_denominators.csv")
    mixed = rank[rank.panel.eq("mixed")].groupby("arm")[METRICS].mean()
    mixed.index = mixed.index.map(NAMES)
    mixed.to_csv(NEW / "smd_mixed_macro.csv")
    text = [
        "# Multisource LEFT on the SMD28 held-out target fold",
        "Each SMD cell gives an equal-weight mean of eligible machines. The complete SMD dataset was excluded from both source fitting and source selection in this fold. PSM and SWaT use the earlier TEP–SMD source pair. All three tasks have historical project exposure. The fixed training seed is 2025.",
        "## Public detection",
        "The primary panel uses every complete W192 test block. All LEFT methods use normal-prefix standardization including target normal validation in this public panel. Z has no target parameter updates. N and the scratch detector use the same target-normal training contract. Interior LEFT scores become available at the window endpoint. PSM uses [0,87744), SWaT uses [495000,944856), and each SMD machine uses [0,192⌊N/192⌋). CATCH uses normal target training with archived FIT statistics (mean of three seeds); PCA uses a fitted linear residual; level deviation uses the normal-prefix scaler and current observation. TimeRadar's native block-grid comparator uses the original blocked inference cache.",
    ]
    for panel, title, domains in [
        ("public", "Complete LEFT axes", ["PSM", "SWAT", "SMD"]),
        ("external_common", "Shared external axes", ["PSM", "SMD"]),
    ]:
        text += ["### " + title]
        if panel == "external_common":
            text += [
                "Each entity uses [99,min(100⌊N/100⌋,192⌊N/192⌋)). PSM and SMD use the verified archive input identities. DADA uses a frozen W100 block model and full normal-prefix scaler. TimeRCD uses original logits, FIT-only statistics and W5000 offline context. The frozen TimeRadar V endpoint comparator reads the trailing W100. SWaT has no equivalent archived DADA comparison in this panel. Source training, target normal fitting and lookahead permissions vary by row."
            ]
        s = public[public.panel.eq(panel)]
        for k in METRICS:
            table = s.pivot(index="method", columns="domain", values=k).reindex(
                columns=domains
            )
            table = table.astype(object)
            if "CATCH" in table.index and "PSM" in table:
                summary = next(
                    r
                    for r in catch_summaries
                    if r["panel"] == panel and r["metric"] == k
                )
                assert abs(summary["mean"] - table.loc["CATCH", "PSM"]) < 1e-12
                table.loc["CATCH", "PSM"] = (
                    f"{summary['mean']:.5f} ± {summary['sample_sd']:.5f}"
                )
            for col in table.columns:
                table[col] = table[col].map(
                    lambda v: v
                    if isinstance(v, str)
                    else f"{v:.5f}"
                    if pd.notna(v)
                    else "NA"
                )
            text += [
                "**" + k + "**",
                table.rename(columns={"SWAT": "SWaT", "SMD": "SMD28"})
                .to_markdown(disable_numparse=True)
                .replace("nan", "NA"),
            ]
    text += [
        "## Independent mixed adaptation",
        mixed.to_markdown(floatfmt=".6f"),
        "CAL is taken from the reserved nominal-normal training record with FIT-only standardization. Each checkpoint sets its own strict-greater q99/higher cutoff on 512 covered CAL points. The operating table gives entity-macro summaries and eligible denominators for each measure, and the event file preserves one-to-one matching, any-hit, retrospective offsets and censored intervals. Per-entity measurement precedes aggregation. A machine without both QUERY classes has undefined ranking metrics. Coverage restricted to long events requires complete events longer than 512 points.",
        "### Paired effects across machines",
        stats.to_markdown(index=False, floatfmt=".6f"),
        "Win, tie and loss counts use full-precision signed differences. Machine counts describe the distribution over this fixed SMD panel with one training seed.",
        "[Operating metrics and denominators](smd_operation_macro.csv) · [Per-machine ranking](smd_entity_metrics.csv) · [Per-machine differences](smd_entity_deltas.csv) · [Per-machine operations](smd_entity_operations.csv) · [Event detail](smd_events.csv) · [Counts and pooled coverage across complete events](smd_event_totals.csv).",
    ]
    text += ["### Normal calibration and event costs"]
    for stream in ("strict", "confirmed3"):
        table = (
            aggregate.xs(stream)
            .loc[
                :,
                [
                    (col, "mean")
                    for col in (
                        "normal_FPR",
                        "Recall",
                        "one_to_one_eventF1",
                        "auxiliary_any_hit_recall",
                        "normal_alarm_fragments",
                        "long_steady_macro_coverage",
                        "long_steady_point_coverage",
                    )
                ],
            ]
            .copy()
        )
        table.columns = [
            "FPR",
            "Point recall",
            "Event F1",
            "Any-hit recall",
            "FP fragments/machine",
            "Long macro coverage",
            "Long pooled coverage per machine",
        ]
        table.index = table.index.map(NAMES)
        text += [
            "**" + stream + "**",
            table.to_markdown(floatfmt=".5f").replace("nan", "NA"),
        ]
    eligible = rank[rank.panel.eq("mixed") & rank.arm.eq("replay")].AP.count()
    ref_ops = ops[(ops.arm == "replay") & (ops.stream == "strict")]
    text += [
        f"Ranking supports both classes on {eligible}/28 mixed suffixes. "
        f"{int(ref_ops.Ntrue.gt(0).sum())}/28 have an eligible complete event; "
        f"{int(ref_ops.long_event_count.gt(0).sum())}/28 have an eligible long event. "
        "Timing is measured in sample points with a known window-end availability time. "
        "The independent CAL has nominal-normal permissions and provides an empirical operating point."
    ]
    review = dict(
        matched_scratch_cache_max_error=cached_error,
        banks_per_role=rank.groupby("panel").size().to_dict(),
        max_ap_roc_absolute_discrepancy=float(rank.check_ap_roc_maxerr.max()),
        operations_rows=int(len(ops)),
        thresholds="q99/higher reproduced by sorted order statistic",
        point_counts="strict and non-backfilled consecutive-three TP/FP independently checked",
        vus="Pinned LEFT VUS implementation; checked aggregation and axes, no second VUS implementation",
        paired_review="Entity joins verified equal interval, point count and class count",
    )
    (NEW / "metric_review.json").write_text(
        json.dumps(review, indent=2), encoding="utf8"
    )
    text.extend(
        [
            "## Source identity and compute",
            "HAI 21.03 train1 and train3 contributed 216,001 and 478,801 consecutive one-second FIT observations, respectively. Independent train2 SELECT contained 226,801 observations. All three files passed the pinned Git blob checks. Each has 79 process columns, zero missing values, and four all-zero attack indicators. The timestamps and all indicators were excluded from the learning arrays. Each FIT file used its own scaler, while train2 used point-pooled FIT statistics. The two source domains each supplied 64,000 training windows. The checkpoint at step 4,000 had the lowest fixed source SELECT loss (0.523464); 1,001 of the 4,000 updates reached the maximum memory mixing weight. [File and scaler identities](source_inputs.json) · [Source training receipt](source_training.json).",
            "The 38-channel source training took 152.54 A40 seconds. The 140 per-machine training runs and one shared source-only run took 12,116.83 A40 seconds. The 392 new SMD score banks took 202.10 seconds. Six pre-update setup attempts stopped at scaler or teacher checks and were recovered; the retry intervals, which include idle time, are listed separately. The six PSM/SWaT permutation trainings took a further 467.24 seconds and their twelve score banks took 12.88 seconds. Successful training and scoring totaled 3.60 A40 hours. CPU evaluation, file transfer and failed setup intervals are excluded from this total. [Training and recovery receipts](training_review.json) · [Metric review](metric_review.json) · [Permutation results and costs](permutation.md).",
        ]
    )
    (NEW / "SMD_RESULTS.md").write_text("\n\n".join(text) + "\n", encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics_dir", type=Path)
    p.add_argument("psm_external", type=Path)
    args = p.parse_args()
    report(args.metrics_dir, args.psm_external)
