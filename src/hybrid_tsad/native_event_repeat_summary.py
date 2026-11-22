"""Summarize the fixed source-training repeats on the published endpoint axes."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANK = ["AP", "ROC_AUC", "VUS_PR", "VUS_ROC"]
OPER = ["normal_FPR", "Recall", "normal_alarm_fragments", "M", "Ntrue", "event_F1",
        "auxiliary_any_hit_events", "long_events", "long_steady_macro_coverage",
        "long_steady_pooled_coverage", "longest_event_miss", "hit_only_median_delay"]
BASE = "frozen_native_V"
COPY = "copy"
POINT = "point_weighted"


def archive_seed(table):
    return table.assign(seed=np.where(table.method == BASE, 0, 2025))


def write_summary(frame, metrics, groups, path):
    grouped = frame.groupby(groups, sort=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    sd = grouped.std(ddof=1).add_suffix("_sd")
    mean.join(sd).join(grouped.size().rename("n")).reset_index().to_csv(path, index=False)


def matched(frame, metrics, keys):
    pairs = []
    for _, group in frame.groupby(keys, sort=False):
        assert len(group) == 7 and not group.duplicated(["method", "seed"]).any()
        official, = group[group.method == BASE].to_dict("records")
        assert official["seed"] == 0
        for seed in (2025, 2026, 2027):
            treatment, = group[(group.seed == seed) & (group.method == POINT)].to_dict("records")
            copy, = group[(group.seed == seed) & (group.method == COPY)].to_dict("records")
            for control in (official, copy):
                pairs.append({**{k: treatment[k] for k in keys}, "seed": seed,
                              "reference": control["method"],
                              **{k: treatment[k] - control[k] for k in metrics}})
    return pd.DataFrame(pairs)


def summarize(root):
    frozen = root.parent / "native_event_rank_public_20261122"
    first_psm = pd.read_csv(frozen / "psm/PSM_RANKING.csv").query("axis_start == 99")
    first_space = pd.read_csv(frozen / "space/ranking_macro.csv")
    rankings = pd.concat([archive_seed(first_psm), archive_seed(first_space),
                          pd.read_csv(root / "psm/ranking.csv"),
                          pd.read_csv(root / "space/ranking_macro.csv")], ignore_index=True)
    old_oper = pd.read_csv(root.parent / "native_event_rank_20261122/methods.csv")
    old_oper = old_oper[old_oper.method.isin([BASE, COPY, POINT])].rename(columns={
        "domain":"dataset", "one_to_one_eventF1":"event_F1", "long_event_count":"long_events",
        "long_steady_point_coverage":"long_steady_pooled_coverage",
        "max_event_miss_points":"longest_event_miss", "delay_hit_only_median":"hit_only_median_delay"})
    operation = pd.concat([archive_seed(old_oper), archive_seed(pd.read_csv(frozen / "space/operating.csv")),
                           pd.read_csv(root/"psm/operating.csv"), pd.read_csv(root/"space/operating.csv")],
                          ignore_index=True)
    for table, metrics, keys, name in [
            (rankings, RANK, ["dataset"], "ranking"),
            (operation, OPER, ["dataset", "stream"], "operating")]:
        table = table[keys + ["method", "seed"] + metrics].copy()
        assert set(table.dataset) == {"PSM", "MSL", "SMAP"} and table[metrics].notna().all().all()
        export = table.copy()
        export["seed"] = export.seed.mask(export.method == BASE).astype("Int64")
        export.to_csv(root/f"{name}_by_seed.csv", index=False)
        write_summary(table, metrics, keys + ["method"], root/f"{name}_methods.csv")
        pairs = matched(table, metrics, keys)
        pairs.to_csv(root/f"{name}_paired_seeds.csv", index=False)
        write_summary(pairs, metrics, keys + ["reference"], root/f"{name}_paired_summary.csv")

    old = []
    for domain in ("MSL", "SMAP"):
        data = pd.read_csv(frozen / f"space/{domain}_methods.csv").query("stream == 'strict'")
        old.append(archive_seed(data))
    entities = pd.concat([*old, pd.read_csv(root/"space/entity_ranking.csv")], ignore_index=True)
    deltas = matched(entities, RANK, ["dataset","entity"])
    records = []
    for (domain, ref, seed), group in deltas.groupby(["dataset","reference","seed"]):
        assert len(group) == {"SMAP":53,"MSL":27}[domain]
        for k in RANK:
            v = group[k]
            expected = rankings.query("dataset == @domain and method == @POINT and seed == @seed")[k].item()
            ref_seed = 0 if ref == BASE else seed
            expected -= rankings.query("dataset == @domain and method == @ref and seed == @ref_seed")[k].item()
            np.testing.assert_allclose(v.mean(), expected, atol=1e-12)
            records.append(dict(dataset=domain,reference=ref,seed=seed,metric=k,
                                wins=(v>0).sum(),ties=(v==0).sum(),losses=(v<0).sum(),median=v.median()))
    pd.DataFrame(records).to_csv(root/"entity_paired_distribution.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("result_root", type=Path)
    summarize(p.parse_args().result_root)
