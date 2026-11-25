"""Render full native learning scope comparisons and independently verify exports."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .native_source_full import ARMS
from .native_source_full_metrics import NAMES,MATCHED
from .native_full_window_metrics import METRICS

DOMAINS=("PSM","SMD","MSL","SMAP")


def table(frame,methods):
    lines=["| Method | PSM | SMD28 | MSL27 | SMAP53 |","|---|---:|---:|---:|---:|"]
    for method,seed in methods:
        rows=frame[(frame.method==method)&(frame.seed==seed)].set_index("domain")
        cells=[" / ".join(f"{rows.loc[d,k]:.4f}" for k in METRICS) if d in rows.index else "NA" for d in DOMAINS]
        lines.append("| "+" | ".join([method+(f" (seed {seed})" if seed else ""),*cells])+" |")
    return lines


def build(root):
    methods=pd.read_csv(root/"methods.csv")
    entities=pd.read_csv(root/"entity_metrics.csv")
    pairs=pd.read_csv(root/"entity_pairs.csv")
    dist=pd.read_csv(root/"paired_distribution.csv")
    keys=["domain","panel","method","seed","timing"]
    recomputed=entities.groupby(keys)[list(METRICS)].mean().sort_index()
    exported=methods.set_index(keys).loc[recomputed.index,list(METRICS)].sort_index()
    np.testing.assert_allclose(recomputed,exported,atol=1e-14,rtol=0)
    for key,sub in pairs.groupby(["domain","panel","method","reference"]):
        for metric in METRICS:
            row=dist.set_index(["domain","panel","method","reference","metric"]).loc[(*key,metric)]
            ds=sub[metric]
            np.testing.assert_allclose([row["mean"],row["median"]],[ds.mean(),ds.median()],atol=1e-14,rtol=0)
            assert (row.wins,row.ties,row.losses)==((ds>0).sum(),(ds==0).sum(),(ds<0).sum())
    selected={}
    for arm in ARMS:
        rec=json.loads((root/arm/"SELECTION.json").read_text())
        rows=rec["source_select"]
        assert [r["update"] for r in rows]==list(range(0,241,40))
        risks=[np.mean(r["risks_event_by_stratum"]) for r in rows]
        np.testing.assert_allclose(risks,[r["risk"] for r in rows],rtol=0,atol=1e-14)
        selected[arm]=rows[np.argmin(risks)]["update"]
        assert selected[arm]==rec["selected_step"]
    lines=["# Native parameter scope and source-supervised deployment statistics","",
        "All cells give AP / ROC-AUC / VUS-PR / VUS-ROC. SMD and NASA are equal-entity means. The new models use seed2025. Different methods retain their respective training and context permissions.","",
        "## Complete method comparison on the external common axis",""]
    frame=methods[methods.panel=="common_offline"]
    lines.extend(table(frame,list(frame[["method","seed"]].drop_duplicates().itertuples(index=False,name=None))))
    lines.extend(["","## Original source contract: scope and criterion on the complete causal axis",""])
    frame=methods[methods.panel=="causal"]
    names=[(n,2025) for a in ARMS for n in (MATCHED[a],NAMES[a])]
    lines.extend(table(frame,names))
    lines.extend(["","## Paired differences on the complete causal axis","",
        "| Contrast | PSM | SMD28 | MSL27 | SMAP53 |","|---|---:|---:|---:|---:|"])
    for arm,ref in [(NAMES[a],MATCHED[a]) for a in ARMS]+[
            (NAMES["point_weighted"],NAMES["copy"]),(NAMES["point_weighted"],NAMES["point_weighted_e"])]:
        sub=dist[(dist.panel=="causal")&(dist.method==arm)&(dist.reference==ref)].set_index(["domain","metric"])
        cells=[" / ".join(f"{sub.loc[(d,k),'mean']:+.4f}" for k in METRICS) for d in DOMAINS]
        lines.append("| "+" | ".join([arm+" minus "+ref,*cells])+" |")
    lines.extend(["","All new checkpoints are selected by their original score-specific source-event risk, including step0. The reconstruction control also uses source labels for SELECT. PSM is an exposed development task. SMD normal training records were source-visible. NASA uses only sensor zero. SWaT's source-labeled intervals are excluded from target benchmarks.","",
        "Operating metrics use the preserved separate PSM suffix, with all512 CAL points attack-labeled. NASA uses small CAL panels and q99/higher equals their sample maxima. SMD calibration is nominally normal. Strict and confirmed-three streams, one-to-one matching, any-hit and long-event coverage are exported without changing these permissions.","",
        "[Full-precision methods](methods.csv) · [Entity metrics](entity_metrics.csv) · [Pairs](entity_pairs.csv) · [Median and wins/ties/losses](paired_distribution.csv) · [Operations](operating.csv) · [Costs](EVALUATION.json)."])
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    (root/"RECHECK.json").write_text(json.dumps(dict(selected=selected,new_entity_rows=len(entities),
        paired_entity_rows=len(pairs),AP_ROC="Independent tied-rank recomputation on raw scores in each entity/panel",
        VUS="Fixed upstream implementation; independent macro and pair aggregation only"),indent=2),encoding="utf8")


if __name__=="__main__":
    import sys
    build(Path(sys.argv[1]))
