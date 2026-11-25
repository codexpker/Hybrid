"""Render full native learning scope comparisons and independently verify exports."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
ARMS=("copy","point_weighted","point_weighted_e")
NAMES={"copy":"Full normal reconstruction / V","point_weighted":"Full V supervision / V",
       "point_weighted_e":"Full E supervision / E"}
MATCHED={"copy":"Source reconstruction continuation","point_weighted":"Source-supervised endpoint V",
         "point_weighted_e":"Decoder E supervision / E"}
METRICS=("AP","ROC_AUC","VUS_PR","VUS_ROC")

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
    cost=json.loads((root/"EVALUATION.json").read_text())
    lines.extend(["","## Interpretation and actual cost","",
        "Full-model V supervision lowered AP relative to decoder-only V in all four domains. PSM and MSL improved ROC-AUC and VUS-ROC while losing AP and VUS-PR. SMD declined on all four metrics, with AP wins in five and losses in twenty-three machines. SMAP's AP mean fell despite27 positive and26 negative entity differences, with a near-zero median.","",
        "With full updates, V supervision exceeded reconstruction in mean AP on PSM, MSL and SMAP but lost12.85 AP percentage points on SMD. Its AP exceeded full E on PSM only. The full reconstruction control improved all four SMD metrics over its decoder counterpart. Full E improved all four measures on MSL and SMAP. These are task-dependent effects of the complete training and source-selection procedures.","",
        "The selected update numbers are40,200,120 for reconstruction,V,E. Nominal exposures at those checkpoints are1,280,6,400,3,840; labeled pair exposures are0,3,000,1,800. All trajectories executed240 updates. Normal reconstruction also used source-event SELECT. Independent initial pretraining, seed repetitions and target adaptation are outside this comparison.","",
        f"Formal training and source-validation time was {cost['train_seconds']/3600:.3f} A40-hours. Causal score generation, input checks and output writing totaled {cost['inference_seconds']/3600:.3f} A40-hours summed over GPUs. The three models each executed their own encoder and adaptive order. All109 objects retained both scores and timestamp footprints. CPU metric workers accumulated {cost['metric_worker_seconds']:.1f} seconds. Focused preliminary gradient checks and source/target bookkeeping are outside the formal GPU totals.","",
        "PSM three-confirmation false-positive rate and recall for full V were25.61% and60.76%, versus22.71% and44.24% for full reconstruction. One-to-one matches were14 versus7 out of48, while both had17 any-hit events. On SMD these rates were20.14%/38.28% for full V and18.92%/47.56% for reconstruction; matched events were148 versus159 out of327. NASA long-event statistics include four MSL and ten SMAP long events. The full operating exports retain these differences and all ineligible events.","",
        "The fixed full-parameter extension does not strengthen the existing V candidate's cross-task AP competitiveness. The archived decoder-only three-seed candidate remains a separate result. No new partial-unfreezing, loss, learning-rate or adaptation search follows this comparison."])
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    (root/"RECHECK.json").write_text(json.dumps(dict(selected=selected,new_entity_rows=len(entities),
        paired_entity_rows=len(pairs),AP_ROC="Independent tied-rank recomputation on raw scores in each entity/panel",
        VUS="Fixed upstream implementation; independent macro and pair aggregation only"),indent=2),encoding="utf8")


if __name__=="__main__":
    import sys
    build(Path(sys.argv[1]))
