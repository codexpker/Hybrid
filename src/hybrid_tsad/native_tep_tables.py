"""Render the two source-support comparison tables from sealed entity aggregates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

METRICS = ("AP","ROC_AUC","VUS_PR","VUS_ROC")
DOMAINS = ("PSM","SMD","MSL","SMAP")
NAMES = {
    "single_V":"SWaT event supervision, matched V",
    "expanded_V":"SWaT + TEP event supervision, V",
    "expanded_E":"SWaT + TEP event supervision, E",
    "official_V":"Official TimeRadar V",
    "official_E":"Official TimeRadar E",
    "official_V_plus_E":"Official weights, fixed V + E",
}
HEADER = "| Method | PSM | SMD28 | MSL27 | SMAP53 |"
RULE = "|---|---:|---:|---:|---:|"


def build(root):
    all_methods=pd.read_csv(root/"methods.csv")
    verify(root,all_methods)
    frame=all_methods[all_methods.panel=="common_offline"]
    methods=frame[["method","seed"]].drop_duplicates()
    lines=["# Fixed source-support comparison","",
           "Each cell gives AP / ROC-AUC / VUS-PR / VUS-ROC. Results for multi-entity datasets are equal-entity means. The three new methods use project seed 2025.","",
           "## Complete detectors on the common external axis","",HEADER,RULE]
    for name,seed in methods.itertuples(index=False,name=None):
        records=frame[(frame.method==name)&(frame.seed==seed)].set_index("domain")
        label=NAMES.get(name,name)
        if seed and name not in NAMES:
            label+=f" (seed {seed})"
        cells=[" / ".join(f"{records.loc[d,k]:.4f}" for k in METRICS) if d in records.index else "NA" for d in DOMAINS]
        lines.append("| "+" | ".join([label,*cells])+" |")
    lines.extend(["","All new scores become available at the window endpoint. Complete LEFT and DADA use full windows. TimeRCD FIT-only uses the whole input block for context. Normalization permissions, input hashes and timestamp intersections are fixed by the preceding external benchmark. PSM PCA uses the current observation. Its CATCH comparisons have target FIT training. The auxiliary historical-context and whole-QUERY normalization panels are preserved separately in methods.csv.","",
        "## Support and deployment statistic on the full causal axis","",
        "| Paired difference | PSM | SMD28 | MSL27 | SMAP53 |",RULE])
    dist=pd.read_csv(root/"paired_distribution.csv")
    for ref in ("single_V","expanded_E"):
        subset=dist[(dist.panel=="causal")&(dist.reference==ref)].set_index(["domain","metric"])
        cells=[" / ".join(f"{subset.loc[(d,k),'mean']:+.4f}" for k in METRICS) for d in DOMAINS]
        lines.append("| "+" | ".join(["Expanded V minus "+NAMES[ref],*cells])+" |")
    lines.extend(["","Each difference was computed within entity on identical points before aggregation. The two V models differ in the source of positive and paired-normal FIT windows. Both use the same combined source-validation panel. The V/E contrast compares training, score-specific selection and deployment together.","",
        "[All panels and full precision](methods.csv) · [Entity scores](entity_metrics.csv) · [Within-entity differences](entity_pairs.csv) · [Median differences and win/tie/loss counts](paired_distribution.csv) · [Operating costs](operating.csv)."])
    costs=json.loads((root/"EVALUATION.json").read_text())
    bank=pd.read_csv(root/"inference_cost.csv")
    lines.extend(["","The development domain is PSM. SMD supplied nominal source records. NASA uses sensor zero per entity. SWaT supplied labeled source events. PSM operating metrics use the separate 64,728-point suffix and its 512 attack-labeled CAL points. SMD uses its nominal CAL. The small NASA CAL sets have q99/higher equal to the sample maximum.","",
        "For expanded V relative to matched V, full-axis AP win/tie/loss counts were 14/0/14 on SMD, 11/0/16 on MSL and 29/3/21 on SMAP. SMAP showed the clearest ranking gain, with a higher three-confirmation false-positive rate. Expanded E achieved higher SMAP AP than expanded V, while frozen official E still had higher SMAP AP than both. TEP-specific improvement of E cannot be estimated from these three arms.","",
        f"Source training and validation used {costs['train_seconds']/3600:.3f} A40-hours. Per-entity causal readout used {costs['inference_seconds']/3600:.3f} A40-hours summed across lanes, including normal CAL, input checks and score writing. Sharing the frozen encoder and reusing byte-identical windows required {int(bank.unique_windows.sum()):,} distinct window computations across entities for the three decoders. There were {int(bank.query_points.sum()):,} QUERY endpoints and {int(bank.cal_points.sum()):,} CAL endpoints. CPU metric workers accumulated {costs['metric_worker_seconds']:.1f} seconds; final aggregation elapsed {costs['aggregation_wall_seconds']:.1f} seconds including waits for the last entity metrics. Data transfer and one-off engineering checks are outside the formal GPU times.","",
        "[Source file identities](INPUT.json) · [Sampling permissions and plan digest](PLAN.json) · [Realized exposures](EXPOSURES.json) · [Timing](EVALUATION.json) · [Numerical verification](RECHECK.json)."])
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def verify(root,summary):
    """Independently regroup the exported entities and reselect from source risks."""
    entities=pd.read_csv(root/"entity_metrics.csv")
    previous=pd.read_csv(root.parent/"native_endpoint_external_20261125"/"domain_metrics.csv")
    new=summary[summary.method.isin(["single_V","expanded_V","expanded_E"])]
    keys=["domain","panel","method","seed","timing"]
    prior=summary[~summary.method.isin(new.method.unique())]
    pd.testing.assert_frame_equal(previous.set_index(keys).sort_index(),prior.set_index(keys).sort_index(),
                                  check_like=True,atol=1e-14,rtol=0)
    maxima=[]
    for key,group in entities.groupby(keys):
        line=new.set_index(keys).loc[key]
        assert len(group)==line.entities and group.points.sum()==line.points
        assert not group.entity.duplicated().any()
        for k in METRICS:
            maxima.append(abs(sum(group[k])/len(group)-line[k]))
    pairs=pd.read_csv(root/"paired_distribution.csv")
    for (domain,panel),group in entities.groupby(["domain","panel"]):
        a=group[group.method=="expanded_V"].set_index("entity").sort_index()
        for ref in ("single_V","expanded_E"):
            b=group[group.method==ref].set_index("entity").sort_index()
            pd.testing.assert_frame_equal(a[["start","stop"]],b[["start","stop"]])
            for k in METRICS:
                ds=a[k].to_numpy()-b[k].to_numpy()
                row,=pairs[(pairs.domain==domain)&(pairs.panel==panel)&(pairs.reference==ref)&(pairs.metric==k)].itertuples()
                assert ((ds>0).sum(),(ds==0).sum(),(ds<0).sum())==(row.wins,row.ties,row.losses)
                maxima.extend([abs(ds.mean()-row.mean),abs(np.median(ds)-row.median)])
    selected={}
    for arm in new.method.unique():
        info=json.loads((root/arm/"SELECTION.json").read_text())
        select=info["selection"]
        assert [r["step"] for r in select]==list(range(0,241,40))
        risk=[(np.mean(r["swat"]["risks_event_by_stratum"])+np.mean(r["tep_by_fault"]))/2 for r in select]
        np.testing.assert_allclose(risk,[r["risk"] for r in select],rtol=2e-7,atol=1e-8)
        selected[arm]=select[np.argmin(risk)]["step"]
        assert selected[arm]==info["selected_step"]
    assert max(maxima)<1e-12
    receipt=dict(archived_external_rows=len(prior),new_entity_metric_rows=len(entities),
                 matched_selection=selected,max_aggregation_difference=max(maxima),
                 rank_check="AP and ROC raw-score tied-rank recheck in each new entity/panel; see EVALUATION.json",
                 VUS_check="Aggregation only; unchanged author implementation, no independent VUS algorithm")
    (root/"RECHECK.json").write_text(json.dumps(receipt,indent=2),encoding="utf8")


if __name__=="__main__":
    import sys
    build(Path(sys.argv[1]))
