"""Readable offline, matched readout and native-axis tables from sealed rankings."""
from pathlib import Path
import pandas as pd

OUT=Path('results/native_full_window_20261124_v1')
METRICS=['AP','ROC_AUC','VUS_PR','VUS_ROC']


def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])


def number(v):return 'NA' if pd.isna(v) else f'{v:.5f}'
def uncertainty(row,key):return f'{row[key+"_mean"]:.5f} ± {row[key+"_sd"]:.5f}'


def build(root):
    macro=pd.read_csv(root/'macro.csv');repeats=pd.read_csv(root/'repeats_common.csv');delta=pd.read_csv(root/'paired_distribution.csv');paired=pd.read_csv(root/'paired_seed_summary.csv')
    view=macro.set_index(['domain','readout','axis','method'])
    rep=repeats.set_index(['domain','readout','family'])
    report=['# Table 1. Fixed-weight complete-window offline comparison','', 'All methods in this table use identical scored points within each entity: the intersection of W100 complete-window outputs, archived causal endpoints and native LEFT coverage. PSM begins at99; SMD and NASA also begin at99 in each independent record. The last point is min(100 floor(N/100), W_LEFT floor(N/W_LEFT))−1. LEFT W192 applies to PSM, SMD and SMAP; MSL uses W96. Every TimeRadar full-window score is available at the containing W100 endpoint. Complete-window output is a project readout of fixed checkpoints. The three-seed mean and sample SD concern project source training at a common official initialization (n=3, ddof=1). Single frozen/E/LEFT models have no invented SD.','']
    rows=[]
    for d in ['PSM','SMD','MSL','SMAP']:
        for name,method in [('Official TimeRadar V','official_V'),('Official E','official_E'),('Official fixed V+E','official_V_plus_E'),('Reconstruction continuation / V','copy'),('Source-supervised native V','point_weighted'),('Source-supervised E / E','Etrain_E'),('LEFT','LEFT')]:
            if method in ('copy','point_weighted'):
                r=rep.loc[d,'full_window',method];values=[uncertainty(r,k) for k in METRICS];n=3
            else:
                r=view.loc[d,'full_window','common',method];values=[number(r[k]) for k in METRICS];n=1
            rows.append([d,name,n,*values])
    report.append(table(['Domain','Method','Project seeds',*METRICS],rows))
    report.extend(['','Official TimeRadar uses no project parameter updates. Reconstruction continuation is selected by real source-event validation but receives only nominal reconstruction gradients. The native-V/E supervised decoders use nominal source reconstruction plus real SWaT source pairs. LEFT uses the full target-normal records with its archived scaler/validation permissions and offline context. Current TimeRadar input scalers and mask identities remain fixed. NASA uses sensor zero only. SMD normal training records were among the project sources. MLP remains in its prior causal panel; no window logit was broadcast. No CAL, thresholds or online latency are evaluated in this readout experiment.','', '[Seed-specific macro values](macro.csv) · [complete per-entity rankings](entity_metrics.csv) · [same-axis repetitions](repeats_common.csv).'])
    (root/'TABLE_OFFLINE.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    report=['# Table 2. Same-checkpoint endpoint and complete-window comparison','', 'Every difference uses the identical per-entity common axis defined in Table1. Full-window minus endpoint measures a readout/context change at fixed parameters. It is separate from differences between training objectives. No result is selected by target performance.','', '## Same weights, different readout','']
    rows=[]
    for d in ['PSM','SMD','MSL','SMAP']:
        for family,name in [('copy','Reconstruction continuation'),('point_weighted','Source-supervised native V')]:
            row=paired[(paired.domain==d)&(paired.readout=='full_minus_endpoint')&(paired.reference==family)].iloc[0]
            a=rep.loc[d,'causal_endpoint',family];b=rep.loc[d,'full_window',family]
            rows.append([d,name,uncertainty(a,'AP'),uncertainty(b,'AP'),*[uncertainty(row,k) for k in METRICS]])
        for method,name in [('official_V','Official V'),('official_E','Official E'),('official_V_plus_E','Official V+E'),('Etrain_E','E-supervised E')]:
            a=view.loc[d,'causal_endpoint','common',method];b=view.loc[d,'full_window','common',method]
            rows.append([d,name,number(a.AP),number(b.AP),*[number(b[k]-a[k]) for k in METRICS]])
    report.append(table(['Domain','Weights / score','Endpoint AP','Complete-window AP',*[f'Δ {k}' for k in METRICS]],rows))
    report.extend(['','## Source supervision within each fixed readout','', 'Differences are candidate minus official V or same-seed reconstruction continuation. Each value is the mean ± sample SD of three paired project-seed differences. Entity macro averaging precedes seed aggregation.'])
    rows=[]
    for _,r in paired[paired.readout.isin(['full_window','causal_endpoint'])].iterrows():
        rows.append([r.domain,r.readout,r.reference,*[uncertainty(r,k) for k in METRICS]])
    report.append(table(['Domain','Candidate readout','Reference',*METRICS],rows))
    report.extend(['','The LEFT comparison always retains LEFT’s native offline readout. Seed-wise and per-entity differences, counts of wins/ties/losses and medians are in [paired_distribution.csv](paired_distribution.csv). The underlying differences are in [paired_entity.csv](paired_entity.csv).'])
    report.extend(['','## Entity distribution of the full-window candidate-minus-LEFT AP','', 'Counts compare each fixed training seed with LEFT on the exact per-entity joint axis; entities are not extra training repetitions.'])
    chosen=delta[(delta.readout=='full_window')&(delta.reference=='LEFT')&(delta.metric=='AP')]
    rows=[[r.domain,r.method,number(r.macro_delta),number(r.median_delta),f'{int(r.wins)}/{int(r.ties)}/{int(r.losses)}'] for _,r in chosen.iterrows()]
    report.append(table(['Domain','Candidate seed','Macro AP delta','Median AP delta','Wins/ties/losses'],rows))
    (root/'TABLE_READOUT.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    native=macro[macro.axis=='native_full'].copy();native.to_csv(root/'native_axis_macro.csv',index=False)
    rows=[[r.domain,r.method,*[number(r[k]) for k in METRICS]] for _,r in native.iterrows()]
    (root/'TABLE_NATIVE_AXES.md').write_text('# Native full axes\n\nTimeRadar uses [0,100 floor(N/100)); LEFT uses [0,W_LEFT floor(N/W_LEFT)). These axes differ. No cross-axis differences are calculated here. The common-axis comparison is Table1.\n\n'+table(['Domain','Method',*METRICS],rows)+'\n',encoding='utf8')

if __name__=='__main__':build(Path.cwd()/OUT)
