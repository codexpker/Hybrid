"""Public method panels and fixed objective/readout contrasts, without model execution."""
from pathlib import Path
import pandas as pd

ROOT=Path('docs/experiments/left_nasa_author_20261123')
E=Path('docs/experiments/native_e_vs_v_20261123')
M=Path('docs/experiments/native_supervised_mlp_20261123')
RANK=['AP','ROC_AUC','VUS_PR','VUS_ROC']
NAMES={'official_V':'Official TimeRadar V','official_E':'Official TimeRadar E','official_V_plus_E':'Official fixed V+E','copy_V':'Reconstruction continuation / V','Vtrain_V':'Source-supervised native V (candidate)','Etrain_E':'Source-supervised E / E','MLP_full':'Supervised MLP','LEFT':'LEFT'}

def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def fmt(v):return 'NA' if pd.isna(v) else f'{v:.5f}'

def build():
    nasa=pd.read_csv(ROOT/'macro.csv').query("axis=='common'").rename(columns={'domain':'dataset'})
    psm=pd.read_csv(M/'readout/PSM/external_compatible.csv')
    psm=psm[psm.method.isin(NAMES)].copy()
    left=pd.read_csv(M/'readout/PSM/external_compatible.csv').query("method=='LEFT native W192'").copy();left['method']='LEFT'
    smd=pd.read_csv(E/'smd/left_common_macro.csv')
    mlp=pd.read_csv(M/'readout/common_macro.csv').query("dataset=='SMD'")
    data=pd.concat([psm,left,smd,mlp,nasa],ignore_index=True)
    data=data[data.method.isin(NAMES)][['dataset','method',*RANK]]
    assert len(data)==32 and not data.duplicated(['dataset','method']).any()
    data.to_csv(ROOT/'common_methods.csv',index=False)
    ix=data.set_index(['dataset','method'])
    out=['# Complete public method comparison','', 'All entries are local measurements. PSM uses [512,87744), 87,232 points; SMD uses [99,192 floor(N/192)) within each machine; NASA uses [99,W floor(N/W)) within each entity, W96 for MSL and W192 for SMAP. Entity means are unweighted within NASA and SMD. These common points align outcomes while preserving distinct source supervision, target-normal training and offline context permissions. Each row below uses seed2025; separate repetition estimates follow.']
    for metric in RANK:
        out.extend(['',f'## {metric}','',table(['Method','PSM','SMD28','MSL27','SMAP53'],[[name,*[fmt(ix.loc[(d,k),metric]) for d in ['PSM','SMD','MSL','SMAP']]] for k,name in NAMES.items()])])
    out.extend(['','The official, reconstruction-continuation and source-supervised candidates use W100 endpoint observations. The project-trained heads and decoders use real SWaT source labels for selection; the supervised arms also use them in training. Official scores have no project updates. The MLP freezes the representation; the V/E arms update its native decoder. LEFT uses domain/entity normal target training and full-window offline scoring. The project candidate has no target-gradient updates in this panel. NASA has sensor zero only. LEFT independent-CAL operation is NA.'])
    (ROOT/'PUBLIC_TABLES.md').write_text('\n'.join(out)+'\n',encoding='utf8')
    full=pd.concat([pd.read_csv(E/'methods.csv'),pd.read_csv(M/'readout/ranking_macro.csv')],ignore_index=True)
    full=full[~full.method.isin(['MLP_240'])]
    full.to_csv(ROOT/'full_axis_ablation.csv',index=False)
    fi=full.set_index(['dataset','method'])
    ab=['# Objective and readout comparison on complete causal axes','', 'V/V and E/E change the supervised decoder objective and deployed evidence; cross-readouts are predetermined descriptive controls. The MLP changes objective, updated component and inference together. Source labels and original FIT/SELECT windows are fixed, but the selected exposure and two-rate development budget differ. All rows retain their original full causal axes. This panel is separate from the LEFT intersections.']
    for metric in RANK:
        rows=[]
        for method in ['copy_V','Vtrain_V','Etrain_E','Vtrain_E','Etrain_V','MLP_full']:
            rows.append([method,*[fmt(fi.loc[(d,method),metric]) for d in ['PSM','SMD','MSL','SMAP']]])
        ab.extend(['',f'## {metric}','',table(['Training / readout','PSM','SMD28','MSL27','SMAP53'],rows)])
    (ROOT/'ABLATIONS.md').write_text('\n'.join(ab)+'\n',encoding='utf8')

    repeated=pd.read_csv('docs/experiments/native_event_rank_repeats_20261122/ranking_methods.csv')
    original=[]
    for _,r in repeated[repeated.method.isin(['copy','point_weighted'])].iterrows():
        original.append([r.dataset,r.method,*[f"{r[k+'_mean']:.5f} ± {r[k+'_sd']:.5f}" for k in RANK]])
    sm=pd.read_csv('docs/experiments/native_event_smd28_20261123/native/ranking_macro.csv')
    for family in ['copy','point_weighted']:
        block=sm[sm['key'].isin([f'{seed}_{family}' for seed in [2025,2026,2027]])]
        assert len(block)==3
        original.append(['SMD',family,*[f'{block[k].mean():.5f} ± {block[k].std(ddof=1):.5f}' for k in RANK]])

    text=['# Project-training repetition and same-axis NASA uncertainty','', 'Each SD is the sample SD across three project-training seeds sharing the official checkpoint (n=3, ddof=1). NASA entities and SMD machines are not extra training repetitions. LEFT has one seed and no SD.','', '## Original complete causal axes','',table(['Domain','Training',*RANK],original),'', 'SMD complete-axis repetition is retained in [the original macro table](../native_event_smd28_20261123/native/ranking_macro.csv) and [paired-repeat summary](../native_event_smd28_20261123/native/pairs_summary.csv). These full axes differ from LEFT common axes.','', '## Recomputed NASA LEFT common axes','']
    rr=pd.read_csv(ROOT/'repeats_common.csv')
    text.append(table(['Domain','Training',*RANK],[[r.domain,r.method,*[f"{r[k+'_mean']:.5f} ± {r[k+'_sd']:.5f}" for k in RANK]] for _,r in rr.iterrows()]))
    (ROOT/'REPEATS.md').write_text('\n'.join(text)+'\n',encoding='utf8')


if __name__=='__main__':build()
