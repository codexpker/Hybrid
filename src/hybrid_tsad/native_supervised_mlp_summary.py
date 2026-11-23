"""Join the fixed MLP benchmark with archived, compatible evidence criteria."""
from pathlib import Path
import pandas as pd

RANK=['AP','ROC_AUC','VUS_PR','VUS_ROC']
METHODS=['official_V','official_E','official_V_plus_E','copy_V','Vtrain_V','Etrain_E','MLP_full','MLP_240']
NAMES=dict(official_V='Official V',official_E='Official E',official_V_plus_E='Official V+E',copy_V='Reconstruction continuation / V',Vtrain_V='Source V supervision / V',Etrain_E='Source E supervision / E',MLP_full='Supervised MLP, full source selection',MLP_240='Supervised MLP, ≤240 selection')

def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])

def fmt(v):
    return 'NA' if pd.isna(v) else f'{v:.5f}'

def main():
    root=Path('docs/experiments/native_supervised_mlp_20261123')
    previous=Path('docs/experiments/native_e_vs_v_20261123')
    ranks=pd.concat([pd.read_csv(previous/'methods.csv'),pd.read_csv(root/'readout/ranking_macro.csv')],ignore_index=True)
    ops=pd.concat([pd.read_csv(previous/'operating.csv'),pd.read_csv(root/'readout/operating.csv')],ignore_index=True)
    ranks=ranks[ranks.method.isin(METHODS)]
    ops=ops[ops.method.isin(METHODS)]
    assert len(ranks)==32 and len(ops)==64
    ranks.to_csv(root/'methods.csv',index=False)
    ops.to_csv(root/'operating.csv',index=False)
    ix=ranks.set_index(['dataset','method']);oi=ops.set_index(['dataset','method','stream'])
    rows,oper,paired=[],[],[]
    for d in ['PSM','MSL','SMAP','SMD']:
        for m in METHODS:
            r=ix.loc[(d,m)]
            rows.append([d,NAMES[m],*[fmt(r[k]) for k in RANK]])
            for stream in ['strict','confirmed3']:
                r=oi.loc[(d,m,stream)]
                oper.append([d,NAMES[m],stream,fmt(r.normal_FPR),fmt(r.Recall),int(r.normal_alarm_fragments),f'{int(r.M)}/{int(r.Ntrue)}',int(r.auxiliary_any_hit_events),fmt(r.event_F1),fmt(r.long_steady_macro_coverage),fmt(r.long_steady_pooled_coverage),fmt(r.longest_event_miss),fmt(r.hit_only_median_delay)])
        for m in ['MLP_full','MLP_240']:
            for ref in ['Vtrain_V','Etrain_E','official_V']:
                paired.append(dict(dataset=d,method=m,reference=ref,**{k:ix.loc[(d,m),k]-ix.loc[(d,ref),k] for k in RANK}))
    pd.DataFrame(paired).to_csv(root/'paired.csv',index=False)
    out=['# Complete supervised-readout comparison, seed 2025','', 'PSM uses its complete causal test ranking axis; its operating panel uses the original 64,728-point suffix. Spacecraft and SMD ranking averages per-entity metrics equally. All operating counts preserve entity boundaries. The full and ≤240 MLP rows are source-selected across two fixed rates. The same heads serve all four targets.','',table(['Dataset','Method',*RANK],rows),'','## Source-selected MLP minus fixed controls','',table(['Dataset','Method','Reference',*RANK],[[r['dataset'],NAMES[r['method']],NAMES[r['reference']],*[fmt(r[k]) for k in RANK]] for r in paired]),'','## Fixed CAL operating points','',table(['Dataset','Method','Stream','Normal FPR','Recall','False fragments','Matched/eligible','Any-hit','Event F1','Long macro','Long pooled','Longest miss','Hit-only delay'],oper)]
    (root/'TABLES.md').write_text('\n'.join(out)+'\n',encoding='utf8')


def external_tables(root):
    previous=Path('docs/experiments/native_e_vs_v_20261123')
    out=['# External comparisons on their own common axes','', 'Each panel fixes its own common points. Native causal rows and offline/target-trained external rows retain their original information permissions. These panels must not be subtracted from full-axis main results.']
    psm=pd.read_csv(root/'readout/PSM/external_compatible.csv')
    out.extend(['','## PSM [512,87744), 87,232 points','',table(['Method',*RANK,'Training and readout permission'],[[r.method,*[fmt(r[k]) for k in RANK],r.permission] for _,r in psm.iterrows()])])
    old_psm=pd.read_csv(previous/'psm/external_axis.csv')
    new=pd.read_csv(root/'readout/common_macro.csv')
    for d,older in [('PSM',old_psm),('SMAP',pd.read_csv(previous/'space/SMAP/common_macro.csv')),('SMD',pd.read_csv(previous/'smd/left_common_macro.csv'))]:
        older=older.copy()
        older['method']=older['method'].replace({'frozen_native_V':'official_V','copy':'copy_V','point_weighted':'Vtrain_V'})
        selected=new[new.dataset==d]
        prior=older[older.method.isin(METHODS+['LEFT W192 endpoint','HistoryMean512','history_mean512','persistence','RoCA_FULL_T','LEFT'])]
        data=pd.concat([prior,selected],ignore_index=True)
        if d=='PSM':axis='[512,87841), 87,329 points'
        elif d=='SMAP':axis='[512,N) per sensor-zero entity, 400,481 points'
        else:axis='[99,192 floor(N/192)) per machine, 703,212 points'
        out.extend(['',f'## {d} {axis}','',table(['Method',*RANK],[[r.method,*[fmt(r[k]) for k in RANK]] for _,r in data.iterrows()])])
    out.extend(['','LEFT is target-normal trained. PSM native W192 LEFT and CATCH use the full offline window. The PSM endpoint LEFT row uses its project causal endpoint connection. SMD LEFT is offline whole-window; its public normal training permissions do not provide the project independent CAL. SMAP RoCA and simple baselines retain their archived permissions. MSL compatible entity-resolved external scores are missing.'])
    (root/'EXTERNAL.md').write_text('\n'.join(out)+'\n',encoding='utf8')

if __name__=='__main__':
    main()
    external_tables(Path('docs/experiments/native_supervised_mlp_20261123'))
