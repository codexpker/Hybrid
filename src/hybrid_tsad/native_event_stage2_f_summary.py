"""Fixed within-initialization and post-F contrasts for the single-event support study."""
import argparse
from pathlib import Path
import pandas as pd

RANK = ['AP','ROC_AUC','VUS_PR','VUS_ROC']
OPS = ['normal_FPR','Recall','normal_alarm_fragments','M','Ntrue','auxiliary_any_hit_events','event_F1','long_steady_macro_coverage','long_steady_pooled_coverage','longest_event_miss','hit_only_median_delay']
NAMES = {'official':'Official initialization','copy':'Reconstruction continuation','point_weighted':'Source V supervision'}

def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])

def fmt(v):
    return 'NA' if pd.isna(v) else f'{v:.5f}'

def summarize(root):
    frame=pd.read_csv(root/'methods.csv')
    lookup=frame.set_index(['domain','initialization','phase','stream'])
    assert lookup.index.is_unique and len(frame)==24
    ranking,operations,paired=[],[],[]
    for domain in ['PSM','machine-3-2']:
        for initial in NAMES:
            for phase in ['Z','F']:
                strict=lookup.loc[(domain,initial,phase,'strict')]
                ranking.append([domain,NAMES[initial],phase,*[fmt(strict[k]) for k in RANK]])
                for stream in ['strict','confirmed3']:
                    r=lookup.loc[(domain,initial,phase,stream)]
                    operations.append([domain,NAMES[initial],phase,stream,fmt(r.normal_FPR),fmt(r.Recall),int(r.normal_alarm_fragments),f'{int(r.M)}/{int(r.Ntrue)}',int(r.auxiliary_any_hit_events),fmt(r.event_F1),fmt(r.long_steady_macro_coverage),fmt(r.long_steady_pooled_coverage),fmt(r.longest_event_miss),fmt(r.hit_only_median_delay)])
        contrasts=[(i,'F',i,'Z') for i in NAMES]+[('point_weighted','F',i,'F') for i in ['copy','official']]
        for a,phase,b,control in contrasts:
            for stream in ['strict','confirmed3']:
                left,right=[lookup.loc[(domain,i,p,stream)] for i,p in [(a,phase),(b,control)]]
                assert left.scored_points==right.scored_points and left.Ntrue==right.Ntrue
                paired.append(dict(domain=domain,initialization=a,phase=phase,reference=b,reference_phase=control,stream=stream,**{k:left[k]-right[k] for k in RANK+OPS}))
    pd.DataFrame(paired).to_csv(root/'paired.csv',index=False)
    text=['# Fixed single-event F comparison','',table(['Task','Initialization','Phase','AP','ROC-AUC','VUS-PR','VUS-ROC'],ranking),'','## Paired changes','',table(['Task','Contrast',*RANK],[[r['domain'],r['initialization']+' '+r['phase']+' minus '+r['reference']+' '+r['reference_phase'],*[fmt(r[k]) for k in RANK]] for r in paired if r['stream']=='strict']),'','## Existing CAL operating points','',table(['Task','Initialization','Phase','Stream','Normal FPR','Recall','False fragments','Matched/eligible','Any-hit','Event F1','Long macro','Long pooled','Longest miss','Hit-only delay'],operations)]
    (root/'TABLES.md').write_text('\n'.join(text)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);summarize(p.parse_args().root)
