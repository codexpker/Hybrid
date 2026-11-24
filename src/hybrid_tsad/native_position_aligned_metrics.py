"""Entity-first rankings for the uniformly selected decoder training comparison."""
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import pandas as pd
from .native_position_aligned_readout import OUT,ARMS
from . import native_full_window_metrics as prior
from .native_v_conditioning_runner import read,dump,sha
from .left_native_swat_metrics import continuous_metrics,load_author_metric

METRICS=list(prior.METRICS)
NAMES=dict(copy='Normal reconstruction control',terminal='Endpoint-supervised control',all_positions='All-position supervised candidate',official_V='Official TimeRadar V',official_E='Official E',official_V_plus_E='Official fixed V+E',LEFT='Complete LEFT')


def entity(args):
    w,d,e=args;root=w/OUT/'readout';p=root/d/(e+'.npz');r=read(p.with_suffix('.json'));assert sha(p)==r['sha256']
    dest=w/OUT/'metrics_parts'/d/(e+'.json');dest.parent.mkdir(exist_ok=True,parents=True)
    if dest.exists():
        old=read(dest);assert old['sha256']==r['sha256'];return old
    # Reuse the latest legal joint axis and old fixed official/LEFT metric values.
    old=read(w/prior.OUT/'metrics_parts'/d/(e+'.json'))
    axisrow=next(x for x in old['rows'] if x['axis']=='common' and x['method']=='LEFT')
    axis=np.arange(axisrow['first'],axisrow['stop']);y=prior.truth(w,d,e)
    assert len(axis)==axisrow['points']
    with np.load(p) as z:
        np.testing.assert_array_equal(z['keys'],ARMS)
        ix=z['index'];score=z['V'];where=np.searchsorted(ix,axis)
        np.testing.assert_array_equal(ix,np.arange(len(y)//100*100));np.testing.assert_array_equal(ix[where],axis)
        np.testing.assert_array_equal(z['available_index'],(ix//100+1)*100-1)
    rows=[];metric=load_author_metric(w/prior.AUTHOR)
    reused_metrics={}
    old_keys=('official_V','2025_copy','2025_point_weighted','2026_copy','2026_point_weighted','2027_copy','2027_point_weighted',None)
    for j,arm in enumerate(ARMS):
        old_index=r['reused_old_columns'][j]
        old_method=old_keys[old_index] if old_index is not None else None
        for name,times,values in [('common',axis,score[where,j]),('native_full',ix,score[:,j])]:
            source_rows=[x for x in old['rows'] if x['method']==old_method and x['readout']=='full_window' and x['axis']==name]
            if source_rows:
                assert len(source_rows)==1 and source_rows[0]['first']==int(times[0]) and source_rows[0]['stop']==int(times[-1]+1)
                m={k:source_rows[0][k] for k in METRICS};reused_metrics[arm]=old_method
            else:
                m,_=continuous_metrics(y[times],values.astype(float),metric)
            rows.append(dict(domain=d,entity=e,method=arm,axis=name,first=int(times[0]),stop=int(times[-1]+1),points=len(times),**{k:m[k] for k in METRICS}))
    rows.extend({k:v for k,v in x.items() if k!='readout'} for x in old['rows'] if x['readout']=='full_window' and x['method'] in ['official_V','official_E','official_V_plus_E','LEFT'])
    result=dict(domain=d,entity=e,sha256=r['sha256'],reused_metric_columns=reused_metrics,axis_binding_sha256=sha(w/prior.OUT/'metrics_parts'/d/(e+'.json')),rows=rows)
    dump(dest,result);return result


def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','|'+'|'.join(['---']*len(head))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def number(v):return 'NA' if pd.isna(v) else f'{v:.5f}'


def summarize(w,results):
    root=w/OUT;frame=pd.DataFrame([r for result in results for r in result['rows']]);frame.to_csv(root/'entity_metrics.csv',index=False)
    macro=frame.groupby(['domain','axis','method'],sort=False)[METRICS].mean().reset_index();macro.to_csv(root/'macro.csv',index=False)
    common=frame[frame.axis=='common'].set_index(['domain','entity','method'])
    deltas=[];distributions=[]
    for d in ['PSM','SMD','MSL','SMAP']:
        a=common.loc[d].xs('all_positions',level='method')[METRICS]
        for ref in ['terminal','copy','official_V','LEFT']:
            b=common.loc[d].xs(ref,level='method')[METRICS];delta=a-b
            deltas.extend(dict(domain=d,entity=e,method='all_positions',reference=ref,**v.to_dict()) for e,v in delta.iterrows())
            for k in METRICS:
                v=delta[k];distributions.append(dict(domain=d,reference=ref,metric=k,macro_delta=v.mean(),median_delta=v.median(),wins=int((v>0).sum()),ties=int((v==0).sum()),losses=int((v<0).sum()),entities=len(v)))
    pd.DataFrame(deltas).to_csv(root/'paired_entity.csv',index=False);pd.DataFrame(distributions).to_csv(root/'paired_distribution.csv',index=False)
    view=macro[macro.axis=='common'].set_index(['domain','method']);rows=[]
    for d in ['PSM','SMD','MSL','SMAP']:
        for arm in NAMES:
            r=view.loc[(d,arm)];rows.append([d,NAMES[arm],*[number(r[k]) for k in METRICS]])
    text=['# Complete-window source-training comparison, seed2025','', 'All three decoder arms use the same official initialization, normal-window plan, 240-update cap and seven full-position SOURCE SELECT checkpoints. The reconstruction arm uses source-event labels for selection but not its training gradient. The supervised arms share the exact positive/normal windows and weights. Point-count weights represent window-sampling strata, not uniform abnormal time-point supervision.','',table(['Domain','Method',*METRICS],rows),'', 'All entries use the latest legal per-entity intersection of W100/stride100 outputs, archived causal coverage and LEFT scored positions. Every new method is an offline complete-window detector. No causal bank or binary threshold was generated. NASA uses sensor0 and entity-first macro averaging; SMD normal records participated in source training. LEFT is target-normal trained, with W96 MSL/W192 other datasets and native offline context. Each new arm has one project seed; no training SD is estimated.','', '## Candidate minus controls','']
    pairs=[]
    for d in ['PSM','SMD','MSL','SMAP']:
        for ref in ['terminal','copy','official_V','LEFT']:
            a=view.loc[(d,'all_positions')];b=view.loc[(d,ref)]
            pairs.append([d,NAMES[ref],*[number(a[k]-b[k]) for k in METRICS]])
    text.append(table(['Domain','Reference',*METRICS],pairs))
    text.extend(['','[Full entity metrics](entity_metrics.csv) · [entity differences](paired_entity.csv) · [wins/ties/losses and median differences](paired_distribution.csv).'])
    (root/'TABLES.md').write_text('\n'.join(text)+'\n',encoding='utf8')
    dump(root/'METRIC_BINDINGS.json',dict(entities=[{k:v for k,v in r.items() if k!='rows'} for r in results],metrics='pinned LEFT AP/ROC/VUS windowSize100 thre250',aggregation='per entity then equal domain mean',calibration='not performed',new_source_selection='uniform full-position risk across arms; original scale'))


def run(w):
    start=time.monotonic();jobs=prior.jobs(w);pending=set(jobs);active={};done=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        while pending or active:
            for d,e in sorted(pending):
                if len(active)>=6:break
                if (w/OUT/'readout'/d/(e+'.json')).exists():
                    f=pool.submit(entity,(w,d,e));active[f]=(d,e);pending.remove((d,e))
            if active:
                ready,_=wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for f in ready:
                    d,e=active.pop(f);done.append(f.result());print(d,e,len(done),len(jobs),flush=True)
            else:time.sleep(5)
    summarize(w,done);dump(w/OUT/'METRIC_COST.json',dict(wall_seconds=time.monotonic()-start,workers=6,GPU_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run']);p.parse_args();run(Path.cwd())
