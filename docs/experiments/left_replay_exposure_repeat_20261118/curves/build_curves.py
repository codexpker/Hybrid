"""CPU-only cached-score explanation; no model import, inference or training."""
from pathlib import Path
import csv, hashlib, json, time
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workspace',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
W=args.workspace.resolve()
OUT=args.output.resolve()
OUT.mkdir(parents=True,exist_ok=False)
R=W/'results' 
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
inputs={}
def bind(p, expected=None):
    h=sha(p)
    if expected is not None: assert h==expected,(p,h,expected)
    inputs[str(p.relative_to(W))]=h
    return p
def csvout(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

started=time.perf_counter()
task=read(bind(R/'left_current_u_exposure_20261118_v1/SWAT/TASK.json'))
seal=read(bind(R/'left_current_u_exposure_20261118_v1/GLOBAL_RAW_BANK_SEAL.json'))
old=read(bind(R/'left_author_swat_source_replay_metrics_20261118_v1/METRICS.json'))
current=read(bind(R/'left_current_u_exposure_metrics_20261118_v1/METRICS.json'))
events=read(bind(R/'left_current_u_exposure_metrics_20261118_v1/EVENTS.json'))
old_events=read(bind(R/'left_author_swat_source_replay_metrics_20261118_v1/EVENTS.json'))
labels=np.load(bind(W/task['labels']['relative'],task['labels']['sha256']))['label']
qaxis=task['score_axes']['QUERY'];caxis=task['score_axes']['CAL']
index=np.arange(*qaxis); assert len(index)==336321
y=labels[index]; select=np.linspace(caxis[0],caxis[1]-1,512,dtype=np.int64)
names=[a+'_'+c for a in ['TT','R'] for c in ['A','B','C']]
lookup={(x['method'],x['condition'],x['stream']):x for x in current['rows'] if x['domain']=='SWAT'}
long_events=sorted([e for e in events if e.get('domain')=='SWAT' and e['method']=='TT' and e['condition']=='B' and e['stream']=='strict' and e['eligible'] and e['total_points']>512],key=lambda e:e['start'])
assert len(long_events)==9 and all(e['complete'] for e in long_events)
curves={};ops=[];event_rows=[];summaries=[]
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
fig,(pr,roc)=plt.subplots(1,2,figsize=(12,5))
colors={'A':'#386cb0','B':'#d95f02','C':'#1b9e77'}
for name in names:
    arm,cond=name.split('_'); banks={}
    for role,axis in [('CAL',caxis),('QUERY',qaxis)]:
        if cond=='A':
            p=R/f'left_author_swat_source_replay_20261118_v1/{arm}/{role}.npz'
            expected=old['replay']['arms'][arm]['score_sha256'][role]
        else:
            p=R/f'left_current_u_exposure_20261118_v1/SWAT/{name}/{role}.npz'
            expected=next(z['sha256'] for z in seal['banks'] if z['relative']==f'SWAT/{name}/{role}.npz')
        with np.load(bind(p,expected)) as z:
            ix=np.arange(*axis);s=z['canonical_score'].copy()
            assert np.array_equal(z['absolute_time_index'],ix) and z['valid'].all()
            assert np.array_equal(s,z['per_channel'].mean(-1)) and np.isfinite(s).all()
            avail=495000+(ix-495000)//192*192+191
            assert np.array_equal(z['available_index'],avail)
            banks[role]=s
    cal=banks['CAL'][select-caxis[0]]; tau=float(np.quantile(cal,.99,method='higher'))
    row=lookup[arm,cond,'strict'];c3row=lookup[arm,cond,'confirmed3']
    assert tau==row['threshold'] and int((cal>tau).sum())==row['CAL_exceedances']
    s=banks['QUERY'];a=s>tau;c3=np.zeros_like(a);c3[2:]=a[2:]&a[1:-1]&a[:-2]
    tp=int((a&(y==1)).sum());fp=int((a&(y==0)).sum())
    assert tp==row['TP'] and fp==row['FP']
    assert int((c3&(y==1)).sum())==c3row['TP'] and int((c3&(y==0)).sum())==c3row['FP']
    precision,recall,pt=precision_recall_curve(y,s)
    fpr,tpr,rt=roc_curve(y,s,drop_intermediate=False)
    ap=float(average_precision_score(y,s));auc=float(roc_auc_score(y,s))
    assert abs(ap-row['AP'])<1e-12 and abs(auc-row['ROC_AUC'])<1e-12
    op=dict(model=name,threshold=tau,CAL_points=512,CAL_exceedances=int((cal>tau).sum()),AP=ap,ROC_AUC=auc,
            QUERY_points=len(y),normal_points=int((y==0).sum()),attack_points=int((y==1).sum()),
            strict_FP=fp,strict_TP=tp,strict_FPR=fp/int((y==0).sum()),strict_recall=tp/int((y==1).sum()),
            strict_precision=tp/int(a.sum()),C3_FP=c3row['FP'],C3_recall=c3row['Recall'])
    ops.append(op)
    style='-' if arm=='TT' else '--';marker='o' if arm=='TT' else '^'
    kw=dict(color=colors[cond],linestyle=style,linewidth=1.5)
    pr.plot(recall,precision,label=f'{name}: AP {ap:.3f}',**kw)
    roc.plot(fpr,tpr,label=f'{name}: AUC {auc:.3f}',**kw)
    pr.scatter(op['strict_recall'],op['strict_precision'],marker=marker,facecolors='white',edgecolors=colors[cond],s=55,zorder=5)
    roc.scatter(op['strict_FPR'],op['strict_recall'],marker=marker,facecolors='white',edgecolors=colors[cond],s=55,zorder=5)
    for key,val in dict(precision=precision,recall=recall,PR_threshold=pt,FPR=fpr,TPR=tpr,ROC_threshold=rt).items():curves[name+'_'+key]=val
    np.savetxt(OUT/(name+'_PR.csv'),np.column_stack([recall,precision,np.r_[pt,np.nan]]),delimiter=',',header='recall,precision,score_threshold_nan_at_terminal',comments='')
    np.savetxt(OUT/(name+'_ROC.csv'),np.column_stack([fpr,tpr,rt]),delimiter=',',header='FPR,TPR,score_threshold',comments='')
    evsource=old_events if cond=='A' else events
    for stream,bits in [('strict',a),('confirmed3',c3)]:
        subset=[e for e in evsource if e['method']==arm and e['stream']==stream and (cond=='A' or (e.get('domain')=='SWAT' and e.get('condition')==cond))]
        records=[]
        for e in long_events:
            cached=next(z for z in subset if z['event_id']==e['event_id'])
            lo,hi=e['start'],e['end_exclusive'];den=hi-lo-512
            full=bits[lo-qaxis[0]:hi-qaxis[0]];steady=bits[lo+512-qaxis[0]:hi-qaxis[0]]
            assert cached['eligible'] and len(steady)==den
            assert abs(float(steady.mean())-cached['steady_after512_coverage'])<1e-12
            assert abs(float(full.mean())-cached['coverage'])<1e-12
            record=dict(model=name,stream=stream,event_id=e['event_id'],start=lo,end_exclusive=hi,
                        full_denominator=hi-lo,steady_start=lo+512,steady_denominator=den,
                        full_alarm_points=int(full.sum()),full_coverage=float(full.mean()),
                        steady_alarm_points=int(steady.sum()),steady_coverage=float(steady.mean()),
                        complete=True,boundary_censored=cached['boundary_censored'],
                        steady_longest_miss_points=cached['steady_longest_miss_points'])
            records.append(record);event_rows.append(record)
        macro=float(np.mean([x['steady_coverage'] for x in records]));den=sum(x['steady_denominator'] for x in records)
        pooled=sum(x['steady_alarm_points'] for x in records)/den
        accepted=lookup[arm,cond,stream]
        assert den==accepted['long_steady_total_points'] and abs(macro-accepted['long_steady_macro_coverage'])<1e-12
        assert abs(pooled-accepted['long_steady_point_coverage'])<1e-12
        summaries.append(dict(model=name,stream=stream,long_events=9,steady_denominator=den,steady_macro_coverage=macro,steady_pooled_coverage=pooled))

pr.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1.02),title='Full precision–recall curves')
roc.set(xlabel='Normal false-positive rate',ylabel='Recall / true-positive rate',xlim=(0,.05),ylim=(0,1.02),title='ROC restricted to normal FPR 0–5%')
for ax in (pr,roc):
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left');ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1))
fig.suptitle('SWaT seed2025 | TT/R × A/B/C | native retrospective cached scores')
fig.text(.5,.01,'Open circles = TT; open triangles = R. Markers use each frozen CAL q99/higher threshold (>). No curve-selected threshold.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.035,1,.95));save(fig,'PR_lowFPR_ROC')
csvout('operating_points.csv',ops);csvout('long_events_all_streams.csv',event_rows);csvout('macro_pooled_summary.csv',summaries)
np.savez_compressed(OUT/'CURVE_ARRAYS.npz',**curves,CAL_indices=select,QUERY_indices=index)

matrix=np.array([[next(x['steady_coverage'] for x in event_rows if x['event_id']==e['event_id'] and x['model']==n and x['stream']=='confirmed3') for n in names] for e in long_events])
den=np.array([e['total_points']-512 for e in long_events]);weights=den/den.sum()
fig,(heat,bar)=plt.subplots(1,2,figsize=(12,6),gridspec_kw={'width_ratios':[2.5,1.4]})
im=heat.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
heat.set_xticks(range(6),names);heat.set_yticks(range(9),[e['event_id'] for e in long_events]);heat.set_title('All nine long events: C3 steady coverage')
for i in range(9):
    for j in range(6):heat.text(j,i,f'{matrix[i,j]*100:.1f}%',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>.55 else 'black')
bar.barh(range(9),den,color='#636363');bar.set_yticks(range(9),[e['event_id'] for e in long_events]);bar.invert_yaxis();bar.set_xscale('log');bar.set_xlim(10,max(den)*2.5)
bar.set_title('Steady denominator / pooled weight');bar.set_xlabel('Observed points (log scale)')
for i,d in enumerate(den):bar.text(d*1.05,i,f'{d:,} / {weights[i]*100:.1f}%',va='center',fontsize=8)
fig.colorbar(im,ax=heat,fraction=.035,pad=.02,format=PercentFormatter(1))
fig.suptitle('Length >512; steady starts at original event start +512 | no event selection')
fig.tight_layout();save(fig,'long_event_coverage_denominators')
fig,ax=plt.subplots(figsize=(9,4));xs=np.arange(6)
sm=[next(z for z in summaries if z['model']==n and z['stream']=='confirmed3') for n in names]
ax.bar(xs-.18,[z['steady_macro_coverage'] for z in sm],.36,label='Macro: each event weight 1/9',color='#756bb1')
ax.bar(xs+.18,[z['steady_pooled_coverage'] for z in sm],.36,label='Pooled: weight by steady points',color='#31a354')
ax.set_xticks(xs,names);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('C3 steady coverage');ax.legend();ax.set_title('Same nine long events: macro versus pooled');fig.tight_layout();save(fig,'macro_vs_pooled')
result=dict(status='CPU_CACHE_CHECKS_PASS',input_sha256=inputs,query_axis=qaxis,query_points=len(index),CAL_axis=caxis,CAL_points=512,
            threshold_rule='np.quantile(q=.99,method=higher); score strictly greater; no curve selection',
            nine_long_events=[dict(event_id=e['event_id'],start=e['start'],end_exclusive=e['end_exclusive'],steady_denominator=e['total_points']-512) for e in long_events],
            event_comparisons='Recomputed strict/C3 coverage agrees exactly with accepted event records and macro/pooled aggregates; event matching NOT recomputed',
            mode='offline full-window, available at window end; not causal delay',cpu_seconds=time.perf_counter()-started)
(OUT/'CHECKS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(ops=ops,summary=summaries,denominators=den.tolist(),cpu_seconds=result['cpu_seconds']),indent=2))
