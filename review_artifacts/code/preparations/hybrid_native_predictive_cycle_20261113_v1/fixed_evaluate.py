"""Frozen common-axis evaluation; no model loading, fitting or score selection.

TEP reuses the existing H2 raw predictions. All methods receive new shared
CAL-q99higher operating points; old conformal reports remain untouched.
"""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name
sys.path.insert(0,str(R/'hybrid_ad/src'))
from hybrid_tsad.alarm_complementarity import confirmed,longest_miss,point_counts
from hybrid_tsad.engine.events import binary_events,greedy_assign

def load(f):
    with np.load(f,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text(encoding='utf8'))
def dump(f,d):
    with Path(f).open('x',encoding='utf8') as h:json.dump(d,h,indent=2,allow_nan=False)
def table(f,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(f).open('x',newline='',encoding='utf8') as h:
        w=csv.DictWriter(h,keys);w.writeheader();w.writerows(rows)
def ratio(a,b):return float(a/b) if b else None
def axis(z,expected):
    ix=z['index'];assert ix.ndim==2 and np.array_equal(ix,np.broadcast_to(expected,ix.shape))
    assert np.isfinite(z['raw_score']).all();return ix
def identity(z):
    rr=z['run'];ff=z['fault'];assert rr.ndim==ff.ndim==2
    assert np.all(rr==rr[:,:1]) and np.all(ff==ff[:,:1]);return np.column_stack([ff[:,0],rr[:,0]]).astype(int)
def model_bank(job,role,key='raw_score'):
    f=O/job/'score'/(role+'.npz');z=load(f);return z,np.asarray(z[key],float),dict(path=str(f),sha256=sha(f),field=key)
def normal_head_diagnostics(domain,jobs):
    rows=[]
    for job in jobs:
        for role in ['CAL']+(['NORMAL_TEST']if domain=='TEP'else[]):
            z=load(O/job/'score'/(role+'.npz'));y=z['target_standardized'];cols=[0]if domain=='TEP'else list(range(25))
            for head,key in [('time','time_prediction_standardized'),('frequency','freq_prediction_standardized'),('multiscale','MS_prediction_standardized')]:
                err=z[key][...,cols]-y[...,cols]
                rows.append(dict(domain=domain,job=job,role=role,head=head,MAE_FIT_standardized=float(np.abs(err).mean()),RMSE_FIT_standardized=float(np.sqrt(np.square(err).mean())),interpretation='current-observed reconstruction'if job.endswith('_native')else'strict-past terminal prediction',selection='diagnostic only; no head/weight/epoch selection'))
    return rows
def canonical_residual(y,p,r):
    low=p-r;upper=p+r
    return np.maximum((p-y)/np.maximum(p-low,1e-8),(y-p)/np.maximum(upper-p,1e-8))

def seal_tep():
    out=O/'evaluation'/'TEP';out.mkdir(parents=True,exist_ok=False);B=out/'banks';B.mkdir()
    expected={'CAL':np.arange(101,501),'QUERY':np.arange(101,961),'NORMAL_TEST':np.arange(101,961)}
    expected_ids={'CAL':np.column_stack([np.zeros(20,int),np.arange(81,101)]),'QUERY':np.array([[f,r]for f in range(1,21)for r in range(201,221)]),'NORMAL_TEST':np.column_stack([np.zeros(20,int),np.arange(201,221)])}
    entries={};sources=[];target=None
    source=load(R/'results/tep_relation_confirmation_20261113_v1/raw/source_role_predictions.npz')
    cmask=(source['run']>=81)&(source['sample']>=101)
    expected_y={'CAL':source['y'][cmask].reshape(20,400)}
    for role,f in [('QUERY','faulty'),('NORMAL_TEST','normal')]:
        actual=load(R/'results/tep_relation_confirmation_20261113_v1/data'/(f+'.npz'))
        assert np.array_equal(actual['fault_run'],expected_ids[role])
        expected_y[role]=actual['values'][:,100:,0]
    for name,job,key in [('LEFT_native','TEP_native','raw_score'),('Hybrid_full','TEP_full','raw_score'),('Hybrid_self_only','TEP_history_only','raw_score'),('Hybrid_nativeoff','TEP_full','nativeoff_score')]:
        e={}
        for role in expected:
            z,s,meta=model_bank(job,role,key);axis(z,expected[role]);assert np.array_equal(identity(z),expected_ids[role]);assert s.shape==z['index'].shape
            assert np.array_equal(z['target_y'],expected_y[role]),'actual target9 raw truth misalignment: '+job+'/'+role
            e[role]=s;sources.append(meta)
            if role=='QUERY':
                if target is None:target=z['target_y'].copy()
                else:assert np.array_equal(target,z['target_y'])
        entries[name]=e
    old=R/'results/tep_relation_confirmation_20261113_v1/raw';source=load(old/'source_role_predictions.npz');mask=(source['run']>=81)&(source['sample']>=101);assert mask.sum()==8000
    for j,name in enumerate(['Ridge_H','Ridge_C','P0','MeanFIT']):
        r=source['radius'][j];s=canonical_residual(source['y'][mask],source['prediction'][mask,j],r).reshape(20,400)
        assert np.array_equal(source['run'][mask].reshape(20,400)[:,0],np.arange(81,101));e={'CAL':s}
        for role,file in [('QUERY','faulty_scores.npz'),('NORMAL_TEST','normal_scores.npz')]:
            z=load(old/file);keep=z['sample']>=101;assert np.array_equal(z['sample'][keep],expected[role]);assert np.array_equal(z['fault_run'],expected_ids[role]);e[role]=z['score'][:,keep,j]
            if role=='QUERY':assert np.array_equal(z['y'][:,keep],target)
        entries[name]=e
    for f in ['source_role_predictions.npz','faulty_scores.npz','normal_scores.npz']:sources.append(dict(path=str(old/f),sha256=sha(old/f)))
    fusion={};scales={}
    for name in ['LEFT_native','Ridge_C']:
        cal=entries[name]['CAL'];med=float(np.median(cal));mad=max(float(np.median(np.abs(cal-med))),1e-6);scales[name]=dict(median=med,MAD=mad,floor=1e-6)
        for role in expected:fusion[role]=fusion.get(role,0)+.5*np.maximum(0,(entries[name][role]-med)/mad)
    entries['Simple_native_RidgeC_half']=fusion
    descriptors=[]
    for name,e in entries.items():
        tau=float(np.quantile(e['CAL'],.99,method='higher'));f=B/(name+'.npz')
        np.savez_compressed(f,**e,CAL_index=expected['CAL'],QUERY_index=expected['QUERY'],NORMAL_TEST_index=expected['NORMAL_TEST'],QUERY_ids=expected_ids['QUERY'],NORMAL_TEST_ids=expected_ids['NORMAL_TEST'])
        descriptors.append(dict(method=name,path=str(f),sha256=sha(f),threshold=tau))
    dump(out/'metric_definition.json',dict(primary='target9 canonical score',CAL='all8000 normal81..100 sample101..500; q99higher strict>',fault_boundary=161,steady_samples=[561,960],common_axis=[101,960],event_end='observed fault interval ends960; recovery right-censored, known onset161 retained; no relabel/drop',event_PRF='one-to-one fault-suffix alarm fragments plus independent20 normal-run fragments once per scope. Extra fragments within one true event penalize fragmentation, not physical normality. Fault prefixes reported separately and not added again.',C3='consecutive3 original samples; no backfill; reset per run, confirm before slicing fault suffix',delay='sample-first minus161; hit-only plus 800point right-censor code; times3minutes',normal='20 FaultFreeTesting runs; fault prefixes hash-deduplicated separately',statistics='20run paired cluster bootstrap seed84551 n2000; each draw retains allfaults. Single trainingseed, not independent domains.',fusion='fixed .5 positive medianMAD native and RidgeC; only same CAL; ownq99',fusion_scales=scales,source_mode='old Ridge weights/raw; no refitting or reuse of old-query-selected thresholds'))
    table(out/'normal_head_diagnostics.csv',normal_head_diagnostics('TEP',['TEP_native','TEP_full','TEP_history_only']))
    dump(out/'RAW_SEAL.json',dict(status='SEALED_BEFORE_NEW_METRICS',methods=descriptors,sources=sources,source_sha256=sha(__file__),protocol_sha256=sha(P/'FIXED_EXPERIMENT.json')))
    return out

def tep_rows(sample,s,gt):
    hit=s;inside=sample>=161;steady=sample>=561;prefix=~inside;h=hit[inside];first=np.flatnonzero(h);delay=int(sample[inside][first[0]]-161)if len(first)else None
    spans=binary_events(h);assign=greedy_assign(spans,[('fault',0,len(h))]);matched=sum(x is not None for x in assign)
    return dict(M=matched,Npred_fault=len(spans),Ntrue=1,anomaly_points=int(inside.sum()),detected_anomaly_points=int(h.sum()),point_coverage=float(h.mean()),steady_coverage=float(hit[steady].mean()),max_miss=longest_miss(h),delay_points=delay,delay_minutes=delay*3 if delay is not None else None,censored_delay_points=delay if delay is not None else 800,delay_right_censored=delay is None,active_at160=bool(hit[sample==160][0]),prefix_FP=int(hit[prefix].sum()),prefix_points=int(prefix.sum()),prefix_FPR=float(hit[prefix].mean()),prefix_fragments=len(binary_events(hit[prefix])))

def evaluate_tep():
    out=O/'evaluation/TEP';seal=read(out/'RAW_SEAL.json');F=[];N=[];prefix=[];prediction=[]
    originals=load(R/'results/tep_relation_confirmation_20261113_v1/data/faulty.npz');prefix_hash=[hashlib.sha256(v[:160].tobytes()).hexdigest()for v in originals['values']]
    for d in seal['methods']:
        assert sha(d['path'])==d['sha256'];z=load(d['path']);tau=d['threshold'];method=d['method'];sample=z['QUERY_index'];gt=sample>=161;seen={}
        for i,(fault,run)in enumerate(z['QUERY_ids']):
            score=z['QUERY'][i];ap=float(average_precision_score(gt,score));roc=float(roc_auc_score(gt,score));hit=score>tau
            for stream,alarm in [('strict',hit),('confirmed3',confirmed(sample,hit))]:
                row=dict(method=method,fault=int(fault),run=int(run),related=int(fault)in[4,11,14],stream=stream,AP=ap,ROC=roc,prevalence=float(gt.mean()),AP_over_prevalence=ap/float(gt.mean()),**tep_rows(sample,alarm,gt));F.append(row)
                key=(prefix_hash[i],stream)
                if key not in seen:
                    seen[key]=True;prefix.append(dict(method=method,run=int(run),representative_fault=int(fault),stream=stream,prefix_sha256=prefix_hash[i],points=row['prefix_points'],FP=row['prefix_FP'],FPR=row['prefix_FPR'],fragments=row['prefix_fragments']))
        for i,(_,run)in enumerate(z['NORMAL_TEST_ids']):
            ns=z['NORMAL_TEST_index'];hit=z['NORMAL_TEST'][i]>tau
            for stream,alarm in [('strict',hit),('confirmed3',confirmed(ns,hit))]:
                n=len(alarm);frag=len(binary_events(alarm));N.append(dict(method=method,run=int(run),stream=stream,points=n,FP=int(alarm.sum()),FPR=float(alarm.mean()),fragments=frag,fragments_per_hour=frag/(n*3/60)))
    table(out/'per_fault_run.csv',F);table(out/'normal_per_run.csv',N);table(out/'normal_prefix_deduplicated.csv',prefix)
    import pandas as pd
    df=pd.DataFrame(F);nd=pd.DataFrame(N);scopes={'all_faults':list(range(1,21)),'related_4_11_14':[4,11,14],**{'fault_'+str(f):[f]for f in range(1,21)}};summ=[];perrun=[]
    for scope,faults in scopes.items():
        for method in df.method.unique():
            for stream in ['strict','confirmed3']:
                x=df[(df.method==method)&(df.stream==stream)&df.fault.isin(faults)];n=nd[(nd.method==method)&(nd.stream==stream)]
                def aggregate(x,n):
                    m=int(x.M.sum());npred=int(x.Npred_fault.sum()+n.fragments.sum());nt=len(x);delays=x.delay_points.dropna()
                    tp=int(x.detected_anomaly_points.sum());fp=int(n.FP.sum());fn=int(x.anomaly_points.sum())-tp
                    return dict(AP=float(x.AP.mean()),ROC=float(x.ROC.mean()),prevalence=float(x.prevalence.mean()),AP_over_prevalence=float(x.AP_over_prevalence.mean()),point_coverage=float(tp/x.anomaly_points.sum()),point_P=ratio(tp,tp+fp),point_F1=ratio(2*tp,2*tp+fp+fn),point_TP=tp,point_FN=fn,event_macro_coverage=float(x.point_coverage.mean()),steady_macro_coverage=float(x.steady_coverage.mean()),M=m,Npred_fault=int(x.Npred_fault.sum()),Npred_independent_normal=int(n.fragments.sum()),event_P=ratio(m,npred),event_R=ratio(m,nt),event_F1=ratio(2*m,npred+nt),normal_FP=fp,normal_FPR=float(fp/n.points.sum()),normal_fragments=int(n.fragments.sum()),normal_fragments_per_hour=float(n.fragments.sum()/(n.points.sum()*3/60)),max_event_miss=int(x.max_miss.max()),mean_longest_miss=float(x.max_miss.mean()),mean_censored_delay_points=float(x.censored_delay_points.mean()),hit_only_delay_median=float(delays.median())if len(delays)else None,right_censored_events=int(x.delay_right_censored.sum()))
                summ.append(dict(scope=scope,method=method,stream=stream,**aggregate(x,n)))
                for run in range(201,221):perrun.append(dict(scope=scope,method=method,stream=stream,run=run,**aggregate(x[x.run==run],n[n.run==run])))
    table(out/'summary.csv',summ);table(out/'paired_run_aggregates.csv',perrun)
    rng=np.random.default_rng(84551);draw=rng.integers(0,20,(2000,20));np.save(out/'bootstrap_run_indices.npy',draw);pdf=pd.DataFrame(perrun);stats=[]
    fields=['AP','ROC','point_coverage','event_macro_coverage','steady_macro_coverage','event_P','event_R','event_F1','normal_FPR','normal_fragments_per_hour','mean_longest_miss','mean_censored_delay_points']
    for scope in ['all_faults','related_4_11_14','fault_4','fault_11','fault_14']:
        for stream in ['strict','confirmed3']:
            x=pdf[(pdf.scope==scope)&(pdf.stream==stream)]
            for baseline in [v for v in df.method.unique()if v!='Hybrid_full']:
                for field in fields:
                    mat=x.pivot(index='run',columns='method',values=field).reindex(range(201,221));delta=(mat['Hybrid_full']-mat[baseline]).to_numpy(float);valid=np.isfinite(delta)
                    if not valid.all():stats.append(dict(scope=scope,stream=stream,contrast='Hybrid_full-'+baseline,metric=field,status='NA_UNDEFINED_RUN_METRIC'));continue
                    bs=delta[draw].mean(1);stats.append(dict(scope=scope,stream=stream,contrast='Hybrid_full-'+baseline,metric=field,status='DESCRIPTIVE_PAIRED_RUN_CLUSTER',delta_mean=float(delta.mean()),delta_sd=float(delta.std(ddof=1)),delta_min=float(delta.min()),delta_max=float(delta.max()),CI95_low=float(np.quantile(bs,.025)),CI95_high=float(np.quantile(bs,.975)),clusters=20,replicates=2000))
    table(out/'paired_CI.csv',stats);dump(out/'EVALUATION_RECEIPT.json',dict(status='COMPLETED',methods=len(seal['methods']),fault_rows=len(F),normal_rows=len(N),prefix_rows=len(prefix),metric_source_sha256=sha(__file__),raw_seal_sha256=sha(out/'RAW_SEAL.json'),no_training=True,no_QUERY_parameter_selection=True))

def seal_psm():
    out=O/'evaluation/PSM';out.mkdir(parents=True,exist_ok=False);B=out/'banks';B.mkdir();entries={};sources=[]
    ci=np.arange(106720,132481);qi=np.arange(512,87841);selected=np.linspace(106720,132480,512,dtype=np.int64)
    raw={r:load(P/'data'/('PSM_'+r+'.npz')) for r in ['CAL','QUERY']}
    for role in raw:
        assert list(raw[role]['values'].shape)==([1,25953,25]if role=='CAL'else[1,87841,25])
        assert np.array_equal(raw[role]['original_start'],[106528]if role=='CAL'else[0])
    expected_y={r:raw[r]['values'][:,(ci-106528)if r=='CAL'else qi,0]for r in raw}
    for name,job,key in [('LEFT_native','PSM_native','raw_score'),('Hybrid_full','PSM_full','raw_score'),('Hybrid_nativeoff','PSM_full','nativeoff_score')]:
        e={}
        for role,ix in [('CAL',ci),('QUERY',qi)]:
            z,s,meta=model_bank(job,role,key);axis(z,ix);assert np.array_equal(z['target_y'],expected_y[role]);assert s.shape==(1,len(ix))
            assert np.array_equal(z['target_all_original'],raw[role]['values'][:,ix-(106528 if role=='CAL'else 0)]),'raw25channel alignment differs'
            assert np.allclose(s,z['channel_score'].mean(-1),atol=0,rtol=0)if key=='raw_score'else np.allclose(s,z['nativeoff_channel_score'].mean(-1),atol=0,rtol=0)
            e[role]=s[0];sources.append(meta)
        entries[name]=e
    sc=read(P/'PSM_FIT_ONLY_SCALER.json');scale=np.asarray(sc['scale']);mean=np.asarray(sc['mean']);assert sc['shape']==[1,79488,25]
    for name in ['P0','MeanFIT']:
        e={}
        for role,ix,offset in [('CAL',ci,106528),('QUERY',qi,0)]:
            v=raw[role]['values'][0];loc=ix-offset;pred=v[loc-1]if name=='P0'else mean
            e[role]=np.mean(np.abs(v[loc]-pred)/scale,axis=1)
        entries[name]=e
    descriptors=[]
    for name,e in entries.items():
        tau=float(np.quantile(e['CAL'][selected-ci[0]],.99,method='higher'));f=B/(name+'.npz');np.savez_compressed(f,**e,CAL_index=ci,QUERY_index=qi,selected_CAL=selected)
        descriptors.append(dict(method=name,path=str(f),sha256=sha(f),threshold=tau))
    dump(out/'metric_definition.json',dict(primary='native mean25 score; target-arrival endpoint only; no future window readout',CAL='independent physical CAL106528..132480;512 equally spaced mature endpoints106720..132480; q99higher',FIT=[0,79488],SELECT=[80032,105984],QUERY=[512,87841],external_common_axis=[512,87744],event_metrics='existing evaluation._coverage/main_eligible and chronological one-to-one matching; incomplete and boundary events retained with exclusions, not relabelled normal',C3='3causal consecutive points; no backfill',steady='after512samples from original event onset; both eventmacro and pooledpoints',simple='P0 andMeanFIT absolute residuals standardized by same newFITscale, mean25; no training/fusion',statistics='single anonymous series; no20run CI, cross-domain confirmation INCOMPLETE',legacy='old PSM LEFT/pretest scaler/SELECT-CALreuse is separate historical permission, excluded from matched claims'))
    table(out/'normal_head_diagnostics.csv',normal_head_diagnostics('PSM',['PSM_native','PSM_full']))
    dump(out/'RAW_SEAL.json',dict(status='SEALED_BEFORE_NEW_METRICS',methods=descriptors,sources=sources,source_sha256=sha(__file__),protocol_sha256=sha(P/'FIXED_EXPERIMENT.json')))

def evaluate_psm():
    from hybrid_tsad.hybrid_v1 import evaluation as k
    out=O/'evaluation/PSM';seal=read(out/'RAW_SEAL.json');cfg=read(R/'preparations/hybrid_left_joint_cycle_20261109_v2/EVALUATION.ACTUAL.json');labelpath=Path(str(cfg['truth']['path']).replace('__RESEARCH_ROOT__',str(R)));truth=load(labelpath)['full_labels'].astype(np.int8);assert truth.shape==(87841,)
    rows=[];events_out=[];alarms_out=[];coverage_rows=[]
    for d in seal['methods']:
        assert sha(d['path'])==d['sha256'];z=load(d['path']);idx=z['QUERY_index'];assert np.array_equal(idx,np.arange(512,87841));hit=z['QUERY']>d['threshold']
        for scope,valid_axis in [('full',idx),('external_common',idx[idx<87744])]:
            keep=idx<87744 if scope=='external_common'else np.ones(len(idx),bool);ix=idx[keep];score=z['QUERY'][keep];scored=np.zeros(len(truth),bool);scored[ix]=True;events=k._coverage(truth,scored,0)
            if d['method']=='LEFT_native':coverage_rows.extend([dict(scope=scope,**e)for e in events])
            for stream,mask in [('strict',hit[keep]),('confirmed3',confirmed(idx,hit)[keep])]:
                full=np.zeros(len(truth),bool);full[ix]=mask;ev,_=k._event_view(truth,scored,full,np.arange(len(truth)),events,0,'main_eligible');norm,_=k._normal(truth,scored,full,0);cov=[];steady=[];stp=stn=0;miss=[];delays=[]
                elook={e['event_id']:e for e in ev['events']}
                for e in events:
                    lo,hi=e['local_start'],e['local_end'];valid=bool(e['main_eligible']);bits=full[lo:hi];info=elook[e['event_id']];er=dict(scope=scope,method=d['method'],stream=stream,**e,**{a:b for a,b in info.items()if a not in ['event_id']},detected_points=int(bits.sum()),observed_point_coverage=float(bits.mean()),longest_miss_points=longest_miss(bits),steady_coverage=None)
                    if valid:
                        cov.append(float(bits.mean()));miss.append(longest_miss(bits))
                        if info['delay_points']is not None:delays.append(info['delay_points'])
                        if hi-lo>512:
                            v=full[lo+512:hi];steady.append(float(v.mean()));stp+=int(v.sum());stn+=len(v);er['steady_coverage']=float(v.mean())
                    events_out.append(er)
                alarms_out.extend([dict(scope=scope,method=d['method'],stream=stream,**a)for a in ev['alarms']])
                rows.append(dict(scope=scope,method=d['method'],stream=stream,AP=float(average_precision_score(truth[ix],score)),ROC=float(roc_auc_score(truth[ix],score)),threshold=d['threshold'],**point_counts(truth,scored,full),**ev['metrics'],**{a:b for a,b in norm.items()if not isinstance(b,list)},event_macro_coverage=float(np.mean(cov))if cov else None,steady_macro_coverage=float(np.mean(steady))if steady else None,steady_pooled_coverage=ratio(stp,stn),steady_points=stn,long_event_count=len(steady),max_event_miss=max(miss)if miss else None,hit_only_delay_median=float(np.median(delays))if delays else None,uncertainty='INCOMPLETE_SINGLE_SERIES'))
    table(out/'summary.csv',rows);table(out/'per_event.csv',events_out);table(out/'alarms.csv',alarms_out);table(out/'event_coverage.csv',coverage_rows)
    dump(out/'EVALUATION_RECEIPT.json',dict(status='COMPLETED',methods=len(seal['methods']),rows=len(rows),label_path=str(labelpath),label_sha256=sha(labelpath),raw_seal_sha256=sha(out/'RAW_SEAL.json'),no_training=True,no_QUERY_parameter_selection=True))

def cpu_tests():
    s=np.arange(101,961);h=np.zeros(860,bool);h[(s>=161)&(s<=170)]=True;c=confirmed(s,h);r=tep_rows(s,c,s>=161);assert r['M']==1 and r['Npred_fault']==1 and r['detected_anomaly_points']==8 and r['delay_points']==2
    h[(s>=200)&(s<=205)]=True;r=tep_rows(s,confirmed(s,h),s>=161);assert r['M']==1 and r['Npred_fault']==2
    h=np.zeros(860,bool);h[s>=159]=True;r=tep_rows(s,confirmed(s,h),s>=161);assert r['delay_points']==0 and not r['active_at160']
    r=tep_rows(s,np.zeros(860,bool),s>=161);assert r['delay_right_censored']and r['max_miss']==800 and r['censored_delay_points']==800
    assert not confirmed(np.array([1,2,4]),np.ones(3,bool)).any();print('CPU_EVALUATION_TESTS_PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['test','seal-tep','evaluate-tep','seal-psm','evaluate-psm']);a=ap.parse_args()
    {'test':cpu_tests,'seal-tep':seal_tep,'evaluate-tep':evaluate_tep,'seal-psm':seal_psm,'evaluate-psm':evaluate_psm}[a.phase]()
