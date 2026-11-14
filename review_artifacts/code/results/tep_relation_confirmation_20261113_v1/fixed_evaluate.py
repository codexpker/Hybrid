"""Fixed CPU confirmation, existing fitted Ridge; no training or model selection."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,pickle,time,hashlib,platform,datetime,warnings
import numpy as np,pandas as pd,sklearn
from sklearn.metrics import average_precision_score,roc_auc_score
P=Path(__file__).resolve().parent;started=time.monotonic();RAW=P/'raw';RAW.mkdir(exist_ok=False)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def dump(name,x):(P/name).write_text(json.dumps(x,indent=2,allow_nan=False))
protocol=json.loads((P/'FIXED_PROTOCOL.json').read_text());methods=['H','C','P0','MeanFIT'];scaler=np.load(P/'legacy/seed0/R_transforms.npz');source=np.load(P/'legacy/normal_R.npz');ym=float(scaler['target_mean']);ys=float(scaler['target_scale'])
with warnings.catch_warnings(record=True)as caught:
    warnings.simplefilter('always');models=[pickle.load((P/'legacy/seed0'/name).open('rb'))for name in ['R_M0.pkl','R_M1.pkl']]
dump('ENVIRONMENT.json',dict(Python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,legacy_sklearn='1.7.2',pickle_warnings=[str(w.message)for w in caught],numerical_model='stored coefficients/intercepts unchanged, sklearn prediction independently replayed',source_sha256=sha(__file__),protocol_sha256=sha(P/'FIXED_PROTOCOL.json'),readout_sha256=sha(P/'FIXED_READOUT.json'),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_hours=0))
def feat(a):
    idx=np.arange(12,len(a));return np.column_stack([a[idx[:,None]-np.arange(1,13),0],a[idx[:,None]-np.arange(1,13),1:].reshape(len(idx),-1)]),a[idx,0],idx+1
def predict(x):
    xx=(x-scaler['mean'])/scaler['scale'];out=[]
    for m,n in zip(models,[12,36]):
        p=m.predict(xx[:,:n]);replay=xx[:,:n]@m.coef_+m.intercept_;assert np.array_equal(p,replay);out.append(p*ys+ym)
    out.extend([x[:,0],np.full(len(x),ym)]);return np.stack(out,1)
sourcepred=predict(source['x']);roles={'FIT':source['run']<=60,'SELECT':(source['run']>=61)&(source['run']<=80),'CAL':source['run']>=81};assert set(source['run'])==set(range(1,101));assert np.array_equal(np.unique(source['sample']),np.arange(13,501))
radius=[];cal=roles['CAL'];rank=min(int(cal.sum())-1,int(np.ceil((int(cal.sum())+1)*.99))-1)
for i in range(4):radius.append(float(np.sort(np.abs(source['y'][cal]-sourcepred[cal,i]))[rank]))
radius=np.asarray(radius);assert(radius>0).all()
np.savez_compressed(RAW/'source_role_predictions.npz',y=source['y'],prediction=sourcepred,run=source['run'],sample=source['sample'],radius=radius,conformal_rank_index=rank,method_names=np.array(methods))
normalprediction=[]
for role,mask in roles.items():
    for run in np.unique(source['run'][mask]):
        q=mask&(source['run']==run)
        for j,name in enumerate(methods):
            err=sourcepred[q,j]-source['y'][q];normalprediction.append(dict(role=role,run=int(run),method=name,n=int(q.sum()),MAE=float(np.abs(err).mean()),RMSE=float(np.sqrt((err**2).mean())),radius=float(radius[j])))
pd.DataFrame(normalprediction).to_csv(P/'normal_source_prediction.csv',index=False)
def confirm(hit):return hit&np.r_[False,hit[:-1]]&np.r_[False,False,hit[:-2]]
def segments(mask):
    dif=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(dif==1),np.flatnonzero(dif==-1)))
def miss(hit):return max([b-a for a,b in segments(~hit)]+[0])
def score(y,p):
    lower=p-radius;upper=p+radius
    return np.maximum((p-y[:,None])/np.maximum(p-lower,1e-8),(y[:,None]-p)/np.maximum(upper-p,1e-8)),lower,upper
normal=np.load(P/'data/normal.npz');faulty=np.load(P/'data/faulty.npz');values=faulty['values'];ids=faulty['fault_run'];nv=normal['values'];nids=normal['fault_run'];assert values.shape==(400,960,3)and nv.shape==(20,960,3)
assert np.array_equal(ids,np.array([[f,r]for f in range(1,21)for r in range(201,221)]));assert np.array_equal(nids[:,1],np.arange(201,221))
# Scientific leak and feature-direction gates use a normal sequence, before TEST scoring.
a=nv[0].copy();x,y,sample=feat(a);changed=a.copy();changed[100:,0]+=1000;x2,_,_=feat(changed);assert np.array_equal(x[88],x2[88]);changed=a.copy();changed[100:,1:]-=1000;x2,_,_=feat(changed);assert np.array_equal(x[88],x2[88]);assert np.array_equal(x[0,:12],a[np.arange(11,-1,-1),0]);assert np.array_equal(x[0,12:],a[np.arange(11,-1,-1),1:].reshape(-1));assert np.array_equal(sample,np.arange(13,961))
prefix=[]
for run in range(201,221):
    pv=values[ids[:,1]==run,:160,:];same=np.array([np.array_equal(p,pv[0])for p in pv]);longnormal=nv[nids[:,1]==run, :160,:][0]
    prefix.append(dict(run=run,fault_prefix_all20_exact=bool(same.all()),fault_prefix_reference_sha256=hashlib.sha256(pv[0].tobytes()).hexdigest(),unique_fault_prefix_hashes=len({hashlib.sha256(p.tobytes()).hexdigest()for p in pv}),fault_vs_normal_same_number_exact=[bool(np.array_equal(p,longnormal))for p in pv],normal_prefix_sha256=hashlib.sha256(longnormal.tobytes()).hexdigest()))
dump('PREFIX_INDEPENDENCE.json',prefix)
dump('CPU_GATES.json',dict(status='PASS',target_current_future_perturbation_no_effect=True,condition_current_future_perturbation_no_effect=True,feature_lag_orientation_exact=True,original_model_prediction_coefficient_replay_exact=True,CAL_runs81_100_only=True,radius=radius.tolist(),rank=rank,raw_prediction_plan_frozen=True,source_models_unchanged=True))
banks={};prefixgroups={}
for role,vv,ii in [('normal',nv,nids),('faulty',values,ids)]:
    preds=[];truth=[];scores=[];lo=[];hi=[]
    for a,(fault,run)in zip(vv,ii):
        x,y,sample=feat(a);p=predict(x);s,l,u=score(y,p);preds.append(p);truth.append(y);scores.append(s);lo.append(l);hi.append(u)
        if role=='faulty':
            h=hashlib.sha256(a[:160].tobytes()).hexdigest();prefixgroups.setdefault(h,[]).append((int(fault),int(run)))
    path=RAW/(role+'_scores.npz');np.savez_compressed(path,prediction=np.stack(preds),y=np.stack(truth),score=np.stack(scores),lower=np.stack(lo),upper=np.stack(hi),radius=radius,method_names=np.array(methods),fault_run=ii,sample=sample)
    banks[role]=dict(path=str(path),sha256=sha(path),series=len(vv))
dump('RAW_SEAL.json',dict(status='ALL_CONTINUOUS_SCORES_SEALED_BEFORE_METRICS',banks=banks,source_sha256=sha(__file__),protocol_sha256=sha(P/'FIXED_PROTOCOL.json'),timestamp_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),labels='fault/sample metadata visible to extraction and protocol, no use for fitting or score calibration; numerical detection metrics start after this seal',GPU_hours=0))
faultrows=[];normalrows=[];prefixrows=[];baselineprefix={};prevalence=800/948
for role in ['normal','faulty']:
    z=np.load(RAW/(role+'_scores.npz'));sample=z['sample'];gt=sample>=161
    for k,(fault,run)in enumerate(z['fault_run']):
        for j,name in enumerate(methods):
            s=z['score'][k,:,j];p=z['prediction'][k,:,j];y=z['y'][k];hit=s>1;c3=confirm(hit);err=p-y
            if role=='normal':
                normalrows.append(dict(run=int(run),method=name,points=len(s),MAE=float(np.abs(err).mean()),RMSE=float(np.sqrt((err**2).mean())),FPR=float(hit.mean()),C3_FPR=float(c3.mean()),false_points=int(hit.sum()),C3_false_points=int(c3.sum()),false_fragments=len(segments(hit)),C3_false_fragments=len(segments(c3)),C3_false_fragments_per_hour=len(segments(c3))/(len(s)*3/60),normal_hours=len(s)*3/60))
            else:
                cp=c3[gt];ix=np.flatnonzero(cp);delay=int(sample[gt][ix[0]]-161)if len(ix)else None
                row=dict(run=int(run),fault=int(fault),method=name,related=int(fault)in[4,11,14],AP=float(average_precision_score(gt,s)),ROC=float(roc_auc_score(gt,s)),prevalence=prevalence,AP_over_prevalence=float(average_precision_score(gt,s))/prevalence,strict_recall=float(hit[gt].mean()),C3_recall=float(cp.mean()),C3_event_hit=bool(cp.any()),strict_event_hit=bool(hit[gt].any()),steady_C3_coverage=float(c3[sample>=561].mean()),longest_C3_miss_points=miss(cp),longest_C3_miss_minutes=miss(cp)*3,first_C3_delay_points=delay,first_C3_delay_minutes=delay*3 if delay is not None else None,censored_C3_delay_points=delay if delay is not None else 800,censored_C3_delay_minutes=delay*3 if delay is not None else 2400,delay_right_censored=delay is None,prefix_C3_active_at_last_normal_point=bool(c3[sample==160][0]),prefix_MAE=float(np.abs(err[~gt]).mean()),prefix_RMSE=float(np.sqrt((err[~gt]**2).mean())),prefix_FPR=float(hit[~gt].mean()),prefix_C3_FPR=float(c3[~gt].mean()))
                faultrows.append(row)
                h=hashlib.sha256(values[k,:160].tobytes()).hexdigest();key=(h,name)
                if key not in baselineprefix:
                    sh=hit[~gt];sc=c3[~gt];baselineprefix[key]=dict(prefix_sha256=h,representative_fault=int(fault),representative_run=int(run),run_ids=sorted({rr for ff,rr in prefixgroups[h]}),multiplicity=len(prefixgroups[h]),method=name,points=int((~gt).sum()),FPR=float(sh.mean()),C3_FPR=float(sc.mean()),C3_false_fragments=len(segments(sc)),C3_false_fragments_per_hour=len(segments(sc))/((~gt).sum()*3/60),MAE=float(np.abs(err[~gt]).mean()),RMSE=float(np.sqrt((err[~gt]**2).mean())))
                else:
                    prior=baselineprefix[key];assert abs(prior['FPR']-float(hit[~gt].mean()))<1e-15
df=pd.DataFrame(faultrows);nd=pd.DataFrame(normalrows);pdf=pd.DataFrame(baselineprefix.values());df.to_csv(P/'per_fault_run.csv',index=False);nd.to_csv(P/'long_normal_per_run.csv',index=False);pdf.to_csv(P/'normal_prefix_unique.csv',index=False)
fields=['AP','ROC','AP_over_prevalence','strict_recall','C3_recall','C3_event_hit','steady_C3_coverage','longest_C3_miss_points','censored_C3_delay_minutes','prefix_FPR','prefix_C3_FPR'];scope_frames=[]
scopes={'all_faults':df,'related_4_11_14':df[df.related],**{'fault_'+str(f):df[df.fault==f]for f in range(1,21)}}
for scope,d in scopes.items():
    t=d.groupby('method')[fields].mean().reset_index();t.insert(0,'scope',scope);scope_frames.append(t)
pd.concat(scope_frames,ignore_index=True).to_csv(P/'summary.csv',index=False)
nd.groupby('method').mean(numeric_only=True).to_csv(P/'long_normal_summary.csv');pdf.groupby('method').mean(numeric_only=True).to_csv(P/'prefix_unique_summary.csv')
# Reuse identical sampled run clusters for every fixed contrast and metric.
rng=np.random.default_rng(84551);draw=rng.integers(0,20,(2000,20));np.save(RAW/'bootstrap_run_indices.npy',draw);stats=[]
for scope in ['all_faults','related_4_11_14','fault_4','fault_11','fault_14']:
    d=scopes[scope]
    for left,right in [('C','H'),('C','P0'),('C','MeanFIT')]:
        for field in fields:
            mat=d.groupby(['run','method'])[field].mean().unstack().reindex(range(201,221));delta=(mat[left]-mat[right]).to_numpy(float);boot=delta[draw].mean(1);stats.append(dict(scope=scope,contrast=left+'-'+right,metric=field,mean_difference=float(delta.mean()),run_delta_sd=float(delta.std(ddof=1)),delta_min=float(delta.min()),delta_max=float(delta.max()),CI95_low=float(np.quantile(boot,.025)),CI95_high=float(np.quantile(boot,.975)),clusters=20,bootstrap=2000))
        # Delay among jointly detected events is a separate selected subset;
        # each resampled run carries all its original fault pairs.
        wide=d.pivot(index=['run','fault'],columns='method',values='first_C3_delay_minutes');paired=wide[[left,right]].dropna();counts=paired.groupby(level=0).size().reindex(range(201,221),fill_value=0).to_numpy();sums=(paired[left]-paired[right]).groupby(level=0).sum().reindex(range(201,221),fill_value=0).to_numpy();bs=sums[draw].sum(1);bn=counts[draw].sum(1);ok=bn>0;dv=float((paired[left]-paired[right]).mean())if len(paired)else None;stats.append(dict(scope=scope,contrast=left+'-'+right,metric='common_hit_delay_minutes',mean_difference=dv,CI95_low=float(np.quantile(bs[ok]/bn[ok],.025))if ok.any()else None,CI95_high=float(np.quantile(bs[ok]/bn[ok],.975))if ok.any()else None,matched_events=len(paired),clusters_with_match=int((counts>0).sum()),conditional_on_common_detection=True))
for field in ['MAE','RMSE','FPR','C3_FPR','C3_false_fragments_per_hour']:
    mat=nd.pivot(index='run',columns='method',values=field).reindex(range(201,221))
    for baseline in ['H','P0','MeanFIT']:
        delta=(mat.C-mat[baseline]).to_numpy();boot=delta[draw].mean(1);stats.append(dict(scope='long_normal',contrast='C-'+baseline,metric=field,mean_difference=float(delta.mean()),CI95_low=float(np.quantile(boot,.025)),CI95_high=float(np.quantile(boot,.975)),clusters=20,bootstrap=2000))
pd.DataFrame(stats).to_csv(P/'paired_statistics.csv',index=False)
dump('EVALUATION_RECEIPT.json',dict(status='COMPLETED',fault_rows=len(df),long_normal_rows=len(nd),unique_prefix_series=len(prefixgroups),unique_prefix_method_rows=len(pdf),CPU_seconds=time.monotonic()-started,GPU_hours=0,no_refit=True,source_models_hash_unchanged=True,statistical_clusters=20,all_run_ids=list(range(201,221)),all_faults=list(range(1,21))))
dump('FINAL_MANIFEST.json',[dict(path=str(f.relative_to(P)),sha256=sha(f),bytes=f.stat().st_size)for f in P.rglob('*')if f.is_file()and f.name!='FINAL_MANIFEST.json'and'__pycache__'not in f.parts])
print(json.dumps(dict(status='STAGE_COMPLETED',seconds=time.monotonic()-started,related=pd.read_csv(P/'summary.csv').query("scope=='related_4_11_14'").to_dict('records'),long_normal=nd.groupby('method').mean(numeric_only=True).reset_index().to_dict('records')),allow_nan=False))

