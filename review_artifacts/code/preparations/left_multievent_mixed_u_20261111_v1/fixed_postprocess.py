"""Fixed five-method SWaT evaluation and offline native-loss exposure accounting."""
import csv,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name
def read(f):return json.loads(Path(f).read_text(encoding='utf-8'))
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def load(f):
    with np.load(f)as z:return{k:z[k].copy()for k in z.files}
def dump(f,d):
    with Path(f).open('x',encoding='utf-8')as out:json.dump(d,out,indent=2,allow_nan=False)
def table(f,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(f).open('x',newline='',encoding='utf-8')as out:w=csv.DictWriter(out,fieldnames=keys);w.writeheader();w.writerows(rows)
s=importlib.util.spec_from_file_location('alarm',R/'hybrid_ad/src/hybrid_tsad/alarm_complementarity.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
def seal():
    start=time.monotonic();assert(P/'ALL_U_RAW_COMPLETE.json').exists();S=O/'fixed_scores';S.mkdir(exist_ok=False);selected=np.linspace(446173,494999,512,dtype=np.int64);sourcecal=load(O/'left_score/CAL.npz');source_tau=float(np.quantile(sourcecal['score'][selected-446173],.99,method='higher'));refs=[('noU',O/'left_score')]+[('U_'+c,O/'U'/c/'score')for c in 'ABC']+[('HistoryMean512',O/'HistoryMean512')];descriptors=[]
    for name,folder in refs:
        q=load(folder/'QUERY.npz');cal=load(folder/'CAL.npz');assert np.array_equal(q['index'],np.arange(608535,944919))and np.array_equal(cal['index'],np.arange(446173,495000));tau=float(np.quantile(cal['score'][selected-446173],.99,method='higher'));f=S/(name+'.npz');d=dict(index=q['index'],score=q['score'],CAL_index=cal['index'],CAL_score=cal['score'],selected_CAL=selected)
        tracks={'own_CAL_q99':tau}
        if name!='HistoryMean512':tracks['fixed_noU_CAL_q99']=source_tau
        for track,t in tracks.items():d[track+'__strict']=q['score']>t;d[track+'__confirmed3']=a.confirmed(q['index'],q['score']>t)
        np.savez_compressed(f,**d);descriptors.append(dict(method=name,path=str(f),sha256=sha(f),thresholds=tracks,raw_QUERY_path=str(folder/'QUERY.npz'),raw_QUERY_sha256=sha(folder/'QUERY.npz'),raw_CAL_path=str(folder/'CAL.npz'),raw_CAL_sha256=sha(folder/'CAL.npz')))
    dump(S/'ALL_SCORES_THRESHOLDS_MASKS_SEALED.json',dict(status='SEALED_BEFORE_NEW_MODEL_SCORE_LABEL_EVALUATION',descriptors=descriptors,rule_sha256=sha(P/'FIXED_EXPERIMENT.json'),noU_threshold=source_tau,QUERY_labels_previously_visible_for_metadata_only=True,no_score_label_join_for_design=True,seconds=time.monotonic()-start))
    print(json.dumps(dict(status='SEALED',methods=len(descriptors),noU_threshold=source_tau)))
def normal_and_exposure():
    start=time.monotonic();E=O/'normal_and_exposure';E.mkdir(exist_ok=False);plan=load(P/'LEARNER_PLAN.npz');mask=plan['replacement_mask'];assert mask.shape==(2000,128)and mask.sum()==25600;lossrows=[];traces=[];selects=[];identities=[];rngs=[];gradrows=[]
    for c in 'ABC':
        root=O/'U'/c/'train';t=read(root/'TRAIN_TRACE.json');identities.append(read(root/'INITIAL_IDENTITY.json'));rngs.append([x['rng_after']for x in t['steps']]);selects.extend([dict(condition=c,**x)for x in t['SELECT_diagnostics']]);step_loss=np.array([x['loss']for x in t['steps']]);sample=load(root/'PER_SAMPLE_NATIVE_LOSS.npz');assert sample['total'].shape==(2000,128)and np.allclose(sample['total'].mean(axis=1),step_loss,rtol=1e-6,atol=1e-6)
        for term in ['ms','cycle_ft','cycle_tf','cycle','total']:
            v=sample[term];allmass=float(v.astype(np.float64).sum());changedmass=float(v[mask].astype(np.float64).sum());lossrows.append(dict(condition=c,term=term,total_native_loss_mass=allmass,designated_replacement_slot_mass=changedmass,replacement_slot_loss_mass_fraction=changedmass/allmass if allmass else None,replacement_slot_mean=float(v[mask].mean()),other_slot_mean=float(v[~mask].mean()),slots_are_not_selected_by_loss=True,A_mask_is_paired_hypothetical_slot_only=c=='A',raw_components_not_anomaly_score=True))
        traces.append(dict(condition=c,SELECT0=t['SELECT_diagnostics'][0]['SELECT_loss'],SELECT2000=t['SELECT_diagnostics'][-1]['SELECT_loss'],first100_loss=float(step_loss[:100].mean()),last100_loss=float(step_loss[-100:].mean()),prototype_t_EMA=t['prototype_EMA']['t_calls'],prototype_f_EMA=t['prototype_EMA']['f_calls'],source_PT_unchanged=t['source_PT_unchanged'],endpoint2000_only=True))
        g=np.array([x['gradient_norm']for x in t['steps']]);gradrows.append(dict(condition=c,grad_median=float(np.median(g)),grad_q95=float(np.quantile(g,.95)),grad_max=float(g.max()),all_finite=bool(np.isfinite(g).all())))
    assert identities[0]['state_sha256']==identities[1]['state_sha256']==identities[2]['state_sha256']and identities[0]['rng']==identities[1]['rng']==identities[2]['rng']and rngs[0]==rngs[1]==rngs[2]
    table(E/'per_native_loss_exposure.csv',lossrows);table(E/'normal_training_summary.csv',traces);table(E/'SELECT_traces.csv',selects);table(E/'gradient_summary.csv',gradrows)
    batch=read(P/'CPU_BATCH_GATE.json');dose=dict(window_presentation_count=256000,B_replaced_window_count=25600,window_fraction=.1,positive_time_exposures=4915200,total_time_exposures=49152000,time_label_positive_fraction=.1,potential_positive_time_channel_exposures=152371200,total_time_channel_exposures=1523712000,potential_scalar_fraction=.1,physical_affected_channel_fraction=None,physical_affected_channel_reason='only attack time labels exist, actual affected channels UNKNOWN',B_C_different_scalar_exposures=batch['B_C_changed_scalar_exposures'],B_C_different_scalar_fraction=batch['B_C_changed_scalar_fraction'],counterfactual_difference_not_physical_channel_truth=True,independent_real_event_donors=13,source_time_windows_not_independent_events=True,native_target_MS_segments=[24,192],native_MS_segment_weights=[.5,.5],native_training_component_coefficients={'MS24':.45,'MS192':.45,'cycleFT':.1,'cycleTF':.1},coefficient_sum=1.1,potential_target_support_weight_fraction=.1,why_same_fraction='every replacement192-window lies fully within one labelled attack; all its192 positions carry positive time labels; MS spectral/coarse targets use whole-window support; actual anomalous channels remain unknown',observed_loss_mass_fraction_file='per_native_loss_exposure.csv',loss_mass_not_gradient_share=True)
    dump(E/'THREE_DENOMINATORS.json',dose)
    stats=[];sealed=read(O/'fixed_scores/ALL_SCORES_THRESHOLDS_MASKS_SEALED.json')
    for d in sealed['descriptors']:
        v=load(d['path'])['CAL_score'];stats.append(dict(method=d['method'],points=len(v),mean=float(v.mean()),std=float(v.std()),median=float(np.median(v)),q95=float(np.quantile(v,.95)),q99=float(np.quantile(v,.99)),q995=float(np.quantile(v,.995)),selected512_q99=d['thresholds']['own_CAL_q99']))
    table(E/'normal_CAL_distributions.csv',stats);dump(E/'NORMAL_REVIEW_RECEIPT.json',dict(status='NORMAL_AND_OFFLINE_LOSS_REVIEW_COMPLETE',paired_initial_model_RNG_exact=True,all2000_step_RNG_paired_exact=True,new_QUERY_labels_not_joined=True,source_PT_track='same-target normal pretraining, SELECT later than UFIT, not crossdatasetZS',seconds=time.monotonic()-start));print(json.dumps(dict(status='NORMAL_REVIEW_READY',training=traces)))
def evaluate():
    start=time.monotonic();S=O/'fixed_scores';sf=S/'ALL_SCORES_THRESHOLDS_MASKS_SEALED.json';seal=read(sf);assert seal['rule_sha256']==sha(P/'FIXED_EXPERIMENT.json');E=O/'fixed_evaluation';E.mkdir(exist_ok=False)
    for d in seal['descriptors']:assert sha(d['path'])==d['sha256']
    dump(E/'RAW_SEAL_BEFORE_SCORE_LABEL_JOIN.json',dict(seal_sha256=sha(sf),score_label_join_for_this_evaluation_not_yet_run=True,labels_previously_metadata_visible=True))
    labels=load(P/'SWAT_LABEL_METADATA.npz')['label'].astype(np.int8);assert len(labels)==944919;sys.path.insert(0,str(R/'preparations/hybrid_left_joint_cycle_20261109_v2/evaluation_runtime_v1/src'));from hybrid_tsad.hybrid_v1 import evaluation as k
    from sklearn.metrics import average_precision_score,roc_auc_score
    rows=[];eventsrows=[];ranks=[]
    for d in seal['descriptors']:
        z=load(d['path']);idx=z['index'];v=z['score'];truth=labels[idx];AP=float(average_precision_score(truth,v));ROC=float(roc_auc_score(truth,v));ranks.append(dict(method=d['method'],AP=AP,ROC_AUC=ROC,points=len(idx),positive_points=int(truth.sum())))
        scored=np.zeros(len(labels),bool);scored[idx]=True;events=k._coverage(labels,scored,0);eligible=[x for x in events if x['main_eligible']];assert len(eligible)==22 and sum(x['total_points']>512 for x in eligible)==9
        for track,tau in d['thresholds'].items():
            for stream in ['strict','confirmed3']:
                full=np.zeros(len(labels),bool);full[idx]=z[track+'__'+stream];view,_=k._event_view(labels,scored,full,np.arange(len(labels)),events,0,'main_eligible');normal,blocks=k._normal(labels,scored,full,0);hit={x['event_id']:x for x in view['events']};cover=[];longcover=[];steady_sum=0;steady_den=0;miss=[];delays=[]
                for e in eligible:
                    lo,hi=e['local_start'],e['local_end'];coverage=float(full[lo:hi].mean());cover.append(coverage);longest=a.longest_miss(full[lo:hi]);miss.append(longest);h=hit[e['event_id']]
                    if h['delay_points']is not None:delays.append(h['delay_points'])
                    steady=None;steady_miss=None
                    if hi-lo>512:
                        steady=float(full[lo+512:hi].mean());longcover.append(steady);steady_sum+=int(full[lo+512:hi].sum());steady_den+=hi-lo-512;steady_miss=a.longest_miss(full[lo+512:hi])
                    eventsrows.append(dict(method=d['method'],track=track,stream=stream,**e,**{x:y for x,y in h.items()if x not in ['event_id','eligible']},coverage=coverage,longest_miss_points=longest,steady_after512_coverage=steady,steady_longest_miss_points=steady_miss))
                rows.append(dict(method=d['method'],track=track,stream=stream,threshold=tau,AP=AP,ROC_AUC=ROC,**a.point_counts(labels,scored,full),**view['metrics'],**{x:y for x,y in normal.items()if not isinstance(y,list)},event_macro_coverage=float(np.mean(cover)),worst_event_coverage=min(cover),long_event_count=len(longcover),long_steady_macro_coverage=float(np.mean(longcover)),long_steady_point_coverage=steady_sum/steady_den,long_steady_total_points=steady_den,delay_hit_only_median=float(np.median(delays))if delays else None,max_event_miss_points=max(miss)))
                folder=E/(d['method']+'__'+track+'__'+stream);folder.mkdir();table(folder/'events.csv',view['events']);table(folder/'alarms.csv',view['alarms']);table(folder/'normal_blocks.csv',blocks)
    table(E/'raw_ranking.csv',ranks);table(E/'method_comparison.csv',rows);table(E/'per_event_coverage_delay_miss.csv',eventsrows)
    lookup={(r['method'],r['track'],r['stream']):r for r in rows};contrasts=[]
    for old,new in [('U_C','U_B'),('U_A','U_C'),('U_A','U_B'),('noU','U_A'),('noU','U_B'),('noU','U_C')]:
        for track in ['own_CAL_q99','fixed_noU_CAL_q99']:
            for stream in ['strict','confirmed3']:
                x=lookup[old,track,stream];y=lookup[new,track,stream];out=dict(old=old,new=new,track=track,stream=stream)
                for key in ['AP','ROC_AUC','TP','FP','Recall','Precision','M','auxiliary_any_hit_events','event_macro_coverage','long_steady_macro_coverage','long_steady_point_coverage','normal_FPR','normal_alarm_fragments_per_1000_points','max_event_miss_points','delay_hit_only_median']:out[key+'_delta']=None if x.get(key)is None or y.get(key)is None else y[key]-x[key]
                contrasts.append(out)
    table(E/'fixed_paired_contrasts.csv',contrasts);dump(E/'RESULT.json',dict(status='COMPLETED_FIXED_EVALUATION',methods=len(ranks),points=len(idx),eligible_events=22,long_events=9,normal_reference_is_nominal=True,one_seed_one_run_not_independent22runs=True,seconds=time.monotonic()-start));print(json.dumps(dict(status='EVALUATION_COMPLETED',ranking=ranks,seconds=time.monotonic()-start)))
if __name__=='__main__':{'seal':seal,'normal':normal_and_exposure,'evaluate':evaluate}[sys.argv[1]]()
