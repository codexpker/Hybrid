"""Final frozen A/B/C evaluation. Labels enter only after all endpoint score seals."""
import csv,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name;E=O/'U_fixed_evaluation'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def load(f):
    with np.load(f)as z:return{k:z[k].copy()for k in z.files}
def dump(f,d):
    with Path(f).open('x')as out:json.dump(d,out,indent=2,allow_nan=False)
def table(f,rows):
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with Path(f).open('x',newline='')as out:
        w=csv.DictWriter(out,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    start=time.monotonic();E.mkdir(parents=True,exist_ok=False);source=O/'U_fixed_scores/ALL_RAW_MASKS_THRESHOLDS_SEALED.json';seal=read(source)
    assert not seal['QUER_NUMERIC_labels_opened']and seal['fixed_rule_sha256']==sha(P/'FIXED_U_SENSITIVITY.json')
    for d in seal['descriptors']:assert sha(d['path'])==d['sha256']
    dump(E/'RAW_SEAL_BEFORE_LABELS.json',dict(source_sha256=sha(source),numeric_QUERY_labels_opened=False,rule_sha256=sha(P/'FIXED_U_SENSITIVITY.json')))
    labels_path=R/'data/prospective_smd_machine15_roles_20261106_v1/audit_only/ORIGINAL_TEST_LABELS.npz';labels=load(labels_path)['labels'];assert labels.shape==(23706,)
    runtime=R/'preparations/hybrid_left_joint_cycle_20261109_v2/evaluation_runtime_v1/src';sys.path.insert(0,str(runtime));from hybrid_tsad.hybrid_v1 import evaluation as kernel
    from sklearn.metrics import average_precision_score,roc_auc_score
    s=importlib.util.spec_from_file_location('alarmfixed',R/'hybrid_ad/src/hybrid_tsad/alarm_complementarity.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    rows=[];events_rows=[];ranking=[]
    for descriptor in seal['descriptors']:
        d=load(descriptor['path']);idx=d['index'];assert np.array_equal(idx,np.arange(8958,23706));score=d['score'];truth=labels[idx];scored=np.zeros(len(labels),bool);scored[idx]=True;available=np.arange(len(labels));events=kernel._coverage(labels,scored,0);assert len(events)==7 and all(e['main_eligible']for e in events)
        AP=float(average_precision_score(truth,score));ROC=float(roc_auc_score(truth,score));ranking.append(dict(condition=descriptor['condition'],AP=AP,ROC_AUC=ROC,points=len(idx),positive_points=int(truth.sum()),primary_endpoint=0 if descriptor['condition']=='base' else 200))
        for track in ['own_CAL_q99','frozen_base_q99']:
            for stream in ['strict','confirmed3']:
                full=np.zeros(len(labels),bool);full[idx]=d[track+'__'+stream];main,_=kernel._event_view(labels,scored,full,available,events,0,'main_eligible');normal,blocks=kernel._normal(labels,scored,full,0)
                row=dict(condition=descriptor['condition'],track=track,stream=stream,threshold=descriptor[track],AP=AP,ROC_AUC=ROC,**a.point_counts(labels,scored,full),**main['metrics'],**{k:v for k,v in normal.items()if not isinstance(v,list)})
                lookup={e['event_id']:e for e in main['events']};miss=[];delays=[]
                for e in events:
                    lo,hi=e['local_start'],e['local_end'];er=lookup[e['event_id']];n=int(full[lo:hi].sum());longest=a.longest_miss(full[lo:hi]);miss.append(longest)
                    if er['delay_points']is not None:delays.append(er['delay_points'])
                    events_rows.append(dict(condition=descriptor['condition'],track=track,stream=stream,source_event_id='machine-1-5:E'+str(int(e['event_id'][1:])-1).zfill(3),**e,**{k:v for k,v in er.items()if k not in ['event_id','eligible']},detected_points=n,coverage=n/e['total_points'],longest_miss_points=longest,steady_after512_coverage=None,steady_NA_reason='NO_EVENT_LONGER_THAN512'))
                row.update(hit_only_delay_median_points=float(np.median(delays))if delays else None,longest_event_miss_points=max(miss),steady_long_events=0,steady_coverage=None,steady_NA_reason='NO_EVENT_LONGER_THAN512');rows.append(row)
                folder=E/(descriptor['condition']+'__'+track+'__'+stream);folder.mkdir();table(folder/'events.csv',main['events']);table(folder/'alarms.csv',main['alarms']);table(folder/'normal_blocks.csv',blocks)
    table(E/'raw_ranking.csv',ranking);table(E/'operating_comparison.csv',rows);table(E/'per_event_coverage_delay_miss.csv',events_rows)
    # Baseline operating identity and cached statistical reference are not rerun.
    old=list(csv.DictReader((R/'results/cross_domain_left_arx_20261111_v1/fixed_evaluation/method_comparison.csv').open()));oldleft={r['stream']:r for r in old if r['method']=='LEFT_native'}
    for r in rows:
        if r['condition']=='base':
            for key in ['TP','FP','M','Ntrue','Recall','normal_FPR']:assert float(r[key])==float(oldleft[r['stream']][key])
    table(E/'cached_ARX_statistical_reference.csv',[r for r in old if r['method']=='ARX_point'])
    lookup={(r['condition'],r['track'],r['stream']):r for r in rows};contrasts=[]
    for oldc,newc in [('C','B'),('A','C'),('A','B'),('base','A'),('base','B'),('base','C')]:
        for track in ['own_CAL_q99','frozen_base_q99']:
            for stream in ['strict','confirmed3']:
                x=lookup[oldc,track,stream];y=lookup[newc,track,stream];out=dict(old=oldc,new=newc,delta='new_minus_old',track=track,stream=stream)
                for key in ['AP','ROC_AUC','TP','FP','Precision','Recall','F1','M','auxiliary_any_hit_events','normal_FPR','normal_alarm_fragments_per_1000_points','hit_only_delay_median_points','longest_event_miss_points']:
                    out[key+'_delta']=None if x.get(key)is None or y.get(key)is None else y[key]-x[key]
                contrasts.append(out)
    table(E/'fixed_paired_condition_contrasts.csv',contrasts)
    calibration=[]
    for c in ['base','A','B','C']:
        for stream in ['strict','confirmed3']:
            x=lookup[c,'frozen_base_q99',stream];y=lookup[c,'own_CAL_q99',stream];r=dict(condition=c,stream=stream,delta='ownCAL_minus_frozenbase')
            for key in ['threshold','TP','FP','Recall','M','normal_FPR','normal_alarm_fragments_per_1000_points']:r[key+'_delta']=y[key]-x[key]
            calibration.append(r)
    table(E/'recalibration_vs_fixed_threshold.csv',calibration)
    dump(E/'RESULT.json',dict(status='COMPLETED_FIXED_NATIVE_LEFT_U_SENSITIVITY',conditions=['base','A','B','C'],tracks=['own_CAL_q99','frozen_base_q99'],primary_U_endpoint=200,QUERY_points=14748,complete_events=7,long_steady_NA=True,no_model_or_threshold_selection=True,one_seed_one_template=True,not_sourceZ_or_unseen_target_transfer=True,window_presentation_dose_not_raw_anomaly_prevalence=True,raw_seal_sha256=sha(source),label_sha256=sha(labels_path),rule_sha256=sha(P/'FIXED_U_SENSITIVITY.json'),seconds=time.monotonic()-start))
    print(json.dumps(dict(status='COMPLETED',seconds=time.monotonic()-start,ranking=ranking)),flush=True)
if __name__=='__main__':main()
