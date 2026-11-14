"""Fixed single-entity comparison; calibration/masks sealed before numerical labels."""
import csv,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def read(f):return json.loads(Path(f).read_text())
def load(f):
    with np.load(f)as z:return{k:z[k].copy()for k in z.files}
def dump(f,d):
    with Path(f).open('x')as out:json.dump(d,out,indent=2,allow_nan=False)
def table(f,rows):
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys:keys.append(k)
    with Path(f).open('x',newline='')as out:
        w=csv.DictWriter(out,fieldnames=keys);w.writeheader();w.writerows(rows)
def main():
    start=time.monotonic();E=O/'fixed_evaluation';E.mkdir(parents=True,exist_ok=False)
    assert read(P/'ALL_MODEL_SCORES_COMPLETE.json')['numeric_QUERY_labels_not_read']
    s=importlib.util.spec_from_file_location('alarmfixed',R/'hybrid_ad/src/hybrid_tsad/alarm_complementarity.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    leftcal=load(O/'left_score/CAL.npz');left=load(O/'left_score/QUERY.npz');arxcal=load(O/'arx/CAL.npz');arx=load(O/'arx/QUERY.npz')
    calidx=np.arange(20020,23705);idx=np.arange(8958,23706);assert np.array_equal(leftcal['index'],calidx)and np.array_equal(arxcal['index'],calidx)and np.array_equal(left['index'],idx)and np.array_equal(arx['index'],idx)
    ends=np.linspace(20020,23704,512,dtype=np.int64);positions=ends-calidx[0]
    cal={'LEFT_native':leftcal['score'],'ARX_point':arxcal['score'],'ARX_accumulated':arxcal['g']};scores={'LEFT_native':left['score'],'ARX_point':arx['score'],'ARX_accumulated':arx['g']}
    tau={name:{str(q):float(np.quantile(values[positions],q,method='higher'))for q in [.99,.995]}for name,values in cal.items()}
    assert tau['LEFT_native']['0.995']>0 and tau['ARX_point']['0.995']>0
    scores['LEFT_OR_ARX_point']=np.maximum(np.asarray(scores['LEFT_native'],np.float64)/np.float64(tau['LEFT_native']['0.995']),np.asarray(scores['ARX_point'],np.float64)/np.float64(tau['ARX_point']['0.995']))
    cal['LEFT_OR_ARX_point']=np.maximum(np.asarray(cal['LEFT_native'],np.float64)/np.float64(tau['LEFT_native']['0.995']),np.asarray(cal['ARX_point'],np.float64)/np.float64(tau['ARX_point']['0.995']))
    masks={};thresholds={name:tau[name]['0.99']for name in tau};thresholds['LEFT_OR_ARX_point']=1.
    for name,score in scores.items():
        if name=='LEFT_OR_ARX_point':
            l=left['score']>tau['LEFT_native']['0.995'];r=arx['score']>tau['ARX_point']['0.995'];strict=l|r;confirmed=a.confirmed(idx,l)|a.confirmed(idx,r);assert np.array_equal(strict,score>1.)
        else:strict=score>thresholds[name];confirmed=a.confirmed(idx,strict)
        masks[name+'__strict']=strict;masks[name+'__confirmed3']=confirmed
    np.savez_compressed(E/'CAL_AND_RAW_QUERY_MASKS.npz',index=idx,available_index=idx,CAL_index=calidx,CAL_selected_index=ends,**{'CAL_'+k:v for k,v in cal.items()},**{'score_'+k:v for k,v in scores.items()},**masks)
    dump(E/'CALIBRATION.json',dict(selected_index=ends.tolist(),thresholds=thresholds,per_branch_tau=tau,quantile_method='higher',single_nominal_budget=.01,OR_each_branch_nominal_budget=.005,CAL_not_USED_FOR_SELECT=True,scaler_FIT_only=True,background_health='provider nominal normal, not independently certified clean'))
    dump(E/'RAW_AND_DECISIONS_SEALED_BEFORE_LABELS.json',dict(numeric_QUERY_labels_opened=False,raw_path=str(E/'CAL_AND_RAW_QUERY_MASKS.npz'),raw_sha256=sha(E/'CAL_AND_RAW_QUERY_MASKS.npz'),calibration_sha256=sha(E/'CALIBRATION.json'),fixed_rule_sha256=sha(P/'FIXED_COMPARISON.json'),source_sha256=sha(P/'evaluate_fixed.py'),source_assets=[dict(path=str(f),sha256=sha(f))for f in [O/'left_score/CAL.npz',O/'left_score/QUERY.npz',O/'arx/CAL.npz',O/'arx/QUERY.npz',O/'left_train/RUNTIME_STATE.json',P/'FIT_ONLY_SCALER.json']]))
    # Numerical label use starts only below the completed raw/mask seal.
    labelpath=R/'data/prospective_smd_machine15_roles_20261106_v1/audit_only/ORIGINAL_TEST_LABELS.npz';labels=load(labelpath)['labels'];assert labels.shape==(23706,)
    runtime=R/'preparations/hybrid_left_joint_cycle_20261109_v2/evaluation_runtime_v1/src';sys.path.insert(0,str(runtime));from hybrid_tsad.hybrid_v1 import evaluation as kernel
    from sklearn.metrics import average_precision_score,roc_auc_score
    scored=np.zeros(len(labels),bool);scored[idx]=True;available=np.arange(len(labels));events=kernel._coverage(labels,scored,0);assert sum(e['main_eligible']for e in events)==7 and all(e['total_points']<=512 for e in events)
    catalog=list(csv.DictReader((R/'data/prospective_smd_machine15_roles_20261106_v1/audit_only/EVENT_CATALOG.csv').open()));sourceids={(int(e['start']),int(e['end'])):e['event_id']for e in catalog}
    for e in events:e['source_event_id']=sourceids[e['start'],e['end_exclusive']]
    table(E/'original_event_coverage.csv',events);rows=[];eventrows=[];sets={}
    for name,score in scores.items():
        AP=float(average_precision_score(labels[idx],score));ROC=float(roc_auc_score(labels[idx],score));sets[name]={}
        for stream in ['strict','confirmed3']:
            full=np.zeros(len(labels),bool);full[idx]=masks[name+'__'+stream]
            main,_=kernel._event_view(labels,scored,full,available,events,0,'main_eligible');normal,blocks=kernel._normal(labels,scored,full,0);lookup={e['event_id']:e for e in main['events']}
            row=dict(method=name,stream=stream,points=len(idx),positive_points=int(labels[idx].sum()),AP=AP,ROC_AUC=ROC,threshold=thresholds[name],**a.point_counts(labels,scored,full),**main['metrics'],**{k:v for k,v in normal.items()if not isinstance(v,list)})
            delays=[e['delay_points']for e in main['events']if e['delay_points']is not None];miss=[]
            for e in events:
                lo,hi=e['local_start'],e['local_end'];er=lookup[e['event_id']];detected=int(full[lo:hi].sum());longest=a.longest_miss(full[lo:hi]);miss.append(longest)
                eventrows.append(dict(method=name,stream=stream,**e,**{k:v for k,v in er.items()if k not in ['event_id','eligible']},detected_points=detected,coverage=detected/e['total_points'],longest_miss_points=longest,steady_after512_points=None,steady_after512_coverage=None,steady_NA_reason='NO_EVENT_LONGER_THAN512'))
            row.update(hit_only_delay_median_points=float(np.median(delays))if delays else None,longest_complete_event_miss_points=max(miss),steady_long_events=0,steady_detected_points=None,steady_scored_points=None,steady_coverage=None,steady_NA_reason='NO_EVENT_LONGER_THAN512')
            rows.append(row);sets[name][stream]={criterion:{e['event_id']for e in main['events']if e['eligible']and e[criterion]}for criterion in ['any_hit','one_to_one_matched']}
            table(E/(name+'__'+stream+'__matched_events.csv'),main['events']);table(E/(name+'__'+stream+'__alarm_segments.csv'),main['alarms']);table(E/(name+'__'+stream+'__normal_blocks.csv'),blocks)
    table(E/'method_comparison.csv',rows);table(E/'per_event_coverage_delay_miss.csv',eventrows)
    overlaps=[];changes=[]
    for right in ['ARX_point','ARX_accumulated']:
        for stream in ['strict','confirmed3']:
            for criterion in ['any_hit','one_to_one_matched']:
                l=sets['LEFT_native'][stream][criterion];r=sets[right][stream][criterion]
                overlaps.append(dict(ARX_branch=right,stream=stream,criterion=criterion,LEFT_only=len(l-r),ARX_only=len(r-l),shared=len(l&r),neither=7-len(l|r),LEFT_only_ids=sorted(l-r),ARX_only_ids=sorted(r-l),shared_ids=sorted(l&r)))
    for stream in ['strict','confirmed3']:
        for criterion in ['any_hit','one_to_one_matched']:
            l=sets['LEFT_native'][stream][criterion];r=sets['LEFT_OR_ARX_point'][stream][criterion];changes.append(dict(stream=stream,criterion=criterion,added=len(r-l),lost=len(l-r),retained=len(l&r),added_ids=sorted(r-l),lost_ids=sorted(l-r)))
    dump(E/'EVENT_COMPLEMENTARITY.json',dict(branch_overlap=overlaps,OR_changes=changes,event_id_source_mapping={e['event_id']:e['source_event_id']for e in events}))
    c3={r['method']:r for r in rows if r['stream']=='confirmed3'};leftc=c3['LEFT_native'];dual=c3['LEFT_OR_ARX_point'];left_any=sets['LEFT_native']['confirmed3']['any_hit'];dual_any=sets['LEFT_OR_ARX_point']['confirmed3']['any_hit']
    screen=dict(additional_one_to_one_C3_events=dual['M']>leftc['M'],LEFT_q99_anyhit_events_preserved=left_any<=dual_any,actual_normal_C3_FPR_le_nominal1pct=dual['normal_FPR']<=.01,LEFT_matched_events=leftc['M'],OR_matched_events=dual['M'],LEFT_normal_FP=leftc['FP'],OR_normal_FP=dual['FP'],OR_normal_C3_FPR=dual['normal_FPR'],normal_CAL_role_isolated=True,normal_health_not_certified=True,prospective_historically_exposed_not_blind=True,independent_entity_count=1,no_lag_specific_or_long_duration_claim=True)
    screen['positive_operational_screen']=all(screen[k]for k in ['additional_one_to_one_C3_events','LEFT_q99_anyhit_events_preserved','actual_normal_C3_FPR_le_nominal1pct'])
    dump(E/'FIXED_OPERATIONAL_SCREEN.json',screen);dump(E/'RESULT.json',dict(status='COMPLETED_FIXED_SMD_MACHINE15_COMPARISON',QUERY_points=14748,original_positive_points=int(labels[idx].sum()),complete_events=7,methods=list(scores),labels_after_raw_seal=True,no_method_or_threshold_search=True,source_label_sha256=sha(labelpath),evaluation_kernel_sha256=sha(runtime/'hybrid_tsad/hybrid_v1/evaluation.py'),seconds=time.monotonic()-start,new_GPU_seconds=0))
    print(json.dumps(dict(status='COMPLETED',seconds=time.monotonic()-start,operational_screen=screen)),flush=True)
if __name__=='__main__':main()
