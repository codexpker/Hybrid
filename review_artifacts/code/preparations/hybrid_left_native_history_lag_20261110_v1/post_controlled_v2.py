"""Fixed controlled-family evaluation; each run remains a distinct timeline."""
import json,time,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name;E=R/'results/hybrid_left_native_history_lag_controlled_evaluation_20261110_v1_cpu_repaired'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 start=time.monotonic();assert(P/'CONTROLLED_ALL_QUERY_COMPLETE.json').exists();E.mkdir(parents=True,exist_ok=False);world=json.loads((P/'CONTROLLED_DATA_MANIFEST.json').read_text());descriptors=[];seals=[];expected_ends=np.linspace(512,4095,256,dtype=np.int64)
 for arm in ['N','L','LS','LC','LSC']:
  q=O/'controlled'/arm/'query';cal=O/'controlled'/arm/'cal_repaired';r=json.loads((q/'RECEIPT.json').read_text());assert r['status']=='COMPLETED'and r['runs']==40;banks=json.loads((q/'BANKS.json').read_text());c=json.loads((cal/'RECEIPT.json').read_text());assert c['records']==512 and c['normal_CAL_runs_independent_of_SELECT']
  for i in range(2):
   with np.load(cal/('normal_run_'+str(i)+'.npz'))as z:assert np.array_equal(z['index'],expected_ends)
  for b in banks:
   assert sha(b['path'])==b['sha256'];seals.append(dict(path=b['path'],sha256=b['sha256']))
  descriptors.append(dict(method=arm,banks=banks,threshold=c['threshold'],kind='native_current_reconstruction',cal=cal))
 base=O/'controlled_baselines_matched512';cal=json.loads((base/'CALIBRATIONS.json').read_text());queries=[e for e in world['roles']if e['role']=='QUERY']
 for method in ['ARX_Ridge512','persistence','HistoryMean512']:
  c=cal[method];assert c['points']==512 and all(np.array_equal(x['index'],expected_ends)for x in c['indices']);banks=[]
  for e in queries:
   path=base/method/(e['meta']['case']+'_'+str(e['meta']['seed'])+'.npz');b=dict(path=str(path),sha256=sha(path),case=e['meta']['case'],seed=e['meta']['seed'],truth=e['truth']);banks.append(b);seals.append(dict(path=str(path),sha256=b['sha256']))
  descriptors.append(dict(method=method,banks=banks,threshold=c['threshold'],kind='strict_past_prediction',cal=base/'CALIBRATIONS.json'))
 dump(E/'ALL_RAW_SEALED_BEFORE_NUMERIC_TRUTH.json',dict(status='ALL_FIXED_SCORES_SEALED',numerical_truth_opened=False,truth_intervention_metadata_predefined_visible=True,not_used_for_learning_selection_normalization_threshold=True,banks=seals,source_generator=sha(P/'controlled_world.py'),source_manifest=sha(P/'CONTROLLED_DATA_MANIFEST.json'),normalization_FIT_only=True,CAL_independent_SELECT=True,CAL_indices_per_run=expected_ends.tolist()))
 sys.path.insert(0,str(R/'preparations/hybrid_left_joint_cycle_20261109_v2/evaluation_runtime_v1/src'))
 from hybrid_tsad.hybrid_v1 import evaluation
 from hybrid_tsad.hybrid_v1.public_native_evaluation import event_diagnostics
 from hybrid_tsad.hybrid_v1.source_candidate_psm_comparison import import_bound,table,corrected_operating_columns
 from hybrid_tsad.hybrid_v1.pretrained_psm_comparison import bound
 from sklearn.metrics import average_precision_score,roc_auc_score
 cfg=json.loads((R/'preparations/hybrid_left_joint_cycle_20261109_v2/EVALUATION.ACTUAL.json').read_text());formatter=import_bound(cfg['operating_formatter'],'formatter').operating_columns;rows=[]
 for d in descriptors:
  for b in d['banks']:
   with np.load(b['path'])as z:idx=z['index'].copy();score=z['score'].copy();available=z['available_index'].copy();assert np.array_equal(idx,np.arange(512,8192));assert np.array_equal(available,idx)
   with np.load(b['truth'])as z:y=z['labels'].copy();assert y.shape==(8192,)
   result=evaluation.evaluate_arrays(index=idx,score=score,valid=np.ones(len(idx),bool),available_index=available,invalid_reasons=[None]*len(idx),full_labels=y,threshold=d['threshold'],bank_complete=True,original_length=8192,warmup_points=512,affiliation_identity=bound(cfg['affiliation_identity']))
   folder=E/d['method']/(b['case']+'_'+str(b['seed']))
   evaluation.save_evaluation(folder,result,index=idx,score=score,valid=np.ones(len(idx),bool),available_index=available,invalid_reasons=[None]*len(idx),full_labels=y,provenance=dict(method=d['method'],case=b['case'],seed=b['seed'],threshold=d['threshold'],score_bank=b,source_manifest_sha256=sha(P/'CONTROLLED_DATA_MANIFEST.json')))
   phases=event_diagnostics(result);op=corrected_operating_columns(formatter,result,phases);truth=y[idx];row=dict(method=d['method'],case=b['case'],world_seed=b['seed'],AP=float(average_precision_score(truth,score))if truth.any()else None,ROC_AUC=float(roc_auc_score(truth,score))if truth.any()else None,positive_points=int(truth.sum()),points=len(idx),threshold=d['threshold'],**op)
   # Additional transition/steady readout uses fixed generator boundaries, not adjusted events.
   for name,lo,hi in [('transition',3072,3584),('steady',3584,5120)]:
    mask=(idx>=lo)&(idx<hi);row[name+'_strict_hits']=int((score[mask]>d['threshold']).sum());row[name+'_points']=int(mask.sum())
   rows.append(row)
 table(E/'all_run_metrics.csv',rows)
 summaries=[]
 for method in [d['method']for d in descriptors]:
  for case in sorted({e['meta']['case']for e in queries}):
   rr=[r for r in rows if r['method']==method and r['case']==case];out=dict(method=method,case=case,independent_worlds=5)
   for key in ['AP','ROC_AUC','strict_Recall','strict_normal_FP_fragments_per1000','confirmed3_Recall','confirmed3_main_events_hit','confirmed3_normal_FP_fragments_per1000','confirmed3_steady_detected_points','confirmed3_max_main_event_observed_miss_points','steady_strict_hits']:
    x=[r[key]for r in rr if r.get(key)is not None];out[key+'_mean']=float(np.mean(x))if x else None;out[key+'_SD']=float(np.std(x,ddof=1))if len(x)>1 else None
   summaries.append(out)
 table(E/'case_summary.csv',summaries)
 paired=[]
 lookup={(r['method'],r['case'],r['world_seed']):r for r in rows}
 for old,new in [('N','L'),('L','LS'),('L','LC'),('L','LSC'),('LS','LSC'),('LC','LSC')]:
  for case in sorted({e['meta']['case']for e in queries}):
   for seed in range(90000,90005):
    a=lookup[old,case,seed];b=lookup[new,case,seed];p=dict(old=old,new=new,case=case,seed=seed)
    for key in ['AP','ROC_AUC','strict_Recall','confirmed3_Recall','confirmed3_main_events_hit','confirmed3_normal_FP_fragments_per1000','confirmed3_steady_detected_points']:
     p[key+'_delta']=None if a.get(key)is None or b.get(key)is None else b[key]-a[key]
    paired.append(p)
 table(E/'paired_world_deltas.csv',paired);dump(E/'RESULT.json',dict(status='COMPLETED_FIXED_CONTROLLED_FAMILY',methods=[d['method']for d in descriptors],runs=len(rows),independent_worlds=5,paired_cases=8,optimization_seeds=1,raw_seal_before_numeric_truth=True,known_interventions_not_blind_labels=True,score_permissions='native current reconstruction vs strict-past linear predictors, common source/CAL/query roles',seconds=time.monotonic()-start));print(json.dumps(dict(status='COMPLETED',RESULT_sha256=sha(E/'RESULT.json'),seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()

