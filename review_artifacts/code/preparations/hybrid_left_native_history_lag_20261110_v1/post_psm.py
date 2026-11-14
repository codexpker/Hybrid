"""PSM fixed family readout using full-history common CAL; no score search."""
import json,time,sys,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;R=P.parents[1];O=R/'results'/P.name;E=R/'results/hybrid_left_native_history_lag_fixed_PSM_evaluation_20261110_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 start=time.monotonic();assert(P/'PSM_ALL_QUERY_COMPLETE.json').exists();E.mkdir(parents=True,exist_ok=False);paths=[]
 for arm in ['L','LS','LC','LSC']:
  for phase in ['query','cal_common']:assert json.loads((O/arm/phase/'RECEIPT.json').read_text())['status']=='COMPLETED'
  paths.extend([O/arm/'query/scores.npz',O/arm/'cal_common/scores.npz',O/arm/'cal_common/RECEIPT.json'])
 assert json.loads((O/'N/cal_common/RECEIPT.json').read_text())['records']==512;paths.extend([O/'N/cal_common/scores.npz',O/'N/cal_common/RECEIPT.json',R/'results/left_causal_endpoint_20261109_v1/QUERY/scores.npz'])
 ends=None
 for arm in ['N','L','LS','LC','LSC']:
  with np.load(O/arm/'cal_common/scores.npz')as z:
   if ends is None:ends=z['index'].copy()
   assert np.array_equal(ends,z['index'])and len(ends)==512 and ends[0]==106496 and ends[-1]==132480
 dump(E/'RAW_SEALED_BEFORE_LABELS.json',dict(labels_read=False,assets=[dict(path=str(p),sha256=sha(p))for p in paths],common_CAL_endpoints=ends.tolist(),SELECT_CAL_reuse_exploratory=True))
 sys.path.insert(0,str(R/'preparations/hybrid_left_joint_cycle_20261109_v2/evaluation_runtime_v1/src'));from hybrid_tsad.hybrid_v1.source_candidate_psm_comparison import _evaluate,import_bound,table;from hybrid_tsad.hybrid_v1.pretrained_psm_comparison import bound
 cfg=json.loads((R/'preparations/hybrid_left_joint_cycle_20261109_v2/EVALUATION.ACTUAL.json').read_text());loader=import_bound(cfg['baseline_loader'],'old_fixed_loader');formatter=import_bound(cfg['operating_formatter'],'old_fixed_formatter').operating_columns
 with np.load(bound(cfg['truth']))as z:labels=z['full_labels'].copy()
 desc=[];data={}
 for arm in ['N','L','LS','LC','LSC']:
  name='LEFT_native_N'if arm=='N'else'Hybrid_'+arm;path=R/'results/left_causal_endpoint_20261109_v1/QUERY/scores.npz'if arm=='N'else O/arm/'query/scores.npz'
  with np.load(path)as z:data[name]={k:z[k].copy()for k in('index','score','valid','available_index')}
  c=json.loads((O/arm/'cal_common/RECEIPT.json').read_text());desc.append(dict(id=name,online=True,threshold=c['threshold'],protocol='native_LEFT_score_W192_history513_if_L_q99higher_common512CAL',role_permissions='target_pretestFIT105984/SELECT26497/fullpretestScaler; SELECT_CAL_reuse_exploratory'))
 for d in cfg['baselines']:
  if d['id']=='HybridV1_Z_source2025':continue
  desc.append(d);data[d['id']]=loader.load_method(d,labels)
 results={}
 for name,axis in [('common',np.arange(512,87744)),('full',np.arange(512,87841))]:
  root=E/name;root.mkdir();rows=[]
  for d in desc:
   if name=='full'and not d['online']:continue
   row,_=_evaluate(data[d['id']],axis,labels,d,root/d['id'],cfg,formatter,reuse=False);rows.append(row)
  results[name]=rows;table(E/('all_methods_'+name+'_axis.csv'),rows)
 dump(E/'RESULT.json',dict(status='COMPLETED_FIXED_HISTORY_LAG_FAMILY_PSM',results=results,query_is_development_historical=True,labels_after_all_raw_sealed=True,CAL_SELECT_reuse=True,no_best_arm_selection=True,seconds=time.monotonic()-start));print(json.dumps(dict(status='COMPLETED',RESULT_sha256=sha(E/'RESULT.json'),seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()
