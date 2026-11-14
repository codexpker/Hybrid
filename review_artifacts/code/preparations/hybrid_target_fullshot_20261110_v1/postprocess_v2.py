"""Single cached merge + fixed existing metric kernels; no model forward."""
import json,hashlib,sys,time,shutil
from pathlib import Path
import numpy as np
R=Path('__RESEARCH_ROOT__');P=Path(__file__).resolve().parent;O=R/'results'/P.name;E=R/'results/hybrid_target_fullshot_fixed_PSM_evaluation_20261110_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def asset(p):return dict(path=str(p),sha256=sha(p))
def main():
 assert (P/'ALL_QUERY_COMPLETE.json').exists();assert not E.exists()
 begin=time.monotonic();runtime=R/'preparations/hybrid_left_joint_cycle_20261109_v2/evaluation_runtime_v1/src';sys.path.insert(0,str(runtime))
 from hybrid_tsad.hybrid_v1 import chunked_bank as cb
 sys.path.insert(0,str(R/'preparations/hybrid_left_joint_cycle_20261109_v2'))
 from cached_chunk_reader import cached_unpack
 shards=[];identity=None;merged=O/'query_merged';merged.mkdir()
 for i in range(4):
  root=O/('query_'+str(i));rec=json.loads((root/'RECEIPT.json').read_text());m=json.loads((root/'bank/manifest.json').read_text())
  assert rec['status']=='COMPLETED' and m['complete'] and rec['bank_manifest_sha256']==sha(root/'bank/manifest.json')
  if identity is None:identity=m['contract']['identity'];template=m
  assert m['contract']['identity']==identity and m['contract']['threshold']==template['contract']['threshold']
  shards.append(root/'bank')
 def records():
  expected=512
  for root in shards:
   for row in cb.read_records(root):
    assert row['index']==expected and row['available_index']==expected;expected+=1;yield row
  assert expected==87841
 with cached_unpack(cb),cb.ChunkedBank(merged/'bank',identity=identity,ranges=[dict(domain='PSM',entity='anonymous_PSM',run='test_original',start=512,end=87841)],threshold=template['contract']['threshold'],chunk_size=64) as bank:
  bank.write(records());bank.finalize();assert bank.manifest['committed_records']==87329
 from hybrid_tsad.hybrid_v1.pretrained_psm_comparison import archive_scalars,read,bound
 from hybrid_tsad.hybrid_v1.source_candidate_psm_comparison import _evaluate,import_bound,table
 E.mkdir();archive=archive_scalars(dict(manifest=asset(merged/'bank/manifest.json'),calibration=asset(O/'cal/calibration.json')),E/'FULLSHOT_CANONICAL_SCALARS.npz')
 dump(E/'RAW_SEALED_BEFORE_LABELS.json',dict(status='ALL_NEW_RAW_SEALED',labels_read=False,archive=archive,all_shards_complete=True,manifest=asset(merged/'bank/manifest.json')))
 # First numerical query label read follows all new raw closure.
 cfg=json.loads((R/'preparations/hybrid_left_joint_cycle_20261109_v2/EVALUATION.ACTUAL.json').read_text())
 loader=import_bound(cfg['baseline_loader'],'original_baselines');formatter=import_bound(cfg['operating_formatter'],'original_formatter').operating_columns
 with np.load(bound(cfg['truth'])) as z:labels=z['full_labels'].copy()
 descriptors=cfg['baselines'];data={d['id']:loader.load_method(d,labels) for d in descriptors}
 scope=json.loads((R/'preparations/psm_u_scope2000_20261107_v1/EVALUATION.FINAL.json').read_text());old=R/'results/hybrid_v1_psm_U_scope2000_fixed_evaluation_20261107_v1'
 for name,path,threshold in [('Z_T_source2025',old/'ZT_original_source_with_target_CAL512.npz',read(scope['ZT_calibration'])['threshold']),('U_FULL2000_source2025',old/'U_FULL2000_source2025_canonical_scalars.npz',read(next(d for d in scope['adapted_methods'] if d['scope']=='FULL')['calibration'])['threshold'])]:
  with np.load(path) as z:data[name]={k:z[k].copy() for k in ('index','score','valid','available_index')}
  descriptors.append(dict(id=name,online=True,threshold=threshold,protocol='reuse_accepted_original_target_protocol',role_permissions='targetCAL only' if name.startswith('Z_') else 'targetFIT[0,79488)2000_updates_hybrid_adaptation_modes_targetCAL'))
 with np.load(archive['path']) as z:data['Hybrid_FULLSHOT8000']={k:z[k].copy() for k in ('index','score','valid','available_index')}
 descriptors.insert(1,dict(id='Hybrid_FULLSHOT8000',online=True,threshold=archive['threshold'],protocol='full_target_FIT8000_all_train_min_SELECT64_h1R3',role_permissions='sourcewarmstart_targetFIT[0,105984)_pretestScaler[0,132481)_isolatedSELECT_CAL_noQUERYfit'))
 from sklearn.metrics import average_precision_score,roc_auc_score
 allrows={}
 for name,axis in [('common',np.arange(512,87744)),('full',np.arange(512,87841))]:
  root=E/name;root.mkdir();rows=[]
  for d in descriptors:
   if name=='full' and not d['online']:continue
   row,phases=_evaluate(data[d['id']],axis,labels,d,root/d['id'],cfg,formatter,reuse=False);rows.append(row)
  # Frozen causal LEFT has no legally calibrated operational threshold.
  with np.load(R/'results/left_causal_endpoint_20261109_v1/QUERY/scores.npz') as z:
   take=np.searchsorted(z['index'],axis);score=z['score'][take];assert np.array_equal(z['index'][take],axis)
  rows.append(dict(method='LEFT_R_targettrained_causal_endpoint',points=len(axis),AP=float(average_precision_score(labels[axis],score)),ROC_AUC=float(roc_auc_score(labels[axis],score)),online=True,threshold=None,role_permissions='target-trained LEFT/checkpoint and fullpretest scaler; prefix endpoint causal',operating_NA='No approved calibration threshold for this existing LEFT readout'))
  table(E/('all_methods_'+name+'_axis.csv'),rows);allrows[name]=rows
 dump(E/'RESULT.json',dict(status='COMPLETED_FULLSHOT_DIAGNOSTIC',results=allrows,labels_after_raw_seal=True,query_domain='PSM development/historical exposure',new_GPU_forward=False,seconds=time.monotonic()-begin))
 print(json.dumps(dict(status='COMPLETED',RESULT=asset(E/'RESULT.json'),seconds=time.monotonic()-begin)),flush=True)
if __name__=='__main__':main()
