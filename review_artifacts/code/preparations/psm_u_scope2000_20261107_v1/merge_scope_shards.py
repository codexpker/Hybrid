"""CPU cached-codec full-axis envelope; no forward, scoring, or truth access."""
import argparse,hashlib,json,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
 assert sha(a.config)==a.trusted_config_sha256;c=read(a.config)
 assert c['operation']=='PSM_SCOPE2000_CACHED_RAW_MERGE' and c['scope'] in ('HEAD','FULL') and c['labels_read'] is False
 for path,digest in c['immutable_sha256'].items():assert sha(path)==digest,path
 sys.path.insert(0,c['runtime_src']);sys.path.insert(0,c['helper_directory'])
 import hybrid_tsad.hybrid_v1.chunked_bank as cb
 from cached_chunk_reader import cached_unpack
 identity=None;tau=None;position=512;receipts=[]
 for d in c['shards']:
  assert sha(d['receipt']['path'])==d['receipt']['sha256'] and sha(d['manifest']['path'])==d['manifest']['sha256']
  r=read(d['receipt']['path']);m=read(d['manifest']['path'])
  assert r['status']=='COMPLETED' and r['phase']=='query' and r['scope']==c['scope'] and r['labels_read'] is False
  assert r['common_plan_sha256']==c['common_plan_sha256'] and r['source_checkpoint_sha256']==c['source_checkpoint_sha256']
  assert r['index_range'][0]==position and r['points']==r['index_range'][1]-position;position=r['index_range'][1]
  assert m['complete'] and m['committed_records']==r['points'] and r['bank_manifest_sha256']==d['manifest']['sha256']
  if identity is None:identity=m['contract']['identity'];tau=m['contract']['threshold']
  assert m['contract']['identity']==identity and m['contract']['threshold']==tau
  assert identity['checkpoint_sha256']==r['checkpoint_sha256'] and identity['calibration_sha256']==r['calibration_sha256']
  receipts.append(r)
 assert position==87841 and len(receipts)==4
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
 def originals():
  i=512
  for d in c['shards']:
   for row in cb.read_records(Path(d['manifest']['path']).parent):
    assert (row['domain'],row['entity'],row['run'],row['index'],row['available_index'])==('PSM','anonymous_PSM','test_original',i,i)
    assert row['replay']['actual_mask_seed']==2025+i;i+=1;yield row
  assert i==87841
 with cached_unpack(cb):
  with cb.ChunkedBank(out/'bank',identity=identity,ranges=[dict(domain='PSM',entity='anonymous_PSM',run='test_original',start=512,end=87841)],threshold=tau,chunk_size=64) as bank:
   bank.write(originals());bank.finalize();assert bank.manifest['committed_records']==87329
 receipt=dict(status='COMPLETED',phase='query',scope=c['scope'],seed=2025,scientific_status='COMPLETE_PENDING_FIXED_EVALUATION',
  points=87329,bank_manifest_sha256=sha(out/'bank/manifest.json'),checkpoint_sha256=identity['checkpoint_sha256'],calibration_sha256=identity['calibration_sha256'],
  source_checkpoint_sha256=c['source_checkpoint_sha256'],common_plan_sha256=c['common_plan_sha256'],bank_identity=identity,
  shard_receipts=c['shards'],new_forward=False,labels_read=False,raw_canonical_unchanged=True,
  continuous_alarms_recomputed_after_full_axis_join=True,config_sha256=a.trusted_config_sha256)
 with (out/'RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
