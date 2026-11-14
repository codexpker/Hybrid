"""Frozen source-CAL controls on native public QUERY roles; no labels/model."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True)
    p.add_argument('--method',choices=['persistence','history_mean512'],required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Frozen source-CAL config required')
    c=json.loads(Path(a.config).read_text())
    sys.path.insert(0,c['source_root'])
    from hybrid_tsad.hybrid_v1 import public_native_zs as native
    from hybrid_tsad.hybrid_v1.stage2_u_matched import load_roles
    for path,h in c['source_sha256'].items():
        if sha(path)!=h:raise PermissionError('Changed frozen CPU score source')
    if sha(c['query_manifest']['path'])!=c['query_manifest']['sha256']:
        raise PermissionError('Changed query-only input manifest')
    spec=c['calibrations'][a.method]
    if sha(spec['path'])!=spec['sha256'] or spec['sha256']!=native.BASELINE_CAL[a.method]:
        raise PermissionError('Original same-source CAL required, never targetCAL')
    out=Path(c['output'])/a.method;out.mkdir(parents=True,exist_ok=False)
    dump(out/'EXECUTION.json',dict(pid=os.getpid(),method=a.method,config_sha256=a.trusted_config_sha256,
        command=sys.argv,labels_read=False,GPUh=0))
    runs=load_roles(c['query_manifest']['path'],{'QUERY'});cal=json.loads(Path(spec['path']).read_text())
    if len(runs)!=54 or sum(len(r.values)-512 for r in runs)!=415230:raise ValueError('Fixed53SMAP+SMD native query coverage')
    began=time.monotonic();banks=[];progress=out/'progress.jsonl';progress.open('x').close()
    checkpoint=hashlib.sha256(('NO_TRAINED_CHECKPOINT::'+a.method+'::'+c['baselines_source_sha256']).encode()).hexdigest()
    for run in runs:
        identity=dict(input_sha256=native.native_identity(run),config_sha256=a.trusted_config_sha256,
            source_sha256=c['score_source_identity_sha256'],checkpoint_sha256=checkpoint,
            calibration_sha256=spec['sha256'],query_manifest_sha256=c['query_manifest']['sha256'])
        path=out/run.domain/run.entity/'bank'
        result=native.write_baseline_entity(run,a.method,cal,path,identity=identity)
        result['directory']=str(path.resolve());result['identity']=identity
        banks.append(result)
        with progress.open('a') as f:f.write(json.dumps(result)+'\n');f.flush()
    for path,h in c['source_sha256'].items():
        if sha(path)!=h:raise PermissionError('CPU source changed during scoring')
    dump(out/'BANKS.json',banks)
    dump(out/'RECEIPT.json',dict(status='COMPLETED_RAW_ONLY_PENDING_HYBRID_AND_UNIFIED_EVALUATION',
        method=a.method,entities=54,records=sum(x['points'] for x in banks),seconds=time.monotonic()-began,
        GPUh=0,labels_read=False,new_fit=False,target_CAL=False,torch_imported='torch' in sys.modules))
    print(json.dumps(dict(method=a.method,entities=54,records=sum(x['points'] for x in banks),seconds=time.monotonic()-began)),flush=True)


if __name__=='__main__':main()
