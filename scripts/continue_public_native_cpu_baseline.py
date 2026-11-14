"""Resume already committed public persistence banks; no score is recomputed.

The retained original resume path omitted QUERY role in remaining_plan entries.
This consumer supplies the one physically bound run's role; source math stays fixed.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Fixed original config required')
    c=json.loads(Path(a.config).read_text());sys.path.insert(0,c['source_root'])
    from hybrid_tsad.hybrid_v1 import public_native_zs as native
    from hybrid_tsad.hybrid_v1.stage2_u_matched import load_roles
    from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank
    from hybrid_tsad.hybrid_v1.baselines import iter_baselines
    from hybrid_tsad.hybrid_v1.cli import score_records
    for path,h in c['source_sha256'].items():
        if sha(path)!=h:raise PermissionError('Original scientific source changed')
    if sha(c['query_manifest']['path'])!=c['query_manifest']['sha256']:raise PermissionError('Native inputs changed')
    out=Path(c['output'])/'persistence';attempt=out/'ROLE_CONSUMER_RECOVERY';attempt.mkdir(exist_ok=False)
    dump(attempt/'EXECUTION.json',dict(pid=os.getpid(),command=sys.argv,script_sha256=sha(__file__),
        science_config_sha256=a.trusted_config_sha256,previous_failures_retained=True,labels_read=False,
        change='Add physically known QUERY role to ChunkedBank.remaining_plan entries; no math/index changes'))
    spec=c['calibrations']['persistence']
    if sha(spec['path'])!=spec['sha256']:raise PermissionError('Original source CAL required')
    cal=json.loads(Path(spec['path']).read_text());runs=load_roles(c['query_manifest']['path'],{'QUERY'})
    tag=hashlib.sha256(('NO_TRAINED_CHECKPOINT::persistence::'+c['baselines_source_sha256']).encode()).hexdigest()
    began=time.monotonic();banks=[]
    with (attempt/'progress.jsonl').open('x') as progress:
        for run in runs:
            path=out/run.domain/run.entity/'bank'
            identity=dict(input_sha256=native.native_identity(run),config_sha256=a.trusted_config_sha256,
                source_sha256=c['score_source_identity_sha256'],checkpoint_sha256=tag,
                calibration_sha256=spec['sha256'],query_manifest_sha256=c['query_manifest']['sha256'])
            with ChunkedBank(path,identity=identity,ranges=native.ranges(run),threshold=cal['threshold'],chunk_size=64) as bank:
                previous=bank.manifest['committed_records']
                if not bank.manifest['complete']:
                    plan=[dict(e,role=run.role) for e in bank.remaining_plan()]
                    for raw in iter_baselines([run],method='persistence',history=512,index_plan=plan):
                        bank.append(score_records([raw],cal)[0])
                    bank.finalize()
            m=json.loads((path/'manifest.json').read_text())
            if not m['complete'] or m['committed_records']!=len(run.values)-512:raise ValueError('Incomplete native bank')
            r=dict(entity=native.entity_key(run),method='persistence',points=m['committed_records'],
                manifest_sha256=sha(path/'manifest.json'),directory=str(path.resolve()),identity=identity,
                already_committed_reused=previous,new_points=m['committed_records']-previous)
            banks.append(r);progress.write(json.dumps(r)+'\n');progress.flush()
    for path,h in c['source_sha256'].items():
        if sha(path)!=h:raise PermissionError('Original scientific source changed')
    dump(out/'BANKS.json',banks)
    dump(out/'RECEIPT.json',dict(status='COMPLETED_RAW_ONLY_AFTER_LIMITED_CONSUMER_RECOVERY',
        original_exit_code=1,first_recovery_exit_code=1,method='persistence',entities=54,
        records=sum(x['points'] for x in banks),reused_committed_points=sum(x['already_committed_reused'] for x in banks),
        new_points=sum(x['new_points'] for x in banks),seconds=time.monotonic()-began,
        GPUh=0,labels_read=False,new_fit=False,target_CAL=False,torch_imported='torch' in sys.modules))
    print(json.dumps(dict(entities=54,records=sum(x['points'] for x in banks),seconds=time.monotonic()-began)),flush=True)

if __name__=='__main__':main()
