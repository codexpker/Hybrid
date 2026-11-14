"""One explicit Windows manifest recovery plus deterministic uncommitted suffix.

No completed entity/point is recomputed. Original failure and before/pending
manifests remain as evidence; scientific scorer/source/calibration stay fixed.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Original fixed config required')
    c=json.loads(Path(a.config).read_text());sys.path.insert(0,c['source_root'])
    from hybrid_tsad.hybrid_v1 import public_native_zs as native
    from hybrid_tsad.hybrid_v1.stage2_u_matched import load_roles
    from hybrid_tsad.hybrid_v1.readout import write_baseline_bank
    from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,_unpack,_key
    for path,h in c['source_sha256'].items():
        if sha(path)!=h:raise PermissionError('Original score source changed')
    if sha(c['query_manifest']['path'])!=c['query_manifest']['sha256']:raise PermissionError('Native manifest changed')
    method='persistence';out=Path(c['output'])/method
    repair=out/'WINDOWS_MANIFEST_RECOVERY';repair.mkdir(exist_ok=False)
    dump(repair/'EXECUTION.json',dict(pid=os.getpid(),command=sys.argv,original_exit_code=1,
        error='WinError5 os.replace manifest.pending -> manifest at SMAP/D-8',
        science_config_sha256=a.trusted_config_sha256,recovery_script_sha256=sha(__file__),labels_read=False))
    spec=c['calibrations'][method]
    if sha(spec['path'])!=spec['sha256']:raise PermissionError('Source CAL changed')
    cal=json.loads(Path(spec['path']).read_text());runs=load_roles(c['query_manifest']['path'],{'QUERY'})
    broken=out/'SMAP/D-8/bank';manifest=broken/'manifest.json';pending=list(broken.glob('manifest.pending-*.json'))
    if len(pending)!=1:raise ValueError('Exactly one retained failed atomic commit required')
    old=json.loads(manifest.read_text());new=json.loads(pending[0].read_text())
    if (new['contract']!=old['contract'] or new['committed_records']!=old['committed_records']+64 or
            new['chunks'][:-1]!=old['chunks'] or new['complete'] is not False):
        raise ValueError('Pending update is not exactly the failed next64 commit')
    with ChunkedBank(broken,identity=old['contract']['identity'],ranges=old['contract']['ranges'],
                     threshold=cal['threshold'],chunk_size=64) as bank:
        last=new['chunks'][-1];chunk=broken/last['file']
        if sha(chunk)!=last['sha256']:raise ValueError('Failed commit chunk changed')
        with np.load(chunk,allow_pickle=False) as z:entries=_unpack(z)
        if len(entries)!=64:raise ValueError('Failed commit payload is not64 records')
        states=copy.deepcopy(bank.states)
        for i,entry in enumerate(entries):
            if _key(entry['raw'])!=bank.expected_key(old['committed_records']+i):raise ValueError('Pending point axes differ')
            if bank._derive(entry['raw'],states)!=entry['derived']:raise ValueError('Pending confirmation state differs')
        if list(states.values())!=new['state']:raise ValueError('Pending ending state differs')
        before_sha=sha(manifest);pending_sha=sha(pending[0])
        (repair/'manifest.before.json').write_bytes(manifest.read_bytes())
        (repair/'manifest.pending.json').write_bytes(pending[0].read_bytes())
    # Both are checked absolute files in the intended generated result directory.
    if not str(manifest.resolve()).startswith(str(out.resolve())) or pending[0].parent!=manifest.parent:
        raise PermissionError('Unexpected recovery filesystem target')
    os.replace(pending[0],manifest)
    dump(repair/'METADATA_REPAIR.json',dict(before_manifest_sha256=before_sha,pending_sha256=pending_sha,
        after_manifest_sha256=sha(manifest),old_points=old['committed_records'],new_points=new['committed_records'],
        point_recomputations=0,repair='Verified existing raw chunk+pending contract+state; committed metadata only'))
    before=time.monotonic();banks=[];progress=repair/'progress.jsonl';progress.open('x').close()
    tag=hashlib.sha256(('NO_TRAINED_CHECKPOINT::'+method+'::'+c['baselines_source_sha256']).encode()).hexdigest()
    for run in runs:
        path=out/run.domain/run.entity/'bank'
        identity=dict(input_sha256=native.native_identity(run),config_sha256=a.trusted_config_sha256,
            source_sha256=c['score_source_identity_sha256'],checkpoint_sha256=tag,
            calibration_sha256=spec['sha256'],query_manifest_sha256=c['query_manifest']['sha256'])
        previous=0
        if (path/'manifest.json').exists():previous=json.loads((path/'manifest.json').read_text())['committed_records']
        write_baseline_bank([run],method,cal,path,identity=identity,history=512,chunk_size=64,resume=True)
        m=json.loads((path/'manifest.json').read_text())
        if not m['complete'] or m['committed_records']!=len(run.values)-512:raise ValueError('Incomplete recovered entity')
        r=dict(entity=native.entity_key(run),method=method,points=m['committed_records'],
            manifest_sha256=sha(path/'manifest.json'),directory=str(path.resolve()),identity=identity,
            already_committed_reused=previous,new_points=m['committed_records']-previous)
        banks.append(r)
        with progress.open('a') as f:f.write(json.dumps(r)+'\n');f.flush()
    for path,h in c['source_sha256'].items():
        if sha(path)!=h:raise PermissionError('Frozen scorer changed')
    dump(out/'BANKS.json',banks)
    dump(out/'RECEIPT.json',dict(status='COMPLETED_RAW_ONLY_AFTER_EXPLICIT_METADATA_RECOVERY',
        original_exit_code=1,method=method,entities=54,records=sum(x['points'] for x in banks),
        reused_committed_points=sum(x['already_committed_reused'] for x in banks),
        new_points=sum(x['new_points'] for x in banks),seconds=time.monotonic()-before,
        GPUh=0,labels_read=False,new_fit=False,target_CAL=False,torch_imported='torch' in sys.modules))
    print(json.dumps(dict(method=method,entities=54,records=sum(x['points'] for x in banks),seconds=time.monotonic()-before)),flush=True)


if __name__=='__main__':main()
