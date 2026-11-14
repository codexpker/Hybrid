"""Native-entity source-only evaluation glue; original kernels/weights unchanged.

The deliberate64-point cost prefix belongs to the final full-entity bank.
Operator job resources never replace the bank's frozen scientific-config SHA.
"""
import hashlib
import json
from pathlib import Path
import time

SOURCE_PT='c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812'
SOURCE_CAL='da239eac4c5ab1e76718cd651c48f6f89b8d5658b5b9b2a20cb400d88f83d585'
BASELINE_CAL={'persistence':'5513840e1454ef3cd84e373c510a336660d716d60971ad6e14b752e1c946d1cd',
              'history_mean512':'16b3a4959a08998ea2dedbf3c2373e7a95e1fae15607f21d065ce42d9825fb9d'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def entity_key(run):
    return run.domain+'__'+run.entity+'__'+run.run


def length_groups(entries,slots=3):
    """Deterministic longest-first static assignment, never scores or labels."""
    if type(slots) is not int or not 1<=slots<=4:raise ValueError('One to four existing GPU slots')
    groups=[[] for _ in range(slots)];loads=[0]*slots
    for e in sorted(entries,key=lambda x:(-(x['expected_shape'][0]-512),x['domain'],x['entity'],x['run'])):
        if e['role']!='QUERY' or e['expected_shape'][0]<=512:raise ValueError('Native QUERY longer thanH512')
        k=min(range(slots),key=lambda i:(loads[i],i));groups[k].append(e);loads[k]+=e['expected_shape'][0]-512
    return dict(groups=groups,points=loads,rule='longest scored length first; lexicographic identity tie; lowest load/index tie')


def ranges(run):
    return [dict(domain=run.domain,entity=run.entity,run=run.run,
                 start=run.start+512,end=run.start+len(run.values))]


def native_identity(run):
    return hashlib.sha256(run.values.tobytes()+run.observed.tobytes()).hexdigest()


def write_record_segment(factory,directory,*,identity,declared_ranges,threshold,
                         max_new_points=None,resume=False):
    """Storage-only original iterator consumer, testable without Torch."""
    from .chunked_bank import ChunkedBank
    if max_new_points not in (None,64):raise ValueError('Only fixed64 profile or complete remainder')
    started=time.monotonic()
    with ChunkedBank(directory,identity=identity,ranges=declared_ranges,threshold=threshold,chunk_size=64) as bank:
        prefix=bank.manifest['committed_records']
        if prefix and not resume:raise PermissionError('Explicit authorized remaining-plan continuation required')
        if bank.manifest['complete']:
            return dict(already_complete=True,prefix=prefix,new_points=0,complete=True,seconds=0.)
        remaining=bank.total_records-prefix
        if max_new_points is not None and prefix:
            raise PermissionError('Never repeat an existing cost profile')
        count=remaining if max_new_points is None else min(max_new_points,remaining)
        iterator=iter(factory(bank.remaining_plan()))
        try:
            for _ in range(count):bank.append(next(iterator))
        finally:
            close=getattr(iterator,'close',None)
            if close:close()
        if count==remaining:bank.finalize()
        else:
            # Exact64 profile means an immutable full chunk is already committed.
            if bank.buffer:raise ValueError('Incomplete profile chunk must not be silently discarded')
        final=bank.manifest['committed_records'];complete=bank.manifest['complete']
        if final-prefix!=count:raise ValueError('Segment did not commit every produced point')
    return dict(prefix=prefix,new_points=count,committed_records=final,complete=complete,
                manifest_sha256=sha(Path(directory)/'manifest.json'),seconds=time.monotonic()-started)


def write_source_entity(model,run,calibration,directory,*,identity,max_new_points=None,resume=False):
    """Original iter_pointwise_infer+score_records, complete tail, no target fit."""
    import torch
    from .chunked_bank import iter_pointwise_infer
    from .cli import score_records
    from .readout import _rng,_restore,_state
    if (identity['checkpoint_sha256'],identity['calibration_sha256'])!=(SOURCE_PT,SOURCE_CAL):
        raise PermissionError('Original source c00f/da239 only; no synthetic adaptation or SOC')
    if identity['input_sha256']!=native_identity(run) or run.role!='QUERY':
        raise PermissionError('Exact native QUERY input identity required')
    directory=Path(directory);model.eval();before=_state(model)
    manifest=directory/'manifest.json';snapshot=directory/'PROFILE64_RNG.pt';binding=directory/'PROFILE64_RNG.json'
    if resume and manifest.exists():
        m=json.loads(manifest.read_text())
        if not m['complete'] and m['committed_records']:
            info=json.loads(binding.read_text())
            if (info['manifest_sha256']!=sha(manifest) or info['rng_sha256']!=sha(snapshot)
                    or info['committed_records']!=m['committed_records']):
                raise PermissionError('Partial prefix RNG/manifest changed; no automatic scientific retry')
            # Only this SHA-bound self-generated state, never a user-supplied pickle.
            _restore(torch.load(snapshot,map_location='cpu',weights_only=False))
    def factory(plan):
        for record in iter_pointwise_infer(model,[run],history=512,horizon=32,seed=2025,index_plan=plan):
            yield score_records([record],calibration)[0]
    result=write_record_segment(factory,directory,identity=identity,declared_ranges=ranges(run),
        threshold=calibration['threshold'],max_new_points=max_new_points,resume=resume)
    if before!=_state(model):raise ValueError('Public inference mutated frozen source weights/buffers')
    if not result['complete']:
        torch.save(_rng(),snapshot)
        dump(binding,dict(committed_records=result['committed_records'],manifest_sha256=sha(manifest),
            rng_sha256=sha(snapshot),purpose='Planned64 profile continuation; do not repeat scored prefix'))
    result.update(entity=entity_key(run),expected_points=len(run.values)-512,
                  source_unchanged=True,labels_read=False,target_fit=False,threshold_fit=False)
    return result


def write_baseline_entity(run,method,calibration,directory,*,identity):
    """Existing deterministic control and matching old source CAL, CPU only."""
    from .readout import write_baseline_bank
    if method not in BASELINE_CAL or identity['calibration_sha256']!=BASELINE_CAL[method]:
        raise PermissionError('Accepted original same-source baseline CAL only')
    if run.role!='QUERY' or identity['input_sha256']!=native_identity(run):
        raise PermissionError('Native query identity differs')
    write_baseline_bank([run],method,calibration,directory,identity=identity,history=512,chunk_size=64,resume=False)
    m=json.loads((Path(directory)/'manifest.json').read_text())
    if not m['complete'] or m['committed_records']!=len(run.values)-512:raise ValueError('Incomplete baseline')
    return dict(entity=entity_key(run),method=method,points=m['committed_records'],
        manifest_sha256=sha(Path(directory)/'manifest.json'),source_calibration_reused=True,
        target_fit=False,threshold_fit=False,labels_read=False)
