"""Candidate pure 20-run normal-control interface; no parser/model/worker.

The post-hoc normal control is a distinct role. Original400fault guards remain
unchanged. Frozen Ridge score_values and checkpoint/save helpers can be reused
after this role validation; no actual runtime/release is implemented here.
"""
import argparse
import hashlib
import json
from pathlib import PurePosixPath, Path
import re

RUNS = tuple(range(51,71))
COLUMNS = ('xmeas_9','xmeas_21','xmv_10')
ENTITIES = tuple('normal_run_%04d'%run for run in RUNS)
OPERATION = 'TEP_FIXED_NORMAL_CONTROL_READOUT'
ROOT = PurePosixPath('/home/a312/tsad_research')

def digest(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('Exact SHA256 required')
    return value

def strict_int(value):
    if type(value) is not int:
        raise ValueError('Integer identity required, not bool/float')
    return value

def control_path(value):
    if not isinstance(value,str) or '\\' in value:
        raise ValueError('Absolute POSIX path required')
    p = PurePosixPath(value)
    base = ROOT/'results/tep_normal_control_inputs_run51_70_20261102_v1/inputs'
    if not p.is_absolute() or '..' in p.parts or p.parent != base:
        raise ValueError('Distinct normal-control input root required')
    return p

def validate_manifest(manifest):
    allowed={'status','records','raw_source','raw_sha256','sample_axis','imputation','standardization','sorting'}
    if set(manifest)!=allowed or manifest['status']!='PUBLIC_SIMULATION_NORMAL_CONTROL_INPUTS':
        raise PermissionError('Closed normal-control manifest schema')
    digest(manifest['raw_sha256'])
    if (manifest['raw_source']!='TEP_FaultFree_Testing.RData' or type(manifest['sample_axis']) is not list
            or any(type(x) is not int for x in manifest['sample_axis']) or manifest['sample_axis']!=[1,960,1]
            or manifest['imputation']!='none' or manifest['standardization']!='none' or manifest['sorting']!='none'):
        raise PermissionError('Unmodified complete FaultFreeTesting input required')
    rows=manifest['records']
    if not isinstance(rows,list) or len(rows)!=20:
        raise ValueError('Exactly20 independent normal runs required')
    fields={'entity_id','role','faultNumber','simulationRun','path','sha256','bytes','shape','dtype','columns','raw_sha256'}
    for row,run,entity in zip(rows,RUNS,ENTITIES):
        if not isinstance(row,dict) or set(row)!=fields:
            raise PermissionError('Closed record; no labels/onset/extra nested fields')
        if (row['entity_id']!=entity or row['role']!='NORMAL_CONTROL'
                or strict_int(row['faultNumber'])!=0 or strict_int(row['simulationRun'])!=run):
            raise PermissionError('Normal identity and ordered51..70runs required')
        if type(row['shape']) is not list or any(type(x) is not int for x in row['shape']) or row['shape']!=[960,3] or row['dtype']!='float64' or type(row['columns']) is not list or any(type(x) is not str for x in row['columns']) or row['columns']!=list(COLUMNS):
            raise ValueError('Original960x3 float64 columns required')
        if control_path(row['path']).name!=entity+'.npz':raise ValueError('Identity/path differs')
        digest(row['sha256'])
        if digest(row['raw_sha256'])!=manifest['raw_sha256'] or strict_int(row['bytes'])<=0:
            raise ValueError('Exact original provenance required')
    return rows

def validate_arrays(arrays):
    import numpy as np
    if set(arrays)!={'values','sample_1based','source_row_0based'}:
        raise PermissionError('Numerical inputs only, no labels or derived targets')
    values,samples,rows=(arrays[k] for k in ('values','sample_1based','source_row_0based'))
    if values.shape!=(960,3) or values.dtype!=np.float64 or not np.isfinite(values).all():
        raise ValueError('Finite original960x3 float64 values')
    if samples.dtype!=np.int64 or not np.array_equal(samples,np.arange(1,961,dtype=np.int64)):
        raise ValueError('No missing/duplicate/reordered original sample axis')
    if rows.dtype!=np.int64 or rows.shape!=(960,) or (rows<0).any() or not (np.diff(rows)>0).all():
        raise ValueError('Increasing original source ordinal required')
    return values

def extract_group_arrays(group,run):
    """Pure frame adapter only; caller's eventual parser/release is not supplied."""
    import numpy as np
    if strict_int(run) not in RUNS or len(group)!=960 or not group['faultNumber'].eq(0).all() or not group['simulationRun'].eq(run).all():
        raise PermissionError('One complete normal fault0run required')
    if any(group[name].dtype!=np.float64 for name in COLUMNS):
        raise ValueError('No implicit measurement coercion')
    arrays=dict(values=group.loc[:,list(COLUMNS)].to_numpy(copy=True),
        sample_1based=group['sample'].to_numpy(copy=True),source_row_0based=group['_source_row_0based'].to_numpy(copy=True))
    validate_arrays(arrays)
    return arrays

def group_normal_control_arrays(frame):
    """One already-parsed full normal table ->20 original-run input banks.

    No read_r, file open, sorting, imputation, scaler fitting or model call.
    Header names are imported from the unchanged actual shared parser dictionary.
    The full500runs axis is checked before selecting51..70; source rows are
    positional ordinals in this exact table, not potentially reset index labels.
    """
    import numpy as np
    from .tep_faulty_inputs import EXPECTED_COLUMNS
    if frame.shape!=(500*960,55) or len(set(frame.columns))!=55 or set(frame.columns)!=set(EXPECTED_COLUMNS):
        raise ValueError('Full500normalruns x960samples / actual55column dictionary required')
    ids=frame.loc[:,['faultNumber','simulationRun','sample']].to_numpy()
    if not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all() or not (ids[:,0]==0).all():
        raise ValueError('Only finite integral fault0 normal identities')
    runs=ids[:,1].astype(np.int64)
    if set(runs.tolist())!=set(range(1,501)):
        raise ValueError('Full normal500run coverage required')
    if not np.array_equal(np.bincount(runs,minlength=501)[1:],np.full(500,960)):
        raise ValueError('Exactly960rows per original run required')
    for run in range(1,501):
        if not np.array_equal(ids[runs==run,2],np.arange(1,961)):
            raise ValueError('Missing/duplicate/reordered original samples in full table')
    selected=frame.loc[np.isin(runs,RUNS),['faultNumber','simulationRun','sample']+list(COLUMNS)].copy()
    selected['_source_row_0based']=np.flatnonzero(np.isin(runs,RUNS)).astype(np.int64)
    # Identities may originate as integral R doubles; verify before lossless int64 axis materialization.
    selected['sample']=selected['sample'].to_numpy(dtype=np.int64)
    return [(run,extract_group_arrays(selected.loc[selected['simulationRun'].eq(run)],run)) for run in RUNS]

def collect_normal_bank(normalized_values,*,run,seed,scorer,fingerprint):
    """Pure injected-callback B1 oracle. No real model import/load/scoring driver.

    Same frozen raw last/mean32 reductions as original collect_bank. Strict
    NORMAL_CONTROL identities never reclassified as CAL or opaque fault query.
    Actual native scorer/timing/provenance/runtime will require a later review.
    """
    import numpy as np
    if strict_int(run) not in RUNS or strict_int(seed) not in (2025,2026,2027):
        raise PermissionError('Fixed20normal/three existing seeds only')
    v=np.asarray(normalized_values)
    if v.shape!=(960,3) or v.dtype not in (np.float32,np.float64) or not np.isfinite(v).all():
        raise ValueError('Finite normalized960x3 required')
    before=json.loads(json.dumps(fingerprint(),sort_keys=True,allow_nan=False))
    if set(before)!={'state','gradients','modes','step','rng'}:raise ValueError('Fullbank-edge fingerprint required')
    raw=np.empty((861,100,3),np.float32)
    for start in range(861):
        batch=np.array(v[start:start+100],dtype=np.float32,copy=True)[None]
        snapshot=batch.copy();score=scorer(batch)
        if not np.array_equal(batch,snapshot):raise RuntimeError('Scorer modified input')
        if not isinstance(score,np.ndarray) or score.shape!=(1,100,3) or score.dtype!=np.float32 or not np.isfinite(score).all():
            raise ValueError('Native B1 synchronous float32 CPU output required')
        raw[start]=score[0]
    after=json.loads(json.dumps(fingerprint(),sort_keys=True,allow_nan=False))
    if before!=after:raise RuntimeError('Frozen model/gradient/mode/step/RNG changed')
    point=raw.mean(axis=2,dtype=np.float32);ends=np.arange(99,960,dtype=np.int64)
    return dict(raw_scores=raw,last=point[:,-1],mean32=point[:,-32:].mean(axis=1,dtype=np.float32),
        channel_last=raw[:,-1,:].copy(),channel_mean32=raw[:,-32:,:].mean(axis=1,dtype=np.float32),
        end_index_0based=ends,available_index_0based=ends.copy(),end_sample_1based=ends+1)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config.get('operation')!=OPERATION:raise PermissionError('Distinct normal-control operation only')
    raise PermissionError('PENDING_NORMAL_INPUT_ACCEPTANCE: candidate has no actual parser/model/worker')

if __name__=='__main__':main()
