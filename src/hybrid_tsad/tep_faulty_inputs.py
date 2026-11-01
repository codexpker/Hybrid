"""Single-parse, bounded TEP public faulty input preparation; no model or metrics.

Preview reads config/metadata only. RData hash/parse/writes require exact root
release plus the external CPU/RSS lifecycle gate. Pure core tests need no resource.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import time

RAW_PATH='/home/a312/tsad_research/data/tep_rieth2017_v1/TEP_Faulty_Testing.RData'
RAW_BYTES=836882037
RAW_MD5='556bdb64c83021bc0c5f92e427753565'
VARIABLES=('xmeas_9','xmeas_21','xmv_10')
IDENTITIES=('faultNumber','simulationRun','sample')
EXPECTED_COLUMNS=IDENTITIES+tuple('xmeas_%d'%i for i in range(1,42))+tuple('xmv_%d'%i for i in range(1,12))
OUTPUT='/home/a312/tsad_research/results/tep_faulty_inputs_run51_70_20261102_v1'


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def config_metadata(path):
    cfg=read_json(path)
    if (cfg.get('operation')!='TEP_FAULTY_INPUT_EXTRACTION' or cfg.get('raw_path')!=RAW_PATH
            or cfg.get('raw_bytes')!=RAW_BYTES or cfg.get('published_MD5')!=RAW_MD5
            or cfg.get('raw_SHA256_before_execution') is not None
            or cfg.get('fault_ids')!=list(range(1,21)) or cfg.get('run_ids')!=list(range(51,71))
            or cfg.get('samples_1based')!=[1,960,1] or tuple(cfg.get('columns',()))!=VARIABLES
            or cfg.get('output')!=OUTPUT or cfg.get('max_seconds')!=900
            or cfg.get('max_RSS_GiB')!=24 or cfg.get('max_AS_GiB')!=32
            or cfg.get('minimum_MemAvailable_GiB')!=48 or cfg.get('minimum_disk_free_GiB')!=5
            or cfg.get('expected_object_name') is not None
            or cfg.get('object_rule')!='exactly_one_DataFrame_record_actual_name'):
        raise PermissionError('Only fixed public400 input extraction contract')
    paths=[cfg['raw_path'],cfg['output']]
    for key in ('official_metadata','contract'):
        ref=cfg.get(key)
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise PermissionError('Exact metadata reference required')
        paths.append(ref['path'])
    for p in paths:
        if (not isinstance(p,str) or '\\' in p or not PurePosixPath(p).is_absolute()
                or not p.startswith('/home/a312/tsad_research/') or '..' in PurePosixPath(p).parts):
            raise PermissionError('Absolute POSIX workspace paths required')
    return cfg


def unique_dataframe(objects):
    import pandas as pd
    if len(objects)!=1:raise ValueError('Exactly one R object required; do not select arbitrary first table')
    name,frame=next(iter(objects.items()))
    if not isinstance(frame,pd.DataFrame):raise ValueError('Unique R object must be DataFrame')
    return name,frame


def validate_table_header(frame):
    if frame.shape!=(9600000,55) or len(set(frame.columns))!=55 or set(frame.columns)!=set(EXPECTED_COLUMNS):
        raise ValueError('Actual full table shape/55 named columns differ')


def select_input_rows(frame):
    """Only selected six columns copied; original ordinal retained, never sort."""
    import numpy as np
    validate_table_header(frame)
    mask=frame['simulationRun'].isin(range(51,71)).to_numpy()
    positions=np.flatnonzero(mask)
    selected=frame.loc[mask,list(IDENTITIES)+list(VARIABLES)].copy()
    selected['_source_row_0based']=positions
    validate_candidate_rows(selected)
    return selected


def validate_candidate_rows(selected):
    import numpy as np
    if len(selected)!=400*960:raise ValueError('Selected candidate row count differs')
    ids=selected.loc[:,list(IDENTITIES)].to_numpy()
    if not np.isfinite(ids).all() or not np.equal(ids,np.floor(ids)).all():
        raise ValueError('Identity fields must be finite integral original values')
    observed=set(zip(selected.faultNumber,selected.simulationRun))
    expected={(f,r) for f in range(1,21) for r in range(51,71)}
    if observed!=expected:raise ValueError('All400 fixed fault/run groups required without extras')
    return True


def group_arrays(group,fault,run):
    import numpy as np
    if fault not in range(1,21) or run not in range(51,71):raise PermissionError('Fixed candidate identities only')
    if len(group)!=960 or not group.faultNumber.eq(fault).all() or not group.simulationRun.eq(run).all():
        raise ValueError('Original complete case identity differs')
    sample=group['sample'].to_numpy()
    if not np.array_equal(sample,np.arange(1,961)):
        raise ValueError('Missing/duplicate/out-of-order samples; no silent repair')
    if any(group[name].dtype!=np.float64 for name in VARIABLES):
        raise ValueError('Every original measurement column must already be float64')
    values=group.loc[:,list(VARIABLES)].to_numpy(copy=True)
    if values.dtype!=np.float64 or values.shape!=(960,3) or not np.isfinite(values).all():
        raise ValueError('Finite original float64 three-variable values required')
    rows=group['_source_row_0based'].to_numpy()
    if rows.dtype!=np.int64 or (rows<0).any() or not (np.diff(rows)>0).all():
        raise ValueError('Strict original source ordinal required')
    return dict(values=values,sample_1based=np.arange(1,961,dtype=np.int64),
                source_row_0based=rows.copy())


def require_release(config_path,release_path,cfg):
    release=read_json(release_path)
    expected=dict(status='PRIMARY_RELEASED_TEP_FAULTY_INPUT_EXTRACTION',approved_by='root',
        config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        helper_hashes=cfg['helper_hashes'],allow_raw_hash_and_one_parse=True,
        allow_fixed_candidate_inputs=True,allow_identity_sealing=True,
        allow_model_scoring=False,allow_label_evaluation=False)
    if any(release.get(k)!=v for k,v in expected.items()):
        raise PermissionError('Exact input-only release required before raw hash/stat/parse')
    return release


def stat_identity(path):
    s=Path(path).stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)


def execute(config_path,release_path):
    cfg=config_metadata(config_path);require_release(config_path,release_path,cfg)
    if set(cfg['helper_hashes'])!={'__init__.py','tep_normal_assets.py','tep_faulty_inputs.py'}:
        raise ValueError('Exact three-helper closure required')
    for name,digest in cfg['helper_hashes'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Own helper snapshot changed')
    # Linux helper reuse only here: resource is deliberately not imported by core/tests.
    from .tep_normal_assets import sha,read_ref,dump
    for name,digest in cfg['helper_hashes'].items():
        if sha(Path(__file__).parent/name)!=digest:raise ValueError('Own helper snapshot changed')
    metadata=read_ref(cfg['official_metadata'])
    if sha(cfg['contract']['path'])!=cfg['contract']['sha256']:raise ValueError('Input extraction contract changed')
    source=next(x['dataFile'] for x in metadata['data']['latestVersion']['files'] if x['dataFile']['id']==3031243)
    version=metadata['data']['latestVersion']
    if (version['versionNumber'],version['versionMinorNumber'],source['filename'],source['filesize'],source['md5'])!=(1,0,'TEP_Faulty_Testing.RData',RAW_BYTES,RAW_MD5):
        raise ValueError('Published file identity/version differs')
    import os,resource,shutil,gc
    from datetime import datetime,timezone
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise PermissionError('CPU-only extraction required')
    available={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    disk=shutil.disk_usage('/home/a312/tsad_research')
    if available['MemAvailable']<48*1024**3 or disk.free<5*1024**3:raise MemoryError('Startup host RAM/disk gate not met')
    # External watchdog handles RSS while C parser runs; AS is a different limit.
    resource.setrlimit(resource.RLIMIT_AS,(32*1024**3,32*1024**3))
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    report=dict(status='RUNNING',read_r_calls=0,GPU_hours=0,actual_raw_sha256=None,
        raw_published_file_id=3031243,raw_published_bytes=RAW_BYTES,raw_published_MD5=RAW_MD5,
        public_source_DOI='10.7910/DVN/6C3JR1',public_source_version=[1,0],
        expected_object_name_preverified=False,old_whole_file_parsed_before=True,
        not_strict_blind=True,no_lag_change_ground_truth=True,
        host_startup={'utc':datetime.now(timezone.utc).isoformat(),
                      'MemAvailable_bytes':available['MemAvailable'],'disk_free_bytes':disk.free})
    def budget():
        if time.monotonic()-start>900 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>24*1024**3:
            raise MemoryError('CPU time/RSS contract exceeded')
    try:
        raw=Path(cfg['raw_path']);before=stat_identity(raw)
        if before['bytes']!=RAW_BYTES:raise ValueError('Raw stat byte count differs')
        md5=hashlib.md5();sha256=hashlib.sha256()
        with raw.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):md5.update(block);sha256.update(block);budget()
        if md5.hexdigest()!=RAW_MD5 or stat_identity(raw)!=before:raise ValueError('Raw MD5/stat identity changed')
        report.update(actual_raw_sha256=sha256.hexdigest(),actual_MD5=md5.hexdigest(),raw_stat_before=before)
        import pyreadr,numpy as np
        if pyreadr.__version__!='0.5.6':raise ValueError('Existing pyreadr0.5.6 required')
        report['read_r_calls']+=1
        objects=pyreadr.read_r(str(raw))
        name,frame=unique_dataframe(objects);validate_table_header(frame);budget()
        report.update(actual_R_object_name=name,actual_columns=list(frame.columns),actual_shape=list(frame.shape),
                      actual_dtypes={k:str(v) for k,v in frame.dtypes.items()},pyreadr=pyreadr.__version__)
        selected=select_input_rows(frame);del objects,frame;gc.collect();budget()
        (out/'inputs').mkdir();(out/'sealed_identity').mkdir()
        groups=selected.groupby(['faultNumber','simulationRun'],sort=False)
        inputs=[];sealed=[]
        for fault in range(1,21):
            for run in range(51,71):
                entity='entity_%04d'%((fault-1)*20+run-50)
                arrays=group_arrays(groups.get_group((fault,run)),fault,run)
                path=out/'inputs'/(entity+'.npz');np.savez(path,**arrays)
                with np.load(path,allow_pickle=False) as z:
                    if set(z.files)!=set(arrays) or any(not np.array_equal(z[k],v) for k,v in arrays.items()):
                        raise ValueError('Original input archive roundtrip differs')
                inputs.append(dict(entity_id=entity,path=str(path),sha256=sha(path),bytes=path.stat().st_size,
                    shape=[960,3],dtype='float64',columns=list(VARIABLES),raw_sha256=sha256.hexdigest()))
                sealed.append(dict(entity_id=entity,faultNumber=fault,simulationRun=run,
                    nominal_first_positive_sample_1based=161,sampling_minutes=3,
                    physical_onset_uncertainty_samples=1,identity_is_not_all_points_abnormal=True))
                budget()
        if stat_identity(raw)!=before:raise ValueError('Raw file changed during parse/extraction')
        dump(out/'inputs/MANIFEST.json',dict(status='PUBLIC_SIMULATION_INPUTS_NO_LABEL_FEATURES',records=inputs,
            raw_source=str(raw),raw_sha256=sha256.hexdigest(),sample_axis='original 1..960; no absolute datetime',
            imputation=False,standardization=False,sorting=False))
        dump(out/'sealed_identity/MANIFEST.json',dict(status='WORKFLOW_SEALED_NOT_CRYPTOGRAPHIC_OR_BLIND',records=sealed,
            provenance='nominal publisher injection convention; not observed physical-effect/lag-change ground truth'))
        report.update(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',entities_written=len(inputs),
            raw_stat_after=stat_identity(raw),no_model_or_metrics=True,
            config_sha256=sha(config_path),release_sha256=sha(release_path),helper_hashes=cfg['helper_hashes'])
    except BaseException as e:
        import traceback
        report.update(status='FAILED_NO_AUTOMATIC_RETRY',error_type=type(e).__name__,traceback=traceback.format_exc());raise
    finally:
        report.update(wall_seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        dump(out/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--release');a=p.parse_args()
    if a.execute:
        if not a.release:raise PermissionError('Exact root input-only release required')
        print(json.dumps(execute(a.config,a.release),indent=2))
    else:
        cfg=config_metadata(a.config)
        print(json.dumps(dict(status='METADATA_PREPARATION_ONLY_NO_RAW_STAT_HASH_OR_PARSE',
            operation=cfg['operation'],candidate_entities=400,actual_raw_sha256=None,
            expected_object_name='UNVERIFIED; unique actual DataFrame name recorded only after release'),indent=2))


if __name__=='__main__':main()
