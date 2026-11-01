"""Bounded TEP normal-only physical three-variable extraction, no models.

Names are resolved from actual R dataframe columns and cached process dictionary.
Only the FaultFree Training file is admitted. Existing roles are never reassigned.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import resource
import time

VARIABLES=('xmeas_9','xmeas_21','xmv_10')
RAW_SHA='12d1055b852763fac09ef36bd7c9612a800c8bf7b6f75f344d12119206f9e940'
CACHE_SHA='43cff9a26cbcf15ffbf66fd40db1b9b10ee114ebd9e8fbbcb10b741b99b15fdd'
ROLES={'fit':list(range(1,21))+list(range(51,61)), 'select':list(range(61,81)), 'cal':list(range(81,101))}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def read_ref(ref):
    if sha(ref['path'])!=ref['sha256']:raise ValueError('Pinned metadata identity differs')
    return json.loads(Path(ref['path']).read_text(encoding='utf-8'))


def dump(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)


def validate_roles(manifest):
    found={k:[] for k in ROLES}
    for row in manifest['records']:
        role=row['role'];rid=row['selector']['simulationRun']
        if (role not in ROLES or row['selector']['faultNumber']!=0 or
            row['source_identity']!=RAW_SHA or (row['local_start'],row['local_end_exclusive'])!=(0,500)):
            raise ValueError('Unexpected old target-normal role/source footprint')
        found[role].append(rid)
    if found!=ROLES:raise ValueError('Frozen normal role lists differ; never reassign runs')
    return found


def lag_profiles(fit_values,max_lag=64):
    """Past-only Pearson, common endpoints64..499, never cross-run boundaries."""
    import numpy as np
    if not fit_values or any(x.shape!=(500,3) or not np.isfinite(x).all() for x in fit_values):
        raise ValueError('Complete finite FIT3 arrays required')
    rows=[]
    for target in range(3):
        y=np.concatenate([x[max_lag:,target] for x in fit_values]);yc=y-y.mean()
        for predictor in range(3):
            for lag in range(1,max_lag+1):
                past=np.concatenate([x[max_lag-lag:500-lag,predictor] for x in fit_values])
                xc=past-past.mean();den=math.sqrt(float(xc@xc)*float(yc@yc))
                corr=float((xc@yc)/den) if den>0 else None
                rows.append({'target':VARIABLES[target],'past_predictor':VARIABLES[predictor],
                    'lag_points':lag,'lag_minutes':lag*3,'paired_points':len(y),'Pearson':corr,
                    'target_index_zero_based':'64..499 within each FIT run',
                    'predictor_index_zero_based':f'{64-lag}..{499-lag} within same run'})
    return rows


def window_schedule():
    windows=30*(500-100+1);steps=math.ceil(windows/128);total=20*steps
    return {'candidate_H':100,'H192_automatically_modified':False,
        'FIT_runs':30,'run_points':500,'stride':1,'FIT_windows':windows,
        'batch128_updates_per_epoch':steps,'last_batch':windows%128,'max_epochs':20,
        'max_optimizer_updates':total,'original_warmup_steps':1000,'original_ramp_steps':2000,
        'first_positive_training_lambda_step':1001,'ramp_completion_step':3000,
        'maximum_ramp_fraction':(total-1000)/2000,'maximum_training_lambda':.3*(total-1000)/2000,
        'original_eval_lambda':.3,'full_ramp_reachable_within_20epochs':False,
        'epochs_to_reach_step3000_if_not_early_stopped':math.ceil(3000/steps),
        'early_stopping_can_reduce_updates':True,
        'official_fault_metadata_only':{'sampling_minutes':3,'fault_injection_hours':8,
            'normal_prefix_points':160,'first_fault_sample_1based':161,
            'boundary_uncertainty':'up to one sampling interval; no faulty file body opened'},
        'prefault_H100_stride1_complete_windows':160-100+1,
        'prefault_H100_nonoverlap_complete_windows':160//100,
        'prefault_H192_complete_windows':max(0,160-192+1),
        'H100_past100_forecast_point_targets_before_fault':160-100,
        'mean128_readout_compatible_with_H100':False,
        'old_C2_weights_reused':False,'hyperparameters_modified':False}


def extract(config_path):
    cfg=json.loads(Path(config_path).read_text(encoding='utf-8'))
    if cfg['variables']!=list(VARIABLES) or cfg['max_seconds']!=300 or cfg['max_RSS_GiB']!=2:
        raise ValueError('Fixed normal extraction scope required')
    raw=Path(cfg['raw']['path'])
    if raw.name!='TEP_FaultFree_Training.RData' or cfg['raw']['sha256']!=RAW_SHA:
        raise PermissionError('Only exact existing normal Training source allowed')
    if sha(raw)!=RAW_SHA or raw.stat().st_size!=24678017:raise ValueError('Raw normal source differs')
    prior=read_ref(cfg['target_normal_manifest']);validate_roles(prior)
    lineage=read_ref(cfg['raw_cache_lineage']);audit=read_ref(cfg['normal_audit'])
    official=read_ref(cfg['official_metadata'])
    if sha(cfg['dictionary']['path'])!=cfg['dictionary']['sha256']:raise ValueError('Process dictionary differs')
    dictionary=Path(cfg['dictionary']['path']).read_text()
    descriptions={'xmeas_9':('XMEAS(9)','Reactor Temperature','Deg C'),
        'xmeas_21':('XMEAS(21)','Reactor Cooling Water Outlet Temp','Deg C'),
        'xmv_10':('XMV(10)','Reactor Cooling Water Flow',None)}
    semantics={}
    for variable,(token,meaning,unit) in descriptions.items():
        matches=[(i+1,line.strip()) for i,line in enumerate(dictionary.splitlines()) if token in line and meaning in line]
        if len(matches)!=1:raise ValueError('Variable physical meaning not closed by dictionary')
        semantics[variable]={'description':meaning,'unit':unit or 'UNSPECIFIED_IN_CACHED_DICTIONARY',
            'dictionary_line':matches[0][0],'dictionary_excerpt':matches[0][1],
            'causal_exogeneity':'not assumed; manipulated variable may reflect closed-loop feedback' if variable=='xmv_10' else 'not a causal claim'}
    version=official['data']['latestVersion']
    desc=next(x for x in version['metadataBlocks']['citation']['fields'] if x['typeName']=='dsDescription')['value'][0]['dsDescriptionValue']['value']
    if not all(text in desc for text in ('every 3 minutes','500','960','1 and 8 hours')):
        raise ValueError('Sampling/fault timing metadata not confirmed')
    normal_file=next(x['dataFile'] for x in version['files'] if x['dataFile']['filename']==raw.name)
    if normal_file['id']!=3031241 or normal_file['md5']!='ec126484534331f85001d8c4ebce6d17':
        raise ValueError('Official normal file identity differs')
    import numpy as np
    import pyreadr
    start=time.monotonic();out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'derived').mkdir()
    def budget():
        if time.monotonic()-start>300 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>2*1024**3:
            raise RuntimeError('Normal-only preparation budget exceeded; preserve partial')
    # Do not use read-directory/glob, and never read Faulty/Testing files.
    objects=pyreadr.read_r(str(raw))
    if set(objects)!={'fault_free_training'}:raise ValueError('Unexpected R object')
    frame=objects['fault_free_training']
    if frame.shape!=(250000,55) or list(frame.columns)!=audit['columns'] or frame.isna().any().any():
        raise ValueError('Actual dataframe schema/missing values differ')
    if not frame['faultNumber'].eq(0).all() or frame.duplicated(['simulationRun','sample']).any():
        raise ValueError('Normal/sample identity violation')
    group=frame.groupby('simulationRun',sort=False)
    if len(group)!=500 or not group.size().eq(500).all():raise ValueError('Expected normal500 runs')
    for rid,run in group:
        if run['sample'].tolist()!=list(range(1,501)):raise ValueError('Original sample sequence changed')
    positions={name:int(frame.columns.get_loc(name)) for name in VARIABLES}
    cache_path=Path(lineage['cache']['path'])
    if lineage['raw']['sha256']!=RAW_SHA or lineage['cache']['sha256']!=CACHE_SHA or sha(cache_path)!=CACHE_SHA:
        raise ValueError('Previously verified cache lineage differs')
    cache=np.load(cache_path,mmap_mode='r',allow_pickle=False)
    if cache.shape!=(100,500,52) or cache.dtype!=np.float32:raise ValueError('Expected legacy rounded cache')
    records=[];fit=[]
    for role,ids in ROLES.items():
        directory=out/'derived'/role;directory.mkdir()
        for rid in ids:
            old=next(x for x in prior['records'] if x['selector']['simulationRun']==rid)
            if old['role']!=role:raise ValueError('Role shift rejected')
            columns=old['columns'];mapping=[columns.index(name) for name in VARIABLES]
            values=group.get_group(rid).loc[:,list(VARIABLES)].to_numpy(copy=True)
            if values.dtype!=np.float64 or values.shape!=(500,3) or not np.isfinite(values).all():
                raise ValueError('Source values are not original finite float64')
            if not np.array_equal(values.astype(np.float32),cache[old['cache_row']][:,mapping]):
                raise ValueError('Selected original columns disagree with validated legacy cache')
            filename=directory/f'run{rid:03d}.npy';np.save(filename,values,allow_pickle=False)
            stored=np.load(filename,allow_pickle=False)
            if not np.array_equal(stored,values):raise ValueError('Float64 extraction roundtrip differs')
            records.append({'role':role,'simulationRun':rid,'faultNumber':0,'raw_source':str(raw),
                'raw_sha256':RAW_SHA,'r_object':'fault_free_training','column_names':list(VARIABLES),
                'raw_dataframe_columns_zero_based':positions,'old_cache_columns_by_actual_names':mapping,
                'old_cache_row':old['cache_row'],'float32_cast_matches_legacy_cache':True,
                'array_path':str(filename),'sha256':sha(filename),'bytes':filename.stat().st_size,
                'shape':[500,3],'dtype':'float64','imputation':False,'standardization':False,
                'array_C_contiguous':bool(values.flags.c_contiguous),'array_F_contiguous':bool(values.flags.f_contiguous),
                'sample_sequence_1based':[1,500,1],'sample_count':500,
                'timestamp_axis':'sample/elapsed_minutes files; no real datetime supplied',
                'sampling_minutes':3,'elapsed_minutes_origin':'first observed sample, zero',
                'new_history_or_window_selected':False})
            if role=='fit':fit.append(values)
            budget()
    np.save(out/'derived/sample_1based.npy',np.arange(1,501,dtype=np.int64),allow_pickle=False)
    np.save(out/'derived/elapsed_minutes_from_first_sample.npy',3*np.arange(500,dtype=np.float64),allow_pickle=False)
    combined=np.concatenate(fit);means=combined.mean(axis=0);scales=combined.std(axis=0,ddof=0)
    if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Invalid FIT scales')
    dump(out/'FIT_statistics.json',{'role':'fit_only','points':15000,'column_names':list(VARIABLES),
        'mean':means.tolist(),'std_ddof0':scales.tolist(),'values_in_derived_files_remain_raw':True,
        'SELECT_CAL_statistics_not_used':True})
    profiles=lag_profiles(fit)
    with (out/'FIT_past_lag_profiles.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(profiles[0]));w.writeheader();w.writerows(profiles)
    dump(out/'window_and_schedule_audit.json',window_schedule())
    dump(out/'asset_manifest.json',{'status':'NORMAL_ONLY_DERIVED_ASSETS_NO_MODEL','records':records,
        'variables':semantics,'raw_run_count_available':500,'selected_existing_roles':ROLES,
        'unused_runs_not_reassigned':True,'normal_raw_columns':list(frame.columns),
        'cache_note':'legacy cache float32; new values taken from original RData float64, not upcast',
        'source_metadata':{k:cfg[k] for k in ('raw','target_normal_manifest','raw_cache_lineage','normal_audit','official_metadata','dictionary')},
        'official_version':[version['versionNumber'],version['versionMinorNumber']],
        'sampling_frequency':'one sample / 3 min; absolute datetime unavailable',
        'dictionary_scope':'classic TE process semantics reference, not proof of identical Rieth implementation',
        'test_or_faulty_file_opened':False,'SELECT_CAL_model_performance_computed':False})
    budget()
    dump(out/'report.json',{'status':'COMPLETED_PENDING_REVIEW','wall_seconds':time.monotonic()-start,
        'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'GPU_hours':0,
        'normal_runs_written':len(records),'fit_runs':30,'select_runs':20,'cal_runs':20,
        'pyreadr_version':pyreadr.__version__,'numpy_version':np.__version__,
        'lag_profile_rows':len(profiles),'raw_shape':[250000,55],'no_model_or_test_data':True})
    return {'output':str(out),'records':len(records),'status':'COMPLETED_PENDING_REVIEW'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    print(json.dumps(extract(args.config) if args.execute else {'status':'PREVIEW_ONLY','config':json.loads(Path(args.config).read_text())},indent=2))


if __name__=='__main__':main()
