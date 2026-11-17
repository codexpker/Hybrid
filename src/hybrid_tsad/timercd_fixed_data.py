"""Original-unit inputs for fixed TimeRCD PSM + all SMD28 comparison.

CPU data binding only. Source hashes and old role bounds remain authoritative;
no label or model score is opened. FLOAT64 raw values are not inverse estimates
from the existing FLOAT32 standardized arrays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

OLD_SHA='d10cd11eca48f570f0e4a122f133cd8be60792c743ba68e0554f293ec4e82149'
FULL_SHA='501184abd4dbd09e9d18a60cc84c88de05eb77f088f706286bb111b28fb51c73'
IDENTITY='timercd_fixed_20261117_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as handle:
        while block:=handle.read(1024*1024):h.update(block)
    return h.hexdigest()


def dump(path,value):
    with Path(path).open('x',encoding='utf8')as handle:json.dump(value,handle,indent=2,allow_nan=False)


def local(w,path):
    text=str(path).replace('\\','/')
    remote='/home/a312/tsad_research/'
    if text.startswith(remote):return w/text[len(remote):]
    marker='/research_workspace/'
    if marker in text:return w/text.split(marker,1)[1]
    return Path(text)if Path(text).is_absolute()else w/text


def check(path,expected):
    if sha(path)!=expected:raise PermissionError('Pinned data identity mismatch: '+str(path))


def records(w):
    old=w/'preparations/timeradar_cross_variable_20261116_v1/data_binding.json'
    full=w/'preparations/timeradar_smd_full_20261116_v1/data_binding_full28.json'
    check(old,OLD_SHA);check(full,FULL_SHA)
    result=[json.loads(old.read_text())['records'][0]]+json.loads(full.read_text())['records']
    if len(result)!=29 or len({r['id']for r in result})!=29:raise ValueError('Exactly PSM plus official complete SMD28 required')
    return result


def raw_sources(w,record):
    sources=record['source'][:1 if record['id']=='PSM'else 2];paths=[]
    for item in sources:
        p=local(w,item['path'])
        if not p.exists()and record['id']=='machine-1-1'and item==sources[0]:
            p=w/'data/smd_machine_1_1_train_20261029/machine-1-1_train.txt'
        check(p,item['sha256']);paths.append(p)
    if record['id']=='PSM':
        # The long-format source contains a separate label channel. Read only
        # feature rows; it is never used to choose a role, point or statistic.
        frame=pd.read_csv(paths[0],usecols=['data','cols'])
        n=record['train_points']+record['query_points']
        values=[]
        for c in record['channel_order']:
            v=frame.loc[frame['cols']==c,'data'].to_numpy(dtype=np.float64)
            if len(v)!=n:raise ValueError('PSM fixed feature mapping/length changed')
            values.append(v)
        matrix=np.column_stack(values)
        return matrix[:record['train_points']],matrix[record['train_points']:],paths
    train,test=[np.loadtxt(p,delimiter=',',dtype=np.float64)for p in paths]
    return train,test,paths


def validate_raw_role(raw,old_x,mean,scale):
    """Reproduce old physical role bitwise; never derive new raw by inversion."""
    if raw.dtype!=np.float64 or raw.shape!=old_x.shape or not np.isfinite(raw).all():
        raise ValueError('Finite original-unit float64 role shape required')
    if not np.array_equal(((raw-mean)/scale).astype(np.float32),old_x):
        raise ValueError('Raw source no longer reproduces the accepted role values')


def prepare_record(w,record):
    began=time.monotonic();rid=record['id'];train,test,sources=raw_sources(w,record)
    if (train.shape!=(record['train_points'],record['channels'])or test.shape!=(record['query_points'],record['channels'])
            or not np.isfinite(train).all()or not np.isfinite(test).all()):raise ValueError('Original source shape/finite mismatch')
    scaler=local(w,record['scaler']['path']);check(scaler,record['scaler']['sha256'])
    with np.load(scaler,allow_pickle=False)as z:mean,scale=z['mean'].copy(),z['scale'].copy()
    physical={r['role']:r for r in record['roles']};lo,hi=physical['FIT']['bounds_half_open']
    fit=train[lo:hi];fitmean=fit.mean(axis=0);fitstd=fit.std(axis=0,ddof=0)
    if not np.allclose(fitmean,mean,rtol=1e-12,atol=1e-12):raise ValueError('Pinned FIT mean mismatch')
    out=w/'data'/IDENTITY/rid;out.mkdir(parents=True,exist_ok=False)
    stats=out/'fit_statistics.npz'
    np.savez(stats,mean=fitmean,std_ddof0=fitstd,official_safe_std=np.where(fitstd==0,1e-8,fitstd),
             original_standard_scaler_mean=mean,original_standard_scaler_scale=scale,fit_bounds=np.array([lo,hi]))
    roles=[]
    for name in ('FIT','CAL','QUERY'):
        old=physical[name];a,b=old['bounds_half_open'];x=(test if name=='QUERY'else train)[a:b]
        original=local(w,old['path']);check(original,old['sha256'])
        with np.load(original,allow_pickle=False)as z:oldx,ix=z['x'].copy(),z['index'].copy()
        if not np.array_equal(ix,np.arange(a,b,dtype=np.int64)):raise ValueError('Original role index changed')
        validate_raw_role(x,oldx,mean,scale)
        path=out/(name+'.npz');np.savez_compressed(path,x=x,index=ix)
        roles.append(dict(role=name,path=path.relative_to(w).as_posix(),sha256=sha(path),shape=list(x.shape),
                          bounds_half_open=[a,b],source_partition='test'if name=='QUERY'else'train',dtype='float64',
                          units='original source values, no normalization',old_role_path=old['path'],old_role_sha256=old['sha256'],
                          raw_to_old_standardized_float32_bitexact=True))
    result=dict(id=rid,dataset=record['dataset'],channels=record['channels'],channel_order=record['channel_order'],
        train_points=len(train),query_points=len(test),source=[dict(path=p.relative_to(w).as_posix(),sha256=sha(p))for p in sources],
        roles=roles,fit_statistics=dict(path=stats.relative_to(w).as_posix(),sha256=sha(stats),fit_bounds=[lo,hi],
            definition='Original raw FIT mean/std(ddof0); official_safe_std uses std==0 -> 1e-8',
            old_standard_scaler_path=record['scaler']['path'],old_standard_scaler_sha256=record['scaler']['sha256']),
        CPU_seconds=time.monotonic()-began,labels_decoded=False,GPU_hours=0,source_sha256=sha(__file__))
    dump(out/'RECEIPT.json',result);return result


def merge(w):
    items=[]
    for original in records(w):
        p=w/'data'/IDENTITY/original['id']/'RECEIPT.json';r=json.loads(p.read_text())
        if r['id']!=original['id']or r['labels_decoded']:raise PermissionError('Wrong raw record receipt')
        for role in r['roles']:check(local(w,role['path']),role['sha256'])
        check(local(w,r['fit_statistics']['path']),r['fit_statistics']['sha256'])
        items.append(r)
    out=w/'preparations'/IDENTITY;out.mkdir(parents=True,exist_ok=True)
    path=out/'data_binding.json'
    dump(path,dict(id=IDENTITY,records=items,expected_records=29,old_binding_sha256=OLD_SHA,full28_binding_sha256=FULL_SHA,
        labels_provided=False,historical_exposure='PSM development and one exposed SMD domain; not blind confirmation',
        track_A='Native TimeRCD separately normalizes full CAL/full QUERY raw roles; offline whole-role statistics',
        track_B='FIT raw mean/std once; bypass native role-wise normalization; same original role indices',
        source_sha256=sha(__file__)))
    return path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--record',action='append');p.add_argument('--merge',action='store_true');a=p.parse_args();w=a.workspace.resolve()
    if a.record:
        lookup={r['id']:r for r in records(w)}
        for rid in a.record:prepare_record(w,lookup[rid])
    if a.merge:print(merge(w))
