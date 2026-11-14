"""Single label-free CPU producer for shared FIT scalers and fixed controls."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from hybrid_tsad.hybrid_v1.data import load_manifest
from hybrid_tsad.roca_cpu_controls import (METHODS,standardized_run,fixed_endpoint_score,
    fixed_calibration,write_score_array,original_long_CAL)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def asset(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def bound(spec):
    if sha(spec['path'])!=spec['sha256']:raise PermissionError('Fixed control input changed')
    return json.loads(Path(spec['path']).read_text())

def prepare(c):
    import sklearn
    from sklearn.preprocessing import StandardScaler
    out=Path(c['scaler_output']);out.mkdir(parents=True,exist_ok=False);rows=[]
    for name,spec in c['FIT_manifests'].items():
        bound(spec);runs=load_manifest(spec['path'],roles={'FIT'})
        for run in runs:
            if not run.observed.all():raise ValueError('No new missing data policy')
            fitted=StandardScaler().fit(np.asarray(run.values,np.float64))
            p=out/(run.domain+'__'+run.entity+'.npz')
            np.savez_compressed(p,mean=fitted.mean_,scale=fitted.scale_,variance=fitted.var_,n_samples_seen=fitted.n_samples_seen_)
            rows.append(dict(domain=run.domain,entity=run.entity,FIT_run=run.run,FIT_points=len(run.values),
                scaler=asset(p),FIT_manifest=spec,method='sklearn.StandardScaler.fit(FITfloat64); transformfloat64->float32'))
    if len(rows)!=54 or len({(r['domain'],r['entity']) for r in rows})!=54:raise ValueError('All53 SMAP plus one SMD scaler required')
    smd=bound(c['SMD_manifest']);cal=next(r for r in smd['runs'] if r['role']=='CAL')
    start=cal['original_start'];end=start+cal['expected_shape'][0]
    indices=np.linspace(start+512,end-1,512,dtype=np.int64)
    if len(set(indices.tolist()))!=512:raise ValueError('Distinct512 original CAL endpoints required')
    plan=[dict(domain=cal['domain'],entity=cal['entity'],run=cal['run'],role='CAL',index=int(t)) for t in indices]
    dump(out/'SMD_CAL512.PLAN.json',plan)
    dump(out/'SCALERS.MANIFEST.json',dict(scalers=rows,sklearn_version=sklearn.__version__,numpy_version=np.__version__,
        sklearn_source=asset(inspect.getfile(StandardScaler)),source_script=asset(__file__),SMD_CAL512=asset(out/'SMD_CAL512.PLAN.json'),
        QUERY_read=False,labels_read=False,normal_health_certified=False))
    return dict(scalers=54,manifest=asset(out/'SCALERS.MANIFEST.json'),SMD_CAL512=asset(out/'SMD_CAL512.PLAN.json'))

def run(c):
    began=time.monotonic();out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    common=bound(c['scalers']);scalers={ (r['domain'],r['entity']):r for r in common['scalers']};allbanks=[];cal_reports=[]
    for domain,job in c['datasets'].items():
        bound(job['CAL_manifest']);bound(job['QUERY_manifest']);plan=bound(job['CAL_plan']);width=job['window']
        if (domain,width) not in [('SMAP',16),('SMD',32)]:raise PermissionError('Fixed short windows only')
        calruns=load_manifest(job['CAL_manifest']['path'],roles={'CAL'});queries=load_manifest(job['QUERY_manifest']['path'],roles={'QUERY'})
        calpoints={}
        for p in plan:calpoints.setdefault((p['domain'],p['entity'],p['run']),[]).append(p['index'])
        short={m:[] for m in METHODS};standard_queries=[]
        for run in calruns+queries:
            desc=scalers[run.domain,run.entity];spec=desc['scaler']
            if sha(spec['path'])!=spec['sha256']:raise PermissionError('Shared FIT scaler changed')
            with np.load(spec['path'],allow_pickle=False) as z:rr=standardized_run(run,z['mean'],z['scale'])
            if run.role=='QUERY':standard_queries.append((rr,desc));continue
            indices=calpoints[run.domain,run.entity,run.run]
            if len(indices)!=(32 if domain=='SMAP' else 512):raise PermissionError('Common fixed CAL endpoint count')
            for method in METHODS:short[method].extend(fixed_endpoint_score(rr,int(t),width,method) for t in indices)
        calibrations={m:fixed_calibration(v) for m,v in short.items()}
        for method in METHODS:
            calfile=out/domain/(method+'_CAL.npz');calfile.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(calfile,score=np.array([r['score'] for r in short[method]]),
                index=np.array([r['index'] for r in short[method]],np.int64),entity=np.array([r['entity'] for r in short[method]]))
            cal_reports.append(dict(domain=domain,method=method,calibration=calibrations[method],CAL_plan=job['CAL_plan'],scores=asset(calfile)))
        for rr,desc in standard_queries:
            # Preserve the old public QUERY axis; no new warmup privilege.
            indices=np.arange(rr.start+512,rr.start+len(rr.values),dtype=np.int64)
            for method in METHODS:
                records=[fixed_endpoint_score(rr,int(t),width,method) for t in indices]
                bank=write_score_array(out/domain/rr.entity/method,records,calibrations[method],
                    dict(shared_scaler=desc['scaler'],CAL_plan=job['CAL_plan'],QUERY_manifest=job['QUERY_manifest'],window=width,
                         target_fit_track=True,source512_ZS_comparable_permissions=False))
                allbanks.append(dict(domain=domain,entity=rr.entity,run=rr.run,method=method,history=width-1,
                    calibration=calibrations[method],**bank))
        if domain=='SMD':
            run=calruns[0];indices=calpoints[run.domain,run.entity,run.run]
            for method,spec in c['original_long_source_calibrations'].items():
                cal=bound(spec);records=original_long_CAL(run,indices,method,cal);newcal=fixed_calibration(records)
                newcal.update(target_fit_scaler_reused=False,original_source_score_units=True)
                np.savez_compressed(out/domain/(method+'_H512_TARGET_CAL.npz'),score=np.array([r['score'] for r in records]),index=np.array(indices,np.int64))
                cal_reports.append(dict(domain=domain,method=method+'_H512_ORIGINAL_SCORE_UNITS_TARGET_CAL',calibration=newcal,
                    original_source_recipe=spec,existing_QUERY_bank=c['original_long_QUERY_banks'][method],
                    new_QUERY_computation=False,RoCA_scaler_applied=False,matched_short_track=False))
    dump(out/'BANKS.json',allbanks);dump(out/'CALIBRATIONS.json',cal_reports);dump(out/'CONFIG.json',c)
    dump(out/'RECEIPT.json',dict(status='ALL_RAW_SEALED_NO_LABELS',short_banks=len(allbanks),
        query_records=sum(r['count'] for r in allbanks),entities=54,methods=list(METHODS),GPUh=0,
        labels_read=False,new_neural_forward=False,source512_QUERY_reused=True,seconds=time.monotonic()-began))
    return dict(banks=len(allbanks),seconds=time.monotonic()-began)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--trusted-config-sha256',required=True);p.add_argument('--phase',choices=['prepare','run'],required=True);a=p.parse_args()
    if sha(a.config)!=a.trusted_config_sha256:raise PermissionError('Fixed CPU config required')
    c=json.loads(Path(a.config).read_text());print(json.dumps(prepare(c) if a.phase=='prepare' else run(c)),flush=True)

if __name__=='__main__':main()
