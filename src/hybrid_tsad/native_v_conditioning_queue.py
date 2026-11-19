"""Bounded owned process orchestration for the single approved protocol."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .native_v_conditioning_runner import read,dump,merge_cache,freeze_trainings,readout,EXPERIMENT
from .native_v_conditioning import sha


def run_jobs(workspace,root,stage,tasks):
    sys.path.insert(0,str(workspace/'preparations/hybrid_left_joint_cycle_20261109_v2/resources'))
    import owned_session
    from gpu_idle_guard import gpu_guard
    devices={int(s.split(',')[0]):s.split(',')[1].strip()for s in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
    def lane(gpu,items):
        results=[]
        for identity,extra,cap,timeout in items:
            life=root/'lifecycle'/stage/identity;life.mkdir(parents=True,exist_ok=False)
            start=time.monotonic();child=None;code=None;cleanup=None;error=None
            try:
                dump(life/'GPU_GUARD.json',gpu_guard(devices[gpu].removeprefix('GPU-'),gpu,cap))
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                argv=[sys.executable,'-B','-m','hybrid_tsad.native_v_conditioning_runner']+extra+['--workspace',str(workspace),'--output',str(root)]
                with(life/'stdout.log').open('xb')as out,(life/'stderr.log').open('xb')as err:
                    child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned_session.claim_group(child)
                    dump(life/'STARTED.json',dict(pid=child.pid,GPU=gpu,argv=argv,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_seconds=timeout,memory_cap_GiB=cap))
                    print(json.dumps(dict(event='RUNNING',stage=stage,identity=identity,pid=child.pid,GPU=gpu)),flush=True)
                    try:code=child.wait(timeout=timeout-10)
                    finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup['status']=='NO_LIVE_GROUP_MEMBERS'
            except BaseException as exc:error=repr(exc)
            result=dict(identity=identity,stage=stage,status='COMPLETED'if error is None else'FAILED',OS_exit_code=code,error=error,cleanup=cleanup,
                actual_GPU_seconds=time.monotonic()-start if child else 0.,guard_elapsed_seconds=time.monotonic()-start if not child else None,
                GPU=gpu,pid=child.pid if child else None,UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat())
            dump(life/'OS_WAIT.json',result);results.append(result)
            if error:raise RuntimeError(result)
        return results
    lanes=[tasks[i::4]for i in range(4)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:groups=list(pool.map(lambda pair:lane(*pair),enumerate(lanes)))
    return [r for group in groups for r in group]


def normal(workspace,root):
    tasks=[(domain+'_'+role,['cache','--domain',domain,'--role',role],20,1800)
        for domain in ['SWAT','PSM']for role in ['SOURCE_FIT','SOURCE_SELECT']]
    jobs=run_jobs(workspace,root,'normal_cache',tasks)
    rows=[]
    for domain in ['SWAT','PSM']:
        for role in ['SOURCE_FIT','SOURCE_SELECT']:
            merge_cache(workspace,root,domain,role);p=root/'cache'/domain/role/'SEALED.json'
            rows.append(dict(domain=domain,role=role,relative=str(p.relative_to(workspace)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    dump(root/'SOURCE_CACHES_SEALED.json',dict(status='COMPLETED',caches=rows,jobs=jobs,labels_read=False,
        actual_GPU_hours=sum(j['actual_GPU_seconds']for j in jobs)/3600))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='SOURCE_CACHES_SEALED',jobs=4)),flush=True)


def learning(workspace,root):
    assert read(root/'SOURCE_CACHES_SEALED.json')['status']=='COMPLETED'
    jobs=run_jobs(workspace,root,'engineering_gates',[(domain,['gates','--domain',domain],20,1800)for domain in ['SWAT','PSM']])
    dump(root/'GATES_COMPLETED.json',dict(status='COMPLETED',jobs=jobs))
    jobs+=run_jobs(workspace,root,'source_training',[(domain+'_'+method,['train','--domain',domain,'--method',method],20,14400)
        for domain in ['SWAT','PSM']for method in ['B','C','D']])
    freeze_trainings(workspace,root)
    dump(root/'TRAINING_PHASE_COMPLETED.json',dict(status='COMPLETED',jobs=jobs))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='ALL_SIX_MODELS_FROZEN',models=6)),flush=True)


def target(workspace,root):
    assert read(root/'ALL_SIX_MODELS_FROZEN.json')['status']=='COMPLETED'
    tasks=[(domain+'_'+role+'_'+str(part),['cache','--domain',domain,'--role',role,'--part',str(part),'--parts',str(parts)],30,7200)
        for domain in ['SWAT','PSM']for role,parts in [('CAL',1),('QUERY',4)]for part in range(parts)]
    jobs=run_jobs(workspace,root,'target_cache',tasks)
    for domain in ['SWAT','PSM']:
        for role in ['CAL','QUERY']:merge_cache(workspace,root,domain,role)
    dump(root/'TARGET_CACHES_SEALED.json',dict(status='COMPLETED',jobs=jobs,labels_read=False))
    tasks=[(domain+'_'+method+'_'+role,['readout','--domain',domain,'--method',method,'--role',role],30,7200)
        for domain in ['SWAT','PSM']for method in ['B','C','D']for role in ['CAL','QUERY']]
    jobs+=run_jobs(workspace,root,'predictor_readout',tasks)
    for domain in ['SWAT','PSM']:
        for role in ['CAL','QUERY']:readout(workspace,root,domain,'A',role)
    rows=[]
    for domain in ['SWAT','PSM']:
        for method in ['A','B','C','D']:
            for role in ['CAL','QUERY']:
                p=root/'banks'/domain/method/role/'RECEIPT.json';row=read(p)
                assert sha(workspace/row['relative'])==row['sha256']
                rows.append(dict(domain=domain,method=method,role=role,relative=row['relative'],sha256=row['sha256'],
                    receipt_relative=str(p.relative_to(workspace)),receipt_sha256=sha(p)))
    alljobs=read(root/'SOURCE_CACHES_SEALED.json')['jobs']+read(root/'TRAINING_PHASE_COMPLETED.json')['jobs']+jobs
    dump(root/'GLOBAL_RAW_BANK_SEAL.json',dict(status='COMPLETED',labels_read=False,no_labels_decoded=True,banks=rows,
        trainings=read(root/'ALL_SIX_MODELS_FROZEN.json')['trainings'],jobs=alljobs,
        actual_GPU_hours=sum(j['actual_GPU_seconds']for j in alljobs)/3600))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage='GLOBAL_RAW_BANK_SEAL',banks=16)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['normal','learning','target','remaining']);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();root=a.output or a.workspace/'results'/EXPERIMENT
    if a.stage=='normal':normal(a.workspace,root)
    elif a.stage=='learning':learning(a.workspace,root)
    elif a.stage=='target':target(a.workspace,root)
    else:learning(a.workspace,root);target(a.workspace,root)
