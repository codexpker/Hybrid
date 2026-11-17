"""Bounded zero-training queue; occupies only initially free A40 devices."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def dump(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2)


def inventory():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    occupied={x.split(',')[0].strip()for x in processes.splitlines()if x.startswith('GPU-')}
    rows=[]
    for line in lines.splitlines():
        gpu,uuid,used,free=[x.strip()for x in line.split(',')]
        rows.append(dict(gpu=int(gpu),uuid=uuid,used_MiB=int(used),free_MiB=int(free),occupied=uuid in occupied))
    return rows


def job(w,config,runtime,out,rid,arm,phase,gpu):
    info=next(x for x in inventory()if x['gpu']==gpu)
    if info['occupied']or info['used_MiB']>1000 or info['free_MiB']<32000:raise RuntimeError('Device busy; no external process eviction')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(runtime/'src'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    cmd=[sys.executable,'-B','-m','hybrid_tsad.timercd_fixed_runner',phase,'--workspace',str(w),'--config',str(config),'--record',rid,'--arm',arm,'--device','cuda:0']
    key=rid+'__'+arm+'__'+phase;begin=time.time();timeout=3600 if phase=='score' else 1200
    with(out/(key+'.log')).open('xb')as f:
        child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        launch=dict(pid=child.pid,record=rid,arm=arm,phase=phase,gpu=gpu,uuid=info['uuid'],start_unix=begin,command=cmd,timeout_seconds=timeout)
        dump(out/(key+'_LAUNCH.json'),launch);print(json.dumps(dict(event='START',**launch)),flush=True)
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            rc=124
    end=time.time();row=dict(launch,exit_code=rc,end_unix=end,seconds=end-begin,GPU_hours=(end-begin)/3600)
    dump(out/(key+'_EXIT.json'),row);print(json.dumps(dict(event='EXIT',**row)),flush=True)
    if rc:raise RuntimeError('Inference failed: '+key)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['parity','formal']);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);a=p.parse_args()
    w=a.workspace.resolve();config=a.config.resolve();runtime=a.runtime.resolve();cfg=json.loads(config.read_text());out=w/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    for rel,expected in json.loads((runtime/'RUNTIME_FREEZE.json').read_text())['files'].items():assert hashlib.sha256((runtime/rel).read_bytes()).hexdigest()==expected,rel
    assert len(cfg['records'])==29 and cfg['arms']==['A_official_role','B_fit_only']
    free=[x for x in inventory()if not x['occupied']and x['used_MiB']<1000 and x['free_MiB']>32000][:4]
    if not free:raise RuntimeError('No free GPU; never evict external processes')
    dump(out/(a.stage+'_CONTROLLER_START.json'),dict(pid=os.getpid(),inventory=inventory(),selected_devices=free,start_unix=time.time(),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
    if a.stage=='formal':assert(out/'PARITY_ALL_PASS.json').exists()
    completed=[]
    if a.stage=='parity':
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(2,len(free)))as pool:
            tasks=[pool.submit(job,w,config,runtime,out,r,'A_official_role','parity',free[i%len(free)]['gpu'])for i,r in enumerate(['PSM','machine-1-1'])]
            for task in tasks:completed.append(task.result())
        dump(out/'PARITY_ALL_PASS.json',dict(status='PASS_ACTUAL_OFFICIAL_API',jobs=completed,GPU_hours=sum(x['GPU_hours']for x in completed)))
    else:
        groups=[cfg['records'][i::len(free)]for i in range(len(free))]
        def worker(records,gpu):
            rows=[]
            for rid in records:
                for arm in cfg['arms']:rows.append(job(w,config,runtime,out,rid,arm,'score',gpu))
            return rows
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(free))as pool:
            tasks=[pool.submit(worker,records,device['gpu'])for records,device in zip(groups,free)]
            for task in concurrent.futures.as_completed(tasks):completed.extend(task.result())
        banks=[dict(path=str(path),relative=path.relative_to(out).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)for path in sorted(out.glob('*/*/*.npz'))if path.name in('CAL.npz','QUERY.npz')]
        assert len(completed)==58 and len(banks)==116
        dump(out/'RAW_BANK_SEAL.json',dict(status='COMPLETED',banks=banks,jobs=completed,GPU_hours=sum(x['GPU_hours']for x in completed),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),no_labels_decoded=True,optimizer_updates=0))
    print(json.dumps(dict(event='STAGE_COMPLETED',stage=a.stage,jobs=len(completed))),flush=True)


if __name__=='__main__':main()
