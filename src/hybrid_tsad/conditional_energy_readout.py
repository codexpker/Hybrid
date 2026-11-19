"""Fixed, disjoint readout shards and a guarded train/readout handoff.

The original training dispatcher/runtime remains unchanged. This reader calls
the same frozen generator/response/energy functions, with noise keyed by time.
"""
import argparse
import concurrent.futures
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import queue as queue_module
import signal
import struct
import subprocess
import sys
import time
import numpy as np
import torch
from .conditional_energy_runner import read,dump,sha,load_parts,arrays,pair,device_setup,hashes
from .conditional_response_energy import fixed_noise


def shard_axis(task,role,shard):
    lo,hi=task['score_axes'][role]
    # Exactly four fixed slices per role, no data-dependent scheduling or fit.
    edges=np.linspace(lo,hi,5,dtype=np.int64)
    return np.arange(edges[shard],edges[shard+1],dtype=np.int64)


def worker(w,root,domain,arm,phase,role,shard):
    start=time.monotonic();device=device_setup();d=root/domain
    assert read(root/'ALL_TRAINING_FROZEN.json')['training_jobs']==8
    task,g,geo,cfg=load_parts(w,d,device);folder=d/arm/phase
    receipt=read(folder/'TRAIN_RECEIPT.json');checkpoint=w/receipt['checkpoint_relative']
    assert sha(checkpoint)==receipt['checkpoint_sha256']
    g.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['model'],strict=True);g.eval()
    before=hashes(g);teacher=hashes(geo.native)
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(device).clone()
    data=arrays(w,d,task,[role])[role];times=shard_axis(task,role,shard)
    scores=np.empty(len(times),dtype=np.float32);out=folder/'readout_v2'/role/f'shard{shard}'
    out.mkdir(parents=True,exist_ok=False)
    with torch.no_grad():
        for i,t in enumerate(times):
            h,y=pair(data,t,device);noise=torch.Generator().manual_seed(192025+int(t))
            v=g(h,*fixed_noise(1,h.shape[2],noise,device=device))
            scores[i]=float(geo.score(h,v,y,arm=='A')[0])
            if(i+1)%10000==0:
                print(json.dumps(dict(event='SCORE_PROGRESS',role=role,shard=shard,points=i+1,total=len(times),elapsed=time.monotonic()-start)),flush=True)
    assert np.isfinite(scores).all()
    assert hashes(g)==before and hashes(geo.native)==teacher
    assert torch.equal(cpu,torch.get_rng_state())and torch.equal(cuda,torch.cuda.get_rng_state(device))
    path=out/'bank.npz';np.savez_compressed(path,absolute_time_index=times,available_index=times,
        canonical_score=scores,role_window_start=times-512,response_window_start=times-99)
    dump(out/'RECEIPT.json',dict(status='COMPLETED',domain=domain,arm=arm,phase=phase,role=role,shard=shard,
        relative=str(path.relative_to(w)),sha256=sha(path),points=len(times),axis=[int(times[0]),int(times[-1])+1],
        checkpoint_sha256=sha(checkpoint),teacher_checkpoint_sha256=cfg['checkpoint_sha256'],
        task_sha256=sha(d/'TASK.json'),runtime_manifest_sha256=sha(root/'RUNTIME_MANIFEST.json'),
        readout_module_sha256=sha(Path(__file__)),model_and_teacher_immutable=True,CPU_CUDA_RNG_unchanged=True,
        labels_read=False,elapsed_GPU_seconds=time.monotonic()-start,peak_GiB=torch.cuda.max_memory_allocated()/2**30))
    print(json.dumps(dict(event='SHARD_COMPLETED',domain=domain,arm=arm,phase=phase,role=role,shard=shard)),flush=True)


def merge(w,root):
    allbanks=[]
    for domain in ['SWAT','PSM']:
        task=read(root/domain/'TASK.json')
        for arm in ['A','B']:
            for phase in ['source','R']:
                folder=root/domain/arm/phase;train=read(folder/'TRAIN_RECEIPT.json');banks=[];parts=[]
                for role in ['CAL','QUERY']:
                    collections={k:[]for k in ['absolute_time_index','available_index','canonical_score','role_window_start','response_window_start']}
                    for shard in range(4):
                        r=read(folder/'readout_v2'/role/f'shard{shard}'/'RECEIPT.json');path=w/r['relative']
                        assert sha(path)==r['sha256'] and r['checkpoint_sha256']==train['checkpoint_sha256']
                        assert r['CPU_CUDA_RNG_unchanged']and r['model_and_teacher_immutable']and not r['labels_read']
                        with np.load(path,allow_pickle=False)as z:
                            assert np.array_equal(z['absolute_time_index'],shard_axis(task,role,shard))
                            for k in collections:collections[k].append(z[k].copy())
                        parts.append(dict(relative=str((folder/'readout_v2'/role/f'shard{shard}'/'RECEIPT.json').relative_to(w)),sha256=sha(folder/'readout_v2'/role/f'shard{shard}'/'RECEIPT.json')))
                    combined={k:np.concatenate(v)for k,v in collections.items()};lo,hi=task['score_axes'][role]
                    assert np.array_equal(combined['absolute_time_index'],np.arange(lo,hi))
                    assert np.array_equal(combined['available_index'],combined['absolute_time_index'])
                    assert np.array_equal(combined['role_window_start'],combined['absolute_time_index']-512)
                    path=folder/(role+'.npz');assert not path.exists(), 'Never overwrite old bank'
                    np.savez_compressed(path,**combined)
                    b=dict(relative=str(path.relative_to(w)),sha256=sha(path),role=role,points=hi-lo)
                    banks.append(b);allbanks.append(dict(b,domain=domain,arm=arm,phase=phase))
                dump(folder/'RAW_BANK_SEAL.json',dict(status='COMPLETED',banks=banks,parts=parts,labels_read=False,
                    checkpoint_sha256=train['checkpoint_sha256'],immutable_model_and_teacher=True,readout_version='fixed_four_disjoint_shards'))
    return allbanks


def run_queue(w,root):
    assert read(root/'HANDOFF.json')['status']=='SAFE_HANDOFF'
    tasks=[]
    for domain in ['SWAT','PSM']:
        task=read(root/domain/'TASK.json')
        for arm in ['A','B']:
            for phase in ['source','R']:
                for role in ['CAL','QUERY']:
                    for shard in range(4):tasks.append((domain,arm,phase,role,shard))
    # Largest fixed mathematical work first, using only axis lengths and arm.
    tasks.sort(key=lambda x:len(shard_axis(read(root/x[0]/'TASK.json'),x[3],x[4]))*(20 if x[1]=='A'else 1),reverse=True)
    q=queue_module.Queue()
    for item in tasks:q.put(item)
    sys.path.insert(0,str(w/'preparations/hybrid_left_joint_cycle_20261109_v2/resources'))
    import owned_session
    from gpu_idle_guard import gpu_guard
    devices={int(v.split(',')[0]):v.split(',')[1].strip()for v in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()}
    def lane(gpu):
        results=[]
        while not q.empty():
            try:item=q.get_nowait()
            except queue_module.Empty:break
            domain,arm,phase,role,shard=item;identity='_'.join(map(str,item))
            life=root/'readout_lifecycles_v2'/identity;life.mkdir(parents=True,exist_ok=False)
            start=time.monotonic();child=None;exit_code=None;cleanup=None;error=None
            try:
                dump(life/'GPU_GUARD.json',gpu_guard(devices[gpu].removeprefix('GPU-'),gpu,20))
                argv=[sys.executable,'-B','-m','hybrid_tsad.conditional_energy_readout','worker','--workspace',str(w),'--output',str(root),
                    '--domain',domain,'--arm',arm,'--phase',phase,'--role',role,'--shard',str(shard)]
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
                with(life/'stdout.log').open('xb')as stdout,(life/'stderr.log').open('xb')as stderr:
                    child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=env,start_new_session=True);owned_session.claim_group(child)
                    dump(life/'STARTED.json',dict(pid=child.pid,GPU=gpu,argv=argv,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),timeout_seconds=28800))
                    print(json.dumps(dict(event='RUNNING',identity=identity,pid=child.pid,GPU=gpu)),flush=True)
                    try:exit_code=child.wait(timeout=28790)
                    finally:cleanup=owned_session.stop_group(child,grace=5)
                assert exit_code==0 and cleanup['status']=='NO_LIVE_GROUP_MEMBERS'
            except BaseException as exc:error=repr(exc)
            r=dict(identity=identity,status='COMPLETED'if error is None else'FAILED',OS_exit_code=exit_code,error=error,cleanup=cleanup,
                actual_GPU_seconds=time.monotonic()-start if child else 0.,guard_elapsed_seconds=time.monotonic()-start if not child else None,
                UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=gpu,pid=child.pid if child else None)
            dump(life/'OS_WAIT.json',r);results.append(r)
            if error:raise RuntimeError(r)
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:groups=list(pool.map(lane,range(4)))
    readout_jobs=[job for group in groups for job in group];assert len(readout_jobs)==64
    banks=merge(w,root)
    trainjobs=[read(path)for path in (root/'lifecycles').glob('*/OS_WAIT.json')]
    assert len(trainjobs)==10 and all(j['status']=='COMPLETED'for j in trainjobs)
    dump(root/'GLOBAL_RAW_BANK_SEAL.json',dict(status='COMPLETED',banks=banks,jobs=trainjobs+readout_jobs,no_labels_decoded=True,
        actual_GPU_hours=sum(j['actual_GPU_seconds']for j in trainjobs+readout_jobs)/3600,
        original_dispatcher='stopped after all training before any score Popen; HANDOFF receipt preserved',readout_version='v2_fixed_shards'))
    print(json.dumps(dict(event='STAGE_COMPLETED',banks=len(banks),readout_jobs=len(readout_jobs))),flush=True)


def live_descendants(pid):
    """All thread children, including the Popen pre-exec fork interval."""
    seen=set();pending=[pid]
    while pending:
        parent=pending.pop();base=Path(f'/proc/{parent}/task')
        if not base.exists():continue
        for task in base.iterdir():
            try:children=(task/'children').read_text().split()
            except FileNotFoundError:continue # a child/task exited during read
            for item in children:
                child=int(item)
                if child not in seen and Path(f'/proc/{child}').exists():
                    seen.add(child);pending.append(child)
    return sorted(seen)


def watch(w,root,pid):
    """Pause only our dispatcher on close-write, verify no score child, hand off.

    If the close-write/Popen race was lost, resume the original dispatcher and
    explicitly fall back. Never kill or overwrite a launched scoring worker.
    """
    expected=b'hybrid_tsad.conditional_energy_runner\x00queue'
    assert expected in Path(f'/proc/{pid}/cmdline').read_bytes()
    if (root/'ALL_TRAINING_FROZEN.json').exists():
        dump(root/'HANDOFF.json',dict(status='ORIGINAL_QUEUE_CONTINUED',reason='Training-close event already occurred before watcher installation'))
        return
    lib=ctypes.CDLL('libc.so.6',use_errno=True);fd=lib.inotify_init1(0)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    assert lib.inotify_add_watch(fd,str(root).encode(),8)>=0 # IN_CLOSE_WRITE
    if (root/'ALL_TRAINING_FROZEN.json').exists():
        dump(root/'HANDOFF.json',dict(status='ORIGINAL_QUEUE_CONTINUED',reason='Training-close event raced watcher installation'))
        return
    dump(root/'HANDOFF_WATCH_STARTED.json',dict(status='WATCHING',dispatcher_pid=pid,watcher_pid=os.getpid(),event='IN_CLOSE_WRITE ALL_TRAINING_FROZEN.json',readout_module_sha256=sha(Path(__file__))))
    while True:
        chunk=os.read(fd,8192);offset=0;hit=False
        while offset<len(chunk):
            wd,mask,cookie,length=struct.unpack_from('iIII',chunk,offset);offset+=16
            name=chunk[offset:offset+length].split(b'\x00',1)[0];offset+=length
            if name==b'ALL_TRAINING_FROZEN.json':hit=True
        if hit:break
    os.kill(pid,signal.SIGSTOP)
    try:
        frozen=read(root/'ALL_TRAINING_FROZEN.json');assert frozen['training_jobs']==8
        waits=[read(p)for p in (root/'lifecycles').glob('train_*/OS_WAIT.json')]
        assert len(waits)==8 and all(r['status']=='COMPLETED'for r in waits)
        score_started=list((root/'lifecycles').glob('score_*/STARTED.json'))
        descendants=live_descendants(pid)
        score_pids=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:cmd=(proc/'cmdline').read_bytes()
            except(OSError,PermissionError):continue
            if b'hybrid_tsad.conditional_energy_runner\x00score' in cmd and str(root).encode()in cmd:score_pids.append(int(proc.name))
        if score_started or score_pids or descendants:
            dump(root/'HANDOFF.json',dict(status='ORIGINAL_QUEUE_RESUMED',reason='Score/fork/child launch won close-write race',started=[str(p.relative_to(root))for p in score_started],score_pids=score_pids,live_descendants=descendants))
            os.kill(pid,signal.SIGCONT);return
        os.kill(pid,signal.SIGKILL)
        dump(root/'HANDOFF.json',dict(status='SAFE_HANDOFF',dispatcher_pid=pid,dispatcher_signal='SIGKILL while stopped after all8 training OS_WAIT',
            no_score_START_present=True,no_score_process_present=True,all_training_frozen_sha256=sha(root/'ALL_TRAINING_FROZEN.json'),
            no_live_descendants=True,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),readout_module_sha256=sha(Path(__file__))))
    except BaseException:
        if Path(f'/proc/{pid}').exists():os.kill(pid,signal.SIGCONT)
        raise
    run_queue(w,root)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['watch','queue','worker']);p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--pid',type=int);p.add_argument('--domain');p.add_argument('--arm');p.add_argument('--phase');p.add_argument('--role');p.add_argument('--shard',type=int)
    a=p.parse_args()
    if a.action=='watch':watch(a.workspace,a.output,a.pid)
    elif a.action=='queue':run_queue(a.workspace,a.output)
    else:worker(a.workspace,a.output,a.domain,a.arm,a.phase,a.role,a.shard)


if __name__=='__main__':main()
