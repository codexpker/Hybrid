"""Four-card fixed-record queue; only this queue's owned process groups cleaned."""
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


def dump(path, value):
    with Path(path).open('x') as f: json.dump(value, f, indent=2)


def inventory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    busy = {line.split(',')[0].strip() for line in processes.splitlines() if line.startswith('GPU-')}
    result = []
    for line in rows.splitlines():
        gpu, uuid, used, free = [x.strip() for x in line.split(',')]
        result.append(dict(gpu=int(gpu), uuid=uuid, used_MiB=int(used), free_MiB=int(free), occupied=uuid in busy))
    return result


def run_job(workspace, config_path, runtime, output, record, arm, phase, gpu):
    info = next(x for x in inventory() if x['gpu'] == gpu)
    if info['occupied'] or info['used_MiB'] >= 1000 or info['free_MiB'] <= 32000:
        raise RuntimeError('GPU is occupied; external processes are never displaced: '+str(info))
    cfg = json.loads(config_path.read_text()); timeout = 7200 if phase == 'train' else (2700 if phase == 'score' else 600)
    env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(runtime/'src'), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    cmd = [sys.executable, '-B', '-m', 'hybrid_tsad.timeradar_adapter_only_runner', phase,
           '--workspace', str(workspace), '--config', str(config_path), '--record', record, '--arm', arm, '--device', 'cuda:0']
    key = record+'__'+arm+'__'+phase; begin = time.time()
    with (output/(key+'.log')).open('xb') as handle:
        child = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        launch = dict(pid=child.pid, gpu=gpu, uuid=info['uuid'], phase=phase, record=record, arm=arm, command=cmd,
                      start_unix=begin, timeout_seconds=timeout, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest())
        dump(output/(key+'_LAUNCH.json'), launch); print(json.dumps(dict(event='START', **launch)), flush=True)
        try:
            rc = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
            rc = 124
    end = time.time(); receipt = dict(launch, exit_code=rc, end_unix=end, seconds=end-begin, GPU_hours=(end-begin)/3600)
    dump(output/(key+'_EXIT.json'), receipt); print(json.dumps(dict(event='EXIT', **receipt)), flush=True)
    if rc != 0: raise RuntimeError('Owned job failed: '+key)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['mechanism', 'formal']); parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args(); w, config, runtime = args.workspace.resolve(), args.config.resolve(), args.runtime.resolve()
    cfg = json.loads(config.read_text()); output = w/cfg['output']; output.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((runtime/'RUNTIME_FREEZE.json').read_text())
    for rel, expected in freeze['files'].items():
        assert hashlib.sha256((runtime/rel).read_bytes()).hexdigest() == expected, rel
    assert cfg['seed'] == 2024 and cfg['updates'] == 2000 and cfg['arms'] == ['self', 'cross']
    devices = [x for x in inventory() if not x['occupied'] and x['used_MiB'] < 1000 and x['free_MiB'] > 32000]
    assert len(devices) == 4, devices
    records = list(cfg['records']); assert records == ['PSM', 'machine-1-1', 'machine-1-2', 'machine-1-3']
    if args.stage == 'formal': assert (output/'MECHANISM_ALL_PASS.json').exists()
    dump(output/(args.stage+'_CONTROLLER_START.json'), dict(pid=os.getpid(), start_unix=time.time(), inventory=devices,
                                                          config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
    def worker(record, gpu):
        jobs = []
        for arm in cfg['arms']:
            phases = ['mechanism'] if args.stage == 'mechanism' else ['train', 'score']
            for phase in phases: jobs.append(run_job(w, config, runtime, output, record, arm, phase, gpu))
        return jobs
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, rid, dev['gpu']) for rid, dev in zip(records, devices)]
        for future in concurrent.futures.as_completed(futures): completed.extend(future.result())
    if args.stage == 'mechanism':
        dump(output/'MECHANISM_ALL_PASS.json', dict(status='PASS_FIT_ONLY', jobs=completed, GPU_hours=sum(r['GPU_hours'] for r in completed)))
    else:
        banks = [dict(path=str(p), relative=p.relative_to(output).as_posix(), bytes=p.stat().st_size,
                      sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.glob('*/*/score/*.npz'))]
        assert len(completed) == 16 and len(banks) == 16
        dump(output/'RAW_BANK_SEAL.json', dict(status='COMPLETED', jobs=completed, banks=banks,
            GPU_hours=sum(r['GPU_hours'] for r in completed), no_labels_decoded=True, no_threshold_fitting=True,
            formal_train_jobs=8, formal_score_jobs=8, entire_native_frozen=True))
    print(json.dumps(dict(event='STAGE_COMPLETED', stage=args.stage, jobs=len(completed))), flush=True)


if __name__ == '__main__': main()
