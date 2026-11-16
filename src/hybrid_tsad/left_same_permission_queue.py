"""One fixed LEFT GPU3 sequence; no retries, tuning, labels or new tasks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .left_same_permission import append, dump
from .timeradar_cross_variable_runner import read, resolve, sha


def now():
    return datetime.now(timezone.utc).isoformat()


def gpu_status(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=uuid,memory.used,memory.total', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, check=True, timeout=20)
    uuid, used, total = [item.strip() for item in result.stdout.strip().split(',')]
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True, timeout=20)
    processes = [line for line in apps.stdout.splitlines() if line.startswith(uuid)]
    return dict(uuid=uuid, used_mib=int(used), total_mib=int(total), processes=processes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--gpu', type=int, default=3)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    config, output = resolve(workspace, args.config), resolve(workspace, args.out)
    cfg = read(config)
    if args.gpu != 3:
        raise ValueError('This fixed queue owns only GPU3')
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'QUEUE_START.json', dict(status='RUNNING', pid=os.getpid(),
        started_utc=now(), gpu=args.gpu, config_sha256=sha(config),
        source_sha256=sha(Path(__file__).with_name('left_same_permission.py')),
        records=list(cfg['records']), scientific_changes='none: native LEFT baseline'))
    jobs, start = [], time.monotonic()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['PYTHONUNBUFFERED'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    try:
        for record in cfg['records']:
            for phase in ['train', 'score']:
                state = gpu_status(args.gpu)
                if state['processes'] or state['used_mib'] > 512:
                    raise RuntimeError('GPU3 is not available: ' + json.dumps(state))
                target = output / record / phase
                target.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-u', '-m', 'hybrid_tsad.left_same_permission', phase,
                    '--workspace', str(workspace), '--config', str(config), '--out', str(target),
                    '--record', record, '--device', 'cuda:0']
                if phase == 'score':
                    command.extend(['--train-out', str(output / record / 'train')])
                began, clock = now(), time.monotonic()
                with (target.parent / (phase + '.log')).open('x') as log:
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    item = dict(record=record, phase=phase, pid=process.pid, gpu=args.gpu,
                        command=command, started_utc=began, status='RUNNING', gpu_preflight=state)
                    append(output / 'JOBS.jsonl', item)
                    timeout = cfg['train_timeout_seconds'] if phase=='train' else cfg['score_timeout_seconds']
                    timed_out = False
                    try:
                        code = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        process.terminate()
                        try:
                            code = process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            code = process.wait(timeout=20)
                seconds = time.monotonic() - clock
                item.update(status='COMPLETED' if code==0 else 'FAILED', exit_code=code,
                    timed_out=timed_out, ended_utc=now(), gpu_seconds=seconds, gpu_hours=seconds/3600)
                jobs.append(item)
                append(output / 'JOBS.jsonl', item)
                if code != 0:
                    raise RuntimeError('No retry: failed native LEFT ' + record + '/' + phase)
        dump(output / 'QUEUE_RECEIPT.json', dict(status='COMPLETED', jobs=jobs,
            gpu_hours=sum(item['gpu_hours'] for item in jobs), queue_seconds=time.monotonic()-start,
            ended_utc=now(), labels_read=False))
    except Exception as error:
        dump(output / 'QUEUE_FAILURE.json', dict(status='FAILED', reason=str(error), jobs=jobs,
             ended_utc=now(), gpu_hours=sum(item['gpu_hours'] for item in jobs)))
        raise
    finally:
        dump(output / 'GPU3_RELEASE.json', dict(status='LEFT_RELEASED_GPU3', gpu=3,
            utc=now(), queue_pid=os.getpid(), final_gpu_status=gpu_status(3),
            instruction='TimeRadar queue must recheck actual availability before using GPU3'))


if __name__ == '__main__':
    main()
