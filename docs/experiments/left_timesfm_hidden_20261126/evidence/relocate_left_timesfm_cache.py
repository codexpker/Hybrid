"""Continue pending U jobs with byte-identical GPU caches at sealed boundaries."""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

w = Path('/home/a312/tsad_research')
r = w / 'results/left_timesfm_hidden_20261126_v1'
assert json.loads((r / 'GPU_CACHE_EQUALITY.json').read_text())['identical_gather']
before = r / 'ADAPTATION_before_gpu_cache.py'
after = w / 'tmp/trajectory_src/src/hybrid_tsad/left_timesfm_adaptation.py'
record = dict(
    reason='Replace host target-feature storage with identical FP32 CUDA storage; no model or update formula change',
    old_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
    new_sha256=hashlib.sha256(after.read_bytes()).hexdigest(),
    lanes=[],
)
for lane, gpu, pid, finished in [
    ('PSM', 1, 3509414, 'machine-1-1'),
    ('SMD', 3, 3509418, 'machine-1-3'),
    ('SWAT', 2, 3452092, 'machine-1-2'),
]:
    while not all((r / 'SMD/U' / finished / a / 'receipt.json').exists() for a in ('A', 'C')):
        time.sleep(1)
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    assert b'hybrid_tsad.left_timesfm_adaptation' in command and lane.encode() in command
    os.kill(pid, signal.SIGSTOP)
    state = json.loads((r / f'U_QUEUE_{lane}.json').read_text())
    if state.get('entity') != finished:
        assert not (r / 'SMD/U' / state['entity'] / 'progress.pt').exists(), state
    stopped = time.time()
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
    for _ in range(150):
        if not Path(f'/proc/{pid}').exists():
            break
        time.sleep(.2)
    assert not Path(f'/proc/{pid}').exists()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(w / 'tmp/trajectory_src/src'))
    log = r / 'logs' / f'u-{lane}-gpu-cache.log'
    with log.open('x') as stream:
        child = subprocess.Popen(
            [str(w / 'envs/timeradar/bin/python'), '-u', '-m',
             'hybrid_tsad.left_timesfm_adaptation', str(w), 'queue', lane],
            cwd=w, env=env, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    record['lanes'].append(dict(lane=lane, gpu=gpu, previous_pid=pid,
        new_pid=child.pid, completed_entity=finished, previous_state=state,
        no_unsealed_optimizer_updates_interrupted=True, stopped_time=stopped,
        log=str(log)))
    (r / 'GPU_CACHE_RELOCATION.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record['lanes'][-1]), flush=True)
