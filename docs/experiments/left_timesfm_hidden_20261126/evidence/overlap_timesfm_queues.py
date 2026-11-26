"""Change waiting hardware gates only, after the PSM updates have sealed."""
import ast
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

w = Path('/home/a312/tsad_research')
r = w / 'results/left_timesfm_hidden_20261126_v1'
before = r / 'ADAPTATION_before_gate_overlap.py'
after = w / 'tmp/trajectory_src/src/hybrid_tsad/left_timesfm_adaptation.py'

def numerical_ast(path):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name != 'queue']
    return ast.dump(tree)

assert numerical_ast(before) == numerical_ast(after)
while not all((r / 'PSM/U/PSM' / a / 'receipt.json').exists() for a in ['A', 'C']):
    time.sleep(10)
while json.loads((r / 'U_QUEUE_PSM.json').read_text())['phase'] == 'training':
    time.sleep(2)
record = dict(reason='Overlap two SMD lanes with streaming SWaT after PSM releases its large cache',
              training_and_scoring_ast_unchanged=True,
              old_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),
              new_sha256=hashlib.sha256(after.read_bytes()).hexdigest(),
              SWAT_active_process_untouched=3452092, lanes=[])
for d, gpu, pid in [('PSM', 1, 3452091), ('SMD', 3, 3452093)]:
    p = Path(f'/proc/{pid}/cmdline')
    assert p.exists()
    command = p.read_bytes().split(b'\0')
    assert b'hybrid_tsad.left_timesfm_adaptation' in command and d.encode() in command
    state = json.loads((r / f'U_QUEUE_{d}.json').read_text())
    assert state['phase'].startswith('waiting'), state
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while Path(f'/proc/{pid}').exists() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not Path(f'/proc/{pid}').exists()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(w / 'tmp/trajectory_src/src'))
    log = r / 'logs' / f'u-{d}-lane{gpu}-overlap.log'
    with log.open('x') as stream:
        child = subprocess.Popen([str(w / 'envs/timeradar/bin/python'), '-u', '-m',
                                  'hybrid_tsad.left_timesfm_adaptation', str(w), 'queue', d],
                                 cwd=w, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record['lanes'].append(dict(domain=d, previous_pid=pid, previous_state=state,
                                new_pid=child.pid, log=str(log)))
(r / 'U_QUEUE_OVERLAP.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
