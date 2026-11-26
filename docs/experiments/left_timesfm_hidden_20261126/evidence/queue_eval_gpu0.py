import os, subprocess, time
from pathlib import Path
root=Path('/home/a312/tsad_research')
os.chdir(root)
os.environ.update(PYTHONPATH=str(root/'tmp/trajectory_src/src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
out=root/'results/left_timesfm_hidden_20261126_v1'
while Path('/proc/3389361').exists():
    time.sleep(15)
assert (out/'SMD/U_SOURCE_FEATURES/receipt.json').exists(), 'shared U source cache failed'
exe=str(root/'envs/timeradar/bin/python')
def run(*args):
    print('RUN',args,flush=True)
    subprocess.run([exe,'-u','-m','hybrid_tsad.left_timesfm_evaluation',str(root),*args],check=True)
run('features','PSM','SWAT','SMD')
pending=['PSM','SWAT','SMD']
while pending:
    ready=[d for d in pending if all((out/d/a/'receipt.json').exists() for a in 'ABCDE')]
    if not ready:
        time.sleep(20)
        continue
    domain=ready[0]
    run('score',domain)
    for role in ['QUERY','CAL']:
        run('score',domain,'--arms','A','C','--role',role)
    if domain!='SMD':
        run('controlled',domain)
    pending.remove(domain)
print('Z_SOURCE_READOUTS_COMPLETED',flush=True)
