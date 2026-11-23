"""Follow the seven fixed U jobs with label-free scoring and one sealed evaluation."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .native_v_conditioning_runner import dump

ROOT = Path("results/native_event_stage2_u_20261123_v1")
MODULE = "hybrid_tsad.native_event_stage2_readout"


def idle_devices():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                                      "--format=csv,noheader,nounits"], text=True)
    rows = [[int(v) for v in line.split(",")] for line in output.splitlines()]
    return [gpu for gpu,used,free,util in rows if used < 8 and free > 8000 and util < 10]


def execute(gpu, lane, n):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    tic = time.monotonic()
    with (ROOT / f"score_gpu{gpu}.log").open("a") as log:
        proc = subprocess.run([sys.executable, "-u", "-m", MODULE, "score", "--lane", str(lane), "--lanes", str(n)],
                              env=env,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
    return dict(gpu=gpu,lane=lane,exit_code=proc.returncode,seconds=time.monotonic()-tic)


def main():
    while not (ROOT/"ALL_MODELS_FROZEN.json").is_file():
        time.sleep(20)
    gpus = idle_devices()
    while not gpus:
        time.sleep(20)
        gpus = idle_devices()
    with ThreadPoolExecutor(len(gpus)) as pool:
        workers = [pool.submit(execute,gpu,lane,len(gpus)) for lane,gpu in enumerate(gpus)]
        status = [j.result() for j in workers]
    dump(ROOT/"READOUT_QUEUE.json",dict(jobs=status))
    if any(s["exit_code"] != 0 for s in status):
        raise RuntimeError("Inspect the failed score lane log before resuming")
    for action in ("merge","evaluate"):
        subprocess.run([sys.executable,"-m",MODULE,action],check=True,timeout=7200)


if __name__ == "__main__":
    main()
