"""Fixed four-model, score-only LEFT grid queue, with owned-process cleanup."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hybrid_tsad.left_frozen_native_grid import dump, sha

ARMS = ["noU", "U_A", "U_B", "U_C"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0,str(args.workspace / "preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    devices = {int(row.split(",")[0]): row.split(",")[1].strip().removeprefix("GPU-")
               for row in rows.splitlines() if row.strip()}
    dump(root/"RUNTIME_FREEZE.json", dict(worker_sha256=sha(Path(__file__).with_name("left_frozen_native_grid.py")),
         queue_sha256=sha(__file__), arms=ARMS, training_updates=0, window=192, stride=192,
         anchor=495000, original_CAL_points=512, labels_read=False))
    def job(item):
        arm, gpu = item
        life = root / "lifecycles" / arm
        life.mkdir(parents=True,exist_ok=False)
        began = time.monotonic()
        utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        code, cleanup, error = None, None, None
        try:
            dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu],gpu,30))
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="GPU-"+devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
            argv = [sys.executable,"-B","-m","hybrid_tsad.left_frozen_native_grid",
                    "--workspace",str(args.workspace),"--output",str(root),"--arm",arm]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True)
                owned=owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,supervisor_pid=os.getpid(),GPU=gpu,
                     UTC=utc,argv=argv,timeout_seconds=900,capGiB=30,owned=owned))
                print(json.dumps(dict(event="RUNNING",arm=arm,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=890)
                finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup["status"]=="NO_LIVE_GROUP_MEMBERS"
        except BaseException as exc:error=repr(exc)
        result=dict(status="COMPLETED"if error is None else"FAILED",arm=arm,GPU=gpu,
                    UTC_start=utc,UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    OS_exit_code=code,error=error,cleanup=cleanup,actual_GPU_seconds=time.monotonic()-began)
        dump(life/"OS_WAIT.json",result)
        if error:raise RuntimeError(result)
        return result
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
            jobs=list(pool.map(job,zip(ARMS,range(4))))
        banks=[]
        for arm in ARMS:
            receipt=json.loads((root/arm/"RECEIPT.json").read_text())
            for bank in receipt["banks"]:
                assert sha(root/bank["relative"])==bank["sha256"]
                banks.append(dict(arm=arm,**bank))
        assert len(banks)==8
        dump(root/"RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,
             banks=banks,jobs=jobs,actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs)/3600))
    except BaseException as exc:
        dump(root/"FAILED.json",dict(error=repr(exc)))
        raise


if __name__=="__main__":
    main()
