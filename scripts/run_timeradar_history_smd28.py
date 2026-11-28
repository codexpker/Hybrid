"""Durable four-GPU queue for the sealed, full SMD28 history experiment."""
import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import subprocess
import sys
import time


MODULE = "hybrid_tsad.timeradar_history_smd"
U_ARMS = ("history", "frozen_history", "native")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stage(workspace, out, gpu, action, entity=None, arm=None, scoring_stage=None):
    command = [sys.executable, "-u", "-m", MODULE, action, str(workspace), str(out)]
    if entity is not None:
        command += ["--entity", entity]
    if arm is not None:
        command += ["--arm", arm]
    if scoring_stage is not None:
        command += ["--stage", scoring_stage]
    filename = "_".join(str(x) for x in (action, scoring_stage, entity, arm) if x is not None)+".log"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (out/"logs"/filename).open("ab") as log:
        subprocess.run(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def queue(tasks, gpus, run, ready):
    """Use independent GPU lanes. Do not start B until its matched A is sealed."""
    devices = list(gpus)
    tasks = list(tasks)
    pending = {}
    failures = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        while tasks or pending:
            while devices:
                pos = next((i for i,t in enumerate(tasks) if ready(*t)), None)
                if pos is None:
                    break
                task = tasks.pop(pos)
                gpu = devices.pop(0)
                pending[pool.submit(run,gpu,*task)] = (gpu,task)
            if not pending:
                if tasks or failures:
                    raise RuntimeError(f"Unfinished queue tasks: {tasks}; failed: {failures}")
                break
            done,_ = wait(pending, return_when=FIRST_COMPLETED)
            for job in done:
                gpu, task = pending.pop(job)
                devices.append(gpu)
                try:
                    job.result()
                except Exception as error:
                    failures.append((task,str(error)))
                    print("FAILED",task,error,flush=True)
                else:
                    print("DONE",task,flush=True)
    if failures:
        raise RuntimeError(f"Failed queue tasks: {failures}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    args=parser.parse_args()
    w,out=args.workspace.resolve(),args.output.resolve()
    gpus=args.gpus.split(",")
    assert len(gpus)==len(set(gpus))==4
    (out/"logs").mkdir(exist_ok=True)
    ids=sorted(read(out/"contract.json")["entities"])
    assert len(ids)==28
    start=time.time()
    for arm in ("native","history"):
        ck=read(out/"source"/arm/"receipt.json")
        assert ck["steps"]==240
    if not (out/"Z_metrics.csv").is_file():
        tasks=[(d,a) for d in ids for a in ("native","history")]
        queue(tasks,gpus,lambda g,d,a:stage(w,out,g,"score",d,a,"Z"),lambda *_:True)
        stage(w,out,"","evaluate-Z")
    print("Z_COMPLETE",time.time()-start,flush=True)

    # Fixed first entity and joint-update arm. This is one actual endpoint of
    # the 84 and remains in the final comparison. No extra pilot is trained.
    pilot=ids[0]
    if not (out/pilot/"history"/"receipt.json").is_file():
        stage(w,out,gpus[0],"U",pilot,"history")
    pilot_receipt=read(out/pilot/"history"/"receipt.json")
    pilot_info=dict(entity=pilot,arm="history",steps=2000,
                    seconds=pilot_receipt["seconds"],seconds_per_step=pilot_receipt["seconds"]/2000,
                    peak_bytes=pilot_receipt["peak_bytes"],gpu_concurrency=4,
                    equal_joint_cost_for_all_arms_envelope_worker_hours=84*pilot_receipt["seconds"]/3600,
                    pilot_included_in_final_results=True)
    (out/"pilot.json").write_text(json.dumps(pilot_info,indent=2)+"\n",encoding="utf-8")
    print("PILOT_COMPLETE",pilot_info,flush=True)

    def run(gpu,entity,arm):
        if not (out/entity/arm/"receipt.json").is_file():
            stage(w,out,gpu,"U",entity,arm)
        stage(w,out,gpu,"score",entity,arm,"U")

    tasks=[(d,a) for d in ids for a in U_ARMS]
    queue(tasks,gpus,run,lambda d,a:a!="frozen_history" or (out/d/"history"/"receipt.json").is_file())
    stage(w,out,"","evaluate")
    (out/"queue_complete.json").write_text(json.dumps(dict(planned_U=84, completed_U=84,
        devices=gpus, queue_wall_seconds=time.time()-start),indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
