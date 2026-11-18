"""Resume only three never-started PSM jobs after GPU0's idle guard refused.

Existing scientific outputs and the original failed controller are immutable.
Wait for this task's GPU2/3 jobs to release their cards; never use GPU0.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hybrid_tsad.left_swat_author import sha,read,dump


def main():
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args();root=args.output
    task=read(root/"TASK.json")
    sys.path.insert(0,str(args.workspace/"preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    devices={int(row.split(",")[0]):row.split(",")[1].strip().removeprefix("GPU-")for row in rows.splitlines()if row.strip()}
    frozen=read(root/"seed2025/PLAN_FREEZE.json")["worker_sha256"]
    assert sha(Path(__file__).with_name("left_domain_replay.py"))==frozen
    for seed,arm in [(2025,"noU"),(2026,"R"),(2027,"TT")]:
        out=root/"noU"if arm=="noU"else root/f"seed{seed}"/arm
        assert not out.exists(),"Refuse to overwrite any existing output"
    def wait_own(seed,arm):
        path=root/f"seed{seed}/lifecycles"/arm/"OS_WAIT.json";start=time.monotonic()
        while not path.exists():
            if time.monotonic()-start>1800:raise TimeoutError("Own dependency has not completed")
            time.sleep(10)
        assert read(path)["status"]=="COMPLETED"
    def job(seed,arm,gpu):
        assert gpu in [2,3]
        sr=root if arm=="noU"else root/f"seed{seed}"
        life=root/"RECOVERY_LIFECYCLES"/f"{seed}_{arm}";life.mkdir(parents=True,exist_ok=False)
        began=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();code=None;cleanup=None;error=None
        try:
            cap=30 if arm=="noU"else 20;dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu],gpu,cap))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="GPU-"+devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
            argv=[sys.executable,"-B","-m","hybrid_tsad.left_author_u_transfer","--workspace",str(args.workspace),
                "--output",str(sr/arm),"--task",str(root/"TASK.json"),"--seed",str(seed)]
            if arm!="noU":argv += ["--replay-arm",arm,"--replay-plan",str(sr/"SAMPLING_PLAN.npz")]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned=owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,supervisor_pid=os.getpid(),GPU=gpu,UTC=utc,argv=argv,capGiB=cap,timeout_seconds=1800,owned=owned))
                print(json.dumps(dict(event="RECOVERY_RUNNING",seed=seed,arm=arm,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=1790)
                finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup["status"]=="NO_LIVE_GROUP_MEMBERS"
        except BaseException as ex:error=repr(ex)
        result=dict(status="COMPLETED"if error is None else"FAILED",seed=seed,arm=arm,GPU=gpu,UTC_start=utc,
            UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),OS_exit_code=code,error=error,cleanup=cleanup,
            actual_GPU_seconds=time.monotonic()-began,recovered_never_started=True)
        dump(life/"OS_WAIT.json",result)
        if error:raise RuntimeError(result)
        return result
    def lane2():
        wait_own(2026,"SS")
        return[job(2025,"noU",2),job(2026,"R",2)]
    def lane3():
        wait_own(2027,"R")
        return[job(2027,"TT",3)]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
            a=pool.submit(lane2);b=pool.submit(lane3);recovered=a.result()+b.result()
        # Finish only after the original non-conflicting lanes have terminated.
        limit=time.monotonic()+1800
        while not(root/"FAILED.json").exists():
            if time.monotonic()>limit:raise TimeoutError("Original controller terminal record missing")
            time.sleep(5)
        jobs=[];banks_all=[]
        for seed in task["seeds"]:
            sr=root/f"seed{seed}";banks=[];seedjobs=[]
            for arm in task["arms"]:
                recovery=root/"RECOVERY_LIFECYCLES"/f"{seed}_{arm}"/"OS_WAIT.json"
                jp=recovery if recovery.exists()else sr/"lifecycles"/arm/"OS_WAIT.json"
                record=read(jp);assert record["status"]=="COMPLETED";seedjobs.append(record)
                for bank in read(sr/arm/"RECEIPT.json")["banks"]:
                    assert sha(sr/arm/bank["relative"])==bank["sha256"]
                    banks.append(dict(bank,relative=arm+"/"+bank["relative"],arm=arm))
            dump(sr/"RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,banks=banks,jobs=seedjobs,
                sampling_plan_sha256=sha(sr/"SAMPLING_PLAN.npz"),actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in seedjobs)/3600))
            jobs.extend(seedjobs);banks_all.extend(dict(bank,relative=f"seed{seed}/"+bank["relative"],seed=seed)for bank in banks)
        nojob=read(root/"RECOVERY_LIFECYCLES/2025_noU/OS_WAIT.json");jobs.append(nojob)
        for bank in read(root/"noU/RECEIPT.json")["banks"]:
            assert sha(root/"noU"/bank["relative"])==bank["sha256"]
            banks_all.append(dict(bank,relative="noU/"+bank["relative"],arm="noU",seed=None))
        failed=read(root/"lifecycles/noU/OS_WAIT.json")
        assert failed["status"]=="FAILED"and failed["OS_exit_code"]is None
        assert len(jobs)==10 and len(banks_all)==20
        dump(root/"GLOBAL_RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,banks=banks_all,jobs=jobs,
             failed_attempts=[dict(failed,model_started=False,actual_owned_GPU_seconds=0)],
             recovery="Only never-started noU/2026R/2027TT; original controller failure preserved; GPU0 excluded",
             task_sha256=sha(root/"TASK.json"),actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs)/3600))
    except BaseException as ex:
        dump(root/"RECOVERY_FAILED.json",dict(error=repr(ex)));raise


if __name__=="__main__":main()
