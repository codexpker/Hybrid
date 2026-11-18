"""Fixed current-U donor preparation and native TT/R B/C job orchestration."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from hybrid_tsad.left_exposure_bundle import construct
from hybrid_tsad.left_swat_author import read,sha,dump


def prepare(args):
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    for domain in ["SWAT","PSM"]:
        d=root/domain;d.mkdir()
        task=read(args.configs/f"left-preservation-{domain.lower()}-20261118.json")
        task.pop("preservation",None);task["experiment"]="left_current_u_exposure_20261118_v1"
        task["exposure"]={"seed":2025,"conditions":["B","C"],"arms":["TT","R"],
            "T_replacement_slots":25600,"T_slots":256000,"all_slots":512000,
            "RNG_entropy":[2025,20261118,10],"preservation_loss":False}
        dump(d/"TASK.json",task)
        plan=(args.workspace/"results/left_author_swat_source_replay_20261118_v1/SAMPLING_PLAN.npz"if domain=="SWAT"else
              args.workspace/"results/left_author_psm_source_replay_20261118_v1/seed2025/SAMPLING_PLAN.npz")
        shutil.copyfile(plan,d/"SAMPLING_PLAN.npz")
        construct(args.workspace,d/"TASK.json",d/"SAMPLING_PLAN.npz",d/"BUNDLE")
        dump(d/"FROZEN.json",dict(task_sha256=sha(d/"TASK.json"),plan_sha256=sha(d/"SAMPLING_PLAN.npz"),
            worker_sha256=sha(Path(__file__).with_name("left_domain_replay.py")),
            constructor_sha256=sha(Path(__file__).with_name("left_exposure_bundle.py")),
            labels_only_constructor=True,seed=2025,updates=2000,batch=256))


def queue(args):
    root=args.output;workspace=args.workspace
    sys.path.insert(0,str(workspace/"preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    devices={int(row.split(",")[0]):row.split(",")[1].strip().removeprefix("GPU-")for row in rows.splitlines()if row.strip()}
    eligible=[d for d in ["SWAT","PSM"]if read(root/d/"BUNDLE/STATUS.json")["status"]=="QUALIFIED"]
    tasks=[(d,arm,condition)for d in eligible for arm in ["TT","R"]for condition in ["B","C"]]
    for domain in eligible:
        f=read(root/domain/"FROZEN.json")
        assert sha(root/domain/"TASK.json")==f["task_sha256"]and sha(root/domain/"SAMPLING_PLAN.npz")==f["plan_sha256"]
        assert sha(Path(__file__).with_name("left_domain_replay.py"))==f["worker_sha256"]
    def job(domain,arm,condition,gpu):
        name=arm+"_"+condition;life=root/"lifecycles"/(domain+"_"+name);life.mkdir(parents=True,exist_ok=False)
        began=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();code=None;cleanup=None;error=None
        try:
            dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu],gpu,20))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="GPU-"+devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
            argv=[sys.executable,"-B","-m","hybrid_tsad.left_author_u_transfer","--workspace",str(workspace),
                "--output",str(root/domain/name),"--task",str(root/domain/"TASK.json"),"--seed","2025",
                "--replay-arm",arm,"--replay-plan",str(root/domain/"SAMPLING_PLAN.npz"),
                "--exposure-bundle",str(root/domain/"BUNDLE/learner"/(condition+"_MANIFEST.json"))]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned=owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,supervisor_pid=os.getpid(),GPU=gpu,UTC=utc,argv=argv,capGiB=20,timeout_seconds=1800,owned=owned))
                print(json.dumps(dict(event="RUNNING",domain=domain,arm=arm,condition=condition,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=1790)
                finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup["status"]=="NO_LIVE_GROUP_MEMBERS"
        except BaseException as ex:error=repr(ex)
        result=dict(status="COMPLETED"if error is None else"FAILED",domain=domain,arm=arm,condition=condition,GPU=gpu,
            UTC_start=utc,UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),OS_exit_code=code,error=error,
            cleanup=cleanup,actual_GPU_seconds=time.monotonic()-began)
        dump(life/"OS_WAIT.json",result)
        if error:raise RuntimeError(result)
        return result
    try:
        n=min(len(args.gpus),len(tasks))
        def lane(i):return[job(d,a,c,args.gpus[i])for d,a,c in tasks[i::n]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n)as pool:
            jobs=[j for group in pool.map(lane,range(n))for j in group]
        allbanks=[]
        for domain in eligible:
            banks=[]
            for arm in ["TT","R"]:
                for condition in ["B","C"]:
                    name=arm+"_"+condition
                    for bank in read(root/domain/name/"RECEIPT.json")["banks"]:
                        assert sha(root/domain/name/bank["relative"])==bank["sha256"]
                        banks.append(dict(bank,relative=name+"/"+bank["relative"],arm=arm,condition=condition))
            dj=[j for j in jobs if j["domain"]==domain]
            dump(root/domain/"RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded_by_learner=True,
                banks=banks,jobs=dj,actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in dj)/3600))
            allbanks.extend(dict(b,relative=domain+"/"+b["relative"],domain=domain)for b in banks)
        dump(root/"GLOBAL_RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded_by_learner=True,
            banks=allbanks,jobs=jobs,eligible_domains=eligible,
            actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs)/3600))
    except BaseException as ex:
        dump(root/"FAILED.json",dict(error=repr(ex)));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["prepare","queue"])
    p.add_argument("--workspace",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--configs",type=Path);p.add_argument("--gpus",type=int,nargs="+",default=[0,1,2,3]);a=p.parse_args()
    prepare(a)if a.phase=="prepare"else queue(a)
