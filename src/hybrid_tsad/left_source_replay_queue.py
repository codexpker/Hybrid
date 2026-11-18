"""Freeze one shared replay plan, then execute three complete native arms."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from hybrid_tsad.left_author_u_transfer import make_replay_plan, replay_batch
from hybrid_tsad.left_swat_author import sha, dump


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=args.output
    root.mkdir(parents=True,exist_ok=False)
    target=args.workspace/"results/left_author_swat_u_transfer_20261118_v1/U/U_PLAN.npz"
    assert sha(target)=="1d4dbd9cfbeb8dcbcf218f7a3bb46176c290a29406392b5eb8a5835f37afa966"
    with np.load(target)as z:plan=make_replay_plan(z["base_starts"])
    plan_path=root/"SAMPLING_PLAN.npz"
    np.savez_compressed(plan_path,**plan)
    assert all(v.shape==(2000,128)and v.min()>=0 for v in plan.values())
    assert plan["T_starts"].max()<=90000-192 and plan["T_prime_starts"].max()<=90000-192
    assert plan["S_starts"].max()<=396000-192 and plan["S_prime_starts"].max()<=396000-192
    # Synthetic numerical check of shared half identity; no scientific data opened.
    t=np.arange(90000,dtype=np.float32)[:,None]
    s=(1000000+np.arange(396000,dtype=np.float32))[:,None]
    r=replay_batch(t,s,plan,0,"R");tt=replay_batch(t,s,plan,0,"TT");ss=replay_batch(t,s,plan,0,"SS")
    assert np.array_equal(r[:128],tt[:128])and np.array_equal(r[128:],ss[128:])
    assert not np.array_equal(tt[:128],tt[128:])and not np.array_equal(ss[:128],ss[128:])
    dump(root/"PLAN_FREEZE.json",dict(status="FROZEN_BEFORE_TRAINING_OR_QUERY",
         plan_sha256=sha(plan_path),target_plan_sha256=sha(target),seed=2025,
         RNG="SeedSequence(2025).spawn(3)",stream_spawn_keys={"S":[0],"T_prime":[1],"S_prime":[2]},
         keys_shapes={k:list(v.shape)for k,v in plan.items()},source_role=[0,396000],target_role=[495000,585000],
         source_starts_inclusive=[0,396000-192],target_starts_inclusive=[0,90000-192],
         arms={"R":["T","S"],"TT":["T","T_prime"],"SS":["S_prime","S"]},
         updates=2000,batch=256,optimizer="Adam lr1e-4",half_identity_CPU_check=True,labels_read=False,
         worker_sha256=sha(Path(__file__).with_name("left_author_u_transfer.py")),queue_sha256=sha(__file__)))
    sys.path.insert(0,str(args.workspace/"preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    devices={int(row.split(",")[0]):row.split(",")[1].strip().removeprefix("GPU-")for row in rows.splitlines()if row.strip()}
    def job(item):
        arm,gpu=item;life=root/"lifecycles"/arm;life.mkdir(parents=True,exist_ok=False)
        began=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();code=None;cleanup=None;error=None
        try:
            dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu],gpu,20))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="GPU-"+devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
            argv=[sys.executable,"-B","-m","hybrid_tsad.left_author_u_transfer","--workspace",str(args.workspace),
                  "--output",str(root/arm),"--replay-arm",arm,"--replay-plan",str(plan_path)]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned=owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,supervisor_pid=os.getpid(),GPU=gpu,UTC=utc,argv=argv,capGiB=20,timeout_seconds=1800,owned=owned))
                print(json.dumps(dict(event="RUNNING",arm=arm,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=1790)
                finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup["status"]=="NO_LIVE_GROUP_MEMBERS"
        except BaseException as ex:error=repr(ex)
        result=dict(status="COMPLETED"if error is None else"FAILED",arm=arm,GPU=gpu,UTC_start=utc,
                    UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),OS_exit_code=code,error=error,
                    cleanup=cleanup,actual_GPU_seconds=time.monotonic()-began)
        dump(life/"OS_WAIT.json",result)
        if error:raise RuntimeError(result)
        return result
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3)as pool:jobs=list(pool.map(job,zip(["R","TT","SS"],range(3))))
        banks=[]
        for arm in ["R","TT","SS"]:
            receipt=json.loads((root/arm/"RECEIPT.json").read_text())
            for bank in receipt["banks"]:
                assert sha(root/arm/bank["relative"])==bank["sha256"]
                banks.append(dict(bank,relative=arm+"/"+bank["relative"],arm=arm))
        assert len(banks)==6
        dump(root/"RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,banks=banks,jobs=jobs,
             sampling_plan_sha256=sha(plan_path),actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs)/3600))
    except BaseException as ex:
        dump(root/"FAILED.json",dict(error=repr(ex)));raise


if __name__=="__main__":
    main()
