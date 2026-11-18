"""Freeze paired plans per seed and run independent native arms on at most four GPUs."""
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
    parser.add_argument("--seed",type=int,default=2025)
    parser.add_argument("--seeds",type=int,nargs="+")
    args=parser.parse_args();root=args.output
    root.mkdir(parents=True,exist_ok=False)
    seeds=args.seeds if args.seeds is not None else [args.seed]
    assert seeds in ([2025],[2026],[2027],[2026,2027])
    legacy_target=args.workspace/"results/left_author_swat_u_transfer_20261118_v1/U/U_PLAN.npz"
    assert sha(legacy_target)=="1d4dbd9cfbeb8dcbcf218f7a3bb46176c290a29406392b5eb8a5835f37afa966"
    with np.load(legacy_target)as z:legacy_generated=make_replay_plan(z["base_starts"],2025)
    legacy_plan=args.workspace/"results/left_author_swat_source_replay_20261118_v1/SAMPLING_PLAN.npz"
    if seeds!=[2025]:
        with np.load(legacy_plan)as z:
            assert set(z.files)==set(legacy_generated)and all(np.array_equal(z[k],legacy_generated[k])for k in z.files)
        dump(root/"LEGACY_2025_PLAN_CHECK.json",dict(status="PASS",legacy_plan_sha256=sha(legacy_plan),
             target_plan_sha256=sha(legacy_target),all_four_arrays_bitexact=True,training_repeated=False))
    seed_roots={seed:(root/f"seed{seed}"if len(seeds)>1 else root)for seed in seeds}
    plan_paths={}
    for seed in seeds:
        seed_root=seed_roots[seed];seed_root.mkdir(parents=True,exist_ok=True)
        if seed==2025:target=legacy_target
        else:
            target=seed_root/"TARGET_PLAN.npz"
            np.savez_compressed(target,base_starts=np.random.default_rng(seed).integers(0,90000-192+1,size=(2000,128)))
        with np.load(target)as z:plan=make_replay_plan(z["base_starts"],seed)
        plan_path=seed_root/"SAMPLING_PLAN.npz";plan_paths[seed]=plan_path
        np.savez_compressed(plan_path,**plan)
        assert all(v.shape==(2000,128)and v.min()>=0 for v in plan.values())
        assert plan["T_starts"].max()<=90000-192 and plan["T_prime_starts"].max()<=90000-192
        assert plan["S_starts"].max()<=396000-192 and plan["S_prime_starts"].max()<=396000-192
        t=np.arange(90000,dtype=np.float32)[:,None];s=(1000000+np.arange(396000,dtype=np.float32))[:,None]
        r=replay_batch(t,s,plan,0,"R");tt=replay_batch(t,s,plan,0,"TT");ss=replay_batch(t,s,plan,0,"SS")
        assert np.array_equal(r[:128],tt[:128])and np.array_equal(r[128:],ss[128:])
        assert not np.array_equal(tt[:128],tt[128:])and not np.array_equal(ss[:128],ss[128:])
        dump(seed_root/"PLAN_FREEZE.json",dict(status="FROZEN_BEFORE_TRAINING_OR_QUERY",
             plan_sha256=sha(plan_path),target_plan_sha256=sha(target),target_plan_file=target.name,seed=seed,
             RNG=f"SeedSequence({seed}).spawn(3)",stream_spawn_keys={"S":[0],"T_prime":[1],"S_prime":[2]},
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
    def job(seed,arm,gpu):
        seed_root=seed_roots[seed];plan_path=plan_paths[seed]
        life=seed_root/"lifecycles"/arm;life.mkdir(parents=True,exist_ok=False)
        began=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();code=None;cleanup=None;error=None
        try:
            dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu],gpu,20))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="GPU-"+devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
            argv=[sys.executable,"-B","-m","hybrid_tsad.left_author_u_transfer","--workspace",str(args.workspace),
                  "--output",str(seed_root/arm),"--replay-arm",arm,"--replay-plan",str(plan_path),"--seed",str(seed)]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned=owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,supervisor_pid=os.getpid(),GPU=gpu,UTC=utc,argv=argv,capGiB=20,timeout_seconds=1800,owned=owned))
                print(json.dumps(dict(event="RUNNING",seed=seed,arm=arm,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=1790)
                finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup["status"]=="NO_LIVE_GROUP_MEMBERS"
        except BaseException as ex:error=repr(ex)
        result=dict(status="COMPLETED"if error is None else"FAILED",seed=seed,arm=arm,GPU=gpu,UTC_start=utc,
                    UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),OS_exit_code=code,error=error,
                    cleanup=cleanup,actual_GPU_seconds=time.monotonic()-began)
        dump(life/"OS_WAIT.json",result)
        if error:raise RuntimeError(result)
        return result
    try:
        tasks=[(seed,arm)for seed in seeds for arm in ["R","TT","SS"]]
        concurrency=min(4,len(tasks))
        def lane(gpu):return[job(seed,arm,gpu)for seed,arm in tasks[gpu::concurrency]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)as pool:
            jobs=[j for lane_jobs in pool.map(lane,range(concurrency))for j in lane_jobs]
        all_banks=[]
        for seed in seeds:
            seed_root=seed_roots[seed];banks=[]
            for arm in ["R","TT","SS"]:
                receipt=json.loads((seed_root/arm/"RECEIPT.json").read_text())
                for bank in receipt["banks"]:
                    assert sha(seed_root/arm/bank["relative"])==bank["sha256"]
                    banks.append(dict(bank,relative=arm+"/"+bank["relative"],arm=arm))
            assert len(banks)==6
            seed_jobs=[j for j in jobs if j["seed"]==seed]
            dump(seed_root/"RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,banks=banks,jobs=seed_jobs,
                 sampling_plan_sha256=sha(plan_paths[seed]),actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in seed_jobs)/3600))
            all_banks.extend(dict(b,relative=str((seed_root/b["relative"]).relative_to(root)),seed=seed)for b in banks)
        if len(seeds)>1:
            dump(root/"GLOBAL_RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,banks=all_banks,jobs=jobs,
                 seeds=seeds,actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs)/3600))
    except BaseException as ex:
        dump(root/"FAILED.json",dict(error=repr(ex)));raise


if __name__=="__main__":
    main()
