"""Role-bound native LEFT replay driver for a frozen author-domain task.

The original SWaT CLI is unchanged. This implementation is selected only by
its explicit --task argument; labels are reserved entirely for evaluation.
"""
import argparse
import copy
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from hybrid_tsad.left_swat_author import sha,read,dump


def plan_arrays(task,seed):
    w=task["scoring"]["window"];steps=task["training"]["updates"];half=task["sampler"]["half_batch"]
    nt=np.diff(task["roles"]["U_FIT"])[0];ns=np.diff(task["roles"]["SOURCE_FIT"])[0]
    streams=np.random.SeedSequence(seed).spawn(3)
    return dict(T_starts=np.random.default_rng(seed).integers(0,nt-w+1,size=(steps,half)),
        S_starts=np.random.default_rng(streams[0]).integers(0,ns-w+1,size=(steps,half)),
        T_prime_starts=np.random.default_rng(streams[1]).integers(0,nt-w+1,size=(steps,half)),
        S_prime_starts=np.random.default_rng(streams[2]).integers(0,ns-w+1,size=(steps,half)))


def tensor_hashes(model):
    import hashlib
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k,v in model.state_dict().items()}


def rng_hashes(torch):
    import hashlib
    return dict(CPU=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        CUDA=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()for v in torch.cuda.get_rng_state_all()])


def run_task(args):
    task=read(args.task);root=args.output;workspace=args.workspace
    root.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    arm=args.replay_arm or "noU";source=task["source"];w=task["scoring"]["window"]
    keep_mode=getattr(args,"preservation_mode",None)
    assert task["domain"]in {"PSM","SWAT"}and w==192 and task["training"]["batch"]==256
    if keep_mode:assert arm=="R"and args.seed==2025
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    cap=30 if arm=="noU"else 20
    torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    checkpoint=workspace/source["checkpoint"]["relative"];scaler_file=workspace/source["scaler"]["relative"]
    config_file=workspace/source["model_config"]["relative"]
    for key,path in [("checkpoint",checkpoint),("scaler",scaler_file),("model_config",config_file)]:
        assert sha(path)==source[key]["sha256"]
    upstream=workspace/source["upstream"]
    assert sha(upstream/"models/Left/LEFT.py")==source["model_sha256"]
    sys.path.insert(0,str(upstream));from models.Left.LEFT import Left
    cfg=read(config_file);c=cfg["enc_in"];assert cfg["seq_len"]==w and c==task["data"]["target_test"]["shape"][1]
    model=Left(SimpleNamespace(**cfg)).cuda()
    loaded=model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=False)
    assert not loaded.missing_keys and all(k.endswith(("total_ops","total_params"))for k in loaded.unexpected_keys)
    model.step=source["python_step"]
    if source["scaler"]["format"]=="json_statistics":
        statistics=read(scaler_file)["statistics"];mean=np.asarray(statistics["mean"]);scale=np.asarray(statistics["scale"])
    else:
        with np.load(scaler_file)as z:mean=z["mean"].copy();scale=z["scale"].copy()
    assert mean.shape==scale.shape==(c,)and np.all(scale>0)
    records={}
    for key,item in task["data"].items():
        path=workspace/item["relative"];assert sha(path)==item["sha256"]
        records[key]=np.load(path,mmap_mode="r",allow_pickle=False)
        assert list(records[key].shape)==item["shape"]
    accesses=[]
    allowed=set()if arm=="noU"else ({"S","T"}if arm=="R"else {"T"}if arm=="TT"else {"S"})
    def load_role(name):
        if name not in allowed:raise PermissionError("Role not allowed in this stage: "+name)
        source_role=name=="S";key="source_train"if source_role else"target_test"
        role={"S":"SOURCE_FIT","T":"U_FIT"}.get(name,name);lo,hi=task["roles"][role]
        values=(np.array(records[key][lo:hi],dtype=np.float64,order="F",copy=True)-mean)/scale
        assert values.shape==(hi-lo,c)and np.isfinite(values).all()
        accesses.append(dict(role=role,record_id=task["data"][key]["record_id"],input=[lo,hi],
                             raw_record_sha256=task["data"][key]["sha256"]))
        return values
    dump(root/"INITIAL_IDENTITY.json",dict(source_checkpoint_sha256=sha(checkpoint),
        scaler_sha256=sha(scaler_file),initial_python_step=model.step,training_seed=args.seed,
        state_sha256=tensor_hashes(model),rng=rng_hashes(torch),task_sha256=sha(args.task),
        actual_training_roles=sorted(allowed),labels_read=False,worker_sha256=sha(__file__)))
    teacher=None;teacher_before=None;keep_trace=[]
    if keep_mode:
        from hybrid_tsad.left_source_preservation import preservation_loss,gradient_norm
        teacher=copy.deepcopy(model).eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        teacher_before=tensor_hashes(teacher)
        assert teacher_before==tensor_hashes(model)
        dump(root/"PRESERVATION_BINDING.json",dict(mode=keep_mode,coefficient=1.0,
            source_only_second_half=[128,256],teacher_checkpoint_sha256=sha(checkpoint),
            teacher_python_step=source["python_step"],teacher_eval=True,teacher_frozen=True,
            monitored_updates=[1,500,1000,1500,2000],
            helper_sha256=sha(Path(__file__).with_name("left_source_preservation.py"))))
    if arm!="noU":
        with np.load(args.replay_plan)as z:plan={k:z[k].copy()for k in z.files}
        expected=plan_arrays(task,args.seed)
        assert set(plan)==set(expected)and all(np.array_equal(plan[k],expected[k])for k in plan)
        role_values={name:load_role(name)for name in sorted(allowed)}
        pairs=task["arms"][arm];mapping={"T":("T","T_starts"),"S":("S","S_starts"),
                                      "T_prime":("T","T_prime_starts"),"S_prime":("S","S_prime_starts")}
        train=root/"train";train.mkdir();model.train();assert all(p.requires_grad for p in model.parameters())
        optimizer=torch.optim.Adam(model.parameters(),lr=task["training"]["learning_rate"])
        trace=[];started=time.monotonic()
        for step in range(task["training"]["updates"]):
            halves=[]
            for part in pairs:
                role,key=mapping[part];values=role_values[role]
                halves.append(np.stack([np.float32(values[int(a):int(a)+w])for a in plan[key][step]]))
            batch=np.concatenate(halves);assert batch.shape==(256,w,c)
            optimizer.zero_grad();batch_tensor=torch.from_numpy(batch).cuda()
            native_loss=model(batch_tensor,None,None,None)[0]
            loss=native_loss
            if keep_mode:
                penalty=preservation_loss(model,teacher,batch_tensor[128:],keep_mode)
                loss=native_loss+penalty
                if step+1 in [1,500,1000,1500,2000]:
                    parameters=[p for p in model.parameters()if p.requires_grad]
                    gn=gradient_norm(native_loss,parameters);gk=gradient_norm(penalty,parameters)
                    assert np.isfinite(gn)and np.isfinite(gk)
                    keep_trace.append(dict(update=step+1,native_loss=float(native_loss.detach()),
                        preservation_loss=float(penalty.detach()),native_gradient_L2=gn,
                        preservation_gradient_L2=gk,gradient_ratio=gk/gn if gn else None))
            assert torch.isfinite(loss);loss.backward()
            grads=[p.grad for p in model.parameters()if p.grad is not None]
            assert grads and all(torch.isfinite(g).all()for g in grads)
            optimizer.step();assert model.step==source["python_step"]+step+1
            trace.append(dict(update=step+1,loss=float(loss.detach()),python_step=model.step))
            if(step+1)%500==0:print(json.dumps(dict(event="UPDATE",seed=args.seed,arm=arm,**trace[-1])),flush=True)
        endpoint=train/"endpoint2000.pth";torch.save(model.state_dict(),endpoint)
        torch.save(optimizer.state_dict(),train/"optimizer2000.pth")
        dump(train/"TRACE.json",dict(steps=trace))
        dump(train/"RECEIPT.json",dict(status="COMPLETED",seed=args.seed,arm=arm,
            source_checkpoint_sha256=sha(checkpoint),checkpoint_sha256=sha(endpoint),scaler_sha256=sha(scaler_file),
            initial_python_step=source["python_step"],final_python_step=model.step,updates=2000,batch=256,
            native_forward_calls=2000,optimizer="fresh Adam lr1e-4",plan_sha256=sha(args.replay_plan),
            training_roles=accesses.copy(),labels_read=False,CAL_QUERY_read=False,endpoint_only=True,
            trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),
            training_seconds=time.monotonic()-started,allocated=torch.cuda.max_memory_allocated(),
            reserved=torch.cuda.max_memory_reserved()))
        if keep_mode:
            assert teacher_before==tensor_hashes(teacher)and teacher.step==source["python_step"]
            assert all(p.grad is None for p in teacher.parameters())
            dump(train/"PRESERVATION_TRACE.json",dict(mode=keep_mode,coefficient=1.0,records=keep_trace,
                teacher_state_unchanged=True,teacher_gradients_none=True,
                native_training_forward_calls=2000,student_source_eval_forward_calls=2000,
                teacher_source_eval_forward_calls=2000,teacher_only_source_FIT=True))
    else:endpoint=checkpoint
    allowed={"CAL","QUERY"};model.eval();before=tensor_hashes(model);rng=rng_hashes(torch);banks=[];coverage=[]
    torch.cuda.reset_peak_memory_stats()
    for role in ["CAL","QUERY"]:
        values=load_role(role);origin,end=task["roles"][role];lo,hi=task["score_axes"][role]
        winlo,winhi=task.get("score_window_axes",task["score_axes"])[role]
        starts=np.arange(winlo,winhi,w,dtype=np.int64)
        assert winlo>=origin and winhi<=end and np.all((starts-task["scoring"]["anchor"])%w==0)
        chunks=[]
        with torch.no_grad():
            for at in range(0,len(starts),128):
                batch=np.stack([np.float32(values[a-origin:a-origin+w])for a in starts[at:at+128]])
                raw=model.infer(torch.from_numpy(batch).cuda(),None,None,None)[0]
                assert raw.shape==(len(batch),w,c)and torch.isfinite(raw).all();chunks.append(raw.cpu().numpy())
        per_channel=np.concatenate(chunks).reshape(-1,c);axis=np.arange(winlo,winhi,dtype=np.int64)
        available=np.repeat(starts+w-1,w);keep=(axis>=lo)&(axis<hi)
        axis=axis[keep];available=available[keep];per_channel=per_channel[keep];score=per_channel.mean(-1)
        assert len(score)==len(axis)and score.dtype==np.float32
        path=root/(role+".npz")
        np.savez_compressed(path,index=axis,absolute_time_index=axis,available_index=available,
            canonical_score=score,score=score,per_channel=per_channel,valid=np.ones(len(axis),bool),
            window_start=starts,window_end_exclusive=starts+w)
        banks.append(dict(role=role,relative=path.name,sha256=sha(path),points=len(axis),axis=[lo,hi]))
        coverage.append(dict(role=role,input=[origin,end],full_windows=[winlo,winhi],score_axis=[lo,hi],windows=len(starts)))
    assert before==tensor_hashes(model)and rng==rng_hashes(torch)
    assert model.step==source["python_step"]+(0 if arm=="noU"else 2000)
    dump(root/"RECEIPT.json",dict(status="COMPLETED",arm=arm,seed=args.seed,checkpoint_sha256=sha(endpoint),
        source_checkpoint_sha256=sha(checkpoint),scaler_sha256=sha(scaler_file),python_step=model.step,
        initial_python_step=source["python_step"],training_updates=0 if arm=="noU"else 2000,
        banks=banks,coverage=coverage,labels_read=False,readout_state_step_RNG_immutable=True,
        source_checkpoint_unchanged=sha(checkpoint)==source["checkpoint"]["sha256"],
        roles=accesses,task_sha256=sha(args.task),worker_sha256=sha(__file__),
        allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),
        total_worker_seconds=time.monotonic()-began))
    print(json.dumps(dict(event="STAGE_COMPLETED",seed=args.seed,arm=arm,banks=banks)),flush=True)


def queue_main():
    parser=argparse.ArgumentParser();parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--task",type=Path,required=True)
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.task,root/"TASK.json");task=read(root/"TASK.json")
    for seed in task["seeds"]:
        seedroot=root/f"seed{seed}";seedroot.mkdir()
        plan=plan_arrays(task,seed);np.savez_compressed(seedroot/"SAMPLING_PLAN.npz",**plan)
        np.savez_compressed(seedroot/"TARGET_PLAN.npz",base_starts=plan["T_starts"])
        for key,v in plan.items():assert v.shape==(2000,128)and v.min()>=0
        dump(seedroot/"PLAN_FREEZE.json",dict(status="FROZEN_BEFORE_TRAINING_OR_QUERY",seed=seed,
            plan_sha256=sha(seedroot/"SAMPLING_PLAN.npz"),target_plan_sha256=sha(seedroot/"TARGET_PLAN.npz"),
            target_plan_file="TARGET_PLAN.npz",task_sha256=sha(root/"TASK.json"),
            RNG=f"T=default_rng({seed}); S,T_prime,S_prime=SeedSequence({seed}).spawn(3)",
            arms=task["arms"],roles=task["roles"],updates=2000,batch=256,
            worker_sha256=sha(__file__),dispatcher_sha256=sha(Path(__file__).with_name("left_author_u_transfer.py"))))
    sys.path.insert(0,str(args.workspace/"preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    devices={int(row.split(",")[0]):row.split(",")[1].strip().removeprefix("GPU-")for row in rows.splitlines()if row.strip()}
    def job(seed,arm,gpu):
        seedroot=root if arm=="noU"else root/f"seed{seed}"
        life=seedroot/"lifecycles"/arm;life.mkdir(parents=True,exist_ok=False)
        began=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();code=None;cleanup=None;error=None
        try:
            cap=30 if arm=="noU"else 20;dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu],gpu,cap))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="GPU-"+devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
            argv=[sys.executable,"-B","-m","hybrid_tsad.left_author_u_transfer","--workspace",str(args.workspace),
                "--output",str(seedroot/arm),"--task",str(root/"TASK.json"),"--seed",str(seed)]
            if arm!="noU":argv += ["--replay-arm",arm,"--replay-plan",str(seedroot/"SAMPLING_PLAN.npz")]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned=owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,supervisor_pid=os.getpid(),GPU=gpu,UTC=utc,argv=argv,capGiB=cap,timeout_seconds=1800,owned=owned))
                print(json.dumps(dict(event="RUNNING",seed=seed,arm=arm,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=1790)
                finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup["status"]=="NO_LIVE_GROUP_MEMBERS"
        except BaseException as ex:error=repr(ex)
        result=dict(status="COMPLETED"if error is None else"FAILED",seed=seed,arm=arm,GPU=gpu,UTC_start=utc,
            UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),OS_exit_code=code,error=error,cleanup=cleanup,
            actual_GPU_seconds=time.monotonic()-began)
        dump(life/"OS_WAIT.json",result)
        if error:raise RuntimeError(result)
        return result
    try:
        tasks=[(2025,"noU")]+[(seed,arm)for seed in task["seeds"]for arm in task["arms"]]
        def lane(gpu):return[job(seed,arm,gpu)for seed,arm in tasks[gpu::4]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
            jobs=[j for group in pool.map(lane,range(4))for j in group]
        allbanks=[]
        for seed in task["seeds"]:
            seedroot=root/f"seed{seed}";banks=[]
            for arm in task["arms"]:
                for b in read(seedroot/arm/"RECEIPT.json")["banks"]:
                    assert sha(seedroot/arm/b["relative"])==b["sha256"]
                    banks.append(dict(b,relative=arm+"/"+b["relative"],arm=arm))
            seedjobs=[j for j in jobs if j["seed"]==seed and j["arm"]!="noU"]
            dump(seedroot/"RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,banks=banks,jobs=seedjobs,
                 sampling_plan_sha256=sha(seedroot/"SAMPLING_PLAN.npz"),actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in seedjobs)/3600))
            allbanks.extend(dict(b,relative=f"seed{seed}/"+b["relative"],seed=seed)for b in banks)
        for b in read(root/"noU/RECEIPT.json")["banks"]:
            assert sha(root/"noU"/b["relative"])==b["sha256"]
            allbanks.append(dict(b,relative="noU/"+b["relative"],arm="noU",seed=None))
        assert len(allbanks)==20 and len(jobs)==10
        dump(root/"GLOBAL_RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,banks=allbanks,jobs=jobs,
             task_sha256=sha(root/"TASK.json"),actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs)/3600))
    except BaseException as ex:
        dump(root/"FAILED.json",dict(error=repr(ex)));raise


if __name__=="__main__":queue_main()
