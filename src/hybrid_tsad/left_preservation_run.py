"""Finite native-state gates and fixed four-job source-preservation execution."""
import argparse
import concurrent.futures
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from hybrid_tsad.left_swat_author import sha,read,dump


def gate(args):
    import torch
    from hybrid_tsad.left_source_preservation import preservation_loss,gradient_norm,views
    from hybrid_tsad.left_domain_replay import tensor_hashes
    began=time.monotonic();task=read(args.task);src=task["source"];workspace=args.workspace
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory)
    random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
    sys.path.insert(0,str(workspace/src["upstream"]));from models.Left.LEFT import Left
    model=Left(SimpleNamespace(**read(workspace/src["model_config"]["relative"]))).cuda()
    pt=workspace/src["checkpoint"]["relative"];assert sha(pt)==src["checkpoint"]["sha256"]
    loaded=model.load_state_dict(torch.load(pt,map_location="cpu",weights_only=True),strict=False)
    assert not loaded.missing_keys and all(k.endswith(("total_ops","total_params"))for k in loaded.unexpected_keys)
    model.step=src["python_step"];model.train();teacher=copy.deepcopy(model).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    scaler=workspace/src["scaler"]["relative"]
    if src["scaler"]["format"]=="npz":
        with np.load(scaler)as z:mean=z["mean"].copy();scale=z["scale"].copy()
    else:st=read(scaler)["statistics"];mean=np.asarray(st["mean"]);scale=np.asarray(st["scale"])
    with np.load(args.plan)as z:plan={k:z[k].copy()for k in z.files}
    role_values={}
    for short,key,role in [("S","source_train","SOURCE_FIT"),("T","target_test","U_FIT")]:
        raw=np.load(workspace/task["data"][key]["relative"],mmap_mode="r");lo,hi=task["roles"][role]
        role_values[short]=(np.array(raw[lo:hi],dtype=np.float64,order="F",copy=True)-mean)/scale
    halves=[]
    for short in ["T","S"]:
        halves.append(np.stack([np.float32(role_values[short][int(a):int(a)+192])for a in plan[short+"_starts"][0]]))
    batch=torch.from_numpy(np.concatenate(halves)).cuda();source=batch[128:]
    assert batch.shape[0]==256 and source.shape[0]==128
    frozen_teacher=tensor_hashes(teacher)
    near={mode:float(preservation_loss(model,teacher,source,mode).detach())for mode in ["P","V"]}
    prediction=views(model,source,grad=True);reference=views(teacher,source,grad=False)
    view_errors=[]
    for name,a,b in zip(["time","inverse_freq","multi_scale"],prediction,reference):
        mse=float((a-b).square().mean().detach());energy=float(b.square().mean())
        view_errors.append(dict(view=name,max_abs=float((a-b).abs().max().detach()),MSE=mse,
                                teacher_energy=energy,relative_MSE=mse/max(energy,1e-12)))
    assert max(x["relative_MSE"]for x in view_errors)<=1e-9
    del prediction,reference
    initial=copy.deepcopy(model.state_dict())
    bias=[p for n,p in model.named_parameters()if n.startswith("Dt.")and n.endswith("bias")][-1]
    with torch.no_grad():bias.add_(.01)
    perturbed={}
    for mode in ["P","V"]:
        value=preservation_loss(model,teacher,source,mode);gn=gradient_norm(value,list(model.parameters()))
        assert float(value.detach())>0 and gn>0
        perturbed[mode]=dict(loss=float(value.detach()),gradient_L2=gn)
    assert perturbed["V"]["loss"]+1e-12>=perturbed["P"]["loss"]
    model.load_state_dict(initial)
    # One heterogeneous mode must survive a grad-enabled auxiliary call.
    dropouts=[m for m in model.modules()if isinstance(m,torch.nn.Dropout)]
    if dropouts:dropouts[0].eval()
    modes={n:m.training for n,m in model.named_modules()}
    _=preservation_loss(model,teacher,source,"P")
    assert modes=={n:m.training for n,m in model.named_modules()};model.train()
    states=copy.deepcopy(model.state_dict());cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()
    py=random.getstate();nr=np.random.get_state();step=model.step
    def tree(value):
        if torch.is_tensor(value):return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if isinstance(value,dict):return{k:tree(v)for k,v in value.items()}
        if isinstance(value,(tuple,list)):return[tree(v)for v in value]
        return repr(value)
    def snapshot():return dict(state=tensor_hashes(model),step=model.step,
        modes={n:m.training for n,m in model.named_modules()},last_x_res=tree(model.last_x_res),cache=tree(model._ms_cache),
        CPU=tree(torch.get_rng_state()),CUDA=[tree(v)for v in torch.cuda.get_rng_state_all()],
        python=hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),numpy=hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest())
    results={}
    for mode in ["native","P","V"]:
        model.load_state_dict(states);model.step=step;model.train();model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda);random.setstate(py);np.random.set_state(nr)
        loss=model(batch,None,None,None)[0];before=snapshot()
        if mode!="native":
            aux=preservation_loss(model,teacher,source,mode);loss=loss+0*aux
            assert before==snapshot()
        loss.backward()
        grads={n:p.grad.detach().cpu().clone()for n,p in model.named_parameters()if p.grad is not None}
        assert all(torch.isfinite(g).all()for g in grads.values())
        results[mode]=dict(loss=float(loss.detach()),gradients=grads,after=snapshot())
    compare={}
    for mode in ["P","V"]:
        assert results[mode]["after"]==results["native"]["after"]
        assert results[mode]["gradients"].keys()==results["native"]["gradients"].keys()
        d=max(float((results[mode]["gradients"][k]-results["native"]["gradients"][k]).abs().max())for k in results[mode]["gradients"])
        assert d<=1e-5 and abs(results[mode]["loss"]-results["native"]["loss"])<=1e-7
        compare[mode]=dict(loss=results[mode]["loss"],max_gradient_difference=d,full_state_cache_modes_all_RNG_exact=True)
    assert frozen_teacher==tensor_hashes(teacher)
    dump(out/"RECEIPT.json",dict(status="PASS",domain=task["domain"],same_weight_near_zero=near,
        same_weight_view_errors=view_errors,relative_MSE_tolerance=1e-9,
        perturbed_source_response=perturbed,lambda_zero=compare,heterogeneous_modes_restored=True,
        input_shape=list(batch.shape),source_shape=list(source.shape),teacher_unchanged=True,
        native_training_forwards_discarded=3,source_only=True,labels_read=False,
        checkpoint_sha256=sha(pt),scaler_sha256=sha(scaler),plan_sha256=sha(args.plan),
        helper_sha256=sha(Path(__file__).with_name("left_source_preservation.py")),
        worker_seconds=time.monotonic()-began,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))


def queue(args):
    root=args.output;root.mkdir(parents=True,exist_ok=args.resume);workspace=args.workspace
    for domain in ["SWAT","PSM"]:
        d=root/domain;d.mkdir(exist_ok=args.resume)
        config=args.configs/f"left-preservation-{domain.lower()}-20261118.json"
        if args.resume:assert sha(config)==sha(d/"TASK.json")
        else:shutil.copyfile(config,d/"TASK.json")
        plan=(workspace/"results/left_author_swat_source_replay_20261118_v1/SAMPLING_PLAN.npz"if domain=="SWAT"else
              workspace/"results/left_author_psm_source_replay_20261118_v1/seed2025/SAMPLING_PLAN.npz")
        if args.resume:assert sha(plan)==sha(d/"SAMPLING_PLAN.npz")
        else:shutil.copyfile(plan,d/"SAMPLING_PLAN.npz")
        freeze=dict(task_sha256=sha(d/"TASK.json"),plan_sha256=sha(d/"SAMPLING_PLAN.npz"),
            replay_arm="R",seed=2025,modes=["P","V"],coefficient=1.0,
            teacher_source_only=True,native_updates=2000,native_batch=256,
            helper_sha256=sha(Path(__file__).with_name("left_source_preservation.py")),
            worker_sha256=sha(Path(__file__).with_name("left_domain_replay.py")))
        if args.resume:assert read(d/"FROZEN.json")==freeze
        else:dump(d/"FROZEN.json",freeze)
    sys.path.insert(0,str(workspace/"preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    rows=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
    devices={int(row.split(",")[0]):row.split(",")[1].strip().removeprefix("GPU-")for row in rows.splitlines()if row.strip()}
    def job(domain,mode,gpu):
        phase=("gate_retry1"if args.resume else"gate")if mode is None else mode
        life=root/"lifecycles"/(domain+"_"+phase);life.mkdir(parents=True,exist_ok=False)
        began=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();code=None;cleanup=None;error=None
        try:
            dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu],gpu,20))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES="GPU-"+devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1")
            if mode is None:
                argv=[sys.executable,"-B","-m","hybrid_tsad.left_preservation_run","gate","--workspace",str(workspace),
                    "--task",str(root/domain/"TASK.json"),"--plan",str(root/domain/"SAMPLING_PLAN.npz"),
                    "--output",str(root/domain/("CUDA_GATE_RETRY1"if args.resume else"CUDA_GATE"))]
            else:
                argv=[sys.executable,"-B","-m","hybrid_tsad.left_author_u_transfer","--workspace",str(workspace),
                    "--task",str(root/domain/"TASK.json"),"--replay-plan",str(root/domain/"SAMPLING_PLAN.npz"),
                    "--replay-arm","R","--seed","2025","--preservation-mode",mode,"--output",str(root/domain/mode)]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned=owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,supervisor_pid=os.getpid(),GPU=gpu,UTC=utc,argv=argv,capGiB=20,timeout_seconds=1800,owned=owned))
                print(json.dumps(dict(event="RUNNING",domain=domain,mode=phase,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=1790)
                finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup["status"]=="NO_LIVE_GROUP_MEMBERS"
        except BaseException as ex:error=repr(ex)
        result=dict(status="COMPLETED"if error is None else"FAILED",domain=domain,mode=phase,GPU=gpu,UTC_start=utc,
            UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),OS_exit_code=code,error=error,cleanup=cleanup,
            actual_GPU_seconds=time.monotonic()-began)
        dump(life/"OS_WAIT.json",result)
        if error:raise RuntimeError(result)
        return result
    try:
        gatejobs=[job(domain,None,args.gpus[0])for domain in ["SWAT","PSM"]]
        for domain in ["SWAT","PSM"]:
            gatepath=root/domain/("CUDA_GATE_RETRY1"if args.resume else"CUDA_GATE")/"RECEIPT.json"
            assert read(gatepath)["status"]=="PASS"
            dump(root/domain/"ACCEPTED_GATE.json",dict(status="ACCEPTED",receipt_relative=str(gatepath.relative_to(root/domain)),
                 receipt_sha256=sha(gatepath),relative_MSE_tolerance=1e-9,original_failed_gate_preserved=args.resume))
        tasks=[("SWAT","P"),("SWAT","V"),("PSM","P"),("PSM","V")]
        def lane(i):return[job(domain,mode,args.gpus[i])for domain,mode in tasks[i::len(args.gpus)]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus))as pool:
            jobs=[j for lanejobs in pool.map(lane,range(len(args.gpus)))for j in lanejobs]
        allbanks=[]
        for domain in ["SWAT","PSM"]:
            banks=[]
            for mode in ["P","V"]:
                for bank in read(root/domain/mode/"RECEIPT.json")["banks"]:
                    assert sha(root/domain/mode/bank["relative"])==bank["sha256"]
                    banks.append(dict(bank,relative=mode+"/"+bank["relative"],mode=mode))
            d_jobs=[j for j in jobs+gatejobs if j["domain"]==domain]
            dump(root/domain/"RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,banks=banks,jobs=d_jobs,
                 actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in d_jobs)/3600))
            allbanks.extend(dict(b,relative=domain+"/"+b["relative"],domain=domain)for b in banks)
        failed=[]
        if args.resume:failed=[read(root/"lifecycles/SWAT_gate/OS_WAIT.json")]
        diagnostic=read(root/"DIAGNOSTIC_COST.json")if(root/"DIAGNOSTIC_COST.json").exists()else None
        cost=sum(j["actual_GPU_seconds"]for j in jobs+gatejobs+failed)
        if diagnostic:cost+=diagnostic["GPU_seconds_upper_bound"]
        dump(root/"GLOBAL_RAW_BANK_SEAL.json",dict(status="COMPLETED",no_labels_decoded=True,banks=allbanks,jobs=jobs,
             gate_jobs=gatejobs,failed_gate_jobs=failed,diagnostic_cost=diagnostic,
             measured_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs+gatejobs+failed)/3600,
             actual_GPU_hours=cost/3600,GPU_cost_includes_diagnostic_upper_bound=diagnostic is not None))
    except BaseException as ex:
        dump(root/("RESUME_FAILED.json"if args.resume else"FAILED.json"),dict(error=repr(ex)));raise


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["gate","queue"])
    p.add_argument("--workspace",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--task",type=Path);p.add_argument("--plan",type=Path);p.add_argument("--configs",type=Path)
    p.add_argument("--gpus",type=int,nargs="+",default=[0,2,3]);p.add_argument("--resume",action="store_true");a=p.parse_args()
    gate(a)if a.phase=="gate"else queue(a)


if __name__=="__main__":main()
