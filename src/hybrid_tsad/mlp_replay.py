"""Plain official CANDI MLP with fixed LEFT endpoint-matched TT/R replay.

No source training, SANA, FPM, schedule or new scoring formula is introduced.
The original192-point exposure plans identify endpoints; only the final10
observations at each endpoint are presented to the original MLP.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np

from .candi_native_a import PIN,score_starts
from .left_swat_author import sha,read,dump

EXPERIMENT="mlp_replay_20261119_v1"


def mapped_windows(original_starts,role):
    original_starts=np.asarray(original_starts)
    assert original_starts.shape==(2000,128)and np.issubdtype(original_starts.dtype,np.integer)
    assert original_starts.min()>=0 and original_starts.max()+192<=role[1]-role[0]
    return original_starts+182,original_starts+role[0]+191


def batch_from_plan(values,starts,arm,step):
    assert arm in ["TT","R"]
    targets=starts["T_starts"][step]+182
    other=starts["T_prime_starts"if arm=="TT"else"S_starts"][step]+182
    a=np.array(values["T"][targets[:,None]+np.arange(10)],dtype=np.float32)
    b=np.array(values["T"if arm=="TT"else"S"][other[:,None]+np.arange(10)],dtype=np.float32)
    return np.concatenate([a,b])


def tensor_hashes(model):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in model.state_dict().items()}


def rng_hashes(torch):
    return dict(CPU=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        CUDA=[hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest()for v in torch.cuda.get_rng_state_all()])


def optimizer_hash(optimizer):
    value=hashlib.sha256();state=optimizer.state_dict()
    value.update(json.dumps(state["param_groups"],sort_keys=True).encode())
    for parameter,items in sorted(state["state"].items()):
        for key,tensor in sorted(items.items()):
            value.update(str((parameter,key)).encode());value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def prepare(args):
    w=args.workspace;root=args.output;root.mkdir(parents=True,exist_ok=False)
    for domain in ["SWAT","PSM"]:
        prior=w/"results/candi_native_a_20261118_v1"/domain;task=read(prior/"TASK.json")
        assert read(prior.parent/"OFFICIAL_SOURCE.json")["commit"]==PIN
        source_receipt=read(prior/"SOURCE_TRAIN_RECEIPT.json")
        source_pt=prior/"SOURCE_SELECTED.pth";scaler=prior/"SCALER.npz";config=prior/"CONFIG.yaml"
        assert sha(source_pt)==source_receipt["source_selected_sha256"]and sha(scaler)==source_receipt["scaler_sha256"]
        task["source"]=dict(checkpoint={"relative":str(source_pt.relative_to(w)),"sha256":sha(source_pt)},
            scaler={"relative":str(scaler.relative_to(w)),"sha256":sha(scaler)},
            config={"relative":str(config.relative_to(w)),"sha256":sha(config)},
            upstream=task["source"]["upstream"],commit=PIN,
            source_manifest={"relative":str((prior.parent/"OFFICIAL_SOURCE.json").relative_to(w)),"sha256":sha(prior.parent/"OFFICIAL_SOURCE.json")},
            normal_FIT=task["source"]["normal_FIT"],normal_SELECT=task["source"]["normal_SELECT"],source_training_repeated=False)
        for stale in ["CANDI","adaptation"]:task.pop(stale,None)
        task.update(experiment=EXPERIMENT,seeds=[2025],arms={"TT":["T10","Tprime10"],"R":["T10","S10"]},
            training=dict(batch=256,updates=2000,optimizer="Adam",learning_rate=1e-4,betas=[.9,.999],eps=1e-8,
                weight_decay=0,all_parameters_trainable=True,full_native_forward_per_update=1,selection="fixed endpoint2000"),
            mapping=dict(original_window=192,actual_window=10,relative_start_add=182,absolute_end_add=191,
                replay_sample_source="accepted seed2025 LEFT A plans; no resampling; each record/role remains separate"))
        d=root/domain;d.mkdir();plan=w/("results/left_author_swat_source_replay_20261118_v1/SAMPLING_PLAN.npz"if domain=="SWAT"else"results/left_author_psm_source_replay_20261118_v1/seed2025/SAMPLING_PLAN.npz")
        shutil.copyfile(plan,d/"SAMPLING_PLAN.npz");derived={}
        with np.load(plan)as z:
            for key in ["T_starts","T_prime_starts","S_starts","S_prime_starts"]:
                role=task["roles"]["U_FIT"if key.startswith("T")else"SOURCE_FIT"]
                start,end=mapped_windows(z[key],role);derived[key+"_relative10"]=start;derived[key+"_absolute_endpoint"]=end
        np.savez_compressed(d/"MAPPED_ENDPOINT_PLAN.npz",**derived)
        task["replay_plan"]=dict(original_relative=str(plan.relative_to(w)),sha256=sha(plan),mapped_sha256=sha(d/"MAPPED_ENDPOINT_PLAN.npz"))
        dump(d/"TASK.json",task)
        dump(d/"FROZEN.json",dict(task_sha256=sha(d/"TASK.json"),plan_sha256=sha(d/"SAMPLING_PLAN.npz"),
            mapped_plan_sha256=sha(d/"MAPPED_ENDPOINT_PLAN.npz"),worker_sha256=sha(Path(__file__))))


def worker(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset,DataLoader
    from sklearn.preprocessing import StandardScaler
    root=args.output;root.mkdir(parents=True,exist_ok=False);w=args.workspace;task=read(args.task);arm=args.arm
    frozen=read(args.task.parent/"FROZEN.json");assert sha(args.task)==frozen["task_sha256"]and sha(Path(__file__))==frozen["worker_sha256"]
    plan_file=args.task.parent/"SAMPLING_PLAN.npz";assert sha(plan_file)==frozen["plan_sha256"]==task["replay_plan"]["sha256"]
    assert sha(w/task["replay_plan"]["original_relative"])==sha(plan_file)
    mapped=args.task.parent/"MAPPED_ENDPOINT_PLAN.npz";assert sha(mapped)==frozen["mapped_plan_sha256"]
    source=task["source"]
    for key in ["checkpoint","scaler","config","source_manifest"]:assert sha(w/source[key]["relative"])==source[key]["sha256"]
    official=read(w/source["source_manifest"]["relative"]);assert official["commit"]==PIN
    upstream=w/source["upstream"]
    for item in official["files"]:assert sha(upstream/item["relative"])==item["sha256"]
    sys.path.insert(0,str(upstream));from config import get_cfg_defaults
    from models.mlp.modeling_mlp import MLP
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats();random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
    phase="setup";phase_started=time.monotonic();exit_error=None
    try:
        cfg=get_cfg_defaults();cfg.merge_from_file(str(w/source["config"]["relative"]))
        shutil.copyfile(w/source["config"]["relative"],root/"SOURCE_CONFIG.yaml")
        assert cfg.DATA.WIN_SIZE==10 and cfg.DATA.N_VAR==task["data"]["target_test"]["shape"][1]
        model=MLP(cfg).cuda();loaded=torch.load(w/source["checkpoint"]["relative"],map_location="cpu",weights_only=False)
        model.load_state_dict(loaded["model_state"],strict=True);assert not any("sana"in k for k in model.state_dict())
        for p in model.parameters():p.requires_grad_(True)
        initial=tensor_hashes(model);nparams=sum(p.numel()for p in model.parameters())
        with np.load(w/source["scaler"]["relative"])as z:
            scaler=StandardScaler();scaler.mean_=z["mean"];scaler.scale_=z["scale"];scaler.var_=z["variance"]
            scaler.n_samples_seen_=z["n_samples_seen"];scaler.n_features_in_=len(scaler.mean_)
        with np.load(plan_file)as z:plans={k:z[k].copy()for k in z.files}
        values={};roles={"T":"U_FIT"}if arm=="TT"else{"T":"U_FIT","S":"SOURCE_FIT"}
        for key,role in roles.items():
            spec=task["data"]["target_test"if key=="T"else"source_train"];path=w/spec["relative"];assert sha(path)==spec["sha256"]
            raw=np.load(path,mmap_mode="r");lo,hi=task["roles"][role]
            values[key]=scaler.transform(np.asarray(raw[lo:hi],dtype=np.float32));assert np.isfinite(values[key]).all()
        for key in ["T_starts","T_prime_starts","S_starts"]:
            start,end=mapped_windows(plans[key],task["roles"]["U_FIT"if key.startswith("T")else"SOURCE_FIT"])
            with np.load(mapped)as z:assert np.array_equal(start,z[key+"_relative10"])and np.array_equal(end,z[key+"_absolute_endpoint"])
        batch=batch_from_plan(values,plans,arm,0);model.eval();x=torch.from_numpy(batch).cuda()
        with torch.no_grad():
            native=model.get_anomaly_scores(x);direct=F.mse_loss(model(x),x,reduction="none").mean((1,2))
        assert torch.equal(native,direct)and initial==tensor_hashes(model)
        dump(root/"INITIAL_IDENTITY.json",dict(seed=2025,arm=arm,source_checkpoint_sha256=source["checkpoint"]["sha256"],
            scaler_sha256=source["scaler"]["sha256"],source_config_sha256=source["config"]["sha256"],
            state_sha256=initial,rng=rng_hashes(torch),parameters=nparams,trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),
            plan_sha256=sha(plan_file),mapped_plan_sha256=sha(mapped),native_score_bitexact=True,
            first_T_numeric_sha256=hashlib.sha256(batch[:128].tobytes()).hexdigest(),first_T_native_score_sha256=hashlib.sha256(native[:128].cpu().numpy().tobytes()).hexdigest(),
            first_T_endpoints=(task["roles"]["U_FIT"][0]+plans["T_starts"][0]+191).tolist(),labels_read=False,CAL_QUERY_read=False))
        dump(root/"SETUP_PHASE_PEAK.json",dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-phase_started))
        phase="training";phase_started=time.monotonic();torch.cuda.reset_peak_memory_stats();model.train()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4);assert len(optimizer.state)==0;trace=[];first_grad=None
        for step in range(2000):
            x=torch.from_numpy(batch_from_plan(values,plans,arm,step)).cuda();prediction=model(x);loss=F.mse_loss(prediction,x)
            assert prediction.requires_grad and torch.isfinite(loss);optimizer.zero_grad();loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all()for p in model.parameters())
            if step==0:first_grad={k:float(p.grad.norm())for k,p in model.named_parameters()}
            optimizer.step();trace.append(dict(update=step+1,loss=float(loss.detach()),batch=256))
        assert any(v>0 for v in first_grad.values());final=tensor_hashes(model)
        assert final!=initial
        dump(root/"TRAINING_PHASE_PEAK.json",dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-phase_started))
        torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),cfg=cfg.dump(),updates=2000,seed=2025,arm=arm),root/"endpoint2000.pth")
        dump(root/"TRACE.json",dict(steps=trace))
        dump(root/"TRAIN_RECEIPT.json",dict(domain=task["domain"],arm=arm,seed=2025,updates=2000,batch=256,native_forward_calls=2000,
            optimizer="fresh Adam lr1e-4 defaults: betas(.9,.999),eps1e-8,weight_decay0",parameters=nparams,all_parameters_trainable=True,
            source_checkpoint_sha256=source["checkpoint"]["sha256"],scaler_sha256=source["scaler"]["sha256"],source_config_sha256=source["config"]["sha256"],
            endpoint_sha256=sha(root/"endpoint2000.pth"),plan_sha256=sha(plan_file),mapped_plan_sha256=sha(mapped),
            first_gradient_norms=first_grad,changed_parameter_tensors=sum(final[k]!=initial[k]for k in final),total_parameter_tensors=len(final),
            training_roles=list(roles.values()),source_training_repeated=False,labels_read=False,CAL_QUERY_read=False,endpoint_selection="fixed2000"))
        phase="readout";phase_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        torch.cuda.set_per_process_memory_fraction(30*1024**3/torch.cuda.get_device_properties(0).total_memory);model.eval()
        before=tensor_hashes(model);rng=rng_hashes(torch);banks=[];optimizer_before=optimizer_hash(optimizer)
        class Windows(Dataset):
            def __init__(self,v,starts):self.v=v;self.starts=starts
            def __len__(self):return len(self.starts)
            def __getitem__(self,i):return np.array(self.v[self.starts[i]:self.starts[i]+10],dtype=np.float32)
        spec=task["data"]["target_test"];path=w/spec["relative"];assert sha(path)==spec["sha256"];raw=np.load(path,mmap_mode="r")
        for role,axis in task["score_axes"].items():
            starts=score_starts(axis);lo,hi=int(starts[0]),int(starts[-1]+10)
            assert lo>=task["roles"][role][0]and hi<=task["roles"][role][1]
            scaled=scaler.transform(np.asarray(raw[lo:hi],dtype=np.float32))
            loader=DataLoader(Windows(scaled,starts-lo),batch_size=256,shuffle=False,drop_last=False,num_workers=0,generator=torch.Generator().manual_seed(2025))
            scores=[]
            with torch.no_grad():
                for x in loader:scores.append(model.get_anomaly_scores(x.cuda()).cpu().numpy())
            scores=np.concatenate(scores);index=np.arange(*axis);assert scores.shape==index.shape and np.isfinite(scores).all()
            path=root/(role+".npz");np.savez_compressed(path,canonical_score=scores,absolute_time_index=index,available_index=index,
                window_start=starts,window_end_exclusive=starts+10)
            banks.append(dict(role=role,relative=role+".npz",sha256=sha(path),axis=axis))
        assert before==tensor_hashes(model)and rng==rng_hashes(torch)and optimizer_before==optimizer_hash(optimizer)
        dump(root/"READOUT_PHASE_PEAK.json",dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-phase_started))
        assert all(sha(w/source[key]["relative"])==source[key]["sha256"]for key in ["checkpoint","scaler","config"])
        dump(root/"RAW_BANK_SEAL.json",dict(status="COMPLETED",banks=banks,seed=2025,arm=arm,
            source_checkpoint_sha256=source["checkpoint"]["sha256"],scaler_sha256=source["scaler"]["sha256"],endpoint_sha256=sha(root/"endpoint2000.pth"),
            native_score="official whole-window MSE; actual10-window endpoint",labels_read=False,readout_state_RNG_immutable=True,optimizer_update_calls_in_readout=0,
            readout_optimizer_state_immutable=True,optimizer_state_sha256=optimizer_before,
            source_assets_unchanged=True,plan_sha256=sha(plan_file),worker_sha256=sha(Path(__file__))))
        print(json.dumps(dict(event="STAGE_COMPLETED",domain=task["domain"],arm=arm,banks=2)),flush=True)
    except BaseException as exc:exit_error=repr(exc);raise
    finally:
        dump(root/"EXIT_PHASE_PEAK.json",dict(phase=phase,error=exit_error,phase_seconds=time.monotonic()-phase_started,
            allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))


def queue(args):
    root=args.output;w=args.workspace;sys.path.insert(0,str(w/"preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    devices={int(x.split(",")[0]):x.split(",")[1].strip()for x in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True).splitlines()}
    tasks=[(d,a,g)for(d,a),g in zip([(d,a)for d in ["SWAT","PSM"]for a in ["TT","R"]],args.gpus)]
    assert len(tasks)==4
    def job(domain,arm,gpu):
        life=root/"lifecycles"/(domain+"_"+arm);life.mkdir(parents=True,exist_ok=False);start=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        code=None;cleanup=None;error=None
        try:
            dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu].removeprefix("GPU-"),gpu,20))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",WANDB_MODE="disabled")
            argv=[sys.executable,"-B","-m","hybrid_tsad.mlp_replay","worker","--workspace",str(w),"--output",str(root/domain/arm),"--task",str(root/domain/"TASK.json"),"--arm",arm]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,GPU=gpu,UTC=utc,argv=argv,train_capGiB=20,readout_capGiB=30,timeout_seconds=1800))
                print(json.dumps(dict(event="RUNNING",domain=domain,arm=arm,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=1790)
                finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup["status"]=="NO_LIVE_GROUP_MEMBERS"
        except BaseException as exc:error=repr(exc)
        result=dict(domain=domain,arm=arm,GPU=gpu,status="COMPLETED"if error is None else"FAILED",OS_exit_code=code,error=error,cleanup=cleanup,
            UTC_start=utc,UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_GPU_seconds=time.monotonic()-start)
        dump(life/"OS_WAIT.json",result)
        if error:raise RuntimeError(result)
        return result
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:jobs=list(pool.map(lambda item:job(*item),tasks))
        banks=[]
        for domain,arm,_ in tasks:
            folder=root/domain/arm
            for bank in read(folder/"RAW_BANK_SEAL.json")["banks"]:
                assert sha(folder/bank["relative"])==bank["sha256"]
                banks.append(dict(bank,domain=domain,arm=arm,relative=f"{domain}/{arm}/"+bank["relative"]))
        dump(root/"GLOBAL_RAW_BANK_SEAL.json",dict(status="COMPLETED",jobs=jobs,banks=banks,labels_read=False,
            source_training_repeated=False,actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs)/3600))
    except BaseException as exc:dump(root/"FAILED.json",dict(error=repr(exc)));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["prepare","worker","queue"]);p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--task",type=Path);p.add_argument("--arm",choices=["TT","R"])
    p.add_argument("--gpus",type=int,nargs="+",default=[0,1,2,3]);a=p.parse_args();{"prepare":prepare,"worker":worker,"queue":queue}[a.phase](a)
