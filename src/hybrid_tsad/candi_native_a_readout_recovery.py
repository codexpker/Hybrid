"""Accept reproducible CANDI endpoint caches and fill only absent native noU.

The original source training and U adaptation are not called. The original
DataLoader RNG guard failure and all scores remain immutable.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .candi_native_a import PIN,score_starts
from .left_swat_author import read,sha,dump


def worker(args):
    import torch
    from torch.utils.data import DataLoader,Dataset
    from sklearn.preprocessing import StandardScaler
    root=args.output;workspace=args.workspace;task=read(root/"TASK.json");receipt=read(root/"ADAPT_RECEIPT.json")
    upstream=workspace/task["source"]["upstream"]
    assert read(root.parent/"OFFICIAL_SOURCE.json")["commit"]==PIN
    for spec in read(root.parent/"OFFICIAL_SOURCE.json")["files"]:assert sha(upstream/spec["relative"])==spec["sha256"]
    assert sha(root/"TASK.json")==read(root/"FROZEN.json")["task_sha256"]
    assert sha(root/"ADAPTED_ENDPOINT.pth")==receipt["endpoint_sha256"]
    source_receipt=read(root/"SOURCE_TRAIN_RECEIPT.json")
    assert sha(root/"SOURCE_SELECTED.pth")==source_receipt["source_selected_sha256"]
    assert sha(root/"SCALER.npz")==source_receipt["scaler_sha256"]
    sys.path.insert(0,str(upstream))
    from config import get_cfg_defaults
    from models.mlp.modeling_mlp import MLP
    from tta.candi.adapter_candi import SANA
    cfg=get_cfg_defaults();cfg.merge_from_file(str(root/"CONFIG.yaml"))
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(30*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    with np.load(root/"SCALER.npz")as z:
        scaler=StandardScaler();scaler.mean_=z["mean"];scaler.scale_=z["scale"];scaler.var_=z["variance"]
        scaler.n_samples_seen_=z["n_samples_seen"];scaler.n_features_in_=len(scaler.mean_)
    source=torch.load(root/"SOURCE_SELECTED.pth",map_location="cpu",weights_only=False)
    endpoint=torch.load(root/"ADAPTED_ENDPOINT.pth",map_location="cpu",weights_only=False)
    assert all(torch.equal(v,endpoint["model_state"][k])for k,v in source["model_state"].items())
    target=task["data"]["target_test"];path=workspace/target["relative"];assert sha(path)==target["sha256"]
    raw=np.load(path,mmap_mode="r");w=10
    def states(model):return{k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in model.state_dict().items()}
    def rng():return dict(cpu=sha_raw(torch.get_rng_state().numpy().tobytes()),cuda=[sha_raw(t.cpu().numpy().tobytes())for t in torch.cuda.get_rng_state_all()])
    class Windows(Dataset):
        def __init__(self,values,starts):self.values=values;self.starts=starts
        def __len__(self):return len(self.starts)
        def __getitem__(self,i):return np.array(self.values[self.starts[i]:self.starts[i]+w],dtype=np.float32)
    def normalized(axis):
        starts=score_starts(axis);lo=int(starts[0]);hi=int(starts[-1]+w)
        assert lo>=task["roles"][current_role][0]and hi<=task["roles"][current_role][1]
        return scaler.transform(np.asarray(raw[lo:hi],dtype=np.float32)),starts,lo
    # Full caches already exist. Verify actual endpoint identity with fixed original batches.
    adapted=MLP(cfg).cuda();adapted.sana_in=SANA(cfg).cuda();adapted.sana_out=SANA(cfg).cuda()
    adapted.load_state_dict(endpoint["model_state"],strict=True);adapted.eval();before=states(adapted);rr=rng();accepted=[];sample_checks=[]
    cached=[(root/"CANDI"/(role+".npz")).exists()for role in task["score_axes"]]
    assert all(cached)or not any(cached)
    reused_cache=all(cached);(root/"CANDI").mkdir(exist_ok=True)
    for current_role,axis in task["score_axes"].items():
        path=root/"CANDI"/(current_role+".npz")
        if not reused_cache:
            values,starts,lo=normalized(axis)
            loader=DataLoader(Windows(values,starts-lo),batch_size=256,shuffle=False,drop_last=False,num_workers=0,
                generator=torch.Generator().manual_seed(2025))
            out=[]
            with torch.no_grad():
                for x in loader:out.append(adapted.get_anomaly_scores(x.cuda()).cpu().numpy())
            index=np.arange(*axis)
            np.savez_compressed(path,canonical_score=np.concatenate(out),absolute_time_index=index,available_index=index,
                window_start=starts,window_end_exclusive=starts+w)
        digest=sha(path)
        with np.load(path,allow_pickle=False)as z:
            index=np.arange(*axis);starts=score_starts(axis);scores=z["canonical_score"].copy()
            assert np.array_equal(z["absolute_time_index"],index)and np.array_equal(z["available_index"],index)
            assert np.array_equal(z["window_start"],starts)and np.array_equal(z["window_end_exclusive"],starts+w)
            assert scores.dtype==np.float32 and scores.shape==index.shape and np.isfinite(scores).all()
        values,starts,lo=normalized(axis);n=len(starts)
        batches=sorted(set([0,(n//2//256)*256,((n-1)//256)*256]))if reused_cache else[]
        for at in batches:
            end=min(at+256,n);x=torch.from_numpy(np.stack([values[s-lo:s-lo+w]for s in starts[at:end]])).cuda()
            with torch.no_grad():pred=adapted.get_anomaly_scores(x).cpu().numpy()
            error=float(np.max(np.abs(pred-scores[at:end])))
            assert np.allclose(pred,scores[at:end],rtol=1e-5,atol=1e-6)
            sample_checks.append(dict(role=current_role,relative_batch_start=at,points=end-at,maxabs=error,
                first_endpoint=int(index[at]),last_endpoint=int(index[end-1])))
        assert sha(path)==digest
        accepted.append(dict(relative=str(path.relative_to(root)),role=current_role,axis=axis,sha256=digest))
    assert before==states(adapted)and rr==rng()
    dump(root/"CANDI/READOUT_RECEIPT.json",dict(banks=accepted,labels_read=False,adapter_called=False,optimizer_update_calls=0,
        state_RNG_immutable=not reused_cache,original_CPU_RNG_guard_failure_retained=reused_cache,model_state_and_optimizer_immutable=True,
        accepted_complete_endpoint_cache=reused_cache,small_replay_state_RNG_immutable=True,sample_checks=sample_checks,
        source_selected_sha256=source_receipt["source_selected_sha256"],adapted_endpoint_sha256=receipt["endpoint_sha256"],
        scaler_sha256=source_receipt["scaler_sha256"],score="official whole-window MSE; actual endpoint"))
    del adapted
    native=MLP(cfg).cuda();native.load_state_dict(source["model_state"],strict=True);native.eval();before=states(native);rr=rng();banks=[]
    folder=root/"noU";folder.mkdir(exist_ok=False)
    for current_role,axis in task["score_axes"].items():
        values,starts,lo=normalized(axis)
        loader=DataLoader(Windows(values,starts-lo),batch_size=256,shuffle=False,drop_last=False,num_workers=0,
            generator=torch.Generator().manual_seed(2025))
        scores=[]
        with torch.no_grad():
            for x in loader:scores.append(native.get_anomaly_scores(x.cuda()).cpu().numpy())
        scores=np.concatenate(scores);index=np.arange(*axis);assert scores.shape==index.shape and np.isfinite(scores).all()
        path=folder/(current_role+".npz")
        np.savez_compressed(path,canonical_score=scores,absolute_time_index=index,available_index=index,window_start=starts,window_end_exclusive=starts+w)
        banks.append(dict(relative=str(path.relative_to(root)),sha256=sha(path),role=current_role,axis=axis))
    assert before==states(native)and rr==rng()
    dump(folder/"READOUT_RECEIPT.json",dict(banks=banks,state_RNG_immutable=True,labels_read=False,adapter_called=False,
        optimizer_update_calls=0,source_selected_sha256=source_receipt["source_selected_sha256"],scaler_sha256=source_receipt["scaler_sha256"]))
    dump(root/"RAW_BANK_SEAL.json",dict(status="COMPLETED",banks=accepted+banks,labels_read=False,
        source_selected_sha256=source_receipt["source_selected_sha256"],adapted_endpoint_sha256=receipt["endpoint_sha256"],
        scaler_sha256=source_receipt["scaler_sha256"],task_sha256=sha(root/"TASK.json"),runtime_patch_sha256=sha(root/"RUNTIME_PATCH.json"),
        original_failure_preserved=True,complete_CANDI_caches_reused=reused_cache,no_source_training_repeated=True,recovery_seconds=time.monotonic()-began,
        allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))
    print(dict(event="STAGE_COMPLETED",domain=task["domain"],complete_CANDI_caches_reused=reused_cache,noU_banks=2),flush=True)


def sha_raw(raw):return hashlib.sha256(raw).hexdigest()


def queue(args):
    root=args.output;workspace=args.workspace
    sys.path.insert(0,str(workspace/"preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    devices={int(line.split(",")[0]):line.split(",")[1].strip()for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True).splitlines()}
    def job(domain,gpu):
        original=root/"lifecycles"/domain/"OS_WAIT.json";deadline=time.monotonic()+7200
        while not original.exists():
            if time.monotonic()>deadline:raise TimeoutError("Original dependency not terminal")
            time.sleep(1)
        prior=read(original);assert prior["OS_exit_code"]==1
        assert not(root/domain/"RAW_BANK_SEAL.json").exists()
        restore_adaptation=not(root/domain/"ADAPTED_ENDPOINT.pth").exists()
        if restore_adaptation:assert domain=="SWAT"and(root/domain/"SOURCE_SELECTED.pth").exists()
        life=root/"readout_recovery_lifecycles"/domain;life.mkdir(parents=True,exist_ok=False)
        start=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();code=None;cleanup=None;error=None
        try:
            dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu].removeprefix("GPU-"),gpu,30))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",WANDB_MODE="disabled")
            module="hybrid_tsad.candi_native_a_adapt_recovery"if restore_adaptation else"hybrid_tsad.candi_native_a_readout_recovery"
            argv=[sys.executable,"-B","-m",module,"worker","--workspace",str(workspace),"--output",str(root/domain)]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,GPU=gpu,argv=argv,UTC=utc,capGiB=30,timeout_seconds=1800))
                print(dict(event="RUNNING_FROZEN_READOUT_RECOVERY",domain=domain,pid=child.pid,GPU=gpu),flush=True)
                try:code=child.wait(timeout=1790)
                finally:cleanup=owned_session.stop_group(child,grace=5)
                assert code==0 and cleanup["status"]=="NO_LIVE_GROUP_MEMBERS"
        except BaseException as exc:error=repr(exc)
        result=dict(domain=domain,status="COMPLETED"if error is None else"FAILED",GPU=gpu,OS_exit_code=code,error=error,
            cleanup=cleanup,UTC_start=utc,UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_GPU_seconds=time.monotonic()-start)
        dump(life/"OS_WAIT.json",result)
        if error:raise RuntimeError(result)
        return result
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:jobs=list(pool.map(lambda item:job(*item),zip(["SWAT","PSM"],args.gpus)))
        banks=[];prior=[]
        for domain in ["SWAT","PSM"]:
            d=root/domain;prior.append(read(root/"lifecycles"/domain/"OS_WAIT.json"))
            for b in read(d/"RAW_BANK_SEAL.json")["banks"]:
                assert sha(d/b["relative"])==b["sha256"];banks.append(dict(b,relative=domain+"/"+b["relative"],domain=domain))
        dump(root/"GLOBAL_RAW_BANK_SEAL.json",dict(status="COMPLETED",jobs=jobs,original_compute_jobs=prior,banks=banks,
            no_source_training_repeated=True,PSM_complete_CANDI_caches_reused=True,SWAT_missing_adaptation_only_restored=True,
            original_CPU_RNG_and_FPM_gate_failures_preserved=True,
            actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs+prior)/3600,labels_read=False))
    except BaseException as exc:dump(root/"READOUT_RECOVERY_FAILED.json",dict(error=repr(exc)));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["worker","queue"]);p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--gpus",type=int,nargs="+",default=[0,1]);a=p.parse_args()
    {"worker":worker,"queue":queue}[a.phase](a)
