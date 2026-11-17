"""Fixed two-arm native source-statistics queue with process/GPU accounting and no score selection."""
import argparse,concurrent.futures,json,os,subprocess,sys,time,signal
from pathlib import Path
from .timercd_fixed import sha
from .timercd_fixed_queue import inventory

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    with Path(p).open("x")as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["profile","train","score"]);p.add_argument("--workspace",type=Path,required=True);p.add_argument("--runtime",type=Path,required=True);p.add_argument("--config",type=Path,required=True);a=p.parse_args()
    w=a.workspace.resolve();rt=a.runtime.resolve();config=a.config.resolve();cfg=read(config);root=w/cfg["output"];out=root/(a.phase+"_controller");out.mkdir(exist_ok=False)
    free=[d for d in inventory()if not d["occupied"]and d["used_MiB"]<1000 and d["free_MiB"]>32000][:(4 if a.phase=="score" else 2)]
    if not free:raise RuntimeError("No free GPU; no external eviction")
    dump(out/"START.json",dict(pid=os.getpid(),start_unix=time.time(),inventory=inventory(),config_sha256=sha(config),runtime_files={str(p.relative_to(rt)):sha(p)for p in sorted(rt.rglob("*.py"))}))
    if a.phase in ["profile","train"]:tasks=[(arm,None,None)for arm in cfg["arms"]]
    else:
        for arm in cfg["arms"]:assert read(root/"train"/arm/"TRAIN_RECEIPT.json")["status"]=="COMPLETED"
        tasks=[(arm,rid,permission)for arm in cfg["arms"]for rid in cfg["records"]for permission in ["B_fit_only"]]
    def job(task,gpu):
        arm,rid,permission=task;info=next(x for x in inventory()if x["gpu"]==gpu)
        for attempt in range(60):
            info=next(x for x in inventory()if x["gpu"]==gpu)
            if not info["occupied"]and info["used_MiB"]<1000:break
            time.sleep(1)
        else:raise RuntimeError("GPU remained occupied; no external eviction")
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(rt/"src"),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
        cmd=[sys.executable,"-B","-m","hybrid_tsad.timercd_reference_runner",a.phase,"--workspace",str(w),"--config",str(config),"--arm",arm,"--device","cuda:0"]
        if rid:cmd.extend(["--record",rid,"--permission",permission])
        key="__".join(x for x in task if x);start=time.time();limit=7200 if a.phase=="train" else 3600
        with(out/(key+".log")).open("xb")as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            receipt=dict(pid=child.pid,arm=arm,record=rid,permission=permission,gpu=gpu,start_unix=start,command=cmd,timeout_seconds=limit)
            dump(out/(key+"_START.json"),receipt);print(json.dumps(dict(event="START",**receipt)),flush=True)
            try:rc=child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                rc=124
        end=time.time();receipt.update(exit_code=rc,end_unix=end,seconds=end-start,GPU_hours=(end-start)/3600);dump(out/(key+"_EXIT.json"),receipt);print(json.dumps(dict(event="EXIT",**receipt)),flush=True)
        if rc:raise RuntimeError("Task failed "+key)
        return receipt
    groups=[tasks[i::len(free)]for i in range(len(free))]
    def worker(group,gpu):return [job(task,gpu)for task in group]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(free))as pool:
        futures=[pool.submit(worker,group,d["gpu"])for group,d in zip(groups,free)]
        for f in concurrent.futures.as_completed(futures):rows.extend(f.result())
    dump(out/"COMPLETED.json",dict(status="COMPLETED",jobs=rows,GPU_hours=sum(x["GPU_hours"]for x in rows)))
    if a.phase=="score":
        banks=[dict(relative=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size)for p in sorted((root/"score").rglob("*.npz"))]
        assert len(banks)==116
        dump(root/"RAW_BANK_SEAL.json",dict(status="COMPLETED",banks=banks,jobs=rows,GPU_hours=sum(x["GPU_hours"]for x in rows),config_sha256=sha(config),no_labels_decoded=True))
    print(json.dumps(dict(event="STAGE_COMPLETED",phase=a.phase,jobs=len(rows))),flush=True)
if __name__=="__main__":main()
