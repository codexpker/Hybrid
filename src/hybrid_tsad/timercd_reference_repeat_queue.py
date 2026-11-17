"""Two preregistered training seeds; immutable source reuse, matched arms, B only."""
import argparse,concurrent.futures,json,os,signal,subprocess,sys,time
from pathlib import Path
from .timercd_fixed import sha
from .timercd_fixed_queue import inventory

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    with Path(p).open("x")as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["train","score"]);p.add_argument("--workspace",type=Path,required=True);p.add_argument("--runtime",type=Path,required=True);p.add_argument("--configs",nargs=2,type=Path,required=True);a=p.parse_args()
    w=a.workspace.resolve();rt=a.runtime.resolve();configs=[p.resolve()for p in a.configs];cfgs=[read(p)for p in configs]
    assert [c["train_seed"]for c in cfgs]==[2026,2027]
    assert len({c["source_asset_sha256"]for c in cfgs})==1
    root=w/"results/timercd_reference_repeat_20261117_v1";root.mkdir(exist_ok=True)
    out=root/(a.phase+"_controller");out.mkdir(exist_ok=False)
    free=[d for d in inventory()if not d["occupied"]and d["used_MiB"]<1000 and d["free_MiB"]>32000][:4]
    if not free:raise RuntimeError("No free GPUs; never evict external processes")
    dump(out/"START.json",dict(pid=os.getpid(),start_unix=time.time(),inventory=inventory(),configs_sha256={str(p):sha(p)for p in configs},runtime_files={str(p.relative_to(rt)):sha(p)for p in sorted(rt.rglob("*.py"))}))
    tasks=[]
    for ci,cfg in enumerate(cfgs):
        for arm in cfg["arms"]:
            if a.phase=="train":tasks.append((ci,arm,None))
            else:
                assert read(w/cfg["output"]/"train"/arm/"TRAIN_RECEIPT.json")["status"]=="COMPLETED"
                tasks.extend((ci,arm,rid)for rid in cfg["records"])
    def job(task,gpu):
        ci,arm,rid=task;cfg=cfgs[ci]
        for attempt in range(60):
            info=next(x for x in inventory()if x["gpu"]==gpu)
            if not info["occupied"]and info["used_MiB"]<1000:break
            time.sleep(1)
        else:raise RuntimeError("GPU remained occupied; external job left intact")
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(rt/"src"),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
        cmd=[sys.executable,"-B","-m","hybrid_tsad.timercd_reference_runner",a.phase,"--workspace",str(w),"--config",str(configs[ci]),"--arm",arm,"--device","cuda:0"]
        if rid:cmd.extend(["--record",rid,"--permission","B_fit_only"])
        key=str(cfg["train_seed"])+"__"+arm+("__"+rid if rid else "");start=time.time();limit=7200 if a.phase=="train"else 3600
        with(out/(key+".log")).open("xb")as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            row=dict(pid=child.pid,seed=cfg["train_seed"],arm=arm,record=rid,permission="B_fit_only"if rid else None,gpu=gpu,start_unix=start,command=cmd,timeout_seconds=limit)
            dump(out/(key+"_START.json"),row);print(json.dumps(dict(event="START",**row)),flush=True)
            try:rc=child.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                rc=124
        end=time.time();row.update(exit_code=rc,end_unix=end,seconds=end-start,GPU_hours=(end-start)/3600);dump(out/(key+"_EXIT.json"),row);print(json.dumps(dict(event="EXIT",**row)),flush=True)
        if rc:raise RuntimeError("Fixed task failed "+key)
        return row
    groups=[tasks[i::len(free)]for i in range(len(free))]
    def worker(group,gpu):return[job(t,gpu)for t in group]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(free))as pool:
        for f in concurrent.futures.as_completed([pool.submit(worker,g,d["gpu"])for g,d in zip(groups,free)]):rows.extend(f.result())
    dump(out/"COMPLETED.json",dict(status="COMPLETED",jobs=rows,GPU_hours=sum(r["GPU_hours"]for r in rows)))
    for cfg,cp in zip(cfgs,configs):
        seedroot=w/cfg["output"];sub=seedroot/(a.phase+"_controller");sub.mkdir(exist_ok=False);owned=[r for r in rows if r["seed"]==cfg["train_seed"]]
        dump(sub/"COMPLETED.json",dict(status="COMPLETED",jobs=owned,GPU_hours=sum(r["GPU_hours"]for r in owned),global_controller=str(out)))
        if a.phase=="score":
            banks=[dict(relative=p.relative_to(seedroot).as_posix(),sha256=sha(p),bytes=p.stat().st_size)for p in sorted((seedroot/"score").rglob("*.npz"))];assert len(banks)==116 and len(owned)==58
            dump(seedroot/"RAW_BANK_SEAL.json",dict(status="COMPLETED",banks=banks,jobs=owned,GPU_hours=sum(r["GPU_hours"]for r in owned),config_sha256=sha(cp),no_labels_decoded=True))
    print(json.dumps(dict(event="STAGE_COMPLETED",phase=a.phase,jobs=len(rows))),flush=True)
if __name__=="__main__":main()
