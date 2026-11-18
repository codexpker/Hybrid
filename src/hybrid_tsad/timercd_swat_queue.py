"""Fixed seven-checkpoint SWaT B inference; no optimization or label access."""
import argparse,concurrent.futures,json,os,signal,subprocess,sys,time
from pathlib import Path
from .timercd_fixed import sha
from .timercd_fixed_queue import inventory

def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    with Path(p).open("x")as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--config",type=Path,required=True);p.add_argument("--runtime",type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();cfg=read(a.config);rt=a.runtime.resolve();root=w/cfg["output"];out=root/"score_controller";out.mkdir(parents=True,exist_ok=False)
    assert len(cfg["models"])==7 and cfg["arms"]==["B_fit_only"]
    free=[d for d in inventory()if not d["occupied"]and d["used_MiB"]<1000 and d["free_MiB"]>32000][:4]
    if not free:raise RuntimeError("No free GPU; external jobs never evicted")
    dump(out/"START.json",dict(pid=os.getpid(),start_unix=time.time(),inventory=inventory(),config_sha256=sha(a.config),runtime_files={p.relative_to(rt).as_posix():sha(p)for p in sorted(rt.rglob("*.py"))}))
    def job(spec,gpu):
        for _ in range(60):
            info=next(d for d in inventory()if d["gpu"]==gpu)
            if not info["occupied"]and info["used_MiB"]<1000:break
            time.sleep(1)
        else:raise RuntimeError("Device remained occupied; do not interfere")
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(rt/"src"),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
        cmd=[sys.executable,"-B","-m","hybrid_tsad.timercd_swat_fixed","--workspace",str(w),"--config",str(a.config.resolve()),"--model",spec["id"],"--device","cuda:0"];begin=time.time()
        with(out/(spec["id"]+".log")).open("xb")as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            row=dict(pid=child.pid,arm=spec["id"],model_id=spec["id"],seed=spec["seed"],permission="B_fit_only",gpu=gpu,start_unix=begin,command=cmd,timeout_seconds=3600);dump(out/(spec["id"]+"_START.json"),row);print(json.dumps(dict(event="START",**row)),flush=True)
            try:rc=child.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                rc=124
        end=time.time();row.update(exit_code=rc,end_unix=end,seconds=end-begin,GPU_hours=(end-begin)/3600);dump(out/(spec["id"]+"_EXIT.json"),row);print(json.dumps(dict(event="EXIT",**row)),flush=True)
        if rc:raise RuntimeError("Frozen inference failed "+spec["id"])
        return row
    def worker(specs,gpu):return[job(s,gpu)for s in specs]
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(free))as pool:
        for f in concurrent.futures.as_completed([pool.submit(worker,cfg["models"][i::len(free)],d["gpu"])for i,d in enumerate(free)]):rows.extend(f.result())
    assert len(rows)==7
    dump(out/"COMPLETED.json",dict(status="COMPLETED",jobs=rows,GPU_hours=sum(r["GPU_hours"]for r in rows)))
    banks=[dict(relative=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size)for p in sorted((root/"score").rglob("*.npz"))];assert len(banks)==14
    dump(root/"RAW_BANK_SEAL.json",dict(status="COMPLETED",banks=banks,jobs=rows,GPU_hours=sum(r["GPU_hours"]for r in rows),config_sha256=sha(a.config),data_binding_sha256=cfg["data_binding_sha256"],no_labels_decoded=True,optimizer_updates=0))
    print(json.dumps(dict(event="STAGE_COMPLETED",models=7,banks=14)),flush=True)
if __name__=="__main__":main()
