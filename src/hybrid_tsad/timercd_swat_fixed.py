"""Seven already frozen native TimeRCD weights on the existing SWaT31 roles.

No optimizer, training, threshold fitting, labels, channel selection or model
change. Native5000 grid is anchored before the established512-point axis crop.
"""
import argparse,json,os,random,time,traceback
from pathlib import Path
import numpy as np
from .timercd_fixed import sha,detector_from_config,predict_with_cache
from .timercd_fixed_runner import state_hash

def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def resolve(w,p):
    p=Path(p);return p if p.is_absolute()else w/p
def dump(p,x):
    with Path(p).open("x",encoding="utf-8")as f:json.dump(x,f,indent=2,allow_nan=False)

class SwatReader:
    def __init__(self,w,cfg,allowed=("CAL","QUERY")):
        self.w=Path(w);self.allowed=set(allowed);self.access=[]
        p=resolve(w,cfg["data_binding"])
        if sha(p)!=cfg["data_binding_sha256"]:raise ValueError("SWaT binding identity changed")
        self.binding=read(p)
        if self.binding["channels_count"]!=31 or len(self.binding["channels"])!=31:raise ValueError("SWaT31 identity changed")
    def load(self,role):
        if role not in self.allowed:raise PermissionError("Role forbidden before file access: "+role)
        e=next(r for r in self.binding["roles"]if r["role"]==role);p=resolve(self.w,e["path"])
        if sha(p)!=e["sha256"]:raise ValueError("Original SWaT role identity changed")
        with np.load(p,allow_pickle=False)as z:
            if set(z.files)!={"values","observed"}:raise ValueError("Original numerical-only role schema changed")
            values=z["values"].copy();observed=z["observed"].copy()
        if values.dtype!=np.float64 or list(values.shape)!=e["shape"]or values.shape[1]!=31 or not observed.all()or not np.isfinite(values).all():raise ValueError("Invalid raw SWaT numerical role")
        index=np.arange(e["original_start"],e["end"],dtype=np.int64)
        if len(index)!=len(values):raise ValueError("Role time index mismatch")
        self.access.append(dict(role=role,path=e["path"],sha256=e["sha256"],fields=["values","observed"],original_start=e["original_start"],end=e["end"]))
        return values,index,e
    def statistics(self):
        e=self.binding["fit_statistics"];p=resolve(self.w,e["path"])
        if sha(p)!=e["sha256"]:raise ValueError("Frozen FIT statistics identity changed")
        with np.load(p,allow_pickle=False)as z:mean=z["mean"].copy();std=z["official_safe_std"].copy()
        if mean.shape!=(31,)or std.shape!=(31,)or not np.isfinite(mean).all()or not np.isfinite(std).all()or np.any(std<=0):raise ValueError("Invalid FIT statistics")
        self.access.append(dict(role="FIT_STATISTICS",path=e["path"],sha256=e["sha256"],fields=["mean","official_safe_std"]))
        return mean,std

def score(w,cfg,out,model_id,device):
    import torch
    spec=next(m for m in cfg["models"]if m["id"]==model_id)
    pt=resolve(w,spec["checkpoint"])
    if sha(pt)!=spec["checkpoint_sha256"]:raise ValueError("Frozen selected checkpoint mismatch")
    reader=SwatReader(w,cfg);mean,std=reader.statistics()
    det=detector_from_config(w,cfg,31,device);model=det._tester.model
    if model_id!="official_zero":
        state=torch.load(pt,map_location=device,weights_only=True)
        if state["step"]!=spec["selected_step"]:raise ValueError("Selected step identity changed")
        model.load_state_dict(state["model_state_dict"],strict=True)
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    before=state_hash(model);banks=[]
    for role in ["CAL","QUERY"]:
        values,index,e=reader.load(role)
        bank,norm=predict_with_cache(det,values,index,"B_fit_only",mean,std)
        bank["eligible_scored"]=index>=e["score_start"]
        if role=="CAL":
            selection=reader.binding["cal_selection"];p=resolve(w,selection["path"])
            if sha(p)!=selection["sha256"]:raise ValueError("Original512 CAL positions changed")
            with np.load(p,allow_pickle=False)as z:chosen=z["index"].copy()
            bank["original_cal_selected"]=np.isin(index,chosen)
            if int(bank["original_cal_selected"].sum())!=512:raise ValueError("Original CAL positions missing")
        path=out/(role+".npz");np.savez_compressed(path,**bank)
        banks.append(dict(role=role,path=str(path),sha256=sha(path),points=len(index),eligible_points=int(bank["eligible_scored"].sum()),padding_points=int(bank["padding_points"]),normalization=norm))
    if state_hash(model)!=before:raise RuntimeError("Frozen parameters or buffers changed during inference")
    dump(out/"SCORE_RECEIPT.json",dict(status="COMPLETED",arm=model_id,model_id=model_id,permission="B_fit_only",record="SWAT",seed=spec["seed"],training_arm=spec["training_arm"],selected_step=spec["selected_step"],checkpoint_sha256=spec["checkpoint_sha256"],data_binding_sha256=cfg["data_binding_sha256"],banks=banks,roles=reader.access,score="abnormal minus normal raw class logit",window=5000,batch_size=1,parameters=sum(p.numel()for p in model.parameters()),optimizer_updates=0,frozen_parameters_and_buffers_unchanged=True,offline=True,no_labels_decoded=True,no_threshold_fitting=True,normalization_count=1,query_axis=cfg["query_axis"],cal_rule=cfg["cal_rule"]))

def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True);p.add_argument("--config",type=Path,required=True);p.add_argument("--model",required=True);p.add_argument("--device",default="cpu");a=p.parse_args();w=a.workspace.resolve();cfg=read(a.config)
    out=resolve(w,cfg["output"])/"score"/a.model;out.mkdir(parents=True,exist_ok=False);begin=time.time();rc=0;device=torch.device(a.device)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);random.seed(cfg["evaluation_seed"]);np.random.seed(cfg["evaluation_seed"]);torch.manual_seed(cfg["evaluation_seed"])
    if device.type=="cuda":
        torch.cuda.set_device(device);torch.cuda.manual_seed_all(cfg["evaluation_seed"]);torch.cuda.set_per_process_memory_fraction(30*2**30/torch.cuda.get_device_properties(device).total_memory,device);torch.cuda.reset_peak_memory_stats(device)
    dump(out/"START.json",dict(pid=os.getpid(),model_id=a.model,config_sha256=sha(a.config),runner_sha256=sha(__file__),CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),start_unix=begin))
    try:score(w,cfg,out,a.model,device)
    except BaseException:
        rc=1;dump(out/"FAILED.json",dict(traceback=traceback.format_exc()));raise
    finally:
        end=time.time();dump(out/"OS.json",dict(pid=os.getpid(),exit_code=rc,start_unix=begin,end_unix=end,seconds=end-begin,GPU_hours=(end-begin)/3600 if device.type=="cuda"else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=="cuda"else 0))
if __name__=="__main__":main()
