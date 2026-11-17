"""Bounded synthetic-supervised TimeRCD continuation and frozen public readout."""
import argparse, copy, hashlib, json, os, pickle, random, sys, time, traceback
from pathlib import Path
import numpy as np
from .timercd_fixed import detector_from_config, predict_with_cache, sha
from .timercd_fixed_runner import RoleReader

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def dump(p, x): Path(p).write_text(json.dumps(x, indent=2, allow_nan=False), encoding="utf-8")
def seed(n):
    import torch
    random.seed(n);np.random.seed(n);torch.manual_seed(n)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(n)
def resolve(w,p):
    p=Path(p);return p if p.is_absolute() else w/p

def loss_terms(model, batch):
    """Original CE plus masked MSE; exclude original padding before binarization."""
    import torch
    import torch.nn.functional as F
    emb=model(time_series=batch["time_series"],mask=batch["attention_mask"])
    logits=model.anomaly_head(emb).mean(dim=-2)
    valid=batch["attention_mask"].bool() & (batch["labels"] >= 0)
    ce=F.cross_entropy(logits[valid],(batch["labels"][valid]>.5).long())
    masked=model(time_series=batch["masked_time_series"],mask=batch["attention_mask"])
    pred=model.reconstruction_head(masked).squeeze(-1)
    selected=batch["mask"].bool() & valid
    mse=F.mse_loss(pred[selected],batch["time_series"][selected])
    return ce+mse,ce,mse

def make_batch(samples, ids, device, mask_seed, patch_size):
    import torch
    # Identical standardization definition to the official collate function.
    xs=[torch.as_tensor(samples[i]["time_series"],dtype=torch.float32) for i in ids]
    labels=[torch.as_tensor(samples[i]["labels"],dtype=torch.long) for i in ids]
    joined=torch.cat(xs,dim=0);mean=joined.mean(0,keepdim=True);std=joined.std(0,keepdim=True)+1e-4
    xs=[(x-mean)/std for x in xs]
    x=torch.nn.utils.rnn.pad_sequence(xs,batch_first=True,padding_value=0)
    y=torch.nn.utils.rnn.pad_sequence(labels,batch_first=True,padding_value=-1)
    valid=torch.arange(x.shape[1])[None,:] < torch.tensor([len(a) for a in xs])[:,None]
    gen=torch.Generator().manual_seed(mask_seed);mask=torch.zeros_like(valid)
    for b,a in enumerate(xs):
        npatch=(len(a)+patch_size-1)//patch_size
        for pos in torch.randperm(npatch,generator=gen)[:int(npatch*.15)]:
            lo=int(pos)*patch_size;mask[b,lo:min(lo+patch_size,len(a))]=True
    masked=x.clone();noise=torch.randn(masked[mask].shape,generator=gen)*.1;masked[mask]=noise
    return {k:v.to(device) for k,v in dict(time_series=x,labels=y,attention_mask=valid,masked_time_series=masked,mask=mask).items()}

def get_model(w,cfg,arm,channels,device):
    from .timercd_patch_residual import install_residual
    det=detector_from_config(w,cfg,channels,device)
    install_residual(det._tester.model,mode=arm,hidden=cfg["hidden"])
    return det

def data(w,cfg,out):
    root=resolve(w,cfg["generator_root"])
    sys.path.insert(0,str(root/"src"))
    os.environ["CHATTS_CONFIG_PATH"]=str(root/"config/synthetic.json")
    from generate_dataset import _generate_single_sample_worker
    samples=[];rows=[];n=cfg["train_instances"]+cfg["select_instances"]
    for i in range(n):
        s=cfg["data_seed"]+i
        sample=_generate_single_sample_worker((cfg["source_length"],True,cfg["source_channels"],.5,False,None,True,s))
        if sample is None: raise RuntimeError("Official generator failed instance "+str(i))
        x=np.asarray(sample["time_series"]);y=np.asarray(sample["labels"])
        if x.shape!=(cfg["source_length"],cfg["source_channels"]) or y.shape!=(len(x),) or not np.isfinite(x).all():
            raise ValueError("Generator invalid instance "+str(i))
        samples.append(sample);rows.append(dict(id=i,seed=s,role="FIT" if i<cfg["train_instances"] else "SELECT",points=len(x),anomaly_points=int((y>.5).sum()),sha256=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()))
        if i%100==0:print(json.dumps(dict(event="GENERATING",instances=i)),flush=True)
    path=out/"source.pkl"
    with path.open("xb") as f:pickle.dump(samples,f,protocol=4)
    dump(out/"SOURCE_MANIFEST.json",dict(generator_commit=cfg["generator_commit"],source_sha256=sha(path),generator_files_sha256={p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob("*.py"))},instances=rows,no_target_access=True))

def evaluate_source(model,samples,ids,cfg,device):
    import torch
    model.eval();values=[];weights=[]
    with torch.no_grad():
        for start in range(0,len(ids),cfg["batch_size"]):
            batch=make_batch(samples,ids[start:start+cfg["batch_size"]],device,cfg["mask_seed"]+900000+start,16)
            v=loss_terms(model,batch);values.append([float(a) for a in v]);weights.append(int(batch["attention_mask"].sum()))
    return np.average(values,axis=0,weights=weights).tolist()

def train(w,cfg,out,arm,device,profile=False):
    import torch
    seed(cfg["train_seed"])
    source_root=resolve(w,cfg["output"]);source_manifest=read(source_root/"SOURCE_MANIFEST.json")
    assert sha(source_root/"source.pkl")==source_manifest["source_sha256"]
    with (source_root/"source.pkl").open("rb") as f:samples=pickle.load(f)
    det=get_model(w,cfg,arm,cfg["source_channels"],device);model=det._tester.model
    seed(cfg["train_seed"])
    for p in model.parameters(): p.requires_grad_(True)
    initial={k:v.detach().cpu().clone() for k,v in model.named_parameters()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    fit=list(range(cfg["train_instances"]));select=list(range(cfg["train_instances"],len(samples)))
    assert sha(resolve(w,cfg["plan"]))==cfg["source_plan_sha256"]
    plan=read(resolve(w,cfg["plan"]))
    batches=plan["batches"][:cfg["profile_steps"]] if profile else plan["batches"]
    checks=[];log=[];best=float("inf");best_step=None
    def validation(step):
        nonlocal best,best_step
        joint,ce,mse=evaluate_source(model,samples,select,cfg,device)
        checks.append(dict(step=step,joint=joint,ce=ce,mse=mse))
        if joint<best:
            best=joint;best_step=step
            torch.save(dict(model_state_dict=model.state_dict(),step=step,source_select_joint=joint,arm=arm),out/"selected.pth")
        dump(out/"SELECT.json",checks);model.train()
    if not profile:validation(0)
    model.train();start=time.time()
    for step,ids in enumerate(batches,1):
        batch=make_batch(samples,ids,device,cfg["mask_seed"]+step,16)
        optimizer.zero_grad(set_to_none=True)
        joint,ce,mse=loss_terms(model,batch)
        if not torch.isfinite(joint):raise RuntimeError("Nonfinite training loss")
        joint.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if step in (1,2,len(batches)):
            groups={}
            for name,param in model.named_parameters():
                key="branch" if ".project." in name or "raw_order" in name else "anomaly_head" if name.startswith("anomaly_head") else "reconstruction_head" if name.startswith("reconstruction_head") else "backbone"
                if param.grad is not None:groups[key]=groups.get(key,0.)+float(param.grad.detach().float().square().sum())
            dump(out/("GRAD_step"+str(step)+".json"),{k:v**.5 for k,v in groups.items()})
        optimizer.step()
        if step==1 or step%10==0 or profile:
            torch.cuda.synchronize() if device.type=="cuda" else None
            row=dict(step=step,joint=float(joint),ce=float(ce),mse=float(mse),grad_norm=float(gn),seconds=time.time()-start)
            log.append(row);print(json.dumps(row),flush=True)
            with (out/"TRAIN.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
        if not profile and (step%cfg["validation_every"]==0 or step==len(batches)):validation(step)
    delta={}
    for name,param in model.named_parameters():
        key="branch" if ".project." in name or "raw_order" in name else "anomaly_head" if name.startswith("anomaly_head") else "reconstruction_head" if name.startswith("reconstruction_head") else "backbone"
        delta[key]=delta.get(key,0.)+float((param.detach().cpu()-initial[name]).double().square().sum())
    dump(out/"PARAMETER_DELTA.json",dict(l2={k:v**.5 for k,v in delta.items()},order=float(model.ts_encoder.embedding_layer.order.detach())if arm!="native" else None))
    if not profile:
        torch.save(dict(model_state_dict=model.state_dict(),step=len(batches),arm=arm),out/"last.pth")
    dump(out/("PROFILE.json" if profile else "TRAIN_RECEIPT.json"),dict(status="COMPLETED",arm=arm,updates=len(batches),selected_step=best_step,best_source_joint=None if profile else best,parameters=sum(p.numel()for p in model.parameters()),trainable=sum(p.numel()for p in model.parameters()if p.requires_grad),source_instances=cfg["train_instances"],sample_exposures=sum(map(len,batches)),timepoints_exposed=sum(map(len,batches))*cfg["source_length"],source_plan_sha256=sha(resolve(w,cfg["plan"])),checkpoint_sha256=sha(out/"selected.pth") if not profile else None,all_original_backbone_and_heads_updated=True,no_target_access=True))

def score(w,cfg,out,arm,rid,permission,device):
    import torch
    reader=RoleReader(w,cfg,rid,["CAL","QUERY"]);mean,std=reader.fit_statistics() if permission=="B_fit_only" else (None,None)
    det=get_model(w,cfg,arm,reader.record["channels"],device)
    checkpoint=resolve(w,cfg["output"])/"train"/arm/"selected.pth"
    state=torch.load(checkpoint,map_location=device,weights_only=True)
    det._tester.model.load_state_dict(state["model_state_dict"],strict=True);det._tester.model.eval()
    banks=[]
    for role in ["CAL","QUERY"]:
        x,index=reader.load(role);bank,norm=predict_with_cache(det,x,index,permission,mean,std)
        np.savez_compressed(out/(role+".npz"),**bank)
        banks.append(dict(role=role,path=str(out/(role+".npz")),sha256=sha(out/(role+".npz")),normalization=norm))
    dump(out/"SCORE_RECEIPT.json",dict(status="COMPLETED",arm=arm,permission=permission,record=rid,selected_step=state["step"],checkpoint_sha256=sha(checkpoint),banks=banks,roles=reader.access,score="head abnormal minus normal logit",offline=True,no_labels_decoded=True))

def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["data","profile","train","score"]);p.add_argument("--workspace",type=Path,required=True);p.add_argument("--config",type=Path,required=True);p.add_argument("--arm",choices=["native","time","frft"],default="native");p.add_argument("--record");p.add_argument("--permission",choices=["A_official_role","B_fit_only"],default="B_fit_only");p.add_argument("--device",default="cpu");a=p.parse_args()
    w=a.workspace.resolve();cfg=read(a.config);root=resolve(w,cfg["output"])
    out=root if a.phase=="data" else root/a.phase/a.arm
    if a.phase=="score":out=out/a.record/a.permission
    out.mkdir(parents=True,exist_ok=False);begin=time.time();rc=0;device=torch.device(a.device)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    if device.type=="cuda":
        torch.cuda.set_device(device);torch.cuda.set_per_process_memory_fraction((30 if a.phase=="score" else 20)*2**30/torch.cuda.get_device_properties(device).total_memory,device);torch.cuda.reset_peak_memory_stats(device)
    dump(out/"START.json",dict(pid=os.getpid(),phase=a.phase,arm=a.arm,record=a.record,config_sha256=sha(a.config),runner_sha256=sha(__file__),CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES"),start_unix=begin))
    try:
        if a.phase=="data":data(w,cfg,out)
        elif a.phase in ["train","profile"]:train(w,cfg,out,a.arm,device,a.phase=="profile")
        else:score(w,cfg,out,a.arm,a.record,a.permission,device)
    except BaseException:
        rc=1;dump(out/"FAILED.json",dict(traceback=traceback.format_exc()));raise
    finally:
        end=time.time();dump(out/"OS.json",dict(pid=os.getpid(),exit_code=rc,start_unix=begin,end_unix=end,seconds=end-begin,GPU_hours=(end-begin)/3600 if device.type=="cuda" else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=="cuda" else 0))
if __name__=="__main__":main()
