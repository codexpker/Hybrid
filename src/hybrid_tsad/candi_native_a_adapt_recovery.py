"""Resume only the never-started SWaT U stage from the accepted source model.

Initial SANA randomness is reconstructed from the recorded, deterministic source
loader order, then checked against the original complete source reference bank.
Reference pools and FPM decisions are unchanged; distance computation is query
microbatched with the original full reference dimension.
"""
import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time
import types

import numpy as np

from .candi_native_a import runtime_label_free_selection
from .left_swat_author import read,sha,dump


def query_batched_distance(query,reference,inverse,chunk=4):
    import torch
    result=[]
    for q in query.split(chunk):
        difference=q[:,None,:]-reference[None,:,:]
        result.append(torch.sum((difference@inverse)*difference,dim=-1).min(1).values[:,None])
    return torch.cat(result)


def worker(args):
    import torch
    from torch.utils.data import DataLoader,Dataset
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import chi2
    root=args.output;workspace=args.workspace;task=read(root/"TASK.json");assert task["domain"]=="SWAT"
    assert not(root/"ADAPTED_ENDPOINT.pth").exists()
    original_failure=read(root.parent/"lifecycles/SWAT/OS_WAIT.json");assert original_failure["OS_exit_code"]==1
    sys.path.insert(0,str(workspace/task["source"]["upstream"]))
    from config import get_cfg_defaults
    from models.mlp.modeling_mlp import MLP
    from tta.candi.adapter_candi import MLPAdapter,CANDIAdapter
    from threshold import Thresholder
    cfg=get_cfg_defaults();cfg.merge_from_file(str(root/"CONFIG.yaml"));cfg.RESULT_DIR=str(root/"adapt_recovery");Path(cfg.RESULT_DIR).mkdir()
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    with np.load(root/"SCALER.npz")as z:
        scaler=StandardScaler();scaler.mean_=z["mean"];scaler.scale_=z["scale"];scaler.var_=z["variance"]
        scaler.n_samples_seen_=z["n_samples_seen"];scaler.n_features_in_=len(scaler.mean_)
    source=read(root/"SOURCE_TRAIN_RECEIPT.json");assert sha(root/"SOURCE_SELECTED.pth")==source["source_selected_sha256"]
    state=torch.load(root/"SOURCE_SELECTED.pth",map_location="cpu",weights_only=False)
    # Same actual order as native worker: MLP init, then train base_seed + sampler
    # seed each epoch, with a SELECT base_seed at 5/10/15/20/25/30.
    torch.manual_seed(2025);torch.cuda.manual_seed_all(2025);model=MLP(cfg).cuda();sequence=[]
    for epoch in range(30):
        for kind in ["train_loader_base_seed","train_RandomSampler_seed"]:
            sequence.append(dict(epoch=epoch,kind=kind,value=int(torch.empty((),dtype=torch.int64).random_())))
        if(epoch+1)%5==0:sequence.append(dict(epoch=epoch,kind="SELECT_loader_base_seed",value=int(torch.empty((),dtype=torch.int64).random_())))
    model.load_state_dict(state["model_state"],strict=True);model.eval()
    data=task["data"]["source_train"];p=workspace/data["relative"];assert sha(p)==data["sha256"]
    raw=np.load(p,mmap_mode="r");lo,hi=task["source"]["normal_SELECT"]
    values=scaler.transform(np.asarray(raw[lo:hi],dtype=np.float32));w=10
    val_inputs=torch.from_numpy(np.lib.stride_tricks.sliding_window_view(values,(w,values.shape[1]))[:,0].copy()).cuda()
    cached=torch.from_numpy(np.load(root/"adapt/val_representations.npy")).cuda();assert cached.shape==(len(values)-w+1,128)
    with torch.no_grad():val_scores=np.concatenate([model.get_anomaly_scores(x).cpu().numpy()for x in val_inputs.split(256)])
    original_repr=type(model).get_representations;initializing=[True];repr_checks=[]
    def get_repr(self,x):
        if initializing[0]and len(x)==len(val_inputs):
            for at in [0,((len(x)-1)//256)*256]:
                end=min(at+256,len(x));a=original_repr(self,x[at:end]);b=cached[at:end]
                error=float((a-b).abs().max());assert torch.allclose(a,b,rtol=1e-5,atol=1e-6)
                repr_checks.append(dict(batch_start=at,points=end-at,maxabs=error,bitexact=bool(torch.equal(a,b))))
            return cached
        return original_repr(self,x)
    model.get_representations=types.MethodType(get_repr,model)
    # Use the exact original label-free full formula, but only microbatch query rows.
    full_function,_=runtime_label_free_selection(CANDIAdapter,chunked=False)
    function,patch=runtime_label_free_selection(CANDIAdapter)
    unwrapped=__import__("inspect").unwrap(function)
    unwrapped.__globals__["chunked_mahalanobis"]=query_batched_distance
    CANDIAdapter.get_samples_to_adapt=function
    adapter=MLPAdapter(cfg,model,Thresholder(cfg,None,None,None,val_scores),val_scores=val_scores,val_inputs=val_inputs,test_labels=None)
    initializing[0]=False
    dump(root/"INITIAL_SANA_RECOVERY.json",dict(source_selected_sha256=source["source_selected_sha256"],
        original_reference_sha256=sha(root/"adapt/val_representations.npy"),ordered_RNG_replay=sequence,reference_reconstruction_checks=repr_checks,
        all_reference_rows_reused=True,source_training_updates_repeated=0,
        assumptions=["source MLP has no dropout or other stochastic forward","Adam has no random draws","native source loaders NUM_WORKERS0; base_seed then sampler seed"] ))
    torch.save(dict(sana_in=model.sana_in.state_dict(),sana_out=model.sana_out.state_dict(),CPU_RNG=torch.get_rng_state(),CUDA_RNG=torch.cuda.get_rng_state_all()),root/"INITIAL_SANA_RECOVERY.pth")
    # Quantify original reference-chunk failure and validate actual chi-square decisions.
    def historical_reference_chunk(query,reference,inverse):
        # Diagnostic reproduction only: the original failed engine, never U training.
        best=torch.full((len(query),),float("inf"),dtype=query.dtype,device=query.device)
        for at in range(0,len(reference),256):
            delta=query[:,None,:]-reference[None,at:at+256,:]
            best=torch.minimum(best,((delta@inverse)*delta).sum(-1).min(1).values)
        return best[:,None]
    checks=[];gate_state={k:v.detach().clone()for k,v in model.state_dict().items()}
    gate_cpu=torch.get_rng_state().clone();gate_cuda=[v.clone()for v in torch.cuda.get_rng_state_all()]
    with torch.no_grad():
        for name,ref in [("hard",adapter.topk_representations),("moderate",adapter.moderate_representations)]:
            q=cached[:4];difference=q[:,None,:]-ref[None,:,:]
            full=((difference@adapter.representations_cov_inv)*difference).sum(-1).min(1).values[:,None]
            reference_chunk=historical_reference_chunk(q,ref,adapter.representations_cov_inv)
            proposed=query_batched_distance(q,ref,adapter.representations_cov_inv,chunk=1)
            threshold=float(chi2.ppf(.05,df=128));maskdiff=int(((full<threshold)!=(proposed<threshold)).sum())
            assert maskdiff==0
            checks.append(dict(pool=name,reference_count=len(ref),full=full.cpu().numpy().tolist(),reference_chunk=reference_chunk.cpu().numpy().tolist(),
                query_chunk=proposed.cpu().numpy().tolist(),reference_chunk_maxabs=float((full-reference_chunk).abs().max()),
                query_chunk_maxabs=float((full-proposed).abs().max()),query_acceptance_mask_differences=maskdiff,chi2_threshold=threshold,
                closest_full_boundary=float((full-threshold).abs().min())))
    full_adapter=copy.deepcopy(adapter);bounded_adapter=copy.deepcopy(adapter)
    def observe(fn):
        def selection(self,x,scores):
            fn(self,x,scores)
            self._gate_selected={k:torch.cat(getattr(self,"samples_to_adapt_"+k)).detach().cpu()if getattr(self,"samples_to_adapt_"+k)else torch.empty(0)
                for k in ["hard","moderate"]}
        return selection
    full_adapter.get_samples_to_adapt=types.MethodType(observe(full_function),full_adapter)
    bounded_adapter.get_samples_to_adapt=types.MethodType(observe(function),bounded_adapter)
    gate_x=torch.cat([adapter.topk_inputs[:16],adapter.moderate_inputs[:16]])
    for item in [full_adapter,bounded_adapter]:
        with torch.no_grad():scores=item.model.get_anomaly_scores(gate_x)
        item.adapt(gate_x,scores)
    assert all(torch.equal(full_adapter._gate_selected[k],bounded_adapter._gate_selected[k])for k in ["hard","moderate"])
    assert full_adapter.n_adapt==bounded_adapter.n_adapt
    update_maxabs=max(float((v-bounded_adapter.model.state_dict()[k]).abs().max())for k,v in full_adapter.model.state_dict().items())
    assert all(torch.allclose(v,bounded_adapter.model.state_dict()[k],rtol=1e-5,atol=1e-6)for k,v in full_adapter.model.state_dict().items())
    gate_steps=full_adapter.n_adapt*cfg.TEST.TTA.STEPS
    assert all(torch.equal(v,model.state_dict()[k])for k,v in gate_state.items())
    assert torch.equal(gate_cpu,torch.get_rng_state())and all(torch.equal(a,b)for a,b in zip(gate_cuda,torch.cuda.get_rng_state_all()))
    dump(root/"SWAT_FPM_RECOVERY_CHECK.json",dict(checks=checks,kernel="query<=4, complete reference dimension, native broadcast Mahalanobis",
        selected_actual_inputs_equal=True,optimizer_steps_per_discarded_copy=gate_steps,nonzero_update_equivalence_tested=gate_steps>0,
        update_state_maxabs=update_maxabs,formal_state_RNG_unchanged=True,
        covariance_regularization_added=False,reference_pool_subsampled=False,source_initialization_verified=True))
    del full_adapter,bounded_adapter,gate_state,gate_x
    # Native adaptation starts here for the first time, never from a partly adapted model.
    target=task["data"]["target_test"];p=workspace/target["relative"];assert sha(p)==target["sha256"]
    ulo,uhi=task["roles"]["U_FIT"];raw=np.load(p,mmap_mode="r");u=scaler.transform(np.asarray(raw[ulo:uhi],dtype=np.float32))
    class Windows(Dataset):
        def __len__(self):return len(u)-w+1
        def __getitem__(self,i):return np.array(u[i:i+w],dtype=np.float32)
    loader=DataLoader(Windows(),batch_size=256,shuffle=False,drop_last=False,num_workers=0)
    trace=[];seen=0;steps=[0];old_step=adapter.optimizer.step
    def counted(*a,**kw):
        r=old_step(*a,**kw);steps[0]+=1;return r
    adapter.optimizer.step=counted;started=time.monotonic();model.eval()
    for i,x in enumerate(loader):
        x=x.cuda();before=(adapter.n_adapt,adapter.total_samples_to_adapt_hard,adapter.total_samples_to_adapt_moderate)
        with torch.no_grad():score=model.get_anomaly_scores(x)
        adapter.adapt(x,score);seen+=len(x)
        trace.append(dict(batch=i,windows=len(x),hard_selected=adapter.total_samples_to_adapt_hard-before[1],
            moderate_selected=adapter.total_samples_to_adapt_moderate-before[2],adaptation_groups=adapter.n_adapt-before[0]))
    assert steps[0]==adapter.n_adapt*cfg.TEST.TTA.STEPS
    assert all(torch.equal(v,model.state_dict()[k].cpu())for k,v in state["model_state"].items())
    torch.save(dict(model_state=model.state_dict(),optimizer_state=adapter.optimizer.state_dict(),cfg=cfg.dump(),
        pending_hard=adapter.n_samples_to_adapt_hard,pending_moderate=adapter.n_samples_to_adapt_moderate),root/"ADAPTED_ENDPOINT.pth")
    dump(root/"ADAPT_TRACE.json",trace);dump(root/"ADAPT_RECEIPT.json",dict(endpoint_sha256=sha(root/"ADAPTED_ENDPOINT.pth"),
        U_FIT=[ulo,uhi],windows_seen=seen,expected_windows=uhi-ulo-w+1,chronological_single_pass=True,
        hard_selected=adapter.total_samples_to_adapt_hard,moderate_selected=adapter.total_samples_to_adapt_moderate,
        rejected_windows=seen-adapter.total_samples_to_adapt_hard-adapter.total_samples_to_adapt_moderate,
        adaptation_groups=adapter.n_adapt,optimizer_steps=steps[0],pending_hard=adapter.n_samples_to_adapt_hard,pending_moderate=adapter.n_samples_to_adapt_moderate,
        source_backbone_unchanged=True,labels_read=False,target_CAL_QUERY_used_for_adaptation=False,FPM_reference="complete normal SOURCE_SELECT",
        source_FPM_threshold=float(adapter.thresholder.threshold),adaptation_seconds=time.monotonic()-started,
        initial_SANA_reconstruction_sha256=sha(root/"INITIAL_SANA_RECOVERY.pth"),kernel="query<=4/full-reference native formula",source_training_repeated=False))
    # Preserve the original failed gate; the recovery check supplements it.
    dump(root/"INCREMENT_CHECK.json",dict(full_reference_retained=True,formal_model_and_RNG_unchanged=True,
        original_gate_failure_preserved=True,recovery_check_sha256=sha(root/"SWAT_FPM_RECOVERY_CHECK.json"),
        source_initialization_recovered=True,source_reference_windows=len(val_scores)))
    del adapter,model,cached,val_inputs;torch.cuda.empty_cache()
    from .candi_native_a_readout_recovery import worker as recover_readout
    recover_readout(args)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["worker"]);p.add_argument("--workspace",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    worker(p.parse_args())
