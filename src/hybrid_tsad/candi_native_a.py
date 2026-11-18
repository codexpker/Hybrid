"""Fixed CANDI source-normal training / natural-U adaptation / frozen readout.

Official implementation is imported from an external SHA-bound checkout, not
vendored or relicensed. Runtime changes remove label-only counters and bound
memory for mathematically identical full-reference FPM operations.
"""
import argparse
import ast
import concurrent.futures
import copy
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap
import time
import types

import numpy as np

from .left_swat_author import read,sha,dump

PIN="28c9679e503832f59e351208cde63657fcb51cad"
EXPERIMENT="candi_native_a_20261118_v1"


def score_starts(axis,window=10):
    """One real preceding window for each scored current endpoint; no padding."""
    return np.arange(axis[0]-window+1,axis[1]-window+1,dtype=np.int64)


def chunked_mahalanobis(query,reference,inverse,chunk=4):
    """Bound query rows while preserving the original complete reference axis."""
    import torch
    result=[]
    for q in query.split(chunk):
        difference=q[:,None,:]-reference[None,:,:]
        distances=torch.sum((difference@inverse)*difference,dim=-1)
        result.append(distances.min(dim=1).values[:,None])
    return torch.cat(result)


def runtime_label_free_selection(cls,chunked=True):
    """Keep official hard/moderate/buffering logic; remove only label statistics."""
    original=inspect.unwrap(cls.get_samples_to_adapt)
    source=textwrap.dedent(inspect.getsource(original))
    tree=ast.parse(source);removed=[]
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            code=ast.unparse(node)
            if any(isinstance(t,ast.Name)and t.id.startswith("selected_labels_")for t in node.targets):
                removed.append(code);return None
            if chunked and any(isinstance(t,ast.Name)and t.id.startswith("mahalanobis_dist_square_")for t in node.targets):
                kind=node.targets[0].id.rsplit("_",1)[1]
                replacement=ast.parse(f"{node.targets[0].id} = chunked_mahalanobis(representations, self.{ 'topk' if kind=='hard' else 'moderate'}_representations, self.representations_cov_inv)").body[0]
                return ast.copy_location(replacement,node)
            return self.generic_visit(node)
        def visit_AugAssign(self,node):
            if isinstance(node.target,ast.Attribute)and node.target.attr.startswith("total_anomalies_in_"):
                removed.append(ast.unparse(node));return None
            return self.generic_visit(node)
    tree=Rewrite().visit(tree);ast.fix_missing_locations(tree)
    namespace=dict(original.__globals__,chunked_mahalanobis=chunked_mahalanobis)
    exec(compile(tree,"<CANDI-fixed-label-free-memory-bounded>","exec"),namespace)
    patched=namespace["get_samples_to_adapt"]
    rewritten=ast.unparse(tree)
    if "self.test_labels"in rewritten or not removed:raise AssertionError("Unexpected official selection source")
    return patched,dict(official_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        runtime_function_sha256=hashlib.sha256(rewritten.encode()).hexdigest(),
        removed_label_statistics=removed,reference_pool_subsampled=False,
        preserved=["hard/moderate inequalities","chi2 threshold","full reference pools","native buffering/update"])


def prepare(args):
    workspace=args.workspace;root=args.output;root.mkdir(parents=True,exist_ok=False)
    repo=workspace/"repos/CANDI_20261029_v1";upstream=root/"upstream";upstream.mkdir()
    assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==PIN
    names=subprocess.check_output(["git","-C",str(repo),"ls-tree","-r","--name-only",PIN],text=True).splitlines()
    selected=[n for n in names if (n.endswith(".py")and not n.startswith("TSB-AD/"))or n in ["LICENSE","README.md","scripts/SWaT/SWaT_alpha_5.0.sh"]]
    manifest=[]
    for name in selected:
        raw=subprocess.check_output(["git","-C",str(repo),"show",PIN+":"+name]);p=upstream/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        manifest.append(dict(relative=name,sha256=sha(p),git_blob=subprocess.check_output(["git","-C",str(repo),"rev-parse",PIN+":"+name],text=True).strip()))
    dump(root/"OFFICIAL_SOURCE.json",dict(repository="https://github.com/kimanki/CANDI",commit=PIN,
        files=manifest,license="Modified MIT Non-Commercial with Permission; external import only"))
    for domain in ["SWAT","PSM"]:
        old=workspace/f"results/left_current_u_exposure_20261118_v1/{domain}/TASK.json"
        task=read(old);d=root/domain;d.mkdir()
        old_source=task.pop("source")
        task.update(experiment=EXPERIMENT,seeds=[2025],source=dict(upstream=str(upstream.relative_to(workspace)),commit=PIN,
            normal_FIT=task["roles"]["SOURCE_FIT"],normal_SELECT=[task["roles"]["SOURCE_FIT"][1],old_source["scaler_fit"][1]],
            all_model_parameters_trained=True,normal_selection="minimum native MSE on SOURCE_SELECT at epochs5/10/15/20/25/30",
            optimizer="Adam lr1e-4 wd1e-4; official cosine warmup5, max30epochs",compatible_checkpoint_found=False),
            scoring=dict(window=10,stride=1,batch=256,definition="official whole-window/channel mean MSE mapped to actual window endpoint",
                available_at="current window endpoint",labels="point labels only in separate evaluator; never window-any-positive",padding=False),
            adaptation=dict(kind="full official FPM + input/output SANA",order="one chronological pass through every legal U_FIT window",
                batch=256,drop_last=False,target_labels_read=False,stop="freeze after last U_FIT batch; no CAL/QUERY updates",
                SWAT_preset="official scripts/SWaT/SWaT_alpha_5.0.sh",PSM_preset="official config defaults; no author PSM preset"),
            CANDI=dict(alpha=5.0 if domain=="SWAT"else .5,learning_rate=.03 if domain=="SWAT"else .0001,
                steps=5 if domain=="SWAT"else 1,gating_init=.5 if domain=="SWAT"else 0.,minimum_samples=16))
        for stale in ["arms","training","sampler","exposure","score_window_axes","source_asset_policy"]:task.pop(stale,None)
        for role,axis in task["score_axes"].items():
            starts=score_starts(axis);lo,hi=task["roles"][role]
            assert starts[0]>=lo and starts[-1]+10<=hi
        dump(d/"TASK.json",task)
        dump(d/"FROZEN.json",dict(task_sha256=sha(d/"TASK.json"),source_manifest_sha256=sha(root/"OFFICIAL_SOURCE.json"),
            worker_sha256=sha(Path(__file__)),seed=2025,window=10,stride=1))


def worker(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset,DataLoader
    from sklearn.preprocessing import StandardScaler
    began=time.monotonic();workspace=args.workspace;root=args.output;task=read(args.task)
    frozen=read(root/"FROZEN.json")
    assert sha(args.task)==frozen["task_sha256"]and sha(Path(__file__))==frozen["worker_sha256"]
    upstream=workspace/task["source"]["upstream"];manifest=read(root.parent/"OFFICIAL_SOURCE.json")
    assert sha(root.parent/"OFFICIAL_SOURCE.json")==frozen["source_manifest_sha256"]
    assert manifest["commit"]==PIN
    for spec in manifest["files"]:assert sha(upstream/spec["relative"])==spec["sha256"]
    sys.path.insert(0,str(upstream))
    from config import get_cfg_defaults
    from models.mlp.modeling_mlp import MLP
    from models.optimizer import construct_optimizer,get_epoch_lr,set_lr
    from tta.candi.adapter_candi import CANDIAdapter,MLPAdapter
    from threshold import Thresholder
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats();random.seed(2025);np.random.seed(2025);torch.manual_seed(2025);torch.cuda.manual_seed_all(2025)
    cfg=get_cfg_defaults();cfg.SEED=2025;cfg.DATA.N_VAR=task["data"]["source_train"]["shape"][1];cfg.DATA.NAME=task["domain"]
    cfg.RESULT_DIR=str(root/"adapt");(root/"adapt").mkdir();cfg.WANDB.ENABLE=False
    cfg.TRAIN.ENABLE=True;cfg.TRAIN.CHECKPOINT_DIR=str(root)
    cfg.TEST.THRESHOLD.ANOMALY_RATIO=task["CANDI"]["alpha"];cfg.TEST.TTA.SOLVER.BASE_LR=task["CANDI"]["learning_rate"]
    cfg.TEST.TTA.STEPS=task["CANDI"]["steps"];cfg.TEST.TTA.CANDI.GATING_INIT=task["CANDI"]["gating_init"]
    (root/"CONFIG.yaml").write_text(cfg.dump());w=cfg.DATA.WIN_SIZE;assert w==10
    data=task["data"]["source_train"];source_path=workspace/data["relative"];assert sha(source_path)==data["sha256"]
    raw=np.load(source_path,mmap_mode="r");fitlo,fithi=task["source"]["normal_FIT"];vallo,valhi=task["source"]["normal_SELECT"]
    source_fit=np.asarray(raw[fitlo:fithi],dtype=np.float32);source_select=np.asarray(raw[vallo:valhi],dtype=np.float32)
    assert np.isfinite(source_fit).all()and np.isfinite(source_select).all()
    scaler=StandardScaler().fit(source_fit);fit=scaler.transform(source_fit);select=scaler.transform(source_select)
    np.savez_compressed(root/"SCALER.npz",mean=scaler.mean_,scale=scaler.scale_,variance=scaler.var_,n_samples_seen=scaler.n_samples_seen_)
    class Windows(Dataset):
        def __init__(self,values,starts):self.values=values;self.starts=np.asarray(starts)
        def __len__(self):return len(self.starts)
        def __getitem__(self,i):return np.array(self.values[self.starts[i]:self.starts[i]+w],dtype=np.float32)
    train_loader=DataLoader(Windows(fit,np.arange(len(fit)-w+1)),batch_size=256,shuffle=True,drop_last=True,num_workers=0)
    select_loader=DataLoader(Windows(select,np.arange(len(select)-w+1)),batch_size=256,shuffle=False,drop_last=False,num_workers=0)
    model=MLP(cfg).cuda();model.eval()
    smoke=torch.from_numpy(np.stack([fit[i:i+w]for i in range(4)])).cuda()
    with torch.no_grad():
        native=model.get_anomaly_scores(smoke);direct=F.mse_loss(model(smoke),smoke,reduction="none").mean((1,2))
    assert torch.equal(native,direct)and torch.isfinite(native).all()
    dump(root/"SOURCE_ROLE_SCORE_SMOKE.json",dict(native_whole_window_MSE_bitexact=True,maximum_absolute_difference=float((native-direct).abs().max()),
        actual_source_window_start=[fitlo+i for i in range(4)],actual_source_score_endpoint=[fitlo+i+w-1 for i in range(4)],
        normal_FIT=[fitlo,fithi],normal_SELECT=[vallo,valhi],scaler_fit_only=[fitlo,fithi],
        training_target_U_CAL_QUERY_values_read=False,labels_read=False))
    optimizer=construct_optimizer(model,cfg);best=float("inf");history=[];updates=0;training_started=time.monotonic()
    for epoch in range(cfg.SOLVER.MAX_EPOCH):
        model.train();total=0.;count=0
        for i,x in enumerate(train_loader):
            set_lr(optimizer,get_epoch_lr(epoch+(i+1)/len(train_loader),cfg));x=x.cuda();loss=F.mse_loss(model(x),x)
            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step();updates+=1;total+=float(loss)*len(x);count+=len(x)
        row=dict(epoch=epoch,train_MSE=total/count,updates=updates)
        if (epoch+1)%cfg.TRAIN.EVAL_PERIOD==0 or epoch+1==cfg.SOLVER.MAX_EPOCH:
            model.eval();vt=0.;vn=0
            with torch.no_grad():
                for x in select_loader:
                    x=x.cuda();value=F.mse_loss(model(x),x);vt+=float(value)*len(x);vn+=len(x)
            row["SELECT_MSE"]=vt/vn
            if row["SELECT_MSE"]<best:
                best=row["SELECT_MSE"];torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),epoch=epoch,updates=updates,cfg=cfg.dump()),root/"SOURCE_SELECTED.pth")
        history.append(row);print(json.dumps(dict(event="SOURCE_EPOCH",domain=task["domain"],**row)),flush=True)
    dump(root/"SOURCE_TRAIN_TRACE.json",history)
    selected=torch.load(root/"SOURCE_SELECTED.pth",map_location="cpu",weights_only=False);model.load_state_dict(selected["model_state"],strict=True);model.eval()
    native_model=copy.deepcopy(model)
    dump(root/"SOURCE_TRAIN_RECEIPT.json",dict(source_selected_sha256=sha(root/"SOURCE_SELECTED.pth"),selected_epoch_zero_based=selected["epoch"],
        selected_update=selected["updates"],total_updates=updates,epochs=cfg.SOLVER.MAX_EPOCH,source_FIT=[fitlo,fithi],source_SELECT=[vallo,valhi],
        scaler_sha256=sha(root/"SCALER.npz"),source_data_sha256=sha(source_path),source_only=True,labels_read=False,
        validation_windows=len(select)-w+1,training_seconds=time.monotonic()-training_started))
    def states():return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in model.state_dict().items()}
    def rng():return dict(cpu=sha_bytes(torch.get_rng_state().numpy().tobytes()),cuda=[sha_bytes(t.cpu().numpy().tobytes())for t in torch.cuda.get_rng_state_all()])
    def readout(name):
        folder=root/name;folder.mkdir(exist_ok=True);before=states();rr=rng();banks=[];steps_before=actual_steps[0]
        target=task["data"]["target_test"];p=workspace/target["relative"];assert sha(p)==target["sha256"]
        raw_target=np.load(p,mmap_mode="r")
        for role,axis in task["score_axes"].items():
            starts=score_starts(axis,w);input_lo=int(starts[0]);input_hi=int(starts[-1]+w)
            # CAL and QUERY are separately materialized; no concatenated context.
            values=scaler.transform(np.asarray(raw_target[input_lo:input_hi],dtype=np.float32))
            dl=DataLoader(Windows(values,starts-input_lo),batch_size=256,shuffle=False,drop_last=False,num_workers=0,
                generator=torch.Generator().manual_seed(2025))
            out=[]
            with torch.no_grad():
                for x in dl:out.append(model.get_anomaly_scores(x.cuda()).cpu().numpy())
            scores=np.concatenate(out);assert len(scores)==axis[1]-axis[0]and np.isfinite(scores).all()
            index=np.arange(*axis);p=folder/(role+".npz")
            np.savez_compressed(p,canonical_score=scores,absolute_time_index=index,available_index=index,
                window_start=starts,window_end_exclusive=starts+w)
            banks.append(dict(relative=str(p.relative_to(root)),sha256=sha(p),role=role,axis=axis))
        assert before==states()and rr==rng()and steps_before==actual_steps[0]
        dump(folder/"READOUT_RECEIPT.json",dict(banks=banks,state_RNG_immutable=True,labels_read=False,adapter_called=False,
            optimizer_update_calls=0,
            source_selected_sha256=sha(root/"SOURCE_SELECTED.pth"),scaler_sha256=sha(root/"SCALER.npz"),score="official whole-window MSE; actual endpoint"))
        return banks
    # Source SELECT supplies the complete FPM reference; it is never target CAL.
    val_inputs=torch.from_numpy(np.lib.stride_tricks.sliding_window_view(select,(w,select.shape[1]))[:,0].copy()).cuda()
    assert tuple(val_inputs.shape)==(len(select)-w+1,w,select.shape[1])
    with torch.no_grad():
        val_scores=np.concatenate([model.get_anomaly_scores(v).cpu().numpy()for v in val_inputs.split(256)])
    thresholder=Thresholder(cfg,None,None,None,val_scores)
    original_repr=type(model).get_representations
    def bounded_repr(self,x):
        with torch.no_grad():return torch.cat([original_repr(self,part)for part in x.split(256)],dim=0)
    model.get_representations=types.MethodType(bounded_repr,model)
    full_function,_=runtime_label_free_selection(CANDIAdapter,chunked=False)
    patched,patch=runtime_label_free_selection(CANDIAdapter);CANDIAdapter.get_samples_to_adapt=patched
    dump(root/"RUNTIME_PATCH.json",patch)
    backbone_before=states();adapter=MLPAdapter(cfg,model,thresholder,val_scores=val_scores,val_inputs=val_inputs,test_labels=None)
    # Incremental bounded-memory formula check; it does not inspect target labels.
    q=adapter.val_representations[:4];inv=adapter.representations_cov_inv;checks=[]
    with torch.no_grad():
        from scipy.stats import chi2
        for name,ref in [("hard",adapter.topk_representations),("moderate",adapter.moderate_representations)]:
            delta=q[:,None,:]-ref[None,:,:];full=((delta@inv)*delta).sum(-1).min(1).values[:,None]
            bounded=chunked_mahalanobis(q,ref,inv)
            assert torch.allclose(full,bounded,rtol=1e-5,atol=1e-5)
            assert torch.equal(full<chi2.ppf(.05,df=128),bounded<chi2.ppf(.05,df=128))
            checks.append(dict(pool=name,full_reference_count=len(ref),maxabs=float((full-bounded).abs().max()),masks_equal=True))
        a=original_repr(model,val_inputs[:4]);b=torch.cat([original_repr(model,x)for x in val_inputs[:4].split(2)])
        assert torch.allclose(a,b,rtol=1e-5,atol=1e-6)
    gate_before=states();gate_rng=rng();gate_start=time.monotonic()
    full_adapter=copy.deepcopy(adapter);bounded_adapter=copy.deepcopy(adapter)
    full_adapter.get_samples_to_adapt=types.MethodType(full_function,full_adapter)
    gate_x=torch.cat([adapter.topk_inputs[:16],adapter.moderate_inputs[:16]])
    for item in [full_adapter,bounded_adapter]:
        with torch.no_grad():gate_scores=item.model.get_anomaly_scores(gate_x)
        item.adapt(gate_x,gate_scores)
    assert full_adapter.total_samples_to_adapt_hard==bounded_adapter.total_samples_to_adapt_hard
    assert full_adapter.total_samples_to_adapt_moderate==bounded_adapter.total_samples_to_adapt_moderate
    assert full_adapter.n_adapt==bounded_adapter.n_adapt
    update_diff=max(float((v-bounded_adapter.model.state_dict()[k]).abs().max())for k,v in full_adapter.model.state_dict().items())
    assert all(torch.allclose(v,bounded_adapter.model.state_dict()[k],rtol=1e-5,atol=1e-6)for k,v in full_adapter.model.state_dict().items())
    gate_steps=full_adapter.n_adapt*cfg.TEST.TTA.STEPS
    assert states()==gate_before and rng()==gate_rng
    del full_adapter,bounded_adapter,delta,full,bounded
    dump(root/"INCREMENT_CHECK.json",dict(full_pool_checks=checks,repr_batch_maxabs=float((a-b).abs().max()),
        full_vs_chunk_selected_and_update_groups_equal=True,comparison_optimizer_steps_per_discarded_copy=gate_steps,
        nonzero_optimizer_update_equivalence_tested=gate_steps>0,gate_input="first16hard+first16moderate normal SOURCE_SELECT references; discarded copies only",
        matched_update_state_maxabs=update_diff,formal_model_and_RNG_unchanged=True,engineering_gate_seconds=time.monotonic()-gate_start,
        source_reference_windows=len(val_scores),hard_references=len(adapter.topk_representations),moderate_references=len(adapter.moderate_representations),
        full_reference_retained=True,labels_unavailable=True))
    actual_steps=[0];original_step=adapter.optimizer.step
    def observed_step(*a,**kw):
        result=original_step(*a,**kw);actual_steps[0]+=1;return result
    adapter.optimizer.step=observed_step
    target=task["data"]["target_test"];p=workspace/target["relative"];assert sha(p)==target["sha256"]
    ulo,uhi=task["roles"]["U_FIT"];u_raw=np.load(p,mmap_mode="r");u=scaler.transform(np.asarray(u_raw[ulo:uhi],dtype=np.float32))
    dl=DataLoader(Windows(u,np.arange(len(u)-w+1)),batch_size=256,shuffle=False,drop_last=False,num_workers=0)
    trace=[];seen=0;adapt_start=time.monotonic();model.eval()
    for batch_index,x in enumerate(dl):
        x=x.cuda();n_before=adapter.n_adapt;hard=adapter.total_samples_to_adapt_hard;moderate=adapter.total_samples_to_adapt_moderate
        with torch.no_grad():scores=model.get_anomaly_scores(x)
        adapter.adapt(x,scores);seen+=len(x)
        trace.append(dict(batch=batch_index,windows=len(x),hard_selected=adapter.total_samples_to_adapt_hard-hard,
            moderate_selected=adapter.total_samples_to_adapt_moderate-moderate,adaptation_groups=adapter.n_adapt-n_before))
    after=states();assert all(after[k]==v for k,v in backbone_before.items())
    assert actual_steps[0]==adapter.n_adapt*cfg.TEST.TTA.STEPS
    assert all(torch.isfinite(v).all()for v in model.state_dict().values())
    torch.save(dict(model_state=model.state_dict(),optimizer_state=adapter.optimizer.state_dict(),cfg=cfg.dump(),
        pending_hard=adapter.n_samples_to_adapt_hard,pending_moderate=adapter.n_samples_to_adapt_moderate),root/"ADAPTED_ENDPOINT.pth")
    dump(root/"ADAPT_TRACE.json",trace);dump(root/"ADAPT_RECEIPT.json",dict(endpoint_sha256=sha(root/"ADAPTED_ENDPOINT.pth"),
        U_FIT=[ulo,uhi],windows_seen=seen,expected_windows=uhi-ulo-w+1,chronological_single_pass=True,
        hard_selected=adapter.total_samples_to_adapt_hard,moderate_selected=adapter.total_samples_to_adapt_moderate,
        rejected_windows=seen-adapter.total_samples_to_adapt_hard-adapter.total_samples_to_adapt_moderate,
        adaptation_groups=adapter.n_adapt,optimizer_steps=actual_steps[0],
        pending_hard=adapter.n_samples_to_adapt_hard,pending_moderate=adapter.n_samples_to_adapt_moderate,
        source_backbone_unchanged=True,labels_read=False,target_CAL_QUERY_used_for_adaptation=False,
        FPM_reference="complete normal SOURCE_SELECT",source_FPM_threshold=float(thresholder.threshold),
        adaptation_seconds=time.monotonic()-adapt_start))
    model.eval();adapt_banks=readout("CANDI");adapted_model=model
    model=native_model;model.eval();noU_banks=readout("noU");model=adapted_model
    dump(root/"RAW_BANK_SEAL.json",dict(status="COMPLETED",banks=noU_banks+adapt_banks,labels_read=False,
        source_selected_sha256=sha(root/"SOURCE_SELECTED.pth"),adapted_endpoint_sha256=sha(root/"ADAPTED_ENDPOINT.pth"),
        scaler_sha256=sha(root/"SCALER.npz"),task_sha256=sha(args.task),runtime_patch_sha256=sha(root/"RUNTIME_PATCH.json"),
        total_seconds=time.monotonic()-began,allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved()))
    print(json.dumps(dict(event="STAGE_COMPLETED",domain=task["domain"],banks=4,windows_adapted=seen)),flush=True)


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()


def queue(args):
    root=args.output;workspace=args.workspace
    sys.path.insert(0,str(workspace/"preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    devices={int(line.split(",")[0]):line.split(",")[1].strip()for line in subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True).splitlines()}
    def job(domain,gpu):
        folder=root/domain;life=root/"lifecycles"/domain;life.mkdir(parents=True,exist_ok=False)
        start=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();child=None;code=None;cleanup=None;error=None
        try:
            dump(life/"GPU_GUARD.json",gpu_guard(devices[gpu].removeprefix("GPU-"),gpu,20))
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",WANDB_MODE="disabled")
            argv=[sys.executable,"-B","-m","hybrid_tsad.candi_native_a","worker","--workspace",str(workspace),"--output",str(folder),"--task",str(folder/"TASK.json")]
            with(life/"stdout.log").open("xb")as out,(life/"stderr.log").open("xb")as err:
                child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);owned_session.claim_group(child)
                dump(life/"STARTED.json",dict(pid=child.pid,GPU=gpu,UTC=utc,argv=argv,capGiB=20,timeout_seconds=7200))
                print(json.dumps(dict(event="RUNNING",domain=domain,pid=child.pid,GPU=gpu)),flush=True)
                try:code=child.wait(timeout=7190)
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
        banks=[]
        for domain in ["SWAT","PSM"]:
            d=root/domain;s=read(d/"RAW_BANK_SEAL.json")
            for b in s["banks"]:
                assert sha(d/b["relative"])==b["sha256"];banks.append(dict(b,relative=domain+"/"+b["relative"],domain=domain))
        dump(root/"GLOBAL_RAW_BANK_SEAL.json",dict(status="COMPLETED",jobs=jobs,banks=banks,
            actual_GPU_hours=sum(j["actual_GPU_seconds"]for j in jobs)/3600,labels_read=False))
    except BaseException as exc:dump(root/"FAILED.json",dict(error=repr(exc)));raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["prepare","worker","queue"])
    parser.add_argument("--workspace",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--task",type=Path);parser.add_argument("--gpus",type=int,nargs="+",default=[0,1]);args=parser.parse_args()
    {"prepare":prepare,"worker":worker,"queue":queue}[args.phase](args)
