"""One endpoint-only native U update from the author-protocol SWaT checkpoint.

Training opens only numerical U-FIT. CAL/QUERY are opened after the endpoint
is sealed; no labels, donor metadata, validation or checkpoint selection occur.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from hybrid_tsad.left_swat_author import sha, read, dump


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    workspace,root=args.workspace,args.output
    root.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    oldprep=workspace/"preparations/left_multievent_mixed_u_20261111_v1"
    oldpath=oldprep/"ordinary_u_worker.py"
    assert sha(oldpath)=="210fc350e433f4febe12744100b992bf4bbfb8e081a48c00c1501d415b476b47"
    spec=importlib.util.spec_from_file_location("original_native_helpers",oldpath)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    torch=old.base.setup_torch(20)
    old.base.seed(torch)
    author=workspace/"results/left_swat_author_20261118_v1"
    source=read(author/"TRAIN_RECEIPT.json")
    checkpoint=author/"model/SWAT/Left/checkpoints_0/checkpoint.pth"
    expected="064747db9b1c24ce10d8678b00b2f72e504b4bcb4bec32b93399747234c5630d"
    assert sha(checkpoint)==source["checkpoint_sha256"]==expected
    assert source["final_python_step"]==52581
    assert sha(oldprep/"configs/SWAT/model_configs_Left.json")==sha(workspace/"preparations/left_swat_author_20261118_v1/configs/SWAT/model_configs_Left.json")
    model=old.base.native(torch).cuda()
    loaded=model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=False)
    assert not loaded.missing_keys and all(k.endswith(("total_ops","total_params"))for k in loaded.unexpected_keys)
    model.step=source["final_python_step"];model.train()
    assert all(p.requires_grad for p in model.parameters())
    original_hash=old.hashes(model)
    scalar_file=author/"SCALER.npz"
    with np.load(scalar_file)as z:mean=z["mean"].copy();scale=z["scale"].copy()
    assert mean.shape==scale.shape==(31,)and np.all(scale>0)
    dump(root/"SOURCE_BINDING.json",dict(author_TRAIN_RECEIPT_sha256=sha(author/"TRAIN_RECEIPT.json"),
         author_SOURCE_BINDING_sha256=sha(author/"SOURCE_BINDING.json"),source_checkpoint_sha256=expected,
         source_final_python_step=52581,scaler_sha256=sha(scalar_file),worker_sha256=sha(__file__),
         accepted_raw_values_sha256=read(author/"SOURCE_BINDING.json")["ALL_VALUES_sha256"],
         U_FIT=[495000,585000],CAL=[585544,607479],QUERY_input=[608023,944919],
         QUERY_score=[608535,944856],labels_read=False,source_readonly=True))
    # The accepted array is mmap-backed. Only this role slice is materialized now.
    values=np.load(oldprep/"data/ALL_VALUES.npy",mmap_mode="r",allow_pickle=False)
    assert values.shape==(944919,31)and values.dtype==np.float64
    fit=(np.array(values[495000:585000],dtype=np.float64,order="F",copy=True)-mean)/scale
    assert fit.shape==(90000,31)and np.isfinite(fit).all()
    plan=np.random.default_rng(2025).integers(0,len(fit)-192+1,size=(2000,128))
    np.savez_compressed(root/"U_PLAN.npz",base_starts=plan)
    train=root/"train";train.mkdir()
    dump(train/"INITIAL_IDENTITY.json",dict(checkpoint_sha256=expected,python_step=model.step,
         state_sha256=original_hash,rng=old.rngs(torch),scaler_sha256=sha(scalar_file),
         plan_sha256=sha(root/"U_PLAN.npz"),U_FIT=[495000,585000],labels_read=False,
         condition="original unlabeled background; no replacement or donor access"))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    trace=[];train_started=time.monotonic()
    for update in range(2000):
        batch=np.stack([np.float32(fit[int(a):int(a)+192])for a in plan[update]])
        optimizer.zero_grad();loss=model(torch.from_numpy(batch).cuda(),None,None,None)[0]
        assert torch.isfinite(loss);loss.backward()
        grads=[p.grad for p in model.parameters()if p.grad is not None]
        assert grads and all(torch.isfinite(g).all()for g in grads)
        optimizer.step();assert model.step==52581+update+1
        trace.append(dict(update=update+1,loss=float(loss.detach()),python_step=model.step))
        if(update+1)%500==0:print(json.dumps(dict(event="U_UPDATE",**trace[-1])),flush=True)
    endpoint=train/"endpoint2000.pth"
    torch.save(model.state_dict(),endpoint);torch.save(optimizer.state_dict(),train/"optimizer2000.pth")
    dump(train/"TRACE.json",dict(steps=trace))
    dump(train/"RECEIPT.json",dict(status="COMPLETED",source_checkpoint_sha256=expected,
        checkpoint_sha256=sha(endpoint),updates=2000,initial_python_step=52581,final_python_step=model.step,
        plan_sha256=sha(root/"U_PLAN.npz"),scaler_sha256=sha(scalar_file),
        trainable_parameters=sum(p.numel()for p in model.parameters()if p.requires_grad),
        training_seconds=time.monotonic()-train_started,allocated=torch.cuda.max_memory_allocated(),
        reserved=torch.cuda.max_memory_reserved(),labels_read=False,CAL_QUERY_read=False,
        train_role=[495000,585000],optimizer="Adam lr1e-4",seed=2025,endpoint_only=True))
    # Open readout roles only now, after the fixed endpoint was persisted.
    model.eval();state_before=old.hashes(model);rng_before=old.rngs(torch)
    banks=[];coverage=[]
    torch.cuda.reset_peak_memory_stats()
    for role,lo,hi,gridlo,gridhi,scorelo in [
        ("CAL",585544,607479,585624,607320,585624),
        ("QUERY",608023,944919,608472,944856,608535)]:
        rolevalues=(np.array(values[lo:hi],dtype=np.float64,order="F",copy=True)-mean)/scale
        starts=np.arange(gridlo,gridhi,192,dtype=np.int64)
        assert np.all((starts-495000)%192==0)and starts[0]>=lo and starts[-1]+192<=hi
        chunks=[]
        with torch.no_grad():
            for a in range(0,len(starts),128):
                ids=starts[a:a+128]
                batch=np.stack([np.float32(rolevalues[s-lo:s-lo+192])for s in ids])
                raw=model.infer(torch.from_numpy(batch).cuda(),None,None,None)[0]
                assert raw.shape==(len(batch),192,31)and torch.isfinite(raw).all()
                chunks.append(raw.cpu().numpy())
        channels=np.concatenate(chunks).reshape(-1,31)
        axis=np.arange(gridlo,gridhi,dtype=np.int64);available=np.repeat(starts+191,192)
        keep=axis>=scorelo;axis=axis[keep];channels=channels[keep];available=available[keep]
        score=channels.mean(axis=-1);assert score.dtype==np.float32
        path=root/(role+".npz")
        np.savez_compressed(path,absolute_time_index=axis,index=axis,available_index=available,
            per_channel=channels,canonical_score=score,score=score,valid=np.ones(len(axis),bool),
            window_start=starts,window_end_exclusive=starts+192)
        banks.append(dict(role=role,relative=path.name,sha256=sha(path),points=len(axis),axis=[int(axis[0]),int(axis[-1])+1]))
        coverage.append(dict(role=role,input=[lo,hi],full_windows=[gridlo,gridhi],score_axis=[int(axis[0]),int(axis[-1])+1],windows=len(starts)))
    assert state_before==old.hashes(model)and rng_before==old.rngs(torch)and model.step==54581
    dump(root/"RECEIPT.json",dict(status="COMPLETED",checkpoint_sha256=sha(endpoint),
        source_checkpoint_sha256=expected,scaler_sha256=sha(scalar_file),python_step=model.step,
        worker_sha256=sha(__file__),banks=banks,coverage=coverage,labels_read=False,
        readout_state_step_RNG_immutable=True,source_checkpoint_unchanged=sha(checkpoint)==expected,
        score="native float32 channel mean all window positions",anchor=495000,window=192,stride=192,
        allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved(),
        total_worker_seconds=time.monotonic()-began))
    print(json.dumps(dict(event="STAGE_COMPLETED",banks=banks)),flush=True)


if __name__=="__main__":
    main()
