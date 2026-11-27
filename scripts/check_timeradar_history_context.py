"""CPU integration check against pinned official TimeRadar and normal TEP input."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from hybrid_tsad.native_v_conditioning import load_native, sha, state_hashes, SOURCE_SHA, WEIGHT_SHA
from hybrid_tsad.timeradar_history_context import HistoryConditionedTimeRadar, record_window, reconstruction_update


def run(workspace):
    torch.set_num_threads(1)
    torch.manual_seed(2025)
    start = time.monotonic()
    source = workspace/"results/native_tep_support_20261125_v1/fault_free_training.npy"
    assert sha(source) == "2c4e24dc41b14a7a521de71a8bbac87cef3788fcf46c9dd321aab5a14f38c698"
    raw = np.load(source)[0, 0, :, :3]
    mean, scale = raw.mean(0), raw.std(0)
    x = torch.from_numpy(((raw-mean)/np.maximum(scale,1e-6)).astype(np.float32))
    history, current = x[:256][None], x[256:356][None]
    h, c, footprint = record_window(x, 256, (0, 500), 256)
    assert torch.equal(h, history) and torch.equal(c, current)
    try:
        record_window(x, 256, (1, 500), 256)
    except ValueError:
        pass
    else:
        raise AssertionError("Cross-role footprint accepted")
    native = load_native(workspace, "cpu")
    model = HistoryConditionedTimeRadar(native)
    official = {k:v.clone() for k,v in native.state_dict().items()}
    with torch.no_grad():
        _, ref = model(history, current, history_start=0, current_start=256, bypass=True)
        _, zero = model(history, current, history_start=0, current_start=256)
    assert torch.equal(ref,zero)
    loss,score = model.loss_and_score(zero,current)
    torch.testing.assert_close(score, native.cal_anomaly_score(current,zero,None,1),rtol=0,atol=0)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
    gradient = {}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, out = model(history,current,history_start=0,current_start=256)
        fit,_ = model.loss_and_score(out,current)
        fit.backward()
        gradient[str(step)] = {name:float(sum(p.grad.abs().sum().item() for p in module.parameters()
                                             if p.grad is not None))
                               for name,module in (("history",model.history),("cross",model.cross),
                                                   ("projection",model.output),("decoder",native.decoder))}
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    assert all(v>0 for v in gradient["1"].values()), gradient
    with torch.no_grad():
        memory, own = model.history(history)
        perm = torch.tensor([2, 0, 1])
        permuted, permuted_own = model.history(history[:, :, perm])
        torch.testing.assert_close(permuted, memory[perm], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(permuted_own, own[perm], rtol=1e-5, atol=1e-6)
        changed = history.clone()
        changed[:, 128:] += 2
        altered, _ = model.history(changed)
        torch.testing.assert_close(memory[:, :4], altered[:, :4], rtol=0, atol=0)
        long_memory, _ = model.history(torch.cat((history, history), 1))
        assert long_memory.shape == (3, 28, 64)
    frozen = [k for k in official if not k.startswith("decoder.")]
    assert all(torch.equal(official[k],native.state_dict()[k]) for k in frozen)
    reconstruction_update(model, optimizer, [(h, c, footprint)])
    model.eval()
    with torch.no_grad():
        _, original = model(history,current,history_start=0,current_start=256)
        _, replacement = model(history.flip(1),current,history_start=0,current_start=256)
        _, bypass = model(history,current,history_start=0,current_start=256,bypass=True)
    assert (original-replacement).abs().max()>0
    before, after = state_hashes(model), None
    rng=torch.get_rng_state().clone()
    model(history,current,history_start=0,current_start=256)
    assert torch.equal(rng,torch.get_rng_state()) and before==state_hashes(model)
    try:
        model(history,current,history_start=1,current_start=256)
    except ValueError:
        pass
    else:
        raise AssertionError("Illegal history footprint accepted")
    # Extending C never changes parameter topology. Current snapshots are ephemeral.
    with torch.no_grad():
        for channels in (1,25,31,38):
            h=history[:,:,:1].expand(1,256,channels).contiguous()
            c=current[:,:,:1].expand(1,100,channels).contiguous()
            _, y=model(h,c,history_start=0,current_start=256)
            assert y.shape==(10,1,100,channels)
    return dict(status="passed",scope="CPU normal-source integration; three engineering updates, no scientific fit",
                source_sha256=sha(source),official_source=SOURCE_SHA,official_weights=WEIGHT_SHA,
                parameters=sum(p.numel() for p in model.parameters()),
                new_parameters=sum(p.numel() for n,p in model.named_parameters() if not n.startswith("native.")),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                zero_context_max_error=float((ref-zero).abs().max()),gradient=gradient,
                history_replacement_output_max_change=float((original-replacement).abs().max()),
                path_bypass_output_max_change=float((original-bypass).abs().max()),
                history_replacement_V_max_change=float((original.var(0)-replacement.var(0)).abs().max()),
                native_nondecoder_unchanged=True,isolated_rng=True,elapsed_seconds=time.monotonic()-start)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    result=run(args.workspace)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
