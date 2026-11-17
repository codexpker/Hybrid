"""Adapter-only factory extension of unchanged native training/scoring kernels."""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import time
import traceback
import torch
from torch import nn
from . import timeradar_cross_variable_runner as base
from .timeradar_adapter_only import AdapterOnlyTimeRadar

HISTORICAL_FACTORY = base.model
ARMS = ('self', 'cross')
CREATED = []


def model(workspace, config, arm, device):
    if arm not in ARMS: raise ValueError('Only historical self/cross adapters')
    wrapped = AdapterOnlyTimeRadar(HISTORICAL_FACTORY(workspace, config, arm, device))
    wrapped.native_initial_hashes = base.state_hashes(wrapped.native)
    CREATED.append(wrapped)
    return wrapped


@contextmanager
def registered_factory():
    original = base.model; base.model = model
    try: yield
    finally: base.model = original


def assert_full_freeze(m):
    after = base.state_hashes(m.native)
    assert after == m.native_initial_hashes, 'Entire native state (including decoder/buffers) changed'
    assert all(not p.requires_grad and p.grad is None for p in m.native.parameters())
    assert all(not module.training for module in m.native.modules())
    return dict(initial=m.native_initial_hashes, final=after, all_native_state_unchanged=True,
                all_native_parameters_frozen=True, all_native_modules_eval=True)


def checkpoint_native_hashes(state):
    import hashlib
    return {k[7:]: hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for k, v in state.items() if k.startswith('native.')}


def mechanism(w, cfg, out, record, arm, device):
    reader = base.RoleReader(w, cfg, record, ['FIT']); values, _ = reader.load('FIT'); spec = cfg['records'][record]
    starts = next(iter(base.native_batches(values, spec['batch_size'])))[:4]
    x = base.tensor_windows(values, starts, device); base.seed(cfg['seed'], device)
    m = model(w, cfg, arm, device).eval()
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == 16912
    cpu = torch.get_rng_state(); cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    m._hook.remove(); m._hook = None
    try:
        with torch.no_grad(): expected = m.native(x, norm=spec['norm'], copies=cfg['copies'])
    finally: m._hook = m.native.decoder.register_forward_pre_hook(m._adapt)
    def restore():
        torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state(cuda, device)
    restore()
    with torch.no_grad(): actual = m(x, norm=spec['norm'], copies=cfg['copies'])
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))
    assert torch.equal(m.cal_anomaly_score(x, expected[1], nn.MSELoss(), L=spec['L']),
                       m.cal_anomaly_score(x, actual[1], nn.MSELoss(), L=spec['L']))
    after_rng = torch.get_rng_state().clone()
    restore()
    with torch.no_grad(): m.adapter_enabled=False; disabled=m(x,norm=spec['norm'],copies=cfg['copies'])
    assert all(torch.equal(a,b) for a,b in zip(expected,disabled))
    assert torch.equal(after_rng,torch.get_rng_state());m.adapter_enabled=True
    initial_adapter = base.state_hashes(m.adapter)
    optimizer=base.optimizer(m,cfg);steps=[];captured=[]
    def retain(module, args, output):
        output.retain_grad();captured.append(output)
    hook=m.adapter.register_forward_hook(retain)
    try:
        for step in range(2):
            m.train();optimizer.zero_grad(set_to_none=True);captured.clear()
            assert all(not q.training for q in m.native.modules()) and m.adapter.training
            _,outputs=m(x,norm=spec['norm'],copies=cfg['copies'])
            loss=base.mse(outputs,x);loss.backward()
            assert captured and captured[-1].grad is not None and torch.isfinite(captured[-1].grad).all()
            assert float(captured[-1].grad.abs().max())>0
            grads={n:float(p.grad.norm()) for n,p in m.adapter.named_parameters() if p.grad is not None}
            assert all(torch.isfinite(p.grad).all() for p in m.adapter.parameters() if p.grad is not None)
            optimizer.step();proof=assert_full_freeze(m)
            steps.append(dict(step=step+1,loss=float(loss.detach()),adapter_gradient_norms=grads,
                              decoder_to_adapter_gradient_max=float(captured[-1].grad.abs().max())))
    finally:hook.remove()
    assert steps[-1]['adapter_gradient_norms']['up.weight']>0 and steps[-1]['adapter_gradient_norms']['down.weight']>0
    mode=[q.training for q in m.modules()];rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None
    before_select=base.state_hashes(m.native)
    with base.validation_mode(m,device):m(x,norm=spec['norm'],copies=cfg['copies'])
    assert mode==[q.training for q in m.modules()] and torch.equal(rng,torch.get_rng_state())
    if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state(device))
    assert base.state_hashes(m.native)==before_select
    base.dump(out/'MECHANISM_RECEIPT.json',dict(status='PASS_FIT_ONLY',record=record,arm=arm,roles=reader.access,
        zero_disabled_native_bitexact=True,initial_adapter_state=initial_adapter,final_adapter_state=base.state_hashes(m.adapter),
        trainable_parameters=16912,steps=steps,native_freeze=proof,mode_RNG_restored=True,no_labels_decoded=True))
    m.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['mechanism','train','score'])
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--record',required=True);p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();w=a.workspace.resolve();config=a.config.resolve();cfg=base.read(config);device=torch.device(a.device)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    out=w/cfg['output']/a.record/a.arm/a.phase;out.mkdir(parents=True,exist_ok=False);start=time.time();rc=0
    if device.type=='cuda':
        torch.cuda.set_device(device);cap=30 if a.phase=='score' else 20
        torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
        torch.cuda.reset_peak_memory_stats(device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    base.dump(out/'START.json',dict(pid=os.getpid(),phase=a.phase,record=a.record,arm=a.arm,start_unix=start,
        config_sha256=base.sha(config),runner_sha256=base.sha(__file__),adapter_policy_sha256=base.sha(Path(__file__).with_name('timeradar_adapter_only.py')),
        base_runner_sha256=base.sha(base.__file__),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        with registered_factory():
            if a.phase=='mechanism':mechanism(w,cfg,out,a.record,a.arm,device)
            elif a.phase=='train':base.train(w,config,cfg,out,a.record,a.arm,device)
            else:base.score(w,config,cfg,out,a.record,a.arm,device)
        proofs=[assert_full_freeze(m) for m in CREATED]
        if a.phase=='train':
            for filename in ['selected.pth','final.pth']:
                state=torch.load(out/filename,map_location='cpu',weights_only=True)
                assert checkpoint_native_hashes(state['state'])==CREATED[0].native_initial_hashes
        base.dump(out/'NATIVE_FREEZE_RECEIPT.json',dict(status='PASS_ENTIRE_NATIVE_STATE',proofs=proofs,
            selected_and_final_checked=a.phase=='train',includes_decoder_and_buffers=True))
    except BaseException:
        rc=1;base.dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
    finally:
        end=time.time();base.dump(out/'OS.json',dict(pid=os.getpid(),start_unix=start,end_unix=end,seconds=end-start,exit_code=rc,
            GPU_hours=(end-start)/3600 if device.type=='cuda' else 0,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)if device.type=='cuda'else 0))


if __name__=='__main__':main()
