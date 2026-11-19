"""Engineering-only fixed-weight CUDA gate for conditional response energy."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from .conditional_response_energy import ConditionalGenerator, ResponseGeometry, fixed_noise, energy_score
from .hybrid_v1.provenance import OfficialSources


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_native(workspace, device):
    cfg=json.loads((workspace/'hybrid_ad/configs/timeradar-cross-variable-20261116.json').read_text())
    source=workspace/cfg['author_source']; checkpoint=workspace/cfg['checkpoint']
    assert sha(source)==cfg['author_source_sha256'] and sha(checkpoint)==cfg['checkpoint_sha256']
    spec=importlib.util.spec_from_file_location('_response_official_timeradar',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.Model(SimpleNamespace(**cfg['model']))
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict' in state: state=state['state_dict']
    state={k.removeprefix('module.'):v for k,v in state.items()}
    for key in sorted(set(state)-set(model.state_dict())):
        if key not in ['tau_n','tau_a','tau_inited']:raise ValueError(key)
        model.register_buffer(key,state[key].clone())
    model.load_state_dict(state,strict=True)
    return model.to(device).eval(),cfg


def hashes(model):
    return {k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()for k,v in model.state_dict().items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--domain',choices=['SWAT','PSM'],required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(20/45,device)
    torch.manual_seed(2025);torch.cuda.manual_seed(2025)
    task=json.loads((a.workspace/'results/mlp_replay_20261119_v1'/a.domain/'TASK.json').read_text())
    rec=task['data']['source_train'];path=a.workspace/rec['relative'];assert sha(path)==rec['sha256']
    raw=np.load(path,mmap_mode='r');lo,hi=task['roles']['SOURCE_FIT']
    fit=np.asarray(raw[lo:hi],dtype=np.float64)
    mean=fit.mean(0);std=fit.std(0);std[std<1e-8]=1
    h=torch.tensor((fit[:512]-mean)/std,dtype=torch.float32,device=device)[None]
    y=torch.tensor((fit[512]-mean)/std,dtype=torch.float32,device=device)[None]
    native,cfg=load_native(a.workspace,device);native_before=hashes(native)
    g=ConditionalGenerator(OfficialSources.from_workspace(a.workspace)).to(device)
    geometry=ResponseGeometry(native,1.);opt=torch.optim.Adam(g.parameters(),lr=1e-4)
    noise=fixed_noise(1,h.shape[2],torch.Generator().manual_seed(2025),device=device)
    torch.cuda.synchronize();t=time.monotonic();v=g(h,*noise);score=geometry.score(h,v,y)
    torch.cuda.synchronize();forward=time.monotonic()-t
    rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
    score2=geometry.score(h,v,y)
    assert torch.equal(rng,torch.get_rng_state())and torch.equal(crng,torch.cuda.get_rng_state())
    torch.testing.assert_close(score,score2,rtol=0,atol=0)
    gv=torch.stack([geometry.feature(h[0],value) for value in v[0]])
    with torch.no_grad(): gy=geometry.feature(h[0],y[0])
    response_only=energy_score(gv[None,:,h.shape[2]:],gy[None,h.shape[2]:]).mean()
    response_grad=torch.autograd.grad(response_only,v,retain_graph=True)[0]
    raw_only=geometry.score(h,v,y,False).mean()
    raw_grad=torch.autograd.grad(raw_only,v,retain_graph=True)[0]
    combined_grad=torch.autograd.grad(score.mean(),v,retain_graph=True)[0]
    assert torch.isfinite(response_grad).all()and response_grad.abs().sum()>0
    assert not torch.equal(raw_grad,combined_grad)
    # Changing a neighbouring batch sample cannot affect the first prediction.
    pair=h.expand(2,-1,-1).clone();pair[1].add_(7)
    shared,variate=fixed_noise(1,h.shape[2],torch.Generator().manual_seed(2025),device=device)
    pairpred=g(pair,shared.expand(2,-1,-1),variate.expand(2,-1,-1,-1))
    torch.testing.assert_close(v[0],pairpred[0],rtol=1e-5,atol=1e-5)
    torch.cuda.synchronize();t=time.monotonic();score.mean().backward();torch.cuda.synchronize();backward=time.monotonic()-t
    finite=all(p.grad is None or torch.isfinite(p.grad).all()for p in g.parameters())
    assert finite and any(p.grad is not None and p.grad.abs().sum()>0 for p in g.parameters())
    grad=float(sum(p.grad.square().sum() for p in g.parameters()if p.grad is not None).sqrt())
    opt.step();assert native_before==hashes(native)and all(p.grad is None for p in native.parameters())
    times=[]
    with torch.no_grad():
        for _ in range(4):
            torch.cuda.synchronize();tt=time.monotonic()
            vp=g(h,*noise);geometry.score(h,vp,y)
            torch.cuda.synchronize();times.append(time.monotonic()-tt)
    # Real raw-only B16 update is a separate timing witness, not scientific training.
    torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True)
    hh=h.expand(16,-1,-1);yy=y.expand(16,-1)
    vv=g(hh,*fixed_noise(16,h.shape[2],torch.Generator().manual_seed(2025),device=device))
    rawscore=geometry.score(hh,vv,yy,False);rawscore.mean().backward();opt.step()
    torch.cuda.synchronize();rawseconds=time.monotonic()-t
    out=dict(status='PASS',engineering_only=True,source_fit_only=True,no_QUERY_opened=True,domain=a.domain,
        forward_B1_seconds=forward,backward_B1_seconds=backward,raw_B16_step_seconds=rawseconds,
        energy=float(score.detach()[0]),generator_gradient_norm=grad,teacher_all_parameters_and_buffers_unchanged=True,
        response_only_input_gradient_norm=float(response_grad.norm()),raw_vs_combined_gradient_max_difference=float((raw_grad-combined_grad).abs().max()),
        neighbour_independence_close=True,steady_B1_readout_seconds=times,
        teacher_no_gradients=True,mask_RNG_restored=True,repeat_bitexact=True,
        trainable_parameters=sum(p.numel()for p in g.parameters()),teacher_parameters=sum(p.numel()for p in native.parameters()),
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,actual_GPU_seconds=time.monotonic()-start,
        provisional_profile_response_rms=1.,profile_discarded_model=True,source_data_sha256=rec['sha256'],
        official_checkpoint_sha256=cfg['checkpoint_sha256'],module_sha256=sha(Path(__file__).with_name('conditional_response_energy.py')))
    (a.output/'GATE.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)


if __name__=='__main__': main()
