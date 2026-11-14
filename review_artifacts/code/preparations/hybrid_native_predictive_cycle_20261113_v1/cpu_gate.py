"""Incremental scientific-path tests, synthetic fixtures only, no QUERY values."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,copy,hashlib,time
import numpy as np,torch
P=Path(__file__).resolve().parent;R=P.parents[1];sys.path.insert(0,str(R/'preparations/left_baseline_build_20261101_v1/upstream'));sys.path.insert(0,str(P/'runtime'))
from models.Left.LEFT import Left
from left_predictive_cycle import predictive_cycle_class
torch.set_num_threads(1);started=time.monotonic();rows=[]
def state(m):return {k:v.detach().clone()for k,v in m.state_dict().items()}
def snapshot_equal(a,b):return set(a)==set(b)and all(torch.equal(a[k],b[k])for k in a)
def gradnorm(m,prefix):return sum(float(p.grad.abs().sum())for n,p in m.named_parameters()if n.startswith(prefix)and p.grad is not None)
for domain in ['TEP','PSM']:
    config=json.loads((P/'configs'/(domain+'.json')).read_text());cfg=SimpleNamespace(**config);H,C=cfg.seq_len,cfg.enc_in;torch.manual_seed(2025);base=Left(cfg);initial=state(base);torch.manual_seed(2025);native=predictive_cycle_class(Left,'native')(cfg);assert snapshot_equal(initial,state(native));torch.manual_seed(2025);full=predictive_cycle_class(Left,'full')(cfg);assert snapshot_equal(initial,state(full));assert not any(isinstance(x,torch.nn.modules.batchnorm._BatchNorm)for x in full.modules())
    g=torch.Generator().manual_seed(1547);arr=torch.randn(2,H+1,C,generator=g);x=arr[:,1:];past=arr[:,:-1]
    # Native mode is exact, including gradient/RNG and explicit EMA updates.
    for model in [base,native]:model.conf_threshold=0.;model.update_q_err=1.;model.update_q_ad=1.;model.step=cfg.warmup_steps+cfg.ramp_steps+5;model.train()
    torch.manual_seed(732);loss0=base(x,None,None,None)[0];rng0=torch.get_rng_state().clone();loss0.backward();s0=state(base);gr0={n:p.grad.detach().clone()for n,p in base.named_parameters()if p.grad is not None}
    torch.manual_seed(732);loss1=native(arr,None,None,None)[0];rng1=torch.get_rng_state().clone();loss1.backward();assert torch.equal(loss0,loss1)and torch.equal(rng0,rng1)and snapshot_equal(s0,state(native));assert all(torch.equal(gr0[n],p.grad)for n,p in native.named_parameters()if n in gr0)
    # Native score/target separation identity when source input == target.
    full.eval();base.load_state_dict(initial);base.eval();full.load_state_dict(initial)
    with torch.no_grad():
        pred=full.prediction_forward(past);a=full._prediction_score(pred,past);b=base.infer(past,None,None,None)[0];retarget_error=float((a-b).abs().max());assert retarget_error==0
    # Past only forward is immutable, RNG-neutral and sensitive to no y_t.
    full.train();full.step=cfg.warmup_steps+cfg.ramp_steps+5;before=state(full);step=full.step;rng=torch.get_rng_state().clone();pred=full.prediction_forward(past);assert full.step==step and snapshot_equal(before,state(full))and torch.equal(rng,torch.get_rng_state())
    terminal,parts=full._terminal(pred,x);full.zero_grad();terminal.backward();grads={p:gradnorm(full,p)for p in ['Et.','Ef.','fusion.','encoder.','ms_decoder.','Dt.','Df.','proto_t.','proto_f.','z_fuse.']};assert all(v>0 and np.isfinite(v)for v in grads.values()),grads
    # Force actual EMA+nonzero memory mixing. Capture first (past) decode before
    # later current branch, perturb only current target, and execute full backward.
    models=[];captures=[];ema_counts=[]
    for changed in [False,True]:
        torch.manual_seed(2025);model=predictive_cycle_class(Left,'full')(cfg);model.train();model.step=step;model.conf_threshold=0.;model.update_q_err=1.;model.update_q_ad=1.;cap={};count={};handles=[]
        for name,module in [('Dt',model.Dt),('Df',model.Df),('MS',model.ms_decoder)]:
            def hook(mod,args,out,name=name):
                if name not in cap:cap[name]=(out[0]if isinstance(out,tuple)else out).detach().clone()
            handles.append(module.register_forward_hook(hook))
        for name,bank in [('t',model.proto_t),('f',model.proto_f)]:
            orig=bank.ema_update_hard;count[name]=0
            def ema(*args,name=name,orig=orig,**kwargs):count[name]+=1;return orig(*args,**kwargs)
            bank.ema_update_hard=ema
        v=arr.clone()
        if changed:v[:,-1,0]+=50
        torch.manual_seed(962);loss=model(v,None,None,None)[0];loss.backward();assert torch.isfinite(loss)and all(torch.isfinite(p.grad).all()for p in model.parameters()if p.grad is not None);assert model.step==step+1 and count=={'t':1,'f':1}
        assert not snapshot_equal(initial,state(model))
        models.append(model);captures.append(cap);ema_counts.append(count)
        for handle in handles:handle.remove()
    assert all(torch.equal(captures[0][k],captures[1][k])for k in captures[0]);proto_difference=float((models[0].proto_t.prototypes-models[1].proto_t.prototypes).abs().sum()+(models[0].proto_f.prototypes-models[1].proto_f.prototypes).abs().sum());assert proto_difference>0
    # Inference output is independent of arrived current truth; only score changes.
    full.eval();v=arr.clone();v[:,-1,0]+=10
    with torch.no_grad():
        pa=full.prediction_forward(arr[:,:-1]);pb=full.prediction_forward(v[:,:-1]);assert all(torch.equal(pa[i],pb[i])for i in [1,3,4,9]);sa=full.infer(arr)[0];sb=full.infer(v)[0];assert not torch.equal(sa,sb)
    # Across-batch target perturbation cannot influence other sample predictions.
    with torch.no_grad():
        pa=full.prediction_forward(arr[:,:-1]);other=arr.clone();other[1]*=7;pb=full.prediction_forward(other[:,:-1]);batch_error=float((pa[4][0]-pb[4][0]).abs().max());assert batch_error<1e-4
    if domain=='TEP':
        torch.manual_seed(2025);ho=predictive_cycle_class(Left,'history_only')(cfg).eval();v=past.clone();v[:,:,1:]+=99
        with torch.no_grad():assert torch.equal(ho.prediction_forward(past)[4],ho.prediction_forward(v)[4])
    rows.append(dict(domain=domain,H=H,C=C,parameters_added=0,initial_native_parameters_exact=True,native_loss_gradient_RNG_EMA_exact=True,retarget_input_equals_truth_score_max_error=retarget_error,past_no_state_RNG_update=True,terminal_only_shared_module_gradient_L1=grads,current_target_perturbation_past_decoders_exact=True,forced_EMA_calls=ema_counts,forced_EMA_prototype_change_from_current_perturbation=proto_difference,full_backward_after_EMA_finite=True,batch_other_sample_prediction_max_error=batch_error,no_BatchNorm=True,history_only_past_condition_invariance=domain=='TEP'))
result=dict(status='PASS',seconds=time.monotonic()-started,fixtures='synthetic seeded, not QUERY',model_sha256=hashlib.sha256((P/'runtime/left_predictive_cycle.py').read_bytes()).hexdigest(),tests=rows)
(P/'CPU_GATE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
