"""Frozen LEFT fusion-token correction of a frozen conditional Ridge response.

T-token-only is NOT frequency-information-off: T is already cross-view fused.
No current response target enters the historical feature extractor.
"""
from contextlib import contextmanager
import numpy as np
import torch


def pooled_tokens(output):
    t,f,_=output
    if t.ndim!=3 or f.ndim!=3 or len(t)!=len(f)or t.shape[-1]!=f.shape[-1]:
        raise ValueError('Expected fused batch,time,embedding tokens')
    return torch.cat((t.mean(1),f.mean(1)),1)


@contextmanager
def fusion_capture(model):
    captured=[]
    def hook(module,args,output):captured.append(pooled_tokens(output).detach())
    handle=model.fusion.register_forward_hook(hook)
    try:yield captured
    finally:handle.remove()


@torch.no_grad()
def frozen_features(model,arrived):
    if arrived.ndim!=3 or arrived.shape[1:]!=(52,9):raise ValueError('Expected52x9')
    if model.training or any(m.training for m in model.modules()):raise ValueError('All frozen LEFT modules must eval')
    if any(p.requires_grad for p in model.parameters()):raise ValueError('LEFT parameters must be frozen')
    with fusion_capture(model)as capture:
        model(arrived[:,:48],None,None,None)
    if len(capture)!=1:raise ValueError('Exactly one original fusion call required')
    return capture[0]


def feature_stats(phi):
    phi=np.asarray(phi,np.float64)
    if phi.ndim!=2 or phi.shape[1]%2:raise ValueError('Two equal feature slots required')
    return phi.mean(0),np.maximum(phi.std(0),1e-6)


def design(phi,mean,scale,variant):
    x=(np.asarray(phi,np.float64)-mean)/scale
    if variant=='T_only':x[:,x.shape[1]//2:]=0
    elif variant!='TF':raise ValueError(variant)
    return x


def fit_head(phi,residual,mean,scale,variant,lam=1.):
    x=design(phi,mean,scale,variant);y=np.asarray(residual,np.float64).reshape(len(x),16)
    alpha=len(x)*16*lam
    # User objective uses a per-scalar mean, so alpha is N*16*lambda.
    weight=np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@y)
    predicted=x@weight
    objective=float(np.mean((y-predicted)**2)+lam*np.sum(weight**2))
    return weight,dict(alpha_sumloss=alpha,lambda_mean_scalar=lam,objective=objective,
                      residual_MSE=float(np.mean((y-predicted)**2)),ridge_penalty=float(lam*np.sum(weight**2)),
                      normal_equation_max_abs=float(np.max(np.abs((x.T@x+alpha*np.eye(x.shape[1]))@weight-x.T@y))))


def ridge_predict(history,coef,intercept):
    history=np.asarray(history,np.float64)
    if history.ndim!=3 or history.shape[1:]!=(48,9):raise ValueError('Strict48x9 history required')
    return (history.reshape(len(history),-1)@coef.T+intercept).reshape(-1,4,4)
