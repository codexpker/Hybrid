"""Source-only response preservation without changing native training state."""
from contextlib import contextmanager
import random

import numpy as np
import torch


@contextmanager
def preserving_eval(model):
    """Keep the eval graph, restore heterogeneous modes and all native state.

    Temporary buffer clones protect both the original native training graph and
    the auxiliary graph: no participating buffer is restored by an inplace copy.
    Parameters are never replaced or modified by this context.
    """
    modules=list(model.modules());modes=[m.training for m in modules]
    buffers=[(m,dict(m._buffers))for m in modules]
    attrs={name:getattr(model,name)for name in ["step","last_x_res","_ms_cache"]if hasattr(model,name)}
    if "_ms_cache"in attrs:attrs["_ms_cache"]=attrs["_ms_cache"].copy()
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all()if torch.cuda.is_initialized()else None
    python=random.getstate();numpy=np.random.get_state()
    try:
        for module,original in buffers:
            module._buffers={k:(v.detach().clone()if v is not None else None)for k,v in original.items()}
        model.eval()
        yield
    finally:
        for module,original in buffers:module._buffers=original
        for module,mode in zip(modules,modes):module.training=mode
        for name,value in attrs.items():setattr(model,name,value)
        torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        random.setstate(python);np.random.set_state(numpy)


def views(model,source,*,grad):
    with preserving_eval(model),torch.set_grad_enabled(grad):
        result=model._forward(source,None,None,None)
        outputs=(result[4],result[3],result[9])
        if any(x.shape!=source.shape for x in outputs):
            raise RuntimeError("All three native reconstructed source views must have input shape/units")
        return outputs


def preservation_loss(student,teacher,source,mode):
    if mode not in {"P","V"}:raise ValueError("Only the two fixed preservation definitions")
    prediction=views(student,source,grad=True)
    reference=views(teacher,source,grad=False)
    delta=torch.stack([a-b for a,b in zip(prediction,reference)],dim=0)
    if not torch.isfinite(delta).all():raise FloatingPointError("Nonfinite preservation response")
    return delta.mean(dim=0).square().mean()if mode=="P"else delta.square().mean()


def gradient_norm(loss,parameters):
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    values=[v.detach().float().square().sum()for v in gradients if v is not None]
    if not values:return 0.0
    return float(torch.stack(values).sum().sqrt())
