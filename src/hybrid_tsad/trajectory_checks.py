"""Arithmetic checks on a fixed source-independent nominal target panel, before fitting."""
import time
import numpy as np
from . import trajectory_subspace as t


def check(raw):
    rng = np.random.default_rng(2025)
    mean,scale=raw.mean(0),raw.std(0)
    x=(raw[:8*t.W]-mean)/np.where(scale>0,scale,1)
    xs=x.reshape(8,t.W,-1)
    x=xs[0]
    h=t.hankel(x)
    dual=rng.normal(size=h.shape)
    np.testing.assert_allclose(np.vdot(h,dual),np.vdot(x,t.adjoint(dual)),rtol=1e-12)
    np.testing.assert_allclose(t.F@t.F.T,np.eye(32),atol=2e-14)
    r=x-rng.normal(size=x.shape)
    o=t.soft(r)
    np.testing.assert_allclose(.5*(r-o)**2+abs(o),np.where(abs(r)<=1,.5*r*r,abs(r)-.5))
    g=t.covariance(xs)
    p=t.planes(g)
    b=t.curvature(p)
    np.testing.assert_allclose(np.sum(x*(b@x))/(2*x.size),
        np.sum((np.eye(8)-p)*t.covariance([x]))/2+t.EPS*np.mean(x*x)/2,atol=1e-13)
    y=rng.normal(size=x.shape)
    direction=rng.normal(size=x.shape)
    value,grad=t.smooth(x,y,b,True)
    step=1e-5
    fd=(t.smooth(x,y+step*direction,b,True)[0]-t.smooth(x,y-step*direction,b,True)[0])/(2*step)
    np.testing.assert_allclose(fd,np.sum(grad*direction),rtol=2e-7,atol=1e-7)
    u,s,v=np.linalg.svd(y,full_matrices=False)
    np.testing.assert_allclose(t.svt(y,.6)[0],(u*np.maximum(s-.6,0))@v,atol=1e-12)
    other=t.planes(t.covariance(xs[::-1]*np.arange(1,9)[:,None,None]))
    updated=t.planes(g,other)
    np.testing.assert_allclose(updated@updated,updated,atol=1e-12)
    mat=g+2*t.ETA*other
    np.testing.assert_allclose(np.sum(updated*mat),np.linalg.eigvalsh(mat)[:,-2:].sum(),atol=1e-12)
    sol=t.Solver(p,x.shape[1])
    clock=time.monotonic()
    diag=np.stack([sol.solve(z)[1] for z in xs])
    assert np.all(diag[:,3]<=diag[:,2]+1e-12)
    return dict(panel="PSM first 8 disjoint normal FIT blocks",blocks=8,checks=["adjoint","DCT",
        "Huber","lag quadratic","gradient","SVT","anchored eigen optimum","descent"],
        max_relative_residual=diag[:,1].max(),iterations=diag[:,0].tolist(),
        mean_initial=diag[:,2].mean(),mean_final=diag[:,3].mean(),seconds=time.monotonic()-clock)
