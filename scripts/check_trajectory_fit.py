"""Summarize frozen planes on nominal SELECT and verify the first real U block objective."""
import argparse
import time
import numpy as np
from pathlib import Path
from threadpoolctl import threadpool_limits

from hybrid_tsad.trajectory_experiment import (OUT, ROLES, blocks, target, dump)
from hybrid_tsad import trajectory_subspace as ts


def main(w):
    out=w/OUT
    summary={}
    start=time.monotonic()
    ps=np.load(out/"source/planes.npy")
    for domain in ROLES:
        dest=out/domain
        with np.load(dest/"Z/scaler.npz") as scaler:
            x=blocks((target(w,domain,"SELECT")-scaler["mean"])/scaler["scale"])
        g=ts.covariance(x)
        summary[domain]=dict(select_blocks=len(x), fits={})
        for arm in ("Z","source_normal","target_normal","source_quadratic","target_robust","source_robust"):
            p=np.load(dest/arm/"planes.npy")
            assert np.isfinite(p).all()
            np.testing.assert_allclose(p@p,p,atol=1e-12)
            np.testing.assert_allclose(np.trace(p,axis1=1,axis2=2),2,atol=1e-12)
            summary[domain]["fits"][arm]=dict(normal_SELECT_trajectory_risk=float(np.sum((np.eye(8)-p)*g)/2),
                                              distance_from_source=float(np.linalg.norm(p-ps)))
        # Independent explicit formula on the first real U block, using the actual saved input grid.
        with np.load(dest/"source_robust/scaler.npz") as scaler:
            part=target(w,domain,"U")[:ts.W]
            x=(part-scaler["mean"])/scaler["scale"]
        y,record=ts.Solver(ps,x.shape[1]).solve(x)
        o=ts.soft(x-y)
        prior=0.
        for start_at in range(ts.SHIFTS):
            coeff=ts.F@y[start_at:start_at+ts.L]
            for b in range(4):
                residue=(np.eye(8)-ps[b])@coeff[8*b:8*b+8]
                prior+=np.sum(residue**2)
        direct=(.5*np.mean((x-y-o)**2)+np.mean(abs(o))
                +prior/(2*ts.L*ts.SHIFTS*x.shape[1])
                +ts.TAU*np.linalg.svd(y,compute_uv=False).sum()/np.sqrt(x.size)
                +ts.EPS*np.mean(y*y)/2)
        saved=np.load(dest/"source_robust/sweep1_diagnostics.npy")[0]
        np.testing.assert_allclose(direct,saved[3],atol=1e-12,rtol=1e-12)
        summary[domain]["first_U_block_objective_abs_error"]=abs(direct-saved[3])
        summary[domain]["first_U_block_iterations"]=int(record[0])
    summary["seconds"]=time.monotonic()-start
    dump(out/"fit_check.json",summary)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("workspace",type=Path)
    with threadpool_limits(1):
        main(p.parse_args().workspace)
