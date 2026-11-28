"""Small sealed-state diagnostics with exact native terminal V."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .timeradar_history_experiment import model,pools,get_source
from .trajectory_experiment import dump
from .native_v_conditioning import sha


def run(w,out):
    data=pools(w)["select"]
    with np.load(out/"plans.npz") as z: panel=z["select"]
    results=[]
    for stage,domain in (("Z",None),("U","PSM"),("U","SWAT")):
        for arm in ("native","history"):
            path=out/("source" if domain is None else domain)/arm/"selected.pt"
            assert (path.parent/"receipt.json").is_file()
            m=model(w,arm).eval()
            m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["model"])
            before={k:v.clone() for k,v in m.state_dict().items()}
            with torch.no_grad():
                for i,row in enumerate(panel):
                    g,r,t=map(int,row)
                    h,x,footprint=get_source(data,row,"cuda")
                    values={}
                    for variant in ("correct","bypass","mismatch"):
                        if variant=="mismatch":
                            if g==0 or t<1024:
                                values[variant]=None
                                continue
                            # Substitute a strictly earlier, nonoverlapping history in the same record/role.
                            other=data[g][r][t-1024:t-512].copy()
                            actual=torch.as_tensor(other,device="cuda")[None]
                        else:
                            actual=h
                        y=m(actual,x,**footprint,bypass=variant=="bypass" or arm=="native")[1]
                        values[variant]=dict(V_terminal=y.var(0,correction=1).mean(-1)[0,-1].item(),
                                            copy=(y-x[None]).square().mean().item())
                    results.append(dict(stage=stage,domain=domain,arm=arm,panel=i,source_domain=g,record=r,
                          current_start=t,history_start=t-len(h[0]),
                          mismatch_history=[t-1024,t-512] if values["mismatch"] else None,
                          mismatch_NA_reason="No legal past nonoverlapping history; no overlap substituted" if values["mismatch"] is None else None,
                          score=values))
            assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    dump(out/"history_diagnostics.json",results)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("workspace",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args();run(a.workspace,a.output)
