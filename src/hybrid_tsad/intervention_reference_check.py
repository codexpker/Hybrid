"""Check raw-score shift invariance and record fixed, unlabeled original/edit pairs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .intervention_reference import (BATCH, ROOT, SUPPORT, source_records, pairs, setup, decoder_model,
    source_median, tensor, forward, load_decoder, target_term, standardizer, target, dump)


def shift_check(original, edited, scale):
    x=original.double().requires_grad_()
    y=edited.double().requires_grad_()
    frozen=x.detach().clone()
    ref=frozen
    rank,anchor=target_term(x,y,frozen,scale,ref)
    offset=7*scale
    shifted,cost=target_term(x+offset,y+offset,frozen,scale,ref)
    gx,gy=torch.autograd.grad(rank,(x,y))
    torch.testing.assert_close(shifted,rank,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(gx+gy,torch.zeros_like(gx),rtol=0,atol=1e-12)
    assert cost>anchor and gy.abs().max()>0
    return dict(rank_change=float((shifted-rank).detach()),anchor_cost=float(cost.detach()),
                max_translation_derivative=float((gx+gy).abs().max()))


def panel(w, domain):
    start=time.monotonic()
    out=w/ROOT
    model=decoder_model(w,setup())
    pools=source_records(w)
    scale=source_median(model,pools["fit"])
    if domain=="source":
        stream=pairs(pools["select"],11,0,56)
        arms=["official","reconstruction","pair"]
    else:
        mu,sd=standardizer(target(w,domain,"FIT"))
        stream=pairs([((target(w,domain,"U")-mu)/sd).astype(np.float32)],31,0,32,source=False)
        arms=["pair","absolute","relative"]
    samples=list(stream)
    rows=[]
    for arm in arms:
        if arm!="official":
            load_decoder(model,out/("source" if arm in ("pair","reconstruction") else domain)/arm/"selected.pt")
        for j,(x,y) in enumerate(samples):
            assert np.array_equal(x[:40],y[:40]) and np.array_equal(x[60:],y[60:])
            diff=(y-x)[SUPPORT]
            with torch.no_grad():
                v,_=forward(model,tensor(x))
                edited,_=forward(model,tensor(y))
            if j==0:
                check=shift_check(v[SUPPORT],edited[SUPPORT],scale)
            rows.append(dict(arm=arm,kind="offset" if (j//2)%2==0 else "time_shift",
                pair=j,mean_V=float(v[SUPPORT].mean()),mean_edited_V=float(edited[SUPPORT].mean()),
                delta=float((edited[SUPPORT]-v[SUPPORT]).mean()),changed_fraction=float((diff!=0).mean()),
                input_rms=float(np.sqrt(np.mean(diff.astype(float)**2)))))
    dump(out/domain/"paired_check.json",dict(rows=rows,shift_check=check,seconds=time.monotonic()-start))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("workspace",type=Path)
    parser.add_argument("domain",choices=("source","PSM","SWAT"))
    args=parser.parse_args()
    panel(args.workspace,args.domain)
