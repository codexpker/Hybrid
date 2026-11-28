"""One in-memory F update on actual isolated support; never a reported F fit."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from hybrid_tsad.timeradar_history_experiment import model
from hybrid_tsad.native_v_conditioning import sha
from hybrid_tsad.native_v_replay import optimizer
from hybrid_tsad.trajectory_experiment import dump


def check(w, out):
    torch.set_num_threads(1)
    started = time.monotonic()
    records = []
    for domain in ("PSM", "SWAT"):
        root = out / domain / "F_support"
        with np.load(root / "inputs.npz") as z:
            ends = z["ends"]
            normal = int(z["normal_ends"][0])
            values = z["footprints"]
        with np.load(root / "plan.npz") as z:
            positive = int(z["positive"][0, 0])
        def window(endpoint):
            x = torch.as_tensor(values[np.searchsorted(ends, endpoint)], device="cuda")
            return x[:512][None], x[512:][None]
        def evidence(m, arm, endpoint):
            h, x = window(endpoint)
            y = m(h, x, history_start=0, current_start=512, bypass=arm == "native")[1]
            return y.var(0, correction=1).mean(-1)[0, -1], (y - x[None]).square().mean()
        official = model(w, "native")
        with torch.no_grad():
            scale = max(1e-8, evidence(official, "native", normal)[0].item())
        del official
        for arm in ("native", "history"):
            path = out / "source" / arm / "selected.pt"
            digest = sha(path)
            m = model(w, arm)
            m.load_state_dict(torch.load(path, weights_only=True, map_location="cpu")["model"])
            frozen = {k: v.clone() for k, v in m.native.state_dict().items() if not k.startswith("decoder.")}
            before = {k: p.detach().clone() for k, p in m.named_parameters() if p.requires_grad}
            opt = optimizer(m)
            vn, copy = evidence(m, arm, normal)
            vp, _ = evidence(m, arm, positive)
            pair = F.softplus(1 - torch.log1p(vp / scale) + torch.log1p(vn / scale))
            loss = copy + pair
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            opt.step()
            assert all(torch.equal(v, m.native.state_dict()[k]) for k, v in frozen.items())
            displacement = sum((p.detach() - before[k]).square().sum().item()
                               for k, p in m.named_parameters() if k in before) ** .5
            assert displacement > 0 and sha(path) == digest
            records.append(dict(domain=domain, arm=arm, scale=scale, copy=copy.item(), pair=pair.item(),
                finite_gradient=True, decoder_and_history_scope_only=True, parameter_displacement=displacement,
                source_checkpoint_unchanged=True, normal_endpoint=normal, positive_endpoint=positive))
    dump(out / "F_entry_check.json", dict(scope="one copy window and one pair; one disposable update per domain/arm",
         reported_F_endpoints=0, seconds=time.monotonic()-started, checks=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    check(args.workspace, args.output)
