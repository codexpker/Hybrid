"""Collect sealed checkpoints, same-axis metrics and source exposure accounting."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hybrid_tsad.left_multisource import (OUT,NATIVE,WIDTH,SOURCE_STEPS,BATCH,WINDOW,
    initialize,source_records,target_inputs,target,sha,read,dump)
from hybrid_tsad.trajectory_evaluation import save_rows
from hybrid_tsad.left_decoder_context_run import normal
from hybrid_tsad.left_smd28_author import bindings


def sample_use(pools,width,seed,steps):
    rng=np.random.default_rng(seed)
    counts=[Counter(),Counter()]
    for _ in range(steps):
        for j in range(BATCH):
            g=j%2
            k=int(rng.integers(len(pools[g])))
            x=pools[g][k]
            start=int(rng.integers(len(x)-WINDOW+1))
            cols=rng.choice(x.shape[1],width,replace=False)
            assert len(set(cols))==width and start+WINDOW<=len(x)
            counts[g][k]+=1
    return counts


def collect(w):
    root=w/OUT
    dest=root/"summary"
    dest.mkdir(exist_ok=True)
    fits=source_records(w)
    rows=[]
    sources=[]
    manifests=[]
    checks={}
    for d,c in WIDTH.items():
        arc=w/NATIVE/d/"native"
        cfg=read(arc/"CONFIG.json")
        scratch=read(arc/"TRAIN_RECEIPT.json")
        score=read(arc/"SCORE_RECEIPT.json")
        assert score["checkpoint_sha256"]==scratch["selected"]["sha256"]
        src=root/d/"source"
        s=read(src/"receipt.json")
        select=[r for r in s["curve"] if "select_loss" in r]
        assert min(select,key=lambda r:(r["select_loss"],r["step"]))["step"]==s["selected"]
        ck=torch.load(src/"selected.pt",weights_only=True,map_location="cpu")
        init=initialize(w,d).cpu()
        pchanges={b:float((ck["state"][b+".prototypes"]-getattr(init,b).prototypes.detach()).norm())
                  for b in ("proto_t","proto_f")}
        _,_,scaler,_=target_inputs(w,d)
        if d=="PSM":
            # PSM random-U initially used the archived trainer's equivalent scaler fit.
            _,_,original,_=normal(w,d)
            x=target(w,d,"U")
            np.testing.assert_array_equal(scaler.transform(x).astype("f"),original.transform(x).astype("f"))
        checks[d]=dict(prototype_parameter_change_norm=pchanges,upstream=cfg["upstream_files"],
            config_files=cfg["configuration"],scaler_sha256=cfg["scaler_sha256"],
            source_selection=select,scratch_checkpoint_sha256=score["checkpoint_sha256"],
            scratch_score_sha256=score["bank_sha256"],stage2_contract=cfg)
        names=sorted(bindings(w))
        for role,seed,steps,records,run_ids in (("FIT",2025,SOURCE_STEPS,fits["fit"],range(1,21)),
                                         ("SELECT",2026,32,fits["select"],range(61,66))):
            counts=sample_use(records,c,seed,steps)
            for group,identities in enumerate((list(run_ids),names)):
                for i,rec in enumerate(identities):
                    sources.append(dict(domain=d,role=role,source="TEP" if group==0 else "SMD",
                        record=rec,presentations=counts[group][i],channels=c))
        for arm in ("source","N","ordinary","random_u","replay"):
            folder=root/d/arm
            rc=read(folder/"receipt.json")
            assert rc["sha256"]==sha(folder/"selected.pt")
            if arm in ("N","ordinary","replay"):
                assert rc["parent"]==s["sha256"]
            if arm=="N":
                assert rc["selected"]==min(rc["curve"],key=lambda r:(r["loss"],r["epoch"]))["epoch"]
            fields=("domain","arm","seconds","peak_bytes","sha256","selected","selected_step","model_step","updates",
                    "source_exposure","target_exposure","warmup","ramp","full_strength_updates","active_updates")
            rows.append({k:rc.get(k) for k in fields})
            if arm=="N":
                checks[d]["normal_validation"]=rc["curve"]
        rows.append(dict(domain=d,arm="scratch_N",seconds=scratch["train_seconds"],selected=scratch["selected"]["epoch"],
            updates=scratch["updates"],target_exposure=len(scratch["validation"])*(cfg["FIT"][1]-WINDOW+1),
            sha256=score["checkpoint_sha256"]))
        for path in sorted((root/d/"banks").glob("*.json")):
            b=read(path)
            assert sha(path.with_suffix(".npz"))==b["sha256"]
            manifests.append(dict(domain=d,file=str(path.with_suffix(".npz").relative_to(w)),**b))
        for role in ("CAL","QUERY"):
            with np.load(root/d/"banks"/("source_"+role+".npz")) as bank:
                with np.load(root/d/"banks/source_TEST.npz") as public:
                    idx=bank["index"]-public["index"][0]
                    a,b=bank["score"],public["score"][idx]
                    checks[d][role+"_Z_max_abs_diff_vs_full_test"]=float(np.max(np.abs(a-b)))
                    np.testing.assert_allclose(a,b,rtol=2e-4,atol=2e-5)
    save_rows(dest/"training.csv",rows)
    save_rows(dest/"source_exposure.csv",sources)
    dump(dest/"checks.json",checks)
    dump(dest/"banks.json",manifests)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("workspace",type=Path)
    collect(p.parse_args().workspace)
