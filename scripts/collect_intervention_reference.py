"""Collect provenance and fit diagnostics while leaving all native score banks on the compute host."""
import hashlib
import json
import platform
import sys
from pathlib import Path
from statistics import mean

from hybrid_tsad.trajectory_experiment import dump,sha,standardizer,target,ROLES,TEP
from hybrid_tsad.trajectory_evaluation import save_rows
from hybrid_tsad.intervention_reference import ROOT


def collect(w):
    root=w/ROOT
    dest=root/"summary"
    sources = {"tep_array_sha256":sha(w/TEP),"smd_fit":"first 80 percent per training record",
               "smd_select":"remaining 20 percent","tep_fit_runs":list(range(1,21)),
               "tep_select_runs":list(range(61,66)),"target_fit_scalers":{}}
    for d in ("PSM","SWAT"):
        mu,sd=standardizer(target(w,d,"FIT"))
        sources["target_fit_scalers"][d]=dict(roles=ROLES[d],mean=mu.tolist(),std=sd.tolist(),
            digest=hashlib.sha256(mu.tobytes()+sd.tobytes()).hexdigest())
    dump(dest/"inputs.json",sources)
    seals=[]
    traces=[]
    validation=[]
    for path in sorted(root.rglob("receipt.json")):
        receipt=json.loads(path.read_text())
        assert sha(path.parent/"selected.pt")==receipt["checkpoint_sha256"]
        seals.append(receipt)
        trace=json.loads((path.parent/"trace.json").read_text())
        for row in trace["training"]:
            traces.append(dict(domain=receipt["domain"],arm=receipt["arm"],**row))
        for row in trace["source_select"]:
            validation.append(dict(arm=receipt["arm"],**row))
        if receipt["domain"]=="source":
            assert min(trace["source_select"],key=lambda r:(r["risk"],r["step"]))["step"]==receipt["selected"]
    assert len(seals)==8
    save_rows(dest/"weights.csv",seals)
    save_rows(dest/"learning.csv",traces)
    save_rows(dest/"source_select.csv",validation)
    banks=[]
    for path in sorted(root.glob("*/banks/*.json")):
        metadata=json.loads(path.read_text())
        assert metadata["sha256"]==sha(path.with_suffix(".npz"))
        banks.append(dict(file=str(path.with_suffix(".npz").relative_to(w)),**metadata))
    assert len(banks)==6
    dump(dest/"banks.json",banks)
    diagnostics=[]
    shifts={}
    for d in ("source","PSM","SWAT"):
        obj=json.loads((root/d/"paired_check.json").read_text())
        shifts[d]=dict(**obj["shift_check"],seconds=obj["seconds"])
        for arm,kind in sorted({(r["arm"],r["kind"]) for r in obj["rows"]}):
            group=[r for r in obj["rows"] if r["arm"]==arm and r["kind"]==kind]
            diagnostics.append(dict(domain=d,arm=arm,kind=kind,n=len(group),
                unmodified=sum(r["input_rms"]==0 for r in group),
                **{k:mean(r[k] for r in group) for k in
                   ("mean_V","mean_edited_V","delta","changed_fraction","input_rms")}))
    save_rows(dest/"edits.csv",diagnostics)
    import torch
    dump(dest/"checks.json",dict(shift_checks=shifts,python=platform.python_version(),
         torch=torch.__version__,cuda=torch.version.cuda,all_eight_weight_hashes_verified=True,
         source_selection_rechecked=True,all_six_banks_verified=True))


if __name__=="__main__":
    collect(Path(sys.argv[1]))
