"""Count realized training windows at the selected checkpoint and at the update cap."""
from collections import Counter
from pathlib import Path
import numpy as np
from . import native_event_rank_training as old
from .native_position_aligned_training import PRIOR
from .native_tep_support import ARMS, OUT
from .native_v_conditioning_runner import read,dump


def exposure(w):
    root=w/OUT
    plan=old.load_plan(w/PRIOR)
    with np.load(root/"SUPPORT_PLAN.npz") as f:
        draws=dict(f)
    results=[]
    for arm in ARMS:
        selected=read(root/arm/"SELECTION.json")["selected_step"]
        for steps in sorted({selected,240}):
            observed={k:Counter() for k in ("copy","positive","paired_normal")}
            def count(role,domain,record,end):
                observed[role][domain,tuple(record),int(end)]+=1
            for i in range(steps):
                counts=np.bincount(draws["copy_domain"][i],minlength=3)
                for end in plan["swat_fit"][i,:counts[0]]:
                    count("copy","SWaT",("nominal",),end)
                for k in range(counts[1]):
                    count("copy","SMD",(int(plan["smd_entity"][i,k]),),plan["smd_fit"][i,k])
                for k in range(counts[2]):
                    count("copy","TEP",(0,int(draws["normal_run"][i,k])+1),draws["normal_end"][i,k])
                for k in range(15):
                    if arm=="single_V" or draws["shared"][i,k]:
                        count("positive","SWaT",(k//3,),plan["positive_fit"][i,k])
                        count("paired_normal","SWaT",("event_source",),plan["normal_fit"][i,k])
                    else:
                        f,r,p,nr,n=map(int,draws["tep_pair"][i,k])
                        count("positive","TEP",(f+1,r+1),p)
                        count("paired_normal","TEP",(0,nr+1),n)
            for role,windows in observed.items():
                for d,c in (("SWaT",31),("SMD",38),("TEP",52)):
                    subset={k:v for k,v in windows.items() if k[0]==d}
                    exposures=sum(subset.values())
                    results.append(dict(arm=arm,steps=steps,selected=steps==selected,role=role,domain=d,
                        presentations=exposures,unique_windows=len(subset),channels=c,
                        channel_window_exposures=c*exposures,records_or_events=len({k[1] for k in subset})))
            assert sum(observed["copy"].values())==32*steps
            assert sum(observed["positive"].values())==sum(observed["paired_normal"].values())==15*steps
    dump(root/"EXPOSURES.json",results)


if __name__=="__main__":
    exposure(Path.cwd())
