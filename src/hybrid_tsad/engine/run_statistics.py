"""TEP fixed macro estimand and paired across-fault run-cluster sensitivity CI."""
import numpy as np

from .readout_manifest import METHODS,SEEDS

COMPARATORS=("temporal","late","history_mean")


def run_cluster_statistics(rows,profile):
    faults,runs=profile["faults"],profile["runs"]
    if not faults or not runs or len(set(faults))!=len(faults) or len(set(runs))!=len(runs):
        raise ValueError("Statistics require unique nonempty fault and run clusters")
    if profile["name"]=="TEP100" and (faults!=list(range(1,21)) or runs!=list(range(31,51))):
        raise ValueError("TEP macro/cluster grid must preserve all20 faults and all20 runs")
    expected={(m,s,f,r) for m in METHODS for s in SEEDS for f in faults for r in runs}
    found={}
    for row in rows:
        key=(row["method"],row["seed"],row["fault_id"],row["run_id"])
        if key not in expected or key in found:raise ValueError("Unknown/duplicate statistics cell")
        found[key]=row
    if set(found)!=expected:raise ValueError("Missing method/seed/fault/run statistics cells")
    arrays={metric:np.array([[[[found[m,s,f,r][metric] for r in runs] for f in faults] for s in SEEDS] for m in METHODS],dtype="float64")
            for metric in ("AP","ROC","positive_fraction")}
    for metric in ("AP","ROC"):
        values=arrays[metric]
        if np.any(np.isfinite(values)&((values<0)|(values>1))):raise ValueError("Entity AP/ROC lies outside [0,1]")
    background=arrays["positive_fraction"]
    if (not np.isfinite(background).all() or np.any(background<=0) or np.any(background>=1)
            or not np.array_equal(background,np.broadcast_to(background[0,0],background.shape))):
        raise ValueError("Compared methods/seeds must retain the same entity label population")
    if profile["name"]=="TEP100" and not np.all(background==800/860):
        raise ValueError("TEP100 AP background must remain800/860")
    macro=[];entity_seed_mean=[];per_fault=[];per_run=[];worst=[]
    for mi,method in enumerate(METHODS):
        for metric in ("AP","ROC"):
            a=arrays[metric][mi]
            entity=a.mean(0)  # first average the same entity over all3 seeds
            seed_macro=a.mean((1,2))
            macro.append(dict(method=method,metric=metric,mean=float(entity.mean()),sample_std=float(seed_macro.std(ddof=1)),
                              per_seed=seed_macro.tolist(),n_seeds=3,independent_datasets=1,
                              estimand="equal entity macro after same-entity three-seed mean"))
            for fi,f in enumerate(faults):
                for ri,r in enumerate(runs):entity_seed_mean.append(dict(method=method,metric=metric,fault_id=f,run_id=r,mean=float(entity[fi,ri])))
                per_fault.append(dict(method=method,metric=metric,fault_id=f,mean=float(entity[fi].mean())))
            for ri,r in enumerate(runs):per_run.append(dict(method=method,metric=metric,run_id=r,mean=float(entity[:,ri].mean())))
            finite=np.isfinite(entity)
            worst_entity=np.unravel_index(np.where(finite,entity,np.inf).argmin(),entity.shape) if finite.any() else None
            fault_mean,run_mean=entity.mean(1),entity.mean(0)
            worst.append(dict(method=method,metric=metric,worst_entity=float(entity[finite].min()) if finite.any() else float("nan"),
                              worst_seed=float(seed_macro.min()),worst_fault=float(fault_mean.min()),worst_run=float(run_mean.min()),
                              worst_entity_fault_id=faults[worst_entity[0]] if worst_entity is not None else None,
                              worst_entity_run_id=runs[worst_entity[1]] if worst_entity is not None else None,
                              worst_seed_id=int(seed_macro.argmin()) if np.isfinite(seed_macro).all() else None,
                              worst_fault_id=faults[fault_mean.argmin()] if np.isfinite(fault_mean).all() else None,
                              worst_run_id=runs[run_mean.argmin()] if np.isfinite(run_mean).all() else None,
                              invalid_entities=int((~finite).sum()),reason=None if finite.all() else "NONFINITE_ENTITIES_RETAINED_IN_FAILURE_TABLE",
                              tie_rule="first frozen-order minimum; full entity/seed tables retain ties"))
    ap=arrays["AP"]
    valid_run=np.isfinite(ap).all((0,1,2)) & np.isfinite(arrays["ROC"]).all((0,1,2))
    differences=np.stack([(ap[METHODS.index("hybrid")]-ap[METHODS.index(c)]).mean(0).mean(0) for c in COMPARATORS],axis=1)
    differences[~valid_run,:]=np.nan
    rng=np.random.default_rng(99174)
    draws=rng.integers(0,len(runs),size=(2000,len(runs)))
    deltas=np.full((2000,len(COMPARATORS)),np.nan)
    for i,draw in enumerate(draws):
        if not valid_run[draw].all():continue  # common NA; never redraw a preferred replicate
        deltas[i]=differences[draw].mean(0)
    intervals=[]
    for ci,comp in enumerate(COMPARATORS):
        valid=np.isfinite(deltas[:,ci]);observed=float(differences[:,ci].mean())
        lo,hi=np.quantile(deltas[valid,ci],[.025,.975]) if valid.any() and np.isfinite(observed) else (float("nan"),float("nan"))
        intervals.append(dict(comparison="hybrid-"+comp,observed_mean_AP_difference=observed,ci95_low=float(lo),ci95_high=float(hi),
                              effect_size_type="absolute paired macro AP difference",
                              replicates=2000,valid_replicates=int(valid.sum()),invalid_replicates=int((~valid).sum()),
                              cluster_unit="run_id across all faults and seeds",clusters=len(runs),rng_seed=99174,
                              scope="exploratory paired run-cluster sensitivity; no multiplicity adjustment"))
    failures=[{**{k:v for k,v in row.items() if k in ("method","seed","fault_id","run_id","AP","ROC")},
               "reason":"NONFINITE_ENTITY_AP_OR_ROC"} for row in rows if not np.isfinite(row["AP"]) or not np.isfinite(row["ROC"])]
    return dict(status="COMPLETE" if not failures else "INCOMPLETE",macro=macro,entity_seed_mean=entity_seed_mean,
                per_fault=per_fault,per_run=per_run,worst=worst,failures=failures,
                positive_fraction=float(arrays["positive_fraction"].mean()),intervals=intervals,
                overall_Aff_F1=None,overall_Aff_reason="NO_OFFICIAL_GLOBAL_DEFINITION_ACROSS_ENTITIES",
                bootstrap_arrays={"draws":draws,"AP_difference":deltas,"run_ids":np.asarray(runs),"paired_run_difference":differences,
                                  "valid_run":valid_run,"comparisons":np.asarray(["hybrid-"+c for c in COMPARATORS]),"rng_seed":np.asarray(99174)})
