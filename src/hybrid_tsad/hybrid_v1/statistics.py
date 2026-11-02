"""Fixed paired AP time-block sensitivity, independent of model/evaluation code.

Original-grid blocks4096, exactly2000 draws, default_rng99173. This is
conditional sensitivity on one developed run, not independent-domain evidence.
No event/Aff bootstrap, file input, training, score ensemble or threshold fit.
"""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

BLOCK_POINTS = 4096
REPLICATES = 2000
RNG_SEED = 99173
DEFAULT_MODELS = ("v1_seed2025", "v1_seed2026", "v1_seed2027")
DEFAULT_BASELINES = ("persistence", "history_mean512")
NA_CODES = {0:"VALID_TWO_CLASS_DRAW",1:"NO_COMMON_POINTS_IN_DRAW",2:"SINGLE_CLASS_DRAW"}


def _array_identity(array):
    a=np.asarray(array)
    return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def prepare_paired_inputs(methods, full_labels, *, original_length, original_start=0,
                         model_names=DEFAULT_MODELS, baseline_names=DEFAULT_BASELINES):
    """Validate full original label axis and keep all-method valid intersection.

    Each method carries index, score, valid, and optional invalid_reasons. Raw
    invalid rows and original-axis absent rows remain distinct exclusion reasons.
    Returned arrays are copies; no caller input is mutated or compressed in time.
    """
    models,baselines=tuple(model_names),tuple(baseline_names); names=models+baselines
    if (len(models)!=3 or len(baselines)!=2 or len(set(names))!=5
            or any(not isinstance(n,str) or not n for n in names) or set(methods)!=set(names)):
        raise ValueError("Exactly three named seeds and two distinct deterministic baselines required")
    labels=np.asarray(full_labels)
    if (type(original_length) is not int or original_length<1 or labels.shape!=(original_length,)
            or labels.dtype.kind not in "biuf" or not np.isfinite(labels).all()
            or not np.isin(labels,[0,1]).all() or type(original_start) is not int or original_start<0):
        raise ValueError("Complete binary original-run labels/length and explicit start required")
    masks=[]; scores=[]; normalized={}; per_method={}; reason_sets=[]
    for name in names:
        row=methods[name]; index=np.asarray(row["index"]); score=np.asarray(row["score"],dtype=np.float64)
        valid=np.asarray(row["valid"])
        if (index.ndim!=1 or not np.issubdtype(index.dtype,np.integer)
                or score.shape!=index.shape or valid.shape!=index.shape or valid.dtype!=np.bool_
                or np.any(index[1:]<=index[:-1]) or np.any(index<original_start)
                or np.any(index>=original_start+original_length) or np.any(valid & ~np.isfinite(score))):
            raise ValueError("Unique ordered original indices, matching bool valid and finite valid scores required: "+name)
        reasons=list(row.get("invalid_reasons",[None]*len(index)))
        if len(reasons)!=len(index): raise ValueError("Invalid-reason axis mismatch: "+name)
        for pos in np.flatnonzero(~valid):
            if not isinstance(reasons[pos],str) or not reasons[pos]:
                raise ValueError("Invalid raw row needs its original reason: "+name)
        local=index-original_start; mask=np.zeros(original_length,bool); mask[local[valid]]=True
        values=np.full(original_length,np.nan); values[local]=score
        reason=["ABSENT_ORIGINAL_SCORE_INDEX"]*original_length
        for pos,i in enumerate(local):reason[i]="VALID" if valid[pos] else reasons[pos]
        normalized[name]=dict(index=index.copy(),score=score.copy(),valid=valid.copy(),invalid_reasons=reasons)
        masks.append(mask);scores.append(values);reason_sets.append(reason)
        unique_reason_counts={}
        for item in reason:unique_reason_counts[item]=unique_reason_counts.get(item,0)+1
        per_method[name]=dict(raw_records=len(index),valid_points=int(mask.sum()),
            invalid_raw_records=int((~valid).sum()),absent_original_points=original_length-len(index),
            exclusion_reason_counts=unique_reason_counts,
            input_identity={k:_array_identity(normalized[name][k]) for k in ("index","score","valid")})
    common=np.logical_and.reduce(masks); original_index=original_start+np.arange(original_length,dtype=np.int64)
    common_index=original_index[common]; common_labels=labels[common].astype(np.int8)
    common_block=(common_index-original_start)//BLOCK_POINTS
    nblocks=(original_length+BLOCK_POINTS-1)//BLOCK_POINTS
    bounds=np.array([[original_start+b*BLOCK_POINTS,
                      min(original_start+(b+1)*BLOCK_POINTS,original_start+original_length)]
                     for b in range(nblocks)],dtype=np.int64)
    patterns={}
    for i in np.flatnonzero(~common):
        pattern=" | ".join(name+":"+reason_sets[j][i] for j,name in enumerate(names) if not masks[j][i])
        patterns[pattern]=patterns.get(pattern,0)+1
    return dict(names=names,models=models,baselines=baselines,full_labels=labels.copy(),
        normalized_inputs=normalized,common_index=common_index,common_labels=common_labels,
        common_scores=np.stack([s[common] for s in scores],axis=1),common_block_id=common_block,
        block_bounds=bounds,per_block_common_count=np.bincount(common_block,minlength=nblocks),
        coverage=dict(original_length=original_length,original_start=original_start,
            common_points=int(common.sum()),common_positive_points=int(common_labels.sum()),
            common_negative_points=int((common_labels==0).sum()),excluded_original_points=int((~common).sum()),
            per_method=per_method,exclusion_patterns=patterns,
            common_axis="original indices retained; no event stream is concatenated",
            sampling_population="all original-grid blocks, including zero-common-point blocks",
            common_intersection_only=True,does_not_replace_method_full_coverage_metrics=True))


def fixed_block_draws(nblocks):
    if type(nblocks) is not int or nblocks<1: raise ValueError("Positive original block count required")
    return np.random.default_rng(RNG_SEED).integers(0,nblocks,size=(REPLICATES,nblocks),dtype=np.int64)


def point_weights_from_draw(block_id, draw, nblocks):
    """One shared block multiplicity vector, applied identically to ALL methods."""
    block_id=np.asarray(block_id);draw=np.asarray(draw)
    if (block_id.ndim!=1 or draw.shape!=(nblocks,) or draw.dtype.kind not in "iu"
            or block_id.dtype.kind not in "iu" or np.any(draw<0) or np.any(draw>=nblocks)
            or np.any(block_id<0) or np.any(block_id>=nblocks)):
        raise ValueError("Original block IDs/draw axis mismatch")
    return np.bincount(draw,minlength=nblocks)[block_id]


def _interval(values):
    values=np.asarray(values); finite=values[np.isfinite(values)]
    return dict(lower=float(np.quantile(finite,.025)) if len(finite) else None,
                upper=float(np.quantile(finite,.975)) if len(finite) else None,
                valid_replicates=len(finite),NA_replicates=REPLICATES-len(finite),
                rule="empirical2.5/97.5percentiles_linear_on_finite_draws; no redraw",
                NA_reason=None if len(finite) else "NO_VALID_TWO_CLASS_DRAWS")


def paired_ap_bootstrap(methods, full_labels, *, original_length, original_start=0,
                        model_names=DEFAULT_MODELS, baseline_names=DEFAULT_BASELINES):
    """Compute exactly2000 paired original-block draws using actual sklearn AP.

    Integer sample_weight equals duplicating every sampled original block. One
    draw is reused across all five methods, and no score average creates an
    ensemble. Single-class/empty draws remain NaN in their original draw row.
    """
    from sklearn.metrics import average_precision_score
    import sklearn
    p=prepare_paired_inputs(methods,full_labels,original_length=original_length,
        original_start=original_start,model_names=model_names,baseline_names=baseline_names)
    names=p["names"];y=p["common_labels"];s=p["common_scores"];nblocks=len(p["block_bounds"])
    draws=fixed_block_draws(nblocks)
    original_AP=np.full(5,np.nan); original_reason=None
    if not len(y):original_reason="NO_COMMON_POINTS"
    elif len(np.unique(y))!=2:original_reason="SINGLE_CLASS_COMMON_INTERSECTION"
    else:
        original_AP=np.array([average_precision_score(y,s[:,i]) for i in range(5)],dtype=np.float64)
    replicate_AP=np.full((REPLICATES,5),np.nan);codes=np.zeros(REPLICATES,np.uint8)
    weighted_positive=np.zeros(REPLICATES,np.int64); weighted_negative=np.zeros(REPLICATES,np.int64)
    for rep,draw in enumerate(draws):
        weight=point_weights_from_draw(p["common_block_id"],draw,nblocks)
        positive=int(weight[y==1].sum());negative=int(weight[y==0].sum())
        weighted_positive[rep]=positive;weighted_negative[rep]=negative
        if positive+negative==0:codes[rep]=1;continue
        if not positive or not negative:codes[rep]=2;continue
        replicate_AP[rep]=[average_precision_score(y,s[:,i],sample_weight=weight) for i in range(5)]
    delta=replicate_AP[:,:3,None]-replicate_AP[:,None,3:]
    mean_delta=delta.mean(axis=1)
    original_delta=original_AP[:3,None]-original_AP[None,3:]
    original_mean_delta=original_delta.mean(axis=0)
    finite_original=np.isfinite(original_AP[:3]).all()
    ap_source=inspect.getsourcefile(inspect.unwrap(average_precision_score))
    summary=dict(parameters=dict(block_points=BLOCK_POINTS,n_replicates=REPLICATES,rng_seed=RNG_SEED),
        status="COMPUTED_ON_SUPPLIED_ARRAYS",names=list(names),models=list(p["models"]),baselines=list(p["baselines"]),
        coverage=p["coverage"],original_AP={n:float(a) if np.isfinite(a) else None for n,a in zip(names,original_AP)},
        original_AP_NA_reason=original_reason,
        model_seed_AP=dict(mean=float(original_AP[:3].mean()) if finite_original else None,
            sample_SD=float(original_AP[:3].std(ddof=1)) if finite_original else None,
            minimum=float(original_AP[:3].min()) if finite_original else None,
            seed_variation="fixed trained banks; not pure optimization variance or new independent domains"),
        original_paired_AP_difference={b:dict(per_seed={m:float(original_delta[i,j]) if finite_original else None
                for i,m in enumerate(p["models"])},mean=float(original_mean_delta[j]) if finite_original else None)
                for j,b in enumerate(p["baselines"])},
        AP_intervals={n:_interval(replicate_AP[:,i]) for i,n in enumerate(names)},
        paired_difference_intervals={b:dict(per_seed={m:_interval(delta[:,i,j]) for i,m in enumerate(p["models"])},
                                          mean=_interval(mean_delta[:,j])) for j,b in enumerate(p["baselines"])},
        draw_counts={NA_CODES[c]:int((codes==c).sum()) for c in NA_CODES},
        shared_draws_for_all_methods=True,redraw_single_class=False,score_ensemble=False,
        interval_scope="conditional time-block sensitivity on one historically developed PSM run; not independent-domain significance",
        not_computed=["Aff_or_event_bootstrap","independent_domain_CI","method_full_coverage_metrics","model_reruns"],
        sklearn=dict(version=sklearn.__version__,AP_callable="sklearn.metrics.average_precision_score",
                     source_file=ap_source,source_sha256=hashlib.sha256(Path(ap_source).read_bytes()).hexdigest() if ap_source else None))
    return dict(summary=summary,plan=p,arrays=dict(block_draws=draws,block_bounds=p["block_bounds"],
        per_block_common_count=p["per_block_common_count"],common_index=p["common_index"],
        common_labels=y,common_block_id=p["common_block_id"],common_scores=s,
        original_AP=original_AP,original_paired_difference=original_delta,
        original_mean_difference=original_mean_delta,replicate_AP=replicate_AP,
        replicate_paired_difference=delta,replicate_mean_difference=mean_delta,
        NA_reason_code=codes,weighted_positive_points=weighted_positive,weighted_negative_points=weighted_negative))


def save_statistics(directory, result, *, provenance):
    """Save exact original inputs, shared draws/replicate values and code identity."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    p=result["plan"];inputs={"full_labels":p["full_labels"]}
    for number,name in enumerate(p["names"]):
        row=p["normalized_inputs"][name]
        for key in ("index","score","valid"):inputs["m%d_%s"%(number,key)]=row[key]
        reasons=row["invalid_reasons"]
        inputs["m%d_reasons"%number]=np.asarray(["" if r is None else r for r in reasons],str)
        inputs["m%d_reason_is_none"%number]=np.asarray([r is None for r in reasons],bool)
    np.savez_compressed(directory/"calculation_inputs.npz",**inputs)
    np.savez_compressed(directory/"bootstrap_arrays.npz",**result["arrays"])
    receipt=dict(provenance=provenance,names=list(p["names"]),NA_reason_codes=NA_CODES,
        input_identity={k:_array_identity(v) for k,v in inputs.items()},
        array_file_SHA256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                          for name in ("calculation_inputs.npz","bootstrap_arrays.npz")},
        implementation=dict(path=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        parameters=result["summary"]["parameters"])
    (directory/"summary.json").write_text(json.dumps(result["summary"],indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (directory/"provenance.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return receipt
