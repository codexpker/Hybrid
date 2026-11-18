"""Construct label-free numeric replacement bundles from the current U-FIT.

Only this construction command opens event labels. Learners receive numerical
window banks and a slot-to-row map; donor starts/event IDs stay in audit_only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hybrid_tsad.left_domain_replay import plan_arrays
from hybrid_tsad.left_swat_author import read,sha,dump

ENTROPY=[2025,20261118,10]


def legal_windows(labels,width=192):
    cs=np.r_[0,np.cumsum(labels,dtype=np.int64)]
    counts=cs[width:]-cs[:-width]
    return np.flatnonzero(counts==width),np.flatnonzero(counts==0),counts


def draw_replacements(b_legal,c_legal,seed=2025):
    entropy=[int(seed),*ENTROPY[1:]]
    streams=np.random.SeedSequence(entropy).spawn(3)
    selected=np.sort(np.random.default_rng(streams[0]).choice(256000,25600,replace=False))
    rows=np.full(256000,-1,np.int32);rows[selected]=np.arange(25600,dtype=np.int32)
    picked={condition:legal[np.random.default_rng(stream).integers(0,len(legal),size=25600)]
            for condition,legal,stream in [("B",b_legal,streams[1]),("C",c_legal,streams[2])]}
    return selected,rows.reshape(2000,128),picked


def construct(workspace,task_path,base_plan,out,seed=2025):
    out.mkdir(parents=True,exist_ok=False);audit=out/"audit_only";audit.mkdir()
    task=read(task_path);domain=task["domain"];w=task["scoring"]["window"]
    assert w==192
    labelpath=workspace/task["labels"]["relative"];assert sha(labelpath)==task["labels"]["sha256"]
    lo,hi=task["roles"]["U_FIT"]
    if labelpath.suffix==".npz":
        with np.load(labelpath)as z:y=z["label"][lo:hi].astype(bool).copy()
    else:y=np.load(labelpath,mmap_mode="r")[lo:hi].astype(bool).copy().reshape(-1)
    assert len(y)==hi-lo
    b_legal,c_legal,counts=legal_windows(y,w)
    edges=np.diff(np.r_[False,y,False].astype(np.int8));events=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    eligible=np.flatnonzero(ends-events>=w)
    qualification=dict(domain=domain,U_FIT=[lo,hi],points=len(y),anomaly_labeled_points=int(y.sum()),
        full_anomaly_windows=len(b_legal),full_normal_windows=len(c_legal),
        eligible_B_event_runs=len(eligible),total_UFIT_event_runs=len(events),
        event_boundaries_at_U_edges_censored=bool(y[0]or y[-1]),
        donor_selection="uniform legal stride-one starts, with replacement; no model scores; no extra distance filter",
        labels_only_constructor=True,channels_physically_affected="UNKNOWN")
    dump(audit/"QUALIFICATION.json",qualification)
    if not len(b_legal)or not len(c_legal):
        dump(out/"STATUS.json",dict(status="NA",reason="No legal full-192 anomaly or normal donors",qualification=qualification))
        return qualification
    with np.load(base_plan)as z:plan={k:z[k].copy()for k in z.files}
    expected=plan_arrays(task,seed)
    assert set(plan)==set(expected)and all(np.array_equal(plan[k],expected[k])for k in plan)
    selected,rows,picked_by_condition=draw_replacements(b_legal,c_legal,seed)
    learner=out/"learner";learner.mkdir()
    np.savez_compressed(learner/"SLOT_MAP.npz",replacement_row=rows)
    source=workspace/task["data"]["target_test"]["relative"]
    assert sha(source)==task["data"]["target_test"]["sha256"]
    raw=np.load(source,mmap_mode="r");rawfit=np.array(raw[lo:hi],dtype=np.float64,order="F",copy=True)
    sf=workspace/task["source"]["scaler"]["relative"];assert sha(sf)==task["source"]["scaler"]["sha256"]
    if task["source"]["scaler"]["format"]=="npz":
        with np.load(sf)as z:mean=z["mean"].copy();scale=z["scale"].copy()
    else:st=read(sf)["statistics"];mean=np.asarray(st["mean"]);scale=np.asarray(st["scale"])
    normalized=(rawfit-mean)/scale;c=rawfit.shape[1]
    exposure={};selected_T_starts=plan["T_starts"].reshape(-1)[selected]
    T_count=counts[plan["T_starts"]];Tp_count=counts[plan["T_prime_starts"]]
    replaced_original=T_count.reshape(-1)[selected]
    base_time=int(T_count.sum());prime_time=int(Tp_count.sum());removed_time=int(replaced_original.sum())
    for condition in ["B","C"]:
        picked=picked_by_condition[condition]
        bankpath=learner/(condition+"_WINDOWS.npy")
        bank=np.lib.format.open_memmap(bankpath,mode="w+",dtype=np.float32,shape=(25600,w,c))
        changed=0
        for at in range(0,25600,256):
            starts=picked[at:at+256];index=starts[:,None]+np.arange(w)
            values=np.float32(normalized[index]);assert np.isfinite(values).all();bank[at:at+len(starts)]=values
            original=np.float32(normalized[selected_T_starts[at:at+len(starts),None]+np.arange(w)])
            changed+=int(np.count_nonzero(values!=original))
        bank.flush();del bank
        uniq,repeats=np.unique(picked,return_counts=True)
        event_counts=None
        event_id=np.full(25600,-1,np.int64)
        if condition=="B":
            event_id=np.searchsorted(events,picked,side="right")-1
            assert np.all(picked+w<=ends[event_id])
            event_counts=np.bincount(event_id,minlength=len(events)).tolist()
        np.savez_compressed(audit/(condition+"_DONORS.npz"),donor_start_U_relative=picked,
            donor_start_target_absolute=picked+lo,event_id=event_id,unique_starts=uniq,repeats=repeats,
            selected_T_flat_slots=selected,original_T_starts=selected_T_starts,
            original_slot_anomaly_labeled_points=replaced_original)
        donor_time=25600*w if condition=="B"else 0
        after_T=base_time-removed_time+donor_time
        exposure[condition]=dict(unique_donor_windows=len(uniq),maximum_repeat_count=int(repeats.max()),
            repeated_presentations=25600-len(uniq),event_presentations=event_counts,
            distinct_B_events_presented=int(np.unique(event_id).size)if condition=="B"else None,
            real_input_float32_changed_scalars=changed,real_input_changed_scalar_fraction_full_batch=changed/(512000*w*c),
            anomaly_labeled_time_exposure=dict(T_after=after_T,TT_after=after_T+prime_time,R_after=after_T),
            TT_event_time_fraction=(after_T+prime_time)/(512000*w),R_event_time_fraction=after_T/(512000*w))
        manifest=dict(format="numeric-only-normalized-window-bundle-v1",domain=domain,
            slot_map={"relative":str((learner/"SLOT_MAP.npz").relative_to(workspace)),"sha256":sha(learner/"SLOT_MAP.npz")},
            window_bank={"relative":str(bankpath.relative_to(workspace)),"sha256":sha(bankpath),
                         "shape":[25600,w,c],"dtype":"float32"},
            scaler_sha256=sha(sf),base_sampling_plan_sha256=sha(base_plan),
            updates=2000,half_batch=128,replacement_rows=25600,source_units="frozen author scaler; do not standardize again",
            donor_labels_or_identity_in_bundle=False,sampling_seed=int(seed))
        dump(learner/(condition+"_MANIFEST.json"),manifest)
    np.savez_compressed(audit/"TIME_LABEL_AUDIT.npz",U_FIT_time_labels=y,event_start_U_relative=events,
        event_end_U_relative=ends,T_window_anomaly_time_count=T_count,T_prime_window_anomaly_time_count=Tp_count,
        selected_T_flat_slots=selected)
    report=dict(qualification=qualification,seed=int(seed),RNG_entropy=[int(seed),*ENTROPY[1:]],spawn_keys={"slots":[0],"B":[1],"C":[2]},
        shared_selected_T_slots=25600,T_exposures=256000,full_batch_exposures=512000,
        nominal_T_replacement_fraction=.1,nominal_full_batch_replacement_fraction=.05,
        baseline_anomaly_labeled_time_exposure=dict(T=base_time,T_prime=prime_time,
            replaced_original_T=removed_time,T_total_time=256000*w,full_batch_total_time=512000*w,
            R_A=base_time,TT_A=base_time+prime_time),conditions=exposure,
        scalar_event_label_copies_are_nominal_not_physical_channel_truth=True,
        effective_supervision=dict(real_full_windows=512000,real_observed_time_positions=512000*w,
            real_observed_scalar_positions=512000*w*c,padding_positions=0,
            exact_anomalous_gradient_contribution="UNKNOWN: native frequency/multiscale losses and prototype eligibility mix time evidence",
            source_S_time_label_exposure="zero under original author normal-history role, not certified physical channel truth"),
        audit_only_not_read_by_learner=True,base_sampling_plan_sha256=sha(base_plan),
        slot_map_sha256=sha(learner/"SLOT_MAP.npz"))
    dump(audit/"EXPOSURE_AUDIT.json",report)
    dump(out/"STATUS.json",dict(status="QUALIFIED",qualification=qualification,
        numeric_manifests={k:str((learner/(k+"_MANIFEST.json")).relative_to(workspace))for k in ["B","C"]},
        exposure_audit_sha256=sha(audit/"EXPOSURE_AUDIT.json")))
    print(json.dumps(dict(event="QUALIFIED",qualification=qualification,conditions=exposure)),flush=True)
    return qualification


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--task",type=Path,required=True);p.add_argument("--base-plan",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=2025);a=p.parse_args()
    construct(a.workspace,a.task,a.base_plan,a.output,a.seed)
