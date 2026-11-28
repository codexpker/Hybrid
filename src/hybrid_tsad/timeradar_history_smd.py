"""Complete SMD-held-out TimeRadar history fold, using pinned normal TEP/HAI."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .left_decoder_context_metrics import AUTHOR
from .left_native_swat_metrics import load_author_metric
from .left_smd28_author import OLD, bindings, checked_data
from .left_smd_reliability_eval import ranked
from .left_smd_source import OUT as HAI_RUN, TEP_DIGEST, source_pools
from .left_smd_targets import normal_scaler, roles
from .native_v_conditioning import WEIGHT_SHA, sha
from .timeradar_history_context import record_window
from .timeradar_history_experiment import FROZEN_HISTORY, SEED, fit, model, source_draws
from .timeradar_history_f import intervals, numeric_intervals
from .trajectory_experiment import dump


ARMS = ("native", "history", FROZEN_HISTORY)
Z_ARMS = ARMS[:2]
METRICS = ("AP", "ROC_AUC", "VUS_PR", "VUS_ROC")
LABEL_MANIFEST = "aee4c628d138cc2355535fb5e0bf24fa103d88fd44a50953b8ed2d4e2a05132b"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def prepare(workspace, out):
    out.mkdir(parents=True, exist_ok=False)
    pools = source_pools(workspace)
    source = {}
    for role, groups in pools.items():
        source[role] = [
            [
                dict(shape=list(array.shape), numeric_sha256=array_hash(array), history=h,
                     legal_windows=len(array)-h-99)
                for array in group
            ]
            for group, h, expected_width in zip(groups, (256, 512), (52, 79))
            if all(a.ndim == 2 and a.shape[1] == expected_width and len(a) >= h+100
                   and np.isfinite(a).all() for a in group)
        ]
        assert len(source[role]) == 2
    rng = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(31)]
    plan = dict(
        source=np.asarray([source_draws(pools["fit"], rng[0], 32) for _ in range(240)]),
        select=np.asarray(source_draws(pools["select"], rng[1], 32)),
        replay=np.asarray([source_draws(pools["fit"], rng[2], 16) for _ in range(2000)]),
    )
    entities = {}
    for slot, (name, rec) in enumerate(sorted(bindings(workspace).items())):
        rr = roles(rec)
        _, scaler = normal_scaler(workspace, rec, False)
        x = checked_data(workspace, rec, "test")
        assert x.shape[1] == 38 and rr["U"][0] == 0 and rr["U"][1] <= rr["QUERY"][0]
        query = [rr["QUERY"][0]+611, rr["QUERY"][1]]
        assert query[0] < query[1] and rr["U"][1] >= 612
        plan[name] = rng[3+slot].integers(512, rr["U"][1]-99, (2000, 16))
        entities[name] = dict(
            roles=rr, query_scored=query, channels=38,
            raw_target_test_sha256=rec["source"][1]["sha256"],
            raw_target_train_sha256=rec["source"][0]["sha256"],
            scaler=dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist()),
            target_window_plan_sha256=array_hash(plan[name]),
            U_numeric_sha256=array_hash(scaler.transform(x[slice(*rr["U"])]).astype(np.float32)),
            QUERY_numeric_sha256=array_hash(scaler.transform(x[slice(*rr["QUERY"])]).astype(np.float32)),
        )
    np.savez_compressed(out/"plans.npz", **plan)
    hai = read(workspace/HAI_RUN/"inputs/receipt.json")
    assert len(hai["features"]) == len(set(hai["features"])) == 79
    dump(out/"contract.json", dict(
        source_datasets=["TEP FaultFree", "HAI 21.03 normal"],
        source_arrays=source, TEP_sha256=TEP_DIGEST, HAI=hai,
        source_channels="full ordered 52 TEP and 79 HAI process channels",
        SMD_from_project_source_excluded=True, source_labels_used=False,
        seed=SEED, mask_seed=84551, source_steps=240, source_warmup=24,
        source_select_steps=list(range(0,241,40)), target_steps=2000, target_warmup=200,
        logical_batch=32, target_slots=16, source_slots={"TEP":8,"HAI":8},
        optimizer="fresh AdamW lr1e-4 betas .9/.95 eps1e-8 decay .1 or 0 for bias/LayerNorm",
        parent_sha256=WEIGHT_SHA, plan_sha256=sha(out/"plans.npz"),
        entities=entities, history_length=dict(TEP=256,HAI=512,SMD=512),
        current_length=100, copies=10, target_labels_read=False,
        ranking="B1 norm0 native endpoint sample variance correction=1 mean across channels",
        Z_on_QUERY_only=True,
    ))
    print("PLAN_SEALED", len(entities), flush=True)


def load_fold(workspace, out):
    contract = read(out/"contract.json")
    assert sha(out/"plans.npz") == contract["plan_sha256"]
    assert contract["SMD_from_project_source_excluded"]
    assert contract["source_datasets"] == ["TEP FaultFree", "HAI 21.03 normal"]
    return source_pools(workspace), contract


def smd_values(workspace, contract, entity, role):
    rec = bindings(workspace)[entity]
    sealed = contract["entities"][entity]
    assert rec["source"][1]["sha256"] == sealed["raw_target_test_sha256"]
    assert rec["source"][0]["sha256"] == sealed["raw_target_train_sha256"]
    assert roles(rec) == sealed["roles"]
    _, scaler = normal_scaler(workspace, rec, False)
    assert scaler.mean_.tolist() == sealed["scaler"]["mean"]
    assert scaler.scale_.tolist() == sealed["scaler"]["scale"]
    lo, hi = sealed["roles"][role]
    x = scaler.transform(checked_data(workspace, rec, "test")[lo:hi]).astype(np.float32)
    digest = array_hash(x)
    assert digest == sealed[f"{role}_numeric_sha256"]
    return x, digest


def checkpoint(out, entity, arm, stage):
    return out/("source" if stage=="Z" else entity)/arm/"selected.pt"


@torch.no_grad()
def score(workspace, out, entity, arm, stage):
    started = time.monotonic()
    dest = out/entity/"scores"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/f"{stage}_{arm}.npz"
    ck = checkpoint(out, entity, arm, stage)
    ck_digest = sha(ck)
    assert ck_digest == read(ck.with_name("receipt.json"))["sha256"]
    if path.with_suffix(".json").is_file():
        cached = read(path.with_suffix(".json"))
        assert cached["bank_sha256"] == sha(path) and cached["checkpoint_sha256"] == ck_digest
        return
    contract = read(out/"contract.json")
    series, digest = smd_values(workspace, contract, entity, "QUERY")
    origin = contract["entities"][entity]["roles"]["QUERY"][0]
    lo, hi = contract["entities"][entity]["query_scored"]
    index = np.arange(lo, hi)
    assert index[0]-611 == origin and index[-1]-origin == len(series)-1
    values = torch.as_tensor(series, device="cuda")
    torch.set_num_threads(1)
    detector = model(workspace, arm)
    detector.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True)["model"])
    raw = []
    for tick in index:
        offset = int(tick)-origin
        history, current, meta = record_window(values, offset-99, (0,len(values)),512)
        copies = detector(history, current, **meta, bypass=arm=="native")[1]
        raw.append(copies.var(0, correction=1).mean(-1)[0,-1])
    scores = torch.stack(raw).cpu().numpy()
    assert np.isfinite(scores).all()
    footprint = np.column_stack((index-611, index-99, index+1))
    temp = path.with_name(path.stem+".partial.npz")
    np.savez_compressed(temp, index=index, available=index, score=scores,
                        historical_start=footprint[:,0], current_start=footprint[:,1])
    temp.replace(path)
    dump(path.with_suffix(".json"), dict(
        entity=entity, stage=stage, arm=arm, start=lo, stop=hi, points=len(index),
        bank_sha256=sha(path), checkpoint_sha256=ck_digest, QUERY_numeric_sha256=digest,
        footprint_sha256=array_hash(footprint), footprint_columns=["history_start","current_start","stop"],
        B=1, labels_read=False, seconds=time.monotonic()-started,
        peak_bytes=torch.cuda.max_memory_allocated(), GPU=torch.cuda.get_device_name(),
    ))
    print("SCORED", entity, stage, arm, len(index), flush=True)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def labels_for(workspace, name, manifest):
    rec = next(r for r in manifest["records"] if r["id"]==name)
    path = workspace/rec["path"]
    assert sha(path) == rec["sha256"]
    with np.load(path, allow_pickle=False) as z:
        y = z["label"].reshape(-1).astype(int)
        np.testing.assert_array_equal(z["index"],np.arange(len(y)))
    assert np.isin(y,[0,1]).all()
    return y


def verify_bank(out, spec, entity, arm, stage):
    path = out/entity/"scores"/f"{stage}_{arm}.npz"
    r = read(path.with_suffix(".json"))
    assert r["bank_sha256"]==sha(path) and not r["labels_read"]
    assert r["checkpoint_sha256"]==sha(checkpoint(out,entity,arm,stage))
    assert r["QUERY_numeric_sha256"]==spec["QUERY_numeric_sha256"] and r["B"]==1
    with np.load(path,allow_pickle=False) as data:
        idx = np.arange(*spec["query_scored"])
        np.testing.assert_array_equal(data["index"],idx)
        np.testing.assert_array_equal(data["available"],idx)
        np.testing.assert_array_equal(data["historical_start"],idx-611)
        np.testing.assert_array_equal(data["current_start"],idx-99)
        assert r["footprint_sha256"]==array_hash(np.column_stack((idx-611,idx-99,idx+1)))
        values=data["score"]
    assert values.shape==idx.shape and np.isfinite(values).all()
    return values, r


def evaluate(workspace, out, *, z_only=False):
    started = time.monotonic()
    contract = read(out/"contract.json")
    stage_arms = [("Z",arm) for arm in Z_ARMS]
    if not z_only:
        stage_arms += [("U",arm) for arm in ARMS]
    banks = {}
    manifest_rows=[]
    for entity,spec in contract["entities"].items():
        for stage,arm in stage_arms:
            values, receipt=verify_bank(out,spec,entity,arm,stage)
            banks[entity,stage,arm]=values
            manifest_rows.append(receipt)
            if stage=="U":
                meta=read(checkpoint(out,entity,arm,stage).with_name("receipt.json"))
                assert meta["steps"] == meta["selected_step"] == 2000 and meta["input_exposures"] == 64000
                assert meta["plan_sha256"] == contract["plan_sha256"]
                parent="history" if arm==FROZEN_HISTORY else arm
                assert meta["source_checkpoint_sha256"] == sha(checkpoint(out,entity,parent,"Z"))
                if arm==FROZEN_HISTORY:
                    audit=read(out/entity/arm/"parameter_audit.json")
                    assert audit["frozen_state_before"]==audit["frozen_state_after"]
                    assert audit["decoder_L2_displacement"]>0 and audit["history_L2_displacement"]==0
    write_rows(out/("Z_bank_manifest.csv" if z_only else "bank_manifest.csv"),manifest_rows)
    label_path=workspace/OLD/"evaluation_manifest_full28.json"
    assert sha(label_path)==LABEL_MANIFEST
    manifest=read(label_path)
    rows=[]; support=[]
    for entity,spec in contract["entities"].items():
        y=labels_for(workspace,entity,manifest)
        lo,hi=spec["query_scored"]
        assert len(y)==spec["roles"]["TEST"][1]
        if not z_only:
            support.append(f_support_eligibility(entity,spec,y))
        # Once Z has been evaluated, reuse its sealed same-axis scalars.
        for stage,arm in stage_arms:
            if stage=="Z" and not z_only:
                continue
            metrics, error=ranked(y[lo:hi],banks[entity,stage,arm],load_author_metric(workspace/AUTHOR))
            rows.append(dict(entity=entity,stage=stage,arm=arm,start=lo,stop=hi,
                positives=int(y[lo:hi].sum()),**metrics,independent_AP_ROC_max_error=error))
        print("EVALUATED",entity,"Z" if z_only else "U",flush=True)
    if z_only:
        write_rows(out/"Z_metrics.csv",rows)
        return
    with (out/"Z_metrics.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            for metric in METRICS+("independent_AP_ROC_max_error",):
                row[metric]=float(row[metric]) if row[metric] else None
            rows.append(row)
    summarize(out,rows)
    dump(out/"F_support_eligibility.json",dict(domains=support, normal_annotation_points_if_eligible=512,
        anomaly_budget="one complete event with three nonempty strata", supports_read_U_FIT_only=True,
        condition="offline qualification only", F_fits=0))
    training=[]
    for arm in Z_ARMS:
        training.append({"entity":"shared source","stage":"source",**read(out/"source"/arm/"receipt.json")})
    for entity in contract["entities"]:
        for arm in ARMS:
            training.append({"entity":entity,"stage":"U",**read(out/entity/arm/"receipt.json")})
    write_rows(out/"training_manifest.csv",training)
    dump(out/"cost.json",dict(
        source_fits=2,U_fits=84,Z_banks=56,U_banks=84,
        source_worker_hours=sum(r["seconds"] for r in training if r["stage"]=="source")/3600,
        U_worker_hours=sum(r["seconds"] for r in training if r["stage"]=="U")/3600,
        Z_scoring_worker_hours=sum(r["seconds"] for r in manifest_rows if r["stage"]=="Z")/3600,
        U_scoring_worker_hours=sum(r["seconds"] for r in manifest_rows if r["stage"]=="U")/3600,
        peak_train_bytes=max(r["peak_bytes"] for r in training),
        peak_score_bytes=max(r["peak_bytes"] for r in manifest_rows),
        score_points=sum(r["points"] for r in manifest_rows),
        convention="sum of actual per-device worker elapsed times, including runtime overhead",
    ))
    dump(out/"metric_review.json",dict(rows=len(rows),U_rows_new=84,Z_rows_reused=56,
        AP_ROC_independent_errors_checked=True,VUS="pinned existing implementation",
        elapsed_seconds=time.monotonic()-started))


def f_support_eligibility(entity,spec,labels):
    origin, stop=spec["roles"]["U"]
    y=labels[origin:stop]
    positive=None
    for start,end in intervals(y):
        if start==0 or end==len(y):
            continue
        endpoints=np.arange(max(start+99,611),end)
        if len(endpoints)>=3:
            positive=(int(start),int(end),endpoints)
            break
    normal=next((int(max(start,512)) for start,end in intervals(1-y)
                 if max(start,512)+512<=end),None)
    row=dict(entity=entity,U_FIT=[origin,stop],CAL=spec["roles"]["CAL"],
             QUERY=spec["roles"]["QUERY"],CAL_record="train",label_source="public U-FIT qualifier")
    if positive is None or normal is None:
        return dict(**row,eligible=False,eligible_complete_positive_event=positive is not None,
                    eligible_normal_block=normal is not None)
    a,b,positive_ends=positive
    normal_ends=np.arange(normal+99,normal+512)
    spans=numeric_intervals(np.r_[positive_ends,normal_ends],origin)
    assert all(origin<=lo<hi<=stop<=spec["roles"]["QUERY"][0] for lo,hi in spans)
    return dict(**row,eligible=True,abnormal_support=[origin+a,origin+b],
                normal_support=[origin+normal,origin+normal+512],
                positive_label_points=b-a,normal_label_points=512,
                positive_endpoints=len(positive_ends),normal_endpoints=len(normal_ends),
                numeric_intervals=spans,
                extra_unlabeled_numeric_points=sum(hi-lo for lo,hi in spans)-512-(b-a))


def summarize(out,rows):
    pairs = {
        "B-A":(("U",FROZEN_HISTORY),("U","history")),
        "B-C":(("U",FROZEN_HISTORY),("U","native")),
        "A-C":(("U","history"),("U","native")),
        "B-History Z":(("U",FROZEN_HISTORY),("Z","history")),
        "A-History Z":(("U","history"),("Z","history")),
        "C-Native Z":(("U","native"),("Z","native")),
    }
    lookup={(r["entity"],r["stage"],r["arm"]):r for r in rows}
    assert len(rows)==len(lookup)==140
    write_rows(out/"metrics.csv",sorted(rows,key=lambda r:(r["entity"],r["stage"],r["arm"])))
    contrasts=[]
    for entity in sorted({r["entity"] for r in rows}):
        for name,((s1,a1),(s0,a0)) in pairs.items():
            r1,r0=lookup[entity,s1,a1],lookup[entity,s0,a0]
            assert (int(r1["start"]),int(r1["stop"]))==(int(r0["start"]),int(r0["stop"]))
            contrasts.append(dict(entity=entity,comparison=name,**{
                m:r1[m]-r0[m] if r1[m] is not None and r0[m] is not None else None for m in METRICS
            }))
    write_rows(out/"contrasts.csv",contrasts)
    summaries=[]
    groups=[(f"{stage}_{arm}",[r for r in rows if (r["stage"],r["arm"])==(stage,arm)])
            for stage,arm in [("Z",a) for a in Z_ARMS]+[("U",a) for a in ARMS]]
    groups += [(name,[r for r in contrasts if r["comparison"]==name]) for name in pairs]
    for group,records in groups:
        assert len(records)==28
        for m in METRICS:
            v=np.array([r[m] for r in records if r[m] is not None])
            diff=group in pairs
            summaries.append(dict(group=group,metric=m,eligible=len(v),
                mean=float(v.mean()) if len(v) else None, sample_SD=float(v.std(ddof=1)) if len(v)>1 else None,
                median=float(np.median(v)) if len(v) else None, win=int((v>0).sum()) if diff else None,
                tie=int((v==0).sum()) if diff else None, loss=int((v<0).sum()) if diff else None))
    write_rows(out/"macro.csv",summaries)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=("prepare","source","U","score","evaluate-Z","evaluate"))
    cli.add_argument("workspace",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--arm",choices=ARMS)
    cli.add_argument("--entity")
    cli.add_argument("--stage",choices=("Z","U"))
    arg=cli.parse_args()
    w,out=arg.workspace,arg.output
    if arg.action=="prepare":
        prepare(w,out)
    elif arg.action in ("source","U"):
        if arg.arm is None or (arg.action=="source" and arg.arm not in Z_ARMS):
            cli.error("Select a valid training arm")
        if arg.action=="U" and (arg.entity not in bindings(w) or not (out/"Z_metrics.csv").is_file()):
            cli.error("U requires a bound SMD entity and the completed Z evaluation")
        fit(w,out,arg.arm,arg.entity if arg.action=="U" else None,smd_fold=True)
    elif arg.action=="score":
        if arg.arm is None or arg.entity not in bindings(w) or arg.stage is None:
            cli.error("Scoring requires an arm, entity and stage")
        score(w,out,arg.entity,arg.arm,arg.stage)
    else:
        evaluate(w,out,z_only=arg.action=="evaluate-Z")


if __name__=="__main__":
    main()
