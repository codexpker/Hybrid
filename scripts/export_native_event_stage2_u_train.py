"""Export compact endpoint, role, and measured-compute records for the seven fixed updates."""
import argparse
import csv
import json
from pathlib import Path

from hybrid_tsad.native_event_stage2_u_train import SEED, STEPS, TARGETS, ARMS, SOURCE, CHECKPOINT, POINT_SHA
from hybrid_tsad.native_v_conditioning import sha
from hybrid_tsad.native_v_conditioning_runner import read


def write_rows(path, rows):
    with path.open("x", newline="") as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish(root, output):
    output.mkdir(exist_ok=True,parents=True)
    seal=read(root/"ALL_MODELS_FROZEN.json")
    expected={(d,a) for d in TARGETS for a in ARMS}|{(SOURCE,"source_pair")}
    assert {(r["domain"],r["arm"]) for r in seal["models"]}==expected and len(seal["models"])==7
    jobs={r["identity"]:r for r in seal["jobs"]}
    entries=[]
    for r in seal["models"]:
        d,a=r["domain"],r["arm"]
        key=f"{d}_{a}"
        p=root/r["checkpoint_relative"]
        ep=read(p.with_name("ENDPOINT.json"))
        tr=read(p.with_name("TRACE.json"))
        assert ep["checkpoint_sha256"]==r["sha256"]==sha(p) and ep["initial_sha256"]==POINT_SHA
        assert ep["trace_sha256"]==sha(p.with_name("TRACE.json"))
        assert ep["update"]==len(tr)==STEPS and tr[-1]["update"]==STEPS
        assert ep["fixed_state_and_rng_verified"] and not ep["target_labels_read"]
        assert jobs[key]["status"]=="COMPLETED" and jobs[key]["OS_exit_code"]==0
        entries.append(dict(target=d,arm=a,seed=SEED,updates=STEPS,source_reconstruction_windows=ep["source_reconstruction_windows"],
                            target_reconstruction_windows=ep["target_reconstruction_windows"],
                            source_positive_windows=ep["additional_positive_windows"],
                            source_paired_normal_windows=ep["additional_paired_normal_windows"],
                            gpu=jobs[key]["GPU"],gpu_seconds=jobs[key]["actual_GPU_seconds"],
                            measured_worker_seconds=ep["elapsed_seconds"],peak_reserved_bytes=ep["peak_reserved_bytes"],
                            checkpoint_sha256=sha(p),trace_sha256=ep["trace_sha256"],source_plan_sha256=ep["source_plan_sha256"]))
    write_rows(output/"training_cost.csv",entries)
    (output/"model_identities.json").write_text(json.dumps(dict(
        status="COMPLETED",initial_checkpoint=str(CHECKPOINT),initial_sha256=POINT_SHA,
        source_plan_and_roles="training_config.json",global_seal_sha256=sha(root/"ALL_MODELS_FROZEN.json"),
        checkpoints=seal["models"],gpu_hours=seal["actual_GPU_hours"],
        checkpoint_access="Server research workspace. Only identities and aggregate results in public Git."
    ),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    publish(args.root,args.output)
