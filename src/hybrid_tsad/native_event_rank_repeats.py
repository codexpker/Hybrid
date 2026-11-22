"""Paired project-source repeats with one fixed official model and SELECT panel."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from .native_event_rank_training import (
    dump, load_plan, prepare, read, require, run_jobs, sha, training,
)

BASE = Path("results/native_event_rank_psm_20261122_v1")
MODULE = "hybrid_tsad.native_event_rank_repeats"
ARMS = ("copy", "point_weighted")
SEEDS = (2026, 2027)
FIT_KEYS = ("swat_fit", "smd_entity", "smd_fit", "positive_fit", "normal_fit")


def make_plans(workspace, root):
    baseline = workspace / BASE
    require(sha(baseline / "PLAN.npz") == "df7e74047bb0fd49d25a2da0bb58e54a2fbe3327fd9f61b6e07ceca6422543e9",
            "Baseline source plan identity mismatch")
    old = load_plan(baseline)
    with TemporaryDirectory() as temp:
        prepare(workspace, Path(temp))
        restored = load_plan(Path(temp))
        for key in old:
            np.testing.assert_array_equal(old[key], restored[key])
    old_scale = read(baseline / "SCALE.json")
    prev = old
    for seed in SEEDS:
        out = root / f"seed{seed}"
        prepare(workspace, out, seed, old)
        current = load_plan(out)
        np.testing.assert_array_equal(current["scale_normal_fit"], old["scale_normal_fit"])
        require(all(not np.array_equal(current[k], prev[k]) and not np.array_equal(current[k], old[k])
                    for k in FIT_KEYS), "Source FIT randomness failed to change")
        prior_bank = baseline / old_scale["bank_file"]
        require(sha(prior_bank) == old_scale["bank_sha256"], "Frozen scale bank changed")
        bank = out / prior_bank.name
        bank.hardlink_to(prior_bank)
        dump(out / "SCALE.json", dict(old_scale, plan_sha256=sha(out / "PLAN.npz"),
                                     reused_scale_sha256=sha(baseline / "SCALE.json")))
        prev = current


def queue(workspace, root):
    make_plans(workspace, root)
    tasks = [(f"{seed}_{arm}", ["train", "--seed", str(seed), "--arm", arm],
              20, 3600) for seed in SEEDS for arm in ARMS]
    jobs = run_jobs(workspace, root, "training", tasks, MODULE)
    seal(root, jobs)


def seal(root, jobs):
    manifests = {}
    for seed in SEEDS:
        out = root / f"seed{seed}"
        arms = {}
        for arm in ARMS:
            path = out / "train" / arm / "SELECTION.json"
            selection = read(path)
            checkpoint = path.with_name("selected.pt")
            require(selection["status"] == "COMPLETED" and selection["seed"] == seed and
                    sha(checkpoint) == selection["selected_checkpoint_sha256"], "Invalid source selection")
            arms[arm] = dict(checkpoint_relative=checkpoint.relative_to(out).as_posix(),
                             selection_relative=path.relative_to(out).as_posix(),
                             checkpoint_sha256=sha(checkpoint), selection_sha256=sha(path),
                             selected_step=selection["selected_step"])
        job_rows = [r for r in jobs if r["identity"].startswith(str(seed))]
        manifest = out / "MODELS_FROZEN.json"
        dump(manifest, dict(status="COMPLETED", seed=seed, arms=arms, jobs=job_rows,
                            actual_GPU_hours=sum(r["actual_GPU_seconds"] for r in job_rows)/3600,
                            target_read=False))
        manifests[str(seed)] = dict(relative=manifest.relative_to(root).as_posix(), sha256=sha(manifest))
    dump(root / "ALL_MODELS_FROZEN.json", dict(status="COMPLETED", seeds=manifests,
         seed2025_plan_reconstructed=True, official_initialization_shared=True,
         source_select_panel_fixed=True, target_read=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train","queue","finish"))
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=SEEDS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--retry-gpu", type=int, default=2)
    args = p.parse_args()
    if args.action == "queue":
        queue(args.workspace, args.output)
    elif args.action == "finish":
        # A failed pre-launch guard consumes no GPU time. Only retry that job.
        jobs = []
        for seed in SEEDS:
            for arm in ARMS:
                identity = f"{seed}_{arm}"
                record = read(args.output / "lifecycle/training" / identity / "OS_WAIT.json")
                if record["status"] == "COMPLETED":
                    jobs.append(record)
                    continue
                require(record["pid"] is None and record["actual_GPU_seconds"] == 0 and
                        "idle/free-memory guard failed" in record["error"], "Unexpected failure")
                retry = [(identity, ["train", "--seed", str(seed), "--arm", arm], 20, 3600)]
                jobs.extend(run_jobs(args.workspace, args.output, "training_retry", retry, MODULE, [args.retry_gpu]))
        seal(args.output, jobs)
    else:
        training(args.workspace, args.output / f"seed{args.seed}", args.arm, args.seed)


if __name__ == "__main__":
    main()
