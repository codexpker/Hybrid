"""One bounded SMD worker per authorized GPU, with shared filesystem claims."""

import argparse
import contextlib
import os
from pathlib import Path
import sys
import time
import traceback

from hybrid_tsad import left_smd_targets as target


def done(root, machine, arm):
    dest = root / machine / "banks"
    if machine == "shared":
        return (root / "source_only" / "receipt.json").is_file()
    role = "TEST" if arm == "N" else "QUERY"
    return (dest / f"{arm}_{role}.json").is_file()


def run(w):
    root = w / target.OUT
    logs = root / "workers"
    logs.mkdir(parents=True, exist_ok=True)
    machines = sorted(target.smd.bindings(w))
    jobs = [("shared", "source_only")]
    jobs += [
        (m, a)
        for m in machines
        for a in (
            "source",
            "N",
            "ordinary",
            "replay",
            "window",
            "source_only",
            "random_u",
        )
    ]
    while True:
        incomplete = False
        for m, a in jobs:
            if done(root, m, a):
                continue
            incomplete = True
            if (
                a != "random_u"
                and not (target.source_path(w) / "receipt.json").exists()
            ):
                continue
            if (
                a == "source_only"
                and m != "shared"
                and not done(root, "shared", "source_only")
            ):
                continue
            name = f"{m}_{a}"
            lock = logs / (name + ".lock")
            try:
                lock.mkdir()
            except FileExistsError:
                continue
            with (logs / (name + ".log")).open("a", buffering=1) as stream:
                with (
                    contextlib.redirect_stdout(stream),
                    contextlib.redirect_stderr(stream),
                ):
                    try:
                        print(
                            "start",
                            time.time(),
                            os.environ["CUDA_VISIBLE_DEVICES"],
                            os.getpid(),
                            flush=True,
                        )
                        sys.argv = ["left_smd_targets", str(w), m, a] + (
                            ["--shared-train"] if m == "shared" else []
                        )
                        target.main()
                        print("end", time.time(), flush=True)
                    except BaseException:
                        traceback.print_exc()
                        print("failed", time.time(), flush=True)
                        raise
            lock.rmdir()
            break
        else:
            if not incomplete:
                return
            # A different active worker already claimed each remaining runnable job.
            return


def verify(w):
    """Bind target banks to sealed checkpoints and match each machine's training plans."""
    import numpy as np

    root = w / target.OUT
    source = target.smd.read(root / "source" / "receipt.json")
    control = target.smd.read(root / "source_only" / "receipt.json")
    assert target.smd.sha(root / "source_only" / "selected.pt") == control["sha256"]
    assert control["source_exposure"] == 64000 and control["target_exposure"] == 0
    assert control["parent"] == source["sha256"] and control["updates"] == 2000
    assert control["selected_step"] == source["selected"] + 2000
    costs = [dict(entity="shared", arm="source_only", seconds=control["seconds"])]
    all_entities = {}
    for entity, rec in sorted(target.smd.bindings(w).items()):
        directory = root / entity
        roles = target.smd.read(directory / "roles.json")
        assert roles == target.roles(rec)
        assert (
            roles["CAL"][0] > roles["scaler_fit"][1]
            and roles["U"][1] < roles["QUERY"][0]
        )
        rows = {}
        for arm in (
            "source",
            "source_only",
            "N",
            "random_u",
            "ordinary",
            "replay",
            "window",
        ):
            if arm in ("source", "source_only"):
                row = source if arm == "source" else control
            else:
                row = target.smd.read(directory / arm / "receipt.json")
                assert target.smd.sha(directory / arm / "selected.pt") == row["sha256"]
                assert row["seed"] == 2025 and row["parent"] == (
                    None if arm == "random_u" else source["sha256"]
                )
                if arm == "N":
                    assert row["loss_history"][row["selected"] - 1] == min(
                        row["loss_history"]
                    )
                    assert row["sampler_cpu_rng_sha256"]
                else:
                    assert (
                        row["updates"] == 2000
                        and row["source_exposure"] + row["target_exposure"] == 64000
                    )
                    assert row["initial_step"] == (
                        0 if arm == "random_u" else source["selected"]
                    )
                    assert row["selected_step"] == row["initial_step"] + 2000
                costs.append(dict(entity=entity, arm=arm, seconds=row["seconds"]))
            bank_hashes = {}
            for role in (
                ("TEST", "CAL", "QUERY")
                if arm == "source"
                else ("TEST",)
                if arm == "N"
                else ("CAL", "QUERY")
            ):
                path = directory / "banks" / f"{arm}_{role}.npz"
                bank = target.smd.read(path.with_suffix(".json"))
                assert (
                    target.smd.sha(path) == bank["sha256"]
                    and row["sha256"] == bank["checkpoint_sha256"]
                )
                bank_hashes[role] = bank["sha256"]
            rows[arm] = dict(
                checkpoint=row["sha256"],
                banks=bank_hashes,
                plan=row.get("shared_target_plan"),
            )
        plans = {rows[a]["plan"] for a in ("random_u", "ordinary", "replay", "window")}
        assert None not in plans and len(plans) == 1
        weights_path = directory / "window" / "target_window_weights.npy"
        weights = np.load(weights_path, allow_pickle=False)
        assert (
            target.smd.sha(weights_path)
            == target.smd.read(directory / "window" / "receipt.json")["weights_sha256"]
        )
        assert weights.shape == (2000, 16) and np.isfinite(weights).all()
        np.testing.assert_allclose(weights.mean(1), 1, rtol=0, atol=2e-7)
        all_entities[entity] = rows
    assert len(all_entities) == 28 and len(costs) == 141
    failed_setups = []
    for log in sorted((root / "workers").glob("machine-*.log")):
        attempts = ("\n" + log.read_text()).split("\nstart ")[1:]
        for run, retry in zip(attempts, attempts[1:]):
            if "\nTraceback " in run:
                assert "\nend " not in run
                start, device, _ = run.splitlines()[0].split()
                failed_setups.append(
                    dict(
                        job=log.stem,
                        gpu=int(device),
                        seconds_until_retry_including_idle=float(retry.split()[0])
                        - float(start),
                    )
                )
    target.smd.dump(
        root / "training_review.json",
        dict(
            entities=all_entities,
            training_costs=costs,
            target_train_seconds=sum(c["seconds"] for c in costs),
            failed_preupdate_setups=failed_setups,
            source_checkpoint=source["sha256"],
            source_only_checked_once=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    (verify if args.verify else run)(args.workspace)
