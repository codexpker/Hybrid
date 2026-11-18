"""Own-process-only four-arm scheduling for the fixed selective LEFT experiment."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from hybrid_tsad.left_selective_worker import sha, dump

ARMS = ["B_trim", "B_random", "C_trim", "C_random"]


def gpu_uuids():
    """Resolve local physical IDs without publishing machine-specific identities."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True, timeout=10)
    mapping = {int(row.split(",")[0]): row.split(",")[1].strip().removeprefix("GPU-")
               for row in result.stdout.splitlines() if row.strip()}
    if not all(index in mapping for index in range(4)):
        raise RuntimeError("This fixed four-arm queue requires physical GPUs 0 through 3")
    return [mapping[index] for index in range(4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    workspace, root = args.workspace, args.root
    GPUS = gpu_uuids()
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(workspace / "preparations/hybrid_left_joint_cycle_20261109_v2/resources"))
    import owned_session
    from gpu_idle_guard import gpu_guard
    paths = [Path(__file__), Path(__file__).with_name("left_selective_worker.py"),
             Path(__file__).with_name("left_selective_update.py")]
    dump(root / "RUNTIME_FREEZE.json", dict(files=[dict(name=p.name, sha256=sha(p)) for p in paths],
         source_plan_sha256="444e61197b6d4951853af6d6396c087d4decc2bc791f25ce52fce0b5ed71e0e6",
         seed=2025, updates=2000, batch=128, retained=103, endpoint_only=True,
         arms=ARMS, CAL_points=512, labels_read=False))

    def job(arm, gpu, phase):
        life = root / "lifecycles" / ((arm or "initial") + "_" + phase)
        life.mkdir(parents=True, exist_ok=False)
        cap = 30 if phase == "score" else 20
        started = time.monotonic()
        utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        code, cleanup, error = None, None, None
        try:
            dump(life / "GPU_GUARD.json", gpu_guard(GPUS[gpu], gpu, cap))
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES="GPU-" + GPUS[gpu], OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            argv = [sys.executable, "-B", "-m", "hybrid_tsad.left_selective_worker", phase,
                    "--workspace", str(workspace), "--root", str(root)]
            if arm:
                argv += ["--arm", arm]
            with (life / "stdout.log").open("xb") as stdout, (life / "stderr.log").open("xb") as stderr:
                child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
                owned = owned_session.claim_group(child)
                dump(life / "STARTED.json", dict(pid=child.pid, supervisor_pid=os.getpid(), GPU=gpu,
                     UUID=GPUS[gpu], UTC=utc, argv=argv, timeout_seconds=900, capGiB=cap, owned=owned))
                print(json.dumps(dict(event="RUNNING", arm=arm, phase=phase, pid=child.pid, GPU=gpu)), flush=True)
                try:
                    code = child.wait(timeout=890)
                finally:
                    cleanup = owned_session.stop_group(child, grace=5)
                assert code == 0 and cleanup["status"] == "NO_LIVE_GROUP_MEMBERS"
        except BaseException as exc:
            error = repr(exc)
        result = dict(status="COMPLETED" if error is None else "FAILED", arm=arm, phase=phase,
                      GPU=gpu, UTC_start=utc, UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      OS_exit_code=code, error=error, cleanup=cleanup,
                      actual_GPU_seconds=time.monotonic()-started)
        dump(life / "OS_WAIT.json", result)
        print(json.dumps(result), flush=True)
        if error:
            raise RuntimeError(result)
        return result

    try:
        gate = job(None, 0, "gate")
        assert json.loads((root / "CUDA_GATE/RECEIPT.json").read_text())["status"] == "PASS"
        def chain(item):
            arm, gpu = item
            return [job(arm, gpu, "train"), job(arm, gpu, "score")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [gate] + [j for pair in pool.map(chain, zip(ARMS, range(4))) for j in pair]
        banks = []
        for arm in ARMS:
            receipt = json.loads((root / arm / "score/RECEIPT.json").read_text())
            for bank in receipt["banks"]:
                assert sha(root / bank["relative"]) == bank["sha256"]
                banks.append(dict(arm=arm, **bank))
        assert len(banks) == 8
        dump(root / "RAW_BANK_SEAL.json", dict(status="COMPLETED", labels_read=False,
             no_labels_decoded=True, banks=banks, jobs=jobs,
             actual_GPU_hours=sum(j["actual_GPU_seconds"] for j in jobs)/3600))
    except BaseException as exc:
        dump(root / "FAILED.json", dict(error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
