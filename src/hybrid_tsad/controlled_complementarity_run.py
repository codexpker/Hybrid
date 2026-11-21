"""Execute the fixed 238-stream public-data complementarity comparison."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time

import numpy as np

from .controlled_complementarity_design import dump, energy, envelope, read, sha, variants, write_csv


MODELS = {"TimeRadar": 99, "TimesFM": 512}


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def array_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def smooth_control(base, target, receiver, start, duration, kind):
    u = np.linspace(0, 1, duration, dtype=np.float64)
    if kind == "H1":
        shape = np.sin(np.pi * u) ** 2 * (2 * u - 1)
    elif kind == "H2":
        shape = envelope(duration, 64)
    else:
        raise ValueError(kind)
    shape[[0, -1]] = 0
    scale = np.sqrt(energy(target, base) / np.dot(shape, shape))
    array = base.copy()
    source = base[start:start + duration, receiver].astype(np.float64)
    array[start:start + duration, receiver] = source + scale * shape
    return array, scale


def control_statistics(array, base, receiver, start, stop):
    delta = array[:, receiver].astype(np.float64) - base[:, receiver]
    second = np.diff(delta, n=2)
    return {
        "peak_source_sigma": float(np.max(np.abs(delta))),
        "mean_source_sigma": float(delta[start:stop].mean()),
        "d2_rms_source_sigma": float(np.sqrt(np.mean(second[start - 2:stop] ** 2))),
        "d2_abs_max_source_sigma": float(np.max(np.abs(second))),
        "d2_energy_over_signal_energy": float(np.dot(second, second) / np.dot(delta, delta)),
    }


def reuse_map(values, cases, times, past):
    """Reuse baseline scores only for identical complete input and target bytes."""
    baseline = values[0]
    assert cases[0] == "base"
    result = np.full((len(cases), len(times)), -1, dtype=np.int32)
    baseline_bits = baseline.view(np.uint32)
    for row, (case, array) in enumerate(zip(cases[1:], values[1:]), 1):
        bits = array.view(np.uint32)
        shifts = (0, int(case.split("sync")[1])) if case.startswith("H3_sync") else (0,)
        for shift in shifts:
            previous = times - shift
            valid = (previous >= times[0]) & (result[row] == -1)
            changed = np.ones(len(baseline), dtype=bool)
            changed[shift:] = np.any(bits[shift:] != baseline_bits[:len(baseline) - shift], axis=1)
            count = np.r_[0, np.cumsum(changed)]
            equal = (count[times + 1] == count[times - past]) & valid
            result[row, equal] = previous[equal] - times[0]
    return result


def prepare(config, frozen_config, cases_path, workload_path, background_path, root):
    config = read(config)
    assert sha(frozen_config) == config["design_config_sha256"]
    assert sha(cases_path) == config["original_cases_sha256"]
    assert sha(workload_path) == config["original_workload_sha256"]
    plan = read(frozen_config)
    frozen = {(row["domain"], int(row["block"]), row["case"]): row for row in csv_rows(cases_path)}
    eligibility = {(row["domain"], int(row["block"]), row["case"]): row for row in csv_rows(workload_path)}
    assert len(frozen) == len(eligibility) == 184 and frozen.keys() == eligibility.keys()
    times = np.arange(*config["score_axis"], dtype=np.int32)
    rows, controls, jobs = [], [], []
    root.mkdir(parents=True, exist_ok=False)
    with np.load(background_path, allow_pickle=False) as backgrounds:
        for domain in plan["domains"]:
            for block in range(4):
                name = f"{domain}_block{block}"
                base = backgrounds[name]
                receiver = int(frozen[domain, block, "base"]["receiver"])
                original = variants(base, receiver, plan, block)
                by_case = {case: array for case, _, array, _ in original}
                case_names, arrays = [], []
                for case, kind, array, meta in original:
                    key = domain, block, case
                    original_row = frozen[key]
                    assert array_sha(array) == original_row["array_sha256"]
                    eligible = eligibility[key]["qualified_for_proposed_inference"] == "True"
                    rows.append(dict(domain=domain, block=block, case=case, kind=kind, eligible=eligible,
                                     array_sha256=array_sha(array), source_start=int(original_row["source_start"])))
                    if eligible:
                        case_names.append(case)
                        arrays.append(array)
                    if kind not in ("H1", "H2"):
                        continue
                    assert eligible and (domain, block, case + "_amp_control") in frozen
                    start = plan["intervention_start"]
                    stop = start + meta["duration"]
                    smooth, scale = smooth_control(base, array, receiver, start, meta["duration"], kind)
                    delta = smooth[:, receiver].astype(np.float64) - base[:, receiver]
                    relative = abs(energy(smooth, base) / energy(array, base) - 1)
                    assert np.isfinite(smooth).all() and relative <= config["total_energy_relative_error_max"]
                    assert not np.any(delta[:start + 1]) and not np.any(delta[stop - 1:])
                    assert np.array_equal(np.delete(smooth, receiver, axis=1), np.delete(base, receiver, axis=1))
                    support = np.flatnonzero(delta)
                    assert (support[0], support[-1]) == (start + 1, stop - 2)
                    if kind == "H1":
                        middle = (start + stop) // 2
                        assert np.all(delta[start + 1:middle] < 0) and np.all(delta[middle:stop - 1] > 0)
                    else:
                        lo, hi = plan["H2"]["primary_core"]
                        assert lo - 514 >= start + 64 and hi <= stop - 64
                        assert np.ptp(delta[lo:hi]) < 2e-6
                    pressure = by_case[case + "_amp_control"]
                    stat = control_statistics(smooth, base, receiver, start, stop)
                    pressure_stat = control_statistics(pressure, base, receiver, start, stop)
                    smooth_name = case + "_smooth_control"
                    controls.append(dict(domain=domain, block=block, case=case, control=smooth_name,
                                         start=start, stop=stop, receiver=receiver, amplitude_scale=float(scale),
                                         energy_target=energy(array, base), energy_control=energy(smooth, base),
                                         relative_total_energy_error=relative, **stat,
                                         stress_d2_energy_over_signal_energy=pressure_stat["d2_energy_over_signal_energy"],
                                         input_sha256=array_sha(smooth), target_sha256=array_sha(array)))
                    rows.append(dict(domain=domain, block=block, case=smooth_name, kind=kind + "_smooth_control",
                                     eligible=True, array_sha256=array_sha(smooth),
                                     source_start=int(original_row["source_start"])))
                    case_names.append(smooth_name)
                    arrays.append(smooth)
                values = np.stack(arrays)
                output = root / f"{name}.npz"
                maps = {model: reuse_map(values, case_names, times, past) for model, past in MODELS.items()}
                np.savez(output, times=times, values=values, cases=case_names, **maps)
                for model, mapping in maps.items():
                    jobs.append(dict(domain=domain, block=block, model=model, inputs=output.name,
                                     input_sha256=sha(output), full_points=int(mapping.size),
                                     forward_points=int(np.sum(mapping == -1))))
    assert len(controls) == 64 and sum(row["eligible"] for row in rows) == config["max_streams"]
    assert len(rows) == 248 and sum(row["eligible"] and row["kind"].endswith("smooth_control") for row in rows) == 64
    assert sum("H1_" in row["control"] for row in controls) == 32
    write_csv(root / "CASES.csv", rows)
    write_csv(root / "SMOOTH_CONTROLS.csv", controls)
    dump(root / "JOBS.json", jobs)
    dump(root / "CPU_GATE.json", dict(result="PASS", original_inputs_bound=184, inferred_streams=238,
                                       original_ineligible=10, added=64, model_forwards=0,
                                       cases_sha256=sha(root / "CASES.csv"), jobs_sha256=sha(root / "JOBS.json")))


def score(workspace, root, model, domain, block):
    import torch
    from . import native_v_conditioning as radar
    from . import timesfm3_frozen as tfm
    from .native_v_conditioning_runner import setup

    start = time.monotonic()
    gate = read(root / "CPU_GATE.json")
    assert gate["result"] == "PASS" and sha(root / "JOBS.json") == gate["jobs_sha256"]
    job = next(row for row in read(root / "JOBS.json")
               if (row["domain"], row["block"], row["model"]) == (domain, block, model))
    inputs = root / job["inputs"]
    assert sha(inputs) == job["input_sha256"]
    with np.load(inputs, allow_pickle=False) as bank:
        cases, times, values, mapping = (bank[key] for key in ("cases", "times", "values", model))
    output = root / f"{domain}_block{block}_{model}"
    output.mkdir(exist_ok=False)
    device = setup(20)
    network = radar.load_native(workspace, device) if model == "TimeRadar" else tfm.load_forecaster(workspace, device)
    state_model = network if model == "TimeRadar" else network.model
    before = radar.state_hashes(state_model)
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(device).clone()
    row, col = np.where(mapping == -1)
    assert len(row) == job["forward_points"]
    score_values = np.empty(mapping.shape, dtype=np.float64)
    channels = values.shape[-1]
    with torch.inference_mode():
        if model == "TimeRadar":
            mask = radar.actual_mask_bits(channels, device)
            # Bind frozen mask bytes to the previous official run, not only to the seed.
            old = workspace / "results/native_v_conditioning_20261119_v1/cache" / domain / "QUERY/part0/RECEIPT.json"
            assert array_sha(mask) == read(old)["mask_bits_sha256"]
            np.save(output / "MASK.npy", mask)
            copies = np.empty((len(row), 10, channels), dtype=np.float32)
            for i, (r, c) in enumerate(zip(row, col)):
                t = times[c]
                x = torch.from_numpy(values[r, t - 99:t + 1].copy()).to(device)[None]
                with radar.fixed_native_rng(device):
                    _, reconstructed = network(x, norm=0, mask_mode="c", copies=10)
                    original = network.cal_anomaly_score(x, reconstructed, anomaly_criterion=None, L=1)
                score_values[r, c] = float(original[0, 99])
                copies[i] = reconstructed[:, 0, 99].detach().cpu().numpy()
                if (i + 1) % 4096 == 0:
                    print(f"{domain} block {block} {model}: {i+1}/{len(row)}", flush=True)
            np.testing.assert_allclose(score_values[row, col], copies.astype(np.float64).var(1, ddof=1).mean(1),
                                       rtol=1e-5, atol=1e-7)
            evidence = {"copies": copies}
        else:
            median = np.empty((len(row), channels), dtype=np.float32)
            actual = np.empty_like(median)
            for begin in range(0, len(row), tfm.OUTER_CHUNK):
                sl = slice(begin, begin + tfm.OUTER_CHUNK)
                rr, cc = row[sl], col[sl]
                contexts = [tfm.history_context(values[r], 0, int(times[c])) for r, c in zip(rr, cc)]
                prediction, _ = tfm.run_predictions(network, contexts)
                target = values[rr, times[cc]]
                median[sl], actual[sl] = prediction, target
                score_values[rr, cc] = tfm.residual_score(prediction, target)[1]
                if (begin + tfm.OUTER_CHUNK) % 4096 == 0:
                    print(f"{domain} block {block} {model}: {begin + tfm.OUTER_CHUNK}/{len(row)}", flush=True)
            evidence = dict(median=median, actual=actual)
    assert radar.state_hashes(state_model) == before
    assert all(not p.requires_grad and p.grad is None for p in state_model.parameters())
    assert all(not module.training for module in state_model.modules())
    assert torch.equal(torch.get_rng_state(), cpu_rng) and torch.equal(torch.cuda.get_rng_state(device), gpu_rng)
    reuse = mapping >= 0
    score_values[reuse] = score_values[0, mapping[reuse]]
    assert np.isfinite(score_values).all() and (score_values >= 0).all()
    bank_path = output / "BANK.npz"
    np.savez(bank_path, cases=cases, time=times, available_index=times, footprint_start=times - MODELS[model],
             footprint_end_exclusive=times + (model == "TimeRadar"), scores=score_values,
             reused_base_col=mapping, forward_row=row, forward_col=col, **evidence)
    dump(output / "RECEIPT.json", dict(**job, bank_sha256=sha(bank_path), state_unchanged=True, rng_restored=True,
                                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                      elapsed_seconds=time.monotonic() - start, pid=os.getpid(),
                                      visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES")))


def dispatch(workspace, root, devices):
    """Run at most one owned process per assigned device and retain exit receipts."""
    assert len(set(devices)) == len(devices) <= 4 and len(devices) > 0
    jobs = read(root / "JOBS.json")
    assert len(jobs) == 16 and read(root / "CPU_GATE.json")["result"] == "PASS"
    per_point = {("TimeRadar", "SWAT"): .012985, ("TimeRadar", "PSM"): .012389,
                 ("TimesFM", "SWAT"): .049333, ("TimesFM", "PSM"): .040644}
    queue = Queue()
    for job in sorted(jobs, key=lambda j: -j["forward_points"] * per_point[j["model"], j["domain"]]):
        queue.put(job)
    logs = root / "lifecycle"
    logs.mkdir(exist_ok=False)

    def worker(gpu):
        receipts = []
        while True:
            try:
                job = queue.get_nowait()
            except Empty:
                return receipts
            domain, block, model = job["domain"], job["block"], job["model"]
            key = f"{domain}_block{block}_{model}"
            elapsed = 0.0
            pid = None
            started = time.time()
            exit_code = None
            reason = None
            usage = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                             "--format=csv,noheader,nounits"], text=True)
            if int(usage.strip()) > 128:
                reason = "device occupied before launch"
            else:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                command = [sys.executable, "-m", __spec__.name, "score", str(workspace), str(root), model, domain, str(block)]
                with (logs / f"{key}.log").open("x", encoding="utf-8") as log:
                    tic = time.monotonic()
                    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                    pid = proc.pid
                    dump(logs / f"{key}.START.json", dict(gpu=gpu, pid=pid, started_unix=started, **job))
                    estimate = job["forward_points"] * per_point[model, domain]
                    try:
                        exit_code = proc.wait(timeout=max(900, estimate * 1.4 + 120))
                    except subprocess.TimeoutExpired:
                        reason = "timeout"
                        proc.terminate()
                        try:
                            exit_code = proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            exit_code = proc.wait()
                    elapsed = time.monotonic() - tic
            receipt = dict(gpu=gpu, pid=pid, started_unix=started, ended_unix=time.time(),
                           occupied_wall_seconds=elapsed, exit_code=exit_code, reason=reason, **job)
            dump(logs / f"{key}.EXIT.json", receipt)
            receipts.append(receipt)
            if exit_code != 0:
                return receipts

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        receipts = [receipt for result in pool.map(worker, devices) for receipt in result]
    succeeded = sum(row["exit_code"] == 0 for row in receipts)
    dump(root / "DISPATCH.json", dict(succeeded=succeeded, total_expected=len(jobs), receipts=receipts,
                                     occupied_gpu_hours=sum(row["occupied_wall_seconds"] for row in receipts) / 3600))
    if succeeded != len(jobs):
        raise RuntimeError(f"{succeeded}/{len(jobs)} jobs completed; see lifecycle receipts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare_cli = actions.add_parser("prepare")
    for key in ("config", "frozen_config", "cases_path", "workload_path", "background_path", "root"):
        prepare_cli.add_argument(key, type=Path)
    scoring = actions.add_parser("score")
    scoring.add_argument("workspace", type=Path)
    scoring.add_argument("root", type=Path)
    scoring.add_argument("model", choices=MODELS)
    scoring.add_argument("domain", choices=("SWAT", "PSM"))
    scoring.add_argument("block", type=int, choices=range(4))
    dispatch_cli = actions.add_parser("dispatch")
    dispatch_cli.add_argument("workspace", type=Path)
    dispatch_cli.add_argument("root", type=Path)
    dispatch_cli.add_argument("devices", type=int, nargs="+")
    arguments = vars(parser.parse_args())
    {"prepare": prepare, "score": score, "dispatch": dispatch}[arguments.pop("action")](**arguments)


if __name__ == "__main__":
    main()
