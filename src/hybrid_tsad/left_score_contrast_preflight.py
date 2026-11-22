"""Verify the paired source probes and native update path before target adaptation."""

import argparse
import copy
import csv
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import torch

from hybrid_tsad.left_domain_replay import rng_hashes, tensor_hashes
from hybrid_tsad.left_score_contrast import (
    adam_displacement, contrast_parts, dot64, native_score, probe_scores,
    probe_windows, reference_gradient, select_pair, source_windows,
)
from hybrid_tsad.left_source_preservation import preserving_eval
from hybrid_tsad.left_swat_author import dump, read, sha


def source_data(workspace, task, validation):
    source = task["source"]
    item = task["data"]["source_train"]
    raw_path = workspace / item["relative"]
    scaler_path = workspace / source["scaler"]["relative"]
    if sha(raw_path) != item["sha256"] or sha(scaler_path) != source["scaler"]["sha256"]:
        raise ValueError("Frozen source data or scaler identity differs")
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if list(raw.shape) != item["shape"] or validation[0] != task["roles"]["SOURCE_FIT"][1]:
        raise ValueError("Source roles are not the fixed separated intervals")
    if tuple(source["scaler_fit"]) != (task["roles"]["SOURCE_FIT"][0], validation[1]):
        raise ValueError("Source validation differs from the original author source prefix")
    if source["scaler"]["format"] == "json_statistics":
        stats = read(scaler_path)["statistics"]
        mean, scale = np.asarray(stats["mean"]), np.asarray(stats["scale"])
    else:
        with np.load(scaler_path, allow_pickle=False) as stats:
            mean, scale = stats["mean"], stats["scale"]
    def role_values(interval):
        lo, hi = interval
        values = (np.array(raw[lo:hi], dtype=np.float64, order="F", copy=True) - mean) / scale
        if not np.isfinite(values).all():
            raise FloatingPointError("Invalid source values")
        return values
    return role_values(task["roles"]["SOURCE_FIT"]), source_windows(role_values(validation))


def load_model(workspace, task):
    source = task["source"]
    path = workspace / source["checkpoint"]["relative"]
    config = workspace / source["model_config"]["relative"]
    upstream = workspace / source["upstream"]
    if (sha(path) != source["checkpoint"]["sha256"] or sha(config) != source["model_config"]["sha256"]
            or sha(upstream / "models/Left/LEFT.py") != source["model_sha256"]):
        raise ValueError("Frozen LEFT source identities differ")
    sys.path.insert(0, str(upstream))
    from models.Left.LEFT import Left
    model = Left(SimpleNamespace(**read(config))).cuda()
    loaded = model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
    if loaded.missing_keys or any(not k.endswith(("total_ops", "total_params")) for k in loaded.unexpected_keys):
        raise ValueError("Archived checkpoint cannot initialize this official LEFT")
    model.step = source["python_step"]
    return model


def native_witness(model, sample, pair):
    model.eval()
    variants = probe_windows(sample, pair).flatten(0, 1)
    with preserving_eval(model):
        official = model.infer(variants, None, None, None)[0]
        actual = native_score(model, variants)
    torch.testing.assert_close(actual.detach(), official, atol=2e-5, rtol=2e-5)
    parity = (actual.detach() - official).abs().max().item()
    del actual, official
    # Use a single real native training forward, then branch at the Adam substep.
    # Repeating the CUDA backward would confound update equality with kernel rounding.
    torch.manual_seed(2025)
    trained = copy.deepcopy(model).train()
    trained(sample, None, None, None)[0].backward()
    assert trained.step == model.step + 1
    source_grads = tuple(p.grad for p in trained.parameters())
    before = tuple(p.detach().clone() for p in model.parameters())
    results = []
    baseline = None
    baseline_opt = None
    for mode in ("plain", "nonconflict", "conflict"):
        student = copy.deepcopy(model).train()
        student.load_state_dict(trained.state_dict())
        student.step = trained.step
        parameters = tuple(student.parameters())
        optimizer = torch.optim.Adam(parameters, lr=1e-4)
        for p, grad in zip(parameters, source_grads):
            p.grad = grad.detach().clone() if grad is not None else None
        buffers = tensor_hashes(student)
        rng = rng_hashes(torch)
        modes = [module.training for module in student.modules()]
        if mode == "plain":
            optimizer.step()
            baseline = tuple(p.detach() - old for p, old in zip(parameters, before))
            baseline_opt = optimizer.state_dict()["state"]
            stats = {}
        else:
            native_grads = tuple(p.grad.detach().clone() if p.grad is not None else None for p in parameters)
            gradient, stats = reference_gradient(student, sample, pair, parameters)
            initial_dot = dot64(baseline, gradient)
            assert initial_dot != 0 and dot64(gradient, gradient) > 0
            # Exercise both branches on the same real score gradient. Flip orientation only in this witness.
            orientation = 1 if ((initial_dot < 0) == (mode == "conflict")) else -1
            normal = tuple(orientation * g if g is not None else None for g in gradient)
            assert tensor_hashes(student) == buffers and rng_hashes(torch) == rng
            assert [module.training for module in student.modules()] == modes
            assert student.step == model.step + 1
            for p, grad in zip(parameters, native_grads):
                if grad is None:
                    assert p.grad is None
                else:
                    assert torch.equal(p.grad, grad)
            stats.update(adam_displacement(optimizer, parameters, normal=normal), witness_orientation=orientation)
            assert stats["triggered"] == (mode == "conflict")
            actual_disp = tuple(p.detach() - old for p, old in zip(parameters, before))
            if mode == "conflict":
                tolerance = 3e-3 * stats["proposal_l2"] * stats["reference_l2"]
                assert abs(stats["dot_applied"]) <= tolerance
                assert any(not torch.equal(a, b) for a, b in zip(actual_disp, baseline))
            else:
                assert all(torch.equal(a, b) for a, b in zip(actual_disp, baseline))
            for key, state in optimizer.state_dict()["state"].items():
                assert all(torch.equal(value, baseline_opt[key][name]) for name, value in state.items())
        assert {int(state["step"]) for state in optimizer.state.values()} == {1}
        results.append(dict(mode=mode, **stats))
    return dict(native_score_max_abs_difference=parity, steps=results)


def main():
    parser = argparse.ArgumentParser()
    for name in ("workspace", "task", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-validation", nargs=2, type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(20 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    task = read(args.task)
    fit, panel = source_data(args.workspace, task, args.source_validation)
    pair, association = select_pair(fit)
    model = load_model(args.workspace, task).eval()
    before = tensor_hashes(model)
    witness = native_witness(model, torch.tensor(panel[:8], dtype=torch.float32, device="cuda"), pair)
    score_rows = []
    statistics = []
    for block, rows in enumerate(np.array_split(np.arange(len(panel)), 4), start=1):
        scores = []
        for indices in np.array_split(rows, (len(rows) + 31) // 32):
            values = torch.tensor(panel[indices], dtype=torch.float32, device="cuda")
            scores.append(probe_scores(model, values, pair, grad=False).cpu().numpy().astype(np.float64))
        arms = np.concatenate(scores, axis=1)
        same, opposite, contrast = contrast_parts(arms)
        numerical_floor = float(32 * np.finfo(np.float32).eps * np.mean(np.abs(arms)))
        statistics.append(dict(block=block, windows=len(rows), start=args.source_validation[0] + int(rows[0]) * 192,
            end=args.source_validation[0] + (int(rows[-1]) + 1) * 192, same_mean=float(same.mean()),
            opposite_mean=float(opposite.mean()), contrast_mean=float(contrast.mean()),
            contrast_p05=float(np.quantile(contrast, .05)), contrast_median=float(np.median(contrast)),
            contrast_p95=float(np.quantile(contrast, .95)), positive=int(np.sum(contrast > numerical_floor)),
            negative=int(np.sum(contrast < -numerical_floor)), near_zero=int(np.sum(np.abs(contrast) <= numerical_floor)),
            numerical_floor=numerical_floor, supports_positive_orientation=bool(contrast.mean() > numerical_floor)))
        for k, index in enumerate(rows):
            score_rows.append(dict(block=block, start=args.source_validation[0] + int(index) * 192,
                plus_plus=arms[0, k], minus_minus=arms[1, k], plus_minus=arms[2, k], minus_plus=arms[3, k],
                same=same[k], opposite=opposite[k], contrast=contrast[k]))
    assert tensor_hashes(model) == before and model.step == task["source"]["python_step"]
    with (args.output / "SOURCE_PROBES.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=tuple(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    dump(args.output / "PREFLIGHT.json", dict(domain=task["domain"], pair=pair, fit_association=association,
        CHANNEL_INDEX_BASE=0, source_fit=task["roles"]["SOURCE_FIT"], source_validation=args.source_validation,
        blocks=statistics, windows=len(score_rows), source_checkpoint_sha256=task["source"]["checkpoint"]["sha256"],
        task_sha256=sha(args.task), witness=witness,
        supports_positive_orientation=all(b["supports_positive_orientation"] for b in statistics),
        source_probes_sha256=sha(args.output / "SOURCE_PROBES.csv"), labels_read=False,
        target_observations_scored=False, seconds=time.monotonic()-started,
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved()))


if __name__ == "__main__":
    main()
