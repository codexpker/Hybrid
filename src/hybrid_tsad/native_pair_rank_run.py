"""CPU pretraining, adaptation and sealed benchmark readout for the fixed PSM fold."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .native_pair_rank import FEATURES, NativePairRank, rank_loss
from .native_v_conditioning_evaluation import read, sha, exact_cal, causal_timing


SEED = 2025
MODES = ("Z", "U-Symmetric", "U-Logistic", "U-NoHistory")
AXIS = (23016, 87744)
ARCHIVE = Path("results/native_v_conditioning_20261119_v1/PSM/TASK.json")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def features(tensor, history):
    if history:
        return tensor
    return torch.cat((tensor[..., :1], torch.zeros_like(tensor[..., 1:])), dim=-1)


def load_sources(path):
    sources = {}
    for family in ("SWAT", "SMD"):
        items = []
        for record in sorted(path.glob(f"{family}*.npz")):
            with np.load(record, allow_pickle=False) as data:
                normal = torch.from_numpy(data["normal"])
                positive = torch.from_numpy(data["positive"])
            if (normal.shape != positive.shape or normal.ndim != 3 or normal.shape[-1] != FEATURES
                    or normal.dtype != torch.float32 or not len(normal)
                    or not torch.isfinite(normal).all() or not torch.isfinite(positive).all()):
                raise ValueError(f"Invalid frozen paired source features: {record.name}")
            items.append((normal, positive))
        if len(items) != (1 if family == "SWAT" else 28):
            raise ValueError(f"Require fixed source record set for {family}")
        if len({item[0].shape for item in items}) != 1:
            raise ValueError(f"Expected equal per-record exposure for {family}")
        sources[family] = tuple(torch.cat(tensors) for tensors in zip(*items))
    return sources


@torch.no_grad()
def validation(model, source, history):
    return float(torch.stack([
        rank_loss(model(features(positive, history)), model(features(normal, history)), symmetric=True).mean()
        for normal, positive in source.values()
    ]).mean())


def train_stage1(fit, select, *, history):
    torch.manual_seed(SEED)
    model = NativePairRank()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(SEED)
    selected_step = 0
    best_loss = validation(model, select, history)
    best = copy.deepcopy(model.state_dict())
    trace = [{"step": 0, "source_select_pair_loss": best_loss}]
    for step in range(1, 241):
        losses = []
        for normal, positive in fit.values():
            ids = rng.integers(len(normal), size=16)
            losses.append(rank_loss(model(features(positive[ids], history)),
                                    model(features(normal[ids], history)), symmetric=True).mean())
        optimizer.zero_grad()
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if step % 40 == 0:
            value = validation(model, select, history)
            trace.append({"step": step, "source_select_pair_loss": value})
            if value < best_loss:
                best_loss, selected_step = value, step
                best = copy.deepcopy(model.state_dict())
    return best, selected_step, trace


def adapt(initial, fit, unlabeled, *, history, symmetric):
    model = NativePairRank()
    model.load_state_dict(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(SEED + 1)
    exposures = []
    for _ in range(120):
        negative_ids = rng.integers(len(unlabeled), size=32)
        compared = model(features(unlabeled[negative_ids], history))
        positive_ids = [rng.integers(len(positive), size=16) for _, positive in fit.values()]
        anchors = torch.cat([model(features(positive[ids], history))
                             for (_, positive), ids in zip(fit.values(), positive_ids)])
        loss = rank_loss(anchors, compared, symmetric=symmetric).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        exposures.append(np.concatenate((negative_ids, *positive_ids)))
    return model.state_dict(), np.stack(exposures)


def train(root):
    started = time.perf_counter()
    if (root / "MODELS_FROZEN.json").exists():
        raise FileExistsError("Model selection is already frozen")
    fit = load_sources(root / "stage1/FIT")
    select = load_sources(root / "stage1/SELECT")
    if any(len(fit[d][0]) != 112 or len(select[d][0]) != 56 for d in fit):
        raise ValueError("Unexpected source sampling budget")
    with np.load(root / "U_FIT.npz", allow_pickle=False) as data:
        unlabeled = torch.from_numpy(data["features"])
    if unlabeled.shape != (224, 25, FEATURES) or unlabeled.dtype != torch.float32 or not torch.isfinite(unlabeled).all():
        raise ValueError("PSM U must be the fixed label-free pool")
    full, best, trace = train_stage1(fit, select, history=True)
    no_history, best_nh, trace_nh = train_stage1(fit, select, history=False)
    checkpoints = {"Z": full}
    plan = None
    for name, initial, history, symmetric in (("U-Symmetric", full, True, True),
                                             ("U-Logistic", full, True, False),
                                             ("U-NoHistory", no_history, False, True)):
        checkpoints[name], positions = adapt(initial, fit, unlabeled, history=history, symmetric=symmetric)
        if plan is not None:
            np.testing.assert_array_equal(plan, positions)
        plan = positions
    np.save(root / "U_EXPOSURES.npy", plan, allow_pickle=False)
    manifest = {}
    for mode, state in checkpoints.items():
        path = root / f"{mode}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save(state, path)
        manifest[mode] = dict(file=path.name, sha256=sha(path))
    write_json(root / "MODELS_FROZEN.json", dict(status="FROZEN_BEFORE_TARGET_SCORE", seed=SEED, modes=manifest,
        source_fit_windows={key: len(value[0]) for key, value in fit.items()},
        source_select_windows={key: len(value[0]) for key, value in select.items()},
        source_selected_step=best, no_history_selected_step=best_nh, source_trace=trace,
        no_history_source_trace=trace_nh, adapt_updates=120, batch=32, optimizer="Adam 0.001",
        parameter_count=sum(v.numel() for v in full.values()), u_exposure_sha256=sha(root / "U_EXPOSURES.npy"),
        cpu_seconds=time.perf_counter()-started, query_or_labels_opened=False))


@torch.no_grad()
def score(root, workspace):
    manifest = read(root / "MODELS_FROZEN.json")
    if manifest["status"] != "FROZEN_BEFORE_TARGET_SCORE" or set(manifest["modes"]) != set(MODES):
        raise ValueError("Four prespecified models must freeze")
    task = read(workspace / ARCHIVE)
    scores = []
    for role in ("CAL", "QUERY"):
        with np.load(root / f"{role}.npz", allow_pickle=False) as bank:
            feature_bank = torch.from_numpy(bank["features"])
            index, native = bank["index"], bank["native_v"]
        points_path = workspace / task["indexes"][role]["relative"]
        if sha(points_path) != task["indexes"][role]["sha256"]:
            raise ValueError("Original scoring axis changed")
        np.testing.assert_array_equal(index, np.load(points_path, allow_pickle=False))
        if feature_bank.shape != (len(index), 25, FEATURES) or native.shape != index.shape:
            raise ValueError("Expected full common pointwise bank")
        output = {"index": index, "TimeRadar-V": native}
        for mode, item in manifest["modes"].items():
            path = root / item["file"]
            if path != root / f"{mode}.pt" or sha(path) != item["sha256"]:
                raise ValueError("Frozen parameter identity changed")
            model = NativePairRank().eval()
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
            output[mode] = torch.cat([model(features(batch, mode != "U-NoHistory"))
                                      for batch in feature_bank.split(512)]).numpy()
        if not all(np.isfinite(array).all() for array in output.values()):
            raise FloatingPointError("Nonfinite scores")
        path = root / f"{role}_SCORES.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **output)
        scores.append(dict(role=role, file=path.name, sha256=sha(path)))
    write_json(root / "SCORES_SEALED.json", dict(models_sha256=sha(root / "MODELS_FROZEN.json"),
        scores=scores, labels_opened=False))


def evaluate(root, workspace, author_metric):
    from .left_native_swat_metrics import continuous_metrics, load_author_metric
    from .timercd_swat_evaluation import evaluate_score

    seal = read(root / "SCORES_SEALED.json")
    if seal["labels_opened"] or seal["models_sha256"] != sha(root / "MODELS_FROZEN.json"):
        raise ValueError("Expected unopened label-free score seal")
    arrays = {}
    for desc in seal["scores"]:
        path = root / desc["file"]
        if path != root / f'{desc["role"]}_SCORES.npz' or sha(path) != desc["sha256"]:
            raise ValueError("Frozen score bank changed")
        with np.load(path, allow_pickle=False) as data:
            arrays[desc["role"]] = dict(data)
    cal, query = arrays["CAL"], arrays["QUERY"]
    np.testing.assert_array_equal(query["index"], np.arange(*AXIS))
    if set(cal) != {"index", "TimeRadar-V", *MODES} or set(query) != set(cal):
        raise ValueError("Incomplete matched model score banks")
    task = read(workspace / ARCHIVE)
    points = np.load(workspace / task["indexes"]["CAL"]["relative"], allow_pickle=False)
    thresholds = {name: exact_cal(cal["index"], cal[name], points) for name in cal if name != "index"}
    label_path = workspace / "data/common_psm_20261028_v1/test_label.npy"
    if sha(label_path) != "309449a288d65de43ff5e58ca2bc06b9913556e46024f5aa0b04153a40b30a69":
        raise ValueError("Labels differ from the existing locked public PSM record")
    labels = np.load(label_path, allow_pickle=False).reshape(-1)
    index = query["index"]
    metric = load_author_metric(author_metric)
    all_rows, all_events = [], []
    for name, threshold in thresholds.items():
        rank, _ = continuous_metrics(labels[index], query[name], metric)
        rows, events, _, _ = evaluate_score(index, query[name], labels, threshold, index,
            query_start=AXIS[0], query_stop=AXIS[1], original_length=len(labels), expected_events=None, ranking=rank)
        rows, events = causal_timing(rows, events)
        all_rows.extend(dict(row, method=name, **rank) for row in rows)
        all_events.extend(dict(row, method=name) for row in events)
    write_json(root / "METRICS.json", dict(axis=list(AXIS), CAL_attack_points=int(labels[points].sum()),
        thresholds=thresholds, results=all_rows, events=all_events))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("train", "score", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--author-metric", type=Path, help="Byte-exact archived LEFT VUS metrics.py")
    options = parser.parse_args()
    torch.set_num_threads(1)
    if options.operation != "train" and options.workspace is None:
        parser.error("--workspace is required for score and evaluate")
    if options.operation == "train":
        train(options.root)
    elif options.operation == "score":
        score(options.root, options.workspace)
    else:
        if options.author_metric is None:
            parser.error("--author-metric is required for evaluate")
        evaluate(options.root, options.workspace, options.author_metric)


if __name__ == "__main__":
    main()
