"""PSM repeat readout: share the frozen B=1 encoder across selected decoder heads."""
import argparse
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .left_native_swat_metrics import continuous_metrics, load_author_metric
from .left_swat_replay_repeats import write_csv
from .native_event_rank_readout import BASE, LABEL_SHA, METRIC_SHA, TARGET_SHA, TASK_SHA, asset
from .native_event_space_summary import group_operating
from .native_rank_psm_extension import AXIS
from .native_v_conditioning import fixed_native_rng, load_native, state_hashes
from .native_v_conditioning_evaluation import causal_timing, exact_cal
from .native_v_conditioning_runner import dump, read, role_index, setup, sha
from .timercd_swat_evaluation import evaluate_score

ROOT = Path("results/native_event_rank_seed_repeats_20261122_v1")
SUFFIX = np.arange(23016, 87744)
SEEDS = (2026, 2027)
ARMS = ("copy", "point_weighted")
FROZEN = ("results/native_rank_psm_extension_20261122_v1/frozen_native_V.npz",
          "83021c4f2175a1d10934073bb0b1b42874f0b807061f7d3b101c02d428a44c9a")


def load_selections(workspace, device):
    root = workspace / ROOT
    all_seal = root / "ALL_MODELS_FROZEN.json"
    if not all_seal.is_file():
        raise ValueError("All four source selections must finish before PSM readout")
    publication = read(all_seal)
    assert publication["status"] == "COMPLETED" and publication["target_read"] is False
    base = load_native(workspace, device)
    original = base.state_dict()
    metadata, models = {}, {}
    for seed in SEEDS:
        directory = root / f"seed{seed}"
        ref = publication["seeds"][str(seed)]
        seal = asset(root, ref["relative"], ref["sha256"])
        assert seal == directory / "MODELS_FROZEN.json"
        selected = read(seal)
        assert selected["status"] == "COMPLETED" and selected["seed"] == seed
        for arm in ARMS:
            key = f"{seed}_{arm}"
            ref = selected["arms"][arm]
            ck = asset(directory, ref["checkpoint_relative"], ref["checkpoint_sha256"])
            state = torch.load(ck, map_location=device, weights_only=True)
            assert state["update"] == ref["selected_step"] and state["arm"] == arm
            model = deepcopy(base)
            model.load_state_dict(state["model"], strict=True)
            assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items()
                       if not k.startswith("decoder."))
            reused = all(torch.equal(v, original[k]) for k, v in model.state_dict().items()
                         if k.startswith("decoder."))
            metadata[key] = dict(seed=seed, method=arm, checkpoint_sha256=ref["checkpoint_sha256"],
                                 selected_step=ref["selected_step"], seed_seal_sha256=sha(seal),
                                 reused_frozen=reused)
            if not reused:
                models[key] = model
    return metadata, models, sha(all_seal)


def score(workspace, out):
    from torch_frft.frft_module import ifrft
    start = time.monotonic()
    device = setup(12)
    metadata, models, seal_sha = load_selections(workspace, device)
    task_dir = workspace / BASE / "PSM"
    task = read(asset(task_dir, "TASK.json", TASK_SHA))
    raw = np.load(asset(workspace, task["data"]["target_test"]["relative"], TARGET_SHA), allow_pickle=False)
    with np.load(asset(task_dir, "SCALER.npz", task["scaler_sha256"])) as z:
        values = ((raw.astype(np.float64) - z["mean"]) / z["std"]).astype(np.float32)
    assert values.shape == (87841, 25) and np.isfinite(values).all()
    x = torch.from_numpy(values).to(device)
    cal = role_index(workspace, task_dir, task, "CAL")
    cols = list(metadata)
    result = np.empty((len(AXIS), len(cols)), dtype=np.float64)
    if any(row["reused_frozen"] for row in metadata.values()):
        with np.load(asset(workspace, *FROZEN)) as cache:
            np.testing.assert_array_equal(cache["index"], AXIS)
            for k, row in metadata.items():
                if row["reused_frozen"]:
                    result[:, cols.index(k)] = cache["score"]
    initial = {key: state_hashes(model) for key, model in models.items()}
    if models:
        keys = list(models)
        base = models[keys[0]]
        angle = []
        hook = base.net.register_forward_hook(lambda module, args, value: angle.append(value.squeeze()))
        part_size = 2048
        for lo in range(0, len(AXIS), part_size):
            part = out / f"part_{lo:05d}.npz"
            receipt = part.with_suffix(".json")
            ix = AXIS[lo:lo+part_size]
            if receipt.is_file():
                info = read(receipt)
                assert info["all_source_seal_sha256"] == seal_sha
                with np.load(asset(out, part.name, info["sha256"])) as saved:
                    np.testing.assert_array_equal(saved["index"], ix)
                    np.testing.assert_array_equal(saved["keys"], keys)
                    result[lo:lo+len(ix), [cols.index(k) for k in keys]] = saved["score"]
                continue
            scores = torch.empty((len(ix), len(keys)), device=device)
            ends = torch.empty((len(ix), len(keys), 10, 25), device=device)
            with torch.inference_mode():
                for i, t in enumerate(ix):
                    window = x[t-99:t+1].unsqueeze(0)
                    angle.clear()
                    with fixed_native_rng(device):
                        enc, primary = base(window, norm=0, mask_mode="c", copies=10)
                    for j, key in enumerate(keys):
                        model = models[key]
                        if j == 0:
                            decoded = primary
                        else:
                            decoded = ifrft(model.decoder(enc), angle[0], dim=-1).abs().reshape(10,1,25,100).permute(0,1,3,2)
                            if lo == 0 and i == 0:
                                with fixed_native_rng(device):
                                    _, direct = model(window, norm=0, mask_mode="c", copies=10)
                                torch.testing.assert_close(decoded, direct, atol=0, rtol=0)
                        scores[i,j] = model.cal_anomaly_score(window, decoded, None, 1)[0,99]
                        ends[i,j] = decoded[:,0,99]
            scores, ends = scores.cpu().numpy(), ends.cpu().numpy()
            np.testing.assert_allclose(scores, ends.astype(np.float64).var(2, ddof=1).mean(2), rtol=1e-5, atol=1e-7)
            np.savez(part, index=ix, keys=keys, score=scores, terminal_copies=ends)
            dump(receipt, dict(sha256=sha(part), all_source_seal_sha256=seal_sha))
            result[lo:lo+len(ix), [cols.index(k) for k in keys]] = scores
            print(int(ix[-1]), round(time.monotonic()-start,1), flush=True)
        hook.remove()
    assert all(state_hashes(models[k]) == v for k,v in initial.items()) and np.isfinite(result).all()
    path = out / "banks.npz"
    np.savez_compressed(path, index=AXIS, available_index=AXIS, window_start=AXIS-99,
                        cal_index=cal, suffix_index=SUFFIX, keys=cols, score=result)
    dump(out/"SCORES_FROZEN.json",dict(sha256=sha(path), all_source_seal_sha256=seal_sha, models=metadata,
        official_frozen_bank=FROZEN, labels_read=False, shared_B1_encoder=True, scaler_sha256=task["scaler_sha256"],
        target_sha256=TARGET_SHA, mask_seed=84551,
        terminal_copy_shards={p.with_suffix(".npz").name: read(p)["sha256"] for p in sorted(out.glob("part_*.json"))},
        gpu=torch.cuda.get_device_name(), slot=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device_elapsed_seconds=time.monotonic()-start, peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def evaluate(workspace, out):
    receipt = read(out/"SCORES_FROZEN.json")
    labels = np.load(asset(workspace, "data/common_psm_20261028_v1/test_label.npy", LABEL_SHA)).reshape(-1)
    metric = load_author_metric(asset(workspace, "preparations/native_pair_rank_psm_20261122_v1/left_author_metric.py", METRIC_SHA))
    ranked, operating, events = [], [], []
    with np.load(asset(out,"banks.npz",receipt["sha256"])) as z:
        np.testing.assert_array_equal(z["index"], AXIS)
        np.testing.assert_array_equal(z["suffix_index"], SUFFIX)
        ci = z["cal_index"]
        assert len(ci) == 512 and labels[ci].sum() == 512
        for i, key in enumerate(z["keys"]):
            data = z["score"][:,i]
            meta = receipt["models"][str(key)]
            binding = dict(dataset="PSM", entity="PSM", **meta)
            rank, _ = continuous_metrics(labels[AXIS], data, metric)
            ranked.append(dict(binding, axis_start=99, axis_stop=87841, points=len(AXIS), **rank))
            cal_scores = data[ci-99]
            tau = exact_cal(ci, cal_scores, ci)
            rows, details, _, _ = evaluate_score(SUFFIX, data[SUFFIX-99], labels, tau, SUFFIX,
                query_start=23016, query_stop=87744, original_length=len(labels), expected_events=None, ranking=None)
            causal_timing(rows,details)
            for row in rows:
                stream_events = [{k: "" if v is None else str(v) for k, v in e.items()} for e in details if e["stream"] == row["stream"]]
                row.update(group_operating([row], stream_events))
                operating.append(dict(row, **binding, axis_start=23016, axis_stop=87744, cal_points=len(ci),
                                      CAL_selected_attack_points=int(labels[ci].sum()), CAL_exceedances=int((cal_scores>tau).sum())))
            for event in details:
                events.append(dict(event, **binding))
    write_csv(out/"ranking.csv", ranked)
    write_csv(out/"operating.csv", operating)
    write_csv(out/"events.csv", events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["score", "evaluate"])
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    out = args.workspace / ROOT / "PSM"
    out.mkdir(parents=True, exist_ok=True)
    (score if args.action == "score" else evaluate)(args.workspace.resolve(), out.resolve())
