"""Read-only native non-overlapping LEFT scores from frozen SWaT U artifacts.

No learner, checkpoint selection, scaler fit, labels or threshold logic occurs.
The historical split/scaler/model loader remains the exact pinned old worker.
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def window_starts(score_start, role_start, role_end, anchor=495000, width=192):
    first = anchor + ((score_start - anchor) // width) * width
    end = anchor + ((role_end - anchor) // width) * width
    starts = np.arange(first, end, width, dtype=np.int64)
    if not len(starts) or starts[0] < role_start or starts[-1] + width > role_end:
        raise ValueError("Native grid does not have legal complete role-contained windows")
    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["noU", "U_A", "U_B", "U_C"], required=True)
    args = parser.parse_args()
    output = args.output / args.arm
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    prep = args.workspace / "preparations/left_multievent_mixed_u_20261111_v1"
    source = prep / "ordinary_u_worker.py"
    assert sha(source) == "210fc350e433f4febe12744100b992bf4bbfb8e081a48c00c1501d415b476b47"
    spec = importlib.util.spec_from_file_location("original_frozen_worker", source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    torch = old.base.setup_torch(30)
    old.base.seed(torch)
    model = old.base.native(torch).cuda()
    if args.arm == "noU":
        identity = read(old.O / "left_train/RUNTIME_STATE.json")
        checkpoint = old.O / "left_train/model/SWAT/Left/checkpoints_0/checkpoint.pth"
    else:
        folder = old.O / "U" / args.arm[-1] / "train"
        identity = read(folder / "RECEIPT.json")
        checkpoint = folder / "endpoint2000.pth"
    assert sha(checkpoint) == identity["checkpoint_sha256"]
    scaler_path = prep / "FIT_ONLY_SCALER.json"
    assert sha(scaler_path) == "8ec724d8b12cebd0d70d33b92228b1f9de093b920c680d76e8e66d1609b3cc12"
    loaded = model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=False)
    assert not loaded.missing_keys and all(k.endswith(("total_ops", "total_params")) for k in loaded.unexpected_keys)
    model.step = identity["final_python_step"]
    model.eval()
    state_before, rng_before = old.hashes(model), old.rngs(torch)
    roles = old.base.RoleData(["CAL", "QUERY"])
    banks, details = [], []
    for role, first_score in [("CAL", 446173), ("QUERY", 608535)]:
        values, origin = roles.load(role)
        starts = window_starts(first_score, origin, origin + len(values))
        chunks = []
        with torch.no_grad():
            for at in range(0, len(starts), 128):
                batch_starts = starts[at:at+128]
                batch = np.stack([np.float32(values[s-origin:s-origin+192]) for s in batch_starts])
                raw = model.infer(torch.from_numpy(batch).cuda(), None, None, None)[0]
                assert raw.shape == (len(batch), 192, 31) and torch.isfinite(raw).all()
                chunks.append(raw.cpu().numpy())
        per_channel_full = np.concatenate(chunks).reshape(-1, 31)
        full_index = np.arange(starts[0], starts[-1]+192, dtype=np.int64)
        availability = np.repeat(starts+191, 192)
        mask = full_index >= first_score
        indices, per_channel, available = full_index[mask], per_channel_full[mask], availability[mask]
        score = np.mean(per_channel, axis=-1)
        assert score.dtype == np.float32
        selected = np.linspace(446173, 494999, 512, dtype=np.int64) if role == "CAL" else None
        missing = [] if selected is None else selected[~np.isin(selected, indices)].tolist()
        path = output / (role + ".npz")
        np.savez_compressed(path, index=indices, absolute_time_index=indices,
            available_index=available, score=score, canonical_score=score,
            per_channel=per_channel, valid=np.ones(len(indices), dtype=bool),
            window_start=starts, window_end_exclusive=starts+192,
            role_input_start=np.int64(origin), role_input_end=np.int64(origin+len(values)))
        banks.append(dict(role=role, relative=str(path.relative_to(args.output)), sha256=sha(path),
                          points=len(indices), axis=[int(indices[0]), int(indices[-1])+1]))
        details.append(dict(role=role, input=[origin, origin+len(values)], windows=len(starts),
            first_window=[int(starts[0]),int(starts[0])+192],
            last_window=[int(starts[-1]),int(starts[-1])+192],
            selected_CAL_missing=missing, original_CAL_selection_unchanged=True))
    assert state_before == old.hashes(model) and rng_before == old.rngs(torch)
    assert model.step == identity["final_python_step"]
    dump(output / "RECEIPT.json", dict(status="COMPLETED", arm=args.arm,
        checkpoint_sha256=sha(checkpoint), python_step=model.step,
        scaler_sha256=sha(scaler_path), historical_worker_sha256=sha(source),
        worker_sha256=sha(__file__), roles=roles.access, banks=banks, coverage=details,
        labels_read=False, training_updates=0, checkpoint_selection_changed=False,
        model_state_step_RNG_unchanged=True, window=192, stride=192, anchor=495000,
        score="native infer; original float32 channel mean; all window positions",
        availability="each actual non-overlapping input window end, inclusive",
        allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved(),
        worker_seconds=time.monotonic()-began))
    print(json.dumps(dict(event="COMPLETED", arm=args.arm, banks=banks)), flush=True)


if __name__ == "__main__":
    main()
