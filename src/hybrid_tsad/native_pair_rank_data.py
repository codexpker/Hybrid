"""Role-bound TimeRadar feature banks for the fixed SWaT/SMD to PSM pilot.

Only preparation invokes the frozen native model. Target CAL/QUERY readout
reuses sealed terminal reconstructions after the learned endpoints are frozen.
"""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .native_pair_rank import window_features
from .native_v_conditioning import SOURCE_SHA, WEIGHT_SHA, actual_mask_bits, fixed_native_rng, load_native, sha, state_hashes
from .native_v_conditioning_runner import dump, read, role_index, role_values


NATIVE_ROOT = Path("results/native_v_conditioning_20261119_v1")
SMD_BINDING = Path("preparations/timeradar_smd_full_20261116_v1/data_binding_full28.json")
SMD_SHA = "501184abd4dbd09e9d18a60cc84c88de05eb77f088f706286bb111b28fb51c73"
TASK_SHA = {
    "SWAT": "f23a8976225cc9d73b44d205c208fbff8c5f732b84b2465d1984c2b0185d9217",
    "PSM": "f65df51946b65518a8e0ce8d40f58726a4a02e7a56d522ebbd07a9fc64df5b38",
}
SEAL_SHA = {
    ("SWAT", "SOURCE_FIT"): "971b0c9e081293999b289ead4c6bfe2710042b49e59132d9097555042aef89d0",
    ("SWAT", "SOURCE_SELECT"): "9463e8842bba1e0a99810385fadde3edc6c35a0edf491e7bb66f36d8455b707a",
    ("PSM", "CAL"): "a5b03f8773addb446d3cc0e6b5a46b8bc6ab204eeb0984a902ab7449c66764e0",
    ("PSM", "QUERY"): "25be8aae2c16ec9ea4a507004f1c15e06a0caca47b38906908975dcda07fb502",
}
SMD_ENTITIES = [f"machine-{group}-{ordinal}" for group, n in [(1, 8), (2, 9), (3, 11)] for ordinal in range(1, n + 1)]
WINDOW = 100
SEED = 2025
MASK_SEED = 84551
U_FIT_BOUNDS = [0, 17568]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def task_binding(workspace, domain):
    folder = workspace / NATIVE_ROOT / domain
    path = folder / "TASK.json"
    require(sha(path) == TASK_SHA[domain], f"Unexpected native {domain} task identity")
    task = read(path)
    require(task["domain"] == domain and task["native_window"] == WINDOW, "Native task axes mismatch")
    require(task["native"] == dict(source_commit="4e06d0ad2bbe4c7e349c5a230fa5787d186e1eb2",
                                   source_sha256=SOURCE_SHA, checkpoint_sha256=WEIGHT_SHA,
                                   norm=0, copies=10, mask_mode="c", mask_seed=MASK_SEED, B1_only=True,
                                   criterion=None, L=1, terminal=WINDOW - 1, correction=1),
            "Native B1 mask or variance convention differs from the locked detector")
    return folder, task


def sample_positions(length, count):
    require(length >= count > 0, "Insufficient candidates for fixed sample size")
    return np.arange(count, dtype=np.int64) * (length - 1) // max(count - 1, 1)


def collect_windows(values, lo, ends):
    offsets = ends - lo
    require(values.ndim == 2 and ends.ndim == 1 and len(ends) > 0, "Illegal window sample shape")
    require(offsets[0] >= WINDOW - 1 and offsets[-1] < len(values) and np.all(np.diff(offsets) > 0),
            "Sample footprints must be valid, unique, ordered and contained in the role")
    indices = offsets[:, None] - np.arange(WINDOW - 1, -1, -1, dtype=np.int64)
    return np.asarray(values[indices], dtype=np.float32)


def variance_from_copies(copies):
    require(copies.ndim == 3 and copies.shape[1] == 10 and copies.dtype == np.float32,
            "Expected 10 float32 terminal reconstructions per record and channel")
    return copies.astype(np.float64).var(axis=1, ddof=1).astype(np.float32)


def official_variance(model, windows, device):
    variance = np.empty((len(windows), windows.shape[-1]), dtype=np.float32)
    max_error = 0.0
    with torch.inference_mode():
        for i, window in enumerate(windows):
            x = torch.from_numpy(window).to(device).unsqueeze(0)
            with fixed_native_rng(device):
                _, copies = model(x, norm=0, copies=10, mask_mode="c")
                score = model.cal_anomaly_score(x, copies, anomaly_criterion=None, L=1)
            require(tuple(copies.shape) == (10, 1, WINDOW, windows.shape[-1]), "Native copy shape changed")
            terminal = copies[:, 0, WINDOW - 1, :].cpu().numpy()[None]
            variance[i] = variance_from_copies(terminal)[0]
            reference = float(score[0, WINDOW - 1])
            actual = float(variance[i].mean(dtype=np.float64))
            require(np.isclose(actual, reference, rtol=1e-5, atol=1e-7), "Native terminal score mismatch")
            max_error = max(max_error, abs(actual - reference))
    return variance, max_error


def old_bank(workspace, domain, role, task):
    path = workspace / NATIVE_ROOT / "cache" / domain / role / "SEALED.json"
    require(sha(path) == SEAL_SHA[domain, role], "Existing replica bank seal differs from the fixed binding")
    seal = read(path)
    require(seal["status"] == "COMPLETED" and seal["domain"] == domain and seal["role"] == role
            and seal["scaler_sha256"] == task["scaler_sha256"] and seal["labels_read"] is False,
            "Existing replica bank has incompatible roles or permissions")
    for part in seal["parts"]:
        receipt = workspace / part["relative"]
        require(sha(receipt) == part["sha256"], "Existing shard receipt hash mismatch")
        receipt = read(receipt)
        require(receipt["source_sha256"] == SOURCE_SHA and receipt["checkpoint_sha256"] == WEIGHT_SHA
                and receipt["mask_seed"] == MASK_SEED and receipt["B1_only"]
                and receipt["channel_order"] == task["channel_order"]
                and receipt["mask_bits_sha256"] == seal["mask_bits_sha256"],
                "Incompatible per-window TimeRadar model/scoring identity")
    bank = workspace / seal["relative"]
    require(sha(bank) == seal["sha256"], "Existing replica bank hash mismatch")
    return bank, seal


def cached_features_input(workspace, domain, role, folder, task, count=None):
    bank, seal = old_bank(workspace, domain, role, task)
    expected = role_index(workspace, folder, task, role)
    with np.load(bank, allow_pickle=False) as archive:
        all_index = archive["absolute_time_index"]
        require(np.array_equal(all_index, expected) and len(all_index) == seal["count"]
                and np.array_equal(archive["available_index"], all_index)
                and np.array_equal(archive["window_start"], all_index - WINDOW + 1)
                and np.array_equal(archive["window_end_exclusive"], all_index + 1),
                "Existing replica bank has incompatible indices or footprints")
        positions = np.arange(len(all_index)) if count is None else sample_positions(len(all_index), count)
        index = all_index[positions]
        copies = archive["reconstruction_terminal"][positions]
        current = archive["current_normalized"][positions]
        native_v = archive["V"][positions]
    values, _, lo, record = role_values(workspace, folder, task, role)
    require(record["record_id"] == seal["record_id"] and seal["record_id"] == task["indexes"][role]["record_id"],
            "Existing replica bank has incompatible record identity")
    require(np.array_equal(values[index - lo], current), "Replica bank and raw role normalization disagree")
    variance = variance_from_copies(copies)
    require(np.allclose(native_v, variance.mean(axis=-1), rtol=1e-5, atol=1e-7), "Cached native V mismatch")
    return index, values, lo, variance, native_v, seal


def source_corruption(windows, entity, role):
    n, _, channels = windows.shape
    entropy = int.from_bytes(hashlib.sha256(f"{entity}/{role}".encode()).digest()[:4], "little")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, entropy]))
    selected = rng.integers(channels, size=n)
    signs = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
    u = np.minimum(np.arange(32, dtype=np.float64) / 23, 1)
    ramp = (0.25 * (1 - np.cos(np.pi * u))).astype(np.float32)
    positive = windows.copy()
    for i, channel in enumerate(selected):
        positive[i, -32:, channel] += signs[i] * ramp
    return positive, selected, signs


def feature_array(windows, variance):
    features = window_features(windows, variance)
    require(features.shape == (len(windows), windows.shape[-1], 7) and features.dtype == np.float32
            and np.isfinite(features).all(), "Model feature interface changed")
    return features


def write_pair(output, entity, role, index, windows, normal_variance, model, device, identity):
    positive, channel, sign = source_corruption(windows, entity, role)
    positive_v, positive_error = official_variance(model, positive, device)
    clean_error = None
    if normal_variance is None:
        normal_variance, clean_error = official_variance(model, windows, device)
    normal = feature_array(windows, normal_variance)
    positive = feature_array(positive, positive_v)
    path = output / "stage1" / role / f"{entity}.npz"
    np.savez_compressed(path, normal=normal, positive=positive, index=index, synthetic_channel=channel, synthetic_sign=sign)
    return dict(identity, path=path.relative_to(output).as_posix(), sha256=sha(path), windows=len(index),
                native_calls=len(index) * (1 + (clean_error is not None)),
                max_native_check_error=max(positive_error, clean_error or 0),
                first_end=int(index[0]), last_end=int(index[-1]), synthetic_channels=channel.tolist(), synthetic_signs=sign.tolist())


def smd_examples(workspace, record, role, count):
    binding, = [r for r in record["roles"] if r["role"] == role]
    scaler = record["scaler"]
    require(binding["run"] == "train" and binding["dataset"] == "SMD"
            and binding["entity"] == record["id"], "SMD must read only train-side roles")
    require(sha(workspace / binding["path"]) == binding["sha256"]
            and sha(workspace / scaler["path"]) == scaler["sha256"], "SMD role or pre-fitted scaler mismatch")
    lo, hi = binding["bounds_half_open"]
    with np.load(workspace / binding["path"], allow_pickle=False) as archive:
        x, index = archive["x"], archive["index"]
    require(x.dtype == np.float32 and list(x.shape) == binding["shape"] and x.shape == (hi - lo, 38)
            and np.array_equal(index, np.arange(lo, hi)), "SMD archived normalized role malformed")
    ends = index[WINDOW - 1 + sample_positions(len(x) - WINDOW + 1, count)]
    return ends, collect_windows(x, lo, ends), dict(role=role, domain="SMD", entity=record["id"],
                                                  bounds=[lo, hi], data_sha256=binding["sha256"],
                                                  scaler_sha256=scaler["sha256"], cached_normal=False)


def prepare(workspace, output, device):
    device = torch.device(device)
    require(device.type == "cuda" and (device.index is None or device.index == 0),
            "Preparation must use one isolated visible CUDA device and the archived CUDA masks")
    require(torch.cuda.device_count() == 1, "Set CUDA_VISIBLE_DEVICES to one idle physical device")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(20 * 1024 ** 3 / torch.cuda.get_device_properties(device).total_memory, 0)
    require(sha(workspace / SMD_BINDING) == SMD_SHA, "Unexpected 28-entity SMD binding")
    smd = {record["id"]: record for record in read(workspace / SMD_BINDING)["records"]}
    require(set(smd) == set(SMD_ENTITIES), "Changing the fixed SMD28 entity inclusion is prohibited")
    psm_folder, psm_task = task_binding(workspace, "PSM")
    swat_folder, swat_task = task_binding(workspace, "SWAT")

    output.mkdir(parents=True, exist_ok=False)
    for role in ("FIT", "SELECT"):
        (output / "stage1" / role).mkdir(parents=True)
    started = time.monotonic()
    model = load_native(workspace, device)
    before = state_hashes(model)
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state(device).clone()
    pairs = []
    masks = {str(c): hashlib.sha256(actual_mask_bits(c, device).tobytes()).hexdigest() for c in (25, 31, 38)}
    for role, swat_count, smd_count in (("FIT", 112, 4), ("SELECT", 56, 2)):
        source_role = "SOURCE_" + role
        index, values, lo, variance, _, seal = cached_features_input(workspace, "SWAT", source_role,
                                                                       swat_folder, swat_task, swat_count)
        require(masks["31"] == seal["mask_bits_sha256"], "Current CUDA mask does not match SWaT cached input")
        windows = collect_windows(values, lo, index)
        pairs.append(write_pair(output, "SWAT", role, index, windows, variance, model, device,
                                dict(domain="SWAT", entity=swat_task["entity"], role=source_role,
                                     bounds=swat_task["roles"][source_role], data_sha256=swat_task["data"]["source_train"]["sha256"],
                                     scaler_sha256=swat_task["scaler_sha256"], cached_normal=True, cached_bank_sha256=seal["sha256"])))
        for entity in SMD_ENTITIES:
            index, windows, identity = smd_examples(workspace, smd[entity], role, smd_count)
            pairs.append(write_pair(output, f"SMD_{entity}", role, index, windows, None, model, device, identity))
        print(f"Prepared SOURCE_{role} with SWaT plus all SMD28", flush=True)

    psm_task["roles"]["U_FIT"] = U_FIT_BOUNDS
    values, _, lo, record = role_values(workspace, psm_folder, psm_task, "U_FIT")
    index = lo + WINDOW - 1 + sample_positions(len(values) - WINDOW + 1, 224)
    windows = collect_windows(values, lo, index)
    variance, error = official_variance(model, windows, device)
    u_path = output / "U_FIT.npz"
    np.savez_compressed(u_path, index=index, features=feature_array(windows, variance))
    require(before == state_hashes(model) and all(not p.requires_grad and p.grad is None for p in model.parameters()),
            "Native frontend updated during feature extraction")
    require(torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_cuda, torch.cuda.get_rng_state(device)),
            "Fixed native masks changed surrounding RNG state")
    dump(output / "PREPARED.json", dict(status="PREPARED_SOURCE_AND_U_FIT_ONLY", seed=SEED, window=WINDOW,
         native=dict(weight_sha256=WEIGHT_SHA, source_sha256=SOURCE_SHA, mask_seed=MASK_SEED, mask_bits_sha256=masks,
                     B1=True, norm=0, copies=10, correction=1, original_frozen=True),
         stage1_domains=["SWAT", "SMD"], stage1_FIT_pairs=224, stage1_SELECT_pairs=112, source_records=pairs,
         synthetic="single-channel +/-0.5 standardized raised-cosine step over positions 68..99; selection independent of signal",
         U_FIT=dict(path=u_path.name, sha256=sha(u_path), windows=len(index), bounds=U_FIT_BOUNDS,
                    record_id=record["record_id"], scaler_sha256=psm_task["scaler_sha256"],
                    raw_declared_sha256=record["sha256"], first_end=int(index[0]), last_end=int(index[-1]),
                    max_native_check_error=error),
         u_normalization="Archived target nominal SOURCE_FIT scaler, also disclosed for Z",
         native_calls=sum(pair["native_calls"] for pair in pairs) + len(index),
         reused_native_windows=168, elapsed_seconds=time.monotonic() - started,
         peak_reserved_bytes=torch.cuda.max_memory_reserved(device), labels_read=False, cal_query_read=False))


def readout(workspace, output, freeze_manifest):
    prepared = read(output / "PREPARED.json")
    require(prepared["status"] == "PREPARED_SOURCE_AND_U_FIT_ONLY" and prepared["stage1_domains"] == ["SWAT", "SMD"],
            "Readout requires the prepared multi-source fold")
    require(freeze_manifest == output / "MODELS_FROZEN.json", "Readout must use the fold's own model freeze")
    freeze = read(freeze_manifest)
    modes = {"Z", "U-Symmetric", "U-Logistic", "U-NoHistory"}
    require(freeze["status"] == "FROZEN_BEFORE_TARGET_SCORE" and freeze["seed"] == SEED
            and set(freeze["modes"]) == modes, "Model freeze lacks all fixed endpoints")
    for mode, endpoint in freeze["modes"].items():
        require(endpoint["file"] == mode + ".pt" and sha(output / endpoint["file"]) == endpoint["sha256"],
                "Model endpoint identity mismatch")
    started = time.monotonic()
    psm_folder, psm_task = task_binding(workspace, "PSM")
    reads = {}
    for role in ("CAL", "QUERY"):
        index, values, lo, variance, native_v, seal = cached_features_input(workspace, "PSM", role, psm_folder, psm_task)
        require(prepared["native"]["mask_bits_sha256"]["25"] == seal["mask_bits_sha256"],
                "Target calibration/query masks differ from the adaptation feature masks")
        features = np.concatenate([feature_array(collect_windows(values, lo, ends), per_channel)
                                   for ends, per_channel in zip(np.array_split(index, (len(index) + 511) // 512),
                                                                np.array_split(variance, (len(index) + 511) // 512))])
        path = output / f"{role}.npz"
        with path.open("xb") as file:
            np.savez_compressed(file, features=features, index=index, native_v=native_v)
        reads[role] = dict(path=path.name, sha256=sha(path), reused_bank_sha256=seal["sha256"],
                           count=len(index), first_end=int(index[0]), last_end=int(index[-1]))
    dump(output / "READOUT.json", dict(status="PSM_CACHE_READOUT_COMPLETE", model_freeze_sha256=sha(freeze_manifest),
         roles=reads, native_calls=0, labels_read=False, elapsed_seconds=time.monotonic() - started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--device", default="cuda:0")
    readout_parser = commands.add_parser("readout")
    readout_parser.add_argument("--freeze-manifest", type=Path, required=True)
    opts = parser.parse_args()
    if opts.command == "prepare":
        prepare(opts.workspace, opts.output, opts.device)
    else:
        readout(opts.workspace, opts.output, opts.freeze_manifest)


if __name__ == "__main__":
    main()
