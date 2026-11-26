"""Matched source replay for native and frozen-TimesFM-conditioned LEFT."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from . import left_timesfm_condition as architecture
from . import left_timesfm_training as training
from . import timesfm_hidden
from . import left_multisource as base
from .left_timesfm_evaluation import role_input

STEPS, HALF = 2000, 16
SOURCE_ROWS = STEPS * HALF


def cache_hash(array):
    digest = hashlib.sha256()
    for start in range(0, len(array), 32):
        digest.update(np.ascontiguousarray(array[start : start + 32]).tobytes())
    return digest.hexdigest()


def source_plan(w, domain):
    pools, ids = training.source_data(w, domain)
    plan, _ = training.make_plan(
        pools["fit"], architecture.WIDTHS[domain], SOURCE_ROWS, 2025
    )
    return pools["fit"], ids["fit"], plan


def source_cache(w, domain):
    """Shared normal-source features; only valid tokens occupy archival storage."""
    folder = w / architecture.OUT / domain / "U_SOURCE_FEATURES"
    folder.mkdir(exist_ok=True, parents=True)
    receipt = folder / "receipt.json"
    groups, ids, plan = source_plan(w, domain)
    if receipt.exists():
        metadata = base.read(receipt)
        assert metadata["plan_array_sha256"] == training.array_hash(plan)
        for name, item in metadata["files"].items():
            assert (folder / name).stat().st_size == item["bytes"]
        return
    started = time.monotonic()
    width = architecture.WIDTHS[domain]
    np.save(folder / "PLAN.npy", plan)
    arrays = [
        np.lib.format.open_memmap(
            folder / f"domain{d}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(SOURCE_ROWS // 2, width, 8 if d == 0 else 16, 1280),
        )
        for d in range(2)
    ]
    encoder = timesfm_hidden.load_encoder(w)
    fingerprint = timesfm_hidden.model_sha(encoder)
    hashes = []
    payload_hash = hashlib.sha256()
    for first in range(0, SOURCE_ROWS, 32):
        history, _, lengths = training.tensors(groups, plan[first : first + 32])
        features, valid = training.feature_call(encoder, history, lengths)
        data = features.cpu().numpy()
        # All 16 positions are reconstructed exactly on the valid mask by the
        # reader. Padded tokens have zero attention mass in the conditioner.
        expanded = np.zeros_like(data)
        for d in (0, 1):
            count = 8 if d == 0 else 16
            block = data[d::2, :, -count:]
            arrays[d][first // 2 : (first + len(data)) // 2] = block
            expanded[d::2, :, -count:] = block
        payload_hash.update(expanded.tobytes())
        hashes.append(
            dict(
                first=first,
                history_sha256=training.array_hash(history.cpu().numpy()),
                valid_token_sha256=training.array_hash(expanded),
            )
        )
        if (first + 32) % 2048 == 0:
            print(domain, "source cache", first + 32, SOURCE_ROWS, flush=True)
    for array in arrays:
        array.flush()
    assert fingerprint == timesfm_hidden.model_sha(encoder)
    files = {
        f"domain{d}.npy": dict(
            sha256=base.sha(folder / f"domain{d}.npy"),
            bytes=(folder / f"domain{d}.npy").stat().st_size,
        )
        for d in (0, 1)
    }
    base.dump(folder / "feature_hashes.json", hashes)
    base.dump(
        receipt,
        dict(
            domain=domain,
            source_records=ids,
            plan_array_sha256=training.array_hash(plan),
            plan_sha256=base.sha(folder / "PLAN.npy"),
            files=files,
            valid_payload_sha256=payload_hash.hexdigest(),
            encoder=fingerprint,
            seconds=time.monotonic() - started,
            rows=SOURCE_ROWS,
            precision="float32",
            source_labels_read=False,
            stored_padding="omitted; zero tokens reconstructed and excluded by attention mask",
        ),
    )


class SourceFeatures:
    def __init__(self, w, domain, plan):
        folder = w / architecture.OUT / domain / "U_SOURCE_FEATURES"
        self.meta = base.read(folder / "receipt.json")
        assert self.meta["plan_array_sha256"] == training.array_hash(plan)
        self.arrays = [
            np.load(folder / f"domain{d}.npy", mmap_mode="r") for d in (0, 1)
        ]
        self.width = plan.shape[1] - 4
        self.digest = hashlib.sha256()

    def take(self, start):
        out = np.zeros((HALF, self.width, 16, 1280), np.float32)
        valid = np.ones((HALF, self.width, 16), bool)
        for d in (0, 1):
            count = 8 if d == 0 else 16
            out[d::2, :, -count:] = self.arrays[d][start // 2 : (start + HALF) // 2]
            valid[d::2, :, : 16 - count] = False
        self.digest.update(out.tobytes())
        return torch.from_numpy(out).cuda(), torch.from_numpy(valid).cuda()


def target_windows(values, starts):
    history = values[starts[:, None] + np.arange(-512, 0)]
    current = values[starts[:, None] + np.arange(192)]
    return torch.from_numpy(history).cuda(), torch.from_numpy(current).cuda()


def checks(w, domain, entity):
    """One discarded matched real batch before the formal U run, without labels."""
    destination = w / architecture.OUT / domain / "U_CHECKS.json"
    if destination.exists():
        return
    started = time.monotonic()
    groups, _, plan = source_plan(w, domain)
    data, _, _, descriptor = role_input(w, domain, entity, "U")
    target_history, target_current = target_windows(data, np.array([512, 513]))
    source_history, source_current, source_lengths = training.tensors(groups, plan[:2])
    history = torch.cat((target_history, source_history))
    current = torch.cat((target_current, source_current))
    lengths = torch.cat((torch.full((2,), 512, device="cuda"), source_lengths))
    encoder = timesfm_hidden.load_encoder(w)
    fingerprint = timesfm_hidden.model_sha(encoder)
    features, valid = training.feature_call(encoder, history, lengths)
    compact = features.clone()
    compact[~valid] = 0
    record = {}
    for arm in ("A", "C"):
        model, parent = architecture.load_model(w, domain, arm)
        before = base.fingerprint(model.state_dict())
        before_step = model.left.step
        with torch.no_grad():
            full_score = model.infer(current, history, lengths, features, valid)[0]
            cached_score = model.infer(current, history, lengths, compact, valid)[0]
        torch.testing.assert_close(full_score, cached_score, rtol=0, atol=0)
        assert before == base.fingerprint(model.state_dict())
        assert before_step == model.left.step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss = model(current, history, lengths, compact, valid)[0]
        assert torch.isfinite(loss)
        loss.backward()
        assert any(
            p.grad is not None and bool(p.grad.abs().sum() > 0)
            for p in model.parameters()
        )
        optimizer.step()
        assert model.left.step == before_step + 1
        assert before != base.fingerprint(model.state_dict())
        record[arm] = dict(
            parent_sha256=parent["sha256"],
            native_step_advanced_once=True,
            initial_eval_state_unchanged=True,
            zero_padded_token_readout_identical=True,
            finite_training_loss=float(loss.detach()),
            parameter_update=True,
        )
        del model, optimizer
    assert fingerprint == timesfm_hidden.model_sha(encoder)
    assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
    base.dump(
        destination,
        dict(
            domain=domain,
            entity=entity,
            target=descriptor,
            checks=record,
            input_history_sha256=training.array_hash(history.cpu().numpy()),
            input_current_sha256=training.array_hash(current.cpu().numpy()),
            frozen_encoder_sha256=fingerprint,
            discarded_check_updates=True,
            labels_read=False,
            seconds=time.monotonic() - started,
        ),
    )


def train(w, domain, entity):
    dest = w / architecture.OUT / domain / "U" / entity
    dest.mkdir(exist_ok=True, parents=True)
    if all((dest / a / "receipt.json").exists() for a in ("A", "C")):
        return
    assert not any((dest / a / "receipt.json").exists() for a in ("A", "C")), (
        "Partial seal requires explicit recovery"
    )
    started = time.monotonic()
    torch.cuda.set_per_process_memory_fraction(
        20 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    groups, source_ids, plan = source_plan(w, domain)
    data, lo, _, descriptor = role_input(w, domain, entity, "U")
    rng = np.random.default_rng(np.random.SeedSequence(2025).spawn(3)[0])
    starts = rng.integers(512, len(data) - 192 + 1, size=(STEPS, HALF))
    unique, inverse = np.unique(starts, return_inverse=True)
    plan_path = dest / "PLANS.npz"
    if plan_path.exists():
        with np.load(plan_path) as previous:
            np.testing.assert_array_equal(previous["target_starts"], starts)
            np.testing.assert_array_equal(previous["source_plan"], plan)
    else:
        np.savez(plan_path, target_starts=starts, source_plan=plan)
    base.dump(dest / "INPUTS.json", dict(target=descriptor, source=source_ids))
    encoder = timesfm_hidden.load_encoder(w)
    encoder_fingerprint = timesfm_hidden.model_sha(encoder)
    cache_path = w / architecture.OUT / domain / "U_SOURCE_FEATURES" / "receipt.json"
    source_features = SourceFeatures(w, domain, plan) if cache_path.exists() else None
    cache_bytes = len(unique) * data.shape[1] * 16 * 1280 * 4
    target_features = None
    cache_tic = time.monotonic()
    if cache_bytes <= 30 * 1024**3:
        target_features = np.empty((len(unique), data.shape[1], 16, 1280), np.float32)
        for first in range(0, len(unique), 32):
            history, _ = target_windows(data, unique[first : first + 32])
            z, _ = training.feature_call(
                encoder, history, torch.full((len(history),), 512, device="cuda")
            )
            target_features[first : first + len(z)] = z.cpu().numpy()
        target_cache_sha = cache_hash(target_features)
    else:
        target_cache_sha = None
    cache_seconds = time.monotonic() - cache_tic
    models, optimizers, parents = {}, {}, {}
    for arm in ("A", "C"):
        models[arm], parents[arm] = architecture.load_model(w, domain, arm)
        models[arm].train()
        optimizers[arm] = torch.optim.Adam(models[arm].parameters(), lr=1e-4)
    common_rng = torch.cuda.get_rng_state().clone()
    initial_steps = {a: m.left.step for a, m in models.items()}
    first_step, previous_seconds = 0, 0.0
    trace, hashes = [], []
    progress = dest / "progress.pt"
    if progress.exists():
        saved = torch.load(progress, map_location="cpu", weights_only=False)
        assert saved["plan"] == base.sha(dest / "PLANS.npz")
        for arm, model in models.items():
            model.load_state_dict(saved["models"][arm])
            model.left.step = saved["native_step"][arm]
            optimizers[arm].load_state_dict(saved["optimizers"][arm])
        common_rng = saved["rng"]
        trace = saved["trace"]
        hashes = saved["hashes"]
        first_step = saved["completed"]
        previous_seconds = saved["seconds"]
        if source_features:
            # Resume validates the full sealed source payload once at completion.
            for i in range(first_step):
                source_features.take(i * HALF)
    loss_sums = {a: 0.0 for a in models}
    for step in range(first_step, STEPS):
        target_history, target_current = target_windows(data, starts[step])
        if target_features is None:
            target_z, target_mask = training.feature_call(
                encoder, target_history, torch.full((HALF,), 512, device="cuda")
            )
        else:
            target_z = torch.from_numpy(
                target_features[inverse.reshape(STEPS, HALF)[step]]
            ).cuda()
            target_mask = torch.ones(
                target_z.shape[:3], device="cuda", dtype=torch.bool
            )
        source_history, source_current, source_lengths = training.tensors(
            groups, plan[step * HALF : (step + 1) * HALF]
        )
        source_z, source_mask = (
            source_features.take(step * HALF)
            if source_features
            else training.feature_call(encoder, source_history, source_lengths)
        )
        history = torch.cat((target_history, source_history))
        current = torch.cat((target_current, source_current))
        lengths = torch.cat((torch.full((HALF,), 512, device="cuda"), source_lengths))
        features = torch.cat((target_z, source_z))
        valid = torch.cat((target_mask, source_mask))
        hashes.append(
            dict(
                step=step + 1,
                history=training.array_hash(history.cpu().numpy()),
                current=training.array_hash(current.cpu().numpy()),
                features=training.array_hash(features.cpu().numpy()),
            )
        )
        endpoint_rng = None
        for arm, model in models.items():
            torch.cuda.set_rng_state(common_rng)
            optimizers[arm].zero_grad(set_to_none=True)
            loss = model(current, history, lengths, features, valid)[0]
            assert torch.isfinite(loss)
            loss.backward()
            optimizers[arm].step()
            loss_sums[arm] += float(loss.detach())
            if endpoint_rng is None:
                endpoint_rng = torch.cuda.get_rng_state().clone()
            else:
                assert torch.equal(endpoint_rng, torch.cuda.get_rng_state())
        common_rng = endpoint_rng
        if (step + 1) % 100 == 0:
            row = dict(
                step=step + 1,
                loss={a: v / 100 for a, v in loss_sums.items()},
                gate=float(models["C"].condition.gate.tanh().detach()),
            )
            trace.append(row)
            loss_sums = {a: 0.0 for a in models}
            print(domain, entity, row, flush=True)
            torch.save(
                dict(
                    models={a: m.state_dict() for a, m in models.items()},
                    optimizers={a: o.state_dict() for a, o in optimizers.items()},
                    native_step={a: m.left.step for a, m in models.items()},
                    rng=common_rng,
                    trace=trace,
                    hashes=hashes,
                    completed=step + 1,
                    plan=base.sha(dest / "PLANS.npz"),
                    seconds=previous_seconds + time.monotonic() - started,
                ),
                dest / "progress.part",
            )
            (dest / "progress.part").replace(progress)
    assert encoder_fingerprint == timesfm_hidden.model_sha(encoder)
    if source_features:
        assert (
            source_features.digest.hexdigest()
            == source_features.meta["valid_payload_sha256"]
        )
    base.dump(dest / "CURVES.json", trace)
    base.dump(dest / "FEATURE_HASHES.json", hashes)
    elapsed = previous_seconds + time.monotonic() - started
    for arm, model in models.items():
        out = dest / arm
        out.mkdir(exist_ok=True)
        assert model.left.step == initial_steps[arm] + STEPS
        torch.save(
            dict(state=model.state_dict(), step=model.left.step), out / "selected.pt"
        )
        base.dump(
            out / "receipt.json",
            dict(
                domain=domain,
                entity=entity,
                arm=arm,
                seed=2025,
                parent=parents[arm]["sha256"],
                sha256=base.sha(out / "selected.pt"),
                updates=STEPS,
                initial_step=initial_steps[arm],
                endpoint=model.left.step,
                selection="fixed endpoint",
                target_exposure=STEPS * HALF,
                source_exposure=STEPS * HALF,
                labels_read=False,
                source_plan=training.array_hash(plan),
                target_plan=training.array_hash(starts),
                encoder_sha256=encoder_fingerprint,
                unique_target_windows=len(unique),
                target_feature_cache_bytes=cache_bytes
                if target_features is not None
                else 0,
                target_feature_sha256=target_cache_sha,
                target_cache_seconds=cache_seconds,
                shared_pair_seconds=elapsed,
                peak_bytes=torch.cuda.max_memory_allocated(),
                source_feature_cache=source_features.meta if source_features else None,
            ),
        )


def queue(w, first_domain):
    """Three fixed hardware lanes; readiness depends only on sealed artifacts."""
    from .left_smd28_author import bindings

    assert first_domain in ("PSM", "SWAT", "SMD")
    lane = ("PSM", "SWAT", "SMD").index(first_domain)
    status = w / architecture.OUT / f"U_QUEUE_{first_domain}.json"

    def wait_for(paths, phase):
        base.dump(status, dict(phase=phase, waiting_for=[str(p) for p in paths]))
        while not all(p.exists() for p in paths):
            time.sleep(20)

    if first_domain != "SMD":
        wait_for(
            [
                w / architecture.OUT / first_domain / a / "receipt.json"
                for a in architecture.ARMS
            ],
            "waiting for all source arms to seal",
        )
        base.dump(
            status, dict(phase="training", domain=first_domain, entity=first_domain)
        )
        checks(w, first_domain, first_domain)
        train(w, first_domain, first_domain)
    wait_for(
        [
            w / architecture.OUT / d / "U" / d / "C" / "receipt.json"
            for d in ("PSM", "SWAT")
        ]
        + [w / architecture.OUT / "SMD" / a / "receipt.json" for a in architecture.ARMS]
        + [w / architecture.OUT / "SMD" / "U_SOURCE_FEATURES" / "receipt.json"],
        "waiting for PSM/SWaT U and SMD source seals",
    )
    entities = sorted(bindings(w))
    if lane == 0:
        checks(w, "SMD", entities[0])
    else:
        wait_for(
            [w / architecture.OUT / "SMD" / "U_CHECKS.json"],
            "waiting for shared SMD check",
        )
    for entity in entities[lane::3]:
        base.dump(status, dict(phase="training", domain="SMD", entity=entity))
        train(w, "SMD", entity)
    base.dump(
        status,
        dict(
            phase="complete", first_domain=first_domain, smd_entities=entities[lane::3]
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("action", choices=("cache", "check", "train", "queue"))
    parser.add_argument("domain", choices=architecture.WIDTHS)
    parser.add_argument("entities", nargs="*")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "cache":
        source_cache(args.workspace, args.domain)
    elif args.action == "queue":
        queue(args.workspace, args.domain)
    elif args.action == "check":
        checks(args.workspace, args.domain, (args.entities or [args.domain])[0])
    else:
        for entity in args.entities or [args.domain]:
            checks(args.workspace, args.domain, entity)
            train(args.workspace, args.domain, entity)
