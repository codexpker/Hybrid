"""Matched source and normal-target learning for pre-view holdout LEFT."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from . import left_holdout as method
from . import left_multisource as base
from . import left_timesfm_condition as prior
from . import left_timesfm_training as prior_train
from . import timesfm_hidden

OUT = Path("results/left_holdout_history_20261127_v1")
ARMS = ("A", "B", "C")
BATCH, SOURCE_STEPS, EPOCHS = 32, 2000, 20


def write_latest(path, record):
    """Replace a rolling record atomically, leaving sealed records immutable."""
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    part.replace(path)


class Features:
    def __init__(self, w):
        self.parallel = os.environ.get("LEFT_HOLDOUT_PARALLEL_HISTORY") == "1"
        self.devices = {
            arm: torch.device("cuda", index if self.parallel else 0)
            for index, arm in enumerate(("B", "C"), 1)
        }
        self.encoders = {}
        for arm, device in self.devices.items():
            with torch.cuda.device(device):
                self.encoders[arm] = timesfm_hidden.load_encoder(
                    w, random=arm == "C", device=device
                )
        self.identities = {
            a: timesfm_hidden.model_sha(m) for a, m in self.encoders.items()
        }
        self.calls = 0
        self.windows = 0
        self.seconds = 0.0
        self.encoder_seconds = dict.fromkeys(self.encoders, 0.0)
        self.pool = ThreadPoolExecutor(max_workers=2) if self.parallel else None

    def extract(self, arm, past, lengths):
        device = self.devices[arm]
        with torch.cuda.device(device):
            start = time.monotonic()
            values = prior_train.feature_call(
                self.encoders[arm], past.to(device), lengths.to(device)
            )
            result = tuple(value.to(past.device) for value in values)
            torch.cuda.synchronize(device)
        return result, time.monotonic() - start

    def __call__(self, past, lengths):
        torch.cuda.synchronize()
        started = time.monotonic()
        if self.pool:
            pending = {
                arm: self.pool.submit(self.extract, arm, past, lengths)
                for arm in self.encoders
            }
            outputs = {arm: job.result() for arm, job in pending.items()}
        else:
            outputs = {arm: self.extract(arm, past, lengths) for arm in self.encoders}
        result = {}
        for arm, (value, seconds) in outputs.items():
            result[arm] = value
            self.encoder_seconds[arm] += seconds
        torch.cuda.synchronize()
        self.calls += len(self.encoders)
        self.windows += len(past) * len(self.encoders)
        self.seconds += time.monotonic() - started
        return result

    def receipt(self):
        if self.pool:
            self.pool.shutdown()
        assert self.identities == {
            a: timesfm_hidden.model_sha(m) for a, m in self.encoders.items()
        }
        return dict(
            calls=self.calls,
            windows=self.windows,
            seconds=self.seconds,
            state_sha256=self.identities,
            encoder_seconds=self.encoder_seconds,
            devices={a: str(d) for a, d in self.devices.items()},
        )


def condition(arm, past, lengths, features):
    return past, lengths, *features.get(arm, (None, None))


def archived_source(w, domain):
    old = w / prior.OUT / domain
    pools, identities = prior_train.source_data(w, domain)
    assert base.read(old / "SOURCE.json") == identities
    with np.load(old / "PLANS.npz") as z:
        plans = {name: z[name] for name in ("fit", "select")}
    return pools, plans, old


@torch.no_grad()
def prior_bypass(w, domain, old):
    model, ck = prior.load_model(w, domain, "C")
    fixture = torch.load(
        old / "SELECT_FEATURE_FIXTURE.pt", weights_only=False, map_location="cpu"
    )
    past, current, lengths = (
        fixture[k].cuda() for k in ("history", "current", "lengths")
    )
    context = past, lengths, *(t.cuda() for t in fixture["features"]["C"])
    state, step = base.fingerprint(model.state_dict()), model.left.step
    tokens = []
    handle = model.left.encoder.register_forward_hook(
        lambda _m, _args, output: tokens.append(output[0])
    )
    on_loss = model(current, *context)[0]
    on_score = model.infer(current, *context)[0]
    handle.remove()
    handles = list(model.left.encoder._forward_hooks.items())
    assert len(handles) == 1 and handles[0][1] == model._inject
    hook_id, hook = handles[0]
    del model.left.encoder._forward_hooks[hook_id]
    handle = model.left.encoder.register_forward_hook(
        lambda _m, _args, output: tokens.append(output[0])
    )
    try:
        off_loss = model(current, *context)[0]
        off_score = model.infer(current, *context)[0]
    finally:
        handle.remove()
        model.left.encoder._forward_hooks[hook_id] = hook
    assert state == base.fingerprint(model.state_dict()) and step == model.left.step
    assert len(tokens) == 4
    assert torch.equal(tokens[0], tokens[1]) and torch.equal(tokens[2], tokens[3])
    delta = on_score - off_score
    return dict(
        rows=len(current),
        checkpoint=ck["sha256"],
        fixture_sha256=base.sha(old / "SELECT_FEATURE_FIXTURE.pt"),
        native_forwards=4,
        frozen_step=step,
        gate=float(model.condition.gate.tanh()),
        residual_rms=float((tokens[0] - tokens[2]).square().mean().sqrt()),
        native_token_rms=float(tokens[2].square().mean().sqrt()),
        residual_to_token_rms=float(
            (tokens[0] - tokens[2]).norm() / tokens[2].norm().clamp_min(1e-12)
        ),
        normal_loss_on=float(on_loss),
        normal_loss_off=float(off_loss),
        score_mean_on=float(on_score.mean()),
        score_mean_off=float(off_score.mean()),
        score_delta_rms=float(delta.square().mean().sqrt()),
        score_delta_absmax=float(delta.abs().max()),
    )


def check(w, domain, pack, cached):
    pools, plans, old = pack
    began = time.monotonic()
    past, x, lengths = prior_train.tensors(pools["select"], plans["select"][:2])
    features = cached(past, lengths)
    masks = torch.stack(
        [method.hidden_positions(i, x.device) for i in range(method.MASKS)]
    )
    assert (masks.sum(0) == 1).all() and (masks.sum(1) == 48).all()
    stats = {}
    for arm in ARMS:
        model, _ = method.initial(w, domain, arm)
        model.eval()
        context = condition(arm, past, lengths, features)
        with torch.no_grad():
            # Use a nonzero gate in this throwaway check so equality exercises
            # the active conditioner and all three native prediction branches.
            if model.condition is not None:
                model.condition.gate.fill_(0.1)
            state, step = base.fingerprint(model.state_dict()), model.left.step
            for mask_id, mask in enumerate(masks):
                changed = x.clone()
                changed[:, mask] += 3.0
                _, pred = model.reconstruct(x, mask_id, *context)
                _, altered = model.reconstruct(changed, mask_id, *context)
                for lhs, rhs in zip(pred, altered, strict=True):
                    torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
                assert not torch.equal(
                    model.objective(x, mask_id, *context),
                    model.objective(changed, mask_id, *context),
                )
            assert (
                state == base.fingerprint(model.state_dict())
                and model.left.step == step
            )
        model.train()
        loss = model.objective(x, 0, *context)
        loss.backward()
        grad = None if arm == "A" else float(model.condition.project.weight.grad.norm())
        assert grad is None or np.isfinite(grad) and grad > 0
        assert model.left.step == step + 1
        stats[arm] = dict(
            exact_hidden_target_isolation=True,
            masks=4,
            condition_projection_gradient=grad,
            training_step_increment=1,
            native_forwards=17,
        )
    return dict(
        arms=stats,
        prior_C=prior_bypass(w, domain, old),
        seconds=time.monotonic() - began,
    )


def model_loss(model, arm, x, mask, past, lengths, features):
    if arm == "native":
        return model(x, None, None, None)[0]
    return model.objective(x, mask, *condition(arm, past, lengths, features))


def core(model, arm):
    return model if arm == "native" else model.left


def save(folder, arm, model, record):
    dest = folder / arm
    dest.mkdir(exist_ok=True, parents=True)
    ck = dest / "selected.pt"
    part = ck.with_suffix(".part")
    torch.save(dict(state=model.state_dict(), step=core(model, arm).step), part)
    part.replace(ck)
    record.update(sha256=base.sha(ck), native_step=core(model, arm).step)
    write_latest(dest / "receipt.json", record)


def load(w, domain, stage, arm, entity=None):
    folder = w / OUT / domain / stage
    if stage == "N":
        folder = folder / entity
    receipt = base.read(folder / arm / "receipt.json")
    ck = folder / arm / "selected.pt"
    assert base.sha(ck) == receipt["sha256"]
    if arm == "native":
        model, _ = prior.load_model(w, domain, "A")
        model = model.left
    else:
        model, _ = method.initial(w, domain, arm)
    state = torch.load(ck, weights_only=True, map_location="cpu")
    model.load_state_dict(state["state"])
    core(model, arm).step = state["step"]
    return model.eval(), receipt


def matched_update(models, optimizers, x, mask, past, lengths, features, common_rng):
    rng, losses = None, {}
    for arm, model in models.items():
        torch.cuda.set_rng_state(common_rng)
        opt = optimizers[arm]
        opt.zero_grad(set_to_none=True)
        before = core(model, arm).step
        loss = model_loss(model, arm, x, mask, past, lengths, features)
        loss.backward()
        opt.step()
        assert core(model, arm).step == before + 1
        losses[arm] = float(loss.detach())
        if not np.isfinite(losses[arm]):
            raise FloatingPointError((arm, losses[arm]))
        next_rng = torch.cuda.get_rng_state()
        if rng is None:
            rng = next_rng
        else:
            assert torch.equal(rng, next_rng)
    return rng, losses


@torch.no_grad()
def validation(models, batches):
    risk = dict.fromkeys(models, 0.0)
    n = 0
    for model in models.values():
        model.eval()
    for past, x, lengths, features in batches:
        n += len(x)
        for arm, model in models.items():
            masks = (0,) if arm == "native" else range(method.MASKS)
            risk[arm] += (
                len(x)
                * sum(
                    float(model_loss(model, arm, x, m, past, lengths, features))
                    for m in masks
                )
                / len(masks)
            )
    assert n and all(np.isfinite(list(risk.values())))
    return {k: v / n for k, v in risk.items()}, n


def train_source(w, domain):
    dest = w / OUT / domain / "Z"
    if (dest / "COMPLETE.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pack = archived_source(w, domain)
    pools, plans, old = pack
    feature = Features(w)
    check_file = dest / "CHECKS.json"
    if not check_file.exists():
        base.dump(check_file, check(w, domain, pack, feature))
    models, parents = {}, {}
    for arm in ARMS:
        models[arm], parents[arm] = method.initial(w, domain, arm)
        models[arm].train()
    assert len({p["sha256"] for p in parents.values()}) == 1
    assert base.fingerprint(models["B"].condition.state_dict()) == base.fingerprint(
        models["C"].condition.state_dict()
    )
    opts = {a: torch.optim.Adam(m.parameters(), lr=1e-4) for a, m in models.items()}
    rng = torch.cuda.get_rng_state()
    plan = plans["fit"]
    assert len(plan) == SOURCE_STEPS * BATCH
    old_hashes = base.read(old / "FEATURE_HASHES.json")
    trace, batch_receipts, intervals = [], [], []
    first, preceding = 0, 0.0
    progress = dest / "progress.pt"
    if progress.exists():
        ck = torch.load(progress, weights_only=False, map_location="cpu")
        assert ck["plan"] == base.sha(old / "PLANS.npz")
        for arm, model in models.items():
            model.load_state_dict(ck["models"][arm])
            model.left.step = ck["steps"][arm]
            opts[arm].load_state_dict(ck["opts"][arm])
        rng, trace, batch_receipts = ck["rng"], ck["trace"], ck["batches"]
        first, preceding, intervals = ck["completed"], ck["seconds"], ck["features"]
    for step in range(first, SOURCE_STEPS):
        past, x, lengths = prior_train.tensors(
            pools["fit"], plan[step * BATCH : (step + 1) * BATCH]
        )
        f = feature(past, lengths)
        expected = old_hashes[step]
        hashes = {
            a: prior_train.array_hash(value[0].cpu().numpy()) for a, value in f.items()
        }
        assert prior_train.array_hash(x.cpu().numpy()) == expected["current"]
        assert prior_train.array_hash(past.cpu().numpy()) == expected["history"]
        if feature.parallel:
            # Off-device execution is an engineering optimization only.
            assert hashes["B"] == expected["C"] and hashes["C"] == expected["E"]
        batch_receipts.append(
            dict(
                step=step + 1,
                **hashes,
                matches_archived_official=hashes["B"] == expected["C"],
                matches_archived_random=hashes["C"] == expected["E"],
            )
        )
        rng, loss = matched_update(
            models, opts, x, step % method.MASKS, past, lengths, f, rng
        )
        trace.append(dict(step=step + 1, **loss))
        if (step + 1) % 100 == 0:
            print(domain, "Z", step + 1, loss, flush=True)
            part = progress.with_suffix(".part")
            torch.save(
                dict(
                    models={a: m.state_dict() for a, m in models.items()},
                    opts={a: o.state_dict() for a, o in opts.items()},
                    rng=rng,
                    steps={a: m.left.step for a, m in models.items()},
                    trace=trace,
                    batches=batch_receipts,
                    completed=step + 1,
                    plan=base.sha(old / "PLANS.npz"),
                    seconds=preceding + time.monotonic() - started,
                    features=intervals
                    + [
                        dict(
                            calls=feature.calls,
                            windows=feature.windows,
                            seconds=feature.seconds,
                        )
                    ],
                ),
                part,
            )
            part.replace(progress)

    def select():
        for i in range(0, len(plans["select"]), BATCH):
            past, x, lengths = prior_train.tensors(
                pools["select"], plans["select"][i : i + BATCH]
            )
            yield past, x, lengths, feature(past, lengths)

    risk, n = validation(models, select())
    for arm, model in models.items():
        save(
            dest,
            arm,
            model,
            dict(
                arm=arm,
                parent=parents[arm]["sha256"],
                seed=2025,
                selected=SOURCE_STEPS,
                updates=SOURCE_STEPS,
                selection="fixed endpoint",
                source_plan_sha256=base.sha(old / "PLANS.npz"),
                normal_select_objective=risk[arm],
            ),
        )
    base.dump(dest / "CURVES.json", trace)
    base.dump(dest / "FEATURES.json", batch_receipts)
    base.dump(
        dest / "COMPLETE.json",
        dict(
            domain=domain,
            seconds=preceding + time.monotonic() - started,
            feature_intervals=intervals + [feature.receipt()],
            select_windows=n,
            native_forwards=SOURCE_STEPS * len(ARMS)
            + len(ARMS) * method.MASKS * (n // BATCH),
            current_presentations_each=SOURCE_STEPS * BATCH,
            peak_bytes=torch.cuda.max_memory_allocated(),
        ),
    )


def normal_input(w, domain, entity):
    if domain == "SMD":
        from . import left_smd_targets as smd

        rec = smd.smd.bindings(w)[entity]
        normal, scaler = smd.normal_scaler(w, rec, True)
        values, cut = scaler.transform(normal), int(len(normal) * 0.8)
    else:
        values, cut, scaler, _ = base.target_inputs(w, domain)
    return values.astype(np.float32), cut, scaler


def normal_cache(values, lo, hi, feature):
    starts = np.arange(lo + prior.HISTORY, hi - prior.WINDOW + 1, prior.WINDOW)
    if not len(starts):
        raise ValueError((lo, hi, "No legal normal windows"))
    pieces = {a: [] for a in ("B", "C")}
    identities = []
    for i in range(0, len(starts), BATCH):
        rows = starts[i : i + BATCH]
        past = torch.from_numpy(
            values[rows[:, None] + np.arange(-prior.HISTORY, 0)]
        ).cuda()
        lengths = torch.full(
            (len(rows),), prior.HISTORY, device="cuda", dtype=torch.long
        )
        for arm, (hidden, mask) in feature(past, lengths).items():
            assert bool(mask.all())
            pieces[arm].append(hidden.cpu())
        identities.extend(prior_train.array_hash(v) for v in past.cpu().numpy())
    current = torch.from_numpy(values[starts[:, None] + np.arange(prior.WINDOW)])
    return current, {a: torch.cat(v) for a, v in pieces.items()}, starts, identities


def normal_batch(cache, indices):
    current, hidden, _, _ = cache
    x = current[indices].cuda()
    mask = torch.ones((len(x), x.shape[2], 16), device="cuda", dtype=torch.bool)
    # The conditioned wrapper only uses the shape of this argument when
    # already-computed, exactly aligned hidden tokens are supplied.
    past_shape = x.new_empty((len(x), 0, x.shape[2]))
    lengths = torch.full((len(x),), prior.HISTORY, device="cuda", dtype=torch.long)
    return (
        past_shape,
        x,
        lengths,
        {a: (v[indices].cuda(), mask) for a, v in hidden.items()},
    )


def train_normal(w, domain, entity):
    dest = w / OUT / domain / "N" / entity
    if (dest / "COMPLETE.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    models, parents = {}, {}
    for arm in ARMS:
        models[arm], parents[arm] = load(w, domain, "Z", arm)
    native, parents["native"] = prior.load_model(w, domain, "A")
    models["native"] = native.left
    opts = {a: torch.optim.Adam(m.parameters(), lr=1e-4) for a, m in models.items()}
    values, cut, scaler = normal_input(w, domain, entity)
    feature = Features(w)
    fit = normal_cache(values, 0, cut, feature)
    select = normal_cache(values, cut, len(values), feature)
    feature_receipt = feature.receipt()
    del feature
    input_file = dest / "INPUTS.json"
    inputs = dict(
        FIT=fit[2].tolist(),
        SELECT=select[2].tolist(),
        role_boundary=cut,
        normalized_sha256=prior_train.array_hash(values),
        scaler_mean_sha256=prior_train.array_hash(scaler.mean_),
        scaler_scale_sha256=prior_train.array_hash(scaler.scale_),
        FIT_history_sha256=fit[3],
        SELECT_history_sha256=select[3],
        feature_encoder_state_sha256=feature_receipt["state_sha256"],
    )
    if input_file.exists():
        assert base.read(input_file) == inputs
    else:
        base.dump(input_file, inputs)
    generator = np.random.default_rng(2025)
    rng = torch.cuda.get_rng_state()
    updates = 0
    best = dict.fromkeys(models, float("inf"))
    trace = []
    plan = np.stack([generator.permutation(len(fit[0])) for _ in range(EPOCHS)])
    order = dest / "ORDER.npy"
    if order.exists():
        np.testing.assert_array_equal(np.load(order), plan)
    else:
        np.save(order, plan)
    scratch = os.environ.get("LEFT_HOLDOUT_N_PROGRESS")
    progress_dir = Path(scratch) / domain / entity if scratch else dest
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress = progress_dir / "progress.pt"
    initial_epoch, preceding = 1, 0.0
    feature_intervals = [feature_receipt]
    if progress.exists():
        ck = torch.load(progress, weights_only=False, map_location="cpu")
        assert ck["inputs"] == base.sha(input_file) and ck["order"] == base.sha(order)
        assert ck["parents"] == {a: r["sha256"] for a, r in parents.items()}
        for arm, model in models.items():
            model.load_state_dict(ck["models"][arm])
            core(model, arm).step = ck["steps"][arm]
            opts[arm].load_state_dict(ck["optimizers"][arm])
        rng, updates, trace, best = ck["rng"], ck["updates"], ck["trace"], ck["best"]
        initial_epoch = ck["completed"] + 1
        preceding = ck["seconds"]
        feature_intervals = ck["features"] + feature_intervals
    for epoch in range(initial_epoch, EPOCHS + 1):
        for model in models.values():
            model.train()
        totals = dict.fromkeys(models, 0.0)
        for i in range(0, len(fit[0]), BATCH):
            ids = plan[epoch - 1][i : i + BATCH]
            past, x, lengths, features = normal_batch(fit, ids)
            rng, loss = matched_update(
                models, opts, x, updates % method.MASKS, past, lengths, features, rng
            )
            updates += 1
            for arm, value in loss.items():
                totals[arm] += len(x) * value
        batches = (
            normal_batch(select, slice(i, i + BATCH))
            for i in range(0, len(select[0]), BATCH)
        )
        risk, n = validation(models, batches)
        for arm, value in risk.items():
            if value < best[arm]:
                best[arm] = value
                save(
                    dest,
                    arm,
                    models[arm],
                    dict(
                        arm=arm,
                        entity=entity,
                        seed=2025,
                        parent=parents[arm]["sha256"],
                        selected=epoch,
                        selected_updates=updates,
                        normal_select_objective=value,
                        selection="earliest minimum of epochs 1–20",
                        input_sha256=base.sha(dest / "INPUTS.json"),
                        order_sha256=base.sha(dest / "ORDER.npy"),
                    ),
                )
        record = dict(
            epoch=epoch,
            updates=updates,
            select=risk,
            fit={a: v / len(fit[0]) for a, v in totals.items()},
        )
        trace.append(record)
        write_latest(dest / "CURVES.json", trace)
        part = progress.with_suffix(".part")
        torch.save(
            dict(
                models={a: m.state_dict() for a, m in models.items()},
                optimizers={a: o.state_dict() for a, o in opts.items()},
                steps={a: core(m, a).step for a, m in models.items()},
                parents={a: r["sha256"] for a, r in parents.items()},
                inputs=base.sha(input_file),
                order=base.sha(order),
                rng=rng,
                updates=updates,
                trace=trace,
                best=best,
                completed=epoch,
                seconds=preceding + time.monotonic() - started,
                features=feature_intervals,
            ),
            part,
        )
        part.replace(progress)
        print(domain, entity, "N", record, flush=True)
    n = len(select[0])
    base.dump(
        dest / "COMPLETE.json",
        dict(
            domain=domain,
            entity=entity,
            seconds=preceding + time.monotonic() - started,
            history_feature_cost=feature_intervals,
            updates_each=updates,
            fit_windows_each=EPOCHS * len(fit[0]),
            select_windows_each=n,
            native_forwards=updates * len(models)
            + EPOCHS * ((n + BATCH - 1) // BATCH) * (len(ARMS) * method.MASKS + 1),
            peak_bytes=torch.cuda.max_memory_allocated(),
        ),
    )
    # Selected weights and all receipts are sealed. The rolling optimizer
    # checkpoint is only needed for interrupted runs.
    progress.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("domain", choices=prior.WIDTHS)
    parser.add_argument("action", choices=("source", "normal"))
    parser.add_argument("--entity")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "source":
        train_source(args.workspace, args.domain)
    else:
        train_normal(args.workspace, args.domain, args.entity or args.domain)
