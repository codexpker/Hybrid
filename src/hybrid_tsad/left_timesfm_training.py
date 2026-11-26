"""Five matched source continuations with streaming frozen history features."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from . import left_timesfm_condition as architecture
from . import timesfm_hidden
from . import left_multisource as base
from . import left_smd_source as smd
from .intervention_reference import source_records
from .left_smd28_author import bindings

STEPS, BATCH = 2000, 32
SELECT_ROWS = 1024


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def source_data(w, domain):
    pools = smd.source_pools(w) if domain == "SMD" else source_records(w)
    identities = {}
    for role, groups in pools.items():
        ids = [
            [f"TEP/run{i}" for i in (range(1, 21) if role == "fit" else range(61, 66))],
            [f"HAI/train{i}" for i in ((1, 3) if role == "fit" else (2,))]
            if domain == "SMD"
            else [f"SMD/{key}" for key in sorted(bindings(w))],
        ]
        identities[role] = [
            [
                dict(
                    record_id=name,
                    role=role,
                    shape=list(x.shape),
                    normalized_sha256=array_hash(x),
                )
                for name, x in zip(names, group, strict=True)
            ]
            for names, group in zip(ids, groups, strict=True)
        ]
    return pools, identities


def make_plan(groups, width, rows, seed):
    rng = np.random.default_rng(seed)
    # domain, record within domain, current_start in role, history length, C columns.
    plan = np.empty((rows, 4 + width), dtype=np.int32)
    for i in range(rows):
        group_id = i % 2
        group = groups[group_id]
        record_id = int(rng.integers(len(group)))
        x = group[record_id]
        history = 256 if group_id == 0 else 512
        assert len(x) >= history + 192
        current = int(rng.integers(history, len(x) - 192 + 1))
        columns = rng.choice(x.shape[1], width, replace=False)
        plan[i] = [group_id, record_id, current, history, *columns]
    # Reassign to earlier histories within each record, preserving channel identity.
    # The same rule remains legal when deployed sequentially on a target record.
    donor = plan.copy()
    shuffle = np.random.default_rng(np.random.SeedSequence([2025, seed, 371]))
    for i, row in enumerate(plan):
        minimum, current = int(row[3]), int(row[2])
        if current > minimum:
            donor[i, 2] = int(shuffle.integers(minimum, current))
    assert np.all(donor[:, 2] <= plan[:, 2])
    return plan, donor


def batch_arrays(groups, plan):
    n, width = len(plan), plan.shape[1] - 4
    history = np.zeros((n, 512, width), np.float32)
    current = np.empty((n, 192, width), np.float32)
    lengths = plan[:, 3].copy()
    for j, row in enumerate(plan):
        domain, record, start, count = map(int, row[:4])
        x, cols = groups[domain][record], row[4:]
        assert 0 <= start - count < start and start + 192 <= len(x)
        history[j, -count:] = x[start - count : start][:, cols]
        current[j] = x[start : start + 192][:, cols]
    return history, current, lengths


def tensors(groups, plan):
    return tuple(torch.as_tensor(v, device="cuda") for v in batch_arrays(groups, plan))


def feature_call(encoder, history, lengths, microbatch=32):
    features, masks = [], []
    for start in range(0, len(history), microbatch):
        z, valid = timesfm_hidden.encode(
            encoder,
            history[start : start + microbatch],
            lengths[start : start + microbatch],
        )
        features.append(z)
        masks.append(valid)
    return torch.cat(features), torch.cat(masks)


def inputs_for_step(groups, plan, donors, start, pretrained, random_encoder):
    rows = plan[start : start + BATCH]
    history, current, lengths = tensors(groups, rows)
    donor_rows = donors[start : start + BATCH]
    shuffled_history, _, shuffled_lengths = tensors(groups, donor_rows)
    official = feature_call(pretrained, history, lengths)
    shuffled = feature_call(pretrained, shuffled_history, shuffled_lengths)
    random = feature_call(random_encoder, history, lengths)
    return history, current, lengths, dict(C=official, D=shuffled, E=random)


def call(model, inputs):
    history, current, lengths, features = inputs
    z, valid = features.get(model.arm, (None, None))
    return model(current, history, lengths, z, valid)[0]


def checks(w, domain):
    started = time.monotonic()
    pools, ids = source_data(w, domain)
    plan, _ = make_plan(pools["select"], architecture.WIDTHS[domain], 2, 2026)
    history, current, lengths = tensors(pools["select"], plan)
    official = timesfm_hidden.load_encoder(w)
    z, mask = feature_call(official, history, lengths)
    del official
    records = {}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        control, _ = architecture.initial(w, domain, "A")
        control.eval()
        with torch.no_grad():
            old_loss = control(current, history, lengths)[0]
            old_score = control.infer(current, history, lengths)[0]
        for arm in ("B", "C"):
            model, _ = architecture.initial(w, domain, arm)
            model.eval()
            original = base.fingerprint(model.state_dict())
            step = model.left.step
            with torch.no_grad():
                new_loss = model(current, history, lengths, z, mask)[0]
                new_score = model.infer(current, history, lengths, z, mask)[0]
            torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
            torch.testing.assert_close(new_score, old_score, rtol=0, atol=0)
            assert (
                original == base.fingerprint(model.state_dict())
                and step == model.left.step
            )
            model.zero_grad(set_to_none=True)
            model(current, history, lengths, z, mask)[0].backward()
            gate_grad = float(model.condition.gate.grad)
            assert np.isfinite(gate_grad) and gate_grad != 0
            opt = torch.optim.Adam(model.parameters(), lr=1e-4)
            opt.step()
            opt.zero_grad(set_to_none=True)
            model(current, history, lengths, z, mask)[0].backward()
            proj_grad = float(model.condition.project.weight.grad.norm())
            assert np.isfinite(proj_grad) and proj_grad > 0
            records[arm] = dict(
                zero_loss_score_exact=True,
                gate_gradient=gate_grad,
                projection_gradient_after_one_step=proj_grad,
                added_parameters=sum(p.numel() for p in model.condition.parameters()),
                inference_frozen=True,
            )
            del model
    dest = w / architecture.OUT / domain
    dest.mkdir(parents=True, exist_ok=True)
    base.dump(
        dest / "CHECKS.json",
        dict(
            domain=domain,
            checks=records,
            source_identities=ids,
            normal_only=True,
            seconds=time.monotonic() - started,
        ),
    )


def train(w, domain):
    folder = w / architecture.OUT / domain
    assert (folder / "CHECKS.json").exists()
    if all((folder / a / "receipt.json").exists() for a in architecture.ARMS):
        for arm in architecture.ARMS:
            architecture.load_model(w, domain, arm)
        return
    assert not any((folder / a / "receipt.json").exists() for a in architecture.ARMS), (
        "Partial seal needs explicit recovery"
    )
    started = time.monotonic()
    torch.cuda.set_per_process_memory_fraction(
        20 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    pools, ids = source_data(w, domain)
    width = architecture.WIDTHS[domain]
    plan, donors = make_plan(pools["fit"], width, STEPS * BATCH, 2025)
    select, select_donors = make_plan(pools["select"], width, SELECT_ROWS, 2026)
    plans_path = folder / "PLANS.npz"
    if plans_path.exists():
        with np.load(plans_path) as old:
            for key, value in dict(
                fit=plan, fit_donor=donors, select=select, select_donor=select_donors
            ).items():
                np.testing.assert_array_equal(old[key], value)
    else:
        np.savez(
            plans_path,
            fit=plan,
            fit_donor=donors,
            select=select,
            select_donor=select_donors,
        )
    if (folder / "SOURCE.json").exists():
        assert base.read(folder / "SOURCE.json") == ids
    else:
        base.dump(folder / "SOURCE.json", ids)
    pretrained = timesfm_hidden.load_encoder(w)
    random_encoder = timesfm_hidden.load_encoder(w, random=True)
    pretrained_hash = base.fingerprint(pretrained.state_dict())
    random_hash = base.fingerprint(random_encoder.state_dict())
    models, opts, parents = {}, {}, {}
    for arm in architecture.ARMS:
        models[arm], parents[arm] = architecture.initial(w, domain, arm)
        opts[arm] = torch.optim.Adam(models[arm].parameters(), lr=1e-4)
        models[arm].train()
    common_rng = torch.cuda.get_rng_state().clone()
    start_rng = array_hash(common_rng.cpu().numpy())
    trace, losses, hashes = [], {a: [] for a in architecture.ARMS}, []
    resume = folder / "progress.pt"
    first_step, preceding_seconds = 0, 0.0
    if resume.exists():
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        assert saved["plan"] == base.sha(folder / "PLANS.npz")
        for arm, model in models.items():
            model.load_state_dict(saved["models"][arm])
            model.left.step = saved["native_step"][arm]
            opts[arm].load_state_dict(saved["optimizers"][arm])
        common_rng = saved["rng"]
        trace, hashes = saved["trace"], saved["hashes"]
        first_step, preceding_seconds = saved["completed"], saved["seconds"]
    for step in range(first_step, STEPS):
        inputs = inputs_for_step(
            pools["fit"], plan, donors, step * BATCH, pretrained, random_encoder
        )
        history, current, lengths, features = inputs
        record = dict(
            step=step + 1,
            history=array_hash(history.cpu().numpy()),
            current=array_hash(current.cpu().numpy()),
            **{a: array_hash(z.cpu().numpy()) for a, (z, _) in features.items()},
        )
        hashes.append(record)
        end_rng = None
        for arm, model in models.items():
            torch.cuda.set_rng_state(common_rng)
            opts[arm].zero_grad(set_to_none=True)
            before = model.left.step
            loss = call(model, inputs)
            loss.backward()
            opts[arm].step()
            assert model.left.step == before + 1
            assert torch.isfinite(loss)
            losses[arm].append(float(loss.detach()))
            if end_rng is None:
                end_rng = torch.cuda.get_rng_state().clone()
            else:
                assert torch.equal(end_rng, torch.cuda.get_rng_state())
        common_rng = end_rng
        if (step + 1) % 100 == 0:
            row = dict(
                step=step + 1,
                loss={a: float(np.mean(v)) for a, v in losses.items()},
                gate={
                    a: float(m.condition.gate.tanh().detach())
                    for a, m in models.items()
                    if m.condition is not None
                },
                seconds=time.monotonic() - started,
            )
            trace.append(row)
            print(domain, row, flush=True)
            losses = {a: [] for a in architecture.ARMS}
            temporary = folder / "progress.part"
            torch.save(
                dict(
                    models={a: m.state_dict() for a, m in models.items()},
                    optimizers={a: o.state_dict() for a, o in opts.items()},
                    native_step={a: m.left.step for a, m in models.items()},
                    rng=common_rng,
                    completed=step + 1,
                    trace=trace,
                    hashes=hashes,
                    plan=base.sha(folder / "PLANS.npz"),
                    seconds=preceding_seconds + time.monotonic() - started,
                ),
                temporary,
            )
            temporary.replace(resume)
        del inputs, features, history, current
    for model in models.values():
        model.eval()
    validation = {a: [] for a in architecture.ARMS}
    with torch.no_grad():
        for start in range(0, len(select), BATCH):
            inputs = inputs_for_step(
                pools["select"],
                select,
                select_donors,
                start,
                pretrained,
                random_encoder,
            )
            if start == 0:
                h, x, history_lengths, features = inputs
                torch.save(
                    dict(
                        history=h.cpu(),
                        current=x.cpu(),
                        lengths=history_lengths.cpu(),
                        features={
                            a: (z.cpu(), ok.cpu()) for a, (z, ok) in features.items()
                        },
                        rows=select[:BATCH],
                        donors=select_donors[:BATCH],
                    ),
                    folder / "SELECT_FEATURE_FIXTURE.pt",
                )
            for arm, model in models.items():
                validation[arm].append(float(call(model, inputs)))
    assert pretrained_hash == base.fingerprint(pretrained.state_dict())
    assert random_hash == base.fingerprint(random_encoder.state_dict())
    assert all(
        p.grad is None and not p.requires_grad
        for m in (pretrained, random_encoder)
        for p in m.parameters()
    )
    base.dump(folder / "FEATURE_HASHES.json", hashes)
    base.dump(folder / "CURVES.json", trace)
    elapsed = preceding_seconds + time.monotonic() - started
    for arm, model in models.items():
        dest = folder / arm
        dest.mkdir(exist_ok=True)
        torch.save(
            dict(state=model.state_dict(), step=model.left.step), dest / "selected.pt"
        )
        base.dump(
            dest / "receipt.json",
            dict(
                domain=domain,
                arm=arm,
                parent=parents[arm]["sha256"],
                sha256=base.sha(dest / "selected.pt"),
                plan_sha256=base.sha(folder / "PLANS.npz"),
                seed=2025,
                updates=STEPS,
                current_windows=STEPS * BATCH,
                endpoint=model.left.step,
                selection="fixed endpoint",
                source_select_loss=float(np.mean(validation[arm])),
                native_rng_start=start_rng,
                native_rng_end=array_hash(common_rng.cpu().numpy()),
                native_parameters=sum(p.numel() for p in model.left.parameters()),
                condition_parameters=sum(
                    p.numel() for p in model.condition.parameters()
                )
                if model.condition
                else 0,
                history_encoder_pretrained_sha=pretrained_hash,
                history_encoder_random_sha=random_hash,
                shared_fold_elapsed_seconds=elapsed,
                peak_bytes=torch.cuda.max_memory_allocated(),
                retained_feature_cache="streamed FP32, per-step hashes; exact plans retained",
                shuffled_singletons=int((donors[:, 2] == plan[:, 2]).sum()),
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("action", choices=("check", "train"))
    p.add_argument("domain", choices=architecture.WIDTHS)
    a = p.parse_args()
    torch.set_num_threads(1)
    (checks if a.action == "check" else train)(a.workspace, a.domain)
