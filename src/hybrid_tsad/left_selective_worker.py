"""Bounded paired LEFT update driver; original workspace assets remain read-only.

Run with --workspace pointing to the research workspace. The pinned historical
worker supplies the original model, role loader, batch constructor and scoring.
Only the gradient objective is replaced by the selective-update implementation.
"""
import argparse
import copy
import datetime
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from hybrid_tsad.left_selective_update import selective_loss

OLD_PREP = "preparations/left_multievent_mixed_u_20261111_v1"
PLAN_SHA = "444e61197b6d4951853af6d6396c087d4decc2bc791f25ce52fce0b5ed71e0e6"
WORKER_SHA = "210fc350e433f4febe12744100b992bf4bbfb8e081a48c00c1501d415b476b47"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)


def load_old(workspace):
    path = workspace / OLD_PREP / "ordinary_u_worker.py"
    assert sha(path) == WORKER_SHA
    spec = importlib.util.spec_from_file_location("pinned_ordinary_u", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    assert sha(old.P / "LEARNER_PLAN.npz") == PLAN_SHA
    return old


def plan_data(old):
    data = old.base.RoleData(["U_FIT", "DONOR", "SELECT"])
    fit, _ = data.load("U_FIT")
    donor, _ = data.load("DONOR")
    val, _ = data.load("SELECT")
    with np.load(old.P / "LEARNER_PLAN.npz") as z:
        plan = {key: z[key].copy() for key in z.files}
    return data, fit, donor, val, plan


def train(torch, old, arm, out):
    condition, mode = arm.split("_")
    model, pt = old.initial(torch)
    model.train()
    assert all(p.requires_grad for p in model.parameters())
    startstep = model.step
    data, fit, donor, val, plan = plan_data(old)
    initial = dict(checkpoint_sha256=sha(pt), state_sha256=old.hashes(model),
                   rng=old.rngs(torch), source_step=startstep, plan_sha256=PLAN_SHA)
    dump(out / "INITIAL_IDENTITY.json", initial)
    np.savez_compressed(out / "INITIAL_RNG.npz", CPU=torch.get_rng_state().numpy(),
                        CUDA=torch.cuda.get_rng_state().cpu().numpy())
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(2025)
    selects = [old.select(torch, model, val, 0)]
    traces, losses, retained = [], [], []
    ema = {"t_calls": 0, "f_calls": 0, "t_vectors": 0, "f_vectors": 0}
    for prefix, bank in [("t", model.proto_t), ("f", model.proto_f)]:
        original = bank.ema_update_hard
        def count(z, nn_idx, momentum=.99, original=original, prefix=prefix):
            ema[prefix + "_calls"] += 1
            ema[prefix + "_vectors"] += int(z.shape[0])
            return original(z, nn_idx, momentum=momentum)
        bank.ema_update_hard = count
    for step in range(2000):
        x = old.constructor.native_batch(fit, donor, plan, step, condition)
        opt.zero_grad()
        loss, indices, each = selective_loss(model, torch.from_numpy(x).cuda(), mode,
                                              generator=generator)
        assert torch.isfinite(loss) and torch.isfinite(each).all()
        assert len(indices) == 103 and len(indices.unique()) == 103
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norm = float(torch.stack([g.detach().float().norm() for g in grads]).norm())
        opt.step()
        assert model.step == startstep + step + 1
        losses.append(each.detach().cpu().numpy().copy())
        retained.append(indices.detach().cpu().numpy().copy())
        traces.append(dict(update=step + 1, loss=float(loss.detach()),
                           gradient_norm=norm, rng_after=old.rngs(torch),
                           python_step=model.step))
        if (step + 1) % 500 == 0:
            selects.append(old.select(torch, model, val, step + 1))
            print(json.dumps(dict(event="UPDATE", arm=arm, update=step + 1,
                                  loss=traces[-1]["loss"], SELECT=selects[-1])), flush=True)
    torch.save(model.state_dict(), out / "endpoint2000.pth")
    torch.save(opt.state_dict(), out / "optimizer2000.pth")
    np.savez_compressed(out / "FINAL_RNG.npz", CPU=torch.get_rng_state().numpy(),
                        CUDA=torch.cuda.get_rng_state().cpu().numpy(),
                        retention_generator=generator.get_state().numpy())
    np.savez_compressed(out / "PER_SAMPLE_NATIVE_LOSS.npz", total=np.stack(losses))
    indices = np.stack(retained)
    keep_mask = np.zeros((2000, 128), dtype=bool)
    np.put_along_axis(keep_mask, indices, True, axis=1)
    np.savez_compressed(out / "RETAINED_SLOTS.npz", indices=indices, keep_mask=keep_mask,
                        update=np.arange(1, 2001))
    delta = {}
    for name, value in model.state_dict().items():
        group = name.split(".")[0]
        d = float((value.detach().cpu().float() - before[name].float()).norm())
        delta[group] = delta.get(group, 0) + d * d
    dump(out / "TRAIN_TRACE.json", dict(arm=arm, steps=traces, SELECT_diagnostics=selects,
         module_delta={k: v**.5 for k, v in delta.items()}, prototype_EMA=ema,
         source_PT_unchanged=sha(pt) == initial["checkpoint_sha256"], final_endpoint_only=True))
    dump(out / "RECEIPT.json", dict(status="COMPLETED", arm=arm, condition=condition,
         selection=mode, updates=2000, initial_step=startstep, final_python_step=model.step,
         checkpoint_sha256=sha(out / "endpoint2000.pth"), plan_sha256=PLAN_SHA,
         roles=data.access, labels_read=False, all_native_parameters_trainable=True,
         full_forward_batch=128, retained_per_update=103, no_prefilter_before_forward=True,
         PER_SAMPLE_NATIVE_LOSS_sha256=sha(out / "PER_SAMPLE_NATIVE_LOSS.npz"),
         RETAINED_SLOTS_sha256=sha(out / "RETAINED_SLOTS.npz")))


def score(torch, old, arm, out, root):
    old.base.seed(torch)
    model = old.base.native(torch).cuda()
    train_dir = root / arm / "train"
    receipt = json.loads((train_dir / "RECEIPT.json").read_text())
    pt = train_dir / "endpoint2000.pth"
    assert sha(pt) == receipt["checkpoint_sha256"]
    loaded = model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True), strict=False)
    assert not loaded.missing_keys and all(k.endswith(("total_ops", "total_params")) for k in loaded.unexpected_keys)
    model.step = receipt["final_python_step"]
    model.eval()
    before, rng = old.hashes(model), old.rngs(torch)
    data = old.base.RoleData(["CAL", "QUERY"])
    banks = []
    for role in ["CAL", "QUERY"]:
        values, start = data.load(role)
        ix = np.arange(start + 512, start + len(values))
        scores, channels = [], []
        with torch.no_grad():
            for a in range(0, len(ix), 256):
                ends = ix[a:a + 256] - start
                x = np.stack([np.float32(values[t-191:t+1]) for t in ends])
                raw = model.infer(torch.from_numpy(x).cuda(), None, None, None)[0].cpu().numpy()
                _, s = old.base.native_score.native_numpy_scores([raw], 31)
                scores.append(s.reshape(len(x), 192)[:, -1])
                channels.append(raw[:, -1])
        path = out / (role + ".npz")
        np.savez_compressed(path, index=ix, score=np.concatenate(scores),
                            per_channel_last=np.concatenate(channels), available_index=ix,
                            valid=np.ones(len(ix), bool))
        banks.append(dict(role=role, relative=str(path.relative_to(root)), sha256=sha(path), points=len(ix)))
    assert before == old.hashes(model) and rng == old.rngs(torch)
    dump(out / "RECEIPT.json", dict(status="COMPLETED", arm=arm, banks=banks,
         checkpoint_sha256=sha(pt), python_step=model.step, eval_state_modes_RNG_immutable=True,
         roles=data.access, labels_read=False))


def gate(torch, old, out):
    model, pt = old.initial(torch)
    model.train()
    _, fit, donor, _, plan = plan_data(old)
    x = torch.from_numpy(old.constructor.native_batch(fit, donor, plan, 0, "B")).cuda()
    snapshot = copy.deepcopy(model.state_dict())
    step = model.step
    cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    results = {}
    for mode in ["native", "all", "trim", "random"]:
        model.load_state_dict(snapshot)
        model.step = step
        model.train()
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        if mode == "native":
            loss = model(x, None, None, None)[0]
            indices = torch.arange(128, device=x.device)
        else:
            loss, indices, each = selective_loss(model, x, "trim" if mode == "all" else mode,
                keep=128 if mode == "all" else 103,
                generator=torch.Generator(device="cpu").manual_seed(2025))
            assert torch.isfinite(each).all()
        assert torch.isfinite(loss)
        loss.backward()
        grads = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(v).all() for v in grads.values())
        state = old.hashes(model)
        rng = old.rngs(torch)
        last = model.last_x_res.detach().cpu().clone() if hasattr(model, "last_x_res") else None
        opt.step()
        params = {k: p.detach().cpu().clone() for k, p in model.named_parameters()}
        results[mode] = dict(loss=float(loss.detach()), gradients=grads, state=state,
                             rng=rng, step=model.step, last=last, params=params,
                             indices=indices.cpu().tolist())
    n, a, t, r = [results[k] for k in ["native", "all", "trim", "random"]]
    assert n["gradients"].keys() == a["gradients"].keys()
    gd = max(float((n["gradients"][k] - a["gradients"][k]).abs().max()) for k in n["gradients"])
    pd = max(float((n["params"][k] - a["params"][k]).abs().max()) for k in n["params"])
    assert abs(n["loss"] - a["loss"]) <= 1e-6 and gd <= 1e-5 and pd <= 2e-6
    assert all(n[k] == a[k] == t[k] == r[k] for k in ["state", "rng", "step"])
    if n["last"] is not None:
        assert all(torch.equal(n["last"], result["last"]) for result in [a, t, r])
    assert len(t["indices"]) == len(set(t["indices"])) == 103
    assert len(r["indices"]) == len(set(r["indices"])) == 103
    assert t["indices"] != r["indices"]
    different = any(not torch.equal(t["gradients"][k], r["gradients"][k]) for k in t["gradients"])
    update_different = any(not torch.equal(t["params"][k], r["params"][k]) for k in t["params"])
    assert different and update_different
    dump(out / "RECEIPT.json", dict(status="PASS", batch=128, checkpoint_sha256=sha(pt),
         native_loss=n["loss"], keepall_loss=a["loss"], max_gradient_abs_difference=gd,
         max_parameter_update_abs_difference=pd, forward_state_RNG_step_last_x_res_exact=True,
         trim_random_distinct_gradients=different, trim_random_distinct_updates=update_different,
         trim_indices=t["indices"], random_indices=r["indices"], labels_read=False,
         discarded_gate_updates=4, plan_sha256=PLAN_SHA))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["gate", "train", "score"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=["B_trim", "B_random", "C_trim", "C_random"])
    args = parser.parse_args()
    old = load_old(args.workspace)
    out = args.root / "CUDA_GATE" if args.phase == "gate" else args.root / args.arm / args.phase
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch = old.base.setup_torch(30 if args.phase == "score" else 20)
    if args.phase == "gate":
        gate(torch, old, out)
    elif args.phase == "train":
        train(torch, old, args.arm, out)
    else:
        score(torch, old, args.arm, out, args.root)
    dump(out / "GPU_PEAK.json", dict(allocated=torch.cuda.max_memory_allocated(),
         reserved=torch.cuda.max_memory_reserved(), seconds=time.monotonic()-start,
         UTC_end=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(event="COMPLETED", phase=args.phase, arm=args.arm)), flush=True)


if __name__ == "__main__":
    main()
