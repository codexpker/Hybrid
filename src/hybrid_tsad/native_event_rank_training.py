"""Fixed source-supervised TimeRadar decoder comparison for the held-out PSM fold."""

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .native_event_rank import WINDOW, event_pair_loss, native_endpoint_and_copy_loss, source_event_roles
from .native_pair_rank_data import SMD_BINDING, SMD_ENTITIES, SMD_SHA, require, sample_positions, task_binding
from .native_v_conditioning import SOURCE_SHA, WEIGHT_SHA, actual_mask_bits, sha, state_hashes
from .native_v_conditioning_queue import run_jobs
from .native_v_conditioning_runner import dump, read, role_values, setup
from .native_v_replay import decoder_model, input_windows, optimizer


MODULE = "hybrid_tsad.native_event_rank_training"
ARMS = ("copy", "point_weighted", "event_balanced")
SEED = 2025
STEPS = 240
WARMUP = 24
SELECT_STEPS = range(0, STEPS + 1, 40)
LABEL_PATH = Path("results/left_current_u_exposure_20261118_v1/SWAT/BUNDLE/audit_only/TIME_LABEL_AUDIT.npz")
LABEL_SHA = "2a9fb52f192067dd258f73be4691678b1c42e09c3dc87d259fea616640818039"


def source_roles(workspace):
    folder, task = task_binding(workspace, "SWAT")
    require(task["roles"]["SOURCE_FIT"] == [0, 396000] and task["roles"]["SOURCE_SELECT"] == [396000, 495000],
            "Unexpected SWaT nominal source boundary")
    require(task["data"]["source_train"] == task["data"]["target_test"], "SWaT source events must share the same record")
    require(sha(workspace / LABEL_PATH) == LABEL_SHA, "Unexpected source-label slice")
    with np.load(workspace / LABEL_PATH, allow_pickle=False) as archive:
        labels = archive["U_FIT_time_labels"]
    require(labels.shape == (90000,), "Expected only the fixed labeled source slice")
    fit, select = source_event_roles(labels, 495000)
    require(fit.bounds == (495000, 502578) and select.bounds == (502578, 585000)
            and len(fit.normal_endpoints) == 4632 and len(select.normal_endpoints) == 79092,
            "Unexpected labeled source boundaries")
    require(sha(workspace / SMD_BINDING) == SMD_SHA, "Unexpected SMD28 binding")
    smd = {record["id"]: record for record in read(workspace / SMD_BINDING)["records"]}
    require(set(smd) == set(SMD_ENTITIES), "SMD28 entity list changed")
    return folder, task, fit, select, smd


def eligible(bounds):
    return np.arange(bounds[0] + WINDOW - 1, bounds[1], dtype=np.int64)


def smd_binding(record, role):
    binding, = [item for item in record["roles"] if item["role"] == role]
    require(binding["dataset"] == "SMD" and binding["entity"] == record["id"] and binding["run"] == "train",
            "Only train-side SMD source data is authorized")
    return binding


def prepare(workspace, output):
    _, task, fit, select, smd = source_roles(workspace)
    output.mkdir(parents=True, exist_ok=True)
    draws = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(6)]
    swat = eligible(task["roles"]["SOURCE_FIT"])
    swat_select = eligible(task["roles"]["SOURCE_SELECT"])
    smd_fit = [eligible(smd_binding(smd[entity], "FIT")["bounds_half_open"]) for entity in SMD_ENTITIES]
    smd_select = [eligible(smd_binding(smd[entity], "SELECT")["bounds_half_open"]) for entity in SMD_ENTITIES]
    domain = draws[1].integers(len(SMD_ENTITIES), size=(STEPS, 16))
    selected_smd = np.empty_like(domain)
    for entity_index, pool in enumerate(smd_fit):
        take = domain == entity_index
        selected_smd[take] = draws[2].choice(pool, size=int(take.sum()))

    fit_strata = [stratum for event in fit.positive_phases for stratum in event]
    select_strata = [stratum for event in select.positive_phases for stratum in event]
    select_sizes = np.asarray([min(16, len(stratum)) for stratum in select_strata], dtype=np.int64)
    select_positives = np.concatenate([stratum[sample_positions(len(stratum), count)]
                                       for stratum, count in zip(select_strata, select_sizes)])
    plan = {
        "swat_fit": draws[0].choice(swat, size=(STEPS, 16)),
        "smd_entity": domain,
        "smd_fit": selected_smd,
        "positive_fit": np.column_stack([draws[3].choice(stratum, size=STEPS) for stratum in fit_strata]),
        "normal_fit": draws[4].choice(fit.normal_endpoints, size=(STEPS, len(fit_strata))),
        "stratum_sizes": np.array([len(stratum) for stratum in fit_strata], dtype=np.int64).reshape(-1, 3),
        "scale_normal_fit": fit.normal_endpoints,
        "swat_select": swat_select[sample_positions(len(swat_select), 112)],
        "smd_select": np.stack([pool[sample_positions(len(pool), 4)] for pool in smd_select]),
        "positive_select": select_positives,
        "normal_select": draws[5].choice(select.normal_endpoints, size=len(select_positives), replace=False),
        "select_stratum_sizes": select_sizes,
    }
    require(len(fit.events) == 5 and len(select.events) == 4
            and sum(map(len, fit_strata)) == 1857 and sum(map(len, select_strata)) == 2439,
            "Unexpected event strata qualification")
    path = output / "PLAN.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **plan)
    dump(output / "PLAN.json", dict(status="PREPARED_BEFORE_MODEL_LOAD", seed=SEED, plan_sha256=sha(path),
         nominal_swat_fit=task["roles"]["SOURCE_FIT"], nominal_swat_select=task["roles"]["SOURCE_SELECT"],
         nominal_swat_scaler_sha256=task["scaler_sha256"], nominal_smd_binding_sha256=SMD_SHA,
         nominal_smd_entities=SMD_ENTITIES, source_label_sha256=LABEL_SHA, event_fit_bounds=fit.bounds,
         event_select_bounds=select.bounds, event_fit=fit.events, event_select=select.events,
         fit_positive_counts=plan["stratum_sizes"].tolist(), fit_normal_pool=len(fit.normal_endpoints),
         select_positive_counts=[len(stratum) for stratum in select_strata],
         select_paired_per_stratum=select_sizes.tolist(), select_normal_pool=len(select.normal_endpoints),
         steps=STEPS, warmup=WARMUP, select_steps=list(SELECT_STEPS),
         optics=dict(optimizer="AdamW", lr=1e-4, betas=[.9, .95], eps=1e-8,
                     weight_decay=.1, norm_bias_weight_decay=0, copy_batch=32, pair_margin=1, pair_weight=1,
                     nominal_window=100, norm=0, native_B=1, native_masks=84551, copies=10,
                     mode="eval", trainable="decoder only", selection="earliest minimum event-balanced source risk"),
         nominal_sampling="SWaT and SMD 16 each. SMD machine uniformly, then a legal endpoint uniformly.",
         permissions="Marked public SWaT source-event slice and nominal SWaT/SMD28 train roles only.",
         target_values_or_labels_read=False))


def load_plan(output):
    path = output / "PLAN.npz"
    require(sha(path) == read(output / "PLAN.json")["plan_sha256"], "Shared plan altered")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_smd(workspace, record, role):
    binding = smd_binding(record, role)
    scaler = record["scaler"]
    require(sha(workspace / binding["path"]) == binding["sha256"]
            and sha(workspace / scaler["path"]) == scaler["sha256"], "SMD role or scaler identity mismatch")
    lo, hi = binding["bounds_half_open"]
    with np.load(workspace / binding["path"], allow_pickle=False) as archive:
        x, index = archive["x"], archive["index"]
    require(x.dtype == np.float32 and x.shape == (hi - lo, 38) and list(x.shape) == binding["shape"]
            and np.array_equal(index, np.arange(lo, hi)) and np.isfinite(x).all(), "Malformed SMD nominal role")
    return x, lo


def event_input(workspace, folder, task, role):
    event_task = dict(task, roles={"EVENT_SOURCE": role.bounds})
    x, _, lo, _ = role_values(workspace, folder, event_task, "EVENT_SOURCE")
    return x, lo


def nominal_swat_input(workspace, folder, task, role):
    x, _, lo, _ = role_values(workspace, folder, task, "SOURCE_" + role)
    return x, lo


def windows(role_data, endpoints, device):
    values, lo = role_data
    return input_windows(values, lo, np.asarray(endpoints) - WINDOW + 1, device)


def native_values(model, role_data, endpoints, device):
    return [native_endpoint_and_copy_loss(model, x) for x in windows(role_data, endpoints, device)]


def mask_ids(device):
    return {str(c): hashlib.sha256(actual_mask_bits(c, device).tobytes()).hexdigest() for c in (31, 38)}


def reference_scale(workspace, output):
    started = time.monotonic()
    plan = load_plan(output)
    folder, task, fit, _, _ = source_roles(workspace)
    device = setup()
    model = decoder_model(workspace, device)
    source = event_input(workspace, folder, task, fit)
    with torch.inference_mode():
        values = np.asarray([float(value) for value, _ in native_values(model, source, plan["scale_normal_fit"], device)],
                            dtype=np.float64)
    require(len(values) == 4632 and np.isfinite(values).all() and (values >= 0).all(), "Invalid native scale scores")
    path = output / "FIT_NORMAL_V.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, V=values, absolute_endpoint=plan["scale_normal_fit"])
    dump(output / "SCALE.json", dict(status="COMPLETED", value=max(1e-8, float(np.median(values))),
         native_source_sha256=SOURCE_SHA, checkpoint_sha256=WEIGHT_SHA, mask_ids=mask_ids(device),
         count=len(values), bank_file=path.name, bank_sha256=sha(path), plan_sha256=sha(output / "PLAN.npz"),
         elapsed_seconds=time.monotonic() - started, peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
         target_read=False))
    print(f"Frozen scale from {len(values)} legal source normals in {time.monotonic()-started:.1f}s", flush=True)


def mean_scalar(values):
    return float(np.asarray(values, dtype=np.float64).mean())


def scalar_pairs(pairs):
    return np.asarray([(float(point), float(loss)) for point, loss in pairs], dtype=np.float64)


def variance_metrics(name, values, scale):
    values = np.asarray(values, dtype=np.float64)
    return {f"{name}_mean_V": mean_scalar(values), f"{name}_mean_V_over_a": mean_scalar(values / scale),
            f"{name}_mean_log1p_V_over_a": mean_scalar(np.log1p(values / scale))}


def validate(model, plan, select_data, scale, device, update):
    event, swat, smd = select_data
    with torch.inference_mode():
        positive = scalar_pairs(native_values(model, event, plan["positive_select"], device))[:, 0]
        normal = scalar_pairs(native_values(model, event, plan["normal_select"], device))[:, 0]
        nominal_swat = scalar_pairs(native_values(model, swat, plan["swat_select"], device))
        nominal_smd = np.concatenate([scalar_pairs(native_values(model, role, endpoint, device))
                                      for role, endpoint in zip(smd, plan["smd_select"])])
    losses = np.logaddexp(0, 1 - np.log1p(positive / scale) + np.log1p(normal / scale))
    split = np.cumsum(plan["select_stratum_sizes"])[:-1]
    stratum_risks = [mean_scalar(loss) for loss in np.split(losses, split)]
    return dict(update=update, risk=mean_scalar(stratum_risks), risks_event_by_stratum=np.asarray(stratum_risks).reshape(-1, 3).tolist(),
                nominal_swat_mean_copy_MSE=mean_scalar(nominal_swat[:, 1]),
                nominal_smd_mean_copy_MSE=mean_scalar(nominal_smd[:, 1]),
                nominal_mean_copy_MSE=mean_scalar(np.concatenate((nominal_swat[:, 1], nominal_smd[:, 1]))),
                **variance_metrics("labeled_positive", positive, scale), **variance_metrics("labeled_normal", normal, scale),
                **variance_metrics("nominal", np.concatenate((nominal_swat[:, 0], nominal_smd[:, 0])), scale))


def train_step(model, opt, plan, fit_data, scale, arm, device, step):
    event, swat, smd = fit_data
    row = step - 1
    opt.zero_grad(set_to_none=True)
    lr = 1e-4 * min(step / WARMUP, 1)
    for group in opt.param_groups:
        group["lr"] = lr
    recon = []
    for role, ends in [(swat, plan["swat_fit"][row])] + [
        (smd[entity], [end]) for entity, end in zip(plan["smd_entity"][row], plan["smd_fit"][row])
    ]:
        for x in windows(role, ends, device):
            point, loss = native_endpoint_and_copy_loss(model, x)
            (loss / 32).backward()
            recon.append((point.detach(), loss.detach()))
    metric = scalar_pairs(recon)
    results = dict(update=step, lr=lr, copy_MSE=mean_scalar(metric[:, 1]),
                   **variance_metrics("nominal_swat", metric[:16, 0], scale),
                   **variance_metrics("nominal_smd", metric[16:, 0], scale))
    if arm != "copy":
        positive = torch.stack([point for point, _ in native_values(model, event, plan["positive_fit"][row], device)]).reshape(-1, 3)
        normal = torch.stack([point for point, _ in native_values(model, event, plan["normal_fit"][row], device)]).reshape(-1, 3)
        pair_loss = event_pair_loss(positive, normal, scale,
                                    phase_sizes=plan["stratum_sizes"] if arm == "point_weighted" else None)
        pair_loss.backward()
        results.update(pair_loss=float(pair_loss.detach()),
                       **variance_metrics("labeled_positive", positive.detach().cpu().numpy(), scale),
                       **variance_metrics("labeled_normal", normal.detach().cpu().numpy(), scale))
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    grad_norm = float(torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in gradients])))
    require(np.isfinite(grad_norm) and grad_norm > 0 and all(p.grad is None for p in model.parameters() if not p.requires_grad),
            "Invalid decoder gradient or frozen gradient")
    opt.step()
    return dict(results, gradient_norm=grad_norm)


def training(workspace, output, arm):
    started = time.monotonic()
    plan = load_plan(output)
    scale_path = output / "SCALE.json"
    scale_record = read(scale_path)
    scale = scale_record["value"]
    require(scale_record["plan_sha256"] == sha(output / "PLAN.npz")
            and scale_record["checkpoint_sha256"] == WEIGHT_SHA
            and sha(output / scale_record["bank_file"]) == scale_record["bank_sha256"], "Reference scale identity mismatch")
    folder, task, fit, select, smd = source_roles(workspace)
    fit_data = (event_input(workspace, folder, task, fit), nominal_swat_input(workspace, folder, task, "FIT"),
                [load_smd(workspace, smd[entity], "FIT") for entity in SMD_ENTITIES])
    select_data = (event_input(workspace, folder, task, select), nominal_swat_input(workspace, folder, task, "SELECT"),
                   [load_smd(workspace, smd[entity], "SELECT") for entity in SMD_ENTITIES])
    arm_dir = output / "train" / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    device = setup()
    require(mask_ids(device) == scale_record["mask_ids"], "Native mask identity mismatch")
    model = decoder_model(workspace, device)
    initial = state_hashes(model)
    rng_cpu = torch.get_rng_state().clone()
    rng_cuda = torch.cuda.get_rng_state(device).clone()
    opt = optimizer(model)
    selection, training_trace = [], []
    best, best_state = None, None
    for step in range(STEPS + 1):
        if step:
            training_trace.append(train_step(model, opt, plan, fit_data, scale, arm, device, step))
        if step in SELECT_STEPS:
            metrics = validate(model, plan, select_data, scale, device, step)
            selection.append(metrics)
            if best is None or metrics["risk"] < best["risk"]:
                best = metrics
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            print(f"{arm} step={step} SELECT={metrics['risk']:.9f}, best={best['update']} ({time.monotonic()-started:.1f}s)", flush=True)
    final = state_hashes(model)
    require(all(initial[name] == final[name] for name in initial if not name.startswith("decoder."))
            and any(initial[name] != final[name] for name in initial if name.startswith("decoder."))
            and all(not module.training for module in model.modules())
            and torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_cuda, torch.cuda.get_rng_state(device)),
            "Training changed the native frozen state, modes or external RNG, or no update occurred")
    endpoint = arm_dir / "selected.pt"
    with endpoint.open("xb") as stream:
        torch.save(dict(model=best_state, update=best["update"], arm=arm), stream)
    trace_path = arm_dir / "TRAIN_TRACE.json"
    dump(trace_path, training_trace)
    dump(arm_dir / "SELECTION.json", dict(status="COMPLETED", arm=arm, seed=SEED, updates=STEPS, selected_step=best["update"],
         source_criterion="earliest minimum event-balanced paired SWaT source SELECT risk",
         optimizer_objective="copy MSE" if arm == "copy" else "copy MSE + unit-margin pair objective",
         event_labels_used_for_gradient=arm != "copy", event_labels_used_for_selection=True,
         native_source_sha256=SOURCE_SHA, initial_checkpoint_sha256=WEIGHT_SHA, selected_checkpoint_sha256=sha(endpoint),
         plan_sha256=sha(output / "PLAN.npz"), scale_sha256=sha(scale_path), scale_value=scale, mask_ids=scale_record["mask_ids"],
         training_trace_sha256=sha(trace_path), selection=selection, frozen_state_identical=True,
         fixed_B1=True, terminal=99, variance_unbiased=True, CUDA_visible=torch.cuda.get_device_name(device),
         elapsed_seconds=time.monotonic() - started, peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
         target_read=False))


def queue(workspace, output):
    reference = run_jobs(workspace, output, "scale", [("fit_normals", ["scale"], 20, 1800)],
                         worker_module=MODULE, gpu_indices=[0])
    jobs = run_jobs(workspace, output, "training", [(arm, ["train", "--arm", arm], 20, 3600) for arm in ARMS],
                    worker_module=MODULE, gpu_indices=[0, 1, 2])
    endpoints = {}
    for arm in ARMS:
        directory = output / "train" / arm
        path = directory / "SELECTION.json"
        selection = read(path)
        checkpoint = directory / "selected.pt"
        require(selection["status"] == "COMPLETED" and sha(checkpoint) == selection["selected_checkpoint_sha256"],
                "Incomplete arm selection")
        endpoints[arm] = dict(checkpoint_relative=checkpoint.relative_to(output).as_posix(),
                              checkpoint_sha256=sha(checkpoint), selection_relative=path.relative_to(output).as_posix(),
                              selection_sha256=sha(path), selected_step=selection["selected_step"])
    dump(output / "MODELS_FROZEN.json", dict(status="COMPLETED", arms=endpoints, seed=SEED, jobs=reference + jobs,
         actual_GPU_hours=sum(job["actual_GPU_seconds"] for job in reference + jobs) / 3600, target_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "scale", "train", "queue"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    opts = parser.parse_args()
    if opts.command == "prepare":
        prepare(opts.workspace, opts.output)
    elif opts.command == "scale":
        reference_scale(opts.workspace, opts.output)
    elif opts.command == "train":
        require(opts.arm is not None, "Training requires a fixed arm")
        training(opts.workspace, opts.output, opts.arm)
    else:
        queue(opts.workspace, opts.output)


if __name__ == "__main__":
    main()
