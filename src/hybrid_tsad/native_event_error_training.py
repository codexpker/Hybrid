"""Single matched seed2025 source training for the endpoint mean-error control."""
import argparse
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from . import native_event_rank_training as tr
from .native_event_rank import terminal_mean_error, native_endpoint_and_copy_loss
from .native_v_conditioning import fixed_native_rng

PLAN_SHA = 'df7e74047bb0fd49d25a2da0bb58e54a2fbe3327fd9f61b6e07ceca6422543e9'


def prepare_scale(workspace, output, original):
    started = time.monotonic()
    tr.require(tr.sha(original / 'PLAN.npz') == PLAN_SHA, 'Unexpected seed2025 source plan')
    for name in ('PLAN.json', 'PLAN.npz'):
        shutil.copyfile(original / name, output / name)
    plan = tr.load_plan(output)
    endpoints = plan['scale_normal_fit']
    old = tr.read(original / 'SCALE.json')
    tr.require(tr.sha(original / old['bank_file']) == old['bank_sha256'], 'Source V bank identity changed')
    with np.load(original / old['bank_file'], allow_pickle=False) as bank:
        np.testing.assert_array_equal(endpoints, bank['absolute_endpoint'])
        original_v = bank['V']
    folder, task, fit, _, _ = tr.source_roles(workspace)
    source = tr.event_input(workspace, folder, task, fit)
    device = tr.setup()
    model = tr.decoder_model(workspace, device)
    initial = tr.state_hashes(model)
    masks = tr.mask_ids(device)
    tr.require(masks == old['mask_ids'], 'Seed2025 fixed masks differ')
    first = next(iter(tr.windows(source, endpoints[:1], device)))
    with torch.no_grad(), fixed_native_rng(device):
        _, copies = model(first, norm=0, mask_mode='c', copies=10)
        double_precision = copies[:, 0, -1].cpu().double().numpy()
        target = first[0, -1].cpu().double().numpy()
        independent_e = float(np.mean(np.square(double_precision.mean(axis=0) - target)))
        independent_v = float(np.mean(double_precision.var(axis=0, ddof=1)))
    grad_e, _ = native_endpoint_and_copy_loss(model, first, 'E')
    grad_e.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    tr.require(grads and all(n.startswith('decoder.') and torch.isfinite(g).all() for n, g in grads.items()), 'Invalid E gradients')
    grad_norm = float(sum(g.square().sum() for g in grads.values()).sqrt())
    tr.require(grad_norm > 0 and tr.state_hashes(model) == initial, 'Frozen state or gradient check failed')
    model.zero_grad(set_to_none=True)
    values = torch.empty((len(endpoints), 2), device=device)
    with torch.inference_mode():
        for row, x in enumerate(tr.windows(source, endpoints, device)):
            with fixed_native_rng(device):
                _, copies = model(x, norm=0, mask_mode='c', copies=10)
            values[row, 0] = terminal_mean_error(copies, x)
            values[row, 1] = model.cal_anomaly_score(x, copies, anomaly_criterion=None, L=1)[0, 99]
    ev, v = values.cpu().double().numpy().T
    np.testing.assert_allclose([ev[0], v[0]], [independent_e, independent_v], rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(v, original_v, rtol=1e-5, atol=1e-8)
    tr.require(len(ev) == 4632 and np.isfinite(ev).all() and (ev >= 0).all(), 'Invalid E source reference')
    bank = output / 'FIT_NORMAL_E.npz'
    with bank.open('xb') as stream:
        np.savez_compressed(stream, E=ev, V=v, absolute_endpoint=endpoints)
    tr.dump(output / 'SCALE.json', dict(status='COMPLETED', evidence='E', value=max(1e-8, float(np.median(ev))),
            count=len(ev), native_source_sha256=tr.SOURCE_SHA, checkpoint_sha256=tr.WEIGHT_SHA,
            plan_sha256=PLAN_SHA, mask_ids=masks, bank_file=bank.name, bank_sha256=tr.sha(bank),
            first_endpoint=int(endpoints[0]), first_E=float(ev[0]), first_E_fp64=independent_e,
            first_V=float(v[0]), first_V_fp64=independent_v, max_abs_archived_V=float(np.max(np.abs(v-original_v))),
            E_decoder_grad_norm=grad_norm, elapsed_seconds=time.monotonic()-started,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device), target_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    args = parser.parse_args()
    original = args.workspace / 'results/native_event_rank_psm_20261122_v1'
    output = args.workspace / 'results/native_e_vs_v_20261123_v1'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'SCALE.json').exists() or (output / 'train/point_weighted_e').exists():
        raise FileExistsError('Source training or scale already exists')
    prepare_scale(args.workspace, output, original)
    tr.training(args.workspace, output, 'point_weighted_e', tr.SEED)


if __name__ == '__main__':
    main()
