"""Process-local factory extension, reusing the audited native runner kernels."""
import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from torch import nn
from . import timeradar_cross_variable_runner as base
from .timeradar_selective_context import ARMS, attach

HISTORICAL_FACTORY = base.model


def model(workspace, config, arm, device):
    if arm not in ARMS:
        raise ValueError('Only two predeclared extension arms')
    wrapper = HISTORICAL_FACTORY(workspace, config, 'self', device)
    return attach(wrapper, arm).to(device)


@contextmanager
def registered_factory():
    original = base.model
    base.model = model
    try:
        yield
    finally:
        base.model = original


def fit_readout(m, x, cfg, spec, device):
    m.eval(); m.adapter.capture_diagnostics = True
    results = {}
    for name, donor, uniform, enabled in [('original', None, False, True), ('donor0_value_zero', 0, False, True),
                                        ('uniform', None, True, True), ('disabled_hook', None, False, False)]:
        m.adapter.zero_donor_value = donor; m.adapter.force_uniform = uniform; m.adapter_enabled = enabled
        base.seed(cfg['seed'], device)
        with torch.no_grad(): rep, outputs = m(x, norm=spec['norm'], copies=cfg['copies'])
        score = m.cal_anomaly_score(x, outputs, nn.MSELoss(), L=spec['L'])
        results[name] = (rep, outputs, score, m.adapter.last_diagnostics if enabled else None)
    assert all(torch.equal(results['original'][0], result[0]) for result in results.values())
    m.adapter_enabled = True; m.adapter.zero_donor_value = None; m.adapter.force_uniform = False
    return results


def save_readout(out, results, starts, m):
    tensors = {}; statistics = {}
    original = results['original']
    for name, result in results.items():
        tensors[name+'_reconstruction'] = result[1].cpu().numpy(); tensors[name+'_L1score'] = result[2].cpu().numpy()
        statistics[name] = dict(reconstruction_max_delta=float((result[1]-original[1]).abs().max()),
                               L1score_max_delta=float((result[2]-original[2]).abs().max()),
                               L1score_mean_absolute_delta=float((result[2]-original[2]).abs().mean()))
        if result[3] is not None:
            tensors[name+'_context'] = result[3]['context'].numpy()
            weights = result[3]['weights']
            if weights is not None:
                tensors[name+'_weights'] = weights.numpy(); c = weights.shape[-1]
                uniform = (1-torch.eye(c))/(c-1)
                relative = (weights-uniform).abs()*(c-1)
                statistics[name].update(relative_uniform_max=float(relative.max()), relative_uniform_mean=float(relative.mean()),
                    entropy_nats_mean=float(-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean()),
                    uniform_entropy_nats=float(np.log(c-1)))
    np.savez_compressed(out/'FIT_MECHANISM.npz', starts=starts, **tensors)
    return dict(interventions=statistics, encoder_representation_bitexact_across_interventions=True,
                zero_up=bool(torch.count_nonzero(m.adapter.up.weight) == 0 and torch.count_nonzero(m.adapter.up.bias) == 0))


def selected_mechanism(w, config_path, cfg, out, record, arm, device):
    folder = w/cfg['output']/record/arm/'train'; receipt = base.read(folder/'TRAIN_RECEIPT.json')
    assert base.sha(folder/'selected.pth') == receipt['selected_sha256']
    state = torch.load(folder/'selected.pth', map_location='cpu', weights_only=True)
    assert state['config'] == cfg and state['arm'] == arm and state['record'] == record
    reader = base.RoleReader(w, cfg, record, ['FIT']); fit, _ = reader.load('FIT'); spec = cfg['records'][record]
    starts = next(iter(base.native_batches(fit, spec['batch_size'])))[:4]; x = base.tensor_windows(fit, starts, device)
    base.seed(cfg['seed'], device); m = model(w, cfg, arm, device); m.load_state_dict(state['state'], strict=True)
    before = base.state_hashes(m)
    result = save_readout(out, fit_readout(m, x, cfg, spec, device), starts, m)
    assert base.state_hashes(m) == before
    base.dump(out/'SELECTED_FIT_RECEIPT.json', dict(status='COMPLETED_FIT_ONLY_NO_UPDATES', record=record, arm=arm,
        selected_update=state['update'], selected_sha256=receipt['selected_sha256'], roles=reader.access,
        negative_results_preserved=True, no_optimizer=True, no_model_or_threshold_selection=True, **result))
    m.close()


def mechanism(w, config_path, cfg, out, record, arm, device):
    """Only fixed FIT windows. No label, CAL or QUERY access or model selection."""
    reader = base.RoleReader(w, cfg, record, ['FIT'])
    fit, _ = reader.load('FIT'); spec = cfg['records'][record]
    starts = next(iter(base.native_batches(fit, spec['batch_size'])))[:4]
    x = base.tensor_windows(fit, starts, device)
    base.seed(cfg['seed'], device); m = model(w, cfg, arm, device).eval()
    expected = 793610+17424
    assert sum(p.numel() for p in m.native.decoder.parameters()) == 793610
    assert sum(p.numel() for p in m.adapter.parameters()) == 17424
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == expected
    # Direct official native forward: no hook, same input and exact RNG replay.
    cpu = torch.get_rng_state(); cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    m._hook.remove(); m._hook = None
    try:
        with torch.no_grad(): witness = m.native(x, norm=spec['norm'], copies=cfg['copies'])
    finally:
        m._hook = m.native.decoder.register_forward_pre_hook(m._adapt)
    def restore():
        torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state(cuda, device)
    restore()
    with torch.no_grad(): zero = m(x, norm=spec['norm'], copies=cfg['copies'])
    assert all(torch.equal(a, b) for a, b in zip(witness, zero))
    assert torch.equal(m.cal_anomaly_score(x, witness[1], nn.MSELoss(), L=spec['L']),
                       m.cal_anomaly_score(x, zero[1], nn.MSELoss(), L=spec['L']))
    m.adapter_enabled = False; restore()
    with torch.no_grad(): disabled = m(x, norm=spec['norm'], copies=cfg['copies'])
    assert all(torch.equal(a, b) for a, b in zip(witness, disabled)); m.adapter_enabled = True
    initial_adapter = base.state_hashes(m.adapter)
    # Two FIT-only diagnostic steps: first moves zero-up, second exposes Q/K.
    opt = base.optimizer(m, cfg); diagnostic_steps = []
    for step in range(2):
        m.train(); base.seed(cfg['seed']+step, device); opt.zero_grad(set_to_none=True)
        _, outputs = m(x, norm=spec['norm'], copies=cfg['copies'])
        loss = base.mse(outputs, x); loss.backward()
        qg, kg = m.adapter.query.weight.grad, m.adapter.key.weight.grad
        grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        diagnostic_steps.append(dict(step=step+1, loss=float(loss.detach()),
            query_grad_norm=float(qg.norm()) if qg is not None else None,
            key_grad_norm=float(kg.norm()) if kg is not None else None))
        opt.step()
    if x.shape[-1] > 2:
        assert diagnostic_steps[-1]['query_grad_norm'] > 0 and diagnostic_steps[-1]['key_grad_norm'] > 0
    results = fit_readout(m, x, cfg, spec, device)
    original, removed, uniform = (results[k] for k in ('original', 'donor0_value_zero', 'uniform'))
    if arm == 'selective' and x.shape[-1] > 2:
        weights = original[3]['weights']
        assert torch.equal(weights, removed[3]['weights'])
        uniform_offdiag = (1-torch.eye(x.shape[-1]))/(x.shape[-1]-1)
        assert float((weights-uniform_offdiag).abs().max()) > 0
        assert not torch.equal(original[1], removed[1]) and not torch.equal(original[2], removed[2])
        assert not torch.equal(original[1], uniform[1]) and not torch.equal(original[2], uniform[2])
    else:
        assert torch.equal(original[1], removed[1]) and torch.equal(original[2], removed[2])
    readout = save_readout(out, results, starts, m)
    base.dump(out/'MECHANISM_RECEIPT.json', dict(status='PASS_FIT_ONLY', record=record, arm=arm,
        zero_and_disabled_native_bitexact=True, zero_native_L1score_bitexact=True, diagnostic_optimizer_steps=diagnostic_steps,
        initial_adapter_state=initial_adapter, final_adapter_state=base.state_hashes(m.adapter),
        trainable_parameters=expected, roles=reader.access, input_windows=4,
        interventions='donor0 value only, fixed Q/K; uniform other-variable weights; FIT only',
        limitations='No query or scientific benefit; C1 falls back to h, C2 has one donor and unidentifiable Q/K selection', **readout))
    m.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['mechanism', 'train', 'score', 'selected_mechanism'])
    p.add_argument('--workspace', type=Path, required=True); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--record', required=True); p.add_argument('--arm', choices=ARMS, required=True)
    p.add_argument('--device', default='cpu'); a = p.parse_args()
    w, config = a.workspace.resolve(), a.config.resolve(); cfg = base.read(config)
    torch.set_num_threads(4); torch.set_num_interop_threads(1); device = torch.device(a.device)
    out = w/cfg['output']/a.record/a.arm/a.phase; out.mkdir(parents=True, exist_ok=False)
    start, rc = time.time(), 0
    if device.type == 'cuda':
        torch.cuda.set_device(device); cap = 30 if a.phase == 'score' else 20
        torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    base.dump(out/'START.json', dict(pid=os.getpid(), phase=a.phase, record=a.record, arm=a.arm,
        start_unix=start, config_sha256=base.sha(config), runner_sha256=base.sha(__file__),
        adapter_sha256=base.sha(Path(__file__).with_name('timeradar_selective_context.py')),
        base_runner_sha256=base.sha(base.__file__), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        with registered_factory():
            if a.phase == 'mechanism': mechanism(w, config, cfg, out, a.record, a.arm, device)
            elif a.phase == 'selected_mechanism': selected_mechanism(w, config, cfg, out, a.record, a.arm, device)
            elif a.phase == 'train': base.train(w, config, cfg, out, a.record, a.arm, device)
            else: base.score(w, config, cfg, out, a.record, a.arm, device)
    except BaseException:
        rc = 1; base.dump(out/'FAILED.json', dict(traceback=traceback.format_exc())); raise
    finally:
        end = time.time(); base.dump(out/'OS.json', dict(pid=os.getpid(), start_unix=start, end_unix=end,
            seconds=end-start, exit_code=rc, GPU_hours=(end-start)/3600 if device.type == 'cuda' else 0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0))


if __name__ == '__main__': main()
