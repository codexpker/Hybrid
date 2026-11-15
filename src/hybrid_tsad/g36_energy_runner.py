"""Fixed G36 energy experiment: train first, QUERY only after normal gate."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from .g36_energy_model import ContextEnergyResponse, fixed_noise
from .g36_runner import Data, batch, dump, hashes, read, seed, sha
from .normality_density_runner import validation_mode


def make(cfg, method, device):
    return ContextEnergyResponse(method, cfg['hidden'], cfg['kernel']).to(device)


def prepare(workspace, config, cfg):
    root = workspace/cfg['output']
    root.mkdir(parents=True, exist_ok=True)
    folder = root/'shared'
    folder.mkdir(exist_ok=False)
    data = Data(workspace, cfg, ['FIT'])
    fit, _ = data.one('FIT')
    source_plan = workspace/cfg['existing_fit_plan']
    with np.load(source_plan, allow_pickle=False) as z:
        order = z['end'].copy()
    assert order.shape == (cfg['updates'], cfg['batch_size'])
    assert order.min() >= 51 and order.max() < len(fit)
    expected = np.random.default_rng(cfg['seed']).integers(51, len(fit), size=order.shape, dtype=np.int64)
    assert np.array_equal(order, expected)
    shutil.copyfile(source_plan, folder/'FIT_PLAN.npz')
    identities = {}
    for method in cfg['methods']:
        seed(cfg['seed'], torch.device('cpu'))
        model = make(cfg, method, torch.device('cpu'))
        torch.save(dict(state=model.state_dict()), folder/(method+'_INITIAL.pth'))
        identities[method] = dict(sha256=sha(folder/(method+'_INITIAL.pth')), state=hashes(model),
                                  parameters=sum(p.numel() for p in model.parameters()))
    assert identities['context_only']['state'] == identities['response_feedback']['state']
    dump(folder/'MANIFEST.json', dict(config_sha256=sha(config), data_manifest_sha256=sha(data.root/'manifest.json'),
         scaler_sha256=sha(data.root/'scaler.npz'), plan_sha256=sha(folder/'FIT_PLAN.npz'),
         reused_plan_source=str(source_plan), initial=identities, FIT_access=data.access,
         no_SELECT_CAL_QUERY_values_access=True, effective_TCN_receptive_field_points=31))


def initial(workspace, config, cfg, method, device):
    folder = workspace/cfg['output']/'shared'
    meta = read(folder/'MANIFEST.json')
    assert meta['config_sha256'] == sha(config)
    assert meta['data_manifest_sha256'] == sha(workspace/cfg['data']/'manifest.json')
    assert meta['scaler_sha256'] == sha(workspace/cfg['data']/'scaler.npz')
    assert sha(folder/(method+'_INITIAL.pth')) == meta['initial'][method]['sha256']
    seed(cfg['seed'], device)
    model = make(cfg, method, device)
    state = torch.load(folder/(method+'_INITIAL.pth'), map_location='cpu', weights_only=True)
    model.load_state_dict(state['state'], strict=True)
    assert hashes(model) == meta['initial'][method]['state']
    return model, meta


def select(model, values, device, cfg):
    total, count = 0., 0
    a, b = [fixed_noise(cfg['select_draw_pairs'], s, device) for s in cfg['select_noise_seeds']]
    with validation_mode(model, device):
        for begin in range(51, len(values), cfg['batch_size']):
            ends = np.arange(begin, min(begin+cfg['batch_size'], len(values)))
            value = model.energy(batch(values, ends, device), a, b)
            assert value.shape == (len(ends),) and torch.isfinite(value).all()
            total += float(value.double().sum())
            count += len(ends)
    return total/count


def score_banks(cfg, device):
    n = cfg['score_reference_pairs']
    seeds = cfg['score_noise_seeds']
    return (fixed_noise(n, seeds[0], device), fixed_noise(n, seeds[1], device),
            fixed_noise(cfg['score_normal_probes'], seeds[2], device))


def train(workspace, config, cfg, out, method, device, l0=False):
    model, meta = initial(workspace, config, cfg, method, device)
    data = Data(workspace, cfg, ['FIT', 'SELECT'])
    fit, _ = data.one('FIT')
    val, _ = data.one('SELECT')
    plan = workspace/cfg['output']/'shared/FIT_PLAN.npz'
    assert sha(plan) == meta['plan_sha256']
    with np.load(plan, allow_pickle=False) as z:
        order = z['end'].copy()
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    best, selected, initial_select = float('inf'), -1, None
    N = 2 if l0 else cfg['updates']
    started = time.monotonic()

    def validation(step):
        nonlocal best, selected, initial_select
        states = [(m, m.training) for m in model.modules()]
        cpu = torch.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
        v = select(model, val, device, cfg)
        assert torch.equal(cpu, torch.get_rng_state()) and all(m.training == s for m, s in states)
        if cuda is not None:
            assert torch.equal(cuda, torch.cuda.get_rng_state(device))
        row = dict(update=step, SELECT_loss=v, CPU_CUDA_modes_restored=True)
        with (out/'SELECT_TRACE.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if step == 0:
            initial_select = v
        if v < best:
            best, selected = v, step
            if not l0:
                torch.save(dict(state=model.state_dict(), method=method, update=step,
                                config=cfg, plan_sha256=meta['plan_sha256']), out/'selected.tmp')
                os.replace(out/'selected.tmp', out/'selected.pth')

    validation(0)
    for step in range(1, N+1):
        model.train()
        x = batch(fit, order[step-1], device)
        opt.zero_grad(set_to_none=True)
        value = model.loss(x, cfg['train_draw_pairs'])
        assert value.ndim == 0 and torch.isfinite(value)
        value.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'])
        assert torch.isfinite(norm)
        response_grad = float(model.input.weight.grad[:, :4].abs().sum())
        context_grad = float(model.input.weight.grad[:, 4:].abs().sum())
        if method == 'context_only':
            assert response_grad == 0
        if l0:
            assert context_grad > 0
            if method == 'response_feedback':
                assert response_grad > 0
        opt.step()
        row = dict(update=step, loss=float(value.detach()), gradient_norm_before_clip=float(norm),
                   response_input_weight_gradient_L1=response_grad, context_input_weight_gradient_L1=context_grad,
                   seconds=time.monotonic()-started)
        with (out/'TRAIN_TRACE.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        if step % cfg['select_every'] == 0 and not l0:
            validation(step)
    if l0:
        validation(N)
        assert any(value != meta['initial'][method]['state'][key] for key, value in hashes(model).items())
        model.eval()
        with torch.no_grad():
            fields = model.components(x, *score_banks(cfg, device))
            assert fields['score'].shape == (len(x),) and torch.isfinite(fields['score']).all()
            assert fields['normal_energy'].shape == (len(x), cfg['score_normal_probes'])
    else:
        torch.save(dict(state=model.state_dict(), update=N), out/'final.pth')
    dump(out/'TRAIN_RECEIPT.json', dict(status='L0_PASS_DISCARDED' if l0 else 'COMPLETED',
         method=method, updates=N, selected_update=selected, SELECT_loss=best, step0_SELECT_loss=initial_select,
         selected_improves_step0=selected > 0 and best < initial_select,
         initial_sha256=meta['initial'][method]['sha256'], initial_state=meta['initial'][method]['state'],
         plan_sha256=meta['plan_sha256'], roles=data.access,
         selected_sha256=None if l0 else sha(out/'selected.pth'), no_CAL_QUERY_values=True,
         no_labels_decoded=True, loss=cfg['training_objective']))


def score(workspace, config, cfg, out, method, device, gate):
    release = read(gate)
    assert release['status'] == 'NORMAL_GATE_PASS_QUERY_RELEASED'
    assert release['config_sha256'] == sha(config)
    context_train = workspace/cfg['output']/'context_only/train'
    context_receipt = read(context_train/'TRAIN_RECEIPT.json')
    assert context_receipt['selected_improves_step0']
    assert release['context_train_receipt_sha256'] == sha(context_train/'TRAIN_RECEIPT.json')
    trainpath = out.parent/'train'
    receipt = read(trainpath/'TRAIN_RECEIPT.json')
    assert sha(trainpath/'selected.pth') == receipt['selected_sha256']
    model, _ = initial(workspace, config, cfg, method, device)
    state = torch.load(trainpath/'selected.pth', map_location='cpu', weights_only=True)
    assert state['method'] == method and state['config'] == cfg
    model.load_state_dict(state['state'], strict=True)
    model.eval()
    before = hashes(model)
    data = Data(workspace, cfg, ['SELECT', 'CAL', 'QUERY'])
    banks = []
    noise = score_banks(cfg, device)
    with torch.no_grad():
        for record in data.manifest['records']:
            if record['role'] not in ['SELECT', 'CAL', 'QUERY']:
                continue
            values, index = data.load(record)
            ends = np.arange(51, len(values))
            parts = {}
            for begin in range(0, len(ends), cfg['batch_size']):
                e = ends[begin:begin+cfg['batch_size']]
                fields = model.components(batch(values, e, device), *noise)
                for key, v in fields.items():
                    parts.setdefault(key, []).append(v.cpu().numpy())
            fields = {k: np.concatenate(v) for k, v in parts.items()}
            ix = index[ends]
            assert ix[0] == record['first_scored_index'] and len(ix) == record['n_scored']
            assert fields['score'].shape == (len(ix),) and np.isfinite(fields['score']).all()
            assert all(np.isfinite(value).all() for value in fields.values() if np.issubdtype(value.dtype, np.floating))
            file = out/(record['run']+'__'+record['role']+'.npz')
            np.savez_compressed(file, index=ix, available_index=ix, max_input_index=ix,
                                max_condition_index=ix-4, min_input_index=ix-51, **fields)
            banks.append(dict(run=record['run'], role=record['role'], path=str(file), sha256=sha(file), points=len(ix)))
    assert hashes(model) == before
    dump(out/'SCORE_RECEIPT.json', dict(status='COMPLETED', method=method, banks=banks, roles=data.access,
         PT_sha256=sha(trainpath/'selected.pth'), selected_update=state['update'], parameter_buffer_immutable=True,
         gate_sha256=sha(gate), no_labels_decoded=True, no_threshold_fitting=True, score=cfg['score'],
         fixed_noise_sha256=[__import__('hashlib').sha256(n.cpu().numpy().tobytes()).hexdigest() for n in noise]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['prepare', 'l0', 'train', 'score'])
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--method', choices=['context_only', 'response_feedback'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--gate', type=Path)
    args = p.parse_args()
    workspace, config = args.workspace.resolve(), args.config.resolve()
    cfg = read(config)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if args.phase == 'prepare':
        return prepare(workspace, config, cfg)
    device = torch.device(args.device)
    out = workspace/cfg['output']/args.method/args.phase
    out.mkdir(parents=True, exist_ok=False)
    begin, rc = time.time(), 0
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        cap = 30 if args.phase == 'score' else 20
        torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(device).total_memory, device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    dump(out/'START.json', dict(pid=os.getpid(), phase=args.phase, method=args.method, start_unix=begin,
         config_sha256=sha(config), runner_sha256=sha(__file__),
         model_sha256=sha(Path(__file__).with_name('g36_energy_model.py')),
         CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))
    try:
        if args.phase == 'score':
            if args.gate is None:
                raise PermissionError('Normal cross-day gate must be explicitly released before QUERY')
            score(workspace, config, cfg, out, args.method, device, args.gate)
        else:
            train(workspace, config, cfg, out, args.method, device, l0=args.phase == 'l0')
    except BaseException:
        rc = 1
        dump(out/'FAILED.json', dict(traceback=traceback.format_exc()))
        raise
    finally:
        end = time.time()
        dump(out/'OS.json', dict(pid=os.getpid(), start_unix=begin, end_unix=end, seconds=end-begin, exit_code=rc,
             GPU_hours=(end-begin)/3600 if device.type == 'cuda' else 0,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0))


if __name__ == '__main__':
    main()
