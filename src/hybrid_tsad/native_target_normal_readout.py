"""Matched zero-update/normal-target-trained native V on entity-local W100 grids."""
import argparse
import fcntl
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .native_v_conditioning import load_native, fixed_native_rng, state_hashes, sha, WEIGHT_SHA
from .native_v_conditioning_runner import read, dump, setup
from . import native_event_space as nasa

OUT = Path('results/native_target_normal_20261124_v1')
SOURCE = Path('results/native_position_aligned_20261124_v1')
KEYS = ('init_official', 'init_source_copy', 'init_source_all',
        'target_official', 'target_source_copy', 'target_source_all')
ARMS = ('official', 'copy', 'all_positions')


def models(w, domain):
    root = w / OUT
    seal = read(root / 'ALL_MODELS_FROZEN.json')
    assert seal['status'] == 'COMPLETED_BEFORE_TEST', seal['status']
    base = load_native(w, 'cuda')
    official_state = base.state_dict()
    nets = [base]
    bindings = [dict(kind='official', checkpoint_sha256=WEIGHT_SHA)]
    for arm, expected in (('copy', 120), ('all_positions', 200)):
        receipt = read(w / SOURCE / 'train' / arm / 'SELECTION.json')
        path = w / SOURCE / receipt['checkpoint_relative']
        assert receipt['selected_step'] == expected and sha(path) == receipt['checkpoint_sha256']
        model = deepcopy(base)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['update'] == expected
        model.load_state_dict(saved['model'], strict=True)
        nets.append(model)
        bindings.append(dict(kind='source_selected', arm=arm, selected_step=expected,
                             checkpoint_sha256=sha(path), path=str(path.relative_to(w))))
    for arm in ARMS:
        folder = root / domain / 'train' / arm
        receipt = read(folder / 'SELECTION.json')
        path = folder / 'selected.pt'
        assert sha(path) == receipt['checkpoint_sha256']
        sealed = seal['domains'][domain][arm]
        assert sealed['checkpoint_sha256'] == sha(path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['epoch'] == receipt['selected_epoch']
        model = deepcopy(base)
        model.load_state_dict(saved['model'], strict=True)
        nets.append(model)
        bindings.append(dict(kind='target_normal_selected', arm=arm,
                             selected_epoch=receipt['selected_epoch'],
                             checkpoint_sha256=sha(path), path=str(path.relative_to(w))))
    for model in nets:
        model.eval().requires_grad_(False)
        assert all(torch.equal(official_state[k], value) for k, value in model.state_dict().items()
                   if not k.startswith('decoder.'))
    aliases = []
    for j, model in enumerate(nets):
        same = [i for i in range(j) if all(torch.equal(value, nets[i].state_dict()[k])
                                         for k, value in model.state_dict().items())]
        aliases.append(same[0] if same else None)
    return nets, bindings, aliases


def raw_test(w, domain, entity):
    if domain == 'PSM':
        path = w / 'data/common_psm_raw_20261029_v1/test.npy'
        assert sha(path) == '6592518615d690ab153592b6dd0220c424441a0d4ee7bac87c3f89fe942893f5'
        value = np.load(path, allow_pickle=False).astype(np.float64)
        assert value.shape == (87841, 25)
    else:
        assert domain == 'MSL' and entity in nasa.entity_list(w, domain)
        path = w / nasa.MSL / 'test' / (entity + '.npy')
        admitted = read(w / 'results/left_nasa_author_20261123_v1/MSL/SCORES.json')
        prior = next(r for r in admitted['banks'] if r['entity'] == entity)
        assert sha(path) == prior['input_sha256']
        value = np.load(path, allow_pickle=False)[:, :1].astype(np.float64)
    assert value.ndim == 2 and np.isfinite(value).all()
    return value, path


def windows(w, domain, entity):
    value, path = raw_test(w, domain, entity)
    scaler = w / OUT / domain / 'scaler.npz'
    with np.load(scaler, allow_pickle=False) as s:
        scaled = ((value - s['mean']) / s['scale']).astype(np.float32)
    n = len(value)
    return scaled[:n // 100 * 100].reshape(-1, 100, value.shape[-1]), dict(
        raw_input_path=str(path.relative_to(w)), raw_input_sha256=sha(path),
        scaler_path=str(scaler.relative_to(w)), scaler_sha256=sha(scaler),
        original_points=n, channels=value.shape[-1],
        scaling='raw public test array -> complete official normal-train mean/scale once')


def jobs(w):
    result = [('PSM', 'PSM')]
    result.extend(('MSL', entity) for entity in nasa.entity_list(w, 'MSL'))
    assert len(result) == 28
    return result


@torch.inference_mode()
def normal_check(w, domain, nets, order):
    from torch_frft.frft_module import ifrft
    from .native_target_normal import normal_raw
    raw = normal_raw(w, domain)
    entity = sorted(raw)[0]
    value = raw[entity][:100]
    with np.load(w / OUT / domain / 'scaler.npz') as s:
        value = ((value - s['mean']) / s['scale']).astype(np.float32)
    assert value.shape == (100, 25 if domain == 'PSM' else 1)
    x = torch.from_numpy(np.ascontiguousarray(value)).cuda()[None]
    initial = [state_hashes(n) for n in nets]
    cpu = torch.get_rng_state().clone()
    gpu = torch.cuda.get_rng_state().clone()
    start = time.monotonic()
    order.clear()
    with fixed_native_rng('cuda'):
        encoded, _ = nets[0](x, norm=0, mask_mode='c', copies=10)
    projected = torch.stack([n.decoder(encoded) for n in nets])
    decoded = ifrft(projected, order[0], dim=-1).abs()
    for j, net in enumerate(nets):
        copies = decoded[j].reshape(10, 1, x.shape[-1], 100).permute(0, 1, 3, 2)
        with fixed_native_rng('cuda'):
            _, direct = net(x, norm=0, mask_mode='c', copies=10)
        torch.testing.assert_close(copies, direct, rtol=0, atol=0)
        v = net.cal_anomaly_score(x, copies, None, 1)[0]
        torch.testing.assert_close(v, direct.var(0, correction=1).mean(-1)[0], rtol=0, atol=0)
    assert initial == [state_hashes(n) for n in nets]
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(gpu, torch.cuda.get_rng_state())
    return dict(domain=domain, entity=entity, role='TARGET_NORMAL_TRAIN', window=[0, 100],
                six_models_native_output_bit_exact=True, native_V_bit_exact=True,
                frozen_verified=True, RNG_restored=True, seconds=time.monotonic() - start)


def archived_initials(w, domain, entity, current, bindings):
    """Reuse only exact scaled inputs, grid and selected weight identities."""
    old_root = w / 'results/native_full_window_20261124_v1'
    path = old_root / domain / (entity + '.npz')
    rec = read(path.with_suffix('.json'))
    assert sha(path) == rec['sha256']
    old_binding = rec['input_binding']
    value, raw_path = raw_test(w, domain, entity)
    if domain == 'PSM':
        folder = w / 'results/native_v_conditioning_20261119_v1/PSM'
        task = read(folder / 'TASK.json')
        scaler_path = folder / 'SCALER.npz'
        assert sha(scaler_path) == task['scaler_sha256'] == old_binding['scaler_sha256']
        with np.load(scaler_path) as s:
            old = ((value - s['mean']) / s['std']).astype(np.float32)
    else:
        assert sha(raw_path) == old_binding['input_sha256']
        old = ((value.astype(np.float32).astype(np.float64) - old_binding['mean']) /
               old_binding['scale']).astype(np.float32)
    old = old[:len(old) // 100 * 100].reshape(-1, 100, old.shape[-1])
    same = current.shape == old.shape and np.array_equal(current, old)
    result = {}
    receipt = dict(scaled_input_bit_exact_to_legacy=same,
                   reason='complete normal-train statistics must not be assumed equal to source-transfer statistics')
    if not same:
        return result, receipt
    assert rec['weights_sha256'][0] == bindings[0]['checkpoint_sha256']
    assert rec['window'] == rec['stride'] == 100 and rec['native_B'] == 1 and rec['copies'] == 10
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['index'], np.arange(len(current) * 100))
        result[0] = z['V'][:, 0].copy()
    aligned = w / SOURCE / 'readout' / domain / (entity + '.npz')
    aligned_receipt = read(aligned.with_suffix('.json'))
    assert sha(aligned) == aligned_receipt['sha256']
    with np.load(aligned, allow_pickle=False) as z:
        for j, old_column in ((1, 0), (2, 2)):
            assert aligned_receipt['weights_sha256'][old_column] == bindings[j]['checkpoint_sha256']
            result[j] = z['V'][:, old_column].copy()
    receipt['reused_bank_sha256'] = [rec['sha256'], aligned_receipt['sha256']]
    return result, receipt


@torch.inference_mode()
def score(w, lane=0, lanes=4):
    from torch_frft.frft_module import ifrft
    setup(12)
    started = time.monotonic()
    root = w / OUT / 'readout'
    root.mkdir(parents=True, exist_ok=True)
    cache = {}
    processed = []
    work = jobs(w)
    for domain, entity in work[lane::lanes] + work:
        dest = root / domain
        dest.mkdir(parents=True, exist_ok=True)
        bank = dest / (entity + '.npz')
        receipt_path = bank.with_suffix('.json')
        with bank.with_suffix('.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if receipt_path.exists():
                receipt = read(receipt_path)
                assert sha(bank) == receipt['sha256']
                continue
            tick = time.monotonic()
            if domain not in cache:
                nets, bindings, aliases = models(w, domain)
                order = []
                nets[0].net.register_forward_hook(lambda _m, _a, output, storage=order: storage.append(output.squeeze()))
                checked = normal_check(w, domain, nets, order)
                dump(root / f'NORMAL_CHECK_{domain}_lane{lane}.json', checked)
                cache[domain] = nets, bindings, aliases, order, [state_hashes(m) for m in nets]
            nets, bindings, aliases, order, identity = cache[domain]
            window, binding = windows(w, domain, entity)
            reused, reuse_receipt = archived_initials(w, domain, entity, window, bindings)
            score_array = np.empty((len(window) * 100, len(KEYS)), np.float32)
            for j, value in reused.items():
                score_array[:, j] = value
            active = [j for j in range(len(KEYS)) if j not in reused and aliases[j] is None]
            check_errors = []
            cpu_rng = torch.get_rng_state().clone()
            gpu_rng = torch.cuda.get_rng_state().clone()
            for b, value in enumerate(window) if active else []:
                x = torch.from_numpy(np.ascontiguousarray(value)).cuda()[None]
                order.clear()
                with fixed_native_rng('cuda'):
                    encoded, _ = nets[0](x, norm=0, mask_mode='c', copies=10)
                decoded = ifrft(torch.stack([nets[j].decoder(encoded) for j in active]), order[0], dim=-1).abs()
                for k, j in enumerate(active):
                    copies = decoded[k].reshape(10, 1, x.shape[-1], 100).permute(0, 1, 3, 2)
                    if b == 0:
                        with fixed_native_rng('cuda'):
                            _, direct = nets[j](x, norm=0, mask_mode='c', copies=10)
                        torch.testing.assert_close(copies, direct, rtol=0, atol=0)
                        v = nets[j].cal_anomaly_score(x, copies, None, 1)[0]
                        terminal = nets[j].cal_anomaly_score(x, direct, None, 1)[0, 99]
                        assert v[99] == terminal
                        check_errors.append(float((copies - direct).abs().max()))
                    score_array[b * 100:(b + 1) * 100, j] = nets[j].cal_anomaly_score(x, copies, None, 1)[0].cpu().numpy()
            for j, alias in enumerate(aliases):
                if alias is not None:
                    score_array[:, j] = score_array[:, alias]
            assert np.isfinite(score_array).all()
            assert identity == [state_hashes(m) for m in nets]
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
            ix = np.arange(len(score_array))
            available = (ix // 100 + 1) * 100 - 1
            temp = bank.with_suffix('.part.npz')
            np.savez_compressed(temp, V=score_array, keys=KEYS, index=ix,
                                window_start=ix // 100 * 100,
                                window_end_exclusive=available + 1, available_index=available)
            temp.replace(bank)
            dump(receipt_path, dict(domain=domain, entity=entity, sha256=sha(bank), keys=KEYS,
                 weights=bindings, aliases=aliases, reused_initial_columns=list(reused), reuse_check=reuse_receipt,
                 input_binding=binding, points=len(ix), omitted_tail=binding['original_points'] - len(ix),
                 seconds=time.monotonic() - tick, device=os.environ['CUDA_VISIBLE_DEVICES'],
                 B=1, copies=10, mask_seed=84551, correction=1, channel_reduction='mean',
                 eval=True, frozen_state_verified=True, RNG_restored=True, labels_read=False,
                 offline=True, window=100, stride=100, same_window_native_output_max_error=max(check_errors, default=0.0)))
            processed.append(f'{domain}/{entity}')
            print(domain, entity, len(ix), round(time.monotonic() - tick, 3), flush=True)
    dump(root / f'LANE_{lane}.json', dict(lane=lane, seconds=time.monotonic() - started,
                                        device=os.environ['CUDA_VISIBLE_DEVICES'], processed=processed,
                                        frozen_verified=all(identity == [state_hashes(m) for m in nets]
                                            for nets, _b, _a, _o, identity in cache.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', type=int, default=0)
    parser.add_argument('--lanes', type=int, default=4)
    args = parser.parse_args()
    score(Path.cwd(), args.lane, args.lanes)
