"""Native LEFT external comparison on physically isolated, FIT-scaled roles.

The official train loop, curriculum, early stopping and native offline scorer
are called unchanged. Only the data loader, logging and resource transport are
replaced. This is an external baseline, not a new Hybrid method.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .timeradar_cross_variable_runner import RoleReader, read, resolve, sha, state_hashes


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


class RoleWindows(Dataset):
    """No padding, cross-role concatenation, refitting or interpolation."""
    def __init__(self, values, index, width=192, stride=192):
        if width < 1 or stride < 1 or len(values) != len(index):
            raise ValueError('Invalid role-local window geometry')
        if len(index) > 1 and not np.all(np.diff(index) == 1):
            raise ValueError('Discontinuous indices cannot form one role')
        self.values, self.index, self.width, self.stride = values, index, width, stride

    def __len__(self):
        return max(0, (len(self.values) - self.width) // self.stride + 1)

    def footprint(self, number):
        start = number * self.stride
        return [int(self.index[start]), int(self.index[start + self.width - 1]) + 1]

    def __getitem__(self, number):
        if not 0 <= number < len(self):
            raise IndexError(number)
        start = number * self.stride
        return self.values[start:start + self.width], np.zeros((self.width, 1), np.float32)


def scientific_state(model, saved, allowed_profile_keys):
    """Ignore exactly recorded THOP counters, strict-load every scientific key."""
    expected = set(model.state_dict())
    extra = set(saved) - expected
    if extra != set(allowed_profile_keys) or any(
        not key.endswith(('total_ops', 'total_params')) for key in extra
    ):
        raise ValueError('Unrecorded/missing profile counter or unexpected model key')
    model.load_state_dict({key: value for key, value in saved.items() if key in expected}, strict=True)


def source_setup(workspace, cfg, out):
    source = resolve(workspace, cfg['official_source'])
    for name, digest in cfg['official_source_hashes'].items():
        if sha(source / name) != digest:
            raise ValueError('Official source identity mismatch: ' + name)
    runtime = out / 'native_configs'
    for dataset, entries in cfg['official_configs'].items():
        for name, entry in entries.items():
            original = resolve(workspace, entry['path'])
            if sha(original) != entry['sha256']:
                raise ValueError('Official configuration identity mismatch')
            destination = runtime / dataset / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
    sys.path.insert(0, str(resolve(workspace, cfg['thop_site'])))
    sys.path.insert(0, str(source))
    return runtime


def set_seed(number):
    random.seed(number)
    np.random.seed(number)
    torch.manual_seed(number)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(number)


def native_model(workspace, cfg, record, device):
    from models.Left.LEFT import Left
    dataset = cfg['records'][record]
    entry = cfg['official_configs'][dataset]['model_configs_Left.json']
    return Left(SimpleNamespace(**read(resolve(workspace, entry['path'])))).to(device)


def train(workspace, cfg, out, record, device):
    runtime = source_setup(workspace, cfg, out)
    from exp import exp_anomaly_detection as official
    set_seed(cfg['seed'])
    data = RoleReader(workspace, cfg, record, ['FIT', 'SELECT'])
    fit, fit_index = data.load('FIT')
    val, val_index = data.load('SELECT')
    trace, saves, updates, exposures = out / 'TRACE.jsonl', [], [0], [0]
    optimizer_started, select_epochs = [False], [0]
    native_early = official.EarlyStopping

    class LoggedEarlyStopping(native_early):
        def save_checkpoint(self, val_loss, model, path):
            super().save_checkpoint(val_loss, model, path)
            event = dict(kind='CHECKPOINT', SELECT_loss=float(val_loss),
                         python_step=int(model.step), update=updates[0],
                         checkpoint_sha256=sha(Path(path) / 'checkpoint.pth'))
            saves.append(event)
            append(trace, event)

    class IsolatedNative(official.Exp_Anomaly_Detection):
        def _acquire_device(self):
            # Physical isolation belongs to CUDA_VISIBLE_DEVICES in the supervisor.
            return device

        def _get_data(self, flag, step=None):
            if flag not in ['train', 'val']:
                raise PermissionError('Native learner cannot access CAL/QUERY')
            values, index = (fit, fit_index) if flag == 'train' else (val, val_index)
            stride = self.model_configs.seq_len if step is None else step
            dataset = RoleWindows(values, index, self.model_configs.seq_len, stride)
            if not len(dataset):
                raise ValueError('Empty independent role')
            append(trace, dict(kind='LOADER', role=flag, windows=len(dataset), stride=stride,
                               first=dataset.footprint(0), last=dataset.footprint(len(dataset)-1)))
            return dataset, DataLoader(dataset, batch_size=self.train_configs.batch_size,
                                       shuffle=flag == 'train', drop_last=False, num_workers=0)

        def _select_optimizer(self):
            optimizer = super()._select_optimizer()
            optimizer_started[0] = True
            def after_step(opt, args, kwargs):
                updates[0] += 1
                gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError('Nonfinite or absent native gradient')
                if updates[0] == 1 or updates[0] % 100 == 0:
                    append(trace, dict(kind='UPDATE', update=updates[0],
                        python_step=int(self.model.step), learning_rate=[g['lr'] for g in opt.param_groups],
                        gradient_norm=float(torch.stack([g.detach().float().norm() for g in gradients]).norm())))
            optimizer.register_step_post_hook(after_step)
            return optimizer

        def vali(self, loader):
            value = super().vali(loader)
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite SELECT')
            select_epochs[0] += 1
            append(trace, dict(kind='SELECT', epoch=select_epochs[0],
                               loss=float(value), update=updates[0], python_step=int(self.model.step)))
            return value

        def test(self, **kwargs):
            raise PermissionError('Learner cannot open QUERY or labels')

        def evaluate_spot(self, **kwargs):
            raise PermissionError('Learner cannot evaluate labels')

    dataset = cfg['records'][record]
    args = SimpleNamespace(data=dataset, data_origin='DADA', model_class='Left', gpu=0,
        use_gpu=device.type == 'cuda', gpu_type=device.type, use_multi_gpu=False,
        configs_path=str(runtime), model_save_path=str(out / 'model'),
        results_save_path=str(out / 'unused_native_results'))
    official.EarlyStopping = LoggedEarlyStopping
    try:
        experiment = IsolatedNative(args, id=0)
        clean_keys = set(experiment.model.state_dict())
        initial = state_hashes(experiment.model)
        parameters = sum(p.numel() for p in experiment.model.parameters())
        # Hook counts actual FIT samples; SELECT and THOP do not count as training exposure.
        def exposure(module, args):
            if module.training and optimizer_started[0]:
                exposures[0] += len(args[0])
        hook = experiment.model.register_forward_pre_hook(exposure)
        dump(out / 'INITIAL_STATE.json', dict(state_sha256=initial, parameters=parameters,
             trainable_parameters=parameters, python_step=int(experiment.model.step)))
        experiment.train()  # unmodified original loop/loss/scheduler/early stopping
        hook.remove()
    finally:
        official.EarlyStopping = native_early
    checkpoint = Path(experiment.model_save_path) / 'checkpoint.pth'
    if not saves or sha(checkpoint) != saves[-1]['checkpoint_sha256']:
        raise ValueError('Selected checkpoint provenance mismatch')
    selected = saves[-1]
    extras = sorted(set(experiment.model.state_dict()) - clean_keys)
    if any(not k.endswith(('total_ops', 'total_params')) for k in extras):
        raise ValueError('Unexpected official profile state')
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    selected_epoch = [i + 1 for i, e in enumerate([e for e in events if e['kind']=='SELECT'])
                      if e['update'] == selected['update']][-1]
    dump(out / 'RECEIPT.json', dict(status='COMPLETED', record=record, seed=cfg['seed'],
        official_commit=cfg['official_commit'], dataset=dataset,
        checkpoint=str(checkpoint.relative_to(workspace)), checkpoint_sha256=sha(checkpoint),
        selected_epoch=selected_epoch, selected_update=selected['update'],
        selected_loss=selected['SELECT_loss'], final_python_step=int(experiment.model.step),
        python_step_semantics='Native posttrain: selected checkpoint params/buffers and final unsaved Python step',
        profile_extra_keys=extras, updates=updates[0], FIT_windows_seen=exposures[0],
        epochs=sum(e['kind']=='SELECT' for e in events), parameters=parameters,
        roles=data.access, labels_read=False, scaler='already FIT-only standardized; no second fit',
        selection='native epoch SELECT loss; patience and tie semantics unchanged'))


def score(workspace, cfg, out, record, device, train_out):
    source_setup(workspace, cfg, out)
    set_seed(cfg['evaluation_seed'])
    receipt = read(train_out / 'RECEIPT.json')
    checkpoint = resolve(workspace, receipt['checkpoint'])
    if receipt['record'] != record or receipt['seed'] != cfg['seed'] or sha(checkpoint) != receipt['checkpoint_sha256']:
        raise ValueError('Training identity mismatch')
    model = native_model(workspace, cfg, record, device)
    scientific_state(model, torch.load(checkpoint, map_location='cpu', weights_only=True), receipt['profile_extra_keys'])
    model.step = receipt['final_python_step']
    model.eval()
    initial = state_hashes(model)
    data, banks = RoleReader(workspace, cfg, record, ['CAL', 'QUERY']), []
    for role in ['CAL', 'QUERY']:
        values, index = data.load(role)
        dataset = RoleWindows(values, index)
        channels, indices, ends, availability = [], [], [], []
        batch_size = cfg['score_batch_size']
        with torch.no_grad():
            for a in range(0, len(dataset), batch_size):
                b = min(a + batch_size, len(dataset))
                x = torch.from_numpy(np.stack([dataset[i][0] for i in range(a, b)])).to(device)
                raw = model.infer(x, None, None, None)[0].cpu().numpy()
                if raw.shape != tuple(x.shape) or not np.isfinite(raw).all():
                    raise ValueError('Invalid native score tensor')
                channels.append(raw.reshape(-1, raw.shape[-1]))
                indices.append(index[a*192:b*192])
                ends.append(np.repeat(index[np.arange(a, b)*192+191], 192))
                availability.append(np.full((b-a)*192, index[b*192-1], dtype=np.int64))
        channel = np.concatenate(channels)
        ix, end, available = map(np.concatenate, (indices, ends, availability))
        point_score = np.mean(channel, axis=-1)  # exact official NumPy channel reduction
        path = out / (role + '.npz')
        np.savez_compressed(path, index=ix, score=point_score, per_channel=channel,
            window_end_index=end, available_index=available, valid=np.ones(len(ix), bool))
        banks.append(dict(role=role, path=str(path.relative_to(workspace)), sha256=sha(path),
            points=len(ix), observed_points=len(index), unscored_tail_index=index[len(ix):].tolist()))
    if initial != state_hashes(model) or model.step != receipt['final_python_step']:
        raise ValueError('Inference mutated scientific state')
    # Threshold independent of QUERY, in this score's own units.
    with np.load(out / 'CAL.npz', allow_pickle=False) as bank:
        threshold = float(np.quantile(bank['score'], cfg['cal_quantile'], method='higher'))
    dump(out / 'CALIBRATION.json', dict(threshold=threshold, quantile=cfg['cal_quantile'],
        quantile_method='higher', comparison='>', role='CAL', labels_read=False))
    dump(out / 'RECEIPT.json', dict(status='COMPLETED', record=record, banks=banks,
        checkpoint_sha256=sha(checkpoint), python_step_restored=model.step,
        eval_state_immutable=True, labels_read=False, roles=data.access,
        native_infer=True, score='official native smoothed per-channel score then NumPy mean(-1)',
        time_permission='offline full role-local W192 nonoverlap; tails UNKNOWN',
        available_index='actual batch final input time; window_end_index marks per-window future dependence'))


def cpu_check(workspace, cfg, out):
    source_setup(workspace, cfg, out)
    checks = []
    for record in ['PSM', 'machine-1-1']:
        set_seed(cfg['seed'])
        model = native_model(workspace, cfg, record, torch.device('cpu'))
        channels = 25 if record == 'PSM' else 38
        x = torch.randn(2, 192, channels)
        model.train()
        loss = model(x, None, None, None)[0]
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        model.eval()
        before = state_hashes(model)
        step = model.step
        with torch.no_grad():
            direct = model.infer(x, None, None, None)[0]
            again = model.infer(x, None, None, None)[0]
        assert direct.shape == x.shape and torch.equal(direct, again)
        assert before == state_hashes(model) and step == model.step
        checks.append(dict(record=record, native_shape=list(direct.shape),
                           native_loss=float(loss.detach()), finite_gradients=True,
                           deterministic_eval=True, state_immutable=True))
    dump(out / 'RECEIPT.json', dict(status='PASS', checks=checks, synthetic_only=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['cpu', 'train', 'score'])
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--record')
    parser.add_argument('--train-out')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    config = resolve(workspace, args.config)
    cfg, out = read(config), resolve(workspace, args.out)
    out.mkdir(parents=True, exist_ok=False)
    start, began = time.monotonic(), utc()
    device = torch.device('cpu' if args.phase == 'cpu' else args.device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        cap = cfg['train_memory_gib'] if args.phase == 'train' else cfg['score_memory_gib']
        torch.cuda.set_per_process_memory_fraction(cap * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
        torch.cuda.reset_peak_memory_stats(device)
    status = 'FAILED'
    try:
        if args.phase == 'cpu':
            cpu_check(workspace, cfg, out)
        elif args.phase == 'train':
            train(workspace, cfg, out, args.record, device)
        else:
            score(workspace, cfg, out, args.record, device, resolve(workspace, args.train_out))
        status = 'COMPLETED'
    except Exception:
        (out / 'ERROR.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        seconds = time.monotonic() - start
        dump(out / 'RESOURCE.json', dict(status=status, phase=args.phase, record=args.record,
            pid=os.getpid(), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
            started_utc=began, ended_utc=utc(), seconds=seconds,
            gpu_hours=seconds / 3600 if device.type == 'cuda' else 0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,
            config_sha256=sha(config), runner_sha256=sha(__file__)))


if __name__ == '__main__':
    main()
