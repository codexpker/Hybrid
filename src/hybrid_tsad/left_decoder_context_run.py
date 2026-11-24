"""Train and score the fixed LEFT decoder-context comparison on public records."""

import argparse
import hashlib
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .left_decoder_context import ARMS, REVISION, channel_context, model_class
from .left_swat_author import dump, read, sha

OUT = Path('results/left_decoder_context_20261124_v1')
BASE = Path('preparations/left_baseline_build_20261101_v1')
SWAT_PREP = Path('preparations/left_multievent_mixed_u_20261111_v1')
SWA = Path('results/left_swat_author_20261118_v1')
PSM = Path('data/common_psm_raw_20261029_v1')
DOMAINS = ('PSM', 'SWAT')
WINDOW = 192


def fingerprint(state):
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def environment(w, domain):
    upstream = w/BASE/'upstream'
    reference = read(w/SWA/'SOURCE_BINDING.json')
    assert all(sha(upstream/path) == digest for path, digest in reference['source_files'].items())
    configs = upstream/'configs' if domain == 'PSM' else w/'preparations/left_swat_author_20261118_v1/configs'
    if domain == 'SWAT':
        assert all(sha(configs/path) == digest for path, digest in reference['configs'].items())
    sys.path[:0] = [str(upstream), str(w/BASE/'deps_thop_transport2_v1/site')]
    from exp import exp_anomaly_detection as official
    return official, configs, reference['source_files']


def normal(w, domain):
    if domain == 'SWAT':
        src = w/SWAT_PREP/'data/ALL_VALUES.npy'
        data = np.load(src, mmap_mode='r', allow_pickle=False)
        assert data.shape == (944919, 31) and data.dtype == np.float64
        raw = np.array(data[:495000], dtype=np.float64, order='F', copy=True)
        cut = 396000
    else:
        src = w/PSM/'prefix.npy'
        assert sha(src) == 'f722bce0b19324d85c7b5691efe73453eb56022e867ccc1ba47bb3406787df38'
        raw = np.array(np.load(src, allow_pickle=False), dtype=np.float64, order='F', copy=True)
        assert raw.shape == (132481, 25)
        cut = 105984
    scaler = StandardScaler().fit(raw)
    if domain == 'SWAT':
        with np.load(w/SWA/'SCALER.npz') as old:
            for key in ('mean', 'scale', 'var'):
                np.testing.assert_array_equal(getattr(scaler, key+'_'), old[key])
    return scaler.transform(raw), cut, scaler, hashlib.sha256(raw.tobytes()).hexdigest()


class Windows(Dataset):
    def __init__(self, values, stride):
        self.values = values
        self.stride = stride

    def __len__(self):
        return (len(self.values)-WINDOW)//self.stride+1

    def __getitem__(self, index):
        start = index*self.stride
        return np.float32(self.values[start:start+WINDOW]), np.zeros((WINDOW, 1), np.float32)


def check(w, domain):
    official, configs, _ = environment(w, domain)
    samples, _, _, _ = normal(w, domain)
    cfg = official.Configs(str(configs/domain/'model_configs_Left.json'))
    torch.manual_seed(2025)
    x = torch.from_numpy(np.stack((samples[:192], samples[192:384])).astype(np.float32)).cuda()
    native = official.Left(cfg).cuda()
    shared = fingerprint({k:v for k,v in native.state_dict().items() if not k.startswith('ms_decoder.to_gb.')})
    # Exercise both sample/channel alignment and the complete native forward.
    tags = torch.arange(2)[:,None]
    torch.testing.assert_close(channel_context(tags, x.shape[-1]).reshape(2,-1),
                               tags.expand(2,x.shape[-1]))
    fitted = {}
    for arm in ARMS:
        candidate = model_class(official.Left, arm)(cfg).cuda()
        candidate.load_state_dict(native.state_dict())
        if arm != 'native':
            with torch.no_grad():
                candidate.ms_decoder.to_gb[-1].weight.zero_()
                candidate.ms_decoder.to_gb[-1].bias.zero_()
        assert shared == fingerprint({k:v for k,v in candidate.state_dict().items()
                                      if not k.startswith('ms_decoder.to_gb.')})
        candidate.eval()
        original = fingerprint(candidate.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(candidate(x,None,None,None)[0], native.eval()(x,None,None,None)[0], rtol=0, atol=0)
            torch.testing.assert_close(candidate.infer(x,None,None,None)[0], native.infer(x,None,None,None)[0], rtol=0, atol=0)
        assert candidate.step == 0 and original == fingerprint(candidate.state_dict())
        candidate.train()
        candidate.step = candidate.warmup_steps + candidate.ramp_steps
        candidate.conf_threshold = 0.0
        candidate.update_q_err = candidate.update_q_ad = 1.0
        calls = [0,0,0,0]
        for i, bank in enumerate((candidate.proto_t,candidate.proto_f)):
            def count(_module, _args, _output, index=i):
                calls[index] += 1
            bank.register_forward_hook(count)
            update = bank.ema_update_hard
            def tracked(*args, index=i, fn=update, **kw):
                calls[index+2] += 1
                return fn(*args, **kw)
            bank.ema_update_hard = tracked
        step = candidate.step
        loss, _ = candidate(x,None,None,None)
        loss.backward()
        assert calls == [1,1,1,1] and candidate.step == step+1
        g = candidate.ms_decoder.to_gb[-1].weight.grad
        norm = 0.0 if g is None else float(g.norm())
        assert (norm > 0) == (arm != 'native')
        assert all(torch.isfinite(p.grad).all() for p in candidate.parameters() if p.grad is not None)
        fitted[arm] = dict(loss_bit_exact_at_zero_modulation=True, score_bit_exact_at_zero_modulation=True,
                          association_and_forced_EMA_calls=calls, native_step_increment=1,
                          to_gb_weight_grad_norm=norm, inference_frozen=True)
    dump(w/OUT/domain/'CHECKS.json', dict(passed=True, public_normal_only=True, results=fitted))


def run(w, domain, arm):
    out = w/OUT/domain/arm
    out.mkdir(parents=True, exist_ok=False)
    assert read(out.parent/'CHECKS.json')['passed']
    official, configs, source = environment(w, domain)
    normalized, cut, scaler, raw_identity = normal(w, domain)
    normal_points = len(normalized)
    config_paths = sorted((configs/domain).glob('*.json'))
    random.seed(2025)
    np.random.seed(2025)
    torch.manual_seed(2025)
    torch.cuda.manual_seed_all(2025)
    kind = model_class(official.Left, arm)
    trace, selected = [], []
    updates = 0
    optimizer = None
    original_stopping = official.EarlyStopping

    class Stopping(original_stopping):
        def save_checkpoint(self, loss, model, path):
            super().save_checkpoint(loss, model, path)
            selected.append(dict(epoch=len(trace), loss=float(loss), step=model.step,
                                 sha256=sha(Path(path)/'checkpoint.pth')))

    class Experiment(official.Exp_Anomaly_Detection):
        def _acquire_device(self):
            return torch.device('cuda:0')

        def _build_model(self):
            return kind(self.hyper_args)

        def _get_data(self, flag, step=None):
            assert flag in ('train', 'val') and self.model_configs.seq_len == WINDOW
            assert step == (1 if flag=='train' else None)
            ds = Windows(normalized[:cut] if flag=='train' else normalized[cut:],
                         1 if flag=='train' else WINDOW)
            return ds, DataLoader(ds,batch_size=self.train_configs.batch_size,
                                  shuffle=flag=='train',drop_last=False,num_workers=0)

        def _select_optimizer(self):
            nonlocal optimizer
            optimizer = super()._select_optimizer()
            def count(_optim, _args, _kwargs):
                nonlocal updates
                updates += 1
            optimizer.register_step_post_hook(count)
            return optimizer

        def vali(self, loader):
            loss = super().vali(loader)
            assert np.isfinite(loss)
            trace.append(dict(epoch=len(trace)+1, validation_loss=float(loss), updates=updates,
                              step=self.model.step, cpu_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
            dump(out/f'epoch{len(trace):02d}.json',trace[-1])
            return loss

    official.EarlyStopping = Stopping
    args = SimpleNamespace(data=domain, data_origin='DADA', model_class='Left', gpu=0,
        use_gpu=True, gpu_type='cuda', use_multi_gpu=False, configs_path=str(configs),
        model_save_path=str(out/'model'), results_save_path=str(out/'unused_results'))
    exp = Experiment(args,id=0)
    config = exp.model
    init = fingerprint({k:v for k,v in config.state_dict().items() if not k.startswith('ms_decoder.to_gb.')})
    np.savez_compressed(out/'SCALER.npz', mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_)
    dump(out/'CONFIG.json', dict(domain=domain, arm=arm, seed=2025, official_reference_revision=REVISION,
         upstream_files=source, configuration={p.name:sha(p) for p in config_paths},
         common_initial_state_sha256=init, scaler_sha256=sha(out/'SCALER.npz'),
         normal_values_sha256=raw_identity, FIT=[0,cut], SELECT=[cut,normal_points],
         FIT_stride=1, SELECT_stride=WINDOW, channels=config.enc_in, window=WINDOW,
         scaler_fit=[0,normal_points], curriculum=dict(warmup=config.warmup_steps,ramp=config.ramp_steps,maximum=config.lambda_max),
         batch=exp.train_configs.batch_size, max_epochs=exp.train_configs.train_epochs,
         early_stop_patience=exp.train_configs.patience, optimizer=exp.train_configs.optim, lr=exp.train_configs.learning_rate,
         lr_schedule=exp.train_configs.lradj, labels_read=False))
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    exp.train()
    torch.cuda.synchronize()
    train_seconds = time.monotonic()-start
    active = [(n,p.numel()) for n,p in config.named_parameters() if p in optimizer.state]
    parameters = dict(registered=sum(p.numel() for p in config.parameters()), updated=sum(v for _,v in active),
                      active_modulation=sum(v for n,v in active if n.startswith('ms_decoder.to_gb.')))
    assert parameters['active_modulation'] == (0 if arm=='native' else 33280)
    ckpt = Path(exp.model_save_path)/'checkpoint.pth'
    assert selected[-1]['sha256'] == sha(ckpt)
    dump(out/'TRAIN_RECEIPT.json', dict(status='COMPLETED', selected=selected[-1], validation=trace,
         selection_history=selected, updates=updates, final_step=config.step, parameters=parameters,
         train_seconds=train_seconds, train_peak_reserved=torch.cuda.max_memory_reserved(),
         test_values_read=False, labels_read=False))
    # Read test values only after the author's normal-validation selection.
    if domain == 'SWAT':
        values = np.load(w/SWAT_PREP/'data/ALL_VALUES.npy', mmap_mode='r', allow_pickle=False)[normal_points:]
    else:
        p = w/PSM/'test.npy'
        assert sha(p) == '6592518615d690ab153592b6dd0220c424441a0d4ee7bac87c3f89fe942893f5'
        values = np.load(p, allow_pickle=False)
    raw_test_hash = hashlib.sha256(values.tobytes()).hexdigest()
    scaled = scaler.transform(np.array(values,dtype=np.float64,order='F',copy=True))
    config.eval()
    before, step = fingerprint(config.state_dict()), config.step
    start = time.monotonic()
    with torch.no_grad():
        pieces = [config.infer(x.cuda(),None,None,None)[0].cpu().numpy()
                  for x,_ in DataLoader(Windows(scaled,WINDOW),batch_size=exp.train_configs.batch_size,shuffle=False)]
    raw_score = np.concatenate(pieces).reshape(-1,config.enc_in)
    score = raw_score.mean(axis=-1)
    assert np.isfinite(score).all() and before==fingerprint(config.state_dict()) and config.step==step
    index = np.arange(normal_points,normal_points+len(score))
    bank = out/'raw_scores.npz'
    np.savez_compressed(bank,canonical_score=score,per_channel=raw_score,absolute_time_index=index,
                        available_index=normal_points+(np.arange(len(score))//WINDOW+1)*WINDOW-1)
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',bank_sha256=sha(bank),raw_test_values_sha256=raw_test_hash,
         checkpoint_sha256=sha(ckpt), scoring_seconds=time.monotonic()-start, points=len(score),
         omitted_tail=len(values)-len(score), state_unchanged=True, labels_read=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=('check','run'))
    p.add_argument('--domain',choices=DOMAINS,required=True)
    p.add_argument('--arm',choices=ARMS)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    root = Path.cwd()
    (root/OUT/a.domain).mkdir(parents=True,exist_ok=True)
    if a.action == 'check':
        check(root,a.domain)
    else:
        assert a.arm is not None
        run(root,a.domain,a.arm)
