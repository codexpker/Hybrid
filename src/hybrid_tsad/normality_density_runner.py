"""Role-isolated training/readout for the fixed next-generation Hybrid study.

Original datasets, baselines and experimental snapshots are read-only inputs.
Model/data configuration is required; this module contains no query tuning.
"""
from contextlib import contextmanager
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def resolve(workspace, path):
    p = Path(path)
    return p if p.is_absolute() else workspace / p


class RoleReader:
    """Check permission and exact source identity before opening values."""
    def __init__(self, workspace, domain, allowed):
        self.workspace, self.domain, self.allowed = workspace, domain, set(allowed)
        self.access = []
        if domain == 'SWAT':
            self.prep = workspace / 'preparations/left_multievent_mixed_u_20261111_v1'
            self.rows = read(self.prep / 'INPUT_QUALIFICATION.json')['roles']
            self.scaler_file = self.prep / 'FIT_ONLY_SCALER.json'
        else:
            self.prep = workspace / 'preparations/hybrid_native_predictive_cycle_20261113_v1'
            self.rows = [e for e in read(self.prep / 'DATA_GATE.json')['roles'] if e['domain'] == domain]
            self.scaler_file = self.prep / (domain + '_FIT_ONLY_SCALER.json')
        self.scaler = read(self.scaler_file)

    def load(self, role):
        if role not in self.allowed:
            raise PermissionError('Role refused before values access: ' + role)
        row = next(e for e in self.rows if e['role'] == role)
        file = self.prep / 'data' / ((role if self.domain == 'SWAT' else self.domain + '_' + role) + '.npz')
        assert sha(file) == row['sha256'], 'Data identity mismatch: ' + str(file)
        with np.load(file, allow_pickle=False) as z:
            assert z['observed'].all()
            values = z['values'].astype(np.float64)
            if values.ndim == 2:
                values = values[None]
            runs = z['run'].copy() if 'run' in z else np.array([0])
            faults = z['fault'].copy() if 'fault' in z else np.array([0])
            offsets = z['original_start'].copy() if 'original_start' in z else np.array([row['original_start']])
        values = np.float32((values - np.asarray(self.scaler['mean'])) / np.asarray(self.scaler['scale']))
        assert np.isfinite(values).all()
        self.access.append(dict(role=role, path=str(file), sha256=row['sha256'], shape=list(values.shape),
                                original_start=np.asarray(offsets).tolist()))
        return values, runs, faults, np.atleast_1d(offsets)


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.default_generators[device.index or 0].manual_seed(seed)


def model_from_config(cfg, domain, variant, device):
    factory = cfg.get('model_factory', 'hybrid_tsad.normality_density:NormalityDensity')
    module, name = factory.split(':')
    klass = getattr(importlib.import_module(module), name)
    spec = cfg['domains'][domain]
    return klass(spec['channels'], spec['history'], variant=variant, **cfg.get('model_kwargs', {})).to(device)


def sample_batch(values, H, run, end, device):
    assert np.all(end >= H) and np.all(end < values.shape[1])
    arr = np.stack([values[r, t-H:t+1] for r, t in zip(run, end)])
    return torch.from_numpy(arr).to(device)


def initialise_fit_model(cfg,model,fit,out):
    """Optional factory hook; historical models/configurations do not enter it."""
    if not cfg.get('initialise_from_fit',False):return
    record=model.initialise_from_fit(fit)
    record['state_sha256']={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                            for k,v in model.state_dict().items()}
    record['active_parameters']=sum(p.numel()for p in model.parameters()if p.requires_grad)
    record['all_parameters']=sum(p.numel()for p in model.parameters())
    record['FIT_only']=True
    dump(out/'INITIALISATION.json',record)


@contextmanager
def validation_mode(model, device):
    modes = [(m, m.training) for m in model.modules()]
    py, nrng = random.getstate(), np.random.get_state()
    devices = [device.index or 0] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            with torch.no_grad():
                yield
    finally:
        for m, mode in modes:
            m.training = mode
        random.setstate(py)
        np.random.set_state(nrng)


def selection(model, values, H, device, batch_size):
    runs, ends = np.meshgrid(np.arange(len(values)), np.arange(H, values.shape[1], H), indexing='ij')
    runs, ends = runs.ravel(), ends.ravel()
    total = count = 0
    with validation_mode(model, device):
        for i in range(0, len(ends), batch_size):
            x = sample_batch(values, H, runs[i:i+batch_size], ends[i:i+batch_size], device)
            loss = model.loss(x)
            assert loss.ndim == 0 and torch.isfinite(loss)
            total += float(loss) * len(x)
            count += len(x)
    assert count > 0
    return total / count


def training(cfg, workspace, out, domain, variant, device):
    spec = cfg['domains'][domain]
    H, B = spec['history'], cfg['batch_size']
    roles = RoleReader(workspace, domain, ['FIT', 'SELECT'])
    fit, _, _, _ = roles.load('FIT')
    val, _, _, _ = roles.load('SELECT')
    assert fit.shape[-1] == spec['channels']
    seed_everything(cfg['seed'], device)
    model = model_from_config(cfg, domain, variant, device)
    initialise_fit_model(cfg,model,fit,out)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    rng = np.random.default_rng(cfg['drawplan_seed'])
    # This complete plan is fixed before the first optimizer update, no labels.
    n = len(fit)*(fit.shape[1]-H)
    per_epoch = int(np.ceil(n/B))
    epochs = spec.get('epochs',cfg.get('epochs',20))
    steps = spec.get('updates',epochs*per_epoch)
    plan_source = None
    if domain in ('TEP','PSM'):
        plan_source = roles.prep/'data'/(domain+'_DRAWPLAN.npz')
        expected = next(e['sha256'] for e in read(roles.prep/'DATA_GATE.json')['plans'] if e['domain']==domain)
        assert sha(plan_source)==expected
        with np.load(plan_source) as z:order=z['order'].copy()
        assert order.shape==(epochs,n)
    else:
        order = np.stack([rng.permutation(n).astype(np.int32) for _ in range(int(np.ceil(steps/per_epoch)))])
    np.savez_compressed(out / 'drawplan.npz', order=order)
    best = float('inf')
    trace = []
    patience = spec.get('patience')
    stale = 0
    stop = False
    start = time.monotonic()
    for step in range(1, steps + 1):
        model.train()
        ep, bi = divmod(step-1,per_epoch)
        ids = order[ep,bi*B:(bi+1)*B]
        run, end = ids//(fit.shape[1]-H),H+ids%(fit.shape[1]-H)
        x = sample_batch(fit, H, run, end, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x)
        assert loss.ndim == 0 and torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
        grad_norm = float(torch.stack([g.detach().norm() for g in grads]).norm())
        optimizer.step()
        row = dict(update=step,epoch=ep+1, loss=float(loss.detach()), gradient_norm=grad_norm,
                   learning_rate=optimizer.param_groups[0]['lr'], seconds=time.monotonic()-start)
        validate_every=cfg.get('validation_every',per_epoch)
        if step % validate_every == 0 or step == steps:
            vloss = selection(model, val, H, device, B)
            row['SELECT_loss'] = vloss
            if vloss < best:
                best = vloss
                stale = 0
                tmp = out / 'selected.tmp'
                torch.save(dict(state=model.state_dict(), optimizer=optimizer.state_dict(),
                                update=step, SELECT_loss=best, config=cfg, domain=domain, variant=variant), tmp)
                os.replace(tmp, out / 'selected.pth')
                row['selected_sha256'] = sha(out / 'selected.pth')
            else:
                stale += 1
            row['patience_counter'] = stale
            stop = patience is not None and stale>=patience
            print(json.dumps(dict(event='VALIDATION', domain=domain, variant=variant, **row)), flush=True)
        trace.append(row)
        with (out / 'TRAIN_TRACE.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        if stop:break
        if step%per_epoch==0:
            epoch=step//per_epoch
            if spec.get('lr_schedule','type2')=='type1':
                lr=cfg['learning_rate']*(.5**epoch)
            else:
                lr={2:5e-5,4:1e-5,6:5e-6,8:1e-6,10:5e-7,15:1e-7,20:5e-8}.get(epoch)
            if lr is not None:
                for group in optimizer.param_groups:group['lr']=lr
    torch.save(dict(state=model.state_dict(), update=step), out / 'final.pth')
    dump(out / 'TRAIN_RECEIPT.json', dict(status='COMPLETED', updates=step,maximum_updates=steps,
         selected_sha256=sha(out/'selected.pth'), final_sha256=sha(out/'final.pth'),
         drawplan_sha256=sha(out/'drawplan.npz'), roles=roles.access,
         historical_plan_sha256=sha(plan_source) if plan_source else None,
         scaler_sha256=sha(roles.scaler_file), CAL_QUERY_unopened=True,
         selected_on_normal_SELECT_only=True, seed=cfg['seed']))


def scoring(cfg, workspace, out, domain, variant, device, identity=False):
    spec = cfg['domains'][domain]
    H, B = spec['history'], cfg.get('score_batch_size', cfg['batch_size'])
    checkpoint = out.parent / 'train/selected.pth'
    if not identity:
        train = read(checkpoint.parent / 'TRAIN_RECEIPT.json')
        assert sha(checkpoint) == train['selected_sha256']
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert state['config'] == cfg and state['domain'] == domain and state['variant'] == variant
    seed_everything(cfg['seed'], device)
    model = model_from_config(cfg, domain, 'full' if identity else variant, device)
    if not identity:model.load_state_dict(state['state'], strict=True)
    model.eval()
    allowed = ['CAL','QUERY'] + (['NORMAL_TEST'] if domain == 'TEP' else [])
    roles = RoleReader(workspace, domain, allowed)
    if not identity:assert sha(roles.scaler_file) == train['scaler_sha256']
    banks = []
    with torch.no_grad():
        for role in allowed:
            values, runs, faults, offsets = roles.load(role)
            scores, indices = [], []
            for r in range(len(values)):
                mature = H if role != 'QUERY' or domain == 'TEP' else max(512,H)
                ends = np.arange(mature, values.shape[1])
                index = ends + offsets[r] + (1 if domain == 'TEP' else 0)
                parts = []
                for i in range(0,len(ends),B):
                    es = ends[i:i+B]
                    x = sample_batch(values,H,np.full(len(es),r),es,device)
                    evidence = model.identity_score(x) if identity else model.score(x)
                    assert evidence.shape == (len(es),spec['channels']) and bool(torch.isfinite(evidence).all())
                    parts.append(evidence.cpu().numpy())
                scores.append(np.concatenate(parts)); indices.append(index)
            channels = np.stack(scores)
            score = channels[...,0] if cfg.get('score_representation')=='replicated_scalar' else channels.mean(-1)
            if cfg.get('score_representation')=='replicated_scalar':
                assert np.array_equal(channels,np.broadcast_to(score[...,None],channels.shape))
            f = out/(role+'.npz')
            np.savez_compressed(f,index=np.stack(indices),raw_score=score,channel_score=channels,
                 run=np.broadcast_to(runs[:,None],score.shape),fault=np.broadcast_to(faults[:,None],score.shape),
                 available_index=np.stack(indices),max_input_index=np.stack(indices),
                 TEP_target9_diagnostic=channels[...,0] if domain=='TEP'and cfg.get('score_representation')!='replicated_scalar'else np.empty(0))
            banks.append(dict(role=role,path=str(f),sha256=sha(f),shape=list(score.shape)))
    dump(out/'SCORE_RECEIPT.json',dict(status='COMPLETED',selected_PT_sha256=None if identity else sha(checkpoint),
         selected_update=None if identity else state['update'],roles=roles.access,banks=banks,labels_unopened=True,
         threshold_not_fitted=True,no_QUERY_statistics=True,
         **({'channel_score_semantics':'replicated joint scalar; not channel attribution',
             'raw_score_taken_from_scalar_copy':True}if cfg.get('score_representation')=='replicated_scalar'else{})))


def l0(cfg,workspace,out,domain,variant,device):
    spec=cfg['domains'][domain]
    H,B=spec['history'],cfg['batch_size']
    roles=RoleReader(workspace,domain,['FIT','SELECT'])
    values,*_=roles.load('FIT')
    val,*_=roles.load('SELECT')
    seed_everything(cfg['seed'],device)
    model=model_from_config(cfg,domain,variant,device)
    initialise_fit_model(cfg,model,values,out)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    x=sample_batch(values,H,np.zeros(B,int),np.arange(H,H+B),device)
    timings=[]
    for _ in range(2):
        begin=time.monotonic();model.train();optimizer.zero_grad();loss=model.loss(x)
        assert loss.ndim==0 and bool(torch.isfinite(loss));loss.backward()
        assert all(bool(torch.isfinite(p.grad).all())for p in model.parameters()if p.grad is not None)
        optimizer.step()
        if device.type=='cuda':torch.cuda.synchronize(device)
        timings.append(time.monotonic()-begin)
    model.eval()
    with torch.no_grad():scores=model.score(x)
    assert scores.shape==(B,spec['channels'])and bool(torch.isfinite(scores).all())
    selected=selection(model,val,H,device,B)
    dump(out/'L0_RECEIPT.json',dict(status='PASS_DISCARDED',domain=domain,variant=variant,
         batch_size=B,loss=float(loss.detach()),normal_SELECT_loss=selected,
         update_seconds=timings,parameters=sum(p.numel()for p in model.parameters()),
         roles=roles.access,no_CAL_QUERY=True,no_checkpoint_saved=True,
         **({'score_scalar_mean':float(scores[:,0].mean()),'score_scalar_min':float(scores[:,0].min()),
             'score_scalar_max':float(scores[:,0].max()),
             'loss_score_semantics':'normalized trajectory NLL for learning; predictive Rosenblatt compliance for detection; equality not asserted'}
            if cfg.get('score_representation')=='replicated_scalar'else{})))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['train','score','identity','l0'])
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--domain',choices=['PSM','SWAT','TEP'],required=True)
    ap.add_argument('--variant',required=True)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    cfg=read(args.config)
    assert args.variant in cfg['variants']
    out=args.output.resolve()/args.domain/args.variant/('score' if args.phase=='identity'else args.phase)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    device=torch.device(args.device)
    start=time.time()
    if device.type=='cuda':
        torch.cuda.set_device(device)
        cap=20 if args.phase in ('train','l0') else 30
        torch.cuda.set_per_process_memory_fraction(cap*1024**3/torch.cuda.get_device_properties(device).total_memory,device)
        torch.cuda.reset_peak_memory_stats(device)
    dump(out/'START.json',dict(pid=os.getpid(),start_unix=start,phase=args.phase,domain=args.domain,
         variant=args.variant,config_sha256=sha(args.config),runner_sha256=sha(__file__),
         CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),device=str(device)))
    rc=0
    try:
        if args.phase=='identity':scoring(cfg,args.workspace.resolve(),out,args.domain,args.variant,device,identity=True)
        else:{'train':training,'score':scoring,'l0':l0}[args.phase](cfg,args.workspace.resolve(),out,args.domain,args.variant,device)
    except BaseException:
        rc=1
        dump(out/'FAILED.json',dict(traceback=traceback.format_exc(),exit_code=1))
        raise
    finally:
        end=time.time()
        dump(out/'OS.json',dict(pid=os.getpid(),start_unix=start,end_unix=end,seconds=end-start,
             GPU_hours=(end-start)/3600 if device.type=='cuda'else 0,exit_code=rc,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda'else 0,
             peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda'else 0,
             CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')))


if __name__=='__main__':
    main()
