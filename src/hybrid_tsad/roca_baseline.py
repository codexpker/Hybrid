"""Legal-role adapter around the fixed official RoCA model and complete objective.

The external MIT source remains byte-for-byte unchanged.  We reuse its model,
augmentation and loss/center functions.  Dataset permissions, endpoint indexing,
fixed-endpoint training and calibration are project adapters, not author results.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

COMMIT = '82405ddca519e772842f55b2c5f2cfe00d885ea4'
CORE = ('models/RoCA/roca_network/model.py',
        'models/RoCA/roca_trainer/trainer.py', 'dataloader/augmentations.py')
TEMPLATES = ('IOpsCompetition', 'UCR', 'SWaT', 'WADI')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def load_official(root, source_sha256):
    """No Merlion import: only exact author computational functions are compiled.

    The AST bodies are unmodified; the omitted imports/Trainer are the author's
    overlapping-label loader, metric reporting and test-selected checkpoint path.
    Namespaces belong to this invocation; there is no global monkey patch.
    """
    import torch
    import torch.nn.functional as F
    root = Path(root)
    needed = set(CORE) | {'conf/roca/' + name + '_Configs.py' for name in TEMPLATES}
    if not needed <= set(source_sha256):
        raise PermissionError('Incomplete fixed official method source closure')
    for relative, digest in source_sha256.items():
        if sha(root / relative) != digest:
            raise PermissionError('Official source changed: ' + relative)

    def module(relative):
        spec = importlib.util.spec_from_file_location(
            '_roca_external_' + source_sha256[relative], root / relative)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    tree = ast.parse((root / CORE[1]).read_text())
    names = {'train', 'center_c', 'model_train', 'model_evaluate', 'get_radius'}
    bodies = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name in names]
    if {n.name for n in bodies} != names:
        raise PermissionError('Official computational function API changed')
    namespace = dict(torch=torch, F=F, np=np)
    exec(compile(ast.Module(body=bodies, type_ignores=[]), str(root / CORE[1]), 'exec'), namespace)
    return SimpleNamespace(Model=module(CORE[0]).base_Model,
                           augmentation=module(CORE[2]).DataTransform,
                           functions=namespace,
                           configs={name: module('conf/roca/' + name + '_Configs.py').Config
                                    for name in TEMPLATES},
                           provenance=dict(commit=COMMIT, source_sha256=dict(source_sha256),
                                           functions=sorted(names), AST_bodies_unmodified=True))


def config_from_template(core, template, input_channels=None, overrides=None):
    if template not in TEMPLATES:
        raise PermissionError('No official SMAP or SMD configuration exists')
    cfg = core.configs[template]()
    if input_channels is not None:
        cfg.input_channels = int(input_channels)
    for key, value in (overrides or {}).items():
        if key not in vars(cfg) or key in ('dataset', 'augmentation'):
            raise PermissionError('Unknown or structural template override: ' + key)
        setattr(cfg, key, value)
    if cfg.input_channels < 1 or cfg.window_size < 1:
        raise ValueError('Positive fixed channel/window shape required')
    return cfg


def config_dict(cfg):
    return {key: dict(vars(value)) if key == 'augmentation' else value
            for key, value in vars(cfg).items()}


def restore_config(value):
    value = dict(value)
    value['augmentation'] = SimpleNamespace(**value['augmentation'])
    return SimpleNamespace(**value)


@dataclass(frozen=True)
class FitScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, runs):
        if not runs or any(r.role != 'FIT' for r in runs):
            raise PermissionError('Only physically admitted FIT values may fit normalization')
        channels = {r.values.shape[1] for r in runs}
        if len(channels) != 1:
            raise PermissionError('RoCA input channels are fixed per model')
        x = np.concatenate([np.asarray(r.values, np.float64) for r in runs])
        if not np.isfinite(x).all() or any(not np.asarray(r.observed, bool).all() for r in runs):
            raise ValueError('Missing-value policy is not inferred from QUERY')
        from sklearn.preprocessing import StandardScaler
        fitted = StandardScaler().fit(x)
        return cls(fitted.mean_.copy(), fitted.scale_.copy())

    def transform(self, values):
        x = np.asarray(values, np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Fixed finite channel shape required')
        return ((x - self.mean) / self.scale).astype(np.float32)


def endpoint_windows(run, width, indices, scaler, required_role):
    """Only [t-W+1,t] of the same original record; current t is available."""
    if run.role != required_role or required_role not in ('FIT', 'SELECT', 'CAL', 'QUERY'):
        raise PermissionError('Role permission mismatch')
    indices = np.asarray(indices, np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError('Explicit unique original endpoint indices required')
    x = scaler.transform(run.values); rows = []
    for index in indices:
        end = int(index) - int(run.start) + 1; begin = end - width
        if width < 1 or begin < 0 or end > len(x):
            raise PermissionError('Endpoint crosses original role/run boundary')
        if not np.asarray(run.observed[begin:end], bool).all():
            raise PermissionError('Unobserved input has no approved imputation')
        rows.append(x[begin:end].T)
    if not rows:
        raise ValueError('Empty window pool')
    return np.stack(rows), [dict(index=int(t), available_index=int(t),
                               input_bounds=[int(t) - width + 1, int(t) + 1],
                               domain=run.domain, entity=run.entity, run=run.run,
                               role=required_role) for t in indices]


def validate_role_boundaries(runs, guard):
    """A separate filename cannot authorize overlapping original record intervals."""
    for a in runs:
        for b in runs:
            if a is b or a.role == b.role or (a.domain, a.entity, a.run) != (b.domain, b.entity, b.run):
                continue
            left, right = sorted((a, b), key=lambda r: r.start)
            gap = right.start - (left.start + len(left.values))
            if gap < guard:
                raise PermissionError('Cross-role original intervals violate fixed guard')


def make_loader(core, windows, cfg, seed, shuffle):
    """Author augmentations are generated once, before DataLoader shuffling."""
    import torch
    x = torch.as_tensor(np.asarray(windows, np.float32))
    if x.ndim != 3 or tuple(x.shape[1:]) != (cfg.input_channels, cfg.window_size):
        raise ValueError('RoCA needs B,C,fixed-W')
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        a, b = core.augmentation(x, cfg)
    finally:
        np.random.set_state(state)
    # Dummy targets solely satisfy the author function interface, never labels.
    dataset = torch.utils.data.TensorDataset(x, torch.zeros(len(x), dtype=torch.int64),
                                            torch.as_tensor(a), torch.as_tensor(b))
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                                       drop_last=cfg.drop_last, num_workers=0, generator=generator)


@contextmanager
def eval_preserving_modes(model):
    before = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, mode in before:
            module.training = mode


def invariance_scores(feature, reconstruction, center):
    """Same normalizations/epsilon/order as official train()'s returned loss_n."""
    import torch.nn.functional as F
    c = F.normalize(center.unsqueeze(0), dim=1)
    q = F.normalize(feature, dim=1); qr = F.normalize(reconstruction, dim=1)
    d = 1 - F.cosine_similarity(q, c, eps=1e-6)
    dr = 1 - F.cosine_similarity(qr, c, eps=1e-6)
    return d + dr


def score_windows(model, windows, center, batch_size=512):
    import torch
    result = []
    with eval_preserving_modes(model), torch.no_grad():
        for begin in range(0, len(windows), batch_size):
            x = torch.as_tensor(windows[begin:begin + batch_size], dtype=torch.float32,
                                device=model.device)
            q, qr = model(x)
            values = invariance_scores(q, qr, center)
            if not torch.isfinite(values).all():
                raise FloatingPointError('Nonfinite original normality score')
            result.append(values.detach().cpu().numpy())
    return np.concatenate(result)


def train_epoch(core, model, optimizer, loader, center, cfg, epoch, trace_sink=None):
    """Label-free operational loop; exact author concatenation/objective/Adam math."""
    import torch
    model.train(); rows = []
    for batch, (x, unused_target, a, b) in enumerate(loader):
        data = torch.cat((x.float().to(model.device), a.float().to(model.device),
                          b.float().to(model.device)), dim=0)
        optimizer.zero_grad()
        q, qr = model(data)
        oc, variance, normal, abnormal = core.functions['train'](
            q, qr, center, epoch, cfg, model.device)
        loss = oc + cfg.omega * variance
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite complete RoCA objective')
        exposed = epoch >= cfg.warmup and cfg.train_method == 'contaminated'
        count = int(len(data) * cfg.nu) if exposed else 0
        selected = torch.topk(normal, count, largest=True, sorted=False).indices if count else None
        parts = dict(normality=oc, variance=variance)
        if count:
            parts['OE'] = cfg.mu * abnormal[selected].mean()
        evidence = {}
        for name, component in parts.items():
            grads = torch.autograd.grad(component, (q, qr), retain_graph=True, allow_unused=True)
            evidence[name] = sum(float(g.detach().abs().sum()) for g in grads if g is not None)
        selected_groups = {name: 0 for name in ('original', 'scaling', 'jitter')}
        if count:
            for group, name in enumerate(selected_groups):
                selected_groups[name] = int(((selected >= group * len(x)) & (selected < (group+1)*len(x))).sum())
        loss.backward()
        gradients = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        if not gradients or not all(torch.isfinite(g).all() for g in gradients.values()):
            raise FloatingPointError('Nonfinite or absent train gradients')
        norm = sum(float(g.detach().abs().sum()) for g in gradients.values())
        optimizer.step()
        row = dict(epoch=epoch, batch=batch, original_batch=len(x), augmented_batch=len(data),
                   objective=float(loss.detach()), normality=float(oc.detach()),
                   variance=float(variance.detach()), OE_selected=count,
                   warmup=not exposed, gradient_L1=norm, component_projection_gradient_L1=evidence,
                   OE_selected_groups=selected_groups, target_labels_used=False)
        rows.append(row)
        if trace_sink is not None:
            trace_sink(row)
    if not rows:
        raise ValueError('Empty training epoch; no replacement sampling is invented')
    return rows


def fit(core, model, loader, cfg, output, select_windows=None, technical=False):
    """Fixed epoch endpoint. SELECT is descriptive only; no QUERY/test calls."""
    import torch
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    dump(output / 'MODEL_CONFIG.json', config_dict(cfg))
    torch.save(model.state_dict(), output / 'initial.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                 weight_decay=cfg.weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    center = torch.zeros(cfg.project_channels, device=model.device)
    center = core.functions['center_c'](loader, model, model.device, center, cfg, eps=cfg.center_eps)
    if not torch.isfinite(center).all() or not torch.any(center != 0):
        raise FloatingPointError('One-class center is zero/nonfinite')
    torch.save(center.detach().clone(), output / 'initial_center.pt')
    center_history = [dict(epoch=0, sha256=hashlib.sha256(center.detach().cpu().numpy().tobytes()).hexdigest())]
    rows = []; selections = []
    with (output / 'trace.jsonl').open('x') as stream:
        def trace(row):
            stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
        try:
            for epoch in range(1, cfg.num_epoch + 1):
                current = train_epoch(core, model, optimizer, loader, center, cfg, epoch, trace)
                rows.extend(current)
                if select_windows is not None:
                    values = score_windows(model, select_windows, center, cfg.batch_size)
                    selections.append(dict(epoch=epoch, mean_normality=float(values.mean()),
                                           labels_used=False, checkpoint_selection=False))
                if epoch < cfg.change_center_epoch:
                    # Preserve the actual author's in-place carry/update rule.
                    center = core.functions['center_c'](loader, model, model.device, center,
                                                         cfg, eps=cfg.center_eps)
                center_history.append(dict(epoch=epoch, updated=epoch < cfg.change_center_epoch,
                    sha256=hashlib.sha256(center.detach().cpu().numpy().tobytes()).hexdigest()))
                scheduler.step(torch.tensor([r['objective'] for r in current]).mean())
        except BaseException as error:
            dump(output / 'FAILURE.json', dict(error=repr(error), completed_updates=len(rows),
                                              no_retry=True))
            raise
    updates = len(rows); exposed = sum(r['OE_selected'] for r in rows)
    valid = updates > 0 and (technical or (cfg.train_method == 'contaminated' and exposed > 0))
    checkpoint = dict(model_state_dict=model.state_dict(), center=center.detach(),
                      model_config=config_dict(cfg), official_provenance=core.provenance)
    torch.save(checkpoint, output / 'endpoint.pt'); torch.save(optimizer.state_dict(), output / 'optimizer.pt')
    report = dict(status='COMPLETED', scientific_status='VALID_FINITE_ENDPOINT' if valid else 'INCOMPLETE_NO_OE_EXPOSURE',
                  epochs=cfg.num_epoch, updates=updates, OE_selected_total=exposed,
                  variance_active=cfg.omega != 0, center_history=center_history,
                  select_diagnostics=selections, fixed_endpoint=True, labels_used=False,
                  technical_discarded=technical, source_provenance=core.provenance)
    dump(output / 'training.json', report); dump(output / 'RECEIPT.json', report)
    return report


def reload_endpoint(core, checkpoint, device):
    import torch
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = restore_config(data['model_config'])
    model = core.Model(cfg, device).to(device)
    model.load_state_dict(data['model_state_dict'], strict=True)
    center = data['center'].to(device)
    return model, center, cfg


def calibrate(scores, role, quantile=.99):
    if role != 'CAL':
        raise PermissionError('Threshold can only use the approved CAL role')
    values = np.asarray(scores, np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite full CAL score pool required')
    threshold = np.sort(values)[max(0, int(np.ceil(quantile*len(values)))-1)]
    return dict(method='RoCA_FULL_NORMALITY_VARIANCE_OE', threshold=float(threshold),
                quantile=quantile, samples=len(values), score='official_loss_n',
                target_labels_used=False, score_normalization='none')
