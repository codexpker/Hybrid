"""Head-only Stage2 kernels and explicit train/calibrate routes; no query route."""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import time

from ..config import read_json, sha256
from .readout_manifest import json_write, record
from .stage2_identity import (STAGE, HEAD_NAMES, load_stage2, approve_stage2, validate_assets)


def learning_rate(step):
    if type(step) is not int or not 1 <= step <= 1000:raise ValueError('Stage2 step is one-based1..1000')
    return .0003 * min(step/100, 1) * (1 if step <= 100 else .5*(1+math.cos(math.pi*(step-100)/900)))


def module_modes(model):
    return {name: module.training for name, module in model.named_modules()}


def frozen_hashes(model):
    def digest(t):
        t=t.detach().cpu().contiguous()
        return hashlib.sha256(str(t.dtype).encode()+str(tuple(t.shape)).encode()+t.numpy().tobytes()).hexdigest()
    return {**{'parameter:'+n:digest(p) for n,p in model.named_parameters() if n not in HEAD_NAMES},
            **{'buffer:'+n:digest(b) for n,b in model.named_buffers()}}


def configure_head_only(model):
    import torch
    params=dict(model.named_parameters())
    if (not set(HEAD_NAMES) <= set(params) or not isinstance(model.head,torch.nn.Linear)
            or model.head.in_features!=128 or model.head.out_features!=5
            or sum(params[n].numel() for n in HEAD_NAMES)!=645 or any(params[n].dtype!=torch.float32 for n in HEAD_NAMES)):
        raise ValueError('Stage2 requires the existing645-parameter Linear128to5 head')
    model.eval()
    for name,p in params.items():p.requires_grad_(name in HEAD_NAMES);p.grad=None
    model.head.train()
    assert_head_modes(model)
    return [params[n] for n in HEAD_NAMES]


def assert_head_modes(model):
    if any(m.training != (n=='head') for n,m in model.named_modules()):
        raise ValueError('Frozen modules must remain eval and only head training')


def seed_job(seed,device):
    import torch
    torch.random.default_generator.manual_seed(seed)
    if torch.device(device).type=='cuda':
        with torch.cuda.device(device):torch.cuda.manual_seed(seed)


@contextmanager
def evaluation_scope(model,device,mask_seed=84551):
    """One reset per call, preserving every module flag and caller RNG."""
    import torch
    import random
    import numpy as np
    device=torch.device(device)
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    modes=module_modes(model)
    python_rng,numpy_rng=random.getstate(),np.random.get_state()
    try:
        model.eval()
        with torch.no_grad(),torch.random.fork_rng(devices=devices):
            seed_job(mask_seed,device)
            yield
    finally:
        random.setstate(python_rng);np.random.set_state(numpy_rng)
        for name,module in model.named_modules():module.training=modes[name]


def head_only_step(model,optimizer,x,y,step):
    """The shared L0/formal kernel. No model.train(), no inference_mode()."""
    import torch
    from ..model import objective
    assert_head_modes(model)
    if x.shape!=(64,100,1) or y.shape!=(64,5,1) or x.dtype!=torch.float32 or y.dtype!=torch.float32:raise ValueError('Head update requires float32 B64/H100/C1/P5')
    lr=learning_rate(step)
    for group in optimizer.param_groups:group['lr']=lr
    if {id(p) for g in optimizer.param_groups for p in g['params']}!={id(p) for n,p in model.named_parameters() if n in HEAD_NAMES}:
        raise ValueError('Optimizer may contain only existing head parameters')
    optimizer.zero_grad(set_to_none=True)
    output=model(x)
    total,parts=objective(output,y)
    head=[p for n,p in model.named_parameters() if n in HEAD_NAMES]
    tf_gradient=None
    tf_path='NO_GRAD_CONNECTION'
    if parts['tf'].requires_grad:
        gradients=torch.autograd.grad(parts['tf'],head,allow_unused=True,retain_graph=True)
        connected=[g for g in gradients if g is not None]
        tf_gradient=max((float(g.abs().max()) for g in connected),default=None)
        tf_path='ZERO_GRAD_CONNECTION' if connected else 'NO_HEAD_CONNECTION'
        if tf_gradient not in (None,0.):raise ValueError('Frozen TF auxiliary unexpectedly differentiates the head')
    if not torch.isfinite(total):raise ValueError('Nonfinite Stage2 objective')
    total.backward()
    gradients={n:float(p.grad.abs().max()) for n,p in model.named_parameters() if n in HEAD_NAMES and p.grad is not None}
    if (set(gradients)!=set(HEAD_NAMES) or any(p.grad is not None for n,p in model.named_parameters() if n not in HEAD_NAMES)
            or any(not torch.isfinite(p.grad).all() for p in head)):
        raise ValueError('Invalid head/frozen gradient path')
    norm=float(torch.nn.utils.clip_grad_norm_(head,1))
    if not math.isfinite(norm):raise ValueError('Nonfinite head gradient norm')
    after_clip={n:float(p.grad.abs().max()) for n,p in model.named_parameters() if n in HEAD_NAMES}
    optimizer.step()
    return dict(step=step,lr=lr,total_loss=float(total.detach()),future_loss=float(parts['future'].detach()),
                TF_aux=float(parts['tf'].detach()),TF_head_gradient_max=tf_gradient,
                TF_aux_requires_grad=parts['tf'].requires_grad,TF_head_gradient_path=tf_path,
                head_gradient_max=gradients,head_gradient_max_after_clip=after_clip,gradient_norm=norm)


def evaluate_c1(model,batch,*,device,count=256):
    """Select/source-cal C1 evaluation in trace order; count64 supports L0."""
    import numpy as np
    import torch
    from ..model import objective
    if count not in (64,256):raise ValueError('C1 evaluation is fixed256 or explicit discarded L0 prefix64')
    arrays={k:[] for k in ('prediction','target','history_scale','valid_target')};losses=[]
    with evaluation_scope(model,device):
        for start in range(0,count,64):
            x,y=batch(start)
            if x.shape!=(64,100,1) or y.shape!=(64,5,1):raise ValueError('C1 evaluation shape mismatch')
            o=model(x);total,parts=objective(o,y)
            if not torch.isfinite(total):raise ValueError('Nonfinite C1 evaluation loss')
            losses.append([float(total),float(parts['future']),float(parts['tf'])])
            for k,v in (('prediction',o['prediction']),('target',y),('history_scale',o['scale']),('valid_target',torch.isfinite(y))):
                arrays[k].append(v.cpu().numpy())
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    valid=arrays['valid_target'];error=(arrays['prediction']-arrays['target'])[valid]
    return dict(total_loss=float(np.mean(losses,axis=0)[0]),future_loss=float(np.mean(losses,axis=0)[1]),
                TF_aux=float(np.mean(losses,axis=0)[2]),physical_MAE=float(np.abs(error).mean()),
                physical_RMSE=float(np.sqrt((error**2).mean())),normalized_MAE=float((np.abs(arrays['prediction']-arrays['target'])/arrays['history_scale'])[valid].mean()),
                windows=count,scope='L0_ONLY' if count==64 else 'FORMAL_CAL_OR_SELECT'),arrays


def evaluate_target_cal(model,target,source_scale,*,device,count=256):
    """B1C52 pool in frozen order, including nonmonotone origins; never a query."""
    import numpy as np
    from ..protocol import score_block
    if count not in (2,256):raise ValueError('Target cal is fixed256 or explicit discarded L0 prefix2')
    arrays={k:[] for k in ('prediction','target','history_scale','canonical_score','run','origin')}
    with evaluation_scope(model,device):
        for i in range(count):
            x,y,run,origin=target.cal_block(i,device=device)
            if x.shape!=(1,100,52) or y.shape!=(5,52):raise ValueError('Target calibration is B1/H100/C52/P5')
            o=model(x);p=o['prediction'].cpu().numpy()[0];scale=o['scale'].cpu().numpy()[0]
            # Exactly the existing score_block. No target scalar is fitted/divided.
            score=score_block(p,y,scale,source_scale)
            if score.dtype!=np.float32 or not np.isfinite(score).all():raise ValueError('Canonical target score must be finite float32')
            for k,v in (('prediction',p),('target',y),('history_scale',scale),('canonical_score',score),('run',run),('origin',origin)):
                arrays[k].append(v)
    return {k:np.asarray(v) for k,v in arrays.items()}


def create_stage2_model(a,device):
    from ..api import create_predictor
    from .stage2_identity import Assets
    job=a['job'];seed_job(job['seed'],device)
    parent_path=a['assets'].file(a['parent']['checkpoint'],a['paths']['source_references'])
    parent_metadata=a['assets'].file(a['parent']['checkpoint_metadata'],a['paths']['source_references'])
    kwargs=dict(mode=job['mode'],history=100,copies=2,temporal_depth=4,device=device)
    if job['initialization']=='source_pretrained':kwargs.update(checkpoint=parent_path,checkpoint_metadata=parent_metadata)
    # random_init never calls torch.load and never borrows the parent head.
    model=create_predictor(a['official'],**kwargs)
    configure_head_only(model)
    return model


def _identity(a,digest,checkpoint_sha):
    job=a['job']
    return dict(stage=STAGE,checkpoint_sha256=checkpoint_sha,mode=job['mode'],seed=job['seed'],history=100,horizon=5,copies=2,temporal_depth=4,
        step=1000,initialization=job['initialization'],parent_checkpoint_sha256=a['parent']['checkpoint']['sha256'],
        initial_checkpoint_sha256=a['parent']['checkpoint']['sha256'] if job['initialization']=='source_pretrained' else None,
        parent_weights_loaded=job['initialization']=='source_pretrained',
        pretraining_domains=['PSM','SMD_SUBSET'] if job['initialization']=='source_pretrained' else [],
        pretraining_excluded_target='TEP' if job['initialization']=='source_pretrained' else None,adaptation_target='TEP',
        paired_reference_parent=a['parent']['checkpoint'],target_labels_used=False,query_access=False,
        target_manifest_sha256=sha256(a['paths']['target_manifest']),target_plan_sha256=sha256(a['paths']['target_plan']),
        source_cal_trace_sha256=sha256(a['paths']['source_cal_trace']),source_references_sha256=sha256(a['paths']['source_references']),
        recipe_sha256=sha256(a['paths']['recipe']),runtime_sha256=sha256(a['paths']['runtime']),
        trainable_parameters=list(HEAD_NAMES),trainable_count=645,operation_manifest_sha256=digest)


def _snapshot(output,cfg,manifest,a):
    output.mkdir(parents=True,exist_ok=False)
    json_write(output/'operation.json',cfg);json_write(output/'operation_manifest.json',manifest)
    hashes={}
    for p in sorted(Path(__file__).resolve().parents[1].rglob('*.py')):
        name=str(p.relative_to(Path(__file__).resolve().parents[1])).replace('\\','/');hashes[name]=sha256(p)
        q=output/'implementation'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    json_write(output/'provenance.json',dict(package_code=hashes,external_inputs={str(p):h for p,h in a['assets'].hashes.items()},upstream_copied=False))
    return hashes


def _finish(a,code):
    a['assets'].recheck()
    package=Path(__file__).resolve().parents[1]
    if any(sha256(package/name)!=h for name,h in code.items()):raise ValueError('Stage2 package changed during execution')


def diagnostic_replay(model,x,y,device):
    """No optimizer step. Compare real gradient paths under restored actual RNG."""
    import random
    import numpy as np
    import torch
    from ..model import objective
    head=[p for n,p in model.named_parameters() if n in HEAD_NAMES]
    cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state(device).clone() if torch.device(device).type=='cuda' else None
    py=random.getstate();nr=np.random.get_state();flags=module_modes(model)
    def restore():
        torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state(cuda,device)
        random.setstate(py);np.random.set_state(nr)
    def pass_once(target,total_gradient):
        o=model(x);loss,parts=objective(o,target)
        grads=torch.autograd.grad(loss if total_gradient else parts['future'],head)
        return o,float(loss),[g.detach().cpu().clone() for g in grads],parts['tf'].requires_grad
    try:
        a,total,g1,tf_connected=pass_once(y,True);restore()
        b,_,g2,_=pass_once(y,False);restore()
        c,perturbed,_,_=pass_once(y+10*a['scale'].detach(),True)
        if (not all(torch.equal(u,v) for u,v in zip(g1,g2)) or not torch.equal(a['prediction'],b['prediction'])
                or not torch.equal(a['prediction'],c['prediction']) or not torch.equal(a['tf_history_normalized'],c['tf_history_normalized'])
                or total==perturbed):
            raise ValueError('L0 gradient/mask/future-only replay assertion failed')
        return dict(total_head_grad_equals_future_only=True,prediction_replay_bitexact=True,
            future_perturbation_changes_loss_only=True,TF_aux_has_grad_connection=tf_connected,
            rng_restored=True,extra_optimizer_steps=0)
    finally:
        restore()
        if module_modes(model)!=flags:raise ValueError('L0 diagnostics changed module flags')


def rng_identity(device):
    import random
    import numpy as np
    import torch
    def h(value):return hashlib.sha256(value).hexdigest()
    nr=np.random.get_state()
    return dict(torch_cpu=h(torch.get_rng_state().numpy().tobytes()),
        torch_device=h(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()) if torch.device(device).type=='cuda' else None,
        numpy=h(nr[0].encode()+nr[1].tobytes()+repr(nr[2:]).encode()),python=h(repr(random.getstate()).encode()))


def source_scale_only(bank):
    """Exact source.calibrate_source prefix, without fitting any L0 threshold."""
    import numpy as np
    e = np.abs(bank["prediction"] - bank["target"]) / bank["history_scale"]
    valid = bank["valid_target"]
    pointvalid = valid.any(-1)
    point = np.divide(np.where(valid, e, 0).sum(-1), valid.sum(-1),
                      out=np.full(pointvalid.shape, np.nan, dtype="float32"), where=pointvalid)
    scale = max(float(np.median(point[pointvalid])), 1e-8)
    return scale


def l0_discard(a,output,*,device,cfg,digest):
    """Exactly five updates; no formal checkpoint/calibration or reusable state."""
    import numpy as np
    import torch
    from .stage2_data import TargetNormal
    from .stage2_identity import input_identity
    from .training import source_batch
    from ..data import ManifestSampler
    started=time.monotonic()
    target=TargetNormal(a['target'],a['plan'],a['assets'].file(a['target']['cache'],a['paths']['target_manifest']))
    model=create_stage2_model(a,device);head=configure_head_only(model)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in HEAD_NAMES}
    before=frozen_hashes(model);flags=module_modes(model);requires={n:p.requires_grad for n,p in model.named_parameters()}
    opt=torch.optim.AdamW(head,lr=.0003,weight_decay=.01);curve=[]
    for step in range(1,6):
        x,y=target.batch('fit',(step-1)*64,device=device)
        old={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in HEAD_NAMES}
        row=head_only_step(model,opt,x,y,step)
        row['head_update_max']={n:float((p.detach().cpu()-old[n]).abs().max()) for n,p in model.named_parameters() if n in HEAD_NAMES}
        if any(v<=0 for v in row['head_update_max'].values()) or any(v<=0 for v in row['head_gradient_max'].values()):raise ValueError('L0 has no effective head update')
        curve.append(row)
        if time.monotonic()-started>120:raise ValueError('L0_DISCARD120second budget exceeded')
    rng_after_updates=rng_identity(device)
    diagnostics=diagnostic_replay(model,x,y,device)
    select,select_arrays=evaluate_c1(model,lambda start:target.batch('select',start,device=device),device=device,count=64)
    src=a['source'];sampler=ManifestSampler(src['paths']['source_manifest'],exclude_dataset='TEP')
    source_metrics,source_arrays=evaluate_c1(model,lambda start:source_batch(sampler,src['trace'],src['rows'],'cal',start,device=device,history=100),device=device,count=64)
    temporary_source_unit=source_scale_only(source_arrays)
    target_arrays=evaluate_target_cal(model,target,temporary_source_unit,device=device,count=2)
    rng_after_evaluations=rng_identity(device)
    if rng_after_updates!=rng_after_evaluations:raise ValueError('L0 select/cal consumed training RNG')
    after=frozen_hashes(model)
    if (before!=after or module_modes(model)!=flags or {n:p.requires_grad for n,p in model.named_parameters()}!=requires):
        raise ValueError('L0 frozen values/module modes/requires_grad changed')
    if time.monotonic()-started>120:raise ValueError('L0_DISCARD120second budget exceeded')
    np.savez_compressed(output/'L0_ONLY_source_routes.npz',**source_arrays)
    np.savez_compressed(output/'L0_ONLY_target_routes.npz',**target_arrays)
    np.savez_compressed(output/'L0_ONLY_select_routes.npz',**select_arrays)
    changes={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in model.named_parameters() if n in HEAD_NAMES}
    result=dict(status='U_NORMAL_HEAD_L0_DISCARD_COMPLETE',stage=STAGE,job_id=a['job']['job_id'],steps=5,exposure_windows=320,
        primary_checkpoint_written=False,optimizer_state_written=False,weights_discarded=True,calibration_reusable=False,
        diagnostic_calibration_status='L0_ONLY_NON_REUSABLE_NO_THRESHOLD_FITTED',temporary_source_unit=temporary_source_unit,
        source_first64_domains=sorted({src['rows'][int(i)]['dataset'] for i in src['trace']['cal_record'][:64]}),
        source_prefix_is_full_domain_coverage=False,curve=curve,head_changes=changes,frozen_before=before,frozen_after=after,
        module_flags_preserved=True,module_modes=flags,requires_grad=requires,diagnostics=diagnostics,
        rng_after_updates=rng_after_updates,rng_after_evaluations=rng_after_evaluations,
        select=select,source_diagnostic=source_metrics,target_cal_order=[dict(run=int(r),origin=int(b)) for r,b in zip(target_arrays['run'],target_arrays['origin'])],
        inputs=input_identity(cfg),parent_checkpoint_sha256=a['parent']['checkpoint']['sha256'],initialization=a['job']['initialization'],
        parent_weights_loaded=a['job']['initialization']=='source_pretrained',operation_manifest_sha256=digest,
        GPU_executed=torch.device(device).type=='cuda',peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if torch.device(device).type=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if torch.device(device).type=='cuda' else None,
        seconds=time.monotonic()-started,query_or_labels_read=False)
    del model,opt,head,target
    return result


def train(a,output,*,device,digest):
    """Formal1000 only. Discarded L0 calls the kernels, not this loop."""
    import numpy as np
    import torch
    from .stage2_data import TargetNormal
    target=TargetNormal(a['target'],a['plan'],a['assets'].file(a['target']['cache'],a['paths']['target_manifest']))
    model=create_stage2_model(a,device);head=configure_head_only(model);frozen=frozen_hashes(model)
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in HEAD_NAMES}
    optimizer=torch.optim.AdamW(head,lr=.0003,weight_decay=.01)
    curve=[];selection=[];best=float('inf');best_step=None;started=time.perf_counter()
    for step in range(1,1001):
        x,y=target.batch('fit',(step-1)*64,device=device)
        curve.append(head_only_step(model,optimizer,x,y,step))
        if step%100==0:
            metrics,arrays=evaluate_c1(model,lambda start:target.batch('select',start,device=device),device=device)
            assert_head_modes(model)
            if frozen_hashes(model)!=frozen:raise ValueError('Frozen backbone/buffer changed during adaptation')
            selection.append(dict(step=step,**metrics));np.savez_compressed(output/f'select_step{step}.npz',**arrays)
            if metrics['future_loss']<best:
                best,best_step=metrics['future_loss'],step;torch.save(model.state_dict(),output/'best.pt')
        if torch.device(device).type=='cuda' and torch.cuda.max_memory_allocated(device)>20*1024**3:raise ValueError('Stage2 exceeded20GiB')
    changes={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in model.named_parameters() if n in HEAD_NAMES}
    if any(v==0 for v in changes.values()):raise ValueError('Head parameter did not actually update')
    checkpoint=output/'step1000.pt';torch.save(model.state_dict(),checkpoint)
    torch.save({'optimizer':optimizer.state_dict(),'step':1000},output/'head_optimizer_step1000.pt')
    meta=_identity(a,digest,sha256(checkpoint));meta.update(frozen_before=frozen,frozen_after=frozen_hashes(model),head_parameter_changes=changes)
    json_write(output/'checkpoint_metadata.json',meta);json_write(output/'curve.json',curve);json_write(output/'select.json',selection)
    return dict(status='U_NORMAL_HEAD_STEP1000_COMPLETE',stage=STAGE,job_id=a['job']['job_id'],primary=record(checkpoint),
        checkpoint_metadata=record(output/'checkpoint_metadata.json'),steps=1000,exposure_windows=64000,trainable_count=645,
        best_step_supplement=best_step,best_checkpoint_supplement=record(output/'best.pt'),optimizer_checkpoint=record(output/'head_optimizer_step1000.pt'),
        primary_step=1000,head_changes=changes,frozen_hashes=frozen,head_mode_after_select=module_modes(model),
        seconds=time.perf_counter()-started,query_access=False,labels_used=False,identity=meta)


def validate_child(a,paths):
    m=read_json(paths['checkpoint_metadata']);summary=read_json(paths['training_summary'])
    archive_root=paths['training_summary'].parent
    archive_path=archive_root/'archive_manifest.json'
    if ((archive_root/'failure.json').exists() or not archive_path.is_file()
            or paths['checkpoint'].parent!=archive_root or paths['checkpoint_metadata'].parent!=archive_root):
        raise ValueError('Stage2 training identity mismatch: incomplete or foreign archive')
    a['assets'].file(record(archive_path),archive_path)
    archive=read_json(archive_path)
    if (archive.get('status')!='STAGE2_ARCHIVE_COMPLETE' or archive.get('stage')!=STAGE or archive.get('operation')!='stage2-train'
            or any(archive['files'].get(p.name)!=sha256(p) for p in (paths['checkpoint'],paths['checkpoint_metadata'],paths['training_summary']))):
        raise ValueError('Stage2 training identity mismatch: frozen archive changed')
    expected=_identity(a,m.get('operation_manifest_sha256'),sha256(paths['checkpoint']))
    if (any(m.get(k)!=v for k,v in expected.items()) or 'excluded_target' in m or m.get('frozen_before')!=m.get('frozen_after')
            or summary.get('status')!='U_NORMAL_HEAD_STEP1000_COMPLETE' or summary.get('job_id')!=a['job']['job_id']
            or summary.get('identity')!=m or summary.get('primary')!=record(paths['checkpoint'])
            or summary.get('checkpoint_metadata')!=record(paths['checkpoint_metadata']) or summary.get('steps')!=1000):
        raise ValueError('Stage2 checkpoint/parent/target/trace/training identity mismatch')
    return m


def calibration_tracks(source_path,target_path,checkpoint_sha):
    """Validate units/roles and expose two thresholds over ONE canonical object.

    No new score array, ranking metric, query accessor or inference is created.
    """
    import numpy as np
    from .stage2_identity import Assets
    from .source import calibrate_source
    from .stage2_data import validate_target
    from ..protocol import score_block
    a=Assets();s=read_json(source_path);t=read_json(target_path)
    if (s.get('stage')!=STAGE or t.get('stage')!=STAGE or s.get('role')!='source_cal_after_adaptation'
            or t.get('role')!='target_threshold_only' or s.get('fitted_checkpoint_sha256')!=checkpoint_sha
            or t.get('fitted_checkpoint_sha256')!=checkpoint_sha or t.get('target_global_rescale') is not False
            or t.get('schema_version')!=2 or t.get('pool_is_contiguous_event') is not False
            or t.get('runtime_sha256')!=s.get('runtime_sha256') or t.get('operation_manifest_sha256')!=s.get('operation_manifest_sha256')
            or set(t)!={'schema_version','stage','role','fitted_checkpoint_sha256','source_calibration','source_error_scale',
                          'target_manifest_sha256','target_plan_sha256','canonical_scores','threshold99','target_global_rescale','pool_is_contiguous_event',
                          'runtime_sha256','operation_manifest_sha256','target_cal_bank','target_manifest','target_plan','seed','fitted_checkpoint_metadata'}):
        raise ValueError('Stage2 calibration role/fitted checkpoint/target-rescale mismatch')
    if a.file(t['source_calibration'],target_path)!=Path(source_path).resolve():raise ValueError('Target threshold source identity mismatch')
    with np.load(a.file(s['source_bank'],source_path),allow_pickle=False) as bank:expected,params=calibrate_source(bank)
    with np.load(a.file(s['canonical_scores'],source_path),allow_pickle=False) as scores:
        if (set(scores.files)!=set(expected) or any(not np.array_equal(scores[k],v,equal_nan=True) for k,v in expected.items())
                or s['normal_error_scale']!=params['normal_error_scale'] or s['threshold99']!=params['threshold99']):
            raise ValueError('Source canonical units differ from the fixed calibration bank')
    if t['source_error_scale']!=s['normal_error_scale']:raise ValueError('Target cannot replace the fitted source scale')
    if (type(t['seed']) is not int or t['seed'] not in (0,1,2)
            or t['target_manifest']['sha256']!=t['target_manifest_sha256'] or t['target_plan']['sha256']!=t['target_plan_sha256']):
        raise ValueError('Target calibration manifest/trace identity mismatch')
    fitted=read_json(a.file(t['fitted_checkpoint_metadata'],target_path))
    if (fitted.get('stage')!=STAGE or fitted.get('checkpoint_sha256')!=checkpoint_sha
            or any(fitted.get(k)!=t[k] for k in ('seed','target_manifest_sha256','target_plan_sha256','runtime_sha256'))):
        raise ValueError('Target calibration fitted checkpoint/trace identity mismatch')
    _,plan=validate_target(a.file(t['target_manifest'],target_path),a.file(t['target_plan'],target_path),t['seed'],a)
    with np.load(a.file(t['target_cal_bank'],target_path),allow_pickle=False) as raw:
        if (set(raw.files)!={'prediction','target','history_scale','run','origin'}
                or any(raw[k].dtype!=np.float32 or raw[k].shape!=shape for k,shape in
                       [('prediction',(256,5,52)),('target',(256,5,52)),('history_scale',(256,1,52))])
                or any(not np.array_equal(raw[k],plan['cal_'+k]) for k in ('run','origin'))):
            raise ValueError('Target calibration bank shape or frozen trace order mismatch')
        expected_target=np.stack([score_block(p,y,scale,s['normal_error_scale'])
            for p,y,scale in zip(raw['prediction'],raw['target'],raw['history_scale'])])
    score_ref=t['canonical_scores'];score_path=a.file(score_ref,target_path)
    with np.load(score_path,allow_pickle=False) as bank:
        score=bank['canonical_score']
        if (set(bank.files)!={'canonical_score','run','origin'}
                or any(not np.array_equal(bank[k],plan['cal_'+k]) for k in ('run','origin'))
                or not np.array_equal(score,expected_target)):
            raise ValueError('Target canonical scores differ from the bound physical calibration bank or trace')
        if score.dtype!=np.float32 or score.shape!=(256,5) or not np.isfinite(score).all() or t['threshold99']!=float(np.quantile(score,.99,method='linear')):
            raise ValueError('Target threshold differs from the single canonical float32 object')
    a.recheck()
    return dict(scope='target_normal_calibration_pool_only_no_query_score_created',source=dict(score_object=score_ref,threshold=s['threshold99']),
                target=dict(score_object=score_ref,threshold=t['threshold99']))


def calibrate(a,output,*,device,digest):
    import numpy as np
    import torch
    from ..api import create_predictor
    from ..data import ManifestSampler
    from .training import source_batch
    from .source import calibrate_source
    from .stage2_data import TargetNormal
    m=validate_child(a,a['paths'])
    model=create_predictor(a['official'],mode=a['job']['mode'],history=100,checkpoint=a['paths']['checkpoint'],
        checkpoint_metadata=a['paths']['checkpoint_metadata'],device=device)
    configure_head_only(model);before=frozen_hashes(model)
    if before!=m['frozen_after']:raise ValueError('Loaded Stage2 frozen backbone identity mismatch')
    src=a['source'];sampler=ManifestSampler(src['paths']['source_manifest'],exclude_dataset='TEP')
    source_metrics,source=evaluate_c1(model,lambda start:source_batch(sampler,src['trace'],src['rows'],'cal',start,device=device,history=100),device=device)
    scores,source_cal=calibrate_source(source)
    np.savez_compressed(output/'source_cal_bank.npz',**source);np.savez_compressed(output/'source_canonical_scores.npz',**scores)
    source_identity=dict(schema_version=1,stage=STAGE,role='source_cal_after_adaptation',fitted_checkpoint_sha256=m['checkpoint_sha256'],
        runtime_sha256=m['runtime_sha256'],operation_manifest_sha256=digest,
        source_cal_trace_sha256=sha256(a['paths']['source_cal_trace']),source_references_sha256=sha256(a['paths']['source_references']),
        source_bank=record(output/'source_cal_bank.npz'),canonical_scores=record(output/'source_canonical_scores.npz'),**source_cal)
    json_write(output/'source_calibration.json',source_identity)
    target=TargetNormal(a['target'],a['plan'],a['assets'].file(a['target']['cache'],a['paths']['target_manifest']))
    target_arrays=evaluate_target_cal(model,target,source_cal['normal_error_scale'],device=device)
    np.savez_compressed(output/'target_cal_bank.npz',**{k:v for k,v in target_arrays.items() if k!='canonical_score'})
    np.savez_compressed(output/'target_canonical_scores.npz',canonical_score=target_arrays['canonical_score'],run=target_arrays['run'],origin=target_arrays['origin'])
    target_identity=dict(schema_version=2,stage=STAGE,role='target_threshold_only',fitted_checkpoint_sha256=m['checkpoint_sha256'],
        runtime_sha256=m['runtime_sha256'],operation_manifest_sha256=digest,
        source_calibration=record(output/'source_calibration.json'),source_error_scale=source_cal['normal_error_scale'],
        target_manifest_sha256=sha256(a['paths']['target_manifest']),target_plan_sha256=sha256(a['paths']['target_plan']),
        target_manifest=record(a['paths']['target_manifest']),target_plan=record(a['paths']['target_plan']),seed=a['job']['seed'],
        fitted_checkpoint_metadata=record(a['paths']['checkpoint_metadata']),
        target_cal_bank=record(output/'target_cal_bank.npz'),
        canonical_scores=record(output/'target_canonical_scores.npz'),threshold99=float(np.quantile(target_arrays['canonical_score'],.99,method='linear')),
        target_global_rescale=False,pool_is_contiguous_event=False)
    json_write(output/'target_threshold.json',target_identity)
    tracks=calibration_tracks(output/'source_calibration.json',output/'target_threshold.json',m['checkpoint_sha256'])
    if before!=frozen_hashes(model):raise ValueError('Frozen state changed during calibration')
    assert_head_modes(model)
    return dict(status='U_NORMAL_HEAD_CALIBRATION_COMPLETE',stage=STAGE,fitted_checkpoint_sha256=m['checkpoint_sha256'],
        source=record(output/'source_calibration.json'),target=record(output/'target_threshold.json'),source_metrics=source_metrics,
        canonical_policy='one source-canonical score object per pool/model; S/T only change threshold',
        threshold_tracks=tracks,module_modes_after_cal=module_modes(model),operation_manifest_sha256=digest,query_access=False,labels_used=False)


def run_stage2(config_path,*,execute=False):
    invoked=time.monotonic()
    cfg,paths,output,manifest,digest=load_stage2(config_path)
    if not execute:return dict(executed=False,operation_manifest=manifest,operation_manifest_sha256=digest,scope='preview only; no weights/data/labels opened')
    approval=approve_stage2(cfg,paths,output,digest)
    a=validate_assets(cfg,paths)
    a['assets'].hashes.update(approval.get('_entry_assets',{}))
    import torch
    torch.set_num_threads(2);device=cfg['parameters']['device']
    if device=='cuda:0':
        if not torch.cuda.is_available():raise ValueError('Approved CUDA device unavailable; no fallback')
        torch.cuda.set_device(device);torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    code=_snapshot(output,cfg,manifest,a)
    try:
        if cfg['operation']=='stage2-train':result=train(a,output,device=device,digest=digest)
        elif cfg['operation']=='stage2-calibrate':result=calibrate(a,output,device=device,digest=digest)
        else:result=l0_discard(a,output,device=device,cfg=cfg,digest=digest)
        if torch.device(device).type=='cuda' and torch.cuda.max_memory_allocated(device)>20*1024**3:raise ValueError('Stage2 exceeded20GiB')
        if cfg['operation']=='stage2-l0' and time.monotonic()-invoked>120:raise ValueError('L0 operation including checks exceeded120seconds')
        result['operation_body_seconds']=time.monotonic()-invoked
        result['timing_scope']='includes input checks/model/computation/output before final summary; OS startup/exit receipt external'
        _finish(a,code);json_write(output/'summary.json',result)
        json_write(output/'archive_manifest.json',dict(status='STAGE2_ARCHIVE_COMPLETE',stage=STAGE,operation=cfg['operation'],
            files={str(p.relative_to(output)).replace('\\','/'):sha256(p) for p in output.rglob('*') if p.is_file()},
            scientific_training_evidence=False if approval.get('scope')=='SYNTHETIC_ENGINEERING_ONLY' else None))
        return result
    except Exception as exc:
        json_write(output/'failure.json',dict(status='STAGE2_FAILED_NO_COMPLETE_SUMMARY',error=str(exc)))
        raise
