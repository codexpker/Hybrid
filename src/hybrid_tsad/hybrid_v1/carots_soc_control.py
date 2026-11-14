"""Bounded published SOC objective control on original Hybrid V1 histories.

No model/routing/scoring definitions are changed. Oracle normal augmentations
are a synthetic mechanism control, not a causal-discovery implementation.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import random
from pathlib import Path
import time

import numpy as np

FEATURE_ID = "V1_FORECAST_T_THEN_F_CHANNEL_MAJOR_FLAT_IDENTITY_V1"
SOC_WEIGHT, SOC_TEMPERATURE, SOC_SIMILARITY = .1, .1, .5
NAMESPACE = "CAROTS_SOC_CONTROL_20261105_V1"
CASES = ("CROSS_N", "CROSS_LAG12_20", "CROSS_BREAK", "NO_DRIVER0_COUPLING",
         "NORMAL_DRIFT", "HARD_NORMAL", "AUGMENT_NORMAL")
SCIENCE_FREEZE_SHA = "35f8daec3d2d40e0880c7e222c573062c6c98999eaf27237081f6183da433192"


def normal_relation_controls(values):
    """Fixed normal drift/hard control; preserve response conditional mechanism."""
    x=np.asarray(values,dtype=np.float64)
    if x.shape!=(2560,3) or not np.isfinite(x).all():
        raise ValueError('Full finite C3 base query required')
    times=np.arange(2560)
    progress=np.clip((times-768)/768.,0,1)
    ramp=.5*(1-np.cos(np.pi*progress))
    envelope=np.where((times>=768)&(times<1536),np.sin(np.pi*progress)**2,0.)
    result={}
    for name,delta in [('NORMAL_DRIFT',ramp),('HARD_NORMAL',envelope*x[:,0])]:
        past=np.r_[np.zeros(12),delta[:-12]]
        y=x.copy();y[:,0]+=delta;y[:,1]+=.3*past;y[:,2]-=.3*past
        result[name]=y.astype(np.float32)
    return result


def prepare_control_inputs(output, science_freeze):
    """Generate once: TRAIN_AUGMENT and new query remain physical separate."""
    from . import r2_learnability as old
    from . import controlled_mechanism as generator
    if old.sha(science_freeze) != SCIENCE_FREEZE_SHA:
        raise PermissionError('Exact root scientific freeze required')
    freeze = json.loads(Path(science_freeze).read_text())
    if (freeze['namespace'] != NAMESPACE or freeze['test']['cases'] != list(CASES) or
            old.sha(generator.__file__) != old.GENERATOR_SHA):
        raise PermissionError('Original generator and fixed scientific identity required')
    out = Path(output);out.mkdir(parents=True, exist_ok=False)
    for name in ('train_augment', 'query', 'audit_only'):
        (out/name).mkdir()
    def save(path, values, role, name):
        x = np.asarray(values, dtype=np.float32)
        if x.shape != (2560, 3) or not np.isfinite(x).all():
            raise ValueError('Fixed finite C3 fixture required')
        with path.open('xb') as f:
            np.savez_compressed(f, values=x, observed=np.ones_like(x, dtype=bool))
        return dict(domain=NAMESPACE, entity='cross_mixed_sign', run=name, role=role,
            path=str(path.resolve()), format='npz', key='values', observed_key='observed',
            sha256=old.sha(path), expected_shape=[2560,3], bounds=[0,2560], original_start=0)
    seeds = {};aug=[]
    for key, rid in (('PARTNER',9410),('DRIVER_A',9411),('DRIVER_B',9412)):
        pair, truth = generator.generate_pair('cross_mixed_sign',run_id=rid,role=NAMESPACE,
            n=2560,burn_in=4096,event=(768,1536),d0=12,d1=20)
        aug.append(save(out/'train_augment'/(key+'.npz'),pair['N']['values'],'FIT',key))
        seeds[key] = dict(latent_run=rid, stream_seeds=truth['seeds'],
                          usage='TRAIN_AUGMENT_ONLY_NORMAL_N',retained_indices=truth['retained_indices'].tolist())
    cases, truth, nulltruth = old.build_query_inputs(query_run=9504,independent_run=9505,role=NAMESPACE)
    pair, extra = generator.generate_pair('cross_mixed_sign',run_id=9506,role=NAMESPACE,
        n=2560,burn_in=4096,event=(768,1536),d0=12,d1=20)
    # Smooth event ramp 0->1 retained after event, extended by zero pre-run.
    # All updates are generated from latent past, never array roll/wrap.
    x = cases['CROSS_N']['values'].astype(np.float64)
    for name,y in normal_relation_controls(x).items():
        cases[name]=dict(values=y)
    cases['AUGMENT_NORMAL'] = dict(values=((x+pair['N']['values'].astype(np.float64))/np.sqrt(2.)).astype(np.float32))
    entries=[save(out/'query'/(case+'.npz'),cases[case]['values'],'QUERY',case) for case in CASES]
    old.dump(out/'AUGMENT_MANIFEST.json',dict(runs=aug,usage='TRAIN_AUGMENT_ONLY',
        namespace=NAMESPACE,conceptual_role='TRAIN_AUGMENT; physical loader role FIT; never old FIT anchors'))
    old.dump(out/'QUERY_MANIFEST.json',dict(runs=entries))
    original=truth['cross_mixed_sign/'+NAMESPACE]
    with (out/'audit_only/QUERY_LABELS.npz').open('xb') as f:
        np.savez_compressed(f,**{case:((np.arange(2560)>=768)&(np.arange(2560)<1536)).astype(np.int8)
            if case in ('CROSS_LAG12_20','CROSS_BREAK') else np.zeros(2560,dtype=np.int8) for case in CASES})
    old.dump(out/'audit_only/SEMANTICS.json',dict(namespace=NAMESPACE,train_augment=seeds,
        query_run=9504,query_seedbook=original['seeds'],query_retained_indices=original['retained_indices'],
        independent_null_run=9505,null_seeds=nulltruth['seeds'],normal_mix_run=9506,mix_seeds=extra['seeds'],
        event=[768,1536],index=[512,2560],SHAM=dict(alias='CROSS_N',extra_forward=False),
        normal_cases=['CROSS_N','NORMAL_DRIFT','HARD_NORMAL','AUGMENT_NORMAL'],
        NULL='NO_DRIVER0_COUPLING mechanism only; health FPR/event metrics NA',
        DRIFT='driver+=g; responses+=+/-0.3*g(t-12); g=.5*(1-cos(pi*clip((t-768)/768,0,1))), zero pre-event,1 post-event',
        HARD_NORMAL='driver*=1+sin(pi*clipped_event_progress)^2; responses+=+/-0.3*(u_new-u_old)(t-12); response noise unchanged; driver innovation regime changed',
        AUGMENT_NORMAL='(N9504+independentN9506)/sqrt2, full joint Gaussian superposition',
        training_augmentation='GP original/independent joint superposition; GN different independent driver replacement only',
        oracle_normality_scope='Known synthetic linear Gaussian mechanism; not public-data causality discovery'))
    result=dict(status='PREPARED_NOT_TRAINED',normal_roles_reused=[9201,9202,9203],
        augmentation_runs=[9410,9411,9412],query_runs=[9504,9505,9506],query_inputs=7,
        augmentation_manifest_sha256=old.sha(out/'AUGMENT_MANIFEST.json'),
        query_manifest_sha256=old.sha(out/'QUERY_MANIFEST.json'),query_labels_read=False,
        science_freeze_sha256=SCIENCE_FREEZE_SHA)
    old.dump(out/'RECEIPT.json',result)
    return result


def relation_views(history, partner, independent_driver):
    """Equal law GP mixture and marginal-matched driver replacement GN.

    Four inputs are independent *training-only* stationary normal realizations
    of the same linear Gaussian generator. This function cannot certify them.
    Inputs and outputs have [H,3] order (driver,positive response,negative).
    """
    x, v, u = (np.asarray(z, dtype=np.float64) for z in
               (history, partner, independent_driver))
    if x.ndim != 2 or x.shape != v.shape or x.shape[1] != 3 or u.shape != (2, len(x)):
        raise ValueError("Same-length C3 normal contexts plus two independent drivers required")
    if not all(np.isfinite(z).all() for z in (x, v, u)):
        raise ValueError("Finite fixed oracle histories required")
    a, b = x.copy(), (x+v)/np.sqrt(2.)
    c, d = a.copy(), b.copy()
    c[:, 0] = u[0]
    d[:, 0] = u[1]
    out = np.stack((a, b, c, d)).astype(np.float32)
    if not np.array_equal(out[0, :, 1:], out[2, :, 1:]) or not np.array_equal(out[1, :, 1:], out[3, :, 1:]):
        raise AssertionError("GN may change only the driver")
    return out


def forecast_features(forecast):
    """Keep channel layout; do not mean channels or append residual scores."""
    import torch
    t, f = forecast["t_features"], forecast["f_features"]
    if t.ndim != 3 or t.shape != f.shape or t.shape[1] != 3:
        raise ValueError("Fixed C3 per-channel time/fractional endpoint features required")
    out = torch.cat((t.flatten(1), f.flatten(1)), dim=1)
    if not bool(torch.isfinite(out).all()):
        raise FloatingPointError("Non-finite fixed-layout relation feature")
    return out


@contextmanager
def soc_randomness(model, seed):
    """Additional objective work must not consume any original SSL RNG stream."""
    import torch
    from .stage2_u_matched import rng_devices
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    states = [(module, module.training) for module in model.modules()]
    devices = rng_devices(model)
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(int(seed))
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        for module, flag in states:
            module.training = flag


def soc_from_views(model, views, *, seed):
    """Four eval-with-grad B1 context encodings, no future input or new head.

    Context includes the arrived current t; the forecast t+1 is NOT scored.
    """
    import torch
    from hybrid_tsad.experimental_controls.carots_soc import soc_objective
    from .stage2_u_matched import evaluation_state
    x = np.asarray(views, dtype=np.float32)
    if x.shape != (4, model.config.history, 3):
        raise ValueError("Exactly four fixed H512 C3 histories required")
    device = next(model.parameters()).device
    with soc_randomness(model, seed):
        features = []
        for row in x:
            with evaluation_state(model, seed), torch.enable_grad():
                value = torch.from_numpy(row[None]).to(device)
                observed = torch.ones_like(value, dtype=torch.bool)
                # Common per-view RNG/masking; no batch-shared FRFT order change.
                result = model.forecast(value, observed, mask_seed=int(seed))
                features.append(forecast_features(result))
        embedding = torch.cat(features, dim=0)
        loss, meta = soc_objective(embedding, feature_identity=FEATURE_ID, enabled=True,
                                  similarity_threshold=SOC_SIMILARITY,
                                  temperature=SOC_TEMPERATURE)
    meta.update(extra_forecast_calls=4, forecast_batch_size=1,
                input="arrived_context_t-H+1_to_t; forecast_t+1_unused",
                augmentation="oracle_joint_GP_mixture_two_independent_driver_replace_GN")
    return loss, meta, embedding


def fit_representation_calibration(features, *, quantile=.99):
    """Same named cosine-to-normal-centroid readout for both trained arms."""
    if quantile != .99:
        raise ValueError("Fixed normal-CAL q99 only")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty normal CAL feature matrix required")
    norms = np.linalg.norm(x, axis=1)
    if (norms <= 0).any():
        raise ValueError("Zero CAL feature norm")
    centroid = (x/norms[:, None]).mean(0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("Invalid normal CAL centroid")
    centroid /= norm
    scores = .5*(1-(x/norms[:, None])@centroid)
    # Same strict > inverse-CDF q99 rule as original residual CAL.
    ordered = np.sort(scores, kind="stable")
    index = min(int(np.ceil(len(x)*quantile))-1, len(x)-1)
    return dict(feature_identity=FEATURE_ID, feature_dimension=x.shape[1],
                centroid=centroid.tolist(), threshold=float(ordered[index]),
                valid_windows=len(x), quantile=float(quantile),
                score="half_one_minus_cosine_to_mean_unit_normal_CAL_feature",
                threshold_rule="inverse empirical CDF q99; strict score>threshold",
                no_query_fitting=True)


def representation_scores(features, calibration):
    x = np.asarray(features, dtype=np.float64)
    if (calibration["feature_identity"] != FEATURE_ID or x.ndim != 2 or
            x.shape[1] != calibration["feature_dimension"] or not np.isfinite(x).all()):
        raise ValueError("Fixed layout and finite representation scores required")
    norm = np.linalg.norm(x, axis=1)
    if (norm <= 0).any():
        raise ValueError("Zero query feature norm")
    result = .5*(1-(x/norm[:, None])@np.asarray(calibration["centroid"], dtype=np.float64))
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite representation score")
    return result


def _gradient_evidence(loss, parameters):
    import torch
    gradients = torch.autograd.grad(loss, [p for _, p in parameters],
                                    retain_graph=True, allow_unused=True)
    groups = {}
    for (name, _), gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("Nonfinite objective-specific gradient")
        key = name.split('.')[0]
        groups[key] = groups.get(key, 0.) + float(gradient.detach().double().square().sum())
    return dict(norm=float(np.sqrt(sum(groups.values()))),
                group_norms={k: float(np.sqrt(v)) for k, v in groups.items()})


def train_soc_arm(source, fit_runs, select_runs, plan, augmentation, *, artifact_dir,
                  technical=False):
    """Only new training arm: original LAG_U SSL + fixed .1 published SOC."""
    import torch
    from . import r2_learnability as old
    from .stage2_u_matched import resolve_u_plan, validate
    from .stage2_adaptation import state_identity
    from .training import passes, tensor_batch
    from .data import window
    from .objectives import ssl_objective
    out = Path(artifact_dir);out.mkdir(parents=True, exist_ok=False)
    count = 2 if technical else 1000
    every = 2 if technical else 200
    if (plan['history'], plan['horizon'], plan['seed'], len(plan['FIT'])) != (512, 32, 2025, count):
        raise ValueError("Original fixed matched FIT plan required")
    if set(augmentation) != {'PARTNER', 'DRIVER_A', 'DRIVER_B'}:
        raise ValueError("Exactly three role-isolated TRAIN_AUGMENT arrays required")
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in augmentation.items()}
    if any(v.shape != (2560, 3) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Three full finite TRAIN_AUGMENT runs required")
    model, names = old.clone_for_arm(source, 'LAG_U')
    before = state_identity(model);source_before = state_identity(source)
    torch.save(model.state_dict(), out/'initial.pt');old.dump(out/'STATE_BEFORE.json', before)
    old.dump(out/'ARM_CONFIG.json', dict(arm='CAROTS_SOC_LAG_U', base_arm='LAG_U',
        model_config=model.config.to_dict(), trainable_prefixes=old.prefixes('LAG_U'),
        feature_identity=FEATURE_ID, SOC=dict(weight=SOC_WEIGHT, temperature=SOC_TEMPERATURE,
        similarity=SOC_SIMILARITY), same_updates_not_same_compute=True,
        candidate_cross_policy=False, no_projection_head=True))
    points = resolve_u_plan(fit_runs, plan['FIT'], 512, 32, 'FIT')
    resolve_u_plan(select_runs, plan['SELECT'], 512, 32, 'SELECT')
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trace = [];path = out/'training_trace.jsonl';path.open('x').close();began=time.monotonic()
    for attempt, (run, t) in enumerate(points):
        old.training_mode(model, 'LAG_U');opt.zero_grad(set_to_none=True)
        batch = tensor_batch(window(run, t, 512, 32), next(model.parameters()).device)
        old.append(path, dict(attempt=attempt, status='ATTEMPT_STARTED', index=batch['index'], footprint=batch['footprint']))
        # Exact old SSL path, mask seed and targets, before additional work.
        f, r = passes(model, batch, 2025+attempt)
        obj = ssl_objective(f, r, batch, 'huber', ['r_tf', 'r_ms', 'r_pred'])
        row = dict(attempt=attempt, index=batch['index'], footprint=batch['footprint'],
                   mask_seed=2025+attempt, updated=False, learning_rate=1e-4)
        if bool(obj['valid_windows'].all()):
            # Arrived context [t-511,t], never future targets or a forecast-t input.
            sl = slice(t-511, t+1)
            views = relation_views(batch['reconstruction_context'][0].detach().cpu().numpy(),
                arrays['PARTNER'][sl], np.stack((arrays['DRIVER_A'][sl, 0], arrays['DRIVER_B'][sl, 0])))
            from .readout import _rng, _rng_json
            rng_before=_rng_json(_rng())
            modes_before=[m.training for m in model.modules()]
            soc, meta, _ = soc_from_views(model, views, seed=731000+attempt)
            if rng_before!=_rng_json(_rng()) or modes_before!=[m.training for m in model.modules()]:
                raise ValueError('SOC work consumed original SSL RNG or changed modes')
            ssl_grad = _gradient_evidence(obj['total'], parameters)
            soc_grad = _gradient_evidence(soc, parameters)
            total = obj['total']+SOC_WEIGHT*soc
            if not bool(torch.isfinite(total)):
                raise FloatingPointError('Nonfinite matched SSL+SOC objective')
            total.backward()
            lag_grads = {n: float(p.grad.detach().norm()) if p.grad is not None else None
                         for n, p in model.named_parameters() if n.startswith('lag.')}
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            row.update(updated=True, loss=float(total.detach()), SSL_loss=float(obj['total'].detach()),
                SOC_loss=float(soc.detach()), weighted_SOC_loss=float(SOC_WEIGHT*soc.detach()),
                SSL_gradient=ssl_grad, SOC_gradient=soc_grad,
                weighted_SOC_gradient_norm=SOC_WEIGHT*soc_grad['norm'],
                SOC_RNG_restored=True,SOC_module_modes_restored=True,
                gradient_norm=float(norm), SOC_metadata=meta, lag_gradient_norms=lag_grads,
                parts={k: float(v.detach()) for k, v in obj['parts'].items()})
        else:
            row['reason']='NO_VALID_GT_NO_REDRAW'
        if (attempt+1)%every == 0 or attempt == count-1:
            row['SELECT_diagnostic'] = validate(model, select_runs, plan['SELECT'], 512, 32,
                                               2025, ['r_tf', 'r_ms', 'r_pred'])
        trace.append(row);old.append(path, dict(row, status='ATTEMPT_COMPLETED'))
    after = state_identity(model);delta = old.state_change(before, after, 'LAG_U')
    if not delta['passed'] or state_identity(source) != source_before:
        raise ValueError('Frozen/source state changed')
    buffers = set(dict(model.named_buffers()))
    if buffers.intersection(delta['changed']):
        raise ValueError('Unexpected frozen buffer update')
    actual_modes = {name: child.training for name, child in model.named_children()}
    if any(flag != (name in old.prefixes('LAG_U')) for name, flag in actual_modes.items()):
        raise ValueError('SOC failed to restore original heterogeneous module modes')
    torch.save(model.state_dict(), out/'adapted.pt')
    torch.save(dict(state_dict=model.state_dict(), optimizer=opt.state_dict(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
        resume_implemented=False), out/'final_training_state.pt')
    exposure = old.lag_update_evidence(trace, delta['changed'], 'LAG_U')
    report = dict(arm='CAROTS_SOC_LAG_U', attempts=count, updates=sum(x['updated'] for x in trace),
        endpoint_attempt=count-1, selection='FIXED_ENDPOINT_NO_BEST', trace=trace,
        state_change=delta, lag_update_evidence=exposure, actual_child_modes=actual_modes,
        SOC_nonzero_gradient_steps=sum(x.get('SOC_gradient', {}).get('norm', 0) > 0 for x in trace),
        SOC_nonself_positive_pairs=sum(x.get('SOC_metadata', {}).get('valid_nonself_positive_pairs', 0) for x in trace),
        source_unchanged=True, seconds=time.monotonic()-began,
        technical_discard_only=technical, updated_names=names, extra_forecast_calls=4*count)
    old.dump(out/'STATE_AFTER.json', after);old.dump(out/'training.json', report)
    old.dump(out/'RECEIPT.json', dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT', arm='CAROTS_SOC_LAG_U',
        attempts=count, updates=report['updates'], lag_policy_verified=bool(delta['lag_changed']),
        lag_update_evidence=exposure, feature_identity=FEATURE_ID,
        scientific_status='VALID_FINITE_ENDPOINT' if report['updates']==count and exposure['update_exposure_gate'] else 'INCOMPLETE',
        checkpoint_sha256=old.sha(out/'adapted.pt'), query_labels_read=False,
        technical_discard_only=technical, SOC_nonzero_gradient_steps=report['SOC_nonzero_gradient_steps']))
    return model, report, opt


def arrived_features(model, run, target, *, seed=2025):
    """Feature index t has arrived input [t-511,t]; prediction t+1 unused."""
    import torch
    from .data import arrived_window
    from .training import tensor_batch
    from .stage2_u_matched import evaluation_state
    batch = tensor_batch(arrived_window(run, target, 512, 32), next(model.parameters()).device)
    with soc_randomness(model, seed+target), evaluation_state(model, seed+target):
        forecast = model.forecast(batch['reconstruction_context'], batch['reconstruction_observed'],
                                  mask_seed=seed+target)
        value = forecast_features(forecast).detach().cpu().numpy()[0]
    return value


def calibrate_both(model, runs, plan, *, artifact_dir, residual_reuse=None, technical=False):
    from . import r2_learnability as old
    from .stage2_u_matched import resolve_u_plan, calibrate
    from .stage2_adaptation import state_identity
    out=Path(artifact_dir);out.mkdir(parents=True,exist_ok=False)
    before=state_identity(model);old.dump(out/'MODEL_STATE_BEFORE.json',before)
    if len(plan) != (2 if technical else 512):
        raise ValueError('Fixed full512 CAL plan, or explicit technical2, required')
    points=resolve_u_plan(runs,plan,512,32,'CAL',arrived=True)
    if residual_reuse is None:
        rec, residual=calibrate(model,runs,plan,history=512,horizon=32,seed=2025,
                               active_components=['r_tf','r_ms','r_pred'])
        old.dump(out/'cal_records.json',rec)
    else:
        residual=json.loads(Path(residual_reuse).read_text())
        old.dump(out/'RESIDUAL_REUSE.json',dict(path=str(residual_reuse),sha256=old.sha(residual_reuse),
            reused_not_recomputed=True,scope='Original LAG_U same normalCAL/source identity verified by worker'))
    old.dump(out/'calibration.json',residual)
    features=np.stack([arrived_features(model,run,t) for run,t in points])
    representation=fit_representation_calibration(features)
    old.dump(out/'representation_calibration.json',representation)
    with (out/'CAL_FEATURES.npz').open('xb') as f:
        np.savez_compressed(f,index=np.array([r.start+t for r,t in points]),features=features,
            representation_score=representation_scores(features,representation))
    after=state_identity(model)
    if before!=after:raise ValueError('CAL mutated model')
    old.dump(out/'MODEL_STATE_AFTER.json',after)
    result=dict(status='COMPLETED_PENDING_INDEPENDENT_AUDIT',phase='calibrate',records=len(points),
        score_views=2,feature_identity=FEATURE_ID,calibration_sha256=old.sha(out/'calibration.json'),
        representation_calibration_sha256=old.sha(out/'representation_calibration.json'),
        source_unchanged=True,query_labels_read=False,technical_discard_only=technical)
    old.dump(out/'RECEIPT.json',result)
    return result


def write_dual_bank(model, run, residual, representation, directory, *, identity, technical=False):
    """Existing full residual forward plus one causal arrived-context feature."""
    from .chunked_bank import ChunkedBank, iter_pointwise_infer
    from .cli import score_records
    from .stage2_adaptation import state_identity
    from . import r2_learnability as old
    before=state_identity(model);model.eval()
    if not technical and (len(run.values)!=2560 or run.run not in CASES):
        raise ValueError('Only the seven fixed query fixtures allowed')
    ranges=[dict(domain=run.domain,entity=run.entity,run=run.run,start=run.start+512,end=run.start+len(run.values))]
    count=0;started=time.monotonic()
    with ChunkedBank(directory,identity=identity,ranges=ranges,
                     threshold=residual['threshold'],chunk_size=64) as bank:
        if bank.manifest['committed_records']:
            raise PermissionError('No automatic repeated query/resume')
        for record in iter_pointwise_infer(model,[run],history=512,horizon=32,seed=2025):
            t=record['index']-run.start
            feature=arrived_features(model,run,t)
            score=float(representation_scores(feature[None],representation)[0])
            record=score_records([record],residual)[0]
            record['representation']=dict(feature=feature,score=score,
                alarm=bool(score>representation['threshold']),threshold=representation['threshold'],
                feature_identity=FEATURE_ID,input_bounds=[record['index']-511,record['index']+1],
                available_index=record['index'],forecast_target_unused=record['index']+1,
                score_view='B_NORMAL_CAL_COSINE',mask_seed=2025+t)
            bank.append(record);count+=1
        bank.finalize()
    if state_identity(model)!=before:raise ValueError('QUERY mutated model')
    expected=len(run.values)-512
    if count!=expected:raise ValueError('Incomplete original query index axis')
    return dict(case=run.run,path=str(Path(directory)/'manifest.json'),sha256=old.sha(Path(directory)/'manifest.json'),
                points=count,seconds=time.monotonic()-started,score_views=2,
                query_labels_read=False,feature_identity=FEATURE_ID)
