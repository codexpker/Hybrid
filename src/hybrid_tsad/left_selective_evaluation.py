"""Fixed SWaT B/C loss-retention update evaluation, no model training.

Historical ordinary-U metrics are hash-reused. Only the four new score banks
are evaluated. Donor annotations are joined after the complete terminal seal;
retention diagnostics never feed a learner or select checkpoints/thresholds.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from . import timercd_swat_evaluation as swat
from . import timeradar_cross_variable_evaluation as io
from .timeradar_smd_full_evaluation import read, local, checked
from .engine.events import binary_events

ARMS = ('B_trim', 'B_random', 'C_trim', 'C_random')
PLAN_SHA = '444e61197b6d4951853af6d6396c087d4decc2bc791f25ce52fce0b5ed71e0e6'
OLD_C_LOSS_SHA = '4dd985d0031ef52d19e72ae35749e1f0f7a776378f392e1d312329090c4f82b4'
TIMERCd_TABLE_SHA = '5b6f1577bc6285bba2dcbdf87668b8442d7d02ebe8b9072e10693ba98d6f850b'


def terminal_gate(seal):
    expected = {f'{arm}/score/{role}.npz' for arm in ARMS for role in ('CAL', 'QUERY')}
    actual = [e['relative'].replace('\\', '/') for e in seal['banks']]
    if (seal['status'] != 'COMPLETED' or not seal['no_labels_decoded']
            or len(actual) != len(set(actual)) or set(actual) != expected):
        raise ValueError('All four arms and eight CAL/QUERY banks must seal before labels open')
    return {e['relative'].replace('\\', '/'): e for e in seal['banks']}


def native_metrics(index, score, labels, tau, available):
    if not np.array_equal(index, available):
        raise PermissionError('Original native LEFT endpoint scorer must not use future observations')
    rows, details, alarms, coverage = swat.evaluate_score(index, score, labels, tau, available)
    for row in rows:
        row.update(mode='CAUSAL_ENDPOINT', delay_hit_only_median=row['offline_first_hit_offset_median'],
                   delay_definition='first available endpoint alarm inside original event; sample positions')
    for row in details:
        row['delay_points'] = row['offline_availability_offset']
        row['timing_status'] = 'CAUSAL_ENDPOINT_SAMPLE_OFFSET_NO_VERIFIED_PHYSICAL_CLOCK'
    return rows, details, alarms, coverage


def inherited(workspace):
    w = Path(workspace)
    table = checked(w / swat.OLD / 'fixed_evaluation/method_comparison.csv', swat.BASELINE_SHA)
    sp = checked(w / swat.OLD / 'fixed_scores/ALL_SCORES_THRESHOLDS_MASKS_SEALED.json', swat.BASELINE_SEAL_SHA)
    seal = read(sp); accepted = swat.parse_csv(table); rows = []
    inputs = {str(table): swat.BASELINE_SHA, str(sp): swat.BASELINE_SEAL_SHA}
    for old_name, name in (('noU', 'noU'), ('U_A', 'U_A'), ('U_B', 'U_B'), ('U_C', 'U_C'),
                            ('HistoryMean512', 'HistoryMean512')):
        desc = next(e for e in seal['descriptors'] if e['method'] == old_name)
        p = checked(w / swat.OLD / 'fixed_scores' / (old_name + '.npz'), desc['sha256'])
        inputs[str(p)] = desc['sha256']
        with np.load(p, allow_pickle=False) as z:
            assert np.array_equal(z['index'], np.arange(swat.QUERY_START, swat.QUERY_END))
            assert np.array_equal(z['CAL_index'], np.arange(swat.CAL_START, swat.CAL_END))
            tau, chosen = swat.threshold(z['CAL_index'], z['CAL_score'])
            assert np.array_equal(z['selected_CAL'], chosen)
            assert tau == desc['thresholds']['own_CAL_q99']
        chosen_rows = [dict(r) for r in accepted if r['method'] == old_name and r['track'] == 'own_CAL_q99']
        if {r['stream'] for r in chosen_rows} != set(swat.STREAMS):
            raise ValueError('Historical methods require both exact SWaT metric streams')
        for row in chosen_rows:
            if row['threshold'] != tau or row['Ntrue'] != 22 or row['long_event_count'] != 9:
                raise ValueError('Historical role/CAL/event protocol differs')
            row.update(method=name, status='REUSED_ACCEPTED_RESULT', metric_origin='HASH_REUSED_ABC2000_CSV',
                event_point_coverage=row['Recall'], mode='CAUSAL_ENDPOINT',
                calibration_permission='Original512 CAL threshold only' if name != 'HistoryMean512'
                    else 'Original512 CAL residual median scale and threshold; extra CAL fitting')
            rows.append(row)
    for method, reason in (
        ('TimeRadar', 'No qualified score bank with this exact SWaT CAL and QUERY role identity; no new inference'),
        ('TSAD-C', 'Pinned official compatibility check stopped with implementation blockers; no score claimed')):
        for stream in swat.STREAMS:
            rows.append(dict(method=method, stream=stream, status='NA_NOT_COMPATIBLE',
                             AP=None, ROC_AUC=None, reason=reason))
    tp = checked(w / 'hybrid_ad/docs/experiments/timercd_reference_swat_20261118/comparison.csv', TIMERCd_TABLE_SHA)
    inputs[str(tp)] = TIMERCd_TABLE_SHA
    timercd = [dict(r, external_scope='Reused frozen source-only TimeRCD; offline5000 context, different model/training permission')
               for r in swat.parse_csv(tp) if r['model_id'] in swat.MODEL_IDS]
    if len(timercd) != 14:
        raise ValueError('Preserve the complete seven-model historical TimeRCD comparison')
    return rows, timercd, inputs


def contrasts(rows):
    lookup = {(r['method'], r['stream']): r for r in rows}
    comparisons = []
    for condition in ('B', 'C'):
        comparisons += [(condition + '_trim', 'U_' + condition, condition + '_trim-minus-ordinary'),
                        (condition + '_trim', condition + '_random', condition + '_trim-minus-random')]
    comparisons += [('U_B', 'U_C', 'ordinary_B-minus-C'),
                    ('B_trim', 'C_trim', 'trim_B-minus-C'),
                    ('B_random', 'C_random', 'random_B-minus-C')]
    rows_out = []
    for stream in swat.STREAMS:
        for a, b, name in comparisons:
            row = dict(comparison=name, candidate=a, reference=b, stream=stream,
                       scope='Single paired training experiment; B/C are conditions, not repeated instances')
            for metric in swat.METRICS:
                va, vb = lookup[a, stream].get(metric), lookup[b, stream].get(metric)
                row[metric + '_delta'] = (float(va-vb) if va is not None and vb is not None
                                         and np.isfinite(va) and np.isfinite(vb) else None)
            rows_out.append(row)
        for reference in ('ordinary', 'random'):
            a = next(r for r in rows_out if r['stream'] == stream and r['comparison'] == 'trim_B-minus-C')
            b = next(r for r in rows_out if r['stream'] == stream and r['comparison'] == reference + '_B-minus-C')
            row = dict(comparison='B-C_trim_difference-minus-' + reference, stream=stream,
                       scope='Descriptive difference-in-differences on the same fixed QUERY; no independent-run inference')
            for metric in swat.METRICS:
                va, vb = a[metric + '_delta'], b[metric + '_delta']
                row[metric + '_delta'] = float(va-vb) if va is not None and vb is not None else None
            rows_out.append(row)
    return rows_out


def hard_nominal_mask(loss, count=25):
    """Audit-only old U-C reference, exact ties favor earlier original slot."""
    if loss.shape != (2000, 128) or not np.isfinite(loss).all() or count != 25:
        raise ValueError('Fixed ordinary-U-C high-loss audit definition')
    order = np.argsort(-loss, axis=1, kind='stable')[:, :count]
    mask = np.zeros(loss.shape, bool); np.put_along_axis(mask, order, True, axis=1)
    return mask


def validate_retained(indices, keep_mask):
    if (indices.shape != (2000, 103) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= 128)
            or keep_mask.shape != (2000, 128) or keep_mask.dtype != np.bool_):
        raise ValueError('Retained slots must preserve the fixed original batch identities')
    rebuilt = np.zeros(keep_mask.shape, bool); np.put_along_axis(rebuilt, indices, True, axis=1)
    if not np.array_equal(rebuilt, keep_mask) or not np.all(keep_mask.sum(1) == 103):
        raise ValueError('Duplicate, mismatched or missing retained slots')


def retention_analysis(workspace, root, labels, receipts):
    w, root = Path(workspace), Path(root)
    pp = checked(w / swat.PREP / 'LEARNER_PLAN.npz', PLAN_SHA)
    lp = checked(w / swat.OLD / 'U/C/train/PER_SAMPLE_NATIVE_LOSS.npz', OLD_C_LOSS_SHA)
    inputs = {str(pp): PLAN_SHA, str(lp): OLD_C_LOSS_SHA}
    with np.load(pp, allow_pickle=False) as z:
        plan = {k:z[k].copy() for k in z.files}
    with np.load(lp, allow_pickle=False) as z:
        hard = hard_nominal_mask(z['total'])
    replacement = plan['replacement_mask']
    if replacement.shape != (2000, 128) or replacement.dtype != np.bool_ or replacement.sum() != 25600:
        raise ValueError('Keep the original replacement schedule and actual exposure count')
    # This annotation join occurs only after all four trainings and score banks
    # have sealed. It never reaches the retained-slot selector or model input.
    donor_lo, donor_hi = 495000, 607479
    events = [(a, b) for a, b in binary_events(labels) if donor_lo <= a and b <= donor_hi]
    if len(events) != 13:
        raise ValueError('Retain all13 original independent attack-event donors')
    ends = np.array([b for a, b in events]); starts = np.array([a for a, b in events])
    bindex = plan['B_donor_starts'] + donor_lo
    cindex = plan['C_donor_starts'] + donor_lo
    prefix = np.concatenate(([0], np.cumsum(labels, dtype=np.int64)))
    for positions, positive in ((bindex, 192), (cindex, 0)):
        p = positions[replacement]
        if np.any(p < donor_lo) or np.any(p + 192 > donor_hi) or not np.all(prefix[p+192]-prefix[p] == positive):
            raise ValueError('Original B attack/C nominal-normal windows must not change or use QUERY donors')
    which = np.full(replacement.shape, -1, dtype=np.int64)
    assigned = np.searchsorted(ends, bindex[replacement] + 192, side='left')
    if np.any(assigned >= len(events)) or not np.all(starts[assigned] <= bindex[replacement]):
        raise ValueError('A donor window must lie wholly within one independent event')
    which[replacement] = assigned
    groups, donor_rows, dose_rows = [], [], []
    methods = list(ARMS) + ['U_B', 'U_C']
    for method in methods:
        condition = method[0] if method in ARMS else method[-1]
        if method in ARMS:
            folder = root / method / 'train'; receipt = receipts[method]
            retain_path = checked(folder / 'RETAINED_SLOTS.npz', receipt['RETAINED_SLOTS_sha256'])
            loss_path = checked(folder / 'PER_SAMPLE_NATIVE_LOSS.npz', receipt['PER_SAMPLE_NATIVE_LOSS_sha256'])
            with np.load(retain_path, allow_pickle=False) as z:
                indices, keep, update = z['indices'].copy(), z['keep_mask'].copy(), z['update'].copy()
            validate_retained(indices, keep)
            if not np.array_equal(update, np.arange(1, 2001)):
                raise ValueError('Retained identities must cover every actual update')
            with np.load(loss_path, allow_pickle=False) as z:
                loss = z['total'].copy()
            inputs[str(retain_path)] = io.sha(retain_path); inputs[str(loss_path)] = io.sha(loss_path)
        else:
            keep = np.ones((2000, 128), bool)
            loss_path = w / swat.OLD / 'U' / condition / 'train/PER_SAMPLE_NATIVE_LOSS.npz'
            inputs[str(loss_path)] = io.sha(loss_path)
            with np.load(loss_path, allow_pickle=False) as z:
                loss = z['total'].copy()
        if loss.shape != keep.shape or not np.isfinite(loss).all():
            raise ValueError('Finite per-original-slot native losses required')
        nominal = ~replacement if condition == 'B' else np.ones_like(replacement)
        definitions = [('all_forward_slots', np.ones_like(replacement)),
            ('background_nominal_slots', ~replacement),
            ('replacement_attack_donor' if condition == 'B' else 'replacement_nominal_normal_donor', replacement),
            ('high_loss_nominal_vs_ordinary_UC', hard & nominal),
            ('other_nominal_vs_ordinary_UC', ~hard & nominal),
            ('high_loss_background_vs_ordinary_UC', hard & ~replacement)]
        if condition == 'C':
            definitions.append(('high_loss_normal_donor_vs_ordinary_UC', hard & replacement))
        for name, mask in definitions:
            offered, retained = int(mask.sum()), int((keep & mask).sum())
            mass = float(loss[keep & mask].astype(np.float64).sum())
            total_mass = float(loss[keep].astype(np.float64).sum())
            groups.append(dict(method=method, condition=condition, group=name, offered_slots=offered,
                retained_loss_slots=retained, retained_fraction=retained/offered if offered else None,
                retained_loss_mass=mass, retained_loss_mass_share=mass/total_mass if total_mass else None,
                loss_mass_is_not_gradient_share=True,
                group_semantics='Audit annotations only; high loss is not contamination; nominal is not certified healthy'))
        if condition == 'B':
            for event_id, (a, b) in enumerate(events):
                mask = replacement & (which == event_id)
                donor_rows.append(dict(method=method, event_id=f'D{event_id+1:02d}', start=a, end_exclusive=b,
                    event_points=b-a, offered_window_presentations=int(mask.sum()),
                    retained_window_presentations=int((keep & mask).sum()),
                    retained_fraction=float((keep & mask).sum()/mask.sum()),
                    independence='One original event; repeated overlapping windows are not independent events'))
        attack = replacement if condition == 'B' else np.zeros_like(replacement)
        retained_attack = int((keep & attack).sum()); retained = int(keep.sum())
        dose_rows.append(dict(method=method, condition=condition, offered_forward_windows=keep.size,
            replacement_windows=int(replacement.sum()), replacement_window_fraction=float(replacement.mean()),
            retained_gradient_windows=retained, retained_gradient_fraction=retained/keep.size,
            retained_attack_label_windows=retained_attack,
            offered_attack_time_label_points=int(attack.sum())*192,
            offered_time_points=keep.size*192,
            retained_attack_time_label_points=retained_attack*192,
            retained_target_time_points=retained*192,
            retained_attack_time_label_support_fraction=retained_attack/retained,
            physical_affected_scalar_fraction=None, physical_affected_scalar_reason='Attack-time labels do not identify actually affected channels',
            all_forward_and_native_prototype_windows=keep.size,
            discarded_loss_slots_still_participate_in_forward_and_prototype_updates=True))
    return groups, donor_rows, dose_rows, inputs


def run(workspace, root, output, config):
    start = time.monotonic(); w = Path(workspace).resolve(); root = Path(root).resolve(); output = Path(output).resolve()
    cp = local(w, config); cfg = read(cp); sp = root / 'RAW_BANK_SEAL.json'; seal = read(sp)
    if (cfg['experiment'] != root.name or tuple(cfg['arms']) != ARMS or cfg['learner_plan_sha256'] != PLAN_SHA
            or cfg['updates'] != 2000 or cfg['batch_size'] != 128 or cfg['retained'] != 103):
        raise PermissionError('Keep the fixed four arms, original draws and 103-of128 loss budget')
    descriptors = terminal_gate(seal)
    for relative, description in descriptors.items():
        checked(root / relative, description['sha256'])
    inputs = {str(cp): io.sha(cp), str(sp): io.sha(sp)}; receipts = {}
    for arm in ARMS:
        p = root / arm / 'train/RECEIPT.json'; train = read(p)
        s = root / arm / 'score/RECEIPT.json'; scored = read(s)
        if (train['status'] != 'COMPLETED' or train['arm'] != arm or train['updates'] != 2000
                or train['plan_sha256'] != PLAN_SHA or train['labels_read']
                or train['full_forward_batch'] != 128 or train['retained_per_update'] != 103
                or not train['no_prefilter_before_forward']
                or scored['status'] != 'COMPLETED' or scored['arm'] != arm or scored['labels_read']
                or scored['checkpoint_sha256'] != train['checkpoint_sha256']
                or scored['python_step'] != train['final_python_step']
                or not scored['eval_state_modes_RNG_immutable']):
            raise PermissionError('Training budget / original plan / endpoint2000 / immutable scoring mismatch')
        checked(root / arm / 'train/endpoint2000.pth', train['checkpoint_sha256'])
        initial = read(root / arm / 'train/INITIAL_IDENTITY.json')
        if initial['checkpoint_sha256'] != cfg['initial_checkpoint_sha256'] or initial['plan_sha256'] != PLAN_SHA:
            raise PermissionError('Do not substitute a different initial model or training schedule')
        inputs[str(p)] = io.sha(p); inputs[str(s)] = io.sha(s); receipts[arm] = train
    inherited_rows, timercd, inherited_inputs = inherited(w); inputs.update(inherited_inputs)
    # The global terminal seal closes both model training and score production.
    # Only now may labels annotate QUERY metrics and donor retention diagnostics.
    lp = checked(w / swat.PREP / 'SWAT_LABEL_METADATA.npz', swat.LABEL_SHA); inputs[str(lp)] = swat.LABEL_SHA
    with np.load(lp, allow_pickle=False) as z:
        labels = z['label'].astype(np.int8)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'EVALUATOR_EXECUTED_SNAPSHOT.py').open('xb') as handle:
        handle.write(Path(__file__).read_bytes())
    new_rows, events, alarms, costs = [], [], [], []
    coverage = None
    for arm in ARMS:
        values = {}
        for role, lo, hi in (('CAL', swat.CAL_START, swat.CAL_END), ('QUERY', swat.QUERY_START, swat.QUERY_END)):
            relative = f'{arm}/score/{role}.npz'; p = root / relative
            with np.load(p, allow_pickle=False) as z:
                index, score, available = z['index'].copy(), z['score'].copy(), z['available_index'].copy()
                valid, channel = z['valid'].copy(), z['per_channel_last'].copy()
            if (not np.array_equal(index, np.arange(lo, hi)) or not np.array_equal(index, available)
                    or score.shape != index.shape or not np.isfinite(score).all()
                    or valid.dtype != np.bool_ or valid.shape != index.shape or not valid.all()
                    or channel.shape != (len(index), 31) or channel.dtype != np.float32
                    or not np.array_equal(score, channel.mean(-1))):
                raise ValueError('Preserve native scalar/channel mean, causal endpoint, original real axis and valid mask')
            values[role] = (index, score, available)
        tau, selected = swat.threshold(values['CAL'][0], values['CAL'][1])
        index, score, available = values['QUERY']
        result, event_rows, alarm_rows, coverage = native_metrics(index, score, labels, tau, available)
        metadata = dict(method=arm, condition=arm[0], selection=arm.split('_')[1], status='OK',
            metric_origin='NEW_FIXED_SELECTIVE_UPDATE_BANK', track='own_CAL_q99',
            checkpoint_rule='endpoint2000_only', checkpoint_sha256=receipts[arm]['checkpoint_sha256'],
            full_forward_batch=128, retained_gradient_slots=103,
            calibration_permission='Same original512 CAL threshold only',
            no_labels_used_in_training_or_selection=True)
        new_rows.extend(dict(**r, **metadata) for r in result)
        events.extend(dict(**r, **metadata) for r in event_rows)
        alarms.extend(dict(**r, **metadata) for r in alarm_rows)
        folder = output / arm; folder.mkdir()
        cal_pos = np.searchsorted(values['CAL'][0], selected)
        np.savez_compressed(folder / 'calculation_inputs.npz', index=index, score=score,
            available_index=available, full_labels=labels, selected_CAL_index=selected,
            selected_CAL_score=values['CAL'][1][cal_pos])
        io._dump(folder / 'metrics.json', dict(rows=[dict(**r, **metadata) for r in result],
            inputs={str(root / f'{arm}/score/{role}.npz'):descriptors[f'{arm}/score/{role}.npz']['sha256']
                    for role in ('CAL', 'QUERY')}, source_sha256=io.sha(__file__)))
        per = dict(method=arm, condition=arm[0], selection=arm.split('_')[1],
                   updates=2000, full_forward_windows=256000, retained_gradient_windows=206000,
                   keep_per_update=103, retained_fraction=103/128, checkpoint_rule='endpoint2000_only')
        for phase in ('train', 'score'):
            jobs = [j for j in seal['jobs'] if j.get('arm') == arm and j['phase'] == phase]
            if len(jobs) != 1 or jobs[0]['OS_exit_code'] != 0:
                raise ValueError('Keep every actual training/scoring lifecycle and exit status')
            per[phase + '_GPU_hours'] = jobs[0]['actual_GPU_seconds'] / 3600
            peak = read(root / arm / phase / 'GPU_PEAK.json')
            per[phase + '_peak_allocated_bytes'] = peak['allocated']
        costs.append(per)
    rows = new_rows + inherited_rows
    groups, donors, doses, audit_inputs = retention_analysis(w, root, labels, receipts); inputs.update(audit_inputs)
    tables = dict(comparison=rows, paired_contrasts=contrasts(rows),
                  per_event_complete_query=[e for e in events if e['main_eligible']],
                  retention_groups=groups, retained_donor_events=donors, supervision_exposure=doses,
                  parameters_and_compute=costs, frozen_TimeRCD_external=timercd)
    for name, entries in tables.items():
        io._write_csv(output / (name + '.csv'), entries)
    io._write_csv(output / 'event_coverage_all_original.csv', coverage)
    io._write_csv(output / 'raw_event_details_all_original.csv', events)
    io._write_csv(output / 'alarm_details.csv', alarms)
    io._dump(output / 'RESULT.json', dict(status='COMPLETED', tables=tables, inputs=inputs,
        CPU_seconds=time.monotonic()-start, GPU_hours=0, actual_GPU_hours=seal['actual_GPU_hours'],
        new_forward=False, newly_evaluated_methods=4, newly_evaluated_stream_rows=8,
        source_sha256=io.sha(__file__), metric_source_sha256=io.sha(swat.__file__),
        legacy_event_kernel_sha256=io.sha(swat.legacy.__file__),
        main_scope='One fixed source/checkpoint/seed2000-update paired experiment. B/C are training conditions, '
                   'not independent repetitions; same22 query events, nine long>512/steady+512',
        retention_scope='Only the103-window gradient loss subset is filtered; all128 still drive forward and native prototype updates. '
                        'High-loss ordinary-U-C nominal slots are an audit proxy, not pollution or authenticated difficult-normal truth.',
        historical_metrics_recomputed=False, new_QUERY_or_donor_labels_used_for_selection=False))
    return tables


def publish(output, public):
    """Publish only derived own-CAL results and post-training audit statistics."""
    output, public = Path(output), Path(public); result = read(output / 'RESULT.json')
    if result['status'] != 'COMPLETED':
        raise ValueError('Complete four-arm evaluation required')
    public.mkdir(parents=True, exist_ok=True)
    names = tuple(result['tables'])
    if any((public / (name + '.csv')).exists() for name in names) or (public / 'RESULTS.md').exists():
        raise FileExistsError('Do not overwrite published scientific results')
    exported = []
    for name in names:
        source = output / (name + '.csv'); text = source.read_text(encoding='utf-8')
        if any(k in text for k in ('/home/', '192.168.', 'D:\\', 'C:\\', 'ghp_', 'token=')):
            raise PermissionError('Potentially private export: ' + name)
        target = public / source.name; target.write_text(text, encoding='utf-8', newline='\n')
        exported.append(dict(name=source.name, raw_sha256=io.sha(source), public_sha256=io.sha(target)))
    tables = result['tables']; lookup = {(r['method'], r['stream']):r for r in tables['comparison']}
    def f(value, percent=False):
        return 'NA' if value is None else f'{value * (100 if percent else 1):.6f}'
    order = ('U_B', 'B_random', 'B_trim', 'U_C', 'C_random', 'C_trim', 'noU', 'U_A',
             'HistoryMean512', 'TimeRadar', 'TSAD-C')
    lines = ['# Fixed SWaT LEFT-native loss retention: B/C matched experiment', '',
        'Four new arms retain the complete native model, per-window reconstruction objectives, '
        'initial checkpoint, original training plan and native endpoint scorer. The sole update factor '
        'is whether103 of128 window losses are selected by lowest loss or a fixed random subset. '
        'The original full128 still drive forward computation and native prototype updates. No '
        'training label, donor event ID or QUERY label enters the loss selector.', '',
        '## Main own-CAL comparison', '',
        '| Method | AP | ROC | Strict point recall (%) | Strict normal FPR (%) | C3 normal FPR (%) | C3 any-hit | C3 one-to-one | '
        'C3 event F1 (%) | C3 anomaly point coverage (%) | Long9 steady macro (%) | Long9 steady pooled (%) | C3 longest miss |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for method in order:
        strict, c3 = lookup[method, 'strict'], lookup[method, 'confirmed3']
        values = [f(strict.get('AP')), f(strict.get('ROC_AUC')), f(strict.get('Recall'), True), f(strict.get('normal_FPR'), True),
            f(c3.get('normal_FPR'), True), str(c3.get('auxiliary_any_hit_events', 'NA')),
            str(c3.get('M', 'NA')), f(c3.get('one_to_one_eventF1'), True),
            f(c3.get('event_point_coverage'), True), f(c3.get('long_steady_macro_coverage'), True),
            f(c3.get('long_steady_point_coverage'), True), str(c3.get('max_event_miss_points', 'NA'))]
        lines.append('| ' + method + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'U_B/U_C are the corresponding ordinary full128-loss updates; B/C are two '
        'matched training conditions on one fixed QUERY, not independent repeats. U_A, noU and '
        'History Mean512 are historical context controls. Only the four new arms are evaluated '
        'here; accepted historical metrics are hash-reused without reevaluation.', '',
        '## Matched update differences', '',
        '| Contrast | AP difference | ROC difference | C3 FPR difference | C3 point coverage difference | Long9 macro difference |',
        '|---|---:|---:|---:|---:|---:|']
    for row in tables['paired_contrasts']:
        if row['stream'] == 'confirmed3':
            fields = ('AP_delta', 'ROC_AUC_delta', 'normal_FPR_delta', 'event_point_coverage_delta',
                      'long_steady_macro_coverage_delta')
            lines.append('| ' + row['comparison'] + ' | ' + ' | '.join(f(row.get(k)) for k in fields) + ' |')
    lines += ['', 'Differences use raw metric fractions. A positive false-alarm difference is a '
        'cost, not a gain. B-minus-C and difference-in-differences are descriptive condition contrasts '
        'with the same source/checkpoint and QUERY, not independent-run causal or significance evidence.', '',
        '## Post-training retention audit', '',
        '| Method | Replacement category | Offered donor slots | Donor loss-retained (%) | High-loss nominal proxy slots | Proxy loss-retained (%) |',
        '|---|---|---:|---:|---:|---:|']
    gl = {(r['method'], r['group']):r for r in tables['retention_groups']}
    for method in ('U_B', 'B_random', 'B_trim', 'U_C', 'C_random', 'C_trim'):
        name = 'replacement_attack_donor' if method in ('U_B', 'B_random', 'B_trim') else 'replacement_nominal_normal_donor'
        donor = gl[method, name]; nominal = gl[method, 'high_loss_nominal_vs_ordinary_UC']
        lines.append('| ' + method + ' | ' + name + f" | {donor['offered_slots']} | "
            + f(donor['retained_fraction'], True) + f" | {nominal['offered_slots']} | "
            + f(nominal['retained_fraction'], True) + ' |')
    lines += ['', 'The high-loss nominal proxy is fixed before reading the new outputs: the top25 '
        'of128 slots per update in the accepted ordinary-U-C native loss archive; exact ties favor '
        'earlier slot identity. For B, attack-replacement slots are excluded from the nominal group; '
        'for C, the paired nominal-normal donor slots are included. The full CSV separates background '
        'and normal-donor subsets. This is relative training-loss difficulty, not authenticated plant '
        'mode, health or contamination truth. Labels are joined only after all four trainings and all '
        'eight score banks seal, never returned to training.', '',
        'B attack donors contain13 independent original events, but their overlapping window '
        'presentations are repeated exposures, not extra events. C donors are label-normal gaps in '
        'the same earlier DONOR interval, not new FIT or certified healthy records. All nominal '
        'background descriptions retain this limitation.', '',
        'The offered replacement fraction is10%. The new gradient subset retains103/128=80.46875% '
        '(206000 of256000 presentations), not exactly80%. supervision_exposure.csv reports offered '
        'window replacement, time-label target support, retained positive-time support and the '
        'unknown physical affected-channel fraction separately. All full256000 windows still '
        'participate in forward/native prototype state: dropping their explicit loss is not proof '
        'that the training state was cleaned. Retained loss mass is not gradient share.', '',
        '## Frozen metric and external permissions', '',
        '- Every method uses its own original512 CAL points linspace(446173,494999,512,int64), '
        'q99/higher and strict greater-than. Only own_CAL_q99 is published. No fixed-noU threshold '
        'extension, QUERY calibration or score search is introduced.',
        '- QUERY stays608535:944919, with22 complete events and9 long events strictly>512; steady '
        'coverage starts at original event start+512. The original strict/C3, chronological one-to-one '
        'matching and normal-fragment definitions are reused. New native scores are actual causal '
        'endpoints; sample offsets are not converted into unverified physical time.',
        '- H512 has extra CAL fitting for per-channel median residual scales as well as its threshold. '
        'The complete accepted seven frozen TimeRCD rows are retained separately in '
        'frozen_TimeRCD_external.csv: source-only training and offline5000 context differ from these '
        'LEFT-native target-normal training/endpoint permissions. No cross-permission mechanism '
        'attribution follows. Compatible TimeRadar banks remain unavailable; TSAD-C remains NA '
        'under its completed official compatibility finding.', '',
        '## Completed resource use', '',
        f"Actual additional GPU time is {result['actual_GPU_hours']:.9f} GPUh, including the finite "
        'CUDA equivalence gate and four training/four scoring lifecycles. Each new arm has2000 '
        'updates and uses only endpoint2000; SELECT is diagnostic. Source weights, old U training, '
        'old scores and historical metrics are not rerun or overwritten. Raw arrays, model weights '
        'and private host paths are withheld from public exports.', '']
    (public / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    io._dump(public / 'PROVENANCE_PUBLIC.json', dict(
        raw_result_sha256=io.sha(output / 'RESULT.json'), executed_evaluator_source_sha256=result['source_sha256'],
        publication_source_sha256=io.sha(__file__), metric_source_sha256=result['metric_source_sha256'],
        accepted_ABC2000_comparison_sha256=swat.BASELINE_SHA, accepted_ABC2000_seal_sha256=swat.BASELINE_SEAL_SHA,
        learner_plan_sha256=PLAN_SHA, ordinary_UC_loss_audit_reference_sha256=OLD_C_LOSS_SHA,
        accepted_frozen_TimeRCD_table_sha256=TIMERCd_TABLE_SHA, label_metadata_sha256=swat.LABEL_SHA,
        exports=exported, withheld='Raw input/label arrays, weights, per-slot audit arrays, private paths and raw receipts.'))
    return public


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); run(a.workspace, a.root, a.output, a.config)
