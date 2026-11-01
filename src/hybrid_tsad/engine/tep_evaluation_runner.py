"""Hash-pinned metadata gate and future fixed TEP evaluation file adapter.

No real all-score seal or label release is bundled. Approval strings are not
signatures: an external root-provided trusted release digest is mandatory.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time

from . import tep_fixed_evaluation as core

OP = 'TEP_FIXED_SCORE_EVALUATION'
ENTITIES = ['entity_%04d' % i for i in range(1, 401)]
CORE_SHA = 'dce4b8f0cf93e57dba219861eae0cd49692b59f3ebd174ebd5f0e0357e280c2f'


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise PermissionError('Exact lowercase SHA256 required')
    return value


def _path(path, workspace):
    p = Path(path)
    root = Path(workspace).resolve(strict=True)
    if not p.is_absolute():
        raise PermissionError('Absolute workspace path required')
    p = p.resolve(strict=True)
    if root not in p.parents or not p.is_file():
        raise PermissionError('File reference must resolve inside workspace')
    return p


def _ref(ref, workspace, suffix='.json'):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise PermissionError('Exact path/hash reference required')
    _digest(ref['sha256'])
    p = _path(ref['path'], workspace)
    if suffix == '.npz':
        if p.name != 'compact_scores.npz':
            raise PermissionError('Only accepted compact_scores.npz payload may be read')
    elif p.suffix.lower() != suffix:
        raise PermissionError('Metadata must be JSON; no raw/parameter body access')
    return p


def _bytes(ref, workspace, suffix='.json'):
    p = _ref(ref, workspace, suffix)
    data = p.read_bytes()
    if sha_bytes(data) != ref['sha256']:
        raise PermissionError('Referenced bytes changed: ' + str(p))
    return data


def _json(ref, workspace):
    def unique_pairs(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(_bytes(ref, workspace), object_pairs_hook=unique_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def source_hashes():
    engine = Path(__file__).resolve().parent
    paths = {'runner': Path(__file__), 'core': Path(core.__file__),
             'events': engine / 'events.py', 'engine_init': engine / '__init__.py',
             'package_init': engine.parent / '__init__.py'}
    return {key: sha_bytes(path.read_bytes()) for key, path in paths.items()}


def _require(condition, message):
    if not condition:
        raise PermissionError(message)


def _manifest_links(manifest, manifest_ref, report_ref, compact_ref, workspace):
    files = manifest.get('files')
    _require(isinstance(files, list), 'Source manifest files missing')
    indexed = {}
    parent = _ref(manifest_ref, workspace).parent
    for item in files:
        rel = Path(item['path'])
        _require(not rel.is_absolute() and '..' not in rel.parts and str(rel) not in indexed,
                 'Unsafe/duplicate manifest path')
        _digest(item['sha256'])
        indexed[str(rel)] = item['sha256']
    for ref in (report_ref, compact_ref):
        p = _ref(ref, workspace, '.npz' if ref is compact_ref else '.json')
        try:
            relative = str(p.relative_to(parent))
        except ValueError:
            raise PermissionError('Report/compact must belong to accepted manifest directory')
        _require(indexed.get(relative) == ref['sha256'], 'Manifest does not bind report/compact bytes')


def verify_metadata(config_ref, release_ref, trusted_release_sha256, workspace):
    """No identity/compact reads until EVERY model metadata chain is verified.

    The caller must obtain trusted_release_sha256 independently from root. This
    routine has no synthetic bypass or path that treats approval text as a signature.
    """
    _require(_digest(trusted_release_sha256) == release_ref.get('sha256'), 'Untrusted release digest')
    release = _json(release_ref, workspace)
    _require(release.get('status') == 'PRIMARY_RELEASED_TEP_FIXED_SCORE_EVALUATION'
             and release.get('approved_by') == 'root' and release.get('operation') == OP
             and release.get('allow_labels') is True and release.get('allow_model_or_fit') is False,
             'Exact root label evaluation release required')
    _require(release.get('config') == config_ref, 'Release config identity differs')
    actual_sources = source_hashes()
    _require(actual_sources['core'] == CORE_SHA and release.get('source_hashes') == actual_sources,
             'Exact frozen source/helper closure required')
    config = _json(config_ref, workspace)
    _require(config.get('status') == 'BOUND_TEP_FIXED_SCORE_EVALUATION' and config.get('operation') == OP,
             'PENDING config cannot open scores or identities')
    seal_ref = config.get('score_seal')
    _require(isinstance(seal_ref, dict) and release.get('score_seal') == seal_ref, 'Exact all-score seal required')
    seal = _json(seal_ref, workspace)
    _require(seal.get('status') == 'PRIMARY_SEALED_ALL_TEP_FIXED_QUERY_SCORES'
             and seal.get('approved_by') == 'root' and seal.get('operation') == OP
             and seal.get('cases') == 400 and seal.get('endpoints_0based') == [99, 959]
             and seal.get('points_per_entity') == 861, 'All-score seal geometry/status differs')
    models = seal.get('score_models')
    _require(isinstance(models, list) and len(models) == 4 and
             [(m.get('family'), m.get('seed')) for m in models] ==
             [('LEFT', 2025), ('LEFT', 2026), ('LEFT', 2027), ('RIDGE_ZMAX', None)],
             'All three LEFT seeds plus CPU model required in fixed order')
    for model in models[:3]:
        core._integer(model['seed'], 'LEFT model seed')
    inputs = _json(seal['input_acceptance'], workspace)
    _require(inputs.get('status') == 'PRIMARY_ACCEPTED_LIMITED_TEP_FAULTY_INPUTS'
             and inputs.get('approved_by') == 'root' and inputs.get('entities') == 400
             and inputs.get('input_manifest_sha256') == seal['input_manifest']['sha256']
             and inputs.get('sealed_identity_manifest_sha256') == seal['identity_manifest']['sha256'],
             'Accepted input and sealed identity chain differs')
    # Validate identity reference shape/path only, without reading/hash-opening it.
    _ref(seal['identity_manifest'], workspace)
    manifest = _json(seal['input_manifest'], workspace)
    _require(manifest.get('status') == 'PUBLIC_SIMULATION_INPUTS_NO_LABEL_FEATURES'
             and [r.get('entity_id') for r in manifest.get('records', [])] == ENTITIES,
             'All accepted opaque input identities required')
    calibrations = []
    for model in models:
        left = model['family'] == 'LEFT'
        acceptance = _json(model['score_acceptance'], workspace)
        status = 'PRIMARY_ACCEPTED_LIMITED_LEFT_TEP_FIXED_QUERY_SCORES' if left else 'PRIMARY_ACCEPTED_LIMITED_TEP_RIDGE_FIXED_QUERY_SCORES'
        _require(acceptance.get('status') == status and acceptance.get('approved_by') == 'root'
                 and acceptance.get('entities') == 400 and acceptance.get('raw_bank_count') == 400
                 and acceptance.get('scored_points_per_entity') == 861
                 and acceptance.get('endpoints_0based') == [99, 959], 'Unaccepted/incomplete score bank')
        for field, ref in [('source_manifest_sha256', model['score_manifest']), ('report_sha256', model['score_report']),
                           ('compact_scores_sha256', model['compact_scores']), ('calibration_sha256', model['calibration']),
                           ('input_manifest_sha256', seal['input_manifest'])]:
            _require(acceptance.get(field) == ref['sha256'], 'Acceptance binding differs: ' + field)
        final = _json(model['score_manifest'], workspace)
        _manifest_links(final, model['score_manifest'], model['score_report'], model['compact_scores'], workspace)
        report = _json(model['score_report'], workspace)
        _require(report.get('status') == 'COMPLETED_PENDING_INDEPENDENT_REVIEW'
                 and [r.get('entity_id') for r in report.get('banks', [])] == ENTITIES,
                 'Report incomplete or reordered banks')
        cal = _json(model['calibration'], workspace)
        _require(cal.get('n') == 8020 and cal.get('rank_1based') == 7941
                 and cal.get('comparison') == 'strict_greater', 'Fixed normal CAL differs')
        keys = set(core.READOUTS) if left else {m + '_' + d for m in core.METHODS[1:] for d in core.READOUTS}
        _require(set(cal.get('thresholds', {})) == keys, 'Exact frozen thresholds required')
        for value in cal['thresholds'].values():
            core._finite_scalar(value, 'threshold')
        scaler = _digest(model['scaler_sha256'])
        _require(cal.get('scaler_sha256') == scaler, 'Wrong normal scaler')
        if left:
            checkpoint = _digest(model['checkpoint_sha256'])
            for value in (acceptance.get('seed'), report.get('seed'), cal.get('seed')):
                core._integer(value, 'LEFT accepted/report/CAL seed')
            _require(acceptance.get('seed') == model['seed'] == report.get('seed') == cal.get('seed')
                     and acceptance.get('checkpoint_sha256') == checkpoint == cal.get('checkpoint_sha256')
                     and acceptance.get('scaler_sha256') == scaler
                     and report.get('compact_scores_sha256') == model['compact_scores']['sha256'],
                     'LEFT seed/checkpoint/scaler/CAL cannot be interchanged')
        else:
            coefficient = _digest(model['coefficients_sha256'])
            refs = report.get('source_refs', {})
            _require(cal.get('coefficient_sha256') == coefficient
                     and refs.get('coefficients', {}).get('sha256') == coefficient
                     and refs.get('scaler', {}).get('sha256') == scaler
                     and refs.get('calibration') == model['calibration']
                     and refs.get('input_manifest') == seal['input_manifest']
                     and report.get('compact_sha256') == model['compact_scores']['sha256']
                     and report.get('calibration_thresholds') == cal['thresholds']
                     and acceptance.get('methods') == list(core.METHODS[1:])
                     and acceptance.get('readouts') == list(core.READOUTS), 'CPU fixed parameter/CAL chain differs')
        calibrations.append(cal)
    # All four chains closed; payload phase remains separate and re-verifies gates.
    return dict(config=config, release=release, seal=seal, calibrations=calibrations)


def load_verified_payload(config_ref, release_ref, trusted_release_sha256, workspace):
    """Re-run all metadata gates internally; no caller-created verified token."""
    state = verify_metadata(config_ref, release_ref, trusted_release_sha256, workspace)
    seal = state['seal']
    identity = _json(seal['identity_manifest'], workspace)
    _require(identity.get('status') == 'WORKFLOW_SEALED_NOT_CRYPTOGRAPHIC_OR_BLIND', 'Identity schema differs')
    records = identity.get('records', [])
    _require([r.get('entity_id') for r in records] == ENTITIES, 'Identity coverage differs')
    pairs = []
    for r in records:
        f, run = core._integer(r['faultNumber'], 'fault'), core._integer(r['simulationRun'], 'run')
        _require(r.get('nominal_first_positive_sample_1based') == 161 and r.get('sampling_minutes') == 3,
                 'Nominal coordinate contract differs')
        pairs.append((f, run))
    _require(len(set(pairs)) == 400 and set(pairs) == {(f, r) for f in core.FAULTS for r in core.RUNS},
             'Incomplete/duplicate fault/run identity mapping')
    import io
    import numpy as np
    compacts = []
    for model in seal['score_models']:
        # Hash and parse the SAME immutable byte buffer, avoiding a reopen race.
        data = _bytes(model['compact_scores'], workspace, '.npz')
        with np.load(io.BytesIO(data), allow_pickle=False) as bank:
            arrays = {k: bank[k].copy() for k in bank.files}
        core.validate_compact(arrays, model['family'])
        compacts.append(arrays)
    return state, records, compacts


def execute(config_ref, release_ref, trusted_release_sha256, workspace):
    """Future released CPU execution; no models, fits or threshold selection."""
    _require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '1' for k in
             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'CPU-only single-thread environment required')
    start = time.monotonic()
    state, identities, banks = load_verified_payload(config_ref, release_ref, trusted_release_sha256, workspace)
    import numpy as np
    cases = []
    for model, cal, bank in zip(state['seal']['score_models'], state['calibrations'], banks):
        for i, row in enumerate(identities):
            for method in (('LEFT',) if model['family'] == 'LEFT' else core.METHODS[1:]):
                for d in core.READOUTS:
                    key = d if method == 'LEFT' else method + '_' + d
                    cases.append(core.evaluate_case(bank[key][i], bank['end_index_0based'][i], cal['thresholds'][key],
                        row['faultNumber'], row['simulationRun'], method, d, model['seed'],
                        bank['available_index_0based'][i] if method == 'LEFT' else None))
                if time.monotonic() - start > 300:
                    raise TimeoutError('300-second fixed CPU budget exceeded; no reduced statistics')
    aggregate = core.aggregate_cases(cases)
    draws = core.bootstrap_draws()
    intervals = core.paired_intervals(aggregate, draws)
    if time.monotonic() - start > 300:
        raise TimeoutError('300-second fixed CPU budget exceeded; no partial success')
    out = Path(state['config']['output'])
    root = Path(workspace).resolve()
    _require(out.is_absolute() and root in out.resolve().parents, 'Output must stay inside workspace')
    out.mkdir(parents=False, exist_ok=False)
    def dump(name, value):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    dump('case_events_and_metrics.json', cases)
    dump('aggregation.json', aggregate)
    dump('paired_intervals.json', intervals)
    np.save(out / 'bootstrap_run_cluster_draws.npy', draws, allow_pickle=False)
    with (out / 'case_metrics.csv').open('x', newline='', encoding='utf-8') as handle:
        fields = ['fault', 'run', 'method', 'readout', 'seed'] + list(core.METRICS)
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for case in cases:
            writer.writerow(dict({k: case[k] for k in fields[:5]}, **case['metrics']))
    dump('evaluation_receipt.json', dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW', config_ref=config_ref,
        release_ref=release_ref, trusted_release_sha256=trusted_release_sha256,
        score_seal=state['config']['score_seal'], source_hashes=source_hashes(),
        elapsed_seconds=time.monotonic() - start, GPU_hours=0, cases=6400,
        delay_note='NA delays are censored; summary valid/NA counts disclose detected/censored denominator',
        scientific_decision='HOLD', innovation='UNSUPPORTED'))
    files = [dict(path=p.name, sha256=sha_bytes(p.read_bytes()), bytes=p.stat().st_size) for p in sorted(out.iterdir())]
    dump('FINAL_MANIFEST.json', dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW', files=files))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--config-sha256', required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--trusted-release-sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    cfg = dict(path=args.config, sha256=args.config_sha256)
    rel = dict(path=args.release, sha256=args.trusted_release_sha256)
    if args.execute:
        execute(cfg, rel, args.trusted_release_sha256, args.workspace)
    else:
        verify_metadata(cfg, rel, args.trusted_release_sha256, args.workspace)
        print('METADATA_ONLY_VERIFIED_NO_IDENTITY_OR_COMPACT_READ')


if __name__ == '__main__':
    main()
