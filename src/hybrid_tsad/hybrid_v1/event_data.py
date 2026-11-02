"""Bounded event-support preparation. CPU/NumPy only; never train or query.

Model manifests have no labels/ledger. NORMAL_SELECT/CAL retain original role
axes and MUST be consumed with emitted explicit plans, never role-wide scans.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .target_data import sha

OPERATION = 'HYBRID_V1_FIXED_SMD_F_INPUTS'
ROLES = {'SUPPORT': [0, 9477], 'NORMAL_SELECT': [10021, 13031], 'NORMAL_CAL': [13575, 16585]}
EVENTS = [(6506, 6528), (7907, 7961), (9340, 9386), (9779, 9805), (12173, 12181), (16136, 16588)]
NORMAL = [(6453, 6475), (7822, 7876), (9263, 9309)]
INPUTS = {'labels': ('416dd20a447f38b8d9394382293ed44f509c1c3f99c4aa2fbff179498626dbdc', 47388, 17129),
          'values': ('2d103661799271be958227907950a1c76803356442d60a4070671cc0abbefb20', 8103348, 16585)}


def prefix_rows(stream, stop, columns):
    """islice bounds numeric decoding; no extra readline beyond stop."""
    if type(stop) is not int or stop < 1 or columns < 1:
        raise ValueError('Positive explicit prefix required')
    rows = []
    for line in itertools.islice(stream, stop):
        fields = line.strip().split(',')
        if len(fields) != columns:
            raise ValueError('Native column count changed')
        rows.append([float(x) for x in fields])
    if len(rows) != stop:
        raise ValueError('Truncated authorized prefix')
    return np.asarray(rows, dtype=np.float64)


def event_intervals(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all() or labels[-1] != 0:
        raise ValueError('Closed binary prefix required; never extend read to find end')
    edges = np.diff(np.r_[0, labels, 0].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _inside(a, b, lo, hi, limit):
    if not 0 <= lo <= a < b <= hi <= limit:
        raise ValueError('Footprint crosses role or authorized numeric prefix')


def _eligible(observed, ts, history):
    cs = np.concatenate([np.zeros((1, observed.shape[1]), np.int64), observed.cumsum(0, dtype=np.int64)])
    return observed[ts].all(1) & ((cs[ts]-cs[ts-history+1]) >= 2).all(1)


def construct(values, labels, *, identity, roles, catalogue, normal, history, horizon,
              counts, seed=2025):
    """Generic original-axis constructor; catalogue IDs supplied, never invented.

    Positives require exact original event bounds and no other anomaly in their
    full footprint. Negatives and normal plans require label-zero full H/P.
    """
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels)
    if values.ndim != 2 or np.isinf(values).any() or history < 3 or horizon < 1:
        raise ValueError('Canonical finite-or-NaN 2D values and legal H/P required')
    if len(labels) < len(values) or set(identity) != {'domain', 'entity', 'run'}:
        raise ValueError('Complete label prefix and original identity required')
    if set(roles) != {'SUPPORT', 'NORMAL_SELECT', 'NORMAL_CAL'}:
        raise ValueError('Only fixed adaptation roles allowed')
    bounds = [roles[r] for r in ('SUPPORT', 'NORMAL_SELECT', 'NORMAL_CAL')]
    for lo, hi in bounds:
        _inside(lo, hi, lo, hi, len(values))
    if any(b[0]-a[1] < history+horizon for a, b in zip(bounds, bounds[1:])):
        raise ValueError('Original roles require H+P guards')
    if event_intervals(labels) != [(e['start'], e['end']) for e in catalogue]:
        raise ValueError('Original event catalogue mismatch; no trimming or renumbering')
    ids = [e['event_id'] for e in catalogue]
    if len(set(ids)) != len(ids):
        raise ValueError('Canonical event IDs must be unique')
    observed = np.isfinite(values)
    eligible_support = []
    lo, hi = roles['SUPPORT']
    for match in normal:
        matches = [e for e in catalogue if e['event_id'] == match['matched_event_id']]
        if len(matches) != 1:
            raise ValueError('Canonical matched event missing')
        event = matches[0]; s, e = event['start'], event['end']
        ns, ne = match['start'], match['end']
        if e-s != ne-ns:
            raise ValueError('Matched normal must have equal target length')
        for a, b in ((s, e), (ns, ne)):
            _inside(a-history, b+horizon-1, lo, hi, len(values))
            if b+horizon-1 > len(labels):
                raise ValueError('Label footprint outside permission')
            if not _eligible(observed, np.arange(a, b), history).all():
                raise ValueError('Every budgeted support point must be observation eligible')
        expected = np.zeros(e+horizon-1-(s-history), dtype=labels.dtype)
        expected[history:history+e-s] = 1
        if not np.array_equal(labels[s-history:e+horizon-1], expected):
            raise ValueError('Positive footprint contains unbudgeted anomaly')
        if np.any(labels[ns-history:ne+horizon-1]):
            raise ValueError('Negative support contamination')
        if ns < e and s < ne:
            raise ValueError('Positive and matched normal targets overlap')
        eligible_support.append((dict(identity, **event), dict(identity, **match)))
    # Fixed order is catalogue order; do not reindex supplied IDs.
    order = {e['event_id']: i for i, e in enumerate(catalogue)}
    eligible_support.sort(key=lambda pair: order[pair[0]['event_id']])
    if len({e['event_id'] for e, _ in eligible_support}) != len(eligible_support):
        raise ValueError('Duplicate matched support event')
    for i, (ev, nm) in enumerate(eligible_support):
        for other, on in eligible_support[i+1:]:
            for x in (ev, nm):
                for y in (other, on):
                    if x['start'] < y['end'] and y['start'] < x['end']:
                        raise ValueError('Support target intervals overlap')
    rng = np.random.default_rng(seed); plans = {}; candidate_counts = {}
    for role in ('NORMAL_SELECT', 'NORMAL_CAL'):
        lo, hi = roles[role]
        candidates = np.arange(lo+history, hi-horizon+1, dtype=np.int64)
        lc = np.r_[0, np.cumsum(labels, dtype=np.int64)]
        clean = (lc[candidates+horizon]-lc[candidates-history]) == 0
        candidates = candidates[clean & _eligible(observed, candidates, history)]
        candidate_counts[role] = len(candidates)
        if len(candidates) < counts[role]:
            raise ValueError('Insufficient normal candidates; no fallback')
        chosen = rng.choice(candidates, size=counts[role], replace=False)
        plans[role] = [dict(identity, role=role, index=int(t), footprint=[int(t)-history, int(t)+horizon]) for t in chosen]
    lo, hi = roles['NORMAL_CAL']; ts = np.arange(lo+history, hi)
    lc = np.r_[0, np.cumsum(labels, dtype=np.int64)]
    candidate_counts['NORMAL_CAL_arrived_not_selected'] = int(np.sum(((lc[ts+1]-lc[ts-history]) == 0) & _eligible(observed, ts, history)))
    combinations = []
    # All three singleton choices retained; K3 unique; not sampled by outcome.
    for k in (1, 3, 5):
        choices = list(itertools.combinations(eligible_support, k))
        if not choices:
            combinations.append(dict(K=k, status='NA', reason='Insufficient independent positive events'))
        for choice in choices:
            combinations.append(dict(K=k, status='AVAILABLE', events=[p[0] for p in choice], normal=[p[1] for p in choice]))
    ledger = dict(label_prefix_points=len(labels), prefix_events=catalogue, normal_matches=normal,
                  K_definition='positive training events only, not total annotation budget',
                  extra_supervision=['normal matched full footprints', 'NORMAL_SELECT exclusion and screening', 'NORMAL_CAL exclusion and screening'],
                  history_exposure='same historically developed machine/record; historical 13-event catalogue known; not blind confirmation',
                  forbidden_as_model_features=True)
    return dict(plans=plans, candidate_counts=candidate_counts, combinations=combinations, ledger=ledger,
                values=values, observed=observed)


def validate_config(cfg):
    if (cfg.get('operation') != OPERATION or cfg.get('roles') != ROLES or cfg.get('history') != 512
            or cfg.get('horizon') != 32 or cfg.get('sampling_seed') != 2025
            or cfg.get('selection_counts') != {'NORMAL_SELECT': 32, 'NORMAL_CAL': 512}
            or cfg.get('identity') != dict(domain='SMD', entity='machine-2-1', run='machine-2-1_test')):
        raise PermissionError('Fixed admitted F fold changed')
    for name, (digest, size, stop) in INPUTS.items():
        spec = cfg['inputs'][name]
        if (spec['sha256'], spec['bytes'], spec['numeric_stop']) != (digest, size, stop):
            raise PermissionError('Original identity/numeric prefix changed')
    expected = [dict(event_id='SMD:machine-2-1:E%d'%i, alias='E%d'%i, event_index=138+i, start=s, end=e) for i, (s, e) in enumerate(EVENTS)]
    normals = [dict(matched_event_id=expected[i]['event_id'], start=s, end=e) for i, (s, e) in enumerate(NORMAL)]
    if cfg['catalogue'] != expected or cfg['normal_matches'] != normals:
        raise PermissionError('Fixed catalogue/canonical mapping/matches changed')


def derive(cfg):
    validate_config(cfg)
    output = Path(cfg['output'])
    if output.exists():
        raise FileExistsError('Fresh external output required')
    for spec in cfg['inputs'].values():
        p = Path(spec['path'])
        if p.stat().st_size != spec['bytes'] or sha(p) != spec['sha256']:
            raise ValueError('Original full-file byte identity mismatch')
    with Path(cfg['inputs']['labels']['path']).open('r', encoding='ascii') as f:
        labels = prefix_rows(f, 17129, 1)[:, 0]
    with Path(cfg['inputs']['values']['path']).open('r', encoding='ascii') as f:
        values = prefix_rows(f, 16585, 38)
    result = construct(values, labels, identity=cfg['identity'], roles=ROLES, catalogue=cfg['catalogue'], normal=cfg['normal_matches'],
                       history=512, horizon=32, counts=cfg['selection_counts'], seed=2025)
    if result['candidate_counts'] != dict(NORMAL_SELECT=1916, NORMAL_CAL=2018, NORMAL_CAL_arrived_not_selected=2049):
        raise ValueError('Accepted candidate counts changed; preserve failure, no fallback')
    output.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    npz = output/'values_observed_original_axis.npz'
    np.savez_compressed(npz, values=result['values'], observed=result['observed'], original_index=np.arange(16585, dtype=np.int64))
    entries = [dict(cfg['identity'], role=r, path=npz.name, format='npz', key='values', observed_key='observed',
                    sha256=sha(npz), expected_shape=[16585, 38], bounds=b, original_start=0) for r, b in ROLES.items()]
    write('f_role_manifest.json', dict(runs=entries, explicit_normal_plans_required=True))
    write('normal_plans.json', dict(history=512, horizon=32, seed=2025, **result['plans']))
    write('annotation_ledger.NOT_MODEL_INPUT.json', result['ledger'])
    heldout = [dict(cfg['identity'], start=9477, end=23694)]
    index = []
    for i, combination in enumerate(result['combinations']):
        if combination['status'] == 'NA':
            index.append(combination); continue
        name = 'support_K%d_choice%d.json'%(combination['K'], i)
        selected = {e['event_id'] for e in combination['events']}
        # Only selected event metadata is exposed; other support event closures
        # are already proved absent by the builder and do not enter F features.
        support = dict(K=combination['K'], events=combination['events'], normal=combination['normal'], forbidden=heldout)
        write(name, support)
        index.append(dict(K=combination['K'], status='AVAILABLE', path=name, event_ids=sorted(selected)))
    write('support_index.json', index)
    report = dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW', candidate_counts=result['candidate_counts'],
                  numeric_parse_ranges=dict(labels=[0,17129], values=[0,16585]), input_identity=cfg['inputs'],
                  query_numeric_read=False, model_or_training=False, explicit_normal_plans_required=True,
                  observed_definition='np.isfinite(canonical float32 benchmark values); not physical missingness truth',
                  phase_warning='Do not scan complete normal roles; E4/E5 remain in original role arrays. No F phase released.')
    write('report.json', report)
    write('ARTIFACT_IDENTITY.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--execute', action='store_true')
    p.add_argument('--release'); p.add_argument('--trusted-release-sha256'); a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding='utf-8')); validate_config(cfg)
    if not a.execute:
        print(json.dumps(dict(preview=True, numeric_read=False))); return
    if not a.release or not a.trusted_release_sha256 or sha(a.release) != a.trusted_release_sha256:
        raise PermissionError('Root exact config/source release required')
    release = json.loads(Path(a.release).read_text(encoding='utf-8'))
    closure = cfg['source_sha256']; base = Path(__file__).resolve().parents[2]
    if (release.get('status') != 'APPROVED' or release.get('approved_by') != 'root'
            or release.get('operation') != OPERATION or release.get('allowed_actions') != ['prepare_f_inputs']
            or release.get('config_sha256') != sha(a.config) or release.get('source_sha256') != closure
            or not {'hybrid_tsad/hybrid_v1/event_data.py', 'hybrid_tsad/hybrid_v1/target_data.py'}.issubset(closure)):
        raise PermissionError('Changed root operation/config/source release')
    for name, digest in closure.items():
        source = (base/name).resolve()
        if base not in source.parents or sha(source) != digest:
            raise PermissionError('Source bytes changed')
    print(json.dumps(derive(cfg)))


if __name__ == '__main__':
    main()
