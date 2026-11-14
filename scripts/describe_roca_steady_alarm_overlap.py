"""Exact existing-alarm overlap on original SMAP long-event steady axes.

No forward, score, threshold fitting, or altered event definition.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_alarms(path):
    """Retain full original axis, including invalid/UNKNOWN time points."""
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    index = np.asarray([int(row['index']) for row in rows], dtype=np.int64)
    if len(index) and not np.array_equal(index, np.arange(index[0], index[-1] + 1)):
        raise ValueError('Expected complete original native axis')
    valid = np.asarray([row['valid'] == 'True' for row in rows])
    strict = np.asarray([row['strict_point'] == 'True' for row in rows])
    confirmed = np.asarray([row['confirmed3_current'] == 'True' for row in rows])
    if np.any((strict | confirmed) & ~valid):
        raise ValueError('An invalid point has an alarm')
    return index, valid, strict, confirmed


def overlap(left, right, use):
    l = left & use
    r = right & use
    return dict(eligible_points=int(use.sum()), left_alarm_points=int(l.sum()),
                right_alarm_points=int(r.sum()), intersection=int((l & r).sum()),
                xor=int((l ^ r).sum()), left_only=int((l & ~r).sum()),
                right_only=int((r & ~l).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--trusted-config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.trusted_config_sha256:
        raise PermissionError('Exact read-only frozen config required')
    config = json.loads(Path(args.config).read_text())
    started = time.monotonic()
    geometry = Path(config['geometry'])
    accepted = Path(config['accepted_evaluation'])
    gates = json.loads((geometry / 'ALL_SCORE_GATES_BEFORE_LABELS.json').read_text())
    entities = sorted({row['entity'] for row in gates if row['domain'] == 'SMAP'})
    names = ['RoCA_FULL_T__q__raw', 'RoCA_INIT_T__q__raw', 'FIT_STATIC_T']
    pairs = [(0, 1), (0, 2), (1, 2)]
    event_rows = []
    entity_rows = []
    bindings = []
    for entity in entities:
        arrays = []
        for name in names:
            root = accepted if name == 'FIT_STATIC_T' else geometry
            path = root / 'entities' / 'SMAP' / entity / name / 'point_coverage.csv'
            arrays.append(read_alarms(path))
            bindings.append(dict(path=str(path), sha256=sha(path)))
        index = arrays[0][0]
        if not all(np.array_equal(index, row[0]) for row in arrays[1:]):
            raise ValueError('Original axes differ across fixed methods')
        common_valid = np.logical_and.reduce([row[1] for row in arrays])
        event_path = accepted / 'entities' / 'SMAP' / entity / 'RoCA_FULL_T' / 'event_coverage.csv'
        bindings.append(dict(path=str(event_path), sha256=sha(event_path)))
        with event_path.open() as handle:
            events = list(csv.DictReader(handle))
        steady_all = np.zeros(len(index), dtype=bool)
        for event in events:
            start, end = int(event['start']), int(event['end_exclusive'])
            if end - start <= 512:
                continue
            use = (index >= start + 512) & (index < end)
            steady_all |= use
            for left, right in pairs:
                for alarm, column in [('strict', 2), ('confirmed3', 3)]:
                    values = overlap(arrays[left][column], arrays[right][column], use & common_valid)
                    event_rows.append(dict(entity=entity, event_id=event['event_id'],
                        source_event_id=event.get('source_event_id'), start=start, end_exclusive=end,
                        steady_start=start + 512, original_steady_points=end - start - 512,
                        run_end_censored=event['boundary_censored'] == 'True',
                        main_eligible=event['main_eligible'] == 'True', left=names[left], right=names[right],
                        alarm=alarm, **values))
        for left, right in pairs:
            for alarm, column in [('strict', 2), ('confirmed3', 3)]:
                entity_rows.append(dict(entity=entity, left=names[left], right=names[right], alarm=alarm,
                    original_steady_points=int(steady_all.sum()),
                    **overlap(arrays[left][column], arrays[right][column], steady_all & common_valid)))
    aggregate = []
    fields = ['original_steady_points', 'eligible_points', 'left_alarm_points', 'right_alarm_points',
              'intersection', 'xor', 'left_only', 'right_only']
    for left, right in pairs:
        for alarm in ['strict', 'confirmed3']:
            selected = [row for row in entity_rows if row['left'] == names[left] and
                        row['right'] == names[right] and row['alarm'] == alarm]
            aggregate.append(dict(left=names[left], right=names[right], alarm=alarm,
                                  **{key: sum(row[key] for row in selected) for key in fields}))
    output = Path(config['steady_overlap_output'])
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status='COMPLETED_READONLY_EXISTING_ALARMS', seconds=time.monotonic() - started,
                  scientific_scope='Exact overlap, not equality inferred from counts; all original long events including boundary censored',
                  steady_rule='[original_event_start+512,original_event_end)',
                  new_score=False, new_tau=False, new_forward=False, GPUh=0,
                  aggregate=aggregate, entities=entity_rows, events=event_rows,
                  input_bindings=bindings, source_config=config)
    (output / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'],
                          long_events=len(event_rows) // 6, aggregate=aggregate)))


if __name__ == '__main__':
    main()
