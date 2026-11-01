"""Synthetic-only contract fixtures; no file-based score/input/checkpoint access."""
import importlib.util
import unittest
import numpy as np

from hybrid_tsad.engine.tep_fixed_evaluation import (
    event_metrics, evaluate_case, aggregate_cases, paired_intervals,
    bootstrap_draws, validate_compact, require_future_release,
    SEEDS, METHODS, READOUTS, METRICS)

ENDS = np.arange(99, 960, dtype=np.int64)


def scores(*segments):
    a = np.zeros(861)
    for start, end in segments:
        a[start - 100:end - 99] = 2.
    return a


def fixture_cases():
    # Scores aren't needed for aggregate arithmetic; metrics are artificial.
    return [dict(fault=f, run=r, method=m, readout=d, seed=s,
                 metrics={k: (r - 51) / 20 + f / 100 +
                          ((s - 2025) / 100 + .1 if m == 'LEFT' else 0.) for k in METRICS})
            for f in range(1, 21) for r in range(51, 71) for m in METHODS
            for d in READOUTS for s in (SEEDS if m == 'LEFT' else (None,))]


class EventContract(unittest.TestCase):
    def test_third_strict_exceedance_and_reset(self):
        a = scores((161, 162), (164, 166))
        a[163 - 100] = 1.  # equality resets, cannot bridge
        out = event_metrics(a, ENDS, 1.)
        self.assertEqual(out['first_eligible_confirmation_sample'], 166)
        self.assertEqual(out['metrics']['detection_delay_minutes'], 15)
        self.assertEqual(out['new_confirmed_segments'], 1)

    def test_181_is_timely_182_is_late_full_precision_still_one(self):
        for start, expected in [(179, 1.), (180, 0.)]:
            out = event_metrics(scores((start, start + 2)), ENDS, 1.)
            self.assertEqual(out['metrics']['eventRecall_60min'], expected)
            self.assertEqual(out['metrics']['eventRecall_full800'], 1.)
            self.assertEqual(out['metrics']['eventPrecision_full800'], 1.)
            self.assertEqual(out['metrics']['detection_delay_minutes'], (start + 2 - 161) * 3)

    def test_boundary_before_after_unconfirmed_and_new_match(self):
        for start, classification in [(158, 'before_injection'), (159, 'after_injection'), (160, 'after_injection')]:
            out = event_metrics(scores((start, 165), (170, 172)), ENDS, 1.)
            self.assertEqual(out['boundary_segments'], 1)
            self.assertEqual(out['segments'][0]['boundary_confirmation'], classification)
            self.assertEqual(out['first_eligible_confirmation_sample'], 172)
            self.assertEqual(out['normal_confirmed_segments'], int(start == 158))
            self.assertEqual(out['metrics']['eventPrecision_full800'], .5 if start == 158 else 1.)
        out = event_metrics(scores((160, 161)), ENDS, 1.)
        self.assertEqual(out['boundary_segments'], 1)
        self.assertIsNone(out['metrics']['eventPrecision_full800'])
        self.assertEqual(out['metrics']['eventRecall_full800'], 0.)

    def test_preconfirmed_crossing_preserves_one_normal_false_alarm(self):
        out = event_metrics(np.full(861, 2.), ENDS, 1.)
        self.assertEqual(out['normal_confirmed_segments'], 1)
        self.assertEqual(out['boundary_confirmed_before'], 1)
        self.assertEqual(out['new_confirmed_segments'], 0)
        self.assertEqual((out['event_TP'], out['event_FP']), (0, 1))
        self.assertEqual(out['metrics']['normal_false_alarms_per_hour_183min'], 60 / 183)
        self.assertEqual(out['metrics']['eventRecall_full800'], 0.)
        self.assertEqual(out['metrics']['eventPrecision_full800'], 0.)
        after = event_metrics(scores((159, 163)), ENDS, 1.)
        self.assertEqual(after['normal_confirmed_segments'], 0)
        self.assertEqual(after['boundary_confirmed_after'], 1)
        self.assertEqual((after['event_TP'], after['event_FP']), (0, 0))
        self.assertIsNone(after['metrics']['eventPrecision_full800'])

    def test_duplicate_normal_precision_exposure_and_censor(self):
        out = event_metrics(scores((100, 102), (161, 163), (958, 960)), ENDS, 1.)
        self.assertEqual((out['event_TP'], out['event_FP'], out['event_FN']), (1, 2, 0))
        self.assertEqual(out['metrics']['eventPrecision_full800'], 1 / 3)
        self.assertEqual(out['metrics']['eventF1_full800'], .5)
        self.assertEqual(out['repeated_new_alarms'], 1)
        self.assertEqual(out['metrics']['normal_false_alarms_per_hour_183min'], 60 / 183)
        self.assertEqual(out['metrics']['normal_false_alarms_per_hour_180min'], 1 / 3)
        empty = event_metrics(np.zeros(861), ENDS, 1.)
        self.assertIsNone(empty['metrics']['eventPrecision_full800'])
        self.assertEqual(empty['metrics']['eventF1_full800'], 0.)
        self.assertTrue(empty['delay_censored'])
        self.assertIsNone(empty['metrics']['detection_delay_minutes'])
        self.assertEqual(empty['censor_horizon_minutes'], 2397)

    def test_end_confirmation_and_independent_entities(self):
        self.assertEqual(event_metrics(scores((958, 960)), ENDS, 1.)['first_eligible_confirmation_sample'], 960)
        self.assertIsNone(event_metrics(scores((959, 960)), ENDS, 1.)['first_eligible_confirmation_sample'])
        self.assertIsNone(event_metrics(scores((100, 100)), ENDS, 1.)['first_eligible_confirmation_sample'])

    def test_geometry_rejects_gaps_nan_noninteger_and_nonfinite_threshold(self):
        for a, ends, t in [(np.zeros(860), ENDS, 1.), (np.zeros(861), ENDS + 1, 1.),
                           (np.zeros(861), ENDS.astype(float), 1.),
                           (np.full(861, np.nan), ENDS, 1.), (np.zeros(861), ENDS, np.inf),
                           (np.zeros(861, dtype=bool), ENDS, 1.)]:
            with self.assertRaises(ValueError):
                event_metrics(a, ends, t)

    def test_invalid_identity_before_sklearn_import(self):
        for kwargs in [dict(seed=2025.), dict(seed=True), dict(seed=2028),
                       dict(fault=0), dict(run=71), dict(readout='best'), dict(method='UNKNOWN')]:
            args = dict(fault=1, run=51, method='LEFT', readout='last', seed=2025)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                evaluate_case(np.zeros(861), ENDS, 1., **args)
        with self.assertRaises(ValueError):
            evaluate_case(np.zeros(861), ENDS, 1., 1, 51, 'MULTI', 'last', 2025)
        with self.assertRaises(ValueError):
            evaluate_case(np.zeros(861), ENDS, 1., 1, 51, 'LEFT', 'last', 2025, ENDS - 1)

    def test_pending_release_always_refuses_including_forged_refs(self):
        for config in [{}, {'status': 'PENDING', 'score_seals': None},
                       {'status': 'RELEASED', 'score_seals': 'forged'}]:
            with self.assertRaises(PermissionError):
                require_future_release(config)


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Existing local environment has no sklearn; no install authorized')
class MatureRanking(unittest.TestCase):
    def test_tied_and_perfect_scores(self):
        out = evaluate_case(np.zeros(861), ENDS, 1., 1, 51, 'LEFT', 'last', 2025)
        m = out['metrics']
        self.assertAlmostEqual(m['AP'], 800 / 861)
        self.assertEqual(m['ROC_AUC'], .5)
        self.assertEqual(m['AP_divided_by_pi'], 1.)
        self.assertAlmostEqual(m['trapezoid_PR_AUC'], (1 + 800 / 861) / 2)
        self.assertIsNone(m['pointPrecision'])
        self.assertEqual(m['pointF1'], 0.)
        out = evaluate_case(scores((161, 960)), ENDS, 1., 1, 51, 'MULTI', 'mean32')
        for key in ['AP', 'ROC_AUC', 'trapezoid_PR_AUC', 'pointPrecision', 'pointRecall', 'pointF1']:
            self.assertEqual(out['metrics'][key], 1.)


class AggregationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture_cases()
        cls.aggregation = aggregate_cases(cls.rows)
        cls.draws = bootstrap_draws()

    def test_exact_counts_and_seed_average(self):
        self.assertEqual(len(self.rows), 6400)
        self.assertEqual(len(self.aggregation['seed_averaged_cells']), 4800)
        cell = self.aggregation['seed_averaged_cells'][0]
        self.assertAlmostEqual(cell['metrics']['AP'], .12)
        summary = next(s for s in self.aggregation['summaries'] if s['group'] == 'related3')
        self.assertEqual(summary['metrics']['AP']['total'], 60)

    def test_missing_duplicate_unknown_seed_invalid_metric_rejected(self):
        for rows in [self.rows[:-1], self.rows + self.rows[:1],
                     [dict(self.rows[0], seed=2025.)] + self.rows[1:],
                     [dict(self.rows[0], method='OTHER')] + self.rows[1:],
                     [dict(self.rows[0], metrics={'AP': 1.})] + self.rows[1:]]:
            with self.assertRaises(ValueError):
                aggregate_cases(rows)

    def test_paired_shared_draws_and_na_never_dropped(self):
        result = paired_intervals(self.aggregation, self.draws)
        self.assertEqual(len(result), 2 * 2 * 5 * len(METRICS))
        for row in result:
            self.assertEqual(row['status'], 'COMPLETE')
            self.assertAlmostEqual(row['mean_difference'], .11)
            np.testing.assert_allclose(row['CI95'], [.11, .11], atol=1e-14)
        changed = [dict(r, metrics=dict(r['metrics'])) for r in self.rows]
        changed[0]['metrics']['pointPrecision'] = None
        aggregate = aggregate_cases(changed)
        summary = next(s for s in aggregate['summaries'] if s['group'] == 'all20' and s['method'] == 'LEFT' and s['readout'] == 'last')
        self.assertEqual(summary['metrics']['pointPrecision']['NA_ids'], ['f01/run51'])
        result = paired_intervals(aggregate, self.draws)
        affected = [r for r in result if r['group'] == 'all20' and r['readout'] == 'last' and r['metric'] == 'pointPrecision']
        self.assertTrue(all(r['status'] == 'INCOMPLETE' and r['CI95'] is None for r in affected))
        self.assertTrue(all(r['status'] == 'COMPLETE' for r in result if r['metric'] == 'AP'))
        with self.assertRaises(ValueError):
            paired_intervals(aggregate, self.draws[::-1])

    def test_nonconstant_run_clusters_keep_all_faults_and_seed_average(self):
        changed = [dict(r, metrics=dict(r['metrics'])) for r in self.rows]
        for row in changed:
            if row['method'] == 'LEFT':
                row['metrics']['AP'] += (row['run'] - 60) * row['fault'] / 10000
        result = paired_intervals(aggregate_cases(changed), self.draws)
        for group, faults in [('all20', range(1, 21)), ('related3', (4, 11, 14))]:
            # Independent explicit cluster reduction: same run has ALL group faults;
            # three training seeds average to .11, not 60 independent run units.
            delta = np.array([sum(.11 + (r - 60) * f / 10000 for f in faults) / len(faults)
                              for r in range(51, 71)])
            expected = np.percentile(np.mean(delta[self.draws], axis=1), [2.5, 97.5])
            matching = [r for r in result if r['group'] == group and r['metric'] == 'AP']
            self.assertEqual(len(matching), 10)
            for row in matching:
                self.assertAlmostEqual(row['mean_difference'], float(delta.mean()))
                np.testing.assert_allclose(row['CI95'], expected, atol=1e-14)


class CompactContract(unittest.TestCase):
    def test_both_schemas_and_missing_paths(self):
        base = dict(entity_id=np.array(['entity_%04d' % i for i in range(1, 401)]),
                    end_index_0based=np.tile(ENDS, (400, 1)))
        left = dict(base, available_index_0based=base['end_index_0based'], last=np.zeros((400, 861)),
                    mean32=np.zeros((400, 861)), channel_last=np.zeros((400, 861, 3)),
                    channel_mean32=np.zeros((400, 861, 3)))
        self.assertTrue(validate_compact(left, 'LEFT'))
        ridge = dict(base, **{m + '_' + r: np.zeros((400, 861)) for m in METHODS[1:] for r in READOUTS})
        self.assertTrue(validate_compact(ridge, 'RIDGE_ZMAX'))
        bad = dict(left); del bad['last']
        with self.assertRaises(ValueError):
            validate_compact(bad, 'LEFT')
        bad = dict(left, available_index_0based=base['end_index_0based'] - 1)
        with self.assertRaises(ValueError):
            validate_compact(bad, 'LEFT')
        bad = dict(left, entity_id=np.repeat('entity_0001', 400))
        with self.assertRaises(ValueError):
            validate_compact(bad, 'LEFT')


if __name__ == '__main__':
    unittest.main()
