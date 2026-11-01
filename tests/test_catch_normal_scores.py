"""Pure CPU metadata/calibration fixtures; no author imports/model/feature IO."""
import unittest
import numpy as np
from hybrid_tsad import catch_normal_scores as c


class CatchScoreTests(unittest.TestCase):
    def test_training_scene_and_root_acceptance_cannot_be_swapped(self):
        import copy
        scene = 'LIN'
        normal = dict(scenario=scene, source_manifest_sha256=c.MANIFEST_SHA, roles={})
        for role, count in [('FIT', 4), ('SELECT', 2)]:
            normal['roles'][role] = [dict(scenario=scene, role=role, run_id=i,
                path=f'/fixture/{scene}/{role}/run{i:02d}_N.npy', shape=[24576,2],
                dtype='float64', allowed_in_training_loader=True, sha256='a'*64) for i in range(count)]
        cfg = dict(scenario=scene, checkpoint={'sha256':'b'*64}, scaler={'sha256':'c'*64},
                   training_report={'sha256':'d'*64}, normal_inputs={'sha256':'e'*64})
        acceptance = dict(status='PRIMARY_ACCEPTED_CATCH_C2_NORMAL_TRAIN', accepted_by='root',
                          scenario=scene, checkpoint_sha256='b'*64, scaler_sha256='c'*64,
                          training_report_sha256='d'*64, normal_inputs_sha256='e'*64,
                          source_manifest_sha256=c.MANIFEST_SHA)
        report = {'formal_data_hashes':normal}
        c.validate_training_scene(cfg, report, normal, acceptance)
        swapped = dict(cfg, scenario='WEAK')
        with self.assertRaises(PermissionError): c.validate_training_scene(swapped, report, normal, acceptance)
        for field, value in [('scenario','WEAK'), ('status','Completed'), ('accepted_by','executor'),
                             ('checkpoint_sha256','d'*64), ('scaler_sha256','d'*64)]:
            bad = dict(acceptance, **{field:value})
            with self.assertRaises(PermissionError): c.validate_training_scene(cfg, report, normal, bad)
        bad = copy.deepcopy(normal); bad['roles']['FIT'][0]['path']='/fixture/WEAK/FIT/run00_N.npy'
        with self.assertRaises(PermissionError): c.validate_training_scene(cfg, {'formal_data_hashes':bad}, bad, acceptance)

    def test_seed_pairs_and_identity(self):
        seeds = [c.bank_seed(s, r, i) for s in ('LIN', 'WEAK') for r in ('CAL', 'DEV') for i in range(8)]
        self.assertEqual(len(set(seeds)), 32)
        self.assertEqual(c.bank_seed('LIN', 'CAL', 0), 3025)
        self.assertEqual(c.bank_seed('WEAK', 'DEV', 7), 104032)
        for args in [('LIN', 'CONFIRM', 0), ('WEAK', 'DEV', True), ('LIN', 'DEV', 8)]:
            with self.assertRaises(PermissionError): c.bank_seed(*args)

    def test_actual_indices_and_availability(self):
        x = c.coordinates()
        np.testing.assert_array_equal(x['absolute_run_index'].reshape(-1), np.arange(24576))
        np.testing.assert_array_equal(x['endpoint'], np.arange(128)*192+191)
        np.testing.assert_array_equal(x['batch_input_end'], np.full(128, 24575))

    def test_cal_only_order_statistic_strict_budget(self):
        records = []
        for i in range(8):
            ref = dict(scenario='LIN', role='CAL', run_id=i, variant='N',
                       path=f'/fixture/LIN/CAL/run{i:02d}_N.npy', sha256='0'*64)
            scores = np.arange(128, dtype=np.float32)+i*128
            records.append(dict(reference=ref, last=scores, mean128=scores))
        result = c.calibrate_frozen(records, 'LIN')
        self.assertEqual(result['last']['threshold'], 1014.)
        self.assertEqual(np.count_nonzero(np.arange(1024)>result['last']['threshold']), 9)
        records[0]['reference']['role'] = 'DEV'
        with self.assertRaises(PermissionError): c.calibrate_frozen(records, 'LIN')


if __name__ == '__main__': unittest.main()
