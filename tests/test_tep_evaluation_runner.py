"""Temporary synthetic JSON/compact gate tests; never touch project score assets."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from hybrid_tsad.engine import tep_evaluation_runner as runner


class SyntheticFiles:
    def __init__(self, root, mutation=None):
        self.root = Path(root)
        self.identity = self.root / 'identity.json'
        records = [dict(entity_id=entity, faultNumber=i // 20 + 1, simulationRun=i % 20 + 51,
                        nominal_first_positive_sample_1based=161, sampling_minutes=3)
                   for i, entity in enumerate(runner.ENTITIES)]
        identity = self.put('identity.json', dict(status='WORKFLOW_SEALED_NOT_CRYPTOGRAPHIC_OR_BLIND', records=records))
        inputs = self.put('inputs.json', dict(status='PUBLIC_SIMULATION_INPUTS_NO_LABEL_FEATURES',
                         records=[dict(entity_id=e) for e in runner.ENTITIES]))
        acceptance = self.put('input_acceptance.json', dict(status='PRIMARY_ACCEPTED_LIMITED_TEP_FAULTY_INPUTS',
            approved_by='root', entities=400, input_manifest_sha256=inputs['sha256'],
            sealed_identity_manifest_sha256=identity['sha256']))
        self.models = []
        for index, seed in enumerate([2025, 2026, 2027, None]):
            left = seed is not None
            directory = self.root / str(index); directory.mkdir()
            keys = list(runner.core.READOUTS) if left else [m + '_' + d for m in runner.core.METHODS[1:] for d in runner.core.READOUTS]
            arrays = dict(entity_id=np.asarray(runner.ENTITIES), end_index_0based=np.tile(np.arange(99, 960), (400, 1)))
            arrays.update({key: np.zeros((400, 861)) for key in keys})
            if left:
                arrays.update(channel_last=np.zeros((400, 861, 3)), channel_mean32=np.zeros((400, 861, 3)),
                              available_index_0based=arrays['end_index_0based'])
            compact_path = directory / 'compact_scores.npz'
            np.savez_compressed(compact_path, **arrays)
            compact = self.reference(compact_path)
            scalar_hash = runner.sha_bytes(('synthetic-scaler-%d' % index).encode())
            parameter_hash = runner.sha_bytes(('synthetic-parameter-%d' % index).encode())
            cal = dict(n=8020, rank_1based=7941, comparison='strict_greater', scaler_sha256=scalar_hash,
                       thresholds={key: 1. for key in keys})
            cal.update(dict(seed=seed, checkpoint_sha256=parameter_hash) if left else dict(coefficient_sha256=parameter_hash))
            if mutation == 'wrong_cal_seed' and index == 1:
                cal['seed'] = 2025
            if mutation == 'wrong_threshold' and index == 3:
                cal['thresholds']['MULTI_last'] = float('inf')
            cal_ref = self.put(str(index) + '/calibration.json', cal)
            report = dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW', seed=seed,
                          banks=[dict(entity_id=e) for e in runner.ENTITIES])
            if left:
                report['compact_scores_sha256'] = compact['sha256']
            else:
                report.update(compact_sha256=compact['sha256'], calibration_thresholds=cal['thresholds'],
                    source_refs=dict(coefficients=dict(sha256=parameter_hash), scaler=dict(sha256=scalar_hash),
                                     calibration=cal_ref, input_manifest=inputs))
            report_ref = self.put(str(index) + '/report.json', report)
            manifest_ref = self.put(str(index) + '/FINAL_MANIFEST.json', dict(files=[
                dict(path='report.json', sha256=report_ref['sha256']),
                dict(path='compact_scores.npz', sha256=compact['sha256'])]))
            accepted = dict(status='PRIMARY_ACCEPTED_LIMITED_LEFT_TEP_FIXED_QUERY_SCORES' if left else 'PRIMARY_ACCEPTED_LIMITED_TEP_RIDGE_FIXED_QUERY_SCORES',
                approved_by='root', entities=400, raw_bank_count=400, scored_points_per_entity=861, endpoints_0based=[99, 959],
                source_manifest_sha256=manifest_ref['sha256'], report_sha256=report_ref['sha256'],
                compact_scores_sha256=compact['sha256'], calibration_sha256=cal_ref['sha256'], input_manifest_sha256=inputs['sha256'])
            if left:
                accepted.update(seed=seed, checkpoint_sha256=parameter_hash, scaler_sha256=scalar_hash)
            else:
                accepted.update(methods=list(runner.core.METHODS[1:]), readouts=list(runner.core.READOUTS))
            if mutation == 'wrong_acceptance_binding' and index == 2:
                accepted['calibration_sha256'] = '0' * 64
            accepted_ref = self.put(str(index) + '/acceptance.json', accepted)
            model = dict(family='LEFT' if left else 'RIDGE_ZMAX', seed=seed, scaler_sha256=scalar_hash,
                score_acceptance=accepted_ref, score_manifest=manifest_ref, score_report=report_ref,
                compact_scores=compact, calibration=cal_ref)
            model['checkpoint_sha256' if left else 'coefficients_sha256'] = parameter_hash
            self.models.append(model)
        if mutation == 'missing_left':
            self.models.pop(1)
        if mutation == 'wrong_model_seed':
            self.models[1]['seed'] = 2025
        if mutation == 'float_seed':
            self.models[1]['seed'] = 2026.
        seal = self.put('seal.json', dict(status='PRIMARY_SEALED_ALL_TEP_FIXED_QUERY_SCORES', approved_by='root',
            operation=runner.OP, cases=400, endpoints_0based=[99, 959], points_per_entity=861,
            input_acceptance=acceptance, input_manifest=inputs, identity_manifest=identity, score_models=self.models))
        cfg = dict(status='BOUND_TEP_FIXED_SCORE_EVALUATION', operation=runner.OP, score_seal=seal,
                   output=str(self.root / 'evaluation'))
        if mutation == 'pending':
            cfg.update(status='PENDING_ALL_SCORE_ACCEPTANCE', score_seal=None)
        self.cfg = self.put('config.json', cfg)
        hashes = runner.source_hashes()
        if mutation == 'wrong_source':
            hashes['events'] = '0' * 64
        release = dict(status='PRIMARY_RELEASED_TEP_FIXED_SCORE_EVALUATION', approved_by='root', operation=runner.OP,
                       allow_labels=True, allow_model_or_fit=False, config=self.cfg, score_seal=seal, source_hashes=hashes)
        if mutation == 'fake_signature':
            release.update(status='RELEASED', signature='root')
        self.release = self.put('release.json', release)
        self.anchor = self.release['sha256']  # trusted only inside this temporary fixture

    def reference(self, path):
        return dict(path=str(path.resolve()), sha256=runner.sha_bytes(path.read_bytes()))

    def put(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data), encoding='utf-8')
        return self.reference(path)


class FileGateTests(unittest.TestCase):
    def test_synthetic_end_to_end_metadata_payload_spine(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticFiles(tmp)
            state = runner.verify_metadata(fixture.cfg, fixture.release, fixture.anchor, tmp)
            self.assertEqual(len(state['calibrations']), 4)
            state, identities, banks = runner.load_verified_payload(fixture.cfg, fixture.release, fixture.anchor, tmp)
            self.assertEqual(len(identities), 400)
            self.assertEqual(len(banks), 4)
            self.assertEqual(banks[0]['last'].shape, (400, 861))

    def test_missing_seed_threshold_source_pending_fake_refuse_before_identity_compact(self):
        for mutation in ['missing_left', 'wrong_model_seed', 'wrong_cal_seed', 'wrong_acceptance_binding',
                         'wrong_threshold', 'wrong_source', 'pending', 'fake_signature', 'float_seed']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                fixture = SyntheticFiles(tmp, mutation)
                touched = []
                original = runner._bytes
                def observe(ref, workspace, suffix='.json'):
                    touched.append(ref['path'])
                    return original(ref, workspace, suffix)
                with patch.object(runner, '_bytes', side_effect=observe):
                    with self.assertRaises((ValueError, PermissionError)):
                        runner.load_verified_payload(fixture.cfg, fixture.release, fixture.anchor, tmp)
                self.assertNotIn(str(fixture.identity), touched)
                self.assertFalse(any(p.endswith('.npz') for p in touched))

    def test_untrusted_release_and_tampered_bytes_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = SyntheticFiles(tmp)
            with self.assertRaises(PermissionError):
                runner.verify_metadata(fixture.cfg, fixture.release, '0' * 64, tmp)
            Path(fixture.release['path']).write_text('{"approved_by":"root"}', encoding='utf-8')
            with self.assertRaises(PermissionError):
                runner.verify_metadata(fixture.cfg, fixture.release, fixture.anchor, tmp)

    def test_raw_paths_relative_paths_and_escape_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'weights.npz'; raw.write_bytes(b'not-a-real-model')
            ref = dict(path=str(raw), sha256=runner.sha_bytes(raw.read_bytes()))
            with self.assertRaises(PermissionError):
                runner._bytes(ref, tmp)
            with self.assertRaises(PermissionError):
                runner._bytes(ref, tmp, '.npz')
            with self.assertRaises(PermissionError):
                runner._ref(dict(path='relative.json', sha256='0' * 64), tmp)
            with tempfile.TemporaryDirectory() as elsewhere:
                outside = Path(elsewhere) / 'metadata.json'; outside.write_text('{}')
                with self.assertRaises(PermissionError):
                    runner._ref(dict(path=str(outside), sha256='0' * 64), tmp)


if __name__ == '__main__':
    unittest.main()
