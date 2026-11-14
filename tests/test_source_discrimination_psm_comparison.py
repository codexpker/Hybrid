"""Small CPU-only wiring fixtures: no production arrays, model or labels."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1.source_discrimination_psm_comparison import (
    admission, crop, exact_fields, corrected_operating_columns, INITIAL, PLAN, RANGES, OLD_SOURCE_CAL)
from hybrid_tsad.hybrid_v1 import source_discrimination_psm_comparison as consumer


class Wiring(unittest.TestCase):
    def test_production_receipt_chain_and_fixed_aux(self):
        asset=lambda name:{'path':name,'sha256':name}
        candidate={k:asset(k) for k in ('initial','source_plan','source_CAL_records','training_receipt',
            'calibration_receipt','query_receipt','training_config','original_source_config','fit_units',
            'checkpoint','training_report','bank_manifest','source_CAL1024')}
        candidate['initial']['sha256']=INITIAL;candidate['source_plan']['sha256']=PLAN
        identity=dict(input_sha256='input',config_sha256='config',checkpoint_sha256='checkpoint',
                      calibration_sha256='source_CAL1024',source_sha256='source')
        candidate.update(expected_five_identity=identity,expected_query_scientific_status='MERGED_ACTUAL')
        common=dict(status='COMPLETED',seed=2025,checkpoint_sha256='checkpoint',labels_read=False,
                    units_sha256='fit_units',module_sha256='module')
        payload={
            'science':dict(baseline=dict(initial=INITIAL,plan=PLAN,seed=2025,normal_updates=8000),
                           loss={'lambda':.1,'margin':.1,'additional_parameters':0,'no_search':True}),
            'training_receipt':dict(common,phase='train',scientific_status='VALID_SOURCE_SELECTED_ENDPOINT',
                attempts=8000,updates=8000,initial_sha256=INITIAL,selected_attempt=6000,config_sha256='training_config'),
            'calibration_receipt':dict(common,phase='cal',scientific_status='VALID_SOURCE_CAL',records=1024,
                                      calibration_sha256='source_CAL1024'),
            'query_receipt':dict(common,phase='query',scientific_status='MERGED_ACTUAL',points=87329,
                                bank_manifest_sha256='bank_manifest',calibration_sha256='source_CAL1024'),
            'training_config':dict(initial={'sha256':INITIAL},original_config={'sha256':'original_source_config'},units={'sha256':'fit_units'}),
            'training_report':dict(attempts=8000,updates=8000,selected_attempt=6000,steps=[{}]*8000),
            'bank_manifest':dict(complete=True,committed_records=87329,contract=dict(ranges=RANGES,threshold=2.,identity=identity)),
            'source_CAL1024':dict(valid_windows=1024,domains=['SMD','Sunspot'],source_domain_equal_weight=True,
                                 weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),threshold=2.),
            'seal':dict(labels_read=False,bank_manifest_sha256='bank_manifest',checkpoint_sha256='checkpoint',calibration_sha256='source_CAL1024'),
            'gate':{'mode_rng_gradient_PASS':True}}
        payload['old_source_cal']=payload['source_CAL1024']
        config=dict(status='FROZEN_READY',seed=2025,common_axis=[512,87744],source_sha256={},science_freeze=asset('science'),candidate=candidate,
            expected_shared_receipt_fields=dict(units_sha256='fit_units',module_sha256='module'),resource_receipts=[asset('os')],
            required_evidence=[dict(asset=asset('gate'),expected_fields={'mode_rng_gradient_PASS':True})],whole_raw_seal=asset('seal'),
            baselines=[dict(id=name) for name in ('HybridV1_Z_source2025','HistoryMean512_ZS','TimeRadar_official_native','LEFT_R_targettrained')])
        config['baselines'][0]['threshold']=2.
        config['fixed_auxiliary_diagnostics']=dict(enabled=True,components=['r_tf','r_ms','r_pred'],selection_allowed=False,
            change_primary_scores=False,axis=[512,87744],original_source_calibration=dict(path='old_source_cal',sha256=OLD_SOURCE_CAL))
        with patch.object(consumer,'read',side_effect=lambda a:payload[a['path']]),patch.object(consumer,'bound',side_effect=lambda a:Path(a['path'])):
            self.assertEqual(admission(config)['status'],'SINGLE_CANDIDATE_ADMITTED')
            payload['query_receipt']['scientific_status']='COMPLETE_SHARD_PENDING_RAW_MERGE'
            with self.assertRaises(ValueError):admission(config)
            payload['query_receipt']['scientific_status']='MERGED_ACTUAL'
            payload['science']['loss']['lambda']=.2
            with self.assertRaises(ValueError):admission(config)

    def test_pending_fails_before_truth_or_candidate_io(self):
        with self.assertRaises(PermissionError):
            admission(dict(status='PENDING_NOT_EXECUTABLE', truth={'path':'must_not_be_read'}))

    def test_full_axis_no_selected_intersection(self):
        data=dict(index=np.arange(512,520),score=np.arange(8.),valid=np.ones(8,bool))
        np.testing.assert_array_equal(crop(data,np.arange(513,519)),np.arange(1,7))
        with self.assertRaises(ValueError):crop(data,np.arange(513,521))
        data['valid'][4]=False
        with self.assertRaises(ValueError):crop(data,np.arange(513,519))

    def test_actual_receipt_status_is_bound_not_guessed(self):
        exact_fields({'scientific_status':'ACTUAL_MERGED_STATUS'},{'scientific_status':'ACTUAL_MERGED_STATUS'})
        with self.assertRaises(ValueError):
            exact_fields({'scientific_status':'COMPLETE_SHARD_PENDING_RAW_MERGE'}, {'scientific_status':'ACTUAL_MERGED_STATUS'})

    def test_unknown_steady_not_a_miss(self):
        phases=[]
        for stream in ('strict_point','confirmed3_current'):
            for eligible,scored,unknown,detected in ((False,0,1000,0),(True,1000,0,0),(True,1000,0,2)):
                phases.append(dict(stream=stream,phase='steady_after512',duration=1600,
                    eligible=eligible,scored_points=scored,unknown_points=unknown,detected_points=detected))
        result={'streams':{s:{'affiliation':{'global_result':{'Aff_F1':None,'NA_reason':'fixture'}}}
                           for s in ('strict_point','confirmed3_current')}}
        row=corrected_operating_columns(lambda r,p:{'unchanged':7},result,phases)
        self.assertEqual(row['unchanged'],7)
        self.assertEqual(row['strict_steady_all_zero_detected_events'],1)
        self.assertEqual(row['confirmed3_steady_main_zero_detected_events'],1)
        self.assertIsNone(row['strict_Aff_F1'])


if __name__=='__main__':unittest.main(verbosity=2)
