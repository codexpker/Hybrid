"""CPU metadata/metric-wiring tests; no production scores or numerical truth."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hybrid_tsad.hybrid_v1 import source_candidate_psm_comparison as c


class SourceCandidate(unittest.TestCase):
    def fixture(self):
        asset = lambda name: dict(path=name, sha256=name)
        d = {k:asset(k) for k in ('initial','source_plan','original_source_config','source_CAL_records',
            'implementation','checkpoint','checkpoint_metadata','training_receipt','calibration_receipt',
            'query_receipt','training_config','training_report','bank_manifest','source_CAL1024')}
        d['initial']['sha256']=c.INITIAL; d['source_plan']['sha256']=c.PLAN
        identity = dict(input_sha256='input', config_sha256='config', source_sha256='source',
            checkpoint_sha256='checkpoint', calibration_sha256='source_CAL1024')
        d['expected_five_identity']=identity
        shared=dict(architecture=c.ARCHITECTURE, implementation_sha256='implementation')
        common=dict(shared, status='COMPLETED', seed=2025, checkpoint_sha256='checkpoint', labels_read=False)
        cal=dict(valid_windows=1024, domains=['SMD','Sunspot'], source_domain_equal_weight=True,
            weights=dict(r_tf=.25,r_ms=.25,r_pred=.5), active_components=['r_tf','r_ms','r_pred'], threshold=2.)
        payload=dict(
            science=dict(candidate='Hybrid_T_anchor_crossview_completion_v1',new_network_parameters=0),
            checkpoint_metadata=dict(shared,checkpoint_sha256='checkpoint',same_original_parameter_keys=True,
                no_extra_trainable_parameters=True,fixed_query='standard_sinusoidal_position_0_to_Nminus1_per_branch'),
            training_receipt=dict(common,phase='train',scientific_status='VALID_SOURCE_SELECTED_ENDPOINT',
                attempts=8000,updates=8000,initial_sha256=c.INITIAL,initial_tensors_exact=True,
                config_sha256='training_config',selected_attempt=6000),
            calibration_receipt=dict(common,phase='cal',scientific_status='VALID_SOURCE_CAL',records=1024,calibration_sha256='source_CAL1024'),
            query_receipt=dict(common,phase='query',scientific_status='COMPLETE_PENDING_FIXED_EVALUATION',points=87329,
                calibration_sha256='source_CAL1024',bank_manifest_sha256='bank_manifest'),
            training_config=dict(operation='HYBRID_T_ANCHOR_COMPLETION',phase='train',seed=2025,
                initial=d['initial'],original_config=d['original_source_config'],immutable_sha256={}),
            original_source_config=dict(plans='plan'),
            training_report=dict(attempts=8000,updates=8000,selected_attempt=6000,steps=[{}]*8000),
            bank_manifest=dict(complete=True,committed_records=87329,contract=dict(ranges=c.RANGES,threshold=2.,identity=identity)),
            source_CAL1024=cal,oldcal=cal,
            seal=dict(shared,labels_read=False,bank_manifest_sha256='bank_manifest',checkpoint_sha256='checkpoint',calibration_sha256='source_CAL1024'),
            gate=dict(PASS=True))
        config=dict(status='FROZEN_READY',seed=2025,common_axis=[512,87744],full_online_axis=[512,87841],original_length=87841,
            source_sha256={'consumer':'hash'},science_freeze=asset('science'),candidate=d,
            expected_shared_receipt_fields=shared,expected_training_config_fields={},
            baselines=[dict(id=name,online=i<2,threshold=2.) for i,name in enumerate(c.BASELINES)],
            original_source_calibration=dict(path='oldcal',sha256=c.OLD_SOURCE_CAL),
            resource_receipts=[asset('os')],required_evidence=[dict(asset=asset('gate'),expected_fields=dict(PASS=True))],
            whole_raw_seal=asset('seal'))
        return config,payload

    def test_completed_chain_and_wrong_architecture_checkpoint_rejected(self):
        cfg,payload=self.fixture()
        with patch.object(c,'read',side_effect=lambda a:payload[a['path']]),patch.object(c,'bound',side_effect=lambda a:Path(a['path'])),patch.object(c,'sha',return_value='hash'):
            self.assertEqual(c.admission(cfg)['status'],'SINGLE_STRUCTURAL_CANDIDATE_ADMITTED')
            payload['checkpoint_metadata']['architecture']='HybridV1'
            with self.assertRaises(ValueError): c.admission(cfg)

    def test_shard_and_label_exposure_rejected(self):
        cfg,payload=self.fixture()
        with patch.object(c,'read',side_effect=lambda a:payload[a['path']]),patch.object(c,'bound',side_effect=lambda a:Path(a['path'])),patch.object(c,'sha',return_value='hash'):
            payload['query_receipt']['scientific_status']='COMPLETE_SHARD_PENDING_RAW_MERGE'
            with self.assertRaises(ValueError): c.admission(cfg)
            payload['query_receipt']['scientific_status']='COMPLETE_PENDING_FIXED_EVALUATION'
            payload['seal']['labels_read']=True
            with self.assertRaises(ValueError): c.admission(cfg)

    def test_no_hinge_or_attribution_gate(self):
        cfg,_=self.fixture();cfg['fixed_auxiliary_diagnostics']={}
        with patch.object(c,'sha',return_value='hash'),patch.object(c,'read') as reader:
            with self.assertRaises(PermissionError): c.admission(cfg)
            reader.assert_not_called()

    def test_fixed_axis_and_canonical_ranking_preserved_offline(self):
        axis=np.arange(512,518);score=np.array([.1,.9,.2,.8,.3,.7])
        data=dict(index=axis,score=score,valid=np.ones(6,bool),available=axis+100)
        labels=np.zeros(520,int);labels[axis[1::2]]=1
        desc=dict(id='offline',online=False,protocol='original',role_permissions='offline')
        with tempfile.TemporaryDirectory() as tmp:
            row,phases=c._evaluate(data,axis,labels,desc,Path(tmp)/'offline',{},None)
            self.assertEqual(row['AP'],1.);self.assertEqual(phases,[])
            with np.load(Path(tmp)/'offline/canonical_inputs.npz') as out:
                np.testing.assert_array_equal(out['score'],score)
                np.testing.assert_array_equal(out['available_index'],axis+100)
        with self.assertRaises(ValueError):c.crop(data,np.arange(512,519))

    def test_online_delegates_original_scores_and_unknown_event_kernel(self):
        from hybrid_tsad.hybrid_v1 import evaluation, public_native_evaluation
        axis=np.arange(512,518);score=np.array([.1,.9,.2,.8,.3,.7])
        labels=np.zeros(87841,int);labels[axis[1::2]]=1
        data=dict(index=axis,score=score,valid=np.ones(6,bool),available_index=axis)
        desc=dict(id='candidate',online=True,threshold=.5,protocol='fixed',role_permissions='source')
        result={'streams':{s:{'affiliation':{'global_result':dict(Aff_P=.3,Aff_R=.4,Aff_F1=.34)}}
            for s in ('strict_point','confirmed3_current')}}
        with tempfile.TemporaryDirectory() as tmp, patch.object(evaluation,'evaluate_arrays',return_value=result) as kernel, \
                patch.object(evaluation,'save_evaluation'), patch.object(public_native_evaluation,'event_diagnostics',return_value=[]), \
                patch('hybrid_tsad.hybrid_v1.controlled_wave_evaluation.reject_dependency_na'),patch.object(c,'bound',return_value=Path('official')):
            row,_=c._evaluate(data,axis,labels,desc,Path(tmp)/'new',dict(affiliation_identity={}),lambda r,p:{})
            self.assertEqual(row['strict_Aff_F1'],.34)
            np.testing.assert_array_equal(kernel.call_args.kwargs['score'],score)
            np.testing.assert_array_equal(kernel.call_args.kwargs['full_labels'],labels)
            self.assertEqual(kernel.call_args.kwargs['threshold'],.5)
            self.assertEqual(kernel.call_count,1)


if __name__=='__main__':unittest.main()
