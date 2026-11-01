import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from hybrid_tsad.normal_b1_readout import collect_bank, checkpoint_key_contract, save_bank, score_identity, preview
from hybrid_tsad.normal_b1_readout import require_runtime_release,bind_checkpoint_schema,execute_cost
from hybrid_tsad.normal_b1_readout import validate_remote_paths


class B1ReadoutPreparation(unittest.TestCase):
    def metadata(self):
        return dict(domain='TEP_REACTOR', role='SELECT', run_id=61, fault_number=0,
                    columns=['xmeas_9','xmeas_21','xmv_10'], seed=2025, window=100,
                    batch_size=1, points=500, array_sha256='a'*64, scaler_sha256='b'*64,
                    checkpoint_sha256='c'*64, feature_source_sha256='d'*64, best_checkpoint_step=376)

    def state(self):
        return dict(state='fixed-science-state', gradients='none', modes={'':False}, step=376,
                    rng={'python':'fixed','numpy':'fixed','torch_cpu':'not-loaded','cuda':'not-loaded'})

    def values(self):
        return np.arange(1500, dtype=np.float64).reshape(500,3) / 1500.

    def test_all401_causal_windows_and_fixed_readouts(self):
        calls=[]; snaps=[]
        def scorer(batch):
            self.assertEqual(batch.shape,(1,100,3)); calls.append(batch.copy()); return batch**2
        def fingerprint():
            snaps.append(1); return self.state()
        bank, report=collect_bank(self.values(),self.metadata(),scorer,fingerprint)
        self.assertEqual((len(calls),len(snaps)),(401,2))
        self.assertEqual(bank['raw_scores'].shape,(401,100,3))
        np.testing.assert_array_equal(bank['end_index_0based'],np.arange(99,500))
        np.testing.assert_array_equal(bank['available_index_0based'],bank['end_index_0based'])
        for i in [0,1,399,400]:
            np.testing.assert_array_equal(calls[i][0],self.values()[i:i+100].astype(np.float32))
        np.testing.assert_array_equal(bank['last'],bank['raw_scores'].mean(2)[:,-1])
        np.testing.assert_array_equal(bank['mean32'],bank['raw_scores'].mean(2)[:,-32:].mean(1))
        self.assertEqual(report['steady_calls'],400)

    def test_wrong_role_or_batch_refused_before_scorer(self):
        for key, value in [('role','CAL'),('run_id',62),('batch_size',128),('fault_number',1)]:
            meta=self.metadata(); meta[key]=value
            with self.assertRaises(PermissionError): collect_bank(self.values(),meta,None,None)

    def test_rawscore_wrongshape_dtype_or_nonfinite_refused(self):
        for value in [np.zeros((100,3),np.float32),np.zeros((1,100,3),np.float64),np.full((1,100,3),np.nan,np.float32)]:
            with self.assertRaises(ValueError): collect_bank(self.values(),self.metadata(),lambda b:value,self.state)

    def test_fingerprint_state_and_rng_changes_rejected(self):
        for key,value in [('step',377),('rng',{'numpy':'changed'}),('state','changed')]:
            state=self.state(); count=[0]
            def scorer(batch): state[key]=value; return batch.copy()
            with self.assertRaises(RuntimeError): collect_bank(self.values(),self.metadata(),scorer,lambda:state)

    def test_incomplete_fingerprint_refused(self):
        with self.assertRaises(ValueError): collect_bank(self.values(),self.metadata(),None,lambda:{'step':1})

    def test_scorer_cannot_mutate_its_input(self):
        def mutating(batch):batch[:]=0;return batch
        with self.assertRaises(RuntimeError):collect_bank(self.values(),self.metadata(),mutating,self.state)

    def test_checkpoint_contract_exact_names_shapes_dtype(self):
        science={'weight':{'shape':[3,3],'dtype':'torch.float32'}}
        extras={'linear.total_ops':{'shape':[1],'dtype':'torch.float64'}}
        actual=dict(science,**extras)
        self.assertEqual(checkpoint_key_contract(actual,science,extras),('weight',))
        for bad in [dict(actual,unknown={'shape':[],'dtype':'torch.int64'}),science,{'weight':{'shape':[1],'dtype':'torch.float32'},**extras}]:
            with self.assertRaises(ValueError):checkpoint_key_contract(bad,science,extras)

    def test_save_keeps_raw_continuous_scores_and_cannot_overwrite(self):
        bank,report=collect_bank(self.values(),self.metadata(),lambda b:b.copy(),self.state)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bank'; result=save_bank(path,bank,report)
            self.assertFalse(result['complete_pipeline_timing_available'])
            with np.load(str(path/'raw_bank.npz'),allow_pickle=False) as z:
                np.testing.assert_array_equal(z['raw_scores'],bank['raw_scores'])
                self.assertIn('end_index_0based',z.files)
            with self.assertRaises(FileExistsError):save_bank(path,bank,report)

    def test_preview_never_numpy_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cfg.json';p.write_text(json.dumps({'operation':'NORMAL_B1_READOUT_COST_PREPARATION','status':'PENDING'}))
            with patch('numpy.load',side_effect=AssertionError('Forbidden data IO')) as load:
                self.assertFalse(preview(p)['allowed_real_execution']);load.assert_not_called()

    def test_pending_runtime_refuses_before_release_file(self):
        with patch('hybrid_tsad.normal_b1_readout.read_json',side_effect=AssertionError('Do not open release')) as load:
            with self.assertRaises(PermissionError):require_runtime_release({'operation':'PENDING'},'missing_cfg','missing_release')
            load.assert_not_called()

    def test_actual_basic_schema_whitelists_only_exact_extras(self):
        before={'w':'hash1'};after=dict(before,**{'linear.total_ops':'hash2'})
        e=[{'kind':'official_basic','before':{'step':0,'state':before},'after':{'step':0,'state':after}}]
        schema={'w':{'shape':[2],'dtype':'torch.float32'},'linear.total_ops':{'shape':[1],'dtype':'torch.float64'}}
        report={'initial_model_state_and_RNG':{'state':before}}
        science,extras=bind_checkpoint_schema(e,schema,report)
        self.assertEqual(set(science),{'w'});self.assertEqual(set(extras),{'linear.total_ops'})
        bad=copy.deepcopy(e);bad[0]['after']['state']['w']='changed'
        with self.assertRaises(ValueError):bind_checkpoint_schema(bad,schema,report)
        bad=copy.deepcopy(schema);bad['linear.total_ops']['dtype']='torch.float32'
        with self.assertRaises(ValueError):bind_checkpoint_schema(e,bad,report)

    def test_wrong_exact_release_refuses_before_runtime_references(self):
        cfg=dict(operation='LEFT_TEP_B1_NORMAL_COST',binding_state='EXACT_REFERENCES_BOUND',
            seed=2025,max_seconds=180,max_allocated_GiB=20,physical_gpu_index=0,
            expected_gpu_uuid='895053ba-a4d6-49a6-d831-865d5982f009',
            checkpoint_initialization='accepted_normal_best',batch_size=1,window=100,
            run_id=61,infer_calls=401,adapter_hashes={})
        cfg.update(output='/home/a312/tsad_research/results/left_tep_b1_normal_cost_seed2025_20261101_v1',
                   source_root='/home/a312/tsad_research/preparations/upstream')
        for key in ('training_acceptance','normal_source_acceptance','contract','training_report','training_events',
                    'checkpoint_schema','scaler','normal_metadata','array','checkpoint'):
            cfg[key]={'path':'/home/a312/tsad_research/results/fixed.json'}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';r=Path(tmp)/'release.json'
            p.write_text(json.dumps(cfg));r.write_text(json.dumps({'status':'PENDING_ROOT_RELEASE'}))
            with patch('hybrid_tsad.normal_b1_readout.verified_ref',side_effect=AssertionError('Forbidden ref IO')) as refs:
                with self.assertRaises(PermissionError):execute_cost(p,r)
                refs.assert_not_called()

    def test_remote_paths_absolute_posix_output_and_refs(self):
        cfg=dict(output='/home/a312/tsad_research/results/left_tep_b1_normal_cost_seed2025_20261101_v1',
                 source_root='/home/a312/tsad_research/preparations/upstream')
        for key in ('training_acceptance','normal_source_acceptance','contract','training_report','training_events',
                    'checkpoint_schema','scaler','normal_metadata','array','checkpoint'):
            cfg[key]={'path':'/home/a312/tsad_research/results/fixed.json'}
        validate_remote_paths(cfg)
        for path in ['\\home\\a312\\tsad_research\\results\\bad','results/relative','/home/a312/tsad_research/../bad']:
            bad=copy.deepcopy(cfg);bad['array']['path']=path
            with self.assertRaises(PermissionError):validate_remote_paths(bad)
        bad=copy.deepcopy(cfg);bad['output']=bad['output'].replace('/','\\')
        with self.assertRaises(PermissionError):validate_remote_paths(bad)


if __name__=='__main__':unittest.main()
