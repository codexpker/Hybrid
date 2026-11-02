"""No Torch/models/public values/labels: suffix parser and injected raw records."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import event_query as q
from hybrid_tsad.hybrid_v1.data import Run,load_manifest,arrived_window
from hybrid_tsad.hybrid_v1.chunked_bank import IDENTITY_KEYS,read_entries


CAL=dict(score_kind='event_logit',q=.99,method='higher',strict_greater=True,threshold=1.)


def bindings():
    parent='c00f3e2c4c094d96572b945474712112ced433a13d82e9fce876b013ee990812'
    assets=dict(checkpoint=dict(sha256='b'*64),training_receipt=dict(sha256='c'*64),parent_source_checkpoint=dict(sha256=parent))
    cfg=dict(operation=q.READOUT_OPERATION,profile=q.PROFILE,score_kind='raw_event_logit',choice='K1_choice0',source_seed=2025,inference_seed=2025,history=512,horizon=32,assets=assets,model=dict(width=128))
    common=dict(choice='K1_choice0',source_checkpoint_sha256=parent,manifest_sha256='d'*64,plans_sha256='e'*64,support_sha256='f'*64)
    docs=dict(source_config=dict(seed=2025,model=cfg['model']),F_train_receipt=dict(common,status='COMPLETED',phase='target_event_train',loaded_roles=['NORMAL_SELECT','SUPPORT'],query_or_labels=False,adapted_checkpoint_sha256='b'*64),
        F_cal_config=dict(common,model=cfg['model'],adapted_checkpoint_sha256='b'*64,adapted_receipt_sha256='c'*64),F_cal_receipt=dict(common,status='COMPLETED',phase='target_event_calibrate',updates=0,loaded_roles=['NORMAL_CAL'],query_or_labels=False,score_kind='event_logit',threshold=1.),calibration=dict(CAL))
    seal=dict(status='ACCEPTED_SMD_EVENT_QUERY_ASSETS',approved_by='root',assets=assets,choice='K1_choice0',score_kind='raw_event_logit')
    return cfg,docs,seal


class EventQuery(unittest.TestCase):
    def test_fixed_original_geometry_and_local_seed(self):
        g=q.geometry();self.assertEqual(g['scored'],[17641,23694]);self.assertEqual(g['records'],6053);self.assertEqual(g['warmup'],[17129,17641])
        self.assertEqual(2025+(17641-17129),2537);self.assertEqual(2025+(23693-17129),8589)
        self.assertEqual(6053%64,37);self.assertEqual(g['tail_retained'],31)

    def test_prefix_is_not_numeric_input_and_missing_is_not_imputed(self):
        a,m=q.parse_query_rows([['unparseable prefix']]*7+[['1','nan'],['-0','3'],['4','5']],total_rows=10,start=7,end=10,channels=2)
        self.assertEqual(a.shape,(3,2));self.assertTrue(np.isnan(a[0,1]));self.assertFalse(m[0,1]);self.assertTrue(np.signbit(a[1,0]));self.assertEqual(a.dtype,np.float32)

    def test_bad_native_shape_count_and_inf_rejected(self):
        for rows,total in [([['1']]*5,5),([['1','2']]*4,5),([['1','inf']]*5,5)]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):q.parse_query_rows(rows,total_rows=total,start=2,end=5,channels=2)

    def test_builder_emits_only_independent_query_run(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'synthetic.csv';raw.write_text('not numeric prefix\n'*7+'1,2\n'*11)
            cfg=dict(raw_csv=dict(path=str(raw),sha256=q.sha(raw),expected_shape=[18,2]),query_bounds=[7,18],history=3,horizon=4,output=str(root/'derived'))
            result=q.prepare_inputs(cfg);self.assertFalse(result['prefix_numeric_converted']);self.assertFalse(result['labels_read'])
            runs=load_manifest(root/'derived/query_manifest.json',roles={'QUERY'});self.assertEqual(len(runs),1);run=runs[0]
            self.assertEqual(run.start,7);self.assertEqual(run.values.shape,(11,2));self.assertEqual(run.role,'QUERY')
            last=arrived_window(run,10,3,4);self.assertEqual(last['index'],17);self.assertEqual(last['footprint'],[14,18]);self.assertFalse(last['future_observed'][0,1:].any())
            with self.assertRaises(FileExistsError):q.prepare_inputs(cfg)

    def test_no_label_permission_and_preview_no_IO(self):
        with patch('pathlib.Path.open',side_effect=AssertionError('No IO')),contextlib.redirect_stdout(io.StringIO()):q.main(['--config','MISSING'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pending.json';path.write_text('{}')
            with self.assertRaises(PermissionError):q.input_admit(path,None,None)
        with self.assertRaises(PermissionError):q._no_annotation(dict(raw_csv=dict(label_path='forbidden')))

    def test_F_checkpoint_CAL_and_parent_binding_not_U_or_source(self):
        cfg,docs,seal=bindings();self.assertEqual(q.event_binding_metadata(cfg,docs,seal)['score_kind'],'raw_event_logit')
        for change in ('U','parent','choice','checkpoint','CALkind','normalplan','CALupdates','trainroles','seal'):
            c,d,s=copy.deepcopy((cfg,docs,seal))
            if change=='U':d['F_train_receipt']['phase']='target_train'
            elif change=='parent':d['F_cal_config']['source_checkpoint_sha256']='0'*64
            elif change=='choice':d['F_cal_receipt']['choice']='K3_choice3'
            elif change=='checkpoint':d['F_cal_config']['adapted_checkpoint_sha256']='0'*64
            elif change=='CALkind':d['calibration']['score_kind']='r_pred'
            elif change=='normalplan':d['F_cal_receipt']['plans_sha256']='0'*64
            elif change=='CALupdates':d['F_cal_receipt']['updates']=1
            elif change=='trainroles':d['F_train_receipt']['loaded_roles']=['QUERY']
            else:s['status']='PENDING'
            with self.subTest(change=change),self.assertRaises((ValueError,PermissionError)):q.event_binding_metadata(c,d,s)

    def test_continuous_logit_NA_and_strict_not_binary_score(self):
        raw=dict(event_features=[[1.,2.,3.]],event_logit=1.,prediction=[[-0.]],alarm=True)
        out=q.event_scored_record(raw,1.,3);self.assertEqual(out['score'],1.);self.assertFalse(out['alarm']);self.assertTrue(out['valid'])
        bad=q.event_scored_record(dict(raw,event_features=[[np.nan,2.,3.]]),1.,3)
        self.assertFalse(bad['valid']);self.assertTrue(np.isnan(bad['score']));self.assertEqual(bad['reason'],'NONFINITE_EVENT_FEATURES');self.assertNotIn('alarm',bad)
        bad=q.event_scored_record(dict(raw,event_logit=float('inf')),1.,3);self.assertEqual(bad['reason'],'NONFINITE_EVENT_LOGIT')
        self.assertEqual(raw['alarm'],True)

    def test_full_6053_tail37_local_mask_and_chunk_confirm_state(self):
        run=Run('SMD','machine-2-1','machine-2-1_test','QUERY',np.ones((6565,3),np.float32),np.ones((6565,3),bool),start=17129)
        def records(model,runs,**kwargs):
            self.assertTrue(kwargs['event_logits']);self.assertEqual(kwargs['seed'],2025)
            self.assertEqual(kwargs['active_components'],['r_tf','r_ms','r_pred'])
            for t in range(512,len(run.values)):
                yield dict(domain=run.domain,entity=run.entity,run=run.run,index=run.start+t,available_index=run.start+t,event_logit=2.,event_features=[[float('nan') if t==576 else .1]*5],prediction=[[-0.,1.,2.]],replay=dict(actual_mask_seed=2025+t))
        with tempfile.TemporaryDirectory() as d:
            bank=Path(d)/'bank';q.write_event_bank(None,[run],CAL,bank,identity={k:'a'*64 for k in IDENTITY_KEYS},feature_dim=5,infer_fn=records)
            m=json.loads((bank/'manifest.json').read_text());rows=list(read_entries(bank))
            self.assertTrue(m['complete']);self.assertEqual(m['committed_records'],6053);self.assertEqual(len(m['chunks']),95);self.assertEqual(m['chunks'][-1]['records'],37)
            self.assertEqual(rows[0]['raw']['index'],17641);self.assertEqual(rows[-1]['raw']['index'],23693);self.assertEqual(rows[-1]['raw']['replay']['actual_mask_seed'],8589)
            self.assertTrue(rows[63]['derived']['confirmed3_current']);self.assertFalse(rows[64]['derived']['strict_alarm']);self.assertFalse(rows[66]['derived']['confirmed3_current']);self.assertTrue(rows[67]['derived']['confirmed3_current'])
            self.assertEqual([r['raw']['index'] for r in rows[-31:]],list(range(23663,23694)))

    def test_global_mask_clock_and_residual_CAL_rejected(self):
        run=Run('SMD','machine-2-1','machine-2-1_test','QUERY',np.ones((5,1),np.float32),np.ones((5,1),bool),start=100)
        def wrong(model,runs,**kwargs):
            yield dict(domain='SMD',entity=run.entity,run=run.run,index=103,available_index=103,event_logit=2.,event_features=[[1.]],replay=dict(actual_mask_seed=2025+103))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):q.write_event_bank(None,[run],CAL,Path(d)/'wrong',identity={k:'a'*64 for k in IDENTITY_KEYS},history=3,horizon=2,feature_dim=1,infer_fn=wrong)
            with self.assertRaises(ValueError):q.write_event_bank(None,[run],dict(CAL,score_kind='r_pred'),Path(d)/'residual',identity={k:'a'*64 for k in IDENTITY_KEYS},history=3,horizon=2,infer_fn=wrong)

    def test_cold_import_has_no_Torch(self):
        src=str(Path(q.__file__).resolve().parents[2]);code='import sys;sys.path.insert(0,'+repr(src)+');import hybrid_tsad.hybrid_v1.event_query;assert "torch" not in sys.modules;print("COLD_NUMPY_NO_TORCH")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('COLD_NUMPY_NO_TORCH',result.stdout)


if __name__=='__main__':unittest.main()
