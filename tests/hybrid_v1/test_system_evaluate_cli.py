"""Synthetic seven-bank W1 tests; no real scores/labels/models/GPU."""
import copy
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import system_evaluate_cli as s,named_evaluate_cli as old
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,IDENTITY_KEYS
from hybrid_tsad.hybrid_v1.scoring import recipe,apply_calibration


def seal(cfg):
    path=Path(cfg['raw_seal']['path']);path.write_text(json.dumps(dict(status='ACCEPTED_SYSTEM_W1_FIXED_BANKS',approved_by='root',profile=cfg['profile'],banks=cfg['banks'])));cfg['raw_seal']['sha256']=s.sha(path)


def fixture(root,gap=False):
    y=np.zeros(18,np.int8);y[1:3]=1;y[5:7]=1;y[11:13]=1;y[16:18]=1
    labels=root/'synthetic.csv';labels.write_text('timestamp_(min),label\n'+''.join('%d,%d\n'%(i,v) for i,v in enumerate(y)))
    ranges=[dict(domain='SYN',entity='E',run='R',start=a,end=b) for a,b in ([(2,8),(10,18)] if gap else [(2,18)])]
    cfg=dict(profile='SYNTHETIC_SYSTEM_W1',banks=[],comparisons=[],statistics=dict(s.STATISTICS_NA),history=2,original_start=0,original_length=18,stream=['SYN','E','R'],output=str(root/'out'),affiliation_identity=None,
        raw_seal=dict(path=str(root/'seal.json'),sha256=None),labels=dict(path=str(labels),sha256=s.sha(labels),shape=[18],format='csv',drop_columns=['timestamp_(min)'],value_columns=['label']))
    methods=[('full_seed2025','full')]+list(s.CONTROLS.items())+[(k,k) for k in ('persistence','history_mean512')]
    for name,control in methods:
        baseline=name in old.BASELINES;active=['r_pred'] if baseline else (['r_ms','r_pred'] if control=='tf_off' else ['r_tf','r_ms','r_pred'])
        cal=dict(scales={k:1. for k in active},threshold=.5,source_domain_equal_weight=True,domains=['SMD','Sunspot'],active_components=active,weights=recipe(active)[1])
        cp=root/(name+'_cal.json');cp.write_text(json.dumps(cal));identity={k:'a'*64 for k in IDENTITY_KEYS};identity['calibration_sha256']=s.sha(cp)
        directory=root/name
        with ChunkedBank(directory,identity=identity,ranges=ranges,threshold=.5,chunk_size=64) as bank:
            for span in ranges:
                for i in range(span['start'],span['end']):
                    values=dict(r_tf=float('nan') if control=='tf_off' else 1.,r_ms=2.,r_pred=3. if i%3 else .1)
                    if i==12:values['r_pred']=float('nan')
                    row=dict(domain='SYN',entity='E',run='R',index=i,available_index=i,**values);row.update(apply_calibration(row,cal));row['reason']=row.get('reason');bank.append(row)
            bank.finalize()
        e=dict(id=name,control_id=control,kind='deterministic_baseline' if baseline else 'source_model',training_seed=None if baseline else 2025,inference_seed=None if baseline else 2025,calibration_origin='source',active_components=active,directory=str(directory),manifest_sha256=s.sha(directory/'manifest.json'),identity=identity,calibration=dict(path=str(cp),sha256=s.sha(cp)),ranges=ranges)
        if not baseline:
            e['model_flags']=s.FLAGS[control];rp=root/(name+'_receipt.json');rp.write_text(json.dumps(dict(phase='calibrate_source',files={'source_calibration.json':s.sha(cp)})));e['calibration_receipt']=dict(path=str(rp),sha256=s.sha(rp))
        cfg['banks'].append(e)
    cfg['comparisons']=[dict(reference='full_seed2025',candidate=k,pairing='same_training_seed') for k in s.CONTROLS]+[dict(reference='full_seed2025',candidate=k,pairing='model_vs_deterministic') for k in ('persistence','history_mean512')];seal(cfg);return cfg,y


class SystemW1(unittest.TestCase):
    def test_seven_methods_six_pairs_real_tf_missing_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,_=fixture(Path(d));s.contract(cfg);reports=s.bank_gates(cfg);arrays=s.scalar_inputs(cfg)
            self.assertEqual(len(reports),7);self.assertEqual(len(cfg['comparisons']),6)
            self.assertTrue(arrays['tf_off_seed2025']['valid'][0]);self.assertFalse(arrays['tf_off_seed2025']['valid'][10])
            self.assertAlmostEqual(arrays['tf_off_seed2025']['score'][0],.5*np.log(3)+.5*np.log(4),delta=1e-15)

    def test_own_tf_recipe_no_full_CAL_or_zero_component_substitution(self):
        for change in ('recipe','weights','target','receipt','CALidentity'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                cfg,_=fixture(Path(d));e=cfg['banks'][1]
                if change=='recipe':e['active_components']=['r_tf','r_ms','r_pred']
                elif change=='CALidentity':e['calibration']=copy.deepcopy(cfg['banks'][0]['calibration'])
                elif change=='receipt':
                    p=Path(e['calibration_receipt']['path']);p.write_text(json.dumps(dict(phase='target_calibrate')));e['calibration_receipt']['sha256']=s.sha(p)
                else:
                    p=Path(e['calibration']['path']);c=json.loads(p.read_text());c['weights']={'r_ms':.25,'r_pred':.5} if change=='weights' else c['weights'];c['source_domain_equal_weight']=change!='target';p.write_text(json.dumps(c));e['calibration']['sha256']=s.sha(p)
                seal(cfg)
                with patch.object(s,'load_labels',side_effect=AssertionError('No label IO')) as loader:
                    with self.assertRaises(ValueError):s.run_evaluation(cfg)
                    loader.assert_not_called()

    def test_missing_changed_or_unaccepted_seal_before_labels(self):
        for change in ('missing','changed','pending'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                cfg,_=fixture(Path(d));path=Path(cfg['raw_seal']['path'])
                if change=='missing':cfg['raw_seal']['sha256']=None
                elif change=='changed':path.write_text('{}')
                else:
                    v=json.loads(path.read_text());v['approved_by']=None;path.write_text(json.dumps(v));cfg['raw_seal']['sha256']=s.sha(path)
                with patch.object(s,'load_labels',side_effect=AssertionError('No labels')),self.assertRaises(PermissionError):s.run_evaluation(cfg)

    def test_no_source_six_synthetic_alias_fake_seed_or_deleted_pair(self):
        for change in ('profile','seed','pair','alias','ensemble'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                cfg,_=fixture(Path(d))
                if change=='profile':cfg['profile']='PSM_A1_SOURCE_SEED2025'
                elif change=='seed':cfg['banks'][1]['training_seed']=2026
                elif change=='pair':cfg['comparisons'].pop()
                elif change=='alias':cfg['banks'][1]['directory']=cfg['banks'][0]['directory']
                else:cfg['ensemble']=True
                with self.assertRaises(ValueError):s.contract(cfg)

    def test_numeric_core_parity_gap_and_saved_all_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,y=fixture(Path(d),gap=True);result=s.run_evaluation(cfg)
            from hybrid_tsad.hybrid_v1.evaluation import evaluate_bank_arrays
            for e in cfg['banks']:
                direct=evaluate_bank_arrays(e['directory'],y,.5,expected_manifest_sha256=e['manifest_sha256'],original_length=18,warmup_points=2,stream=cfg['stream'])
                saved=result['methods'][[x['descriptor']['id'] for x in result['methods']].index(e['id'])]
                self.assertEqual(saved['ranking'],direct['ranking']);self.assertEqual(saved['coverage'],direct['coverage'])
                self.assertEqual(direct['coverage']['missing_arrived_raw_records'],2)
            self.assertEqual(len(list((Path(cfg['output'])/'pairs').glob('*/paired_events.csv'))),6);self.assertIsNone(result['model_seed_mean'])

    def test_readout_7d_all_four_flags_supported_without_mutation(self):
        from test_named_readout import fixture as readout_fixture
        from hybrid_tsad.hybrid_v1 import readout
        for gpu,control in enumerate(s.CONTROLS.values()):
            with self.subTest(control=control),tempfile.TemporaryDirectory() as d:
                active=['r_ms','r_pred'] if control=='tf_off' else None
                cfg,seal_data=readout_fixture(Path(d),control_id=control,gpu=gpu,active=active,**s.FLAGS[control]);before=copy.deepcopy(cfg)
                value=readout.named_binding_metadata(cfg,seal_data);self.assertTrue(value['pure_source_only_Z']);self.assertEqual(cfg,before)

    def test_preview_pending_and_label_alias_no_IO(self):
        with patch('pathlib.Path.open',side_effect=AssertionError('No IO')),contextlib.redirect_stdout(io.StringIO()):s.main(['--config','MISSING'])
        with tempfile.TemporaryDirectory() as d:
            cfg,_=fixture(Path(d));cfg['raw_seal']['path']=cfg['labels']['path']
            with patch.object(s,'bound',side_effect=AssertionError('No label hash')),self.assertRaises(PermissionError):s.bank_gates(cfg)

    def test_cold_import_and_old_entry_unchanged(self):
        src=str(Path(s.__file__).resolve().parents[2]);code='import sys;sys.path.insert(0,'+repr(src)+');import hybrid_tsad.hybrid_v1.system_evaluate_cli;assert "torch" not in sys.modules;print("W1_COLD_NO_TORCH")'
        r=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=20);self.assertEqual(r.returncode,0,r.stderr);self.assertIn('W1_COLD_NO_TORCH',r.stdout)
        self.assertEqual(s.sha(Path(old.__file__)),'d0fe221e2ce6ba9f6f18f3a143ee6aaca05179b92d018cc2e0d4964afef85957')

    def test_exact_signed_engineering_admit_and_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cfg,_=fixture(root);base=Path(s.__file__).resolve().parents[2]
            required=old.REQUIRED_SOURCE|{'hybrid_tsad/hybrid_v1/'+n for n in ('system_evaluate_cli.py','bank_recalibration.py','named_evaluate_cli.py','scoring.py')}
            cfg.update(operation=s.OPERATION,permissions=dict(labels=True,evaluation=True,training=False,target_fit=False,GPU=False),source_sha256={k:s.sha(base/k) for k in required})
            config=root/'config.json';release=root/'release.json'
            def signed():
                config.write_text(json.dumps(cfg));release.write_text(json.dumps(dict(status='APPROVED',approved_by='root',operation=s.OPERATION,allowed_actions=['evaluate_system_w1_fixed_banks'],allow_labels=True,config_sha256=s.sha(config),raw_seal_sha256=cfg['raw_seal']['sha256'],source_sha256=cfg['source_sha256'])))
            signed();self.assertEqual(s.admit(config,release,s.sha(release))['profile'],'SYNTHETIC_SYSTEM_W1')
            cfg['source_sha256']['hybrid_tsad/hybrid_v1/system_evaluate_cli.py']='0'*64;signed()
            with patch.object(s,'load_labels',side_effect=AssertionError('No labels')),self.assertRaises(PermissionError):s.admit(config,release,s.sha(release))


if __name__=='__main__':unittest.main()
