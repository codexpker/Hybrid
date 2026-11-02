"""Synthetic CPU only; actual execution requires root's separate release."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import bank_recalibration as r,target_evaluate_cli as t,named_evaluate_cli as n
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,IDENTITY_KEYS,read_records,read_entries,_fingerprint
from hybrid_tsad.hybrid_v1.scoring import apply_calibration


def write(path,value):path.write_text(json.dumps(value));return dict(path=str(path),sha256=r.sha(path))
def seal(cfg):cfg['raw_seal']=write(Path(cfg['output']).parent/'seal.json',dict(status='ACCEPTED_TARGET_FIXED_BANKS',approved_by='root',profile=cfg['profile'],banks=cfg['banks']))


def fixture(root,gap=False,same=False):
    ranges=[dict(domain='SYN',entity='E',run='R',start=a,end=b) for a,b in ([(2,70),(72,103)] if gap else [(2,103)])]
    entries=[]
    for id,(origin,mode) in t.METHODS.items():
        scales=dict(r_tf=1.,r_ms=10.,r_pred=1.) if origin=='source' or same else dict(r_tf=10.,r_ms=1.,r_pred=1.)
        cal=write(root/(id+'_cal.json'),dict(scales=scales,threshold=.2,active_components=['r_tf','r_ms','r_pred'],weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),source_domain_equal_weight=origin=='source'))
        identity={k:'a'*64 for k in IDENTITY_KEYS};identity['calibration_sha256']=cal['sha256']
        if mode=='U':identity['checkpoint_sha256']='c'*64
        receipt=dict(phase='calibrate_source' if origin=='source' else 'target_calibrate',status='COMPLETED')
        if origin=='target':receipt.update(mode=mode,updates=0,query_or_labels=False,loaded_roles=['CAL'],target_calibration_contact=True,source_checkpoint_sha256='a'*64,calibrated_checkpoint_sha256=identity['checkpoint_sha256'],plans_sha256='d'*64,manifest_sha256='e'*64)
        entry=dict(id=id,calibration_origin=origin,adaptation_mode=mode,training_seed=2025,inference_seed=2025,active_components=['r_tf','r_ms','r_pred'],identity=identity,ranges=ranges,directory=str(root/id),calibration=cal,calibration_receipt=write(root/(id+'_receipt.json'),receipt),receipt_expect={k:v for k,v in receipt.items() if k in ('phase','plans_sha256','manifest_sha256')})
        if mode=='U':entry['parent_source_checkpoint_sha256']='a'*64
        entries.append(entry)
        if mode=='Z_TCAL':continue
        c=json.loads(Path(cal['path']).read_text())
        with ChunkedBank(entry['directory'],identity=identity,ranges=ranges,threshold=.2,chunk_size=64) as bank:
            for span in ranges:
                for i in range(span['start'],span['end']):
                    raw=dict(domain='SYN',entity='E',run='R',index=i,available_index=i,r_tf=9. if i%2 else 1.,r_ms=1. if i%2 else 9.,r_pred=1.,payload=np.array([-0.,np.nan],np.float32))
                    if i==13:raw['r_tf']=float('nan')
                    raw.update(apply_calibration(raw,c));raw['reason']=raw.get('reason');bank.append(raw)
            bank.finalize()
        entry['manifest_sha256']=r.sha(Path(entry['directory'])/'manifest.json')
    z,d,u=entries;d['identity']['config_sha256']=r.recipe_identity(z,d);cfg=dict(operation=r.OPERATION,producer=z,destination=d)
    cfg['input_seal']=write(root/'input_seal.json',dict(status='ACCEPTED_RECALIBRATION_INPUTS',approved_by='root',producer=z,destination=d))
    receipt=r.run(cfg);d['manifest_sha256']=receipt['manifest_sha256']
    d['overlay']=dict(producer=z,destination=receipt['destination'],receipt=dict(path=str(Path(d['directory'])/'recalibration_receipt.json'),sha256=r.sha(Path(d['directory'])/'recalibration_receipt.json')))
    # JSON serialization already saved the destination; retain that independent
    # descriptor instead of a Python object alias later extended by the fixture.
    d['overlay']['destination']=json.loads(Path(d['overlay']['receipt']['path']).read_text())['destination']
    y=np.zeros(103,np.int8);y[1:3]=1;y[69:74]=1;y[99:103]=1
    path=root/'labels.csv';path.write_text('timestamp_(min),label\n'+''.join('%d,%d\n'%(i,v) for i,v in enumerate(y)))
    out=dict(profile='SYNTHETIC_TARGET',banks=entries,history=2,original_length=103,original_start=0,stream=['SYN','E','R'],statistics=dict(n.STATISTICS_NA),labels=dict(path=str(path),sha256=r.sha(path),shape=[103],format='csv',drop_columns=['timestamp_(min)'],value_columns=['label']),output=str(root/'evaluation'),affiliation_identity=None,
        comparisons=[dict(reference='Z_S_seed2025',candidate='Z_Tcal_seed2025',pairing='calibration_only'),dict(reference='Z_Tcal_seed2025',candidate='U_Tcal_seed2025',pairing='same_seed_equal_target_CAL'),dict(reference='Z_S_seed2025',candidate='U_Tcal_seed2025',pairing='combined_update_and_calibration')])
    seal(out);return out,cfg,y


class RecalibrationTarget(unittest.TestCase):
    def test_same_CAL_original_raw_bytes_indices_NA_and_tail_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,recal,_=fixture(Path(d),same=True);z,v,_=cfg['banks']
            before={p.name:r.sha(p) for p in Path(z['directory']).iterdir() if p.is_file()}
            a=list(read_records(z['directory']));b=list(read_records(v['directory']))
            self.assertEqual([x['index'] for x in b],list(range(2,103)))
            self.assertEqual([x['index'] for x in b[-31:]],list(range(72,103)))
            for x,y in zip(a,b):
                self.assertEqual(x['valid'],y['valid'])
                if x['valid']:self.assertEqual(x['score'],y['score'])
                for k in ('r_tf','r_ms','r_pred'):self.assertEqual(_fingerprint({k:x[k]}),_fingerprint({k:y[k]}))
            t.gates(cfg)
            full=list(r.iter_materialized(v['overlay']))
            for original,materialized in zip(a,full):
                for key in set(original)-{'score','valid','reason','alarm'}:
                    self.assertEqual(_fingerprint({key:original[key]}),_fingerprint({key:materialized[key]}))
                self.assertEqual(materialized['producer_derived']['valid'],original['valid'])
                self.assertEqual(materialized['producer_derived']['reason'],original['reason'])
            self.assertEqual(before,{p.name:r.sha(p) for p in Path(z['directory']).iterdir() if p.is_file()})

    def test_component_scales_can_reverse_rank_not_scalar_monotonic(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,_,_=fixture(Path(d));a,b=[list(read_records(e['directory'])) for e in cfg['banks'][:2]]
            self.assertLess(a[0]['score'],a[1]['score']);self.assertGreater(b[0]['score'],b[1]['score'])
            self.assertAlmostEqual(b[0]['score'],.25*np.log1p(.1)+.25*np.log1p(9)+.5*np.log(2),delta=1e-15)

    def test_gaps_reset_confirm_and_not_joined_by_evaluator(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,_,_=fixture(Path(d),gap=True);rows=list(read_entries(cfg['banks'][1]['directory']));lookup={x['raw']['index']:x['derived'] for x in rows}
            self.assertTrue(lookup[69]['confirmed3_current']);self.assertFalse(lookup[72]['confirmed3_current']);self.assertTrue(lookup[74]['confirmed3_current']);t.gates(cfg)
            from hybrid_tsad.hybrid_v1.evaluation import evaluate_bank_arrays
            e=cfg['banks'][1];labels=np.zeros(103,np.int8)
            result=evaluate_bank_arrays(e['directory'],labels,.2,expected_manifest_sha256=e['manifest_sha256'],original_length=103,warmup_points=2,stream=cfg['stream'])
            self.assertEqual(result['coverage']['unscored_points'],5) # warmup2 + gap2 + invalid raw1

    def test_changed_or_missing_seal_refused_before_label_open(self):
        for change in ('missing','changed'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                cfg,_,_=fixture(Path(d));cfg['raw_seal']['sha256']=None if change=='missing' else 'b'*64
                with patch.object(n,'load_labels',side_effect=AssertionError('labels')) as loader:
                    with self.assertRaises(PermissionError):t.run(cfg)
                    loader.assert_not_called()

    def test_incomplete_raw_producer_rejected_without_output(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,recal,_=fixture(Path(d));recal=copy.deepcopy(recal);recal['destination']['directory']=str(Path(d)/'fresh')
            p=Path(recal['producer']['directory'])/'manifest.json';m=json.loads(p.read_text());m['complete']=False;p.write_text(json.dumps(m));recal['producer']['manifest_sha256']=r.sha(p)
            recal['input_seal']=write(Path(d)/'newseal.json',dict(status='ACCEPTED_RECALIBRATION_INPUTS',approved_by='root',producer=recal['producer'],destination=recal['destination']))
            with self.assertRaises(ValueError):r.run(recal)
            self.assertFalse(Path(recal['destination']['directory']).exists())

    def test_CAL_phase_source_flag_checkpoint_or_plan_cannot_masquerade(self):
        for key in ('calibration_origin','adaptation_mode','checkpoint','plan'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                cfg,_,_=fixture(Path(d));e=cfg['banks'][2]
                if key=='checkpoint':e['parent_source_checkpoint_sha256']='f'*64
                elif key=='plan':e['receipt_expect']['plans_sha256']='f'*64
                else:e[key]='source' if key=='calibration_origin' else 'Z_TCAL'
                seal(cfg)
                with patch.object(n,'load_labels',side_effect=AssertionError('labels')),self.assertRaises(ValueError):t.run(cfg)
        for change in ('source_flag','phase','mode'):
            with self.subTest(actual_CAL_guard=change),tempfile.TemporaryDirectory() as d:
                cfg,_,_=fixture(Path(d));e=cfg['banks'][1]
                if change=='source_flag':
                    value=json.loads(Path(e['calibration']['path']).read_text());value['source_domain_equal_weight']=True
                    e['calibration']=write(Path(e['calibration']['path']),value);e['identity']['calibration_sha256']=e['calibration']['sha256']
                else:
                    value=json.loads(Path(e['calibration_receipt']['path']).read_text());value['phase' if change=='phase' else 'mode']='calibrate_source' if change=='phase' else 'U'
                    e['calibration_receipt']=write(Path(e['calibration_receipt']['path']),value)
                with self.assertRaises(ValueError):r.calibration(e)

    def test_fake_three_seed_mean_and_changed_primary_recipe_rejected(self):
        for change in ('mean','recipe','seed'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
                cfg,_,_=fixture(Path(d))
                if change=='mean':cfg['model_mean_SD']={}
                elif change=='recipe':cfg['banks'][1]['active_components']=['r_pred']
                else:cfg['banks'][2]['training_seed']=2026
                with self.assertRaises(ValueError):t.contract(cfg)

    def test_real_core_numerical_parity_and_saved_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,_,y=fixture(Path(d),gap=True);result=t.run(cfg)
            from hybrid_tsad.hybrid_v1.evaluation import evaluate_bank_arrays
            for e in cfg['banks']:
                direct=evaluate_bank_arrays(e['directory'],y,.2,expected_manifest_sha256=e['manifest_sha256'],original_length=103,warmup_points=2,stream=cfg['stream'])
                self.assertEqual(result['methods'][e['id']],direct)
            self.assertIsNone(result['model_seed_mean']);self.assertEqual(len(result['comparisons']),3)
            self.assertEqual(len(list((Path(cfg['output'])/'pairs').glob('*/paired_events.csv'))),3)

    def test_label_alias_refused_before_hash_or_body(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,_,_=fixture(Path(d));cfg['raw_seal']['path']=cfg['labels']['path']
            with patch.object(t,'bound',side_effect=AssertionError('No prelabel open')):
                with self.assertRaises(PermissionError):t.gates(cfg)

    def test_preview_has_no_IO(self):
        for module in (r,t):
            with patch('pathlib.Path.open',side_effect=AssertionError('No IO')),contextlib.redirect_stdout(io.StringIO()):module.main(['--config','missing'])


if __name__=='__main__':unittest.main()
