"""Synthetic complete original axis and phase counterexamples, no true banks."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from hybrid_tsad.hybrid_v1 import smd_recalibration as o, normal_readout as q, event_adaptation as event
from hybrid_tsad.hybrid_v1.chunked_bank import ChunkedBank,IDENTITY_KEYS,read_records,read_entries,_fingerprint
from hybrid_tsad.hybrid_v1.scoring import apply_calibration

class SMDOverlay(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.identity={k:'a'*64 for k in IDENTITY_KEYS};self.identity.update(input_sha256=q.helper.RAW_SHA,checkpoint_sha256=q.normal.SOURCE,calibration_sha256=q.SOURCE_CAL,
            input_runs=[dict(domain='SMD',entity='machine-2-1',run='machine-2-1_test',start=17129,length=6565,sha256='e'*64,
                input_identity=dict(values_float32_sha256='v'*64,observed_sha256='o'*64,start=17129,shape=[6565,38]))])
        self.pconfig=dict(operation=q.OPERATION,profile=q.PROFILE,mode='Z_SOURCE_R3',method_id='Z_SOURCE_R3',choice='COMMON_SOURCE',source_sha256={'fake':'code'},
            model={'fixture':True},sources={},history=512,horizon=32,active_components=['r_tf','r_ms','r_pred'],assets=dict(checkpoint=dict(sha256=q.normal.SOURCE),calibration=dict(sha256=q.SOURCE_CAL)))
        self.config_asset=self.asset('producer_config',self.pconfig);self.identity['config_sha256']=self.config_asset['sha256']
        self.parent=dict(directory=str(self.root/'producer'),identity=self.identity,ranges=copy.deepcopy(o.RANGES))
        # Gaps in validity are real missing scores, not fabricated normal samples.
        self.rows=[dict(domain='SMD',entity='machine-2-1',run='machine-2-1_test',index=t,available_index=t,
            r_tf=float('nan') if t==17704 else 1.,r_ms=2.,r_pred=3.,score=float(np.log(2)),valid=True,alarm=True,reason=None,
            prediction=np.array([-0.,np.nan],dtype=np.float32),replay=dict(actual_mask_seed=2025+t-17129,input_identity=self.identity['input_runs'][0]['input_identity'])) for t in range(17641,23694)]
        self.rows[63].update(score=float('nan'),valid=False,reason='MISSING_REQUIRED_COMPONENT');self.rows[63].pop('alarm')
        with ChunkedBank(self.parent['directory'],identity=self.identity,ranges=o.RANGES,threshold=.1,chunk_size=64) as bank:
            for row in self.rows:bank.append(row)
            bank.finalize()
        self.parent['manifest_sha256']=q.sha(Path(self.parent['directory'])/'manifest.json')
        self.receipt=dict(operation=q.OPERATION,mode='Z_SOURCE_R3',score_kind='source_calibrated_R3',status='COMPLETED_PENDING_RAW_ACCEPTANCE',updates=0,
            labels_read=False,model_state_unchanged=True,records=6053,bank_manifest_sha256=self.parent['manifest_sha256'])
        self.ev=dict(config=self.config_asset,receipt=self.asset('producer_receipt',self.receipt),
            acceptance=self.asset('producer_acceptance',dict(approved_by='root',status='ACCEPTED_SYNTHETIC_SCHEMA',bank_manifest_sha256=self.parent['manifest_sha256'],checkpoint_sha256=q.normal.SOURCE,config_sha256=self.config_asset['sha256'])),
            execution_identity=self.asset('producer_execution',dict(config_sha256=self.config_asset['sha256'],source_sha256=self.pconfig['source_sha256'])))
        cal=dict(source_domain_equal_weight=False,domains=['SMD'],valid_windows=512,active_components=['r_tf','r_ms','r_pred'],weights=dict(r_tf=.25,r_ms=.25,r_pred=.5),scales=dict(r_tf=1.,r_ms=2.,r_pred=3.),threshold=.1)
        self.cal=cal;self.e=dict(directory=str(self.root/'overlay'),identity=dict(self.identity),ranges=copy.deepcopy(o.RANGES),active_components=cal['active_components'],calibration_origin='target',adaptation_mode='Z_TCAL_R3',calibration=self.asset('target_cal',cal))
        self.e['identity']['calibration_sha256']=self.e['calibration']['sha256'];self.e['identity']['config_sha256']=o.original.recipe_identity(self.parent,self.e)
        for key in ('normal_plans','normal_manifest'):self.e[key]=self.asset(key,dict(synthetic_metadata_only=True))
        self.patch=patch.dict(event.ACCEPTED,dict(plans=self.e['normal_plans']['sha256'],manifest=self.e['normal_manifest']['sha256']),clear=True);self.patch.start();self.addCleanup(self.patch.stop)
        cc=dict(self.pconfig,operation=q.normal.OPERATION,mode='Z_TCAL_R3',scope='public_target',choice='COMMON_SOURCE',permissions=dict(phase='source_target_calibrate',query=False,labels=False),
            source_checkpoint_sha256=q.normal.SOURCE,select_count=32,cal_count=512,seed=2025,plans_sha256=event.ACCEPTED['plans'],manifest_sha256=event.ACCEPTED['manifest'])
        self.e['calibration_config']=self.asset('target_cal_config',cc)
        cr=dict(operation=q.normal.OPERATION,mode='Z_TCAL_R3',phase='source_target_calibrate',status='COMPLETED',updates=0,loaded_roles=['NORMAL_CAL'],query_or_labels=False,
            target_calibration_contact=True,choice='COMMON_SOURCE',calibrated_checkpoint_sha256=q.normal.SOURCE,source_checkpoint_sha256=q.normal.SOURCE,
            calibration_file='target_r3_calibration.json',threshold=.1,plans_sha256=event.ACCEPTED['plans'],manifest_sha256=event.ACCEPTED['manifest'])
        self.e['calibration_receipt']=self.asset('target_cal_receipt',cr)
        self.e['calibration_acceptance']=self.asset('target_cal_acceptance',dict(approved_by='root',status='ACCEPTED_SYNTHETIC_SCHEMA',calibration_sha256=self.e['calibration']['sha256'],receipt_sha256=self.e['calibration_receipt']['sha256'],config_sha256=self.e['calibration_config']['sha256']))
        self.cfg=dict(operation=o.OPERATION,permissions=dict(raw_bank=True,labels=False,model_forward=False,threshold_fit=False),producer=self.parent,producer_evidence=self.ev,destination=self.e)
        self.seal()
    def asset(self,name,value):
        path=self.root/(name+'.json');path.write_text(json.dumps(value));return dict(path=str(path),sha256=q.sha(path))
    def seal(self):self.cfg['input_seal']=self.asset('root_input_seal',dict(status='ACCEPTED_SMD_Z_RECALIBRATION_INPUTS',approved_by='root',producer=self.parent,producer_evidence=self.ev,destination=self.e))
    def test_complete_overlay_raw_fingerprint_and_cross_chunk_confirm_reset(self):
        original={p.name:q.sha(p) for p in Path(self.parent['directory']).iterdir() if p.is_file()}
        result=o.run(self.cfg);spec=dict(receipt=dict(path=str(self.root/'overlay/recalibration_receipt.json'),sha256=q.sha(self.root/'overlay/recalibration_receipt.json')),producer=self.parent,destination=self.e)
        materialized=list(o.iter_materialized(spec));m=q.read(self.root/'overlay/manifest.json');entries=list(read_entries(self.root/'overlay'))
        self.assertEqual((result['records'],len(m['chunks']),m['chunks'][-1]['records']),(6053,95,37))
        self.assertEqual([r['index'] for r in materialized],[r['index'] for r in self.rows])
        for source,row in zip(self.rows,materialized):
            for key in ('prediction','replay','r_tf','r_ms','r_pred'):self.assertEqual(_fingerprint(source[key]),_fingerprint(row[key]))
            expected=apply_calibration(source,self.cal);self.assertEqual(row['valid'],expected['valid'])
            if expected['valid']:self.assertEqual(row['score'],expected['score'])
            else:self.assertNotIn('alarm',row);self.assertEqual(row['producer_derived']['valid'],False);self.assertEqual(row['producer_derived']['reason'],'MISSING_REQUIRED_COMPONENT')
        self.assertFalse(entries[63]['raw']['valid']);self.assertFalse(entries[64]['derived']['confirmed3_current']);self.assertTrue(entries[66]['derived']['confirmed3_current'])
        self.assertEqual(original,{p.name:q.sha(p) for p in Path(self.parent['directory']).iterdir() if p.is_file()})
    def test_U_N_F_source_false_receipts_cannot_masquerade(self):
        for phase in ('target_calibrate','target_normal_calibrate','target_event_calibrate'):
            e=copy.deepcopy(self.e);r=q.read(e['calibration_receipt']['path']);r['phase']=phase;e['calibration_receipt']=self.asset('bad_receipt',r)
            with self.assertRaises(ValueError):o.target_calibration(e,self.pconfig)
    def test_same_source_scale_but_wrong_normal_pool_rejected(self):
        e=copy.deepcopy(self.e);e['normal_plans']['sha256']='c'*64
        with self.assertRaises(PermissionError):o.target_calibration(e,self.pconfig)
    def test_borrowed_N_producer_or_shifted_axis_rejected(self):
        self.pconfig['mode']='N_NEG300';self.ev['config']=self.asset('wrong_producer',self.pconfig);self.seal()
        with self.assertRaises(ValueError):o.gate(self.cfg)
    def test_pending_inputseal_stops_before_rawbank_read(self):
        self.cfg['input_seal']=self.asset('pending_seal',dict(status='PENDING'))
        with patch.object(o.original,'bank_metadata',side_effect=AssertionError('Must not read bank')):
            with self.assertRaises(PermissionError):o.gate(self.cfg)
    def test_incomplete_bank_or_changed_producer_bytes_rejected(self):
        path=self.root/'producer/manifest.json';m=q.read(path);m['complete']=False;path.write_text(json.dumps(m))
        with self.assertRaises(PermissionError):o.gate(self.cfg)
    def test_overlay_cannot_change_checkpoint_or_original_index(self):
        self.e['identity']['checkpoint_sha256']='c'*64;self.e['identity']['config_sha256']=o.original.recipe_identity(self.parent,self.e);self.seal()
        with self.assertRaises(ValueError):o.gate(self.cfg)
    def test_public_overlay_rejects_discontinuous_time_axis(self):
        self.e['ranges']=[dict(o.RANGES[0],end=18000),dict(o.RANGES[0],start=18001)];self.seal()
        with self.assertRaises(ValueError):o.gate(self.cfg)

class MetadataBoundary(unittest.TestCase):
    def test_annotation_or_arrays_rejected_before_open(self):
        for path in ('/never/test_label.json','/never/annotation_ledger.json','/never/query.npz','/never/weights.pt'):
            with patch.object(o.original,'bound',side_effect=AssertionError('Must not open file')):
                with self.assertRaises(PermissionError):o.metadata(dict(path=path,sha256='a'*64))

if __name__=='__main__':unittest.main()
