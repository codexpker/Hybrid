"""Temporary synthetic metadata/compact fixtures only, never actual new scores."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
from hybrid_tsad import tep_normal_control_evaluation_runner as r

class Loader(unittest.TestCase):
    def fixture(self,root):
        def dump(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value),encoding='utf-8');return ref(path)
        def ref(path):return dict(path=str(path.resolve()),sha256=r.sha(path.read_bytes()))
        rows=[dict(entity_id=e,role='NORMAL_CONTROL',faultNumber=0,simulationRun=run,
            path='/home/a312/tsad_research/results/tep_normal_control_inputs_run51_70_20261102_v1/inputs/'+e+'.npz',
            sha256='b'*64,bytes=24000,shape=[960,3],dtype='float64',columns=list(r.identity.COLUMNS),raw_sha256=r.RAW_SHA)
            for e,run in zip(r.identity.ENTITIES,r.core.RUNS)]
        inputs=dump(root/'input_manifest.json',dict(status='PUBLIC_SIMULATION_NORMAL_CONTROL_INPUTS',records=rows,
            raw_source='TEP_FaultFree_Testing.RData',raw_sha256=r.RAW_SHA,sample_axis=[1,960,1],imputation='none',standardization='none',sorting='none'))
        accepted=dump(root/'input_acceptance.json',dict(status='PRIMARY_ACCEPTED_LIMITED_TEP_NORMAL_CONTROL_INPUTS',approved_by='root',entities=20,
            source_sha256=r.RAW_SHA,input_manifest_sha256=inputs['sha256']))
        models=[]
        for family,seed in [('LEFT',2025),('LEFT',2026),('LEFT',2027),('RIDGE_ZMAX',None)]:
            p=root/(family+str(seed));p.mkdir()
            left=family=='LEFT';ends=np.tile(np.arange(99,960),(20,1))
            a=dict(entity_id=np.asarray(r.identity.ENTITIES),end_index_0based=ends)
            keys=['last','mean32'] if left else [m+'_'+d for m in r.core.METHODS[1:] for d in r.core.READOUTS]
            a.update({k:np.zeros((20,861)) for k in keys})
            if left:a.update(available_index_0based=ends.copy(),channel_last=np.zeros((20,861,3)),channel_mean32=np.zeros((20,861,3)))
            np.savez(p/'compact_scores.npz',**a);compact=ref(p/'compact_scores.npz')
            cal=dict(n=8020,rank_1based=7941,comparison='strict_greater',thresholds={k:1. for k in keys},scaler_sha256='c'*64)
            if left:cal.update(seed=seed,checkpoint_sha256='d'*64)
            else:cal.update(coefficient_sha256='e'*64)
            calibration=dump(p/'calibration.json',cal)
            refs=dict(input_manifest=inputs,calibration=calibration,scaler=dict(path='not_opened_scaler',sha256='c'*64))
            refs['checkpoint' if left else 'coefficients']=dict(path='not_opened_parameters',sha256=('d' if left else 'e')*64)
            report=dict(status='COMPLETED_PENDING_INDEPENDENT_REVIEW',role='NORMAL_CONTROL',banks=[dict(entity_id=e) for e in r.identity.ENTITIES],
                compact_scores_sha256=compact['sha256'],source_refs=refs,thresholds=cal['thresholds'])
            if left:report['seed']=seed
            report_ref=dump(p/'report.json',report)
            final=dump(p/'FINAL_MANIFEST.json',dict(files=[dict(path='compact_scores.npz',sha256=compact['sha256']),dict(path='report.json',sha256=report_ref['sha256'])]))
            accept=dict(status='PRIMARY_ACCEPTED_LIMITED_NORMAL20_LEFT_SCORES' if left else 'PRIMARY_ACCEPTED_LIMITED_NORMAL20_RIDGE_SCORES',
                approved_by='root',entities=20,raw_bank_count=20,scored_points_per_entity=861,endpoints_0based=[99,959],
                source_manifest_sha256=final['sha256'],report_sha256=report_ref['sha256'],compact_scores_sha256=compact['sha256'],
                calibration_sha256=calibration['sha256'],input_manifest_sha256=inputs['sha256'])
            if left:accept.update(seed=seed,checkpoint_sha256='d'*64)
            else:accept['coefficients_sha256']='e'*64
            model=dict(family=family,seed=seed,score_acceptance=dump(p/'acceptance.json',accept),score_manifest=final,
                score_report=report_ref,compact_scores=compact,calibration=calibration,scaler_sha256='c'*64)
            model['checkpoint_sha256' if left else 'coefficients_sha256']=('d' if left else 'e')*64
            models.append(model)
        seal=dump(root/'score_seal.json',dict(status='PRIMARY_SEALED_ALL_TEP_NORMAL_CONTROL_SCORES',approved_by='root',operation=r.OP,
            entities=20,points_per_entity=861,endpoints_0based=[99,959],input_acceptance=accepted,input_manifest=inputs,score_models=models))
        contract=root/'contract.md';contract.write_text('# Synthetic contract, not JSON\n',encoding='utf-8')
        (root/'results').mkdir()
        config=dump(root/'config.json',dict(status='BOUND_NORMAL_CONTROL_EVALUATION',score_seal=seal,contract=ref(contract),output=str(root/'results/tep_normal_control_evaluation_20261102_v1')))
        release=dump(root/'release.json',dict(status='PRIMARY_RELEASED_TEP_NORMAL_CONTROL_EVALUATION',approved_by='root',operation=r.OP,
            config=config,score_seal=seal,allow_normal_score_evaluation=True,allow_model_or_fit=False,allow_bootstrap=False,source_hashes=r.source_hashes()))
        return config,release

    def test_synthetic_four_compact_loader_and_all320_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,rel=self.fixture(root)
            fake_resource=types.SimpleNamespace(RUSAGE_SELF=0,getrusage=lambda x:types.SimpleNamespace(ru_maxrss=1234))
            with mock.patch.dict('os.environ',dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')), \
                 mock.patch.dict('sys.modules',{'resource':fake_resource}):
                r.execute(c,rel,rel['sha256'],root)
            out=root/'results/tep_normal_control_evaluation_20261102_v1'
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['result_rows'],320);self.assertEqual(summary['stream_count'],16)
            self.assertEqual(summary['independent_normal_runs'],20)
            self.assertTrue(all(row['result']['metrics']['AP'] is None for row in summary['rows']))
            self.assertEqual(len(json.loads((out/'FINAL_MANIFEST.json').read_text())['files']),4)

    def test_pending_rejects_before_release_or_payload(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'pending.json';p.write_text(json.dumps(dict(status='PENDING',score_seal=None)))
            c=dict(path=str(p),sha256=r.sha(p.read_bytes()))
            with self.assertRaisesRegex(PermissionError,'PENDING'):
                r.load_verified_scores(c,{'path':'never_opened','sha256':'a'*64},'a'*64,td)

    def test_untrusted_release_and_bad_compact_identity_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,rel=self.fixture(root)
            with self.assertRaises(PermissionError):r.load_verified_scores(c,rel,'a'*64,root)
            state,banks=r.load_verified_scores(c,rel,rel['sha256'],root)
            a=copy.deepcopy(banks[0]);a['entity_id'][0]='entity_0001'
            with self.assertRaises(ValueError):r.validate_compact(a,'LEFT')
            a=copy.deepcopy(banks[0]);a['available_index_0based'][0,0]=98
            with self.assertRaises(ValueError):r.validate_compact(a,'LEFT')

    def test_corrupted_payload_sha_never_parsed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,rel=self.fixture(root)
            p=root/'LEFT2025/compact_scores.npz';p.write_bytes(b'corrupt synthetic payload')
            with self.assertRaisesRegex(PermissionError,'bytes changed'):
                r.load_verified_scores(c,rel,rel['sha256'],root)

    def test_existing_output_refuses_in_metadata_before_compact_read(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);c,rel=self.fixture(root)
            (root/'results/tep_normal_control_evaluation_20261102_v1').mkdir()
            with self.assertRaisesRegex(PermissionError,'Fresh'):
                r.verify_metadata(c,rel,rel['sha256'],root)

    def test_changed_seed_or_calibration_chain_rejected(self):
        for field in ['seed','calibration']:
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);c,rel=self.fixture(root)
                cfg=json.loads(Path(c['path']).read_text());seal_ref=cfg['score_seal'];seal=json.loads(Path(seal_ref['path']).read_text())
                if field=='seed':seal['score_models'][0]['seed']=2026
                else:seal['score_models'][0]['calibration']=seal['score_models'][1]['calibration']
                Path(seal_ref['path']).write_text(json.dumps(seal));seal_ref['sha256']=r.sha(Path(seal_ref['path']).read_bytes())
                Path(c['path']).write_text(json.dumps(cfg));c['sha256']=r.sha(Path(c['path']).read_bytes())
                release=json.loads(Path(rel['path']).read_text());release['score_seal']=seal_ref;release['config']=c
                Path(rel['path']).write_text(json.dumps(release));rel['sha256']=r.sha(Path(rel['path']).read_bytes())
                with self.assertRaises(PermissionError):r.load_verified_scores(c,rel,rel['sha256'],root)

if __name__=='__main__':unittest.main(verbosity=2)
